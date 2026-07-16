import asyncio
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Union

import ray
import zmq
from ray.util.queue import Queue
from tqdm import tqdm

from ..arguments import EvaluationArguments
from ..benchmarks import BaseBenchmark
from ..inference_wrappers import BaseInferenceWrapper
from .vlm_worker import VLMWorker


def filter_metadata(data: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if isinstance(data[key], (dict, list, tuple)):
                new_data[key] = filter_metadata(value)
            elif isinstance(data[key], (int, float, bool, str)):
                new_data[key] = value
        return new_data
    elif isinstance(data, (list, tuple)):
        new_data = []
        for item in data:
            if isinstance(item, (dict, list, tuple)):
                new_data.append(filter_metadata(item))
            elif isinstance(item, (int, float, bool, str)):
                new_data.append(item)
        return new_data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


class Dispatcher(object):
    def __init__(
        self,
        benchmarks: Dict[str, BaseBenchmark],
        request_queues: List[Queue],
        model_endpoints: List[str],
        max_concurrency: int,
    ):
        self.benchmarks = benchmarks
        self.request_queues = request_queues
        self.max_concurrency = max_concurrency

        context = zmq.Context()
        self.poller = zmq.Poller()
        self.sockets = []

        for port in model_endpoints:
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.CONFLATE, 1)
            socket.connect(port)
            self.poller.register(socket, zmq.POLLIN)
            self.sockets.append(socket)

    async def _add_request(self, data: Dict[str, Any]):
        queue_id = random.randint(0, len(self.request_queues) - 1)

        # min_load = float("inf")
        # socks = dict(self.poller.poll())
        # for i, socket in enumerate(self.sockets):
        #     if socket in socks:
        #         load = int(socket.recv_string())
        #         if load < min_load:
        #             min_load = load
        #             queue_id = i

        await asyncio.to_thread(self.request_queues[queue_id].put, data)

    async def start(self):
        pbar = tqdm(
            total=sum(len(benchmark) for benchmark in self.benchmarks.values()),
            desc="Dispatch tasks",
            position=0,
        )

        sem = asyncio.Semaphore(self.max_concurrency)
        running_tasks = set()

        def callback(task):
            running_tasks.discard(task)
            sem.release()
            pbar.update(1)

        for name, benchmark in self.benchmarks.items():
            for data in benchmark:
                await sem.acquire()
                data["benchmark"] = name
                task = asyncio.create_task(self._add_request(data))
                running_tasks.add(task)
                task.add_done_callback(callback)

        await asyncio.gather(*running_tasks)
        pbar.close()

        for queue in self.request_queues:
            queue.put(None)


class Collector(object):
    def __init__(
        self,
        benchmarks: Dict[str, BaseBenchmark],
        result_queue: Queue,
    ):
        self.benchmarks = benchmarks
        self.result_queue = result_queue
        self.total = sum(benchmark.n_samples for benchmark in benchmarks.values())

    async def _process_response(
        self,
        data_id: str,
        response: str,
        benchmark: BaseBenchmark,
        results: List[Dict[str, Any]],
    ):
        prediction = await benchmark.process_response(data_id, response)
        score = await benchmark.get_matching_score(data_id, prediction)
        result = {
            "data_id": data_id,
            "response": response,
            "prediction": prediction,
            "score": score,
        }
        results.append(result)

    async def start(self):
        tasks = []
        results = defaultdict(list)

        for _ in tqdm(range(self.total), desc="Collect results", position=1):
            response = await self.result_queue.get_async()
            task = asyncio.create_task(
                self._process_response(
                    response["data_id"],
                    response["text"],
                    self.benchmarks[response["benchmark"]],
                    results[response["benchmark"]],
                )
            )
            tasks.append(task)

        await asyncio.gather(*tasks)
        return results


class Evaluator(object):
    def __init__(
        self,
        args: EvaluationArguments,
        inference_wrapper: BaseInferenceWrapper,
        benchmarks: Union[BaseBenchmark, List[BaseBenchmark]],
    ):
        if not isinstance(benchmarks, (list, tuple)):
            benchmarks = [benchmarks]
        self.args = args
        self.benchmarks = benchmarks
        self.inference_wrapper = inference_wrapper

    async def _eval(self):
        resources = ray.cluster_resources()
        num_gpus = int(resources.get("GPU", 0))

        num_gpus_per_worker = self.args.tensor_parallel_size * self.args.pipeline_parallel_size
        num_workers = num_gpus // num_gpus_per_worker

        model_workers = []
        model_endpoints = []
        request_queues = []
        result_queue = Queue()

        for i in range(num_workers):
            request_queue = Queue(maxsize=8)
            model_worker = VLMWorker.options(num_gpus=num_gpus_per_worker).remote(
                backend=self.args.backend,
                inference_wrapper=self.inference_wrapper,
                processing_params=self.args.processing_params,
                sampling_params=self.args.sampling_params,
                parallel_params=self.args.parallel_params,
                num_processor_workers=self.args.num_processor_workers,
                input_queue=request_queue,
                output_queue=result_queue,
            )
            model_port = ray.get(model_worker.get_endpoint.remote())
            model_worker.start.remote()

            model_workers.append(model_worker)
            request_queues.append(request_queue)
            model_endpoints.append(model_port)

        benchmarks = {benchmark.__class__.__name__: benchmark for benchmark in self.benchmarks}

        dispatcher = Dispatcher(
            benchmarks=benchmarks,
            request_queues=request_queues,
            model_endpoints=model_endpoints,
            max_concurrency=len(model_workers) * 2,
        )
        asyncio.create_task(dispatcher.start())

        collector = Collector(
            benchmarks=benchmarks,
            result_queue=result_queue,
        )
        results = await collector.start()

        prefix = datetime.now().strftime("%Y%m%d%H%M%S")

        for name in benchmarks:
            benchmark = benchmarks[name]
            metrics = benchmark.compute_metrics(results[name])

            print("=" * 20, f"Results on {name}", "=" * 20)
            print(json.dumps(metrics, indent=4))

            save_path = os.path.join(self.args.save_dir, f"{prefix}_{name}.json")

            for result in results[name]:
                result["metadata"] = benchmark.data_dict[result["data_id"]]
                result["metadata"] = filter_metadata(dict(result["metadata"]))

            os.makedirs(self.args.save_dir, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(
                    {
                        "model_path": self.args.model_path,
                        "benchmark": name,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "prompt_format": self.args.prompt_format,
                        "enable_thinking": self.args.enable_thinking,
                        "sampling_params": self.args.sampling_params,
                        "processing_params": self.args.processing_params,
                        "metrics": metrics,
                        "metadata": results[name],
                    },
                    f,
                    indent=4,
                )

    def eval(self):
        asyncio.run(self._eval())
