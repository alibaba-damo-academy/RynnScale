import asyncio
from typing import Any, Dict

import ray
import torch
import torch.multiprocessing as mp
import zmq
from ray.util.queue import Empty, Queue
from torch.utils.data import DataLoader, IterableDataset

from ..inference_wrappers import BaseInferenceWrapper


class ProcessorWorker(IterableDataset):
    def __init__(
        self,
        backend: str,
        inference_wrapper: BaseInferenceWrapper,
        processing_params: Dict[str, Any],
        input_queue: mp.Queue,
    ):
        self.backend = backend
        self.inference_wrapper = inference_wrapper
        self.processing_params = processing_params
        self.input_queue = input_queue

    def _preprocess(self, data):
        enable_thinking = data.pop("enable_thinking", False)
        image_inputs, video_inputs = {}, {}

        for data_id, conversation in zip(data["data_ids"], data["conversations"]):
            prompt = self.inference_wrapper.apply_chat_template(conversation, enable_thinking=enable_thinking)

            if len(image_inputs) == 0 and len(video_inputs) == 0:
                images, videos = [], []
                for message in conversation:
                    for content in message["content"]:
                        if content["type"] == "image":
                            images.append(content["image"])
                        elif content["type"] == "video":
                            videos.append(content["video"])

                if len(images):
                    images = self.inference_wrapper.load_images(images, processing_params=self.processing_params)
                    image_inputs = self.inference_wrapper.process_images(
                        images, processing_params=self.processing_params
                    )

                if len(videos):
                    videos = self.inference_wrapper.load_videos(videos, processing_params=self.processing_params)
                    video_inputs = self.inference_wrapper.process_videos(
                        videos, processing_params=self.processing_params
                    )

            model_inputs = self.inference_wrapper.process_text(
                text=prompt,
                image_inputs=image_inputs,
                video_inputs=video_inputs,
            )

            if self.backend == "sglang":
                image_data, video_data = {"format": "processor_output"}, {"format": "processor_output"}
                for name, value in model_inputs.items():
                    if name in self.inference_wrapper.processor.image_processor.model_input_names:
                        image_data[name] = value
                    if name in self.inference_wrapper.processor.video_processor.model_input_names:
                        video_data[name] = value

                model_inputs = {
                    "input_ids": model_inputs["input_ids"][0].tolist(),
                    "image_data": image_data if len(image_data) > 1 else None,
                    "video_data": video_data if len(video_data) > 1 else None,
                }

            request = {
                "benchmark": data["benchmark"],
                "data_id": data_id,
                "model_inputs": model_inputs,
            }
            yield request

    def __iter__(self):
        while True:
            data = self.input_queue.get()
            if data is None:
                self.input_queue.put(data)
                break
            yield from self._preprocess(data)


class HFModelRunner(object):
    def __init__(
        self,
        inference_wrapper: BaseInferenceWrapper,
        sampling_params: Dict[str, Any],
        parallel_params: Dict[str, Any],
        data_loader: DataLoader,
        output_queue: mp.Queue,
    ):
        self.data_loader = data_loader
        self.output_queue = output_queue

        self.inference_wrapper = inference_wrapper

        self.sampling_params = sampling_params.copy()
        if self.sampling_params.get("temperature", None) == 0.0:
            self.sampling_params["do_sample"] = False

    def start(self):
        for data in self.data_loader:
            model_inputs = data.pop("model_inputs").to("cuda")
            with torch.inference_mode():
                texts = self.inference_wrapper.generate(
                    model_inputs=model_inputs,
                    sampling_params=self.sampling_params,
                )
            text = texts[0]

            response = {**data, "text": text}
            self.output_queue.put(response)


class SGLangModelRunner(object):
    def __init__(
        self,
        inference_wrapper: BaseInferenceWrapper,
        sampling_params: Dict[str, Any],
        parallel_params: Dict[str, Any],
        data_loader: DataLoader,
        output_queue: mp.Queue,
    ):
        import sglang as sgl

        self.engine = sgl.Engine(
            model_path=inference_wrapper.model_path,
            mem_fraction_static=0.8,
            **parallel_params,
        )

        self.sampling_params = sampling_params
        self.data_iterator = iter(data_loader)
        self.output_queue = output_queue

    async def _process_request(self, request: Dict[str, Any]):
        result = await self.engine.async_generate(
            **request.pop("model_inputs"),
            sampling_params=self.sampling_params,
        )
        response = {
            **request,
            "text": result["text"],
        }
        await asyncio.to_thread(self.output_queue.put, response)

    async def _main_loop(self):
        running_tasks = set()
        sem = asyncio.Semaphore(32)

        def callback(task):
            running_tasks.discard(task)
            sem.release()

        while True:
            try:
                await sem.acquire()
                data = await asyncio.to_thread(next, self.data_iterator, None)

                if data is None:
                    break

                task = asyncio.create_task(self._process_request(data))
                running_tasks.add(task)
                task.add_done_callback(callback)

            except Empty:
                pass

            # self._publish_load(len(running_tasks))

        await asyncio.gather(*running_tasks)
        self.output_queue.put(None)

    def start(self):
        asyncio.run(self._main_loop())


def _collate_fn(data_list):
    return data_list[0]


def start_model_runner(
    backend: str,
    inference_wrapper: BaseInferenceWrapper,
    processing_params: Dict[str, Any],
    sampling_params: Dict[str, Any],
    parallel_params: Dict[str, Any],
    num_processor_workers: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
):
    if backend == "hf":
        runner_class = HFModelRunner
    elif backend == "sglang":
        runner_class = SGLangModelRunner
    else:
        raise ValueError

    dataset = ProcessorWorker(
        backend=backend,
        inference_wrapper=inference_wrapper,
        processing_params=processing_params,
        input_queue=input_queue,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_processor_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    runner_class(
        inference_wrapper=inference_wrapper,
        sampling_params=sampling_params,
        parallel_params=parallel_params,
        data_loader=data_loader,
        output_queue=output_queue,
    ).start()


@ray.remote
class VLMWorker(object):
    def __init__(
        self,
        backend: str,
        inference_wrapper: BaseInferenceWrapper,
        processing_params: Dict[str, Any],
        sampling_params: Dict[str, Any],
        parallel_params: Dict[str, Any],
        num_processor_workers: int,
        input_queue: Queue,
        output_queue: Queue,
    ):
        self.backend = backend
        self.inference_wrapper = inference_wrapper
        self.processing_params = processing_params
        self.sampling_params = sampling_params
        self.parallel_params = parallel_params
        self.num_processor_workers = num_processor_workers
        self.input_queue = input_queue
        self.output_queue = output_queue

        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind_to_random_port("tcp://*")
        self.socket = socket
        self.endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)

    def _publish_load(self, load: int):
        self.socket.send_string(str(load))

    def get_endpoint(self):
        return self.endpoint

    async def _handle_input(self, processor_input_queue):
        while True:
            data = await self.input_queue.get_async()
            await asyncio.to_thread(processor_input_queue.put, data)
            if data is None:
                break

    async def start(self):
        mp.set_start_method("spawn", force=True)
        processor_input_queue = mp.Queue(maxsize=1)
        inner_output_queue = mp.Queue()

        model_process = mp.Process(
            target=start_model_runner,
            args=(
                self.backend,
                self.inference_wrapper,
                self.processing_params,
                self.sampling_params,
                self.parallel_params,
                self.num_processor_workers,
                processor_input_queue,
                inner_output_queue,
            ),
        )
        model_process.start()

        task = asyncio.create_task(self._handle_input(processor_input_queue))

        while True:
            data = await asyncio.to_thread(inner_output_queue.get)
            if data is None:
                break
            await self.output_queue.put_async(data)

        await asyncio.gather(task)
