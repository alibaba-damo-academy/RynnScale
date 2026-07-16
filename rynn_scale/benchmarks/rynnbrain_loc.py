import os
from collections import defaultdict
from typing import Any, Dict, List, Union
import json
import re
import math
import numpy as np

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY

TASKS = {
    "Referring": "rynnbrain_referring_2000.jsonl",
    "trajectory": "rynnbrain_traj_2000.jsonl",
    "affordance": "rynnbrain_affordance_2000.jsonl",
    "area": "rynnbrain_area_2000.jsonl",
}


@BENCHMARK_REGISTRY.register()
class RynnBrainLoc(BaseBenchmark):
    # THINKING_MODE = True
    THINKING_MODE = False

    REFERRING_TYPES = ["situational referring", "direct referring"]
    AREA_TYPES = ["area"]
    TRAJ_TYPES = ["traj"]
    AFFORDANCE_TYPES = ["affordance"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_jsonl_file(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"JSONL file not found: {path}")
        data_list = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
                data_list.append(obj)
        return data_list

    def load_data(self, data_root: str) -> Dict[Union[str, int], Any]:
        """
        Load data from the ERQA TFRecord file.
        data_root is expected to be the path to the .tfrecord file.
        """
        data_dict = {}

        for task_name, json_path in TASKS.items():
            data_path = os.path.join(data_root, json_path)
            data_folder = os.path.join(data_root, "data")
            # data_folder = os.path.join(data_root, "localization") # for test

            # print('data_path', data_path)
            data_list = self._load_jsonl_file(data_path)
            # print('len(data_list)',len(data_list))
            for i, data in enumerate(data_list):
                # if i<1:
                    # continue
                # if i>1:
                #     break
                data_id = data.get("id", 0)
                data_id = f"{task_name}_{data_id}"
                task_type = data.get("task_type", "unknown")
                conversation = data.get("conversation", [])


                # resolve data
                for msg in conversation:
                    if msg["role"] == "assistant":
                        # print('msg["content"]', msg["content"])
                        assistant_content = msg.get('content', [])
                        last_content = assistant_content[-1] if assistant_content else {}
                        # answer = assistant_content[-1].get("text", "") if assistant_content else ""
                        answer = (
                            last_content.get("text")
                            or last_content.get("area")
                            or last_content.get("traj")
                            or last_content.get("affordance")
                            or ""
                        )
                    elif msg["role"] == "user":
                        user_content = msg.get('content', [])
                        image_path = [os.path.join(data_folder, item['image']) for item in user_content if item.get('type') == 'image']
                        question = user_content[-1].get("text", "") if user_content else ""

                data_dict[data_id] = {
                    "images": image_path,
                    "ground_truth": answer,
                    "task_type": task_type,
                    "question": question,
                    # custom fields
                }
                # print('data_dict[data_id]',data_dict[data_id])

        print(f"Loaded {len(data_dict)} samples from {data_root}.")
        return data_dict

    def generate_instruction(self, data_id: Union[int, str]) -> List[Dict[str, Any]]:
        """Generate instruction for model inference (the question itself)."""
        meta_data = self.data_dict[data_id]
        question = meta_data["question"].rstrip(".")
        image_path = meta_data["images"]
        task_type = meta_data["task_type"]

        if self.prompt_format == "RynnBrain":
            if task_type in self.REFERRING_TYPES:
                prompt = f"Output the bounding box in the format <object> <frame n>: ...; (x1,y1), (x2,y2) </object>. n is the chosen frame index."
                thinking_prompt = "\nOutput format: `#### <answer><object><frame i> (X_min, Y_min), (X_max, Y_max) </object></answer>`. Coordinates normalized to 0-1000."
                self.SYSTEM_PROMPT = SYSTEM_PROMPT_REFERRING
            elif task_type in self.TRAJ_TYPES:
                prompt = f"First predict the frame containing the trajectory start point, then output up to 10 key trajectory points as a list of tuples in the format: <trajectory> <frame n>: ...; (x1, y1), (x2, y2), .... </trajectory> All coordinates must be normalized between 0 and 1000."
                thinking_prompt = f"\nFirst predict the frame containing the trajectory start point, then output up to 10 key trajectory points as a list of tuples.\nOutput format: `#### <answer><trajectory><frame i> (X_1, Y_1), (X_2, Y_2), ..., (X_N, Y_N) </trajectory></answer>`. Coordinates normalized to 0-1000."
                self.SYSTEM_PROMPT = SYSTEM_PROMPT_TRAJ
            elif task_type in self.AFFORDANCE_TYPES:
                prompt = f"First predict the key frame, then output a single affordance point as coordinates (x, y).\nOutput format: <affordance> <frame n>: ...; (x, y) </affordance>\n Both x and y values must be normalized between 0 and 1000."
                thinking_prompt = f"\nFirst predict the key frame, then output a single affordance point as coordinates (x, y).\nOutput format: `#### <answer><affordance><frame i> (X, Y) </affordance></answer>`. Coordinates normalized to 0-1000."
                self.SYSTEM_PROMPT = SYSTEM_PROMPT_AFFORDANCE
            elif task_type in self.AREA_TYPES:
                prompt = f"First predict the key frame, then output coordinates as a series of tuples. \nOutput format: <area> <frame n>: ...; (x1, y1), (x2, y2), .... </area>\n All coordinates must be normalized between 0 and 1000."
                thinking_prompt = f"\nFirst predict the key frame, then output coordinates as a series of tuples.\nOutput format: `#### <answer><area><frame i> (X_1, Y_1), ... </area></answer>`. Coordinates normalized to 0-1000."
                self.SYSTEM_PROMPT = SYSTEM_PROMPT_AREA
        elif self.prompt_format == "RynnBrain1.1":
            if task_type in self.REFERRING_TYPES:
                prompt = "Predict the bounding box. Output the result in JSON format."
                # prompt = 'Output the result in JSON format. Example:\n```json\n[{"bbox": [x_min, y_min, x_max, y_max], "frame_idx": 0, "label": "object"}]\n```'
                thinking_prompt = "\nOutput format: `#### <answer><object><frame i> (X_min, Y_min), (X_max, Y_max) </object></answer>`. Coordinates normalized to 0-1000."
                self.SYSTEM_PROMPT = SYSTEM_PROMPT_REFERRING
            elif task_type in self.TRAJ_TYPES:
                prompt = "Predict the trajectory points. Output the result in JSON format."
                # prompt = "Execute sequential trajectory analysis:\n1. Identify the frame containing trajectory initiation point\nOutput the result in JSON format."
                thinking_prompt = f"\nFirst predict the frame containing the trajectory start point, then output up to 10 key trajectory points as a list of tuples.\nOutput format: `#### <answer><trajectory><frame i> (X_1, Y_1), (X_2, Y_2), ..., (X_N, Y_N) </trajectory></answer>`. Coordinates normalized to 0-1000."
                self.SYSTEM_PROMPT = SYSTEM_PROMPT_TRAJ
            elif task_type in self.AFFORDANCE_TYPES:
                prompt = "Predict the affordance point. Output the result in JSON format."
                # prompt = "Follow this exact sequence:\n1. Predict key frame\n2. Output one affordance point as Python tuple\nOutput the result in JSON format."
                thinking_prompt = f"\nFirst predict the key frame, then output a single affordance point as coordinates (x, y).\nOutput format: `#### <answer><affordance><frame i> (X, Y) </affordance></answer>`. Coordinates normalized to 0-1000."
                self.SYSTEM_PROMPT = SYSTEM_PROMPT_AFFORDANCE
            elif task_type in self.AREA_TYPES:
                prompt = "Predict the area polygon. Output the result in JSON format."
                # prompt = "Follow this exact sequence:\n1. Predict key frame\n2. Output tuple series\nOutput the result in JSON format."
                thinking_prompt = f"\nFirst predict the key frame, then output coordinates as a series of tuples.\nOutput format: `#### <answer><area><frame i> (X_1, Y_1), ... </area></answer>`. Coordinates normalized to 0-1000."
                self.SYSTEM_PROMPT = SYSTEM_PROMPT_AREA
        else:
            raise NotImplementedError

        if self.THINKING_MODE:

            instruction = f'{question}{thinking_prompt}' if question[-1] == "." else f'{question}.{thinking_prompt}'
            # instruction = f'{question}.{thinking_prompt}'
            # if task_type in self.AFFORDANCE_TYPES or task_type in self.AREA_TYPES or task_type in self.REFERRING_TYPES:
            #     instruction = f'{question}. {prompt}{thinking_prompt}'
        else:
            instruction = f'{question}. {prompt}'
        # print('instruction', instruction)

        if isinstance(instruction, str):
            if not isinstance(image_path, list):
                image_path = [image_path]
            content = []
            for i, path in enumerate(image_path):
                content.append({"type": "text", "text": f"<frame {i}>: "})
                content.append({"type": "image", "image": path})

            messages = []
            if self.THINKING_MODE:
                messages.append({"role": "system", "content": self.SYSTEM_PROMPT})
            messages.append(
                {
                    "role": "user",
                    "content": content + [{"type": "text", "text": instruction}],
                }
            )
            # print('messages', messages)
        return messages

    async def process_response(self, data_id: Union[int, str], response: str) -> Any:
        """Process the raw model response."""
        if self.prompt_format == "RynnBrain":
            # Normalize the response similarly to the ground truth for fair comparison
            raw_response = response.strip()
            meta_data = self.data_dict[data_id]
            image_path = meta_data["images"]

            # resolve frame
            # match = re.search(r"<frame\s*(\d+)>", outputs)
            # matches = re.findall(r"<frame\s*(\d+)>", outputs)
            # match = list(re.finditer(r"<frame\s*(\d+)>:", outputs))
            match = list(re.finditer(r"<frame\s*(\d+)>:?", raw_response))
            total_frame = len(image_path)
            if match:
                if 0 <= int(match[-1].group(1)) < total_frame:
                    frame_idx = int(match[-1].group(1))  # 提取并转为整数
                    # print('Predicted frame idx', frame_idx)
                else:
                    frame_idx = total_frame - 1
                    # print('Predicted frame out of index, using the last frame', frame_idx)
            else:
                frame_idx = total_frame - 1
                # print("Frame ID not found, using the last frame", frame_idx)

            if self.THINKING_MODE:
                match = re.findall(r'<answer>(.*?)</answer>', raw_response, re.DOTALL)
                response = match[0].strip() if match else raw_response

            pm = [(int(x), int(y)) for x, y in re.findall(r"\((\d+)\s*,\s*(\d+)\)", response)][:10]
            
            results = json.dumps(
                {
                    'frame_idx': frame_idx,
                    "response": raw_response,
                    'outputs': pm
                }
            )

        elif self.prompt_format == "RynnBrain1.1":
            pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            match = re.search(pattern, response)
            if match:
                response = match.group(1)
            response = response.strip()

            meta_data = self.data_dict[data_id]
            task_type = meta_data["task_type"]
            total_frame = len(meta_data["images"])

            try:
                results = json.loads(response)
                if isinstance(results, list):
                    results = results[0]
                frame_idx = results["frame_idx"]

                if task_type in self.REFERRING_TYPES:
                    points = results["bbox"]
                elif task_type in self.TRAJ_TYPES:
                    points = results["trajectory"]
                elif task_type in self.AFFORDANCE_TYPES:
                    points = [results["point"]]
                elif task_type in self.AREA_TYPES:
                    points = results["polygon"]
                else:
                    raise NotImplementedError

            except Exception:
                import traceback; traceback.print_exc()
                print(f"Match failed: {response}")
                frame_idx = total_frame - 1
                points = [[0, 0]]

            results = json.dumps(
                {
                    "frame_idx": frame_idx,
                    "response": response,
                    "outputs": points,
                }
            )

        else:
            raise NotImplementedError

        return results
    
    async def get_matching_score(self, data_id, prediction):
        """
        Compute the matching score between model prediction and ground truth.
        """
        meta_data = self.data_dict[data_id]
        ground_truth = meta_data["ground_truth"]
        task_type = meta_data["task_type"]
        image_path = meta_data["images"]

        prediction = json.loads(prediction)
        frame_idx = prediction["frame_idx"]
        pred_points = prediction["outputs"]

        if task_type in self.REFERRING_TYPES:
            label = process_referring_label(ground_truth, image_path, frame_idx)
            score = calculate_iou_bbox(pred_points[-2:], label) # 取最后一个prediction计算
        elif task_type in self.TRAJ_TYPES:
            label = process_area_traj_affor_label(ground_truth, frame_idx)
            pred_points = [(pred[0]/1000, pred[1]/1000) for pred in pred_points if len(pred)==2] # normalize
            score = calculate_dfd(pred_points, label, num_samples = 15)
        elif task_type in self.AFFORDANCE_TYPES:
            label = process_area_traj_affor_label(ground_truth, frame_idx)
            pred_points = [(pred[0]/1000, pred[1]/1000) for pred in pred_points if len(pred)==2] # normalize
            score = calculate_nearest_distances(pred_points, label)
        elif task_type in self.AREA_TYPES:
            label = process_area_traj_affor_label(ground_truth, frame_idx)
            pred_points = [(pred[0]/1000, pred[1]/1000) for pred in pred_points if len(pred)==2] # normalize
            score = calculate_points_in_polygon(pred_points, label)
        else:
            raise NotImplementedError
        print(f'| dataid: {data_id} | pred: {prediction} | score: {score} |')
        return score

    def compute_metrics(self, results):
        sample_scores = defaultdict(list)
        for data in results:
            data_id = data["data_id"]
            score = data["score"]
            task_type = self.data_dict[data_id]['task_type']
            sample_scores[task_type].append(score)

        task_metrics = {}
        for cat, scores in sample_scores.items():
            if cat in self.REFERRING_TYPES:
                task_metrics[cat] = sum(score > 0.5 for score in scores) / len(scores)  # ACC@0.5
            elif cat in self.TRAJ_TYPES:
                task_metrics[cat] = sum(scores) / len(scores)
            elif cat in self.AFFORDANCE_TYPES:
                task_metrics[cat] = sum(scores) / len(scores)
            elif cat in self.AREA_TYPES:
                task_metrics[cat] = sum(scores) / len(scores)

        # Object referring metrics
        referring_scores = [
            score for t, scores in sample_scores.items()
            if t in self.REFERRING_TYPES
            for score in scores
        ]
        referring_metrics = (
            {'Object Referring': sum(score > 0.5 for score in referring_scores) / len(referring_scores)}
            if referring_scores
            else {}
        )

        metrics = {**referring_metrics, **task_metrics}

        return metrics

SYSTEM_PROMPT_REFERRING = [
    {
        "type": "text",
        "text": (
            "You are an embodied agent. You are given a video to solve an object detection problem. "
            "Put your final answer in the format of `#### <answer><object><frame i> (X_min, Y_min), (X_max, Y_max) </object></answer>`."
        ),
    }
]    

SYSTEM_PROMPT_TRAJ = [
    {
        "type": "text",
        "text": (
            "You are an embodied agent. You are given a video to solve a trajectory prediction problem."
            "Put your final answer in the format of `#### <answer><trajectory><frame i> (X_1, Y_1), (X_2, Y_2), ..., (X_N, Y_N) </trajectory></answer>`."
        ),
    }
]

SYSTEM_PROMPT_AFFORDANCE = [
    {
        "type": "text",
        "text": (
            "You are an embodied agent. You are given a video to solve an affordance prediction problem."
            "Put your final answer in the format of `#### <answer><affordance><frame i> (X, Y) </affordance></answer>`."
        ),
    }
]

SYSTEM_PROMPT_AREA = [
    {
        "type": "text",
        "text": (
            "You are an embodied agent. You are given a video to solve an area prediction problem."
            "Put your final answer in the format of `#### <answer><area><frame i> (X_1, Y_1), ..., (X_N, Y_N) </area></answer>`."
        ),
    }
]


def process_area_traj_affor_label(point_list: List[dict], frame_idx: int):
    # # affordance、traj、area label 格式目前为[yx]->[xy]
    # frame_count = sum(1 for item in user_text[0]['content'] if item['type'] == 'image')
    # label = [(item[1],item[2]) for item in point_list if round(item[0] * (frame_count-1)) == frame_idx]

    # affordance、traj、area label 格式目前均为[xy], 0-1000
    label = [(item[1] / 1000, item[2] / 1000) for item in point_list if item[0] == frame_idx]

    # print('image_count', frame_count)
    # print('frame_idx', frame_idx)
    # print('user_text', user_text)
    # print('point_list', point_list)
    # print('label', label)
    return label


def process_referring_label(point_list: List[dict], image_path: List, frame_idx: int):
    # extract img_name from user_msg
    # image_name_list = [
    #     (os.path.basename(item['image']).split('.')[0].lstrip('0') or '0')
    #     for item in user_msg[0]['content'] if item['type'] == 'image'
    # ]
    image_name_list = [(os.path.basename(item).split(".")[0].lstrip("0") or "0") for item in image_path]
    # print('image_name_list', image_name_list)
    img_name = image_name_list[frame_idx]
    # print('img_name', img_name)

    label = point_list[0].get(img_name, None)
    if label:
        label = [(label[0], label[1]), (label[2], label[3])]
    return label


def calculate_iou_bbox(box1, box2):
    """
    计算两个边界框的IoU。

    参数:
    box1: 第一个边界框，格式为 [(x1, y1), (x2, y2)]
    box2: 第二个边界框，格式为 [(x1, y1), (x2, y2)]

    返回:
    iou: 交并比，浮点数，范围在 [0, 1] 之间
    """
    if not box1 or not box2:
        return 0.0
    
    if len(box1) == 0 or len(box2) == 0:
        return math.exp(-float("inf"))

    # 1. 获取每个 box 的坐标
    (x1_a, y1_a), (x2_a, y2_a) = box1[0], box1[1]
    (x1_b, y1_b), (x2_b, y2_b) = box2[0], box2[1]

    # 2. 计算交集区域的坐标
    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    # 3. 计算交集面积
    # 如果两个框没有重叠，宽度或高度会是负数，取 0
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height

    # 4. 计算两个原始框的面积
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)

    # 5. 计算并集面积
    union_area = area_a + area_b - intersection_area

    # 6. 计算 IoU
    # 避免除以零的错误
    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area

    return iou


def calculate_dfd(seq1, seq2, num_samples=15, dist_func=None):
    """
    discrete_frechet_distance
    seq1: 第一个序列，形状为 (m, d) 的数组，m是点数，d是维度
    seq2: 第二个序列，形状为 (n, d) 的数组，n是点数，d是维度

    return:
    float: 两个序列的离散弗雷歇距离
    """

    if not seq1 or not seq2:
        return math.exp(-float("inf"))

    if len(seq1) == 0 or len(seq2) == 0:
        return math.exp(-float("inf"))

    seq1 = uniform_trajectory_sampling(seq1, num_samples)
    seq2 = uniform_trajectory_sampling(seq2, num_samples)

    seq1 = np.array(seq1)
    seq2 = np.array(seq2)
    m, n = len(seq1), len(seq2)

    if dist_func is None:

        def dist_func(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

    C = np.zeros((m, n))
    C[0, 0] = dist_func(seq1[0], seq2[0])

    for i in range(1, m):
        C[i, 0] = max(C[i - 1, 0], dist_func(seq1[i], seq2[0]))

    for j in range(1, n):
        C[0, j] = max(C[0, j - 1], dist_func(seq1[0], seq2[j]))

    for i in range(1, m):
        for j in range(1, n):
            C[i, j] = max(min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1]), dist_func(seq1[i], seq2[j]))

    return math.exp(-C[m - 1, n - 1])


def uniform_trajectory_sampling(trajectory, n_points):
    """
    轨迹进行均匀采样
    trajectory: 原始轨迹，形状为 (m, d) 的数组，m是点数，d是维度
    n_points: 采样后的点数

    return:
    numpy.array: 采样后的轨迹，形状为 (n_points, d)
    """

    if trajectory is None or len(trajectory) == 0:
        raise ValueError("Trajectory can not be None or empty")

    if n_points <= 0:
        raise ValueError("采样点数必须大于0")

    trajectory = np.array(trajectory, dtype=float)

    # 计算每段长度
    segment_lengths = np.sqrt(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1))

    # 累积长度，并在开头插入0
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
    total_length = cumulative_lengths[-1]

    if total_length == 0:
        return np.tile(trajectory[0], (n_points, 1))

    # 生成均匀分布的N个采样点
    sampled_distances = np.linspace(0, total_length, n_points)

    # np.interp需要一维数据，分别对x和y（以及更高维度）坐标进行插值
    sampled_points = np.zeros((n_points, trajectory.shape[1]))
    for i in range(trajectory.shape[1]):  # 对每一个维度（x, y, ...）进行操作
        sampled_points[:, i] = np.interp(sampled_distances, cumulative_lengths, trajectory[:, i])

    # 转置并返回结果
    return sampled_points


def calculate_nearest_distances(pred_points, gt_points):
    """
    计算 pred 路径中每个点到 gt 路径中最近点的欧氏距离。

    参数:
    pred_points (list of lists/tuples): 预测的路径点，形状为 [(x1, y1), (x2, y2), ...]。
    gt_points (list of lists/tuples): 真实的路径点 (Ground Truth)，形状类似。

    返回:
    np.ndarray: 一个一维数组，包含了 pred_points 中每个点到 gt_points 的最近距离。
                数组的长度与 pred_points 的长度相同。
    """
    # 检查输入列表是否为空，避免后续计算出错
    if not pred_points or not gt_points:
        return math.exp(-float("inf"))

    pred_arr = np.array(pred_points)
    gt_arr = np.array(gt_points)

    # 计算距离矩阵
    from scipy.spatial.distance import cdist

    dist_matrix = cdist(pred_arr, gt_arr, "euclidean")

    # 对矩阵的每一行（代表 pred 中的一个点）取最小值
    # axis=1 表示沿着行的方向操作，即找出每个 pred 点到所有 gt 点的距离中的最小值
    min_distances = np.min(dist_matrix, axis=1).mean()

    return math.exp(-min_distances)


def calculate_points_in_polygon(pred_points, gt_points):
    """
    判断 pred_list 中每个点是否在由 point_list 定义的多边形内部。

    参数:
        pred_points: list of [x, y]，待判断的预测点列表
        gt_points: list of [x, y]，定义多边形顶点（顺序闭合或不闭合均可）
    返回:
        list of bool，对应每个预测点是否在多边形内部（含边界）
    """
    from shapely.geometry import Polygon, Point

    if not pred_points or not gt_points:
        return 0.0

    if len(pred_points) == 0 or len(gt_points) == 0:
        return 0.0

    polygon = Polygon(gt_points)
    points_inside = sum(1 for p in pred_points if polygon.intersects(Point(p)))
    return points_inside / len(pred_points)
