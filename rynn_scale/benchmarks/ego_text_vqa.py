import json
import os

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY


@BENCHMARK_REGISTRY.register()
class EgoTextVQAIndoor(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}

        data_path = os.path.join(data_root, "data", "indoor", "total.json")
        video_dir = os.path.join(data_root, "data", "indoor", "fps6_video")

        with open(data_path, "r") as f:
            json_data = json.load(f)

        for i, item in enumerate(json_data[0]["data"]):
            video_path = os.path.join(video_dir, f"{item['video_id']}.mp4")
            if not os.path.exists(video_path):
                continue

            data_dict[str(i)] = {
                # Required fields for data loading
                "videos": [video_path],
                # Required fields for evaluation
                "task_type": item["question_type"],
                "ground_truth": item["correct_answer"],
                # Custom fields for instruction generation
                "question": item["question"],
            }

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        prompt = (
            "You are a person in the situation shown in the following consecutive images from a video. \n"
            "You can answer questions that humans ask to help them make decisions. "
            "Now you are observing your surroundings and answering questions based on the current situation. "
            "Understanding the scene text around you is important for answering questions. "
            "Answer the questions in the first-person perspective. "
            "Answer the question as detailed as possible, covering all relevant aspects and providing comprehensive context."
            f"\n\nQuestion: {meta_data['question']}"
        )
        contents = [{"type": "video", "video": video} for video in meta_data["videos"]]
        contents.append({"type": "text", "text": prompt})
        instruction = [{"role": "user", "content": contents}]
        return instruction

    async def process_response(self, data_id, response):
        return response.strip()

    async def _gpt_evaluate(self, question, ground_truth, prediction):
        def remove_special_characters(text):
            return text.replace("\n", "")

        ground_truth = remove_special_characters(str(ground_truth)).lower()
        prediction = remove_special_characters(str(prediction)).lower()

        try:
            system_prompt = (
                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Focus on the meaningful match between the predicted answer and the correct answer. "
                "Please note that not only matches of noun phrases between answers, but also matches of prepositional phrases. "
                'For example, "at the car wash on your right" does not exactly match "car wash". '
                '"at the gas station beside the sign \'gas sale\'" does not exactly match "gas station".\n'
                "- Consider synonyms or paraphrases as valid matches. "
                "Note that the predicted answer must be consistent with the string type of the correct answer, which may include phone numbers, email addresses, numbers, dates, etc. "
                'For example, the string types of "www.usps.com" and "visit their website" are inconsistent, '
                'and the string types of "9849041316" and "advertiser\'s contact number" are inconsistent.\n'
                "- Evaluate the correctness of the prediction compared to the answer."
            )

            user_prompt = (
                "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {ground_truth}\n"
                f"Predicted Answer: {prediction}\n\n"
                "Provide your eval_code only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. "
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'pred': 'yes', 'score': 5}, {'pred': 'no', 'score': 1}."
            )

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini-0718-global",  # Use the correct IDEALAB model name
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=50,
            )

            response_text = response.choices[0].message.content.strip()
            import ast

            eval_dict = ast.literal_eval(response_text)

            return {
                "is_correct": int(eval_dict.get("pred", "no").lower() == "yes") * 100.0,
                "score": eval_dict.get("score", 0.0),
            }

        except Exception as e:
            print(f"GPT evaluation error: {e}")
            pred_lower = prediction.lower().strip()
            gt_lower = ground_truth.lower().strip()
            is_match = pred_lower == gt_lower or pred_lower in gt_lower or gt_lower in pred_lower
            return {
                "is_correct": int(is_match) * 100.0,
                "score": 5.0 if is_match else 0.0,
            }

    async def get_matching_score(self, data_id, prediction):
        meta_data = self.data_dict[data_id]
        ground_truth = meta_data["ground_truth"]
        question = meta_data["question"]
        eval_result = await self._gpt_evaluate(question, ground_truth, prediction)
        return eval_result

    def compute_metrics(self, results):
        scoring_results, correctness_results = [], []
        for result in results:
            score = result.pop("score")
            scoring_results.append({**result, "score": score["score"]})
            correctness_results.append({**result, "score": score["is_correct"]})

        metrics = {
            "Score": self._summarize_scores(scoring_results, category_key="task_type"),
            "Accuracy": self._summarize_scores(correctness_results, category_key="task_type"),
        }

        return metrics
