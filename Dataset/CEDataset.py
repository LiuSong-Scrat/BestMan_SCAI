import os
import base64
import json
import cv2
from openai import AzureOpenAI
import yaml


class CEDataset:
    def __init__(self, img_mode="use_base64", cfg=None):
        # 接受外部传入的 cfg
        self.cfg = cfg
        self.client = AzureOpenAI(
            azure_endpoint=self.cfg['LLM']['azure_endpoint'],
            api_key=self.cfg['LLM']['api_key'],
            api_version=self.cfg['LLM']['api_version'],
        )
        self.img_mode = img_mode

    def encode_image(self, image_path, target_width=800):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image file not found or unreadable: {image_path}")

        original_height, original_width = image.shape[:2]
        aspect_ratio = original_height / original_width
        target_height = int(target_width * aspect_ratio)

        resized_image = cv2.resize(
            image, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
        _, buffer = cv2.imencode(
            ".jpg", resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        return base64.b64encode(buffer).decode("utf-8")

    def analyze_image(self, base64_image=None, url_link=None):
        if self.img_mode == "use_url":
            if not url_link:
                raise ValueError("URL link is required when img_mode is 'use_url'.")
            image_content = {
                "type": "image_url",
                "image_url": {"url": url_link, "detail": "high"},
            }
        elif self.img_mode == "use_base64":
            if not base64_image:
                raise ValueError("Base64 image is required when img_mode is 'use_base64'.")
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpg;base64,{base64_image}",
                    "detail": "high",
                },
            }
        else:
            raise ValueError("Invalid img_mode. Choose 'use_url' or 'use_base64'.")

        messages = [
            {
                "role": "system",
                "content": self.cfg['LLM']['dataset_prompts']['image_analysis'],
            },
            {"role": "user", "content": [image_content]},
        ]

        response = self.client.chat.completions.create(
            model=self.cfg['LLM']['model'],
            messages=messages,
            max_tokens=self.cfg['LLM']['max_tokens'],
            temperature=self.cfg['LLM']['temperature'],
        )

        return response.choices[0].message.content

    def analyze_text(self, prompt):
        messages = [
            {"role": "system", "content": self.cfg['LLM']['dataset_prompts']['text_analysis']},
            {"role": "user", "content": {"type": "text", "text": prompt}},
        ]

        response = self.client.chat.completions.create(
            model=self.cfg['LLM']['model'],
            messages=messages,
            max_tokens=self.cfg['LLM']['max_tokens'],
            temperature=self.cfg['LLM']['temperature'],
        )

        return response.choices[0].message.content

    def save_json_response(self, response_text, save_path):
        try:
            cleaned_text = response_text.strip("```json").strip("```").strip()
            parsed_json = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON: {e}\nResponse: {response_text}")

        # 如果文件存在，追加内容
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as json_file:
                try:
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    existing_data = []

            # 如果是字典，转为列表
            if isinstance(existing_data, dict):
                existing_data = [existing_data]
            existing_data.append(parsed_json)

            with open(save_path, "w", encoding="utf-8") as json_file:
                json.dump(existing_data, json_file, indent=4, ensure_ascii=False)
        else:
            # 文件不存在，直接创建
            with open(save_path, "w", encoding="utf-8") as json_file:
                json.dump([parsed_json], json_file, indent=4, ensure_ascii=False)

        print(f"Data successfully saved to {save_path}")

if __name__ == "__main__":
    """
    Test example
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #config = os.path.join(current_dir, "../Config/default.yaml")
    config = os.path.join(current_dir, "/home/yixuan/BestMan_Chemistry/Config/default.yaml")
    image_path = "/home/yixuan/det/lang-segment-anything/output_objects/object6.png"
    save_path = "newobjects_json.json"

    with open(config, 'r', encoding='utf-8') as fin:
        file = yaml.load(fin, Loader=yaml.FullLoader)

    llm_ce = CEDataset(cfg=file)

    # Encode image to base64
    image = llm_ce.encode_image(image_path)

    # Analyze the image
    text = llm_ce.analyze_image(base64_image=image)

    # Modify the response to update the "source" field to the image path
    try:
        cleaned_text = text.strip("```json").strip("```").strip()
        parsed_json = json.loads(cleaned_text)

        # Update the "source" field
        for container in parsed_json.get("containers", []):
            container["source"] = image_path

        for chemical in parsed_json.get("chemicals", []):
            chemical["source"] = image_path

        # Save the updated response
        llm_ce.save_json_response(json.dumps(parsed_json, indent=4), save_path)
    except json.JSONDecodeError as e:
        print(f"Failed to process the model's response: {e}\nResponse: {text}")
