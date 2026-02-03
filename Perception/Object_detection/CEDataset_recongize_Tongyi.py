# import os
# from openai import OpenAI
# import base64
# import os

# client = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
#     api_key="sk-f501e73e8f214b739b7403a7f874bb5f",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

# #  Base64 编码格式
# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")


# image_path = "/home/yan/BestMan_Chemistry_Test_SONG/Perception/Object_detection/Lang_SAM/output_objects/object1.png"


# base64_image = encode_image(image_path)

# completion = client.chat.completions.create(
#     model="qwen-vl-plus",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#     messages=[{"role": "user","content": [
#             {"type": "text","text": "这是什么"},
#             {"type": "image_url",
#              "image_url": {"url": f"data:image/png;base64,{base64_image}"},}
#             ]}]
#     )
# print(completion.model_dump_json())







import os
import base64
import json
import cv2
from openai import AzureOpenAI
from openai import OpenAI
import yaml
import time

from Utils import *

class CEDataset:
    def __init__(self, img_mode="use_base64", cfg=None):
        # 接受外部传入的 cfg
        self.cfg = cfg
        self.client = AzureOpenAI(
            azure_endpoint=self.cfg['LLM']['azure_endpoint'],
            api_key=self.cfg['LLM']['api_key'],
            api_version=self.cfg['LLM']['api_version'],
        )
        self.client_tongyi = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key="sk-f501e73e8f214b739b7403a7f874bb5f",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
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



    def analyze_image(self, base64_image=None, other_prompt = None, url_link=None):
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
                "content": "You are a helpful assistant that can analyze images and strictly follow the user's instructions.",
            },
            {"role": "user", "content": [image_content, {
          "type": "text",
          "text": self.cfg['LLM']['dataset_prompts']['image_analysis']
        }] },
        ]

        # response = self.client.chat.completions.create(
        #     model=self.cfg['LLM']['model'],
        #     messages=messages,
        #     max_tokens=self.cfg['LLM']['max_tokens'],
        #     temperature=self.cfg['LLM']['temperature'],
        # )
        
        response = self.client_tongyi.chat.completions.create(
            model="qwen2.5-vl-32b-instruct",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=messages,
            max_tokens=self.cfg['LLM']['max_tokens'],
            temperature=self.cfg['LLM']['temperature'],
        )
        # print(response.model_dump_json())
        # print(json.loads(response.model_dump_json())['choices'][0]['message']['content'])

        
        return json.loads(response.model_dump_json())['choices'][0]['message']['content']
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

def get_unique_filename(directory, base_name, extension):
    """根据 base_name 生成唯一文件名，避免重名"""
    counter = 1
    new_name = f"{base_name}{extension}"
    while os.path.exists(os.path.join(directory, new_name)):
        new_name = f"{base_name}_{counter}{extension}"
        counter += 1
    return new_name

if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    input = Submodule()
    pkl_file = os.path.abspath("./data.pkl")
    input.deserialize(pkl_file)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = os.path.join(current_dir, "/home/yan/BestMan_Chemistry_Test_SONG/Config/default.yaml")
    image_path = "/home/yan/BestMan_Chemistry_Test_SONG/Perception/Object_detection/Lang_SAM/output_objects"
    load_json_path = "/home/yan/BestMan_Chemistry_Test_SONG/Perception/Object_detection/Lang_SAM/output_objects/objects_metadata.json"
    save_path = "/home/yan/BestMan_Chemistry_Test_SONG/Perception/Object_detection/Lang_SAM/output_objects/objects_gpt.json"

    with open(config, 'r', encoding='utf-8') as fin:
        file = yaml.load(fin, Loader=yaml.FullLoader)

    with open(load_json_path, "r") as f:
        data = json.load(f)
    #print(data['object5'])

    llm_ce = CEDataset(cfg=file)
    for i in range(3):
        # 遍历每个对象并提取 name 和 box
        for object_name, object_info in data.items():
            image_name = str(object_name)+".png"
            image_path_new = os.path.join(image_path, image_name)
            box = object_info["box"]
            start = time.time()
            image = llm_ce.encode_image(image_path_new)

            # Analyze the image
            text = llm_ce.analyze_image(base64_image=image, other_prompt=box)
            try:
                cleaned_text = text.strip("```json").strip("```").strip() #tongyi doesn't need to cut
                parsed_json = json.loads(cleaned_text)
                # Update the "source" field
                for container in parsed_json.get("containers", []):
                    container["source"] = image_path_new

                for chemical in parsed_json.get("chemicals", []):
                    chemical["source"] = image_path_new

                for others in parsed_json.get("others", []):
                    others["source"] = image_path_new

                # Save the updated response
                llm_ce.save_json_response(json.dumps(parsed_json, indent=4), save_path)
                
            except json.JSONDecodeError as e:
                print(f"Failed to process the model's response: {e}\nResponse: {text}")
            end = time.time()    # 程序结束时间
            print("end_time:", end)
            run_time = end - start   # 程序的运行时间，单位为秒
            print("run_time:", run_time)
    # 1. 读取JSON文件
    with open(save_path, 'r') as f:
        data = json.load(f)  # 注意这里用你的实际文件路径

    # 2. 创建结果字典
    item_bbox_dict = {}
    item_info_dict = {}
    item_count_dict = {}

    for entry in data:
        # 处理容器类物品
        for container in entry["containers"]:
            N = container["name"]
            if N not in item_count_dict:
                item_count_dict[N] = 1
            else:
                item_count_dict[N] += 1
            item_bbox_dict[N+' '+str(item_count_dict[N])] = container["bounding box"]
            if "containing" in container:
                item_info_dict[N+' '+str(item_count_dict[N])] = {}
                item_info_dict[N+' '+str(item_count_dict[N])]["containing"] = container["containing"]
        
        # 处理化学品
        for chemical in entry["chemicals"]:
            N = chemical["name"]
            if N not in item_count_dict:
                item_count_dict[N] = 1
            else:
                item_count_dict[N] += 1
            item_bbox_dict[N+' '+str(item_count_dict[N])] = chemical["bounding box"]
        
        # 处理其他物品
        for other in entry["others"]:
            N = other["name"]
            if N not in item_count_dict:
                item_count_dict[N] = 1
            else:
                item_count_dict[N] += 1
            item_bbox_dict[N+' '+str(item_count_dict[N])] = other["bounding box"]

    #-----------SONG-Q-0326----------#
    sorted_objects = dict(sorted(item_bbox_dict.items(), key=lambda item: (-item[1][3], item[1][0])))

    input.clear()
    input.add("sorted_objects", sorted_objects)
    input.add("objects_info", item_info_dict)
    input.serialize(pkl_file)
