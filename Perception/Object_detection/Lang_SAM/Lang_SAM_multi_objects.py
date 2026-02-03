import numpy as np
from PIL import Image

from lang_sam.models.gdino import GDINO
from lang_sam.models.sam import SAM
import os
import json
import cv2
import matplotlib.pyplot as plt
from Utils import *
import shutil

class LangSAM:
    def __init__(self, sam_type="sam2.1_hiera_small", ckpt_path: str | None = None):
        self.sam_type = sam_type
        self.sam = SAM()
        self.sam.build_model(sam_type, ckpt_path)
        self.gdino = GDINO()
        self.gdino.build_model()

    def predict(
        self,
        images_pil: list[Image.Image],
        texts_prompt: list[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        """Predicts masks for given images and text prompts using GDINO and SAM models.

        Parameters:
            images_pil (list[Image.Image]): List of input images.
            texts_prompt (list[str]): List of text prompts corresponding to the images.
            box_threshold (float): Threshold for box predictions.
            text_threshold (float): Threshold for text predictions.

        Returns:
            list[dict]: List of results containing masks and other outputs for each image.
            Output format:
            [{
                "boxes": np.ndarray,
                "scores": np.ndarray,
                "masks": np.ndarray,
                "mask_scores": np.ndarray,
            }, ...]
        """

        gdino_results = self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)
        all_results = []
        sam_images = []
        sam_boxes = []
        sam_indices = []
        for idx, result in enumerate(gdino_results):
            processed_result = {
                **result,
                "masks": [],
                "mask_scores": [],
            }

            if result["labels"]:
                processed_result["boxes"] = result["boxes"].cpu().numpy()
                processed_result["scores"] = result["scores"].cpu().numpy()
                sam_images.append(np.asarray(images_pil[idx]))
                sam_boxes.append(processed_result["boxes"])
                sam_indices.append(idx)

            all_results.append(processed_result)
        if sam_images:
            print(f"Predicting {len(sam_boxes)} masks")
            masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
            for idx, mask, score in zip(sam_indices, masks, mask_scores):
                all_results[idx].update(
                    {
                        "masks": mask,
                        "mask_scores": score,
                    }
                )
            print(f"Predicted {len(all_results)} masks")
        return all_results
    
def Det_res(image, text_prompt="objects"):

    model = LangSAM()
    out = model.predict(
        [image],
        [text_prompt],
    )
    # 预测出的 bounding box 列表，格式为 [x_min, y_min, x_max, y_max]
    det_res = out[0]
    return det_res

def Save_mask_bbox_image(det_res, image, save_path):

    input_image = image
    mask_image_path = save_path

    # 可视化每个 bounding box
    bboxes = det_res["boxes"]
    masks = det_res["masks"]
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)  # 转换为整数
        color = (255, 0, 0)  # 红色框，格式为 (R, G, B)
        thickness = 3  # 边框厚度
        input_image = cv2.rectangle(input_image, (x_min, y_min), (x_max, y_max), color, thickness)
    colors = [255,0 , 0]
    overlay = input_image.copy()

    # 遍历每个 mask，将其绘制在叠加图层上
    for i, mask in enumerate(masks):
        color = colors
        overlay[mask == 1] = color  # 将 mask 区域设置为对应颜色

    # 将叠加层与原图进行透明度混合
    alpha = 0.6  # 透明度系数
    result = cv2.addWeighted(overlay, alpha, input_image, 1 - alpha, 0)
    cv2.imwrite(mask_image_path, result)

def Split_instance_object_images(det_res, image, save_path):
    input_image = image
    json_data = {}

    det_res['scores'] = np.atleast_1d(det_res['scores'])
    det_res['labels'] = np.atleast_1d(det_res['labels'])
    det_res['boxes'] = np.atleast_1d(det_res['boxes'])
    det_res['masks'] = np.atleast_1d(det_res['masks'])
    det_res['mask_scores'] = np.atleast_1d(det_res['mask_scores'])

    # 遍历每个对象
    for i, (score, label, box, mask, mask_score) in enumerate(zip(det_res['scores'], det_res['labels'], det_res['boxes'], det_res['masks'], det_res['mask_scores'])):
        x1, y1, x2, y2 = map(int, box)  # 转为整数
        object_image = input_image[y1:y2, x1:x2]  # 根据 bounding box 裁剪

        # 保存对象图像
        object_name = f"object{i+1}.png"
        object_path = os.path.join(output_dir, object_name)
        cv2.imwrite(object_path, object_image)

        # 保存对象信息到 JSON
        json_data[f"object{i+1}"] = {
            "score": float(score),
            "label": label,
            "box": box.tolist(),
            #"mask": mask.tolist(),
            "mask_score": float(mask_score),
            "image_path": object_path
        }

    # 保存 JSON 文件
    with open(os.path.join(save_path, "objects_metadata.json"), "w") as f:
        json.dump(json_data, f, indent=4)
    object_path = os.path.join(save_path, "input_image.png")
    cv2.imwrite(object_path, input_image)
    print(f"所有对象图像和 JSON 文件已保存到 {output_dir}")

if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    input = Submodule()
    pkl_file = "/home/yan/BestMan_Chemistry_Test_SONG/Perception/Object_detection/Lang_SAM/data.pkl"
    input.deserialize(pkl_file)
    image = input.get("input_img", Image.Image)

    output_dir = "/home/yan/BestMan_Chemistry_Test_SONG/Perception/Object_detection/Lang_SAM/output_objects"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    data = Det_res(image = image)
    mask_image_path = os.path.join(output_dir, "mask_image.png")
    image = np.array(image)

    Save_mask_bbox_image(det_res=data, image = image, save_path=mask_image_path)
    # 假设输入图像路径
    Split_instance_object_images(det_res=data, image=image,save_path=output_dir)