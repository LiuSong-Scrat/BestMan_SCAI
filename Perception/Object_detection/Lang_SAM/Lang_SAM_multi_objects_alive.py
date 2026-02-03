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

from typing import Optional, Union
from typing import List
from typing import Tuple


import pandas as pd
from sklearn.decomposition import PCA

class LangSAM:
    def __init__(self,  sam_ckpt_path: Union[str, None]= None, gd_ckpt_path: Union[str, None]= None, sam_type="sam2.1_hiera_small"):
        self.sam_type = sam_type
        self.sam = SAM()
        self.sam.build_model(self.sam_type, ckpt_path=sam_ckpt_path)
        self.gdino = GDINO()
        self.gdino.build_model(ckpt_path=gd_ckpt_path)

    def predict(
        self,
        images_pil: List[Image.Image],
        texts_prompt: List[str],
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
    
# def Det_res(image, text_prompt="objects"):
#     #time1 = time.s
#     model = LangSAM()
#     out = model.predict(
#         [image],
#         [text_prompt],
#     )
#     # 预测出的 bounding box 列表，格式为 [x_min, y_min, x_max, y_max]
#     det_res = out[0]
#     return det_res


def Det_res_alive(model, image, text_prompt="objects"):
    
    out = model.predict(
        [image],
        [text_prompt],
    )
    # 预测出的 bounding box 列表，格式为 [x_min, y_min, x_max, y_max]
    det_res = out[0]
    return det_res






class Det_res_alive:
    def __init__(self, text_prompt="objects"):

        # # Online Cost Lots if Time
        self.model_mode = "offline" # online or offline
        if self.model_mode == "offline":
            # Offline 
            self.sam_ckpt_path = "/home/yan/BestMan_Chemistry_Test_SONG/Perception/lang-segment-anything/local_load_ckpt/sam2.1_hiera_small.pt"
            self.gd_path = "/home/yan/BestMan_Chemistry_Test_SONG/Perception/lang-segment-anything/local_load_ckpt/grounding-dino-base/"
        elif self.model_mode == "online":
            # Online 
            self.sam_ckpt_path = None
            self.gd_path = None

        # self.sam_ckpt_path = "/home/yan/BestMan_Chemistry_Test_SONG/Perception/lang-segment-anything/local_load_ckpt/sam2.1_hiera_small.pt"
        # self.gd_path = None

        self.model = LangSAM(sam_ckpt_path = self.sam_ckpt_path,gd_ckpt_path=self.gd_path)
        self.text_prompt = text_prompt

        self.pkl_file = "/home/yan/BestMan_Chemistry_Test_SONG/Perception/Object_detection/Lang_SAM/data.pkl"
        self.output_dir = "/home/yan/BestMan_Chemistry_Test_SONG/Perception/Object_detection/Lang_SAM/output_objects"
        
        self.masks = []

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        

    def Det_inference_res(self,image):
        out = self.model.predict(
            [image],
            [self.text_prompt],
        )
        # 预测出的 bounding box 列表，格式为 [x_min, y_min, x_max, y_max]
        det_res = out[0]
        return det_res
    def linear_regression(self,x, y): 
        N = len(x)
        sumx = sum(x)
        sumy = sum(y)
        sumx2 = sum(x**2)
        sumxy = sum(x*y)
        
        A = np.mat([[N, sumx], [sumx, sumx2]])
        b = np.array([sumy, sumxy])
        
        return np.linalg.solve(A, b)
    
    def Get_mask_pts(self,valid_pts,mask,pt1,pt2,pt3,pt4):
        mask_center =np.array([int(np.average(valid_pts[0])), int(np.average(valid_pts[1]))])

        points = valid_pts.T
        pca = PCA(n_components=2)
        pca.fit(points)
        direction_vector = pca.components_[0]  # 第一主成分的方向向量
        # 计算斜率 k 和截距 b
        slope = direction_vector[1] / direction_vector[0]  # 斜率 (y分量 / x分量)
        mean_point = np.mean(points, axis=0)  # 数据点的均值
        intercept = mean_point[1] - slope * mean_point[0]  # 截距 (通过均值点计算)

        if slope==0:
            print("slope is 0 !!!")
            exit()
        
        move_pt_right = mask_center
        move_pt_left = mask_center
        move_pt_right_ = mask_center
        move_pt_left_ = mask_center

        if abs(slope)<=1:
            while pt1==[] or pt2==[] or pt3==[] or pt4==[]:
                if pt1==[]:
                    next_move_pt_x = move_pt_right[0]+1
                    next_move_pt_y = int(slope * next_move_pt_x + intercept)
                    if next_move_pt_x>mask.shape[1]-1 or next_move_pt_y<0 or next_move_pt_y>mask.shape[0]-1:
                        pt1 = move_pt_right
                    elif  mask[next_move_pt_y][next_move_pt_x]==0:
                        move_pt_right = np.array([next_move_pt_x, next_move_pt_y])
                        pt1 = move_pt_right
                    move_pt_right = np.array([next_move_pt_x, next_move_pt_y])

                if pt2==[]:
                    next_move_pt_x = move_pt_left[0]-1
                    next_move_pt_y = int(slope * next_move_pt_x + intercept)
                    if next_move_pt_x<0 or next_move_pt_y<0 or next_move_pt_y>mask.shape[0]-1:
                        pt2 = move_pt_left
                    elif  mask[next_move_pt_y][next_move_pt_x]==0:
                        move_pt_left = np.array([next_move_pt_x, next_move_pt_y])
                        pt2 = move_pt_left
                    move_pt_left = np.array([next_move_pt_x, next_move_pt_y])

                if pt3==[]:
                    next_move_pt_y = move_pt_right_[1]+1
                    next_move_pt_x = int(mask_center[0]-slope*(next_move_pt_y-mask_center[1]))
                    if next_move_pt_y>mask.shape[0]-1 or next_move_pt_x<0 or next_move_pt_x>mask.shape[1]-1:
                        pt3 = move_pt_right_
                    elif  mask[next_move_pt_y][next_move_pt_x]==0:
                        move_pt_right_ = np.array([next_move_pt_x, next_move_pt_y])
                        pt3 = move_pt_right_
                    move_pt_right_ = np.array([next_move_pt_x, next_move_pt_y])
                    
                   

                if pt4==[]:
                    next_move_pt_y = move_pt_left_[1]-1
                    next_move_pt_x = int(mask_center[0]-slope*(next_move_pt_y-mask_center[1]))
                    if next_move_pt_y<0 or next_move_pt_x<0 or next_move_pt_x>mask.shape[1]-1:
                        pt4 = move_pt_left_
                    elif  mask[next_move_pt_y][next_move_pt_x]==0:
                        move_pt_left_ = np.array([next_move_pt_x, next_move_pt_y])
                        pt4 = move_pt_left_
                    move_pt_left_ = np.array([next_move_pt_x, next_move_pt_y])

        elif abs(slope)>1:
            while pt1==[] or pt2==[] or pt3==[] or pt4==[]:
                if pt1==[]:
                    next_move_pt_y = move_pt_right[1]+1
                    next_move_pt_x = int((next_move_pt_y-intercept)/slope)
                    if next_move_pt_y>mask.shape[0]-1 or next_move_pt_x<0 or next_move_pt_x>mask.shape[1]-1:
                        pt1 = move_pt_right
                    elif  mask[next_move_pt_y][next_move_pt_x]==0:
                        move_pt_right = np.array([next_move_pt_x, next_move_pt_y])
                        pt1 = move_pt_right
                    move_pt_right = np.array([next_move_pt_x, next_move_pt_y])

                if pt2==[]:
                    next_move_pt_y = move_pt_left[1]-1
                    next_move_pt_x = int((next_move_pt_y-intercept)/slope)
                    if next_move_pt_y<0 or next_move_pt_x<0 or next_move_pt_x>mask.shape[1]-1:
                        pt2 = move_pt_left
                    elif  mask[next_move_pt_y][next_move_pt_x]==0:
                        move_pt_left = np.array([next_move_pt_x, next_move_pt_y])
                        pt2 = move_pt_left
                    move_pt_left = np.array([next_move_pt_x, next_move_pt_y])
                    
                if pt3==[]:
                    next_move_pt_x = move_pt_right_[0]+1
                    next_move_pt_y = int(-(next_move_pt_x-mask_center[0])/slope)+mask_center[1] 
                    if next_move_pt_x>mask.shape[1]-1 or next_move_pt_y<0 or next_move_pt_y>mask.shape[0]-1:
                        pt3 = move_pt_right_
                    elif  mask[next_move_pt_y][next_move_pt_x]==0:
                        move_pt_right_ = np.array([next_move_pt_x, next_move_pt_y])
                        pt3 = move_pt_right_
                    move_pt_right_ = np.array([next_move_pt_x, next_move_pt_y])

                if pt4==[]:
                    next_move_pt_x = move_pt_left_[0]-1
                    next_move_pt_y= int(-(next_move_pt_x-mask_center[0])/slope)+mask_center[1] 
                    if next_move_pt_x<0 or next_move_pt_y<0 or next_move_pt_y>mask.shape[0]-1:
                        pt4 = move_pt_left_
                    elif  mask[next_move_pt_y][next_move_pt_x]==0:
                        move_pt_left_ = np.array([next_move_pt_x, next_move_pt_y])
                        pt4 = move_pt_left_
                    move_pt_left_ = np.array([next_move_pt_x, next_move_pt_y])
                  
              
        return pt1,pt2,pt3,pt4,slope
    

    def Save_mask_bbox_image(self, det_res, image, save_path):

        input_image = image
        mask_image_path = save_path

        # 可视化每个 bounding box
        bboxes = det_res["boxes"]
        masks = det_res["masks"]
        self.masks = masks

        for index,bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = map(int, bbox)  # 转换为整数
            color = (255, 0, 0)  # 红色框，格式为 (R, G, B)
            thickness = 3  # 边框厚度
            input_image = cv2.rectangle(input_image, (x_min, y_min), (x_max, y_max), color, thickness)
            input_image = cv2.putText(input_image, str(index+1), (int((x_min+x_max)/2) ,int((y_min+y_max)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
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

    def Split_instance_object_images(self,det_res, image):
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
            object_path = os.path.join(self.output_dir, object_name)
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
        with open(os.path.join(self.output_dir, "objects_metadata.json"), "w") as f:
            json.dump(json_data, f, indent=4)
        object_path = os.path.join(self.output_dir, "input_image.png")
        cv2.imwrite(object_path, input_image)
        print(f"所有对象图像和 JSON 文件已保存到 {self.output_dir}")

        
    def data_process(self,data,image):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        mask_image_path = os.path.join(self.output_dir, "mask_image.png")
        image = np.array(image)
        self.Save_mask_bbox_image(det_res=data, image = image, save_path=mask_image_path)
        # 假设输入图像路径
        self.Split_instance_object_images(det_res=data, image=image)

if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    input = Submodule()
    # pkl_file = "/home/yan/BestMan_Chemistry_Test_SONG/Perception/Object_detection/Lang_SAM/data.pkl"
    # input.deserialize(pkl_file)
    # image = input.get("input_img", Image.Image)


    image = Image.open("/home/liusong/图片/new_sample8.png")

    det_res_alive = Det_res_alive()
    data = det_res_alive.Det_inference_res(image)
    det_res_alive.data_process(data,image)
   