temp_img = camera_overhead.get_rgb_image()
temp_img =  cv2.resize(temp_img,(640,480),interpolation=cv2.INTER_LINEAR)
cv2.imwrite("/home/liusong/ProgramFiles/BestMan/Dataset/Images/cube_stak.png",cv2.cvtColor(temp_img,cv2.COLOR_BGR2RGB))
cv2.waitKey(0)




import numpy as np
import open3d as o3d

points_n6=cloud_rgb

# 分离 XYZ 和 RGB
xyz = points_n6[:, :3]               # shape (N, 3)
rgb = points_n6[:, 3:] / 255.0       # 转换到 [0, 1]

# 创建 PointCloud 对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)  # 注意：colors 也是 Vector3dVector

# 保存为 PLY 文件
o3d.io.write_point_cloud("/home/liusong/temp/output.ply", pcd)

print("PLY file saved as output.ply")



import torch
import numpy as np
import pickle

obs_dict = {
    'image': torch.randn(3, 64, 64),          # PyTorch 张量
    'agent_pos': np.array([0.1, 0.2, 0.3]),   # NumPy 数组
    'task': 'pick_red_block'                  # 字符串
}

# 保存
with open('obs_dict.pkl', 'wb') as f:
    pickle.dump(obs_dict, f)

# 读取
with open('obs_dict.pkl', 'rb') as f:
    data = pickle.load(f)

print(type(data['image']))      # <class 'torch.Tensor'>
print(type(data['agent_pos']))  # <class 'numpy.ndarray'>




bgr = camera_overhead.get_rgb_image()
rgb =cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
img_overhead_rgb_resized = cv2.resize(rgb,(640,480),interpolation=cv2.INTER_LINEAR)
cv2.imwrite("/home/liusong/ProgramFiles/BestMan/Dataset/Images/cube_stak.png",img_overhead_rgb_resized)


frame = cv2.imread("/home/liusong/ProgramFiles/BestMan/Dataset/Images/cube_stak.png")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
width, height = frame.shape[:2][::-1]
cv2.imshow("overhead_frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)