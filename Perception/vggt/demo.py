import torch
import open3d as o3d
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
# model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model = VGGT().to(device)
state_dict = torch.load('/home/yan/hny/model.pt')
model.load_state_dict(state_dict)

# Load and preprocess example images (replace with your own image paths)
image_names = []
#for i in range(13,14):

#name = "/home/yan/hny/pic/" + str(i) + ".png"
name = "/home/yan/hny/vggt/captured_images/color_20250508_144308.png"
image_names.append(name)
    # image_names.append = ["/home/yan/hny/pic/3.jpg"]  
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)

rgb_pcl = predictions["world_points"]
points = rgb_pcl.squeeze(0).permute(1, 2, 3, 0).contiguous().cpu().numpy().reshape(-1, 3)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])
import cv2
cv2.imshow("image",images[0].permute(1,2,0).cpu().numpy())
cv2.waitKey(0)



depth = predictions['depth'].squeeze(0).squeeze(0).squeeze(2).cpu().numpy()



print('1')