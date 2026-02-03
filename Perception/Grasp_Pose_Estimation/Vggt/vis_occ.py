import numpy as np
import pyvista as pv
from tqdm import tqdm
# Function to create voxel mesh using individual cubes
def create_voxel_mesh_stand1(occupied_voxels, voxel_size, voxel_colors):
    """Create a voxel mesh more efficiently."""
    cubes = []  # To accumulate cubes
    colors_list = []  # To accumulate colors
    
    # Loop to generate each voxel (as a cube)
    for voxel, color in tqdm(zip(occupied_voxels, voxel_colors), total=len(occupied_voxels)):
        x, y, z = voxel
        cube = pv.Cube(center=(x, y, z), x_length=voxel_size[0], y_length=voxel_size[1], z_length=voxel_size[2])
        cubes.append(cube)
        colors_list.append(np.tile(color, (6, 1)))  # Store color information

    # Merge all cubes at once
    combined_mesh = pv.PolyData().merge(cubes)
    
    # Apply colors (you may need to ensure colors are applied correctly based on your needs)
    combined_mesh.cell_data['colors'] = np.vstack(colors_list)
    
    return combined_mesh

def create_voxel_mesh_stand2(occupied_voxels, voxel_size, voxel_colors):
    """Efficiently create a voxel mesh using individual cubes to represent each voxel."""
    cubes = []  # List to store cubes
    color_list = []  # List to store color data for all cubes

    # Loop to generate each voxel (as a cube)
    for voxel, color in tqdm(zip(occupied_voxels, voxel_colors), total=len(occupied_voxels)):
        x, y, z = voxel
        # Create a cube centered at the voxel position with specified size
        cube = pv.Cube(center=(x, y, z), x_length=voxel_size[0], y_length=voxel_size[1], z_length=voxel_size[2])
        cubes.append(cube)  # Store the cube
        
        # Store the color for all 6 faces of the cube
        color_list.append(np.tile(color, (6, 1)))  
    
    # Merge all cubes into a single mesh
    combined_mesh = pv.PolyData().merge(cubes)
    
    # Combine all color arrays into a single array
    all_colors = np.vstack(color_list)
    
    # Apply colors to the combined mesh
    combined_mesh.cell_data['colors'] = all_colors
    
    return combined_mesh
"""
# Load the occupancy grid from the .npz file
data = np.load("scene_reconstructed_scene0000_00.npz")
print(data["occupancy"].shape)
occ_grid = data["occupancy"]  # Shape: (120, 120, 32)
occ_grid = np.squeeze(occ_grid, axis=0)  # 现在 occ_grid 的形状为 (120, 120, 32)
print(np.max(occ_grid), np.min(occ_grid))
# 创建一个掩码，满足条件的元素会被设置为 0
mask = occ_grid < 0.51

# 使用掩码将小于 0.7 的值设置为 0
occ_grid[mask] = 0
non_zero_count = np.count_nonzero(occ_grid)
print(non_zero_count)
occ_grid = np.expand_dims(occ_grid, axis=-1)
# Remove the last singleton dimension to work with the occupancy values (120, 120, 32)
#occ_grid = np.unsqueeze(occ_grid, axis=-1)  # Shape: (120, 120, 32)
occupied_grids = np.argwhere(np.any(occ_grid > 0, axis=-1)) 
# Get the occupied voxels (voxels with occupancy value > 0)
#occupied_grids = np.argwhere(occ_grid > 0)  # Identify all occupied voxels
voxel_size = [0.08333333, 0.08333333, 0.09375]  # Set voxel size
print(occupied_grids.shape)
# Create the voxel mesh

# Calculate voxel coordinates in actual 3D space by scaling
occupied_voxels = occupied_grids * voxel_size  # Calculate actual coordinates of voxels

# Assign a default color (e.g., light gray) for each occupied voxel
voxel_colors = np.tile([200 / 255.0, 200 / 255.0, 200 / 255.0], (len(occupied_voxels), 1))  # Light gray color

# Create the voxel mesh using the function
voxel_mesh1 = create_voxel_mesh_stand1(occupied_voxels, voxel_size, voxel_colors)



data = np.load("scene_gt_epoch_26.npz")
print(data["occupancy"].shape)
occ_grid = data["occupancy"]  # Shape: (120, 120, 32)
occ_grid = np.squeeze(occ_grid, axis=0)  # 现在 occ_grid 的形状为 (120, 120, 32)
print(np.max(occ_grid), np.min(occ_grid))
# 创建一个掩码，满足条件的元素会被设置为 0
mask = occ_grid < 0.5

# 使用掩码将小于 0.7 的值设置为 0
occ_grid[mask] = 0
non_zero_count = np.count_nonzero(occ_grid)
print(non_zero_count)
occ_grid = np.expand_dims(occ_grid, axis=-1)
# Remove the last singleton dimension to work with the occupancy values (120, 120, 32)
#occ_grid = np.unsqueeze(occ_grid, axis=-1)  # Shape: (120, 120, 32)
occupied_grids = np.argwhere(np.any(occ_grid > 0, axis=-1)) 
# Get the occupied voxels (voxels with occupancy value > 0)
#occupied_grids = np.argwhere(occ_grid > 0)  # Identify all occupied voxels
voxel_size = [0.08333333, 0.08333333, 0.09375]  # Set voxel size
print(occupied_grids.shape)
# Create the voxel mesh

# Calculate voxel coordinates in actual 3D space by scaling
occupied_voxels = occupied_grids * voxel_size  # Calculate actual coordinates of voxels

# Assign a default color (e.g., light gray) for each occupied voxel
voxel_colors = np.tile([200 / 255.0, 200 / 255.0, 200 / 255.0], (len(occupied_voxels), 1))  # Light gray color

# Create the voxel mesh using the function
voxel_mesh2 = create_voxel_mesh_stand1(occupied_voxels, voxel_size, voxel_colors)
"""

# # Load the RGB voxel space
# data = np.load("/home/yixuan/Trans2OccGrasp/data/150_transparent_3_7scenes/occ_data_new/scene_20250423_195422_438/scene_20250423_195422_438_random_view_0002.npz")
# rgb_voxel = data["color_grid"]  # Shape: (120, 120, 32, 3)
# #print(rgb_voxel)
# # Get the occupied voxels and colors
# occupied_grids = np.argwhere(np.any(rgb_voxel >0, axis=-1))  # Use all voxels as occupied
# print(occupied_grids.shape)

# #voxel_colors = rgb_voxel.reshape((-1, 3)) # Normalize to [0, 1] for visualization
# x_range, y_range, z_range = 1.0, 1.0, 0.5  # 假设你知道这个范围
# voxel_size = [
#     x_range / rgb_voxel.shape[0],
#     y_range / rgb_voxel.shape[1],
#     z_range / rgb_voxel.shape[2],
# ]  # Set voxel size
# print(occupied_grids.shape)
# # Create the voxel mesh

# occupied_voxels = occupied_grids * voxel_size  # 计算体粒的实际坐标
# voxel_colors = rgb_voxel[occupied_grids[:, 0], occupied_grids[:, 1], occupied_grids[:, 2]] / 255.0  # Normalize to [0, 1] # 归一化颜色到 [0, 1]
# print(voxel_colors.shape)
# voxel_mesh2 = create_voxel_mesh_stand2(occupied_voxels, voxel_size, voxel_colors)

data2 = np.load("/home/yan/Downloads/150_transparent_3_7scenes/occ_data_new/scene_20250423_195425_305/scene_20250423_195425_305_random_view_0002.npz")
rgb_voxel2 = data2["color_grid"]  # Shape: (120, 120, 32, 3)
#print(rgb_voxel)
# Get the occupied voxels and colors
occupied_grids2 = np.argwhere(np.any(rgb_voxel2 >0, axis=-1))  # Use all voxels as occupied
print(occupied_grids2.shape)

#voxel_colors = rgb_voxel.reshape((-1, 3)) # Normalize to [0, 1] for visualization
x_range, y_range, z_range = 1.0, 1.0, 0.5  # 假设你知道这个范围
voxel_size2 = [
    x_range / rgb_voxel2.shape[0],
    y_range / rgb_voxel2.shape[1],
    z_range / rgb_voxel2.shape[2],
]  # Set voxel size
print(occupied_grids2.shape)
# Create the voxel mesh

occupied_voxels2 = occupied_grids2 * voxel_size2  # 计算体粒的实际坐标
voxel_colors2 = rgb_voxel2[occupied_grids2[:, 0], occupied_grids2[:, 1], occupied_grids2[:, 2]] / 255.0  # Normalize to [0, 1] # 归一化颜色到 [0, 1]
print(voxel_colors2.shape)
voxel_mesh = create_voxel_mesh_stand2(occupied_voxels2, voxel_size2, voxel_colors2)

#voxel_mesh1 = create_voxel_mesh_stand2(occupied_voxels, voxel_size, voxel_colors)
# Plot the voxel mesh
plotter = pv.Plotter()
plotter.add_mesh(voxel_mesh, scalars="colors", rgb=True, opacity=1)#
#plotter.add_mesh(voxel_mesh2, scalars="colors", rgb=True, opacity=1)
plotter.show()