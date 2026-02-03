# AnyGrasp Installation Guide

This guide provides step-by-step instructions to install AnyGrasp, including required dependencies and modules.

## 1. Install MinkowskiEngine

> **Note**: If you are running this on a laptop, you may encounter issues due to limited resources. To address this, add `export MAX_JOBS=2;` before `pip install`. Refer to the [NVIDIA MinkowskiEngine Issue #228](https://github.com/NVIDIA/MinkowskiEngine/issues/228) for more details.

Follow the steps below to set up your environment and install MinkowskiEngine:

```bash
# Step 1: Create and activate a new conda environment
conda create -n targetanygrasp python=3.9.20
conda activate targetanygrasp

# Step 2: Install PyTorch with CUDA support
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Step 3: Install OpenBLAS
conda install openblas-devel -c anaconda

# Step 4: Set CUDA path
export CUDA_HOME=/usr/local/cuda-11.8

# Step 5: Clone and install MinkowskiEngine
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

## 2. Install Additional Dependencies

Install other required Python packages using ```pip```:
```bash
pip install graspnetAPI
pip install open3d
pip install numpy==1.21.6
pip install yacs
```

## 3. Install the ```pointnet2``` Module

Follow the steps below to install the ```pointnet2``` module:
```bash
cd pointnet2
python setup.py install
```

## 4. License Registration

To use the AnyGrasp SDK, you need to register and set up a valid license. Follow the steps below to complete the setup.

---

### 4.1 Obtain and Setup the License

1. **Apply for the SDK License**  
   Apply for the license from [AnyGrasp SDK](https://github.com/graspnet/anygrasp_sdk).  
   
2. **Place the License**  
   Name this license folder as `license` and place this inside the `AnyGrasp` directory.

> **Reference**: If you haven't applied for a license yet, refer to the [license registration guide](../license_registration/README.md).

---

### 4.2 Place Model Weights

Along with the license you will receive the `checkpoint.tar`. Place this in `checkpoints/`
