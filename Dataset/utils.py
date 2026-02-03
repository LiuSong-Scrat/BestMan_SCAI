import os

def get_project_root():
    """
    获取项目的根目录路径。
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_database_path():
    """
    获取Database文件夹的绝对路径。
    """
    return os.path.join(get_project_root(), "Dataset")

def get_image_path(image_name):
    """
    获取指定图像文件的绝对路径。

    Args:
        image_name (str): 图像文件名（例如 "cup.png"）。

    Returns:
        str: 图像文件的绝对路径。

    Raises:
        FileNotFoundError: 如果文件不存在。
    """
    image_path = os.path.join(get_database_path(), "Images", image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    return image_path

def verify_file_exists(file_path):
    """
    检查文件是否存在。

    Args:
        file_path (str): 文件路径。

    Raises:
        FileNotFoundError: 如果文件不存在。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
