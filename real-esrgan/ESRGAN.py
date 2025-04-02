import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet  # 正确的模型架构
from realesrgan import RealESRGANer  # Real-ESRGAN 推理核心

# 确保 Python 能正确访问 Real-ESRGAN
sys.path.insert(0, "/root/autodl-tmp/real-ESRGAN/Real-ESRGAN-master")  

# **设定输入 & 输出目录**
input_root = "/root/autodl-tmp/dataset_ori/data0resume_en/1_resumedata/en_resume_1/Resumes_Datasets/"
output_root = "/root/autodl-tmp/dataset_train/data0resume_en/1_resumedata/resume_en/"

# **需要处理的三个大文件夹**
folders = ["Scrapped_Resumes", "resume_database", "Bing_images"]

# **选择 GPU 进行加速**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **加载 Real-ESRGAN 超分辨率模型**
model_path = "/root/autodl-tmp/real-ESRGAN/Real-ESRGAN-master/weights/RealESRGAN_x2plus.pth"  # 确保路径正确
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ 模型文件 {model_path} 不存在，请检查路径是否正确！")

# **使用 x2 超分辨率的 RRDBNet**
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

# **加载模型权重**
checkpoint = torch.load(model_path, map_location=device)

# **检查是否有 'generator' 键**
if "generator" in checkpoint:
    print("🔧 检测到 'generator'，正在提取...")
    checkpoint = checkpoint["generator"]  # 只加载 generator 作为模型权重

model.load_state_dict(checkpoint, strict=False)  # `strict=False` 忽略不匹配的键
model.to(device).eval()

# **创建超分辨率推理对象**
upsampler = RealESRGANer(scale=2, model_path=model_path, model=model, device=device, tile=0, tile_pad=10, pre_pad=0, half=True)

# **定义超分辨率处理函数**
def enhance_image(input_path, output_path):
    try:
        # **读取图片**
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"❌ 无法读取图片: {input_path}")
            return
        
        # **执行超分辨率**
        with torch.no_grad():
            output, _ = upsampler.enhance(image, outscale=2)  # x2 放大

        # **保存增强后的图片**
        cv2.imwrite(output_path, output)
        print(f"✅ 处理完成: {output_path}")

    except Exception as e:
        print(f"❌ 处理失败 {input_path}: {e}")

# **遍历输入目录并处理图像**
def process_images(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片文件
                input_image_path = os.path.join(root, file)

                # **计算输出路径（保持原始目录结构）**
                relative_path = os.path.relpath(root, input_dir)
                output_folder_path = os.path.join(output_dir, relative_path)
                os.makedirs(output_folder_path, exist_ok=True)

                # **添加 `_train` 后缀**
                file_name, file_ext = os.path.splitext(file)
                output_image_path = os.path.join(output_folder_path, f"{file_name}_train{file_ext}")

                # **如果增强版图片已存在，则跳过**
                if os.path.exists(output_image_path):
                    print(f"⏩ 跳过 {output_image_path}，文件已存在")
                    continue

                # **处理图片**
                enhance_image(input_image_path, output_image_path)

# **处理所有文件夹**
for folder in folders:
    input_folder = os.path.join(input_root, folder)
    output_folder = os.path.join(output_root, folder)
    process_images(input_folder, output_folder)

print("🚀 所有图像处理完成！")
