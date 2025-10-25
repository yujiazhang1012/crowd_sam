import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import sys

# ====== 修改这里：设置你的项目路径 ======
sys.path.append("/home/ccnu-train/zyj/crowd_sam/crowd_sam")  # 你的 CrowdSAM 项目路径

from crowdsam.model import CrowdSAM
import crowdsam.utils as utils


def visualize_results(image, mask_data, action_classes):
    """
    在图像上绘制检测框和动作标签
    """
    # 定义颜色（每个动作一个颜色）
    colors = [
        (0, 255, 0),    # write - 绿色
        (255, 0, 0),    # read - 蓝色
        (0, 0, 255),    # lookup - 红色
        (255, 255, 0),  # turn_head - 青色
        (255, 0, 255),  # raise_hand - 品红
        (0, 255, 255),  # stand - 黄色
        (128, 128, 128) # discuss - 灰色
    ]
    
    # 转换为 BGR 格式（OpenCV 使用 BGR）
    if image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image.copy()
    
    # 绘制每个检测结果
    if 'actions' in mask_data and len(mask_data['actions']) > 0:
        boxes = mask_data['boxes'].numpy() if hasattr(mask_data['boxes'], 'numpy') else mask_data['boxes']
        actions = mask_data['actions']
        
        for i, (box, action) in enumerate(zip(boxes, actions)):
            x1, y1, x2, y2 = map(int, box)
            action_id = action['action_id']
            action_name = action['action_name']
            confidence = action['confidence']
            
            # 获取颜色
            color = colors[action_id % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{action_name}: {confidence:.2f}"
            cv2.putText(image_bgr, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image_bgr


def main():
    # ====== 修改这里：设置你的文件路径 ======
    CONFIG_PATH = "configs/crowdhuman.yaml"        # 配置文件路径
    IMAGE_PATH = "dataset/crowdhuman/Images/05.jpg"       # 输入图像路径
    OUTPUT_PATH = "demo_output.jpg"                # 输出图像路径
    DEVICE = "cuda"                                # 设备 (cuda 或 cpu)
    ACTION_HEAD_PATH = "weights/action_head.pth"   # 动作识别头权重路径

    # 加载配置
    config = utils.load_config(CONFIG_PATH)
    config['environ']['device'] = DEVICE

    # 初始化模型
    print("Loading CrowdSAM model...")
    model = CrowdSAM(config, logger=None)
    model.eval()
    
    # 加载动作识别头权重（如果存在）
    if os.path.exists(ACTION_HEAD_PATH):
        model.action_head.load_state_dict(
            torch.load(ACTION_HEAD_PATH, map_location=model.device)
        )
        print(f"Loaded action head weights from {ACTION_HEAD_PATH}")
    else:
        print(f"Warning: Action head weights not found at {ACTION_HEAD_PATH}!")

    # 加载图像
    print(f"Loading image from {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise ValueError(f"Could not load image from {IMAGE_PATH}")

    print(f"Image shape: {image.shape}")  # 添加调试
    print(f"Image dtype: {image.dtype}")  # 添加调试
    
    
    # 转换为 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"RGB image shape: {image_rgb.shape}")  # 添加调试

    # 推理
    print("Running inference...")
    with torch.no_grad():
        mask_data = model.generate(image_rgb)
    # 调试：检查 mask_data 内容
    print(f"mask_data keys: {list(mask_data._stats.keys())}")
    if 'masks' in mask_data._stats:
        print(f"Number of masks: {len(mask_data['masks'])}")
    else:
        print("No masks generated! Check your input image and model weights.")

    # 可视化结果
    print("Visualizing results...")
    result_image = visualize_results(image_rgb, mask_data, model.action_classes)

    # 保存结果
    cv2.imwrite(OUTPUT_PATH, result_image)
    print(f"Results saved to {OUTPUT_PATH}")

    # 打印统计信息
    if 'actions' in mask_data and len(mask_data['actions']) > 0:
        num_detections = len(mask_data['actions'])
        print(f"Detected {num_detections} persons")
        for i, action in enumerate(mask_data['actions']):
            print(f"  Person {i+1}: {action['action_name']} (conf: {action['confidence']:.2f})")
    else:
        print("No persons detected!")


if __name__ == '__main__':
    main()