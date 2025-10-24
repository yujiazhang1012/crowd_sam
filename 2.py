import os
import json
from datetime import datetime

import PIL.Image as Image
import matplotlib.pyplot as plt

import torch
from crowdsam.model import CrowdSAM
from crowdsam.utils import (
    load_config,
    setup_logger,
    visualize_result,
)
from crowdsam.data import data_meta


def load_model(config_file, output_dir):
    """
    初始化配置、logger、模型，返回 (config, logger, model, class_names)
    """
    # 1. 读取config
    config = load_config(config_file)

    # 2. 创建输出文件夹 & 日志
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(output_dir, 'log'))
    logger.info(f"[Init] Using config: {config_file}")
    logger.info(f"[Init] Output dir: {output_dir}")

    # 3. 取类别信息
    n_class, class_names = data_meta[config['data']['dataset']][1:]
    logger.info(f"[Init] Classes ({n_class}): {class_names}")

    # 4. 构建模型
    model = CrowdSAM(config, logger)
    logger.info("[Init] CrowdSAM model loaded.")

    return config, logger, model, class_names


def run_inference_on_image(image_path, config, logger, model, class_names,
                           output_dir, mode='seg'):
    """
    对单张图片做推理、保存可视化结果并显示
    """
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    logger.info(f"[Infer] Processing image: {image_path}")

    # 1. 打开图片
    image = Image.open(image_path)

    # 2. 模型推理
    result = model.generate(image)
    # result 里通常包含:
    # 'boxes': Nx4 tensor (xyxy)
    # 'scores': Nx1 tensor
    # 'categories': Nx1 tensor (class id)
    # 'rles': list/seg info (如果是seg模式)

    # 3. 组织输出字典（方便后续调试/保存成JSON）
    instance_dict = {
        'image_file': image_path
    }
    for k, v in result.items():
        if k in ['boxes', 'scores', 'categories']:
            instance_dict[k] = v.tolist()
        elif k in ['rles']:
            instance_dict[k] = v

    # 4. 生成时间戳子目录，存可视化结果 & json
    now = datetime.now()
    time_str = now.strftime("%Y_%m-%d_%H-%M-%S")
    save_dir = os.path.join(output_dir, time_str)
    os.makedirs(save_dir, exist_ok=True)

    # 可视化输出文件名
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    vis_path = os.path.join(save_dir, f"{img_name}_vis.jpg")

    # 5. 画框/掩膜并保存可视化图
    visualize_result(
        image,
        result,
        class_names,
        vis_path,
        conf_thresh=config['vis']['vis_thresh'],
        vis_masks=(mode == 'seg')
    )
    logger.info(f"[Infer] Visualization saved to: {vis_path}")

    # 6. 保存检测结果到 json
    json_path = os.path.join(save_dir, f"{img_name}_result.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(instance_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"[Infer] JSON result saved to: {json_path}")

    # 7. 弹窗展示可视化后的结果
    vis_img = Image.open(vis_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(vis_img)
    plt.axis('off')
    plt.title("CrowdSAM result")
    plt.show()

    return {
        "vis_image_path": vis_path,
        "json_result_path": json_path,
        "raw_result": instance_dict,
    }


if __name__ == "__main__":
    # ======= 你只需要改这三个路径/变量 =======
    IMAGE_PATH   = "/home/ccnu-train/zyj/crowd_sam/crowd_sam/013.jpg"              # 要推理的那张图片
    CONFIG_FILE  = "./configs/crowdhuman.yaml"         # 训练/推理所用的yaml
    OUTPUT_DIR   = "/home/ccnu-train/zyj/crowd_sam/crowd_sam/014.jpg"            # 输出放在哪
    MODE         = "seg"                               # "seg" 画mask, "bbox" 只画框
    # ========================================

    # 可选：如果你强行想用CPU推理，确保模型内部支持 .to(device)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 取消注释可强制禁GPU

    # 1. 初始化模型
    config, logger, model, class_names = load_model(CONFIG_FILE, OUTPUT_DIR)

    # 2. 对单张图片做推理 + 可视化 + 展示
    result_pack = run_inference_on_image(
        image_path=IMAGE_PATH,
        config=config,
        logger=logger,
        model=model,
        class_names=class_names,
        output_dir=OUTPUT_DIR,
        mode=MODE
    )

    # 打印一下输出文件位置，方便你在命令行看
    print("[Done]")
    print("Visualized image:", result_pack["vis_image_path"])
    print("JSON result:", result_pack["json_result_path"])
