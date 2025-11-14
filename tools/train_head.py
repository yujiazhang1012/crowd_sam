import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import random
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 防止在无GUI环境报错
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
import sys
sys.path.append("/home/ccnu-train/zyj/crowd_sam/crowd_sam")

from crowdsam.model import CrowdSAM

from crowdsam.data import CrowdHuman, collate_fn_crowdhuman
import crowdsam.utils as utils
def extract_geometric_features_from_bbox_mask(bbox, mask_np, image_shape):
    """
    从 bbox 和 mask_np 提取几何特征
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = image_shape[:2]
    h_mask, w_mask = mask_np.shape[:2]

    height_norm = (y2 - y1) / h_img

    # 手高于头部 (改进版)
    mask_bool = mask_np.astype(bool)
    ys, xs = np.where(mask_bool)
    if len(ys) > 0:
        upper_body_height = 0.4 * (y2 - y1)
        upper_body_y_min = y1
        upper_body_y_max = y1 + upper_body_height
        hand_in_upper = (ys >= upper_body_y_min) & (ys <= upper_body_y_max)
        hand_above_head = float(hand_in_upper.any())
    else:
        hand_above_head = 0.0

    aspect_ratio = (x2 - x1) / (y2 - y1 + 1e-6)

    mask_float = mask_np.astype(float)
    weighted_y = (np.arange(mask_np.shape[0])[:, None] * mask_float).sum() / (mask_float.sum() + 1e-6)
    vertical_center_norm = weighted_y / h_img

    is_tall = float(height_norm > 0.3)

    # 头部存在性 (改进版)
    head_region_height = 0.2 * (y2 - y1)
    head_y_min = y1
    head_y_max = y1 + head_region_height
    head_pixels = mask_np[int(head_y_min):int(head_y_max), :].sum()
    head_area_ratio = head_pixels / (mask_float.sum() + 1e-6)
    has_head = head_area_ratio > 0.05

    if not has_head:
        return None

    # 手部相对位置 (简化，基于上半身判断)
    if len(ys) > 0:
        body_center_y = (y1 + y2) / 2
        if hand_above_head:
            # 如果手在上半身，计算其重心
            hand_ys_in_upper = ys[hand_in_upper]
            if len(hand_ys_in_upper) > 0:
                hand_center_y = hand_ys_in_upper.min()
                hand_relative_y = (hand_center_y - body_center_y) / (y2 - y1 + 1e-6)
            else:
                hand_relative_y = 0.0
        else:
            hand_relative_y = 0.0
    else:
        hand_relative_y = 0.0

    # 头部朝向 (简化)
    head_mask = mask_np[int(head_y_min):int(head_y_max), :]
    if head_mask.sum() > 0:
        head_ys, head_xs = np.where(head_mask)
        if len(head_ys) > 0:
            head_center_x = np.mean(head_xs)
            head_width = head_xs.max() - head_xs.min()
            head_aspect = head_width / (head_ys.max() - head_ys.min() + 1e-6)
            head_orientation = float(head_center_x > w_mask / 2)
        else:
            head_orientation = 0.0
    else:
        head_orientation = 0.0

    # 身体倾斜度 (简化，使用重心)
    if len(xs) > 0 and len(ys) > 0:
        cx = (np.arange(mask_np.shape[1]) * mask_float.sum(axis=0)).sum() / (mask_float.sum() + 1e-6)
        cy = (np.arange(mask_np.shape[0]) * mask_float.sum(axis=1)).sum() / (mask_float.sum() + 1e-6)
        body_center_y = (y1 + y2) / 2
        body_tilt = (cy - body_center_y) / (y2 - y1 + 1e-6)
    else:
        body_tilt = 0.0

    geo_feat = np.array([
        height_norm,
        hand_above_head,
        aspect_ratio,
        vertical_center_norm,
        is_tall,
        hand_relative_y,
        head_orientation,
        body_tilt
    ], dtype=np.float32)

    return geo_feat

def get_train_transform():
    return T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
class AugmentedCrowdHuman(CrowdHuman):
    def __init__(self, dataset_root, annot_path, use_sam_masks=True, class_counts=None, is_train=True):
        super().__init__(dataset_root, annot_path, transform=None, use_sam_masks=use_sam_masks)
        self.class_counts = class_counts or [520, 920, 4045, 915, 569, 58, 6]
        self.rare_classes = {i for i, count in enumerate(self.class_counts) if count < 200}
        self.train_transform = get_train_transform()
        self.val_transform = get_val_transform()
        self.is_train = is_train  # 控制是否使用增强

    def __getitem__(self, idx):
        image_path, bbox, label = self.samples[idx]
        original_image_pil = Image.open(image_path).convert('RGB')
        original_image_np = np.array(original_image_pil)
        image_shape = original_image_np.shape
        x, y, w, h = bbox
        h_img, w_img = image_shape[:2]
        mask_np = np.zeros((h_img, w_img), dtype=np.uint8)
        y1, y2 = max(0, int(y)), min(h_img, int(y + h))
        x1, x2 = max(0, int(x)), min(w_img, int(x + w))
        mask_np[y1:y2, x1:x2] = 1
        bbox_xyxy = [int(x), int(y), int(x + w), int(y + h)]
        geo_features = extract_geometric_features_from_bbox_mask(bbox_xyxy, mask_np, image_shape)
        if geo_features is None:
            # 如果几何特征计算失败，使用默认值
            geo_features = np.zeros(8, dtype=np.float32)
            # print(f"Warning: Could not compute geo features for sample {idx}, using zeros.")
        roi_pil = original_image_pil.crop((x, y, x + w, y + h))
        roi_w, roi_h = roi_pil.size
        aspect_ratio = roi_w / roi_h
        if aspect_ratio < 0.3 or aspect_ratio > 3.0 or roi_h < 64 or roi_w < 64:
            # 如果 ROI 不符合要求，用一个默认图像填充
            roi_pil = Image.new('RGB', (224, 224), color='black')
            geo_features = np.zeros(8, dtype=np.float32) # 也重置 geo_features
            label = 0

        if not self.is_train:
            roi_tensor = self.val_transform(roi_pil)
        else:
            if label in self.rare_classes or label in [0, 1, 3]:
                roi_tensor = self._strong_augment(roi_pil)
            else:
                if random.random() < 0.3:
                    roi_tensor = self._weak_augment(roi_pil)
                else:
                    roi_tensor = self.val_transform(roi_pil)
        geo_features_tensor = torch.from_numpy(geo_features).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        # 返回包含 geo_features 的元组
        return roi_tensor, geo_features_tensor, label_tensor
    def _strong_augment(self, img):

        # 对小样本类别使用更强的增强
        img = T.Resize((256, 256))(img)
        img = T.RandomCrop(224)(img)
        if random.random() < 0.5:
            img = T.RandomHorizontalFlip(p=0.5)(img)
        img = T.RandomRotation(degrees=20)(img)
        img = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15)(img)
    
        img = T.ToTensor()(img)
        img = T.RandomErasing(p=0.4, scale=(0.02, 0.15))(img)  
        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img
    def _weak_augment(self, img):
        img = T.Resize((256, 256))(img)
        img = T.RandomCrop(224)(img)
        if random.random() < 0.5:
            img = T.RandomHorizontalFlip(p=1.0)(img)
        if random.random() < 0.3:
            img = T.ColorJitter(brightness=0.1, contrast=0.1)(img)
        img = T.ToTensor()(img)
        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img
def collate_fn_for_training(batch):
    rois, geo_features, labels = zip(*batch)
    rois = torch.stack(rois)
    geo_features = torch.stack(geo_features)
    labels = torch.stack(labels)
    return rois, geo_features, labels
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.weight = weight

    def forward(self, x, target):
        log_probs = F.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()

def validate_model(model, val_dataloader, device, class_names, result_dir, epoch):
    model.eval()
    all_preds = []
    all_labels = []

    

    with torch.no_grad():
        for rois, geo_features, labels in tqdm(val_dataloader, desc="Validating"):
            rois = rois.to(device)
            geo_features = geo_features.to(device)
            labels = labels.to(device)
            cnn_features = model.action_backbone(rois)
            cnn_features_flat = F.adaptive_avg_pool2d(cnn_features, 1).flatten(1)
            combined_features = torch.cat([cnn_features_flat, geo_features], dim=1)
            logits = model.action_head(combined_features)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\n Validation Accuracy: {acc:.4f}")
    print("\n Classification Report:")
    labels = list(range(len(class_names)))  # [0,1,2,3,4,5,6]
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        labels=labels,      # 👈 关键修复
        digits=4, 
        zero_division=0
    ))

    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    cm_path = os.path.join(result_dir, f'confusion_matrix_epoch{epoch+1:02d}.png')    
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to:{cm_path}")
    plt.close()
    model.train()
    return acc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', default='configs/crowdhuman.yaml')
    args = parser.parse_args()
    config = utils.load_config(args.config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 混淆矩阵保存路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_parent_dir = "results"               
    result_dir = os.path.join(results_parent_dir, f"results_{timestamp}")  
    os.makedirs(result_dir, exist_ok=True)
    print(f" Training logs and confusion matrices will be saved to: {result_dir}")

    # 初始化模型
    model = CrowdSAM(config, logger=None)
    model.to(model.device)

    for param in model.parameters():
        param.requires_grad = False
    # 解冻 EfficientNet 最后两个 block（features.7 和 features.8）
    for name, param in model.action_backbone.named_parameters():
        if "features.7" in name or "features.8" in name:
            param.requires_grad = True
            print(f"Unfrozen: {name}")
    for param in model.action_head.parameters():
        param.requires_grad = True
    # 为不同的类别分配不同的权重
    class_counts = torch.tensor([520, 920, 4045, 915, 569, 58, 6], dtype=torch.float).to(device)
    beta=0.9999
    effective_num=1.0 - torch.pow(beta, class_counts)
    class_weights = (1.0 - beta)/ effective_num
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # 归一化
    print(f"[INFO] Class Weights: {class_weights.tolist()}")
    # 优化器分层设置学习率
    optimizer = optim.AdamW([
        {'params': model.action_head.parameters(), 'lr': 1e-5},  # 分类头学习率高
        {'params': [p for n, p in model.action_backbone.named_parameters() if "7" in n or "8" in n], 'lr': 1e-5}  # backbone学习率低
    ], weight_decay=1e-5)

    class_counts_list = [520, 920, 4045, 915, 569, 58, 6]
    raw_train_dataset = CrowdHuman(
        dataset_root=config['data']['dataset_root'],
        annot_path=config['data']['train_file'],
        transform=None, # 在 AugmentedCrowdHuman 中处理
        use_sam_masks=True
    )
    raw_val_dataset = CrowdHuman(
        dataset_root=config['data']['dataset_root'],
        annot_path=config['data']['json_file'], # 修复路径
        transform=None, # 在 AugmentedCrowdHuman 中处理
        use_sam_masks=True
    )

    train_dataset = AugmentedCrowdHuman(
        dataset_root=config['data']['dataset_root'],
        annot_path=config['data']['train_file'],
        use_sam_masks=True,
        class_counts=class_counts_list,
        is_train=True
    )
    val_dataset = AugmentedCrowdHuman(
        dataset_root=config['data']['dataset_root'],
        annot_path=config['data']['json_file'], # 修复路径
        use_sam_masks=True,
        class_counts=class_counts_list,
        is_train=False  
    )

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn_for_training)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn_for_training)

    criterion = LabelSmoothingCrossEntropy(weight = class_weights, smoothing = 0.1).to(device)

    best_acc = 0.0
    for epoch in range(50):
        model.train()
        total_loss = 0
        for rois, geo_features, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            rois = rois.to(model.device)
            geo_features = geo_features.to(device)
            labels = labels.to(model.device)
            optimizer.zero_grad()
            cnn_features = model.action_backbone(rois)
            cnn_features_flat = F.adaptive_avg_pool2d(cnn_features, 1).flatten(1)
            
            
            combined_features = torch.cat([cnn_features_flat, geo_features], dim=1)
           
            logits = model.action_head(combined_features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == 29:
            val_acc = validate_model(
                model, 
                val_dataloader, 
                model.device, 
                class_names=["write", "read", "lookup", "turn_head", "raise_hand", "stand", "discuss"],
                result_dir = result_dir,
                epoch = epoch
            )
            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs("weights", exist_ok=True)
                torch.save(model.action_head.state_dict(), "weights/action_head_best.pth")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.action_head.state_dict(), "weights/action_head_final.pth")
    print("Training completed!")
