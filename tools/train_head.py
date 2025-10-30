import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import random
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
import sys
sys.path.append("/home/ccnu-train/zyj/crowd_sam/crowd_sam")

from crowdsam.model import CrowdSAM
from crowdsam.data import CrowdHuman, collate_fn_crowdhuman
import crowdsam.utils as utils


def get_train_transform():
    return T.Compose([
        T.Resize((256, 256)),  # 先放大，再随机裁剪
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
    def __init__(self, dataset_root, annot_path, use_sam_masks=True, class_counts=None):
        super().__init__(dataset_root, annot_path, transform=None, use_sam_masks=use_sam_masks)
        self.class_counts = class_counts or [520, 920, 4045, 915, 569, 58, 6]
        # 定义小样本类别（样本数 < 100）
        self.rare_classes = {i for i, count in enumerate(self.class_counts) if count < 100}
        self.train_transform = get_train_transform()
        self.val_transform = get_val_transform()

    def __getitem__(self, idx):
        # 调用父类获取原始 roi 和 label
        roi, label, img_info = super().__getitem__(idx)
        
        # 转为 PIL Image（假设 roi 是 numpy array）
        if isinstance(roi, torch.Tensor):
            roi = T.ToPILImage()(roi)
        elif isinstance(roi, np.ndarray):
            roi = Image.fromarray(roi.astype('uint8'), 'RGB')
        else:
            roi = Image.open(roi).convert('RGB')

        # 对小样本类别应用更强增强
        if label in self.rare_classes:
            roi = self._strong_augment(roi)
        else:
            # 普通增强
            if random.random() < 0.3:  # 30% 概率增强
                roi = self.train_transform.transforms[0](roi)  # Resize
                roi = self.train_transform.transforms[1](roi)  # RandomCrop
                if random.random() < 0.5:
                    roi = T.RandomHorizontalFlip(p=1.0)(roi)
                if random.random() < 0.3:
                    roi = T.ColorJitter(brightness=0.1, contrast=0.1)(roi)
                roi = T.ToTensor()(roi)
                roi = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(roi)
            else:
                roi = self.val_transform(roi)
        return roi, label, img_info

    def _strong_augment(self, img):
        """对小样本类别应用强增强"""
        # 1. 随机水平翻转
        if random.random() < 0.5:
            img = T.functional.hflip(img)
        # 2. 随机旋转 (-15° ~ 15°)
        angle = random.uniform(-15, 15)
        img = T.functional.rotate(img, angle, fill=0)
        # 3. 颜色抖动
        img = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)(img)
        # 4. 转 tensor + normalize
        img = T.ToTensor()(img)
        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img
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



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', default='configs/crowdhuman.yaml')
    args = parser.parse_args()
    config = utils.load_config(args.config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型（只训练 action_head）
    model = CrowdSAM(config, logger=None)
    model.to(model.device)
    model.train()

    # 冻结所有参数，只解冻 action_head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.action_head.parameters():
        param.requires_grad = True

    # 优化器
    optimizer = optim.AdamW(
        model.action_head.parameters(),
        lr=3e-4,
        weight_decay=1e-5
    )
    # criterion = nn.CrossEntropyLoss()

    # 数据集
    class_counts = [520, 920, 4045, 915, 569, 58, 6]
    dataset = AugmentedCrowdHuman(
        dataset_root=config['data']['dataset_root'],
        annot_path=config['data']['train_file'],
        use_sam_masks=True,
        class_counts=class_counts
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_crowdhuman
    )
    # 类别权重（解决不平衡）
    class_counts = torch.tensor([520, 920, 4045, 915, 569, 58, 6], dtype=torch.float)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # 归一化

    print(f"[INFO] Class Weights: {class_weights.tolist()}")

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights).to(device)

    # 训练循环
    for epoch in range(30):
        total_loss = 0
        for rois, labels, _ in tqdm(dataloader):
            rois = rois.to(model.device)
            labels = labels.to(model.device)

            optimizer.zero_grad()
            
            # 前向传播
            features = model.action_backbone(rois)
            logits = model.action_head(features)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    # 保存权重
    os.makedirs("weights", exist_ok=True)
    torch.save(model.action_head.state_dict(), "weights/action_head_2.pth")
    print("Action head training completed!")