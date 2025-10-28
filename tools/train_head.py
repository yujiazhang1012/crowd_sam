import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
import sys
sys.path.append("/home/ccnu-train/zyj/crowd_sam/crowd_sam")

from crowdsam.model import CrowdSAM
from crowdsam.data import CrowdHuman, collate_fn_crowdhuman
import crowdsam.utils as utils


def get_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
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
    dataset = CrowdHuman(
        dataset_root=config['data']['dataset_root'],
        annot_path=config['data']['train_file'],
        transform=get_transform(),
        use_sam_masks=True  # 确保使用 SAM 生成的 mask
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_crowdhuman
    )
    # 类别权重（解决不平衡）
    class_counts = torch.tensor([520, 920, 4045, 915, 569, 58], dtype=torch.float)
    class_weights = (1.0 / class_counts) * len(class_counts) / (1.0 / class_counts).sum()
    class_weights = class_weights.to(device)
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
    torch.save(model.action_head.state_dict(), "weights/action_head_1.pth")
    print("Action head training completed!")