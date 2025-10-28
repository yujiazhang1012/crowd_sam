import torch
from crowdsam.model import CrowdSAM
import crowdsam.utils as utils

config = utils.load_config('configs/crowdhuman.yaml')
model = CrowdSAM(config, logger=None)
model.eval()

# 检查模型结构
print("=== Model Structure ===")
print("SAM predictor:", model.predictor is not None)
print("Action backbone:", model.action_backbone is not None)
print("Action head:", model.action_head is not None)
print("crop_n_layers:", model.crop_n_layers)

# 检查权重加载
action_head_path = "weights/action_head.pth"
if torch.cuda.is_available():
    weights = torch.load(action_head_path)
else:
    weights = torch.load(action_head_path, map_location='cpu')
print("Action head weights keys:", list(weights.keys()))
print("Action head output dim:", weights['5.weight'].shape[0])  # 最后一层