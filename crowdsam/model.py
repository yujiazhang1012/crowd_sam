import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import os
import cv2
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch.nn as nn
from loguru import logger
import math
from torchvision.ops.boxes import batched_nms, box_area
import crowdsam.utils as utils
# from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_weights
import torchvision.models as models
from segment_anything_cs.utils.amg import (
    MaskData,
    batch_iterator,
    batched_mask_to_box,
    calculate_stability_score,
    generate_crop_boxes,
    remove_small_regions,
    mask_to_rle_pytorch,
    coco_encode_rle,
)

class EfficientNetWithCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        # weight = EfficientNet_B0_weights.IMAGENET1K_V1
        # backbone = efficientnet_b0(weights=weights)
        backbone = models.efficientnet_b0(pretrained=True)
        self.features = backbone.features
        # 最后的 stage 后加一个 CBAM
        self.cbam = CBAM(1280)

    def forward(self, x):
        # x = self.backbone(x)
        x= self.features(x)
        x = self.cbam(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes,ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding =kernel_size//2
        self.conv1 = nn.Conv2d(
            2,
            1, 
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            groups=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out,_ = torch.max(x,dim=1,keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out,max_out],dim=1)))
class CBAM(nn.Module):
    def __init__(self, in_planes,ratio=16,kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self,x):
        return x * self.sa(x) * self.ca(x)

class CrowdSAM(nn.Module):
    vis_img_id = 0
    def __init__(self,config,logger):
        super().__init__()
        self.device = torch.device(config['environ']['device'])
        
        #hard-coded setting        
        legacy_mode = False
        self.train_free=False
        #model 
        dino_model =  torch.hub.load(config['model']['dino_repo'],
                                    config['model']['dino_model'],
                                    source='local',pretrained=False).to(self.device)
        dino_model.load_state_dict(torch.load(config['model']['dino_checkpoint']))
        self.predictor =self.load_sam_model(config['model']['sam_model'], 
                                            config['model']['sam_arch'],
                                            config['model']['sam_checkpoint'], 
                                            config['model']['sam_adapter_checkpoint'],
                                            dino_model, 
                                            config['model']['n_class'],
                                            )

        # 增加
        self.action_classes = ["write", "read", "lookup", "turn_head", "raise_hand", "stand","discuss"]
        self.num_action_classes = len(self.action_classes)
        # 使用 EfficientNet-B0 作为 backbone
        # self.action_backbone = torchvision.models.efficientnet_b0(pretrained=True).features
        self.action_backbone = EfficientNetWithCBAM()
        self.action_backbone.eval()
        self.action_backbone.to(self.device)

        # 分类头
        self.action_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280 , 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_action_classes)
        ).to(self.device)
        # 加载训练好的权重（如果有）
        action_head_path = "weights/action_head_best.pth"
        if os.path.exists(action_head_path):
            self.action_head.load_state_dict(torch.load(action_head_path, map_location=self.device))

        
        self.mask_selection = config['test']['mask_selection']
        self.apply_box_offsets = config['test']['apply_box_offsets'] #apply_box_offsets
        #eps settings
        self.max_prompts =config['test']['max_prompts']
        self.filter_thresh = config['test']['filter_thresh']
        #other test settings     
        self.max_size =config['test']['max_size']#resize image to this
        self.grid_size =config['test']['grid_size']
        self.pred_iou_thresh =  config['test']['pred_iou_thresh'] #iou_score filter
        # self.score_thresh = kwargs.get('score_thresh')
        self.fuse_simmap = config['test']['fuse_simmap']
        self.stability_score_thresh = config['test']['stability_score_thresh']
        self.stability_score_offset = config['test']['stability_score_offset']
        self.box_nms_thresh = config['test']['box_nms_thresh']
        self.points_per_batch = config['test']['points_per_batch']
        self.crop_n_layers = config['test']['crop_n_layers']
        self.crop_nms_thresh = config['test']['crop_nms_thresh']
        self.crop_overlap_ratio = config['test']['crop_overlap_ratio']
        self.min_mask_region_area = config['test']['min_mask_region_area']
        self.pos_sim_thresh = config['test']['pos_sim_thresh']
        self.output_rles = config['test']['output_rles']



        if legacy_mode:
            self.patch_size = config['model']['patch_size'] # vit_l for dino
            self.feat_size = feat_size
            self.feat_dim = feat_dim # vit_l for dino
            self.transform = T.Compose([
                Resize(patch_size),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        if config['model']['trainfree']:
            self.train_free = True
            self.patch_size = 14
            self.image_encoder = dino_model
            self.ref_feature = torch.load(config['model']['ref_feature'])['f'].mean(dim=0).to(self.device)#.mean(dim=0)
            self.alpha = config['model']['score_fusion']
        #  if not config['model']['trainfree'] else True
            
        #original sam automask generater args
     
        #other parameters

    #load sam model according to specifiedd arguments
    def load_sam_model(self, sam_model, sam_arch, sam_checkpoint, sam_adapter_checkpoint, dino_model, n_class):
        if sam_arch =='crowdsam':
            from segment_anything_cs import sam_model_registry, SamPredictor
            # from per_segment_anything_person_specific import sam_model_registry,SamPredictor
            sam = sam_model_registry[sam_model](checkpoint=sam_checkpoint,n_class=n_class)
            sam.mask_decoder.load_state_dict(torch.load(sam_adapter_checkpoint),strict=False)
            predictor = SamPredictor(sam, dino_model)

        elif sam_arch =='sam_hq':
            from segment_anything_hq import sam_model_registry, SamPredictor
            # from per_segment_anything_person_specific import sam_model_registry,SamPredictor
            sam_model = sam_model[2:]
            sam = sam_model_registry[sam_model](checkpoint=sam_checkpoint,n_class=n_class)
            sam.mask_decoder.load_state_dict(torch.load(sam_adapter_checkpoint), strict=False)
            predictor = SamPredictor(sam, dino_model)
        elif sam_arch == 'mobile_sam':
            from mobile_sam import sam_model_registry, SamPredictor
            sam_model = sam_model[6:]
            sam = sam_model_registry[sam_model](checkpoint=sam_checkpoint)
            sam.mask_decoder.load_state_dict(torch.load(sam_adapter_checkpoint))
            predictor = SamPredictor(sam, dino_model)
        else:
            from segment_anything import sam_model_registry,SamPredictor
            sam = sam_model_registry[sam_model](checkpoint=sam_checkpoint)
            predictor = SamPredictor(sam)
        dino_model = dino_model.to(self.device)
        sam = sam.to(self.device)
        return predictor
    
    # 教室场景几何特征
    def extract_geometric_features(self, mask: np.ndarray, bbox: list, image_shape: tuple):
        """
        您设计的 8 维几何特征
        """
        x1, y1, x2, y2 = bbox
        h_img, w_img = image_shape[:2]
        h_mask, w_mask = mask.shape[:2]

        # 归一化高度
        height_norm = (y2 - y1) / h_img

        # 手是否高于头部
        ys, xs = np.where(mask)
        if len(ys) == 0:
            hand_above_head = 0.0
        else:
            # 在 mask 的尺度下判断
            upper_body_y = (y1 + y2) // 2
            hand_in_upper = ys < upper_body_y
            if hand_in_upper.any():
                hand_center_y = ys[hand_in_upper].min()
                head_center_y = upper_body_y
                hand_above_head = float(hand_center_y < head_center_y)
            else:
                hand_above_head = 0.0

        aspect_ratio = (x2 - x1) / (y2 - y1 + 1e-6)

        mask_float = mask.astype(float)
        weighted_y = (np.arange(mask.shape[0])[:, None] * mask_float).sum() / (mask_float.sum() + 1e-6)
        vertical_center_norm = weighted_y / h_img

        is_tall = float(height_norm > 0.3)

        # 头部存在性判断
        head_region_height = 0.2 * (y2 - y1)
        head_y_min = y1
        head_y_max = y1 + head_region_height
        head_pixels = mask[int(head_y_min):int(head_y_max), :].sum()
        has_head = head_pixels > 10

        if not has_head:
            return None

        # 手部相对位置
        if len(ys) > 0:
            body_center_y = (y1 + y2) / 2
            hand_relative_y = (hand_center_y - body_center_y) / (y2 - y1 + 1e-6) if 'hand_center_y' in locals() else 0.0
        else:
            hand_relative_y = 0.0
        
        # 头部朝向
        head_mask = mask[int(head_y_min):int(head_y_max), :]
        if head_mask.sum() > 0:
            head_ys, head_xs = np.where(head_mask)
            if len(head_ys) > 0:
                head_center_x = np.mean(head_xs)
                head_width = head_xs.max() - head_xs.min()
                head_aspect = head_width / (head_ys.max() - head_ys.min() + 1e-6)
                head_orientation = float(head_center_x > w_mask / 2)  # 0: 左，1: 右
            else:
                head_orientation = 0.0
        else:
            head_orientation = 0.0

        # 身体倾斜度
        if len(xs) > 0 and len(ys) > 0:
            cov_matrix = np.cov(np.vstack([xs, ys]))
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            major_axis_angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            body_tilt = np.sin(major_axis_angle)
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


    def classify_roi_from_mask(self, image: np.ndarray, mask: torch.Tensor):
        """
        对单个 mask 区域进行动作分类
        """
        print(f"🔍 Processing mask of shape: {mask.shape if isinstance(mask, np.ndarray) else mask.shape}")
        if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
        else:
                mask_np = mask

        mask_bool = mask_np.astype(bool)
        ys, xs = np.where(mask_bool)
        if len(ys) == 0:
                return 0, "unknown", 0.0, [0, 0, 0, 0]

        # 计算 tight bbox
        # x1_mask, x2_mask = xs.min(), xs.max()
        # y1_mask, y2_mask = ys.min(), ys.max()
        bbox = self.get_tight_bbox_from_mask(mask_bool)
        if bbox is None:
            return 0, "unknown", 0.0, [0, 0, 0, 0]
        x1_mask, y1_mask, x2_mask, y2_mask = bbox
    
            # 缩放到原始图像坐标
        h_mask, w_mask = mask_bool.shape
        h_img, w_img = image.shape[:2]
        scale_h = h_img / h_mask
        scale_w = w_img / w_mask
        x1_orig = int(x1_mask * scale_w)
        y1_orig = int(y1_mask * scale_h)
        x2_orig = int(x2_mask * scale_w)
        y2_orig = int(y2_mask * scale_h)
        tight_box = [x1_orig, y1_orig, x2_orig, y2_orig]

        # 面积过滤
        area = (x2_orig - x1_orig) * (y2_orig - y1_orig)
        total_area = h_img * w_img
        area_ratio = area / total_area
        if area_ratio < 0.001 or area_ratio > 0.25:
            return 0, "unknown", 0.0, [0, 0, 0, 0]

        # 裁剪 ROI
        min_margin = 20
        margin_w = max(min_margin, int((x2_orig - x1_orig) * 0.15))
        margin_h = max(min_margin, int((y2_orig - y1_orig) * 0.15))
        x1_crop = max(0, x1_orig - margin_w)
        y1_crop = max(0, y1_orig - margin_h)
        x2_crop = min(w_img, x2_orig + margin_w)
        y2_crop = min(h_img, y2_orig + margin_h)

        roi = image[y1_crop:y2_crop, x1_crop:x2_crop]
        if roi.size == 0:
            return 0, "unknown", 0.0, [0, 0, 0, 0]

        roi_h, roi_w = roi.shape[:2]
        aspect_ratio = roi_w / roi_h
        if aspect_ratio < 0.3 or aspect_ratio > 3.0 or roi_h < 64 or roi_w < 64:
            return 0, "unknown", 0.0, [0, 0, 0, 0]

        # 提取几何特征
        geo_features = self.extract_geometric_features(mask_np, tight_box, image.shape)
        if geo_features is None:
            return 0, "unknown", 0.0, [0, 0, 0, 0]
        geo_features_tensor = torch.from_numpy(geo_features).float().to(self.device).unsqueeze(0)

        ## 图像预处理
        roi_pil = Image.fromarray(roi).convert('RGB')
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        roi_tensor = transform(roi_pil).unsqueeze(0).to(self.device)

        # 特征提取 + 分类
        with torch.no_grad():
            cnn_features = self.action_backbone(roi_tensor)
            cnn_features_flat = F.adaptive_avg_pool2d(cnn_features, 1).flatten(1)
            print(f"🔍 CNN features shape: {cnn_features_flat.shape}")  # 应该是 (1, 1280)
            print(f"🔍 Geo features shape: {geo_features_tensor.shape}") 
            combined_features = torch.cat([cnn_features_flat, geo_features_tensor], dim=1)
            print(f"🔍 Combined features shape: {cnn_features.shape}")
            logits = self.action_head(cnn_features)
            # 添加 Temperature Scaling
            # temperature = 1.5
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)

        action_id = pred.item()
        confidence = conf.item()
        action_name = self.action_classes[action_id]

        print(f"📊 Prediction: {action_name} (conf: {confidence:.3f}, id: {action_id})")
        print(f"   All probs: {[f'{name}:{prob:.3f}' for name, prob in zip(self.action_classes, probs[0].cpu().numpy())]}")
        return action_id, action_name, confidence, tight_box
    
    
    def get_tight_bbox_from_mask(self, mask_bool):
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        return [x, y, x + w, y + h]
        
    

    def crop_image(self, image, crop_box, sim_map=None):
        
        #crop and then resize image
        x0,y0,x1,y1 = crop_box
        #adapt crop region guided by semantic prior
        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint8)
        self.orig_image = image
        #crop area represents the area of image to operate
        image = image[y0:y1, x0:x1,:]
        image,r = utils.resize_image(image, self.max_size)
        self.image = image
        self.downscale = r

    @torch.no_grad()
    def generate(self, image: np.ndarray):
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.
        Returns:
           predictions (MaskData): The dict that stores predictions in np.ndarray format
             boxes: xyxy format
             scores: >0
             masks: only exists when visualize is enabled
        """

        
        # Generate masks
        mask_data = self._generate_masks(image)
        # Filter small disconnected regions and holes in masks

        # 增加
        # 对每个 mask 输出动作类别
        if len(mask_data['masks']) > 0:
            masks = mask_data['masks']
            actions = []
            tight_boxes = []

            for mask in masks:
                # 提取 mask 特征（可选）
                # 使用 mask decoder 的中间特征进行分类
                action_id, action_name, conf, tight_box = self.classify_roi_from_mask(image, mask)
                actions.append({
                    'action_id': action_id,
                    'action_name': action_name,
                    'confidence': conf
                })
                tight_boxes.append(tight_box)

            mask_data['boxes'] = torch.tensor(tight_boxes, dtype=torch.float32)
            mask_data['actions'] = actions
        else:
            mask_data['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            mask_data['actions'] = []
        return mask_data

    def _generate_masks(self, image):
        img_size = np.array(image).shape[:2]
        # print(f"🔍 Input image size: {img_size}")  # 调试1
         
        # print(f"🔧 crop_n_layers: {self.crop_n_layers}")
        # print(f"🔧 crop_overlap_ratio: {self.crop_overlap_ratio}")
        # print(f"🔧 max_size: {self.max_size}")
               
        crop_boxes, layer_idxs = generate_crop_boxes(
            img_size, self.crop_n_layers, self.crop_overlap_ratio
        )
        # print(f"🔍 Generated {len(crop_boxes)} crop boxes")  # 调试2
        for i, box in enumerate(crop_boxes):
            print(f"  Crop {i+1}: {box}")
        layer_idxs = np.ones(len(crop_boxes))    
        data = MaskData()
        #===============> Step 2. Process Crops   
        for i, (crop_box, layer_idx) in enumerate(zip(crop_boxes, layer_idxs)):
            # print(f"🔍 Processing crop {i+1}/{len(crop_boxes)}: {crop_box}")
            crop_data = self._process_crop(image, crop_box)
        
            if crop_data is None:
                # print(f"  ❌ Crop {i+1} returned None")
                continue
            
            if 'masks' in crop_data._stats and len(crop_data['masks']) > 0:
                # print(f"  ✅ Crop {i+1} generated {len(crop_data['masks'])} masks")

                # 调试
                before_cat_masks = len(data['masks']) if 'masks' in data._stats else 0
                before_cat_keys = list(data._stats.keys())
                # print(f"  📊 Before cat: keys={before_cat_keys}, masks_count={before_cat_masks}")

                data.cat(crop_data)

                after_cat_masks = len(data['masks']) if 'masks' in data._stats else 0
                after_cat_keys = list(data._stats.keys())
                # print(f"  📊 After cat: keys={after_cat_keys}, masks_count={after_cat_masks}")


            else:
                print(f"  ❌ Crop {i+1} generated 0 masks")
            del crop_data
            
            logger.debug(f"#{layer_idx} crop area {str(crop_box)}")
        # Remove duplicate masks between crops
        if len(crop_boxes) > 1 and 'crop_boxes' in data._stats and len(data['crop_boxes']) > 0:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)
            del data['crop_boxes']
        if len(data._stats.keys()) > 0:
            del data['iou_preds']
        else:
            data['boxes'] = torch.zeros(0, 4)
            data['scores'] = torch.zeros(0, 4)
        if 'rles' in data._stats:
            data["rles"] = [coco_encode_rle(rle) for rle in data["rles"]] 
        else:
            data['rles'] = []
        

        # 调试
        # print(f"🔍 Before to_numpy: keys={list(data._stats.keys())}")
        # if 'masks' in data._stats:
        #     print(f"✅ Before to_numpy: masks_count={len(data['masks'])}")
        # else:
        #     print(f"❌ Before to_numpy: NO MASKS!")

        data.to_numpy()    
        # print(f"🔍 Total masks after all crops: {len(data['masks']) if 'masks' in data._stats else 0}")
        # print(f"🔍 [DEBUG] _generate_masks: Final data keys = {list(data._stats.keys())}")
        # if 'masks' in data._stats:
        #     print(f"✅ [DEBUG] _generate_masks: Total masks = {len(data['masks'])}")
        # else:
        #     print(f"❌ [DEBUG] _generate_masks: NO MASKS in final data!")
        return data
    
    def _process_crop(self, image, crop_box):
    
        self.crop_image(image, crop_box)
        # print(f"📸 [DEBUG] _process_crop: Cropped image shape = {self.image.shape}")
        self.predictor.set_image(self.image)
        orig_h, orig_w = self.orig_image.shape[:2]
        img_size = torch.tensor(self.image.shape[:2])
        if not self.train_free:
            # dino_feats = self.predictor.dino_feats
            feat_size = (img_size * min(self.grid_size/img_size)).int()         
            sim_map = self.predictor.predict_fg_map(img_size)
            sim_map = torch.nn.functional.interpolate(sim_map, (self.grid_size, self.grid_size), mode='bilinear')
            sim_map = sim_map.sigmoid().max(dim=1)[0]
            sim_map = sim_map[0,:feat_size[0], :feat_size[1]]
            sim_thresh =self.pos_sim_thresh
        else:
        #used when  dino_feats = self.predictor.dino_feats
            transform = T.Compose([
                    T.Resize((1022, 1022)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            dino_feats = self.extract_features(self.image, transform)
            feat_size = dino_feats.shape[:2]
            feature_sim = F.cosine_similarity(self.ref_feature, dino_feats.flatten(0,1)) 
            feature_sim = feature_sim.reshape(*feat_size)
            sim_map = feature_sim
            sim_thresh =  self.pos_sim_thresh
        coords = self.match_ref(sim_map, sim_thresh).cpu()
        inv_factor = torch.tensor([feat_size[1]/self.image.shape[1], feat_size[0]/self.image.shape[0]])
        coords = (coords ) /  inv_factor
        # prompt_coords = utils.composite_clustering(coords, self.num_prompts, self.device)[0]
        points_for_image = coords.cpu().numpy()
        # print(f"📍 [DEBUG] _process_crop: Generated {len(points_for_image)} prompt points")
        # if len(points_for_image) == 0:
        #     print("❌ [DEBUG] _process_crop: No prompt points, returning None")
        #     return None

        logger.debug(f'len points {len(points_for_image)}')
        #No change here
        data = MaskData()
        occupy_mask = torch.zeros(*img_size, dtype=torch.bool)
        
        def efficient_batch_iterator(batch_size: int, points):
            points = points.astype('int')
            np.random.shuffle(points)
            count = 0
            while len(points)>0 and count < self.max_prompts:
                batch_size = min(len(points), batch_size)
                sel_pts= points[:batch_size]
                points = points[batch_size:]
                yield sel_pts
                keep = (~occupy_mask[points[:,1], points[:,0]]).numpy()
                points = points[keep]
                count += batch_size
                
        for points in efficient_batch_iterator(self.points_per_batch, points_for_image):
            if len(points) ==0:
                continue
            batch_data = self._process_batch(points, self.predictor.original_size, crop_box)
            if batch_data is None:
                # print(f"  ❌ [DEBUG] _process_crop: No masks for batch")
                continue
            occupy_mask = (batch_data['masks'][batch_data['iou_preds']> self.filter_thresh]).any(0).cpu()

            # 调试
            before_cat = len(data['masks']) if 'masks' in data._stats else 0
            data.cat(batch_data)
            after_cat = len(data['masks']) if 'masks' in data._stats else 0
            # print(f"📊 [DEBUG] data.cat: {before_cat} → {after_cat} masks")

            del batch_data
        self.predictor.reset_image()
        
        # The data maybe empty here since prompts are dynamic
        if len(data.items()) == 0:
            return None
        if len(data['masks']) == 0:
            return None

        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)
        
        if self.min_mask_region_area > 0:
            min_mask_region_area = self.min_mask_region_area  #* (self.downscale)**2
            data = self.postprocess_small_regions(
                data,
                min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )
        #Implement joint classification scores here 
        if self.fuse_simmap:
            sim_map_high_res = F.interpolate(sim_map.unsqueeze(0).unsqueeze(0), self.image.shape[:2],mode='bilinear')[0,0].cuda()  
            # cls_scores = self.evaluate_cls_scores(data['masks'], sim_map_high_res, clf)
            cls_scores = []
            for mask in data['masks']:
                if mask.sum() > 0:
                    cls_score = sim_map_high_res[mask].mean()
                else:
                    cls_score = 0
                # cls_score = (cls_score -cls_score.min())/ (cls_score.max() - cls_score.min())
                cls_score = torch.clamp(cls_score + 0.5,0, 1)
                cls_scores.append(cls_score)
            cls_scores = torch.tensor(cls_scores).to(self.device)
            data['scores'] = data['iou_preds'] ** 0.5  * cls_scores ** 0.5
        
        else:
            data['scores'] = data['iou_preds']
            
        # data["rles"] = mask_to_rle_pytorch((utils.uncrop_masks(data["masks"], crop_box, orig_h, orig_w)))
        data['rles'] = mask_to_rle_pytorch(data["masks"])
        # data['rles_info'] = [crop_box,[orig_h, orig_w]]
        if 'masks' in data._stats and len(data['masks']) > 0:
            data['rles_info'] = [ [crop_box, [orig_h, orig_w]] for _ in range(len(data['masks'])) ]
        else:
            # 如果没有 masks，不要添加这个键
            pass
        # del data['masks']
            
        #Implement the box_offsets herehere 
            #apply box offsets here
        data["boxes"] = utils.uncrop_boxes_xyxy(data["boxes"], crop_box, self.downscale)
        data['points'] = utils.uncrop_points(data['points'], crop_box, self.downscale)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["boxes"]))])
        if self.apply_box_offsets:
            ext_boxes = utils.apply_box_offsets(data['boxes'], data['box_offsets'])
            data['fboxes'] = ext_boxes
        else:
            data['fboxes'] = data['boxes']

        # 将mask调整到原始图像尺寸
        if 'masks' in data._stats and len(data['masks']) > 0:
            orig_h, orig_w = self.orig_image.shape[:2]
            uncropped_masks = []
        
            for mask in data['masks']:
                # 创建原始尺寸的空 mask
                full_mask = torch.zeros(orig_h, orig_w, dtype=mask.dtype, device=mask.device)
                # 获取 crop 区域坐标
                x0, y0, x1, y1 = crop_box
                # 调整 mask 尺寸到 crop 区域大小
                if mask.shape[0] != (y1 - y0) or mask.shape[1] != (x1 - x0):
                    mask_resized = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(y1 - y0, x1 - x0),
                        mode='nearest'
                    ).squeeze(0).squeeze(0) > 0.5
                else:
                    mask_resized = mask > 0.5
            
                # 将 mask 放回原始位置
                full_mask[y0:y1, x0:x1] = mask_resized
                uncropped_masks.append(full_mask)
        
            # 替换为原始尺寸的 masks
            data['masks'] = torch.stack(uncropped_masks, dim=0)

        # print(f"✅ [DEBUG] _process_crop: Returning {len(data['masks'])} masks")
        masks_count = len(data['masks']) if 'masks' in data._stats else 0
        print(f"✅ [DEBUG] _process_crop: Returning {masks_count} masks")
        
        return data

    @torch.no_grad()
    def extract_features(self, image, transform):
        t = transform(Image.fromarray(image))
        _,h,w = t.shape
        feat_h, feat_w = h//self.patch_size, w//self.patch_size
        features_dict = self.image_encoder.forward_features(t.unsqueeze(0).to(self.device))
        features = features_dict['x_norm_patchtokens'].flatten(0,1)
        feat_size = torch.tensor((feat_h, feat_w))
        return features.reshape(feat_h, feat_w, -1)
    
    def select_mask(self, masks, iou_preds):
        bin_masks = masks  > self.predictor.model.mask_threshold
        if self.mask_selection == 'max_area':
            ind = bin_masks.sum(dim=[-1,-2]).max(dim=-1)[1] # L
        elif self.mask_selection == 'min_area':
            ind = bin_masks.sum(dim=[-1,-2]).min(dim=-1)[1] # L
        elif self.mask_selection == 'max_iou':
            ind = iou_preds.max(dim=-1)[1]
        elif self.mask_selection == 'all':
            return masks.flatten(0,1)
        else:
            raise NotImplementedError
        indices = torch.arange(len(masks)), ind
        return indices
        # 

    def _process_batch(
        self,
        points: np.ndarray,
        im_size,
        crop_box
    ) -> MaskData:
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)

        masks, iou_preds, cls_scores = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )[:3]
        if not self.train_free:
            conf, categories = cls_scores.sigmoid().max(dim=-1)            
            indices = self.select_mask(masks, iou_preds)
            iou_preds = iou_preds  * conf
            masks, iou_preds, points, categories = masks[indices], iou_preds[indices], torch.as_tensor(points.repeat(1, axis=0)), categories[indices]
        else:
            iou_preds = torch.clamp(iou_preds, 0.) * cls_scores.squeeze(2).sigmoid() 
            indices = self.select_mask(masks, iou_preds)  
            masks, iou_preds, points,  = masks[indices], iou_preds[indices], torch.as_tensor(points.repeat(1, axis=0))
            categories = torch.zeros(len(masks)).int()
        # return 
        #Feature: Select proper mask according to IoU 
            
        #, (torch.clamp(iou_preds,0) * conf.sigmoid()), points, categories)
        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks,
            iou_preds=iou_preds,
            points=points,
            categories = categories
        )
        print(f"📦 [DEBUG] _process_batch: Raw masks shape = {masks.shape}")
        print(f"📦 [DEBUG] _process_batch: Raw iou_preds shape = {iou_preds.shape}")
        del masks
        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            print(f"📦 [DEBUG] _process_batch: IoU filter - {keep_mask.sum()}/{len(keep_mask)} kept")
            data.filter(keep_mask)
        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold ,self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            print(f"📦 [DEBUG] _process_batch: Stability filter - {keep_mask.sum()}/{len(keep_mask)} kept")
            data.filter(keep_mask)
        
        print(f"📦 [DEBUG] _process_batch: Final masks = {len(data['masks']) if 'masks' in data._stats else 0}")
        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])
        
        orig_h, orig_w = self.orig_image.shape[:2]
        keep_mask = ~utils.is_box_near_crop_edge(data["boxes"], crop_box,  [0, 0, orig_w, orig_h], self.downscale)
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        if len(data['masks']) == 0:
            print("❌ [DEBUG] _process_batch: All masks filtered out!")
            return None
        return data
    

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["masks"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        # import time
        # t = time.time()
        for mask in mask_data["masks"].cpu().numpy():

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed
            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))
        # print(time.time() -t)

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_data["boxes"][i_mask] = torch.as_tensor(boxes[i_mask], device=mask_data["masks"].device)  # update res directly
                mask_data['masks'][i_mask] = torch.as_tensor(masks[i_mask], device=mask_data["masks"].device)
        mask_data.filter(keep_by_nms)

        return mask_data

    def match_ref(self, sim_map, pos_sim_thresh):
        #TODO: It remains a question whether it is better to do some sim map fusion here
        fg_mask = sim_map > pos_sim_thresh
        coords = fg_mask.nonzero()[:,[1,0]]
        return coords
