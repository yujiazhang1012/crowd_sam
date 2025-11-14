import torch
import json
import os
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from .coco_names import coco_classes
data_meta = {'crowdhuman': ["./datasets/crowdhuman", 7,
                   {1: 'write', 2: 'read', 3: 'lookup', 4: 'turn_head',
                    5: 'raise_hand', 6: 'stand', 7: 'discuss'}],
             'occhuman': ["./datasets/OCHuman", 7,
                   {1: 'write', 2: 'read', 3: 'lookup', 4: 'turn_head',
                    5: 'raise_hand', 6: 'stand', 7: 'discuss'}],
             'coco_occ':["./datasets/coco", 80, coco_classes],
             'coco':["./datasets/occ_coco", 80, coco_classes], 
             }
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, image_directory, transform=None):
        """
        Initialize COCO dataset
        
        Args:
            annotation_file (str): Path to COCO annotation file (JSON)
            image_directory (str): Path to directory containing images
            transform (callable, optional): Optional transform to be applied to the image
        """
        self.coco = COCO(annotation_file)
        self.image_directory = image_directory
        self.transform = transform
        self.image_ids = self.coco.getImgIds()
        self.mapping = self.create_category_mapping()
    def __len__(self):
        """
        Return the total number of samples in the dataset
        """
        return len(self.image_ids)
    def create_category_mapping(self):
        """
        Create a mapping from COCO category IDs to a continuous range from 0 to 80
        
        Returns:
            dict: Mapping from COCO category IDs to a continuous range from 0 to 80
        """
        coco_categories = self.coco.loadCats(self.coco.getCatIds())
        coco_category_ids = sorted([category['id'] for category in coco_categories])
        category_mapping = {coco_id: idx for idx, coco_id in enumerate(coco_category_ids)}
        return category_mapping
    def __getitem__(self, idx):
        """
        Get the sample at the specified index
        
        Args:
            idx (int): Index of the sample to retrieve
        
        Returns:
            dict: Dictionary containing the image and its associated masks and categories
        """
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = f"{self.image_directory}/train2017/{image_info['file_name']}"
        image = Image.open(image_path)

        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        
        masks = [self.coco.annToMask(annotation) for annotation in annotations]
        category_ids = [self.mapping[annotation['category_id']] for annotation in annotations]

        sample = {
            'image': image,
            'masks': masks,
            'categories': category_ids
        }


        return sample

class CrowdHuman(torch.utils.data.Dataset):
    def __init__(self, dataset_root, annot_path, transform, use_sam_masks=True):
        super().__init__()
        self.dataset_root = dataset_root
        self.transform = transform
        img_dir = 'Images'        
        with open(annot_path, 'r') as f:
            annots = json.load(f)
        annotations = annots['annotations']
        images = annots['images']
        self.image_ids = [img['id'] for img in images]
        self.image_files = [
            os.path.join(dataset_root, img_dir, img['file_name'])
            for img in images
        ]
        # 创建 image_id -> file_path 映射
        image_id_to_path = {
            img['id']: os.path.join(dataset_root, img_dir, img['file_name'])
            for img in images
        }
        self.samples = []
        for annot in annotations:
            image_id = int(annot['image_id'])
            if image_id in image_id_to_path:
                image_path = image_id_to_path[image_id]
                bbox = annot['bbox']
                label = annot['category_id'] - 1  # 转为 0~6
                self.samples.append((image_path, bbox, label))
    def __getitem__(self, item):
        image_path, bbox, label = self.samples[item]
        img = Image.open(image_path).convert('RGB')
        x, y, w, h = bbox
        roi = img.crop((x, y, x + w, y + h))
        
        if self.transform:
            roi = self.transform(roi)
        
        geo_features = torch.zeros(8)
        return roi, label, geo_features
    def __len__(self):
        return len(self.image_files)
    
# def collate_fn_crowdhuman(data):
#     rois, labels, geo_features = zip(*data)
#     rois = torch.stack(rois, dim=0)
#     labels = torch.tensor(labels, dtype=torch.long)
#     geo_features = torch.stack(geo_features, dim=0)
#     return rois, labels, geo_features
def collate_fn_crowdhuman(batch):
    """
    自定义collate函数, 处理包含bbox和image_shape的数据
    预期每个batch item是字典格式: {'roi': ..., 'label': ..., 'bbox': ..., 'image_shape': ...}
    """
    # 检查数据格式
    if isinstance(batch[0], dict):
        # 处理字典格式数据
        rois = torch.stack([item['roi'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        
        # 处理边界框 - 确保是float类型
        bboxes = []
        for item in batch:
            bbox = item['bbox']
            if isinstance(bbox, np.ndarray):
                bbox = torch.from_numpy(bbox).float()
            elif isinstance(bbox, list):
                bbox = torch.tensor(bbox, dtype=torch.float)
            bboxes.append(bbox)
        bboxes = torch.stack(bboxes)
        
        # 处理图像尺寸 - 确保是int类型
        image_shapes = []
        for item in batch:
            shape = item['image_shape']
            if isinstance(shape, np.ndarray):
                shape = torch.from_numpy(shape).int()
            elif isinstance(shape, list) or isinstance(shape, tuple):
                shape = torch.tensor(shape, dtype=torch.int)
            image_shapes.append(shape)
        image_shapes = torch.stack(image_shapes)
        
        return rois, labels, bboxes, image_shapes
        
    else:
        # 向后兼容：处理旧格式 (rois, labels, geo_features)
        rois, labels, geo_features = zip(*batch)
        rois = torch.stack(rois, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        geo_features = torch.stack(geo_features, dim=0)
        return rois, labels, geo_features


def collate_fn_coco(data):
    images = [d['image'] for d in  data]
    masks = [d['masks'] for d in data]
    categories = [d['categories'] for d in data]
    # images,boxes, boxe_full= zip(*data.values())
    return images, masks, categories

