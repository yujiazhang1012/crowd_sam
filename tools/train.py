
import os
import json
import itertools 
import tqdm

# import cv2
import torch
import torchvision.transforms as T

import numpy as np
import sys
# from utils import draw_mask,draw_point,draw_box
from segment_anything_cs import sam_model_registry, SamPredictor
from loguru import logger
import crowdsam.utils as utils
from crowdsam.data import COCODataset, collate_fn_coco, CrowdHuman, collate_fn_crowdhuman
from crowdsam.engine import train_loop


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-c','--config_file', default='configs/crowdhuman.yaml')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    config = utils.load_config(args.config_file)
    #fixed config
    model_arch = 'dino'
    mode = 'training'
    #data related
    #===========>fix seed for reproduction
    np.random.seed(42)
    torch.random.manual_seed(42)
    #===========>set arguments or options 
    #datasets
    # #set data stream
    sam = sam_model_registry[config['model']['sam_model']](checkpoint= config['model']['sam_checkpoint'],n_class=config['model']['n_class'])
    sam.cuda() 
    dino_repo = config['model']['dino_repo']
    dino = torch.hub.load(dino_repo, config['model']['dino_model'],source='local',pretrained=False).cuda()
    dino.load_state_dict(torch.load(config['model']['dino_checkpoint']))
    predictor = SamPredictor(sam, dino)
    learnable_params = list(itertools.chain(predictor.model.mask_decoder.parallel_iou_head.parameters(),
                                                         predictor.model.mask_decoder.point_classifier.parameters(),
                                                         predictor.model.mask_decoder.dino_proj.parameters(),
                                                         ))
    size = 0
    for param in predictor.model.parameters():
        param.requires_grad = False
    for param in learnable_params:
        param.requires_grad_()
        size += param.numel()
    print('total learnable parameters:', size)
    optimizer = torch.optim.AdamW(params= learnable_params,
                                  lr= config['train']['lr'], weight_decay= config['train']['weight_decay'])
    # prompts = generate_prompts(patch_h, patch_w).cuda()
    if config['data']['dataset'] == 'coco':
        torch.nn.init.constant_(predictor.model.mask_decoder.point_classifier.layers[-1].bias,-5)
        dataset = COCODataset(image_directory=config['data']['dataset_root'], annotation_file=config['data']['train_file'], transform=T.ToTensor())
        train_dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=True, num_workers=1, drop_last=False,collate_fn=collate_fn_coco)
    elif config['data']['dataset'] == 'crowdhuman':
        dataset = CrowdHuman(config['data']['dataset_root'], config['data']['train_file'], transform=T.ToTensor())
        train_dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=True, num_workers=1, drop_last=False,collate_fn=collate_fn_crowdhuman)

    loss_type = 'coco' if config['data']['dataset'] == 'coco' else 'crowdhuman'
    train_loop(train_dataloader, predictor, optimizer,config['train']['steps'], config['train']['n_shot'], config['train']['samples_per_batch'],config['train']['neg_samples_per_batch'], config['model']['n_class'], loss_type=loss_type,debug=args.debug) 
    #save path
    save_path = config['train']['save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(predictor.model.mask_decoder.state_dict(),  save_path)
    logger.info('done')
   