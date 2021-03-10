# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
from multiprocessing import Process, Queue

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm

import _init_paths
import models
from collections import OrderedDict

from config import cfg
from config import check_config
from config import update_config
from dataset import make_test_dataloader
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from fp16_utils.fp16util import network_to_half
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.vis import save_debug_images
from utils.vis import save_valid_image
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size

from IPython import embed
import cv2
import numpy as np
import json

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    parser = argparse.ArgumentParser(description="Test keypoints network")
    # general
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--world_size",
        help="Modify config options using the command-line",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    return args


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info("| Arch " + " ".join(["| {}".format(name) for name in names]) + " |")
    logger.info("|---" * (num_values + 1) + "|")

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + "..."
    logger.info(
        "| "
        + full_arch_name
        + " "
        + " ".join(["| {:.3f}".format(value) for value in values])
        + " |"
    )



def worker(gpu_id, dataset, indices, cfg, logger, final_output_dir, pred_queue):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(cfg, is_train=False)

    dump_input = torch.rand((1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE))

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.TEST.MODEL_FILE:
        logger.info("=> loading model from {}".format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(final_output_dir, "model_best.pth.tar")
        logger.info("=> loading model from {}".format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
        
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    sub_dataset = torch.utils.data.Subset(dataset, indices)
    data_loader = torch.utils.data.DataLoader(
        sub_dataset, sampler=None, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )

    if cfg.MODEL.NAME == "pose_hourglass":
        transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    parser = HeatmapParser(cfg)
    all_preds = []
    all_scores = []

    pbar = tqdm(total=len(sub_dataset)) if cfg.TEST.LOG_PROGRESS else None
    for i, (images, annos) in enumerate(data_loader):
        assert 1 == images.size(0), 'Test batch size should be 1'

        image = images[0].cpu().numpy()
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
        )

        with torch.no_grad():
            final_heatmaps = None
            tags_list = []
            for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                input_size = cfg.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                )
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                    cfg.TEST.PROJECT2IMAGE, base_size
                )

                final_heatmaps, tags_list = aggregate_results(
                    cfg, s, final_heatmaps, tags_list, heatmaps, tags
                )

            final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)

            visual = False
            if visual:
                visual_heatmap = torch.max(final_skelemaps[0], dim=0, keepdim=True)[0]
                visual_heatmap = (
                    visual_heatmap.cpu().numpy().repeat(3, 0).transpose(1, 2, 0)
                )

                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                visual_img = (
                    image_resized[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
                )
                visual_img = visual_img[:, :, ::-1] * np.array(std).reshape(
                    1, 1, 3
                ) + np.array(mean).reshape(1, 1, 3)
                visual_img = visual_img * 255
                test_data = cv2.addWeighted(
                    visual_img.astype(np.float32),
                    0.0,
                    visual_heatmap.astype(np.float32) * 255,
                    1.0,
                    0,
                )
                cv2.imwrite("test_data/{}.jpg".format(i), test_data)

            grouped, scores = parser.parse(
                final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
            )

            final_results = get_final_preds(
                grouped, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )

        if cfg.TEST.LOG_PROGRESS:
            pbar.update()
        
        data_idx = indices[i]
        img_id = dataset.ids[data_idx]
        file_name = dataset.coco.loadImgs(img_id)[0]["file_name"]

        for idx in range(len(final_results)):
            all_preds.append({
                "keypoints": final_results[idx][:,:3].reshape(-1,).astype(np.float).tolist(),
                "image_id": int(file_name[-16:-4]),
                "score": float(scores[idx]),
                "category_id": 1
            })

    if cfg.TEST.LOG_PROGRESS:
        pbar.close()
    pred_queue.put_nowait(all_preds)

def main():
    args = parse_args()

    update_config(cfg, args)
    check_config(cfg)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, "valid")

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    _, dataset = make_test_dataloader(cfg)

    total_size = len(dataset)
    pred_queue = Queue(100)
    workers = []
    for i in range(args.world_size):
        indices = list(range(i, total_size, args.world_size))
        p = Process(
            target = worker,
            args = (
                i, dataset, indices, cfg, logger, final_output_dir, pred_queue
            )
        )
        p.start()
        workers.append(p)
        logger.info("==>" + " Worker {} Started, responsible for {} images".format(i, len(indices)))
    
    all_preds = []
    for idx in range(args.world_size):
        all_preds += pred_queue.get()
    
    for p in workers:
        p.join()

    res_folder = os.path.join(final_output_dir, "results")
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    res_file = os.path.join(res_folder, "keypoints_%s_results.json" % dataset.dataset)

    json.dump(all_preds, open(res_file, 'w'))

    info_str = dataset._do_python_keypoint_eval(res_file, res_folder)
    name_values = OrderedDict(info_str)
    
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(logger, name_value, cfg.MODEL.NAME)
    else:
        _print_name_value(logger, name_values, cfg.MODEL.NAME)


if __name__ == "__main__":
    main()
