from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
import _init_paths
import models
from config import cfg
from config import update_config
from torchsummaryX import summary


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    cfg.defrost()
    cfg.freeze()

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    model.eval()
    dump_input = torch.rand(
            (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
        )
    summary(model, dump_input)

if __name__ == '__main__':
    main()
