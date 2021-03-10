# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import os.path

import cv2
import numpy as np
from torch.utils.data import Dataset
from glob import glob

logger = logging.getLogger(__name__)


class InferDataset(Dataset):
    """`

    Args:
        root (string): Root directory where dataset is located to.
        extension (string): extension of tested images.
    """

    def __init__(self, root, extension=".jpg"):
        self.root = root
        self.extension = extension
        self.img_paths = glob(os.path.join(self.root, "*"+extension))

    def __getitem__(self, index, return_path=True):
        """
        Args:
            index (int): Index
            return_path (bool): If return the path of images

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        img_path = self.img_paths[index]

        img = cv2.imread(
            img_path, 
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )

        if return_path:
            return img, img_path
        else:
            return img

    def __len__(self):
        return len(self.img_paths)
