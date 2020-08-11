"""DBox
Original author: YutaroOgawa
https://github.com/YutaroOgawa/pytorch_advanced/blob/master/2_objectdetection/utils/ssd_model.py
"""

import torch

from itertools import product
from math import sqrt

class DBox(object):
    def __init__(self, **cfg):
        super(DBox, self).__init__()

        self.img_size = cfg['img_size']
        self.feature_maps = cfg['feature_maps']
        self.num_priors = len(cfg['feature_maps'])
        self.steps = cfg['steps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.aspect_ratios = cfg['aspect_ratios']

    def make_dbox_list(self):

        mean = []
        # features_maps: [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                # size of feature map
                # 300 / steps:[8, 16, 32, 64, 100, 300]
                f_k = self.img_size / self.steps[k]

                # center coords (x, y) of DBox, normalized[0, 1]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # small DBox [cx, cy, width, heigh]
                s_k = self.min_sizes[k] / self.img_size
                mean += [cx, cy, s_k, s_k]

                # large DBox [cx, cy, width, heigh]
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.img_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # torch.Size([8372, 4])
        output = torch.Tensor(mean).view(-1, 4)

        # clamp if size of DBox is larger than 1
        output.clamp_(max=1, min=0)

        return output


if __name__ == "__main__":

    ssd_cfg = {
        'n_classes': 21,
        'img_size': 300,
        'bbox_aspect_ratios': [4, 6, 6, 6, 4, 4],
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 11, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }

    dbox = DBox(**ssd_cfg)
    dbox_list = dbox.make_dbox_list()

    import pandas as pd
    print(pd.DataFrame(dbox_list))