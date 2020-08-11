"""SSD
Original author: YutaroOgawa
https://github.com/YutaroOgawa/pytorch_advanced/blob/master/2_objectdetection/utils/ssd_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.ssd.modules.vgg import make_vgg
from modeling.ssd.modules.extras import make_extras
from modeling.ssd.modules.l2norm import L2Norm
from modeling.ssd.modules.loc_conf import make_loc_conf
from modeling.ssd.modules.dbox import DBox
from modeling.ssd.detection import Detect


class SSD(nn.Module):
    def __init__(self, **cfg):
        super(SSD, self).__init__()

        self.n_classes = cfg['n_classes']

        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(
            cfg['n_classes'], cfg['bbox_aspect_num']
        )

        # make DBox: torch.Size([8372, 4])
        dbox = DBox(**cfg)
        self.dbox_list = dbox.make_dbox_list().to(cfg['device'])

        self.detect = Detect(variances=cfg['variances'], conf_thresh=cfg['conf_thresh'], top_k=cfg['top_k'], nms_thresh=cfg['nms_thresh'])

    def forward(self, x, phase='train'):
        sources = list() # input for loc and conf
        loc = list() # output for loc
        conf = list() # output for conf

        # calculate until vgg conv4_3
        for k in range(23):
            x = self.vgg[k](x)

        source1 = self.L2Norm(x)
        sources.append(source1)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        # extras conv and ReLU
        # add source 3~6 to sources 
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # convolution with l(x) and c(x)
            # l(x), c(x): torch.size([batch_num, 4*aspect, featueremap height, featuremap width])
            # permute => [num minibatch, num feature map, num feature map, 4*aspect]
            # torch.contiguous: an instruction to reposition elements sequentially in memory. (to use view, it's necessary)

        # loc: torch.Size([batch_num, 34928])
        # conf: torch.Size([batch_num, 183372])
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # loc: torch.Size([batch_num, 8732, 4])
        # conf: torch.Size([batch_num, 8732, 21])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.n_classes)

        output = (loc, conf, self.dbox_list)

        if phase == 'test':
            # Detect forward
            # outputs: torch.Size([batch_num, 21, 200, 5])
            # 5: [conf, xmin, ymin, xmax, ymax]
            detect_outputs = self.detect(loc, conf, self.dbox_list)
            return output, detect_outputs

        # output: (loc, conf, dbox_list)
        return output

if __name__ == "__main__":
    ssd_cfg = {
        'n_classes': 21,
        'img_size': 300,
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 11, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }

    print(SSD(phase='train', **ssd_cfg))