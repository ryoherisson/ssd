"""Loc and Conf
Original author: YutaroOgawa
https://github.com/YutaroOgawa/pytorch_advanced/blob/master/2_objectdetection/utils/ssd_model.py
"""

import torch.nn as nn

def make_loc_conf(n_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):
    
    loc_layers = []
    conf_layers = []

    # conv of vgg 22nd (source1)
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0]*n_classes, kernel_size=3, padding=1)]

    # conv of vgg last (source2)
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1]*n_classes, kernel_size=3, padding=1)]

    # conv of extras (source3)
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2]*n_classes, kernel_size=3, padding=1)]

    # conv of extras (source4)
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3]*n_classes, kernel_size=3, padding=1)]

    # conv of extras (source5)
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4]*n_classes, kernel_size=3, padding=1)]

    # conv of extras (source6)
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5]*4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5]*n_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

if __name__ == "__main__":
    loc_test, conf_test = make_loc_conf()
    print(loc_test)
    print(conf_test)