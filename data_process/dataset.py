"""Dataset process
Original author: YutaroOgawa
https://github.com/YutaroOgawa/pytorch_advanced/blob/master/2_objectdetection/utils/ssd_model.py
"""
from pathlib import Path
import cv2
import numpy as np
import xml.etree.ElementTree as ET

import torch
import torch.utils.data as data

from data_process.utils.data_augmentation import Compose, ConvertFromInts, ToAbsoluteCoords, \
                                                 PhotometricDistort, Expand, RandomSampleCrop,\
                                                 RandomMirror, ToPercentCoords, Resize, SubtractMeans


class Anno_xml2list(object):
    def __init__(self, classes):
        self.classes = classes
    
    def __call__(self, xml_path, width, height):
        """Annotation transform

        Parameters
        ----------
        xml_path : str
            path to xml file
        width : int
            image width
        height : int
            image height

        Returns
        -------
        ret : 
            [[xmin, ymin, xmax, ymax, label_ind], ... ]
        """

        ret = []

        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):

            # exclude difficult
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            bndbox = []

            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
                # for voc, the origin is (1, 1), so need to subtract 1
                cur_pixel = int(bbox.find(pt).text) - 1

                # normalize with width and height
                if pt == 'xmin' or pt == 'xmax':
                    cur_pixel /= width
                else:
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            # add [xmin, ymin, xmax, ymax, label_ind] to ret
            ret += [bndbox]

        return np.array(ret) # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class DataTransform(object):
    """Preprocessing Class

    Attributes
    ----------
    img_size : int
        resized img size
    color_mean : (R, G, B)
        average values of each channels
    mode : str
        'train' or 'test'
    """

    def __init__(self, img_size, color_mean, mode):
        if mode == 'train':
            self.data_transform = Compose([
                    ConvertFromInts(),
                    ToAbsoluteCoords(),
                    PhotometricDistort(),
                    Expand(color_mean),
                    RandomSampleCrop(),
                    RandomMirror(),
                    ToPercentCoords(),
                    Resize(img_size),
                    SubtractMeans(color_mean),
            ])
        elif mode == 'test':
            self.data_transform = Compose([
                    ConvertFromInts(),
                    Resize(img_size),
                    SubtractMeans(color_mean),
            ])
        else:
            raise ValueError(f'mode should be train or test. now mode is {mode}')

    def __call__(self, img, boxes, labels):
        return self.data_transform(img, boxes, labels)


class VOCDataset(data.Dataset):
    """VOC Dataset class

    Attributes
    ----------
    img_list : list
        image file path list
    annot_list : list
        annotation xml file path list
    transform : object
        preprocess instance
    transform_annot : object
        annotation xml file converted to annotation list
    """

    def __init__(self, img_list, annot_list, transform, transform_annot):
        self.img_list = img_list
        self.annot_list = annot_list
        self.transform = transform
        self.transform_annot = transform_annot

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        im, gt, h, w, img_path = self.pull_item(index)
        return im, gt, h, w, img_path

    def pull_item(self, index):
        # read img
        img_path = self.img_list[index]
        img = cv2.imread(img_path) # [height][width][BGR]
        height, width, channels = img.shape

        # create a list of annot info
        annot_path = self.annot_list[index]
        annot_list = self.transform_annot(annot_path, width, height)

        # preprocess
        img, boxes, labels = self.transform(
            img, annot_list[:, :4], annot_list[:, 4]
        )

        # change color channel from BGR to RGB
        # change array from (height, width, color) to (color, height, width)
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # np.array of BBoxes and label
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width, Path(img_path)


def od_collate_fn(batch):

    targets = []
    imgs = []
    heights = []
    widths = []
    fnames = []

    for sample in batch:
        imgs.append(sample[0]) # sample[0] is img
        targets.append(torch.FloatTensor(sample[1])) # sample[1] is annot gt
        heights.append(sample[2])
        widths.append(sample[3])
        fnames.append(sample[4])

    # imgs is mini-batch list
    # each element has torch.Size([3, 300, 300])
    # convert this list to torch.Tensor with torch.Size([batch_num, 3, 300, 300])
    imgs = torch.stack(imgs, dim=0)

    # targets is list of gt
    # list size is mini-batch size
    # each element has ([n, 5])
    # n is a number of object in the img
    # 5: [xmin, ymin, xmax, ymax, class_index]

    return imgs, targets, heights, widths, fnames