"""Metrics class
calculate iou and mean iou, save results images
"""

import csv
from logging import getLogger

import pandas as pd

import torch

from modeling.metrics.detect import Detect

logger = getLogger(__name__)
pd.set_option('display.unicode.east_asian_width', True)

class Metrics(object):
    def __init__(self, **metric_cfg):
        self.n_classes = metric_cfg['n_classes']
        self.classes = metric_cfg['classes']
        self.img_size = metric_cfg['img_size']
        self.writer = metric_cfg['writer']
        self.metrics_dir = metric_cfg['metrics_dir']
        self.imgs_dir = metric_cfg['imgs_dir']

        # initialize output list, intersection, union, iou and mean_iou
        self.initialize()

        self.loss = 0

        self.conf_thresh = metric_cfg['conf_thresh']
        self.top_k = metric_cfg['top_k']
        self.nms_thresh = metric_cfg['nms_thresh']
        self.detect = Detect(conf_thresh=self.conf_thresh, top_k=self.top_k, nms_thresh=self.nms_thresh)

    def initialize(self):
        self.output_list = []

        # intersection for iou : torch.Size([n_classes])
        self.intersection = torch.zeros(self.n_classes, dtype=torch.long)
        # union for iou : torch.Size([n_classes])
        self.union = torch.zeros(self.n_classes, dtype=torch.long)
        # iou
        self.ious = 0
        # mean iou
        self.mean_iou = 0

    def update(self, loc, conf, dbox_list, targets, loss, fnames):
        # preds: (loc, conf, dbox_list)
        # targets: [mini-batch, n_elem, [xmin, ymin, xmax, ymax, class_index]]
        # loss: loss
        # fnames: filenames

        # Detect forward
        # outputs: torch.Size([batch_num, 21, 200, 5])
        outputs = self.detect(loc, conf, dbox_list)
        self.output_list.append(outputs)

        # mini-batch size
        n_batch = outputs.size(0) 
    
        for idx in range(n_batch):

            for label in range(0, self.n_classes):

                # compare target_tensor and output_tensor
                target_tensor = torch.zeros(self.img_size, self.img_size, dtype=torch.long)
                output_tensor = torch.zeros(self.img_size, self.img_size, dtype=torch.long)

                # targets indices same label
                target_indices = (targets[idx][:, -1] == label).nonzero().squeeze(1)

                for t_i in target_indices:
                    xmin = (targets[idx][t_i][0] * self.img_size).long()
                    ymin = (targets[idx][t_i][1] * self.img_size).long()
                    xmax = (targets[idx][t_i][2] * self.img_size).long()
                    ymax = (targets[idx][t_i][3] * self.img_size).long()

                    target_tensor[ymin:ymax, xmin:xmax] = label

                # outputs
                for k in range(self.top_k):
                    xmin = (outputs[idx][label][k][0] * self.img_size).long()
                    ymin = (outputs[idx][label][k][0] * self.img_size).long()
                    xmax = (outputs[idx][label][k][0] * self.img_size).long()
                    ymax = (outputs[idx][label][k][0] * self.img_size).long()

                    output_tensor[ymin:ymax, xmin:xmax] = label

                self.intersection[label] += torch.sum((target_tensor == label) & (output_tensor == label)).long()
                self.union[label] += torch.sum((target_tensor == label) | (output_tensor == label)).long()

    def calc_metrics(self, epoch, mode='train'):
        self.iou()
        self.logging(epoch, mode)
        self.save_csv(epoch, mode)

    def iou(self):
        # to avoid inf when union is 0

        # self.union += 1e-9
        self.union = self.union.float() + 1e-9
        self.intersection = self.intersection.float()

        # exclude background
        self.union = self.union[1:]
        self.intersection = self.intersection[1:]

        # calc ious
        self.ious = self.intersection / self.union
        self.mean_iou = self.ious.mean()

    def logging(self, epoch, mode):
        logger.info(f'{mode} metrics...')
        logger.info(f'loss:         {self.loss}')

        # ious per class
        df = pd.DataFrame(index=self.classes)
        df['IoU'] = self.ious.tolist()
        logger.info(f'\nmetrics value per classes: \n{df}\n')

        # micro mean iou
        logger.info(f'mean iou:    {self.mean_iou}')

        # Change mode from 'test' to 'val' to change the display order from left to right to train and test.
        mode = 'val' if mode == 'test' else mode

        self.writer.add_scalar(f'loss/{mode}', self.loss, epoch)
        self.writer.add_scalar(f'mean_iou/{mode}', self.mean_iou, epoch)

    def save_csv(self, epoch, mode):
        csv_path = self.metrics_dir / f'{mode}_metrics.csv'

        if not csv_path.exists():
            with open(csv_path, 'w') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow(['epoch', f'{mode} loss', f'{mode} iou'])

        with open(csv_path, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, self.loss, self.mean_iou])