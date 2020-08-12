"""Metrics class
calculate iou and mean iou, save results images
"""

import csv
from logging import getLogger

import pandas as pd

import torch

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
        self.confidence_level = metric_cfg['confidence_level']

        # initialize output list, intersection, union, iou and mean_iou
        self.initialize()

        self.loss = 0

    def initialize(self):
        # detect output list
        self.preds_list = []
        # target list
        self.targets = []

        # intersection for iou : torch.Size([n_classes])
        self.intersection = torch.zeros(self.n_classes-1, dtype=torch.long)
        # union for iou : torch.Size([n_classes])
        self.union = torch.zeros(self.n_classes-1, dtype=torch.long)
        # iou
        self.ious = 0
        # mean iou
        self.mean_iou = 0

    def update(self, preds, targets, loss):
        # preds: [1, 21, 200, 5] ([mini-batch, n_classes, [class_conf, xmin, ymin, xmax, ymax]])
        # targets: [mini-batch, n_elem, [xmin, ymin, xmax, ymax, class_index]]
        # loss: loss
        # fnames: filenames

        self.preds_list.append(preds)
        self.loss = loss
        self.targets.extend(targets)

    def calc_metrics(self, epoch, mode='train'):

        preds = torch.cat([o for o in self.preds_list], 0)
        
        # input data size
        num_inputs = preds.size(0)
    
        for idx in range(num_inputs):

            # self.n_classes-1: exclude background
            for label in range(0, self.n_classes-1):

                # compare target_tensor and output_tensor
                target_tensor = torch.zeros(self.img_size, self.img_size, dtype=torch.long) + self.n_classes
                pred_tensor = torch.zeros(self.img_size, self.img_size, dtype=torch.long) + self.n_classes

                # targets indices same label
                target_indices = (self.targets[idx][:, -1] == label).nonzero().squeeze(1)

                for t_i in target_indices:
                    xmin = (self.targets[idx][t_i][0] * self.img_size).long()
                    ymin = (self.targets[idx][t_i][1] * self.img_size).long()
                    xmax = (self.targets[idx][t_i][2] * self.img_size).long()
                    ymax = (self.targets[idx][t_i][3] * self.img_size).long()

                    target_tensor[ymin:ymax, xmin:xmax] = label

                # preds
                # background label is 0, so p_label = label + 1
                p_label = label + 1
                find_index = (preds[idx][p_label][:, 0] >= self.confidence_level).nonzero().squeeze(1)

                for k in find_index:
                    xmin = (preds[idx][p_label][k][1] * self.img_size).long()
                    ymin = (preds[idx][p_label][k][2] * self.img_size).long()
                    xmax = (preds[idx][p_label][k][3] * self.img_size).long()
                    ymax = (preds[idx][p_label][k][4] * self.img_size).long()

                    pred_tensor[ymin:ymax, xmin:xmax] = p_label

                self.intersection[label] += torch.sum((target_tensor == label) & (pred_tensor == p_label)).long()
                self.union[label] += torch.sum((target_tensor == label) | (pred_tensor == p_label)).long()

        self.calc_iou()
        self.logging(epoch, mode)
        self.save_csv(epoch, mode)

        return preds

    def calc_iou(self):
        # If union is 0, set nan to ignore them
        zero_idx= (self.union == 0).nonzero()
        self.union = self.union.float()
        self.union[zero_idx] = float('nan')

        self.intersection = self.intersection.float()

        # calc ious
        self.ious = self.intersection / self.union
        self.mean_iou = self.ious[~torch.isnan(self.ious)].mean()

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
            logwriter.writerow([epoch, self.loss, self.mean_iou.item()])