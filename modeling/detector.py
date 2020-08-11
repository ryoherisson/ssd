from tqdm import tqdm

from logging import getLogger
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision

logger = getLogger(__name__)


class ObjectDetection(object):
    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.network = kwargs['network']
        self.optimizer = kwargs['optimizer']
        self.criterion = kwargs['criterion']
        self.train_loader, self.test_loader = kwargs['data_loaders']
        self.metrics = kwargs['metrics']
        self.box_vis = kwargs['box_vis']
        self.img_size = kwargs['img_size']
        self.writer = kwargs['writer']
        self.save_ckpt_interval = kwargs['save_ckpt_interval']
        self.ckpt_dir = kwargs['ckpt_dir']
        self.img_outdir = kwargs['img_outdir']


    def train(self, n_epochs, start_epoch=0):

        best_test_iou = 0

        for epoch in range(start_epoch, n_epochs):
            logger.info(f'\n\n==================== Epoch: {epoch} ====================')
            logger.info('### train:')
            self.network.train()

            train_loss = 0

            with tqdm(self.train_loader, ncols=100) as pbar:
                for idx, (inputs, targets, heights_, widths_, img_paths_) in enumerate(pbar):
                    inputs = inputs.to(self.device)
                    targets_device = [annot.to(self.device) for annot in targets]

                    outputs = self.network(inputs)

                    loss_l, loss_c = self.criterion(outputs, targets_device)
                    loss = loss_l + loss_c

                    loss.backward()

                    nn.utils.clip_grad_value_(self.network.parameters(), clip_value=2.0)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    train_loss += loss.item()

                    ### metrics update
                    # self.metrics.update(preds=detect_outputs,
                    #                     targets=targets,
                    #                     loss=test_loss,
                    #                     fnames=fnames_)

                    ### logging train loss and accuracy
                    pbar.set_postfix(OrderedDict(
                        epoch="{:>10}".format(epoch),
                        loss="{:.4f}".format(train_loss)))

            if epoch % self.save_ckpt_interval == 0:
                logger.info('\nsaving checkpoint...')
                self._save_ckpt(epoch, train_loss/(idx+1))

            # logger.info('\ncalculate metrics...')
            # self.metrics.calc_metrics(epoch, mode='train')
            # self.metrics.initialize()

            ### test
            logger.info('\n### test:')
            test_mean_iou = self.test(epoch)

            if test_mean_iou > best_test_iou:
                logger.info(f'\nsaving best checkpoint (epoch: {epoch})...')
                best_test_iou = test_mean_iou
                self._save_ckpt(epoch, train_loss/(idx+1), mode='best')


    def test(self, epoch, inference=False):
        self.network.eval()

        test_loss = 0
        height_list = []
        width_list = []
        img_path_list = []

        with torch.no_grad():
            with tqdm(self.test_loader, ncols=100) as pbar:
                for idx, (inputs, targets, heights, widths, img_paths) in enumerate(pbar):

                    inputs = inputs.to(self.device)
                    targets_device = [annot.to(self.device) for annot in targets]

                    outputs, detect_outputs = self.network(inputs, phase='test')

                    loss_l, loss_c = self.criterion(outputs, targets_device)
                    loss = loss_l + loss_c

                    self.optimizer.zero_grad()

                    test_loss += loss.item()

                    height_list.extend(heights)
                    width_list.extend(widths)
                    img_path_list.extend(img_paths)

                    ### metrics update
                    self.metrics.update(preds=detect_outputs,
                                        targets=targets,
                                        loss=test_loss)

                    ### logging test loss and accuracy
                    pbar.set_postfix(OrderedDict(
                        epoch="{:>10}".format(epoch),
                        loss="{:.4f}".format(test_loss)))

            ### metrics
            logger.info('\ncalculate metrics...')
            # preds: [n_imgs, n_classes, top_k, 5]
            # 5: [class_conf, xmin, ymin, xmax, ymax]
            preds = self.metrics.calc_metrics(epoch, mode='test') 
            test_mean_iou = self.metrics.mean_iou

            if inference:
                self._save_images(img_paths, preds, height_list, width_list)

            self.metrics.initialize()

        return test_mean_iou


    def _save_ckpt(self, epoch, loss, mode=None, zfill=4):
        if isinstance(self.network, nn.DataParallel):
            network = self.network.module
        else:
            network = self.network

        if mode == 'best':
            ckpt_path = self.ckpt_dir / 'best_iou_ckpt.pth'
        else:
            ckpt_path = self.ckpt_dir / f'epoch{str(epoch).zfill(zfill)}_ckpt.pth'

        torch.save({
            'epoch': epoch,
            'network': network,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, ckpt_path)

    def _show_imgs(self, img_paths, preds, img_size, prefix='train'):
        """Show result image on Tensorboard

        Parameters
        ----------
        img_paths : list
            original image path
        preds : tensor
            [1, 21, 200, 5] ([mini-batch, n_classes, [class_conf, xmin, ymin, xmax, ymax]])
        img_size : int
            show img size
        prefix : str, optional
            'train' or 'test', by default 'train'
        """
        


        pass

    def _save_images(self, img_paths, preds, height_list, width_list):
        """Save Image

        Parameters
        ----------
        img_paths : list
            original image paths
        preds : tensor
            [1, 21, 200, 5] ([mini-batch, n_classes, [class_conf, xmin, ymin, xmax, ymax]])
        height_list : list
            original height list
        width_list : list
            original width list
        """

        for i, img_path in enumerate(img_paths):
            # preds[i] has background label 0, so exclude background class
            pred = preds[i][1:]
            height = height_list[i]
            width = width_list[i]

            annotated_img = self.box_vis.draw_box(img_path, pred, height, width)

            outpath = self.img_outdir / img_path.name
            self.box_vis.save_img(annotated_img, outpath)