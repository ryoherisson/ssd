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
            n_total = 0

            with tqdm(self.train_loader, ncols=100) as pbar:
                for idx, (inputs, targets, heights_, widths_, fnames_) in enumerate(pbar):
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

                    n_total += len(targets)

                    ### metrics update
                    # self.metrics.update(loc=outputs[0].cpu().detach().clone(),
                    #                     conf=outputs[1].cpu().detach().clone(),
                    #                     dbox_list=outputs[2].cpu().detach().clone(),
                    #                     targets=targets,
                    #                     loss=train_loss / n_total,
                    #                     fnames=fnames_)

                    ### logging train loss and accuracy
                    pbar.set_postfix(OrderedDict(
                        epoch="{:>10}".format(epoch),
                        loss="{:.4f}".format(train_loss / n_total)))

            if epoch % self.save_ckpt_interval == 0:
                logger.info('saving checkpoing...')
                self._save_ckpt(epoch, train_loss/(idx+1))

            # self.metrics.calc_metrics(epoch, mode='train')
            # self.metrics.initialize()

            ### test
            logger.info('### test:')
            self.test(epoch)

    def test(self, epoch, inference=False):
        self.network.eval()

        test_loss = 0
        n_total = 0

        with torch.no_grad():
            with tqdm(self.test_loader, ncols=100) as pbar:
                for idx, (inputs, targets, heights_, widths_, fnames_) in enumerate(pbar):

                    inputs = inputs.to(self.device)
                    targets_device = [annot.to(self.device) for annot in targets]

                    outputs = self.network(inputs)

                    loss_l, loss_c = self.criterion(outputs, targets_device)
                    loss = loss_l + loss_c

                    self.optimizer.zero_grad()

                    test_loss += loss.item()

                    n_total += len(targets)

                    ### metrics update
                    self.metrics.update(loc=outputs[0].cpu().detach().clone(),
                                        conf=outputs[1].cpu().detach().clone(),
                                        dbox_list=outputs[2].cpu().detach().clone(),
                                        targets=targets,
                                        loss=test_loss / n_total,
                                        fnames=fnames_)

                    ### logging test loss and accuracy
                    pbar.set_postfix(OrderedDict(
                        epoch="{:>10}".format(epoch),
                        loss="{:.4f}".format(test_loss / n_total)))

            self.metrics.calc_metrics(epoch, mode='test')
            self.metrics.initialize()


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