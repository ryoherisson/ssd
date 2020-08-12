"""SSD
Original author: YutaroOgawa
https://github.com/YutaroOgawa/pytorch_advanced/blob/master/2_objectdetection/utils/ssd_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.criterions.functions.match import match

class MultiBoxLoss(nn.Module):
    """Loss for SSD"""

    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh  # 0.5 jaccard's coef threshold
        self.negpos_ratio = neg_pos  # 3:1 ratio of Hard Negative Mining neg:pos
        self.device = device

    def forward(self, predictions, targets):
        """
        Loss for SSD
        Parameters
        ----------
        predictions : SSD net output(tuple)
            (loc=torch.Size([n_batch, 8732, 4]), conf=torch.Size([n_batch, 8732, 21]), dbox_list=torch.Size [8732,4])。
        targets : [n_batch, num_objs, 5]
            5: [xmin, ymin, xmax, ymax, label_ind]
        Returns
        -------
        loss_l : torch.tensor
            loc's loss
        loss_c : torch.tensor
            conf's loss
        """

        loc_data, conf_data, dbox_list = predictions

        # elem num
        n_batch = loc_data.size(0)  # mini-batchsize
        n_dbox = loc_data.size(1)  # num dbox
        n_classes = conf_data.size(2)  # num classes

        # ceate a variable to store what is used to calculate the loss
        # conf_t_label： store the label of the nearest correct BBox in each DBox
        # loc_t: store the location information of the nearest correct BBox to each DBox
        conf_t_label = torch.LongTensor(n_batch, n_dbox).to(self.device)
        loc_t = torch.Tensor(n_batch, n_dbox, 4).to(self.device)


        for idx in range(n_batch):

            # BBox
            truths = targets[idx][:, :-1].to(self.device)  # BBox
            # labels [label1, label2, …]
            labels = targets[idx][:, -1].to(self.device)

            # default box
            dbox = dbox_list.to(self.device)

            # update the contents of loc_t and conf_t_label
            variance = [0.1, 0.2]
            # coefficients in the formula used to calculate the correction from DBox to BBox
            match(self.jaccard_thresh, truths, dbox,
                  variance, labels, loc_t, conf_t_label, idx)

        # ----------
        # Loc Loss with smooth l1 loss
        # ----------
        # create mask
        pos_mask = conf_t_label > 0  # torch.Size([n_batch, 8732])

        # pos_mask size => loc_data size
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        # loc of Positive DBox and target
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        # loc loss
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # ----------
        # Class Loss with Cross-Entropy
        # ----------
        batch_conf = conf_data.view(-1, n_classes)

        loss_c = F.cross_entropy(
            batch_conf, conf_t_label.view(-1), reduction='none')

        # -----------------
        # Hard Negative Mining
        # -----------------

        num_pos = pos_mask.long().sum(1, keepdim=True)
        loss_c = loss_c.view(n_batch, -1)  # torch.Size([n_batch, 8732])
        loss_c[pos_mask] = 0 

        # Hard Negative Mining
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_neg = torch.clamp(num_pos*self.negpos_ratio, max=n_dbox)

        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        conf_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)
                             ].view(-1, n_classes)

        conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]

        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')

        N = num_pos.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c