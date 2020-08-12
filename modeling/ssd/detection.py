"""Detect
Original author: YutaroOgawa
https://github.com/YutaroOgawa/pytorch_advanced/blob/master/2_objectdetection/utils/ssd_model.py
"""

import torch
import torch.nn as nn
from torch.autograd import Function

from modeling.ssd.modules.decode import decode
from modeling.ssd.modules.non_maximum_suppression import nm_suppression

class Detect(nn.Module):
    def __init__(self, variances, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        super(Detect, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.variances = torch.Tensor(variances)

    def forward(self, loc_data, conf_data, dbox_list):

        n_batch = loc_data.size(0)    # batchsize
        n_dbox = loc_data.size(1)     # number of DBox = 8732
        n_classes = conf_data.size(2) # number of classes

        conf_data = self.softmax(conf_data)

        # output tensor.size[mini batch, num classes, 200, 5]
        output = torch.zeros(n_batch, n_classes, self.top_k, 5)
        
        # conf_data [batch_num, 8732, num_classes] to [batch_num, num_classes, 8732]
        conf_preds = conf_data.transpose(2, 1)

        for i in range(n_batch):

            decoded_boxes = decode(loc_data[i], dbox_list, self.variances)

            conf_scores = conf_preds[i].clone()

            for cl in range(1, n_classes):

                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # conf_scores: torch.Size([21, 8732])
                # c_mask     : torch.Size([8732])

                # scores: torch.Size([num bbox with value over threshold])
                scores = conf_scores[cl][c_mask]

                if scores.nelement() == 0:
                    continue
                
                # change c_mask size to adapt 
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask: torch.Size([8732, 4])

                boxes = decoded_boxes[l_mask].view(-1, 4)
                # boxes: torch.Size([num bbox with value over threshold, 4])

                # 3, Non-Maximum Suppression
                ids, count = nm_suppression(
                    boxes, scores, self.nms_thresh, self.top_k)
                # ids: index after nm_suppression
                # count: num of bbox after nm_suppression
                
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes[ids[:count]]), 1)
        
        return output # torch.Size([1, 21, 200, 5])
            




