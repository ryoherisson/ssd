"""Non-Maximum Supression
Original author: YutaroOgawa
https://github.com/YutaroOgawa/pytorch_advanced/blob/master/2_objectdetection/utils/ssd_model.py
"""

def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """Non-Maximum Supression
    delete boxes with overlap larger than threshold

    Parameters
    ----------
    boxes : [number of BBox with conf over 0.01, 4]
        bbox info
    scores : [number of BBox with conf over 0.01]
        conf info
    overlap : float, optional
        overlap threshold, by default 0.45
    top_k : int, optional
        number of bbox to search, by default 200

    Returns
    -------
    keep : list
        stores the index that passed the nms in descending order of conf
    count : int
        number of BBoxes passing through nms
    """
    
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    # keep: torch.Size([number of bbox over conf threshold])

    # calc area of bbox
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    # copy boxes
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    # reorder scores in ascending order
    v, idx = scores.sort(0)

    # extract the index of the top_k (200) BBoxes (it can be less than 200)
    idx = idx[-top_k:]

    while idx.numel() > 0:
        i = idx[-1] # index with maximum value of conf

        keep[count] = i
        count += 1

        if idx.size(0) == 1:
            break

        # delete the index which is inserted in keep
        idx = idx[:-1]

        # -------------------
        # extract and remove BBoxes that have a large overlap 
        # with the BBoxes stored in keep.
        # -------------------
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        # for all BBoxes, set current BBox=index to the value covered by i (clamp)
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        # reduce the w and h tensor size by one index
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        # find the width and height of the BBox
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        # delete the coords with negative value
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # calculate area
        inter = tmp_w * tmp_h

        # IoU = intersect / (area(a) + area(b) - intersect)
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union

        # leave only the idx smaller than the overlap
        idx = idx[IoU.le(overlap)]

    return keep, count

