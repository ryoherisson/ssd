import argparse
import yaml
from pathlib import Path
from datetime import datetime

from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from torchsummary import summary

from utils.path_process import Paths
from utils.setup_logger import setup_logger
from utils.vis_img import BoxVis
from data_process.data_path_process import make_datapath_list
from data_process.dataset import Anno_xml2list, DataTransform
from data_process.dataset import VOCDataset
from data_process.dataset import od_collate_fn
from modeling.ssd.ssd import SSD
from modeling.criterions.loss import MultiBoxLoss
from modeling.metrics.metrics import Metrics
from modeling.detector import ObjectDetection

logger = getLogger(__name__)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configfile', type=str, default='./configs/default.yml')
    parser.add_argument('--inference', action='store_true', default=False)
    args = parser.parse_args()
    return args

def main():
    args = parser()

    with open(args.configfile) as f:
        configs = yaml.safe_load(f)

    ## path process (path definition, make directories)
    now = datetime.now().isoformat()
    log_dir = Path(configs['log_dir']) / now
    paths = Paths(log_dir=log_dir)

    ### setup logs and summary writer ###
    setup_logger(logfile=paths.logfile)

    writer = SummaryWriter(str(paths.summary_dir))

    ### setup GPU or CPU ###
    if configs['n_gpus'] > 0 and torch.cuda.is_available():
        logger.info('CUDA is available! using GPU...\n')
        device = torch.device('cuda')
    else:
        logger.info('using CPU...\n')
        device = torch.device('cpu')
    
    ### Dataset ###
    logger.info('preparing dataset...')
    data_root = configs['data_root']
    logger.info(f'==> dataset path: {data_root}\n')

    train_img_list, train_annot_list, test_img_list, test_annot_list = make_datapath_list(rootpath=data_root, train_data=configs['train_txt'], test_data=configs['test_txt'])

    train_transform = DataTransform(img_size=configs['img_size'], color_mean=configs['color_mean'], mode='train')
    test_transform = DataTransform(img_size=configs['img_size'], color_mean=configs['color_mean'], mode='test')
    transform_annot = Anno_xml2list(configs['classes'])

    train_dataset = VOCDataset(train_img_list, train_annot_list, transform=train_transform, transform_annot=transform_annot)
    test_dataset = VOCDataset(test_img_list, test_annot_list, transform=test_transform, transform_annot=transform_annot)

    ### DataLoader ###
    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, collate_fn=od_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=configs['batch_size'], shuffle=True, collate_fn=od_collate_fn)

    ### Network ###
    logger.info('preparing network...')

    ssd_cfg = {
        'n_classes': configs['n_classes'],
        'img_size': configs['img_size'],
        'bbox_aspect_num': configs['bbox_aspect_num'],  # number of aspect ratios of dbox
        'feature_maps': configs['feature_maps'],  # feature map size of each source
        'steps': configs['steps'],  # size of dbox
        'min_sizes': configs['min_sizes'],  # size of dbox
        'max_sizes': configs['max_sizes'],  # size of dbox
        'aspect_ratios': configs['aspect_ratios'],  # aspect ratios
        'variances': configs['variances'], # variances for decode
        'conf_thresh': configs['conf_thresh'],
        'top_k': configs['top_k'],
        'nms_thresh': configs['nms_thresh'],
        'device': device,
    }

    network = SSD(**ssd_cfg)
    network = network.to(device)
    criterion = MultiBoxLoss(jaccard_thresh=configs['jaccord_thresh'], neg_pos=configs['neg_pos'], device=device)
    optimizer = optim.Adam(network.parameters(), lr=configs['lr'])

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    if configs['pretrained']:
        # Load pretrained model
        logger.info('==> Pretrained VGG...\n')
        if not Path(configs['pretrained']).exists():
            logger.info('No pretrained model found!')
        
        vgg_weights = torch.load(configs['pretrained'])
        network.vgg.load_state_dict(vgg_weights)

    network.extras.apply(weights_init)
    network.loc.apply(weights_init)
    network.conf.apply(weights_init)

    if configs['resume']:
        # Load checkpoint
        logger.info('==> Resuming from checkpoint...\n')
        if not Path(configs['resume']).exists():
            logger.info('No checkpoint found !')
            raise ValueError('No checkpoint found !')

        ckpt = torch.load(configs['resume'])
        network.load_state_dict(ckpt)
        start_epoch = 0
        # network.load_state_dict(ckpt['model_state_dict'])
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # start_epoch = ckpt['epoch']
        # loss = ckpt['loss']
    else:
        logger.info('==> Building model...\n')
        start_epoch = 0

    # logging
    logger.info('model summary: ')
    logger.info(summary(network, input_size=(configs['n_channels'], configs['img_size'], configs['img_size'])))

    if configs["n_gpus"] > 1:
        network = nn.DataParallel(network)

    ### Metrics ###
    metrics_cfg = {
        'n_classes': configs['n_classes'],
        'classes': configs['classes'],
        'img_size': configs['img_size'],
        'confidence_level': configs['confidence_level'],
        'writer': writer,
        'metrics_dir': paths.metrics_dir,
        'imgs_dir': paths.img_outdir,
    }
    
    metrics = Metrics(**metrics_cfg)

    ### Visualize Results ###
    box_vis = BoxVis(configs['confidence_level'], configs['classes'], configs['label_color_map'], configs['font_path'])

    ### Train or Inference ###
    kwargs = {
        'device': device,
        'network': network,
        'optimizer': optimizer,
        'criterion': criterion,
        'data_loaders': (train_loader, test_loader),
        'metrics': metrics,
        'box_vis': box_vis,
        'img_size': configs['img_size'],
        'writer': writer,
        'save_ckpt_interval': configs['save_ckpt_interval'],
        'ckpt_dir': paths.ckpt_dir,
        'img_outdir': paths.img_outdir,
    }

    object_detection = ObjectDetection(**kwargs)

    if args.inference:
        if not configs['resume']:
            logger.info('No checkpoint found for inference!')
        logger.info('mode: inference\n')
        object_detection.test(epoch=start_epoch, inference=True)
    else:
        logger.info('mode: train\n')
        object_detection.train(n_epochs=configs['n_epochs'], start_epoch=start_epoch)

if __name__ == "__main__":
    main()