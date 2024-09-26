import torch
import random
import os
import numpy as np
import argparse
import pdb
import torch.distributed as dist

from dataset.dataset import EgoData
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tools.utils.load_cfg import load_config
from tools.modules.EgoChoir import EgoChoir
from tools.utils.logger import Logger
from tools.utils.trainer import train_ddp, train

def main(args):
    if not os.path.exists(args.TRAIN.save_ckpt_path):
        os.makedirs(args.TRAIN.save_ckpt_path)

    if args.USE_GPU and args.train_device == 'ddp':
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        if args.USE_GPU:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    train_dataset = EgoData(args, split='train')
    val_dataset = EgoData(args, split='val')
    model = EgoChoir(args, device)

    model.to(device)
    logger = Logger(os.path.join(args.TRAIN.save_ckpt_path, 'log.txt'), title="eval_matrix")
    logger.set_names(["Epoch", 'AUC', 'aIOU', 'SIM', 'Precision', 'Recall', 'F1', 'geo', 'Acc_1'])
    if args.train_device == 'ddp':
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.HYPER.batch, sampler=train_sampler, num_workers=8, drop_last=True)
        val_sampler = DistributedSampler(val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=args.HYPER.batch, sampler=val_sampler, shuffle=False, num_workers=8)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)
        train_ddp(args, model, train_loader, train_sampler, val_dataset, val_loader, logger, rank, device)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.HYPER.batch, shuffle=True, num_workers=8, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.HYPER.batch, shuffle=False, num_workers=8)
        train(args, model, train_loader, val_dataset, val_loader, logger, device)
    logger.close()

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest="cfg_file", type=str, default='configs/EgoChoir.yaml', help='config path')
    parser.add_argument('--use_gpu', dest="use_gpu", type=bool, default=True, help='set device')
    parser.add_argument('--train_device', dest="train_device", type=str, default='ddp', help='use ddp or single')
    args = parser.parse_args()
    cfg = load_config(args)
    main(cfg)