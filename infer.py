import torch
import argparse

from tools.utils.load_cfg import load_config
from torch.utils.data import DataLoader
from tools.modules.EgoChoir import EgoChoir
from dataset.dataset import EgoData, EgoData_infer
from tools.utils.trainer import inference_sample, inference_whole

def infer(args):
    if args.USE_GPU:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    #val_dataset = EgoData(args, split='infer')
    val_dataset = EgoData_infer(args, split='infer')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=val_dataset.collate_fn)
    ckpt = 'runs/ckpt/EgoChoir_824/AUC_best.pt'
    param = torch.load(ckpt, map_location='cpu')
    model = EgoChoir(args, device)
    model.load_state_dict(param)
    model.to(device)
    model.eval()
    #inference_sample(args, model, val_loader, device)
    inference_whole(args, model, val_loader, device)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest="cfg_file", type=str, default='configs/EgoChoir.yaml', help='config path')
    parser.add_argument('--use_gpu', dest="use_gpu", type=bool, default=True, help='set device')
    args = parser.parse_args()
    cfg = load_config(args)
    infer(cfg)