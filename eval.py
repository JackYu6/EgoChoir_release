import torch
import random
import os
import numpy as np
import argparse
import pdb

from dataset.dataset import EgoData
from torch.utils.data import DataLoader
from tools.utils.load_cfg import load_config
from tools.modules.EgoChoir import EgoChoir
from tools.utils.evaluation import evaluate, evaluate_per_class

def run_eval(args):
    if args.USE_GPU:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    val_dataset = EgoData(args, 'infer')
    val_loader = DataLoader(val_dataset, batch_size=args.HYPER.batch, shuffle=False, num_workers=8, collate_fn=val_dataset.collate_fn)
    ckpt = args.eval_ckpt_path
    param = torch.load(ckpt, map_location='cpu')

    model = EgoChoir(args, device)
    model.load_state_dict(param)
    model.to(device)
    model.eval()
    pr_aff, gt_aff = [], []
    pr_contact, gt_contact = [], []
    pr_logits, gt_logits = [], []
    Interactions = []
    with torch.no_grad():
        for i, data_info in enumerate(val_loader):
            V = data_info['frames'].to(device)
            M = data_info['camera_pose'].float().to(device)
            O = data_info['pts'].float().to(device)
            B = O.size(0)
            contact_gt = data_info['contact_label'].to(device)
            affordance_gt = data_info['affordance_label'].float().to(device)
            logits_labels = data_info['aff_logits'].to(device)
            obj_path = data_info['obj_path']
            for iter in range(B):
                Interactions.append(obj_path[iter].split('/')[4])

            pre_contact, pre_affordance, pre_logits = model(V, M, O)

            pr_aff.extend(pre_affordance)
            gt_aff.extend(affordance_gt)
            pr_contact.extend(pre_contact)
            gt_contact.extend(contact_gt)
            pr_logits.extend(pre_logits)
            gt_logits.extend(logits_labels)
        aff_preds, aff_targets = torch.stack(pr_aff, 0), torch.stack(gt_aff, 0)
        contact_preds, contact_targets = torch.stack(pr_contact, 0), torch.stack(gt_contact, 0)
        logits_preds, logits_gt = torch.stack(pr_logits, 0), torch.stack(gt_logits, 0)
        #AUC_, IOU_, SIM_, precision_, recall_, F1_, geo, top1_acc = evaluate_per_class(contact_preds, contact_targets, aff_preds, aff_targets, logits_preds, logits_gt, Interactions)
        AUC_, IOU_, SIM_, precision_, recall_, F1_, geo, top1_acc = evaluate(contact_preds, contact_targets, aff_preds, aff_targets, logits_preds, logits_gt)
        print("Overall---------")
        print(f'AUC:{np.around(AUC_,2)} | aIOU:{np.around(IOU_*100,2)} | SIM:{np.around(SIM_,3)} | Precision:{np.around(precision_,2)} | Recall:{np.around(recall_,2)} | F1:{np.around(F1_,2)} | geo:{np.around(geo*100,2)} | Acc_1:{np.around(top1_acc,2)}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest="cfg_file", type=str, default='configs/EgoChoir.yaml', help='config path')
    parser.add_argument('--use_gpu', dest="use_gpu", type=bool, default=True, help='set device')
    parser.add_argument('--train_device', dest="train_device", type=str, default='ddp', help='use ddp or single')
    args = parser.parse_args()
    cfg = load_config(args)
    run_eval(cfg)