import torch
import torch.nn as nn
import logging
import os
import pdb
import torch.nn.functional as F
import torch.distributed as dist

from .loss import L_ca
from. evaluation import evaluate
from .visual import visual_affordance, visual_contact, visual_affordance_seq

def train_ddp(args, model, train_loader, train_sampler, val_dataset, val_loader, logger, rank, device):

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.TRAIN.save_ckpt_path, "train.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def log_string(str):
        screen_logger.info(str)
        print(str)

    batches_train, batches_val = len(train_loader), len(val_loader)

    loss_ca = L_ca().to(device)
    loss_ce = nn.CrossEntropyLoss().to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(pg, lr=args.HYPER.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.HYPER.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.HYPER.epoch, eta_min=1e-6)

    best_current = {'AUC':0, 'aIOU':0, 'SIM':0, 'precision':0, 'Recall':0, 'f1':0, 'geo':100000, 'Acc_1':0}
    if rank ==0 :
        log_string(f'train_batch:{batches_train} | val_batch:{batches_val}')
        print(f"start training...")

    for epoch in range(args.HYPER.epoch):
        if rank ==0 :
            log_string(f'{epoch + 1} Start -------')
        train_sampler.set_epoch(epoch)
        model = model.train()
        loss_sum = 0
        for i, data_info in enumerate(train_loader):
            optimizer.zero_grad()
            V = data_info['frames'].to(device)
            M = data_info['camera_pose'].float().to(device)
            O = data_info['pts'].float().to(device)

            contact_gt = data_info['contact_label'].to(device)
            affordance_gt = data_info['affordance_label'].float().to(device)
            logits_labels = data_info['aff_logits'].to(device)

            pre_contact, pre_affordance, pre_logits = model(V, M, O)

            loss_c = loss_ca(pre_contact.view(-1, 6890, 1), contact_gt.view(-1, 6890, 1))
            loss_a = loss_ca(pre_affordance, affordance_gt)
            loss_s = loss_ce(pre_logits, logits_labels)
            loss = 10*(loss_c + loss_a + loss_s)

            if rank ==0 :
                print(f'Iteration {i}/{batches_train} | loss: {loss.item()}')
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
        avg_loss = loss_sum / (batches_train)
        if rank ==0 :
            log_string(f'Epoch: {epoch + 1} : loss: {avg_loss}')
        if(epoch % 2 ==0):
            model = model.eval()
            val_loss, best_results = val(epoch, args, model, best_current, val_dataset, val_loader, batches_val, loss_ca, loss_ce, logger, device)
            if rank ==0 :
                log_string(f'Epoch: {epoch + 1} : val_loss: {val_loss}')
                if 'AUC' in best_results:
                    best_current['AUC'], best_current['aIOU'], best_current['SIM'] = best_results['AUC'], best_results['aIOU'], best_results['SIM']
                    best_model_path = os.path.join(args.TRAIN.save_ckpt_path, 'AUC_best.pt')
                    torch.save(model.module.state_dict(), best_model_path)
                    log_string(f'AUC best saved in {best_model_path}')
                if 'f1' in best_results:
                    best_current['f1'] = best_results['f1']
                    best_current['precision'] = best_results['precision']
                    best_current['Recall'] = best_results['Recall']
                    best_current['geo'] = best_results['geo']
                    best_model_path = os.path.join(args.TRAIN.save_ckpt_path, 'F1_best.pt')
                    torch.save(model.module.state_dict(), best_model_path)
                    log_string(f'F1 best saved in {best_model_path}')
                if 'Acc_1' in best_results:
                    best_current['Acc_1'] = best_results['Acc_1']
        scheduler.step()
    if rank ==0 :
        log_string(f'Best Results----AUC:{best_current["AUC"]} | aIOU:{best_current["aIOU"]} | SIM:{best_current["SIM"]} | \
                Precision:{best_current["precision"]} | Recall:{best_current["Recall"]} | F1:{best_current["f1"]} | geo:{best_current["geo"]} \
                Acc_1:{best_current["Acc_1"]}')

def train(args, model, train_loader, val_dataset, val_loader, logger, device):

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.TRAIN.save_ckpt_path, "train.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)
    def log_string(str):
        screen_logger.info(str)
        print(str)

    batches_train, batches_val = len(train_loader), len(val_loader)

    loss_ca = L_ca().to(device)
    loss_ce = nn.CrossEntropyLoss().to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(pg, lr=args.HYPER.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.HYPER.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.HYPER.epoch, eta_min=1e-6)

    best_current = {'AUC':0, 'aIOU':0, 'SIM':0, 'precision':0, 'Recall':0, 'f1':0, 'geo':100000, 'Acc_1':0}

    log_string(f'train_batch:{batches_train} | val_batch:{batches_val}')
    print(f"start training...")

    for epoch in range(args.HYPER.epoch):
        log_string(f'{epoch + 1} Start -------')
        model = model.train()
        loss_sum = 0
        for i, data_info in enumerate(train_loader):
            optimizer.zero_grad()
            V = data_info['frames'].to(device)
            M = data_info['camera_pose'].float().to(device)
            O = data_info['pts'].float().to(device)

            contact_gt = data_info['contact_label'].to(device)
            affordance_gt = data_info['affordance_label'].float().to(device)
            logits_labels = data_info['aff_logits'].to(device)

            pre_contact, pre_affordance, pre_logits = model(V, M, O)

            loss_c = loss_ca(pre_contact.view(-1, 6890, 1), contact_gt.view(-1, 6890, 1))
            loss_a = loss_ca(pre_affordance, affordance_gt)
            loss_s = loss_ce(pre_logits, logits_labels)
            loss = 15*(loss_c + loss_a + loss_s)
            print(f'Iteration {i}/{batches_train} | loss: {loss.item()}')
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()
        avg_loss = loss_sum / (batches_train)

        log_string(f'Epoch: {epoch + 1} : loss: {avg_loss}')
        if(epoch % 1 ==0):
            model = model.eval()
            val_loss, best_results = val(epoch, args, model, best_current, val_dataset, val_loader, batches_val, loss_ca, loss_ce, logger, device)
            log_string(f'Epoch: {epoch + 1} : val_loss: {val_loss}')
            if 'AUC' in best_results:
                best_current['AUC'], best_current['aIOU'], best_current['SIM'] = best_results['AUC'], best_results['aIOU'], best_results['SIM']
                best_model_path = os.path.join(args.TRAIN.save_ckpt_path, 'AUC_best.pt')
                torch.save(model.state_dict(), best_model_path)
                log_string(f'AUC best saved in {best_model_path}')
            if 'f1' in best_results:
                best_current['f1'] = best_results['f1']
                best_current['precision'] = best_results['precision']
                best_current['Recall'] = best_results['Recall']
                best_current['geo'] = best_results['geo']
                best_model_path = os.path.join(args.TRAIN.save_ckpt_path, 'F1_best.pt')
                torch.save(model.state_dict(), best_model_path)
                log_string(f'F1 best saved in {best_model_path}')
            if 'Acc_1' in best_results:
                best_current['Acc_1'] = best_results['Acc_1']
        scheduler.step()
    log_string(f'Best Results----AUC:{best_current["AUC"]} | aIOU:{best_current["aIOU"]} | SIM:{best_current["SIM"]} | \
            Precision:{best_current["precision"]} | Recall:{best_current["Recall"]} | F1:{best_current["f1"]} | geo:{best_current["geo"]} \
               Acc_1:{best_current["Acc_1"]}')

def val(epoch, args, model, best_current, val_dataset, val_loader, batches_val, loss_ca, loss_ce, logger, device):
    best_results = {}
    loss_sum = 0

    pr_aff, gt_aff = [], []
    pr_contact, gt_contact = [], []
    pr_logits, gt_logits = [], []

    logits_preds = torch.zeros((len(val_dataset), len(args.DATA.affordances)))
    logits_gt = torch.zeros((len(val_dataset), ))
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        for i, data_info in enumerate(val_loader):
            V = data_info['frames'].to(device)
            M = data_info['camera_pose'].float().to(device)
            O = data_info['pts'].float().to(device)

            contact_gt = data_info['contact_label'].to(device)
            affordance_gt = data_info['affordance_label'].float().to(device)
            logits_labels = data_info['aff_logits'].to(device)

            pre_contact, pre_affordance, pre_logits = model(V, M, O)

            loss_c = loss_ca(pre_contact.view(-1, 6890, 1), contact_gt.view(-1, 6890, 1))
            loss_a = loss_ca(pre_affordance, affordance_gt)
            loss_s = loss_ce(pre_logits, logits_labels)
            loss = 10*(loss_c + loss_a + loss_s)
            loss_sum += loss.item()

            if args.train_device == 'ddp':
                cur_aff = [torch.ones_like(pre_affordance) for _ in range(dist.get_world_size())]
                cur_aff_gt = [torch.ones_like(affordance_gt) for _ in range(dist.get_world_size())]
                cur_contact = [torch.ones_like(pre_contact) for _ in range(dist.get_world_size())]
                cur_contact_gt = [torch.ones_like(contact_gt) for _ in range(dist.get_world_size())]
                cur_logits = [torch.ones_like(pre_logits) for _ in range(dist.get_world_size())]
                cur_logits_gt = [torch.ones_like(logits_labels) for _ in range(dist.get_world_size())]

                dist.all_gather(cur_aff, pre_affordance)
                dist.all_gather(cur_aff_gt, affordance_gt)
                dist.all_gather(cur_contact, pre_contact)
                dist.all_gather(cur_contact_gt, contact_gt)
                dist.all_gather(cur_logits, pre_logits)
                dist.all_gather(cur_logits_gt, logits_labels)
                
                pr_aff.extend(cur_aff)
                gt_aff.extend(cur_aff_gt)
                pr_contact.extend(cur_contact)
                gt_contact.extend(cur_contact_gt)
                pr_logits.extend(pre_logits)
                gt_logits.extend(logits_labels)
                aff_preds, aff_targets = torch.cat(pr_aff, 0), torch.cat(gt_aff, 0)
                contact_preds, contact_targets = torch.cat(pr_contact, 0), torch.cat(gt_contact, 0)
                logits_preds, logits_gt = torch.stack(pr_logits, 0), torch.stack(gt_logits, 0)
            else:
                pre_contact, contact_gt = pre_contact.cpu(), contact_gt.cpu()
                pr_aff.extend(pre_affordance)
                gt_aff.extend(affordance_gt)
                pr_contact.extend(pre_contact)
                gt_contact.extend(contact_gt)
                pr_logits.extend(pre_logits)
                gt_logits.extend(logits_labels)
                aff_preds, aff_targets = torch.stack(pr_aff, 0), torch.stack(gt_aff, 0)
                contact_preds, contact_targets = torch.stack(pr_contact, 0), torch.stack(gt_contact, 0)
                logits_preds, logits_gt = torch.stack(pr_logits, 0), torch.stack(gt_logits, 0)

        AUC_, IOU_, SIM_, precision_, recall_, F1_, geo, top1_acc = evaluate(contact_preds, contact_targets, aff_preds, aff_targets, logits_preds, logits_gt)

    if(AUC_ > best_current['AUC']):
        best_results['AUC'] = AUC_
        best_results['aIOU'] = IOU_
        best_results['SIM'] = SIM_
    if(F1_ > best_current['f1']):
        best_results['precision'] = precision_
        best_results['Recall'] = recall_
        best_results['f1'] = F1_
        best_results['geo'] = geo
    if top1_acc > best_current['Acc_1']:
        best_results['Acc_1'] = top1_acc

    logger.append([int(epoch+1), AUC_, IOU_, SIM_, precision_, recall_, F1_, geo, top1_acc])

    avg_loss = loss_sum / batches_val
    return avg_loss, best_results

def inference_sample(args, model, val_loader, device):
    with torch.no_grad():
        for i, data_info in enumerate(val_loader):
            V = data_info['frames'].to(device)
            M = data_info['camera_pose'].float().to(device)
            O = data_info['pts'].float().to(device)
            B = O.size(0)
            contact_gt = data_info['contact_label'].to(device)
            frame_paths = data_info['frame_path']
            json_key = data_info['json_key']
            obj_path = data_info['obj_path']

            pre_contact, pre_affordance, _ = model(V, M, O)

            for j in range(B):
                contact_ = pre_contact[j].squeeze(dim=-1)
                affordance_ = pre_affordance[j].detach().cpu().numpy()
                pts = O[j].detach().cpu().numpy().transpose()
                affordance_gt_path = obj_path[j]
                frame_path = frame_paths[j]
                visual_affordance(pts, affordance_, affordance_gt_path, args.INFER.save_visual_path)
                visual_contact(contact_, frame_path, json_key[j], args.INFER.save_visual_path)
    print(f'Finish, see results in {args.INFER.save_visual_path}')

def inference_whole(args, model, val_loader, device):
    with torch.no_grad():
        for i, data_info in enumerate(val_loader):
            V = data_info['frames'].to(device)
            _,_,T,H,W = V.shape
            V = V.squeeze(dim=0).contiguous().view(-1,3,args.DATA.num_frames, H, W)
            B = V.shape[0]

            O = data_info['pts'].float().to(device)
            O = O.expand(B,-1,-1)
            M = data_info['camera_pose'].float().to(device)
            _,_,cam_1, cam_2 = M.shape
            M = M.squeeze(dim=0).contiguous().view(-1,args.DATA.num_frames, cam_1, cam_2)
            frame_paths = data_info['frame_path'][0]
            json_key = data_info['json_key'][0]
            obj_path = data_info['obj_path'][0]
 
            pre_contact, pre_affordance, _ = model(V, M, O)
            pre_contact = pre_contact.view(-1, 6890)
            visual_contact(pre_contact, frame_paths, json_key, args.INFER.save_visual_path)
            for j in range(B):
                affordance_ = pre_affordance[j].detach().cpu().numpy()
                pts = O[j].detach().cpu().numpy().transpose()
                visual_affordance_seq(pts, affordance_, obj_path, args.INFER.save_visual_path, j)
    print(f'Finish, see results in {args.INFER.save_visual_path}')