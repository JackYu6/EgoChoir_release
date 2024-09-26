import numpy as np
import torch
import pdb
import pandas as pd
from sklearn.metrics import roc_auc_score
from numpy import nan

def evaluate(contact_pred, contact_gt, aff_pred, aff_gt, logits_preds, logits_gt):

    contact_gt = torch.flatten(contact_gt, start_dim=0, end_dim=1)
    contact_pred = torch.flatten(contact_pred, start_dim=0, end_dim=1)
    contact_pred = contact_pred.cpu().detach().numpy()
    contact_gt = contact_gt.cpu().detach().numpy()
    aff_pred = aff_pred.cpu().detach().numpy()
    aff_gt = aff_gt.cpu().detach().numpy()
    logits_preds = logits_preds.cpu().detach().numpy()
    logits_gt = logits_gt.cpu().detach().numpy()

    dist_matrix = np.load('data/smpl_neutral_geodesic_dist.npy')
    dist_matrix = torch.tensor(dist_matrix)

    AUC_aff = np.zeros((aff_gt.shape[0], aff_gt.shape[2]))
    IOU_aff = np.zeros((aff_gt.shape[0], aff_gt.shape[2]))

    SIM_matrix = np.zeros(aff_gt.shape[0])

    IOU_thres = np.linspace(0, 1, 20)
    human_num = contact_gt.shape[0]
    aff_num = aff_gt.shape[0]
    f1_avg = 0
    precision_avg = 0
    recall_avg = 0
    false_positive_dist_avg = 0
    false_negative_dist_avg = 0

    for b in range(human_num):
        contact_tp_idx = contact_gt[b, contact_pred[b,:,0]>=0.5, 0]
        contact_tp_num = np.sum(contact_tp_idx)
        contact_precision_denominator = np.sum(contact_pred[b, :, 0]>=0.5)
        contact_recall_denominator = np.sum(contact_gt[b, :, 0])

        precision_contact = contact_tp_num / (contact_precision_denominator + 1e-10)
        recall_contact = contact_tp_num / (contact_recall_denominator + 1e-10)
        f1_contact = 2 * precision_contact * recall_contact / (precision_contact + recall_contact + 1e-10)

        gt_columns = dist_matrix[:, contact_gt[b, :, 0]==1] if any(contact_gt[b, :, 0]==1) else dist_matrix
        error_matrix = gt_columns[contact_pred[b, :, 0] >= 0.5, :] if any(contact_pred[b, :, 0] >= 0.5) else gt_columns

        false_positive_dist = error_matrix.min(dim=1)[0].mean()
        false_negative_dist = error_matrix.min(dim=0)[0].mean()

        f1_avg += f1_contact
        precision_avg += precision_contact
        recall_avg += recall_contact
        false_positive_dist_avg += false_positive_dist
        false_negative_dist_avg += false_negative_dist
        #sim
    for b in range(aff_num):
        SIM_matrix[b] = SIM(aff_pred[b], aff_gt[b])

        #AUC_IOU
        aff_t_true = (aff_gt[b] >= 0.5).astype(int)
        aff_p_score = aff_pred[b]

        if np.sum(aff_t_true) == 0:
            AUC_aff[b] = np.nan
            IOU_aff[b] = np.nan
        else:
            try:
                auc_aff = roc_auc_score(aff_t_true, aff_p_score)
                AUC_aff[b] = auc_aff
            except ValueError:
                AUC_aff[b] = np.nan

            temp_iou = []
            for thre in IOU_thres:
                p_mask = (aff_p_score >= thre).astype(int)
                intersect = np.sum(p_mask & aff_t_true)
                union = np.sum(p_mask | aff_t_true)
                temp_iou.append(1.*intersect/union)
            temp_iou = np.array(temp_iou)
            aiou = np.mean(temp_iou)
            IOU_aff[b] = aiou

    AUC_ = np.nanmean(AUC_aff)
    IOU_ = np.nanmean(IOU_aff)
    SIM_ = np.mean(SIM_matrix)

    top1_acc = calculate_top_k_accuracy(logits_gt, logits_preds, k=1)

    f1_avg = f1_avg / human_num
    precision_avg = precision_avg / human_num
    recall_avg = recall_avg / human_num
    fp_error, fn_error = false_positive_dist_avg / human_num, false_negative_dist_avg / human_num
    return AUC_, IOU_, SIM_, precision_avg, recall_avg, f1_avg, fp_error.numpy(), top1_acc

def evaluate_per_class(contact_pred, contact_gt, aff_pred, aff_gt, logits_preds, logits_gt, Interactions):

    metrics = {'Metrics':['Precision','Recall','F1','geo','AUC','aIOU','SIM']}
    data_df = pd.DataFrame(metrics)
    def set_round(data):
        return np.around(data, 4)

    interaction_list = ['grasp', 'open', 'lay', 'sit', 'wrapgrasp', 'pour', 'pull', 'play', 'stab', 'contain', 'cut', 'mix']

    for interaction in interaction_list:
        exec(f'{interaction} = [[], [], [], [], [], [], []]')
    contact_pred = contact_pred.cpu().detach().numpy()
    contact_gt = contact_gt.cpu().detach().numpy()
    aff_pred = aff_pred.cpu().detach().numpy()
    aff_gt = aff_gt.cpu().detach().numpy()
    logits_preds = logits_preds.cpu().detach().numpy()
    logits_gt = logits_gt.cpu().detach().numpy()

    dist_matrix = np.load('data/smpl_neutral_geodesic_dist.npy')
    dist_matrix = torch.tensor(dist_matrix)

    AUC_aff = np.zeros((aff_gt.shape[0], aff_gt.shape[2]))
    IOU_aff = np.zeros((aff_gt.shape[0], aff_gt.shape[2]))

    SIM_matrix = np.zeros(aff_gt.shape[0])

    IOU_thres = np.linspace(0, 1, 20)
    human_num = contact_gt.shape[0]
    aff_num = aff_gt.shape[0]
    f1_avg = 0
    precision_avg = 0
    recall_avg = 0
    false_positive_dist_avg = 0
    false_negative_dist_avg = 0

    for i in range(human_num):
        current_pred = contact_pred[i]
        current_gt = contact_gt[i]
        T = current_gt.shape[0]
        for b in range(T):
            contact_tp_idx = current_gt[b, current_pred[b,:,0]>=0.5, 0]
            contact_tp_num = np.sum(contact_tp_idx)
            contact_precision_denominator = np.sum(current_pred[b, :, 0]>=0.5)
            contact_recall_denominator = np.sum(current_gt[b, :, 0])

            precision_contact = contact_tp_num / (contact_precision_denominator + 1e-10)
            recall_contact = contact_tp_num / (contact_recall_denominator + 1e-10)
            f1_contact = 2 * precision_contact * recall_contact / (precision_contact + recall_contact + 1e-10)

            gt_columns = dist_matrix[:, current_gt[b, :, 0]==1] if any(current_gt[b, :, 0]==1) else dist_matrix
            error_matrix = gt_columns[current_pred[b, :, 0] >= 0.5, :] if any(current_pred[b, :, 0] >= 0.5) else gt_columns

            false_positive_dist = error_matrix.min(dim=1)[0].mean()
            false_negative_dist = error_matrix.min(dim=0)[0].mean()

            interaction_cls = Interactions[i]
            exec(f'{interaction_cls}[3].append({f1_contact})')
            exec(f'{interaction_cls}[4].append({precision_contact})')
            exec(f'{interaction_cls}[5].append({recall_contact})')
            exec(f'{interaction_cls}[6].append({false_positive_dist})')

            f1_avg += f1_contact
            precision_avg += precision_contact
            recall_avg += recall_contact
            false_positive_dist_avg += false_positive_dist
            false_negative_dist_avg += false_negative_dist
        #sim
    for b in range(aff_num):
        interaction_cls = Interactions[b]
        SIM_matrix[b] = SIM(aff_pred[b], aff_gt[b])
        exec(f'{interaction_cls}[2].append({SIM_matrix[b]})')
        #AUC_IOU
        aff_t_true = (aff_gt[b] >= 0.5).astype(int)
        aff_p_score = aff_pred[b]

        if np.sum(aff_t_true) == 0:
            AUC_aff[b] = np.nan
            IOU_aff[b] = np.nan
            inter_auc = AUC_aff[b]
            inter_iou = IOU_aff[b]
            exec(f'{interaction_cls}[0].append({inter_auc})')
            exec(f'{interaction_cls}[1].append({inter_iou})')
        else:
            try:
                auc_aff = roc_auc_score(aff_t_true, aff_p_score)
                AUC_aff[b] = auc_aff
            except ValueError:
                AUC_aff[b] = np.nan

            temp_iou = []
            for thre in IOU_thres:
                p_mask = (aff_p_score >= thre).astype(int)
                intersect = np.sum(p_mask & aff_t_true)
                union = np.sum(p_mask | aff_t_true)
                temp_iou.append(1.*intersect/union)
            temp_iou = np.array(temp_iou)
            aiou = np.mean(temp_iou)
            IOU_aff[b] = aiou
            inter_auc = AUC_aff[b]
            inter_iou = IOU_aff[b]
            exec(f'{interaction_cls}[0].append({inter_auc})')
            exec(f'{interaction_cls}[1].append({inter_iou})')

    AUC_ = np.nanmean(AUC_aff)
    IOU_ = np.nanmean(IOU_aff)
    SIM_ = np.mean(SIM_matrix)

    top1_acc = calculate_top_k_accuracy(logits_gt, logits_preds, k=1)
    f1_avg = f1_avg / human_num
    precision_avg = precision_avg / human_num
    recall_avg = recall_avg / human_num
    fp_error, fn_error = false_positive_dist_avg / human_num, false_negative_dist_avg / human_num

    print('------Object-------')
    for i,interaction in enumerate(interaction_list):
        auc_ = set_round(np.nanmean(eval(interaction)[0]))
        aiou = set_round(np.nanmean(eval(interaction)[1]))
        sim_ = set_round(np.mean(eval(interaction)[2]))
        f1_ = set_round(np.mean(eval(interaction)[3]))
        precision_ = set_round(np.mean(eval(interaction)[4]))
        recall_ = set_round(np.mean(eval(interaction)[5]))
        geo_ = set_round(np.mean(eval(interaction)[6]))

        data_df.insert(i+1,interaction,[np.round(precision_,2), np.round(recall_,2), np.round(f1_,2), geo_*100, auc_*100, aiou*100, np.round(sim_,2)])
        print(f'{interaction} | AUC:{auc_*100} | IOU:{aiou*100} | SIM:{sim_} | F1:{f1_} | Precision:{precision_} | Recall:{recall_} | geo:{geo_*100}')

    data_df.to_csv('eval_results.csv', mode='w', header=True,index=False)
    return AUC_, IOU_, SIM_, precision_avg, recall_avg, f1_avg, fp_error.numpy(), top1_acc

def SIM(map1, map2, eps=1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)

def calculate_top_k_accuracy(true_labels, predictions, k=1):
    """
    Calculate Top-k accuracy for classification predictions.

    Parameters:
    - true_labels: numpy array of true labels
    - predictions: numpy array of predicted probabilities for each class
    - k: top k predictions to consider for accuracy

    Returns:
    - accuracy: Top-k accuracy of the predictions
    """
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
    match_array = np.any(top_k_preds == true_labels[:, None], axis=1)
    accuracy = np.mean(match_array)
    
    return accuracy
