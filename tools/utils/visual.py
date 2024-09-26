import open3d as o3d
import numpy as np
import os
import torch
import pdb

Vector3dVector = o3d.utility.Vector3dVector

def visual_affordance(pts, affordance_pred, GT_path, results_folder):

    pred_point = o3d.geometry.PointCloud()
    pred_point.points = o3d.utility.Vector3dVector(pts)

    reference_color = np.array([255, 0, 0])
    back_color = np.array([190, 190, 190])

    pred_color = np.zeros((2000,3))
    for i, pred in enumerate(affordance_pred):
        scale_i = pred
        pred_color[i] = (reference_color-back_color) * scale_i + back_color

    pred_point.colors = o3d.utility.Vector3dVector(pred_color.astype(np.float64) / 255.0)

    results_folder = os.path.join(results_folder, 'affordance')
    os.makedirs(results_folder, exist_ok=True)
    file_name = GT_path.split('/')[2] + '_' + GT_path.split('/')[3] + '_' + (GT_path.split('/')[-1]).split('.')[0]
    pred_file_name = file_name + '_pred.ply'
    o3d.io.write_point_cloud(os.path.join(results_folder, pred_file_name), pred_point)

def visual_contact(pred_contact, frame_path, json_key, results_folder):

    frames = pred_contact.shape[0]
    contact_color = np.array([255.0, 191.0, 0.])

    if 'ego' in json_key:
        scene, seq = json_key.split('-')[0], json_key.split('-')[1]
        ego_frame, clip = json_key.split('-')[2], json_key.split('-')[-2]
        save_folder = os.path.join(results_folder, 'GIMO_contact', scene, seq, ego_frame, clip)
        os.makedirs(save_folder, exist_ok=True)
    else:
        take_name, clip = json_key.split('-')[0], json_key.split('-')[1]
        obj_aff = json_key.split('-')[-1]
        save_folder = os.path.join(results_folder, 'EgoExo_contact', take_name, clip, obj_aff)
        os.makedirs(save_folder, exist_ok=True)

    for i in range(frames):
        pred_contact_id = torch.where(pred_contact[i] > 0.5)[0].cpu()
        pred_contact_id = np.asarray(pred_contact_id)

        colors = np.array([255.0, 255.0, 255.0])[None, :].repeat(6890, axis=0)
        colors[pred_contact_id] = contact_color
        colors = colors / 255.0        

        source_human_path = 'dataset/support_files/smpl_template.ply'
        source_human = o3d.io.read_triangle_mesh(source_human_path)
        source_human.vertex_colors = Vector3dVector(colors)

        pred_save_file = os.path.join(save_folder, frame_path[i].split('/')[-1].split('.')[0] + '_pred.ply')
        o3d.io.write_triangle_mesh(pred_save_file, source_human)

def visual_affordance_seq(pts, affordance_pred, GT_path, results_folder, clip):
    pred_point = o3d.geometry.PointCloud()
    pred_point.points = o3d.utility.Vector3dVector(pts)

    reference_color = np.array([255, 0, 0])
    back_color = np.array([190, 190, 190])

    pred_color = np.zeros((2000,3))
    for i, pred in enumerate(affordance_pred):
        scale_i = pred
        pred_color[i] = (reference_color-back_color) * scale_i + back_color

    pred_point.colors = o3d.utility.Vector3dVector(pred_color.astype(np.float64) / 255.0)

    results_folder = os.path.join(results_folder, 'affordance')
    os.makedirs(results_folder, exist_ok=True)
    file_name = GT_path.split('/')[2] + '_' + GT_path.split('/')[3] + '_' + (GT_path.split('/')[-1]).split('.')[0]
    pred_file_name = file_name + '_' + str(clip) + '_pred.ply'
    o3d.io.write_point_cloud(os.path.join(results_folder, pred_file_name), pred_point)