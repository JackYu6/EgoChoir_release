import torch
import numpy as np
import os
import json
import random
import pdb
import open3d as o3d
from torch.utils.data import Dataset
from .dataset_utils import pack_frames_to_video_clip, retry_load_images
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class EgoData(Dataset):
    def __init__(self, args, split):
        super(EgoData).__init__()

        self.args = args
        self.split = split
        self.data_json = json.load(open(args.DATA.json_file, 'r'))

        if self.split == 'train':
            self.seqs_file = args.DATA.seq_file_train
        else:
            self.seqs_file = args.DATA.seq_file_test
        self.seq_keys = self.read_file(self.seqs_file)
        if self.split == 'train':
            number_dict = {
                'Bed_sit':0, 'Bed_lay':0, 'Bottle_contain':0, 'Bottle_open':0, 'Bottle_wrapgrasp':0,'Bottle_pour':0, 'Bowl_contain':0, 'Bowl_wrapgrasp':0,
                'Chair_sit':0, 'Dishwasher_open':0, 'Faucet_open':0, 'Fork_stab':0, 'Fork_wrapgrasp':0, 'Guitar_play':0, 'Kettle_contain':0, 'Kettle_grasp':0,
                'Kettle_open':0, 'Kettle_pour':0, 'Knife_cut':0, 'Knife_grasp':0, 'Knife_stab':0, 'Mug_wrapgrasp':0, 'Mug_pour':0, 'Mug_grasp':0, 'Mug_contain':0,
                'Piano_play':0, 'Refrigerator_open':0, 'Spatula_mix':0, 'Spatula_wrapgrasp':0, 'Spoon_contain':0, 'Spoon_mix':0, 'Spoon_wrapgrasp':0, 'Suitcase_pull':0,
                'Vase_wrapgrasp':0, 'Violin_play':0
            }
            self.obj_pts_path, number_dict = self.read_file(args.DATA.obj_pts_train, number_dict)
            self.obj_list = list(number_dict.keys())
            self.pts_split = {}
            start_index = 0
            for obj_ in self.obj_list:
                temp_split = [start_index, start_index + number_dict[obj_]]
                self.pts_split[obj_] = temp_split
                start_index += number_dict[obj_]
        else:
            self.obj_pts_path = self.read_file(args.DATA.obj_pts_test)
        self.affordance_list = args.DATA.affordances
        all_clips = []
        self.all_clip, self.json_key = self.get_clips_folder(args, self.seq_keys, all_clips)
        self.transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms_video.NormalizeVideo(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_clips_folder(self, args, seq_keys, folders):
        json_key = []
        for seq_key in seq_keys:
            scene_id = seq_key.split('-')[0]
            if "ego" in seq_key.split('-')[2]:
                scene_folder = os.path.join(args.DATA.root_folder, 'GIMO', scene_id)
                seq_ids = os.listdir(scene_folder)
                seq_id_last = seq_key.split('-')[1]
                seq_id = list(filter(lambda seq: seq.split('-')[-1] == seq_id_last, seq_ids))[0]
                ego_frame = seq_key.split('-')[2]
                clip = seq_key.split('-')[3]
                clip_folder = os.path.join(scene_folder, seq_id, ego_frame, clip)
            else:
                clip = seq_key.split('-')[1]
                clip_folder = os.path.join(args.DATA.root_folder, 'Ego-EXO', scene_id, clip)
            folders.append(clip_folder)
            json_key.append(seq_key)
        return folders, json_key

    def read_file(self, path, number_dict=None):
        file_list = []
        with open(path,'r') as f:
            files = f.readlines()
            for file in files:
                file = file.strip('\n')
                if number_dict != None:
                    object_ = file.split('/')[2]
                    affordance = file.split('/')[4]
                    key = object_ + '_' + affordance
                    number_dict[key] += 1
                file_list.append(file)
            f.close()
        if number_dict != None:
            return file_list, number_dict
        else:
            return file_list
        
    def get_contact(self, contact_paths, idx, contact_folder):
        contact_files = [contact_paths[i] for i in idx]
        contact_data = []
        for file in contact_files:
            data = np.load(os.path.join(contact_folder, file), allow_pickle=True)[:,None]
            contact_data.append(data)
        contact_data = np.stack(np.array(contact_data), axis=0)
        return contact_data, contact_files
        
    def __len__(self):
        return len(self.all_clip)
    
    def __getitem__(self, index):
        
        clip_folder = self.all_clip[index]
        json_key = self.json_key[index]
        img_folder = os.path.join(clip_folder, 'img')
        camera_pose_path = os.path.join(clip_folder, 'camera_pose.txt')

        if 'ego' in json_key:
            camera_pose = self.get_camera_pose(camera_pose_path, 'GIMO')
        else:
            camera_pose = self.get_camera_pose(camera_pose_path, 'EgoEXO')

        img_paths = os.listdir(img_folder)
        if "ego" in json_key:
            img_paths = sorted(img_paths, key=lambda x: int((x.split('/')[-1]).split('.')[0]))
            contact_path = os.path.join(clip_folder, 'smpl_contact.npy')
            contact_gt = np.load(contact_path, allow_pickle=True)
            interaction_imgs = img_paths
        else:
            interaction_imgs = sorted(img_paths, key=lambda x: int((x.split('-')[-1]).split('.')[0]))
            contact_path = os.path.join(clip_folder, 'smpl_contact_'+ json_key.split('-')[-1].split('_')[0] + '.npy')
            contact_gt = np.load(contact_path, allow_pickle=True)

        fast_frames_paths, fast_frames, fast_idx = pack_frames_to_video_clip(self.args, interaction_imgs, img_folder, self.split)
        fast_frames = fast_frames.permute(3, 0, 1, 2).float()
        fast_frames = self.transform(fast_frames)

        contact_label = contact_gt[np.array(fast_idx),:]
        camera_pose = camera_pose[fast_idx, :, :]

        if self.split == 'train':
            obj_path = None
        else:
            obj_path = self.obj_pts_path[index]
        point_clous, affordance_label, affordance_logits = self.get_interaction(self.data_json, json_key, obj_path)
        
        data_info = {}
        data_info['frames'] = fast_frames
        data_info['contact_label'] = torch.from_numpy(contact_label).unsqueeze(dim=-1)
        data_info['pts'] = point_clous
        data_info['affordance_label'] = affordance_label
        data_info['aff_logits'] = torch.tensor(affordance_logits)
        data_info['camera_pose'] = torch.from_numpy(camera_pose)
        if self.split == 'infer':
            data_info['frame_path'] = fast_frames_paths
            data_info['obj_path'] = obj_path
            data_info['json_key'] = json_key
        return data_info
    
    def get_interaction(self, data_dict, key, obj_path):
        obj_affordance = data_dict[key]["interaction"]
        pts, aff_label = self.get_object(obj_affordance, obj_path)
        affordance_semantic = obj_affordance.split('_')[-1]
        aff_logits = self.affordance_list.index(affordance_semantic)

        return pts, aff_label, aff_logits
    
    def get_object(self, object, obj_path):
        if obj_path == None:
            obj_range = self.pts_split[object]
            point_sample_idx = random.sample(range(obj_range[0], obj_range[1]), 1)
            obj_data = np.load(self.obj_pts_path[point_sample_idx[0]], allow_pickle=True)
        else:
            obj_data = np.load(obj_path, allow_pickle=True)
        pts = obj_data[:, :3]
        affordance = obj_data[:, -1][:, None]
        pts = pc_normalize(pts)
        pts = pts.transpose()
        pts = torch.from_numpy(pts)
        affordance = torch.from_numpy(affordance)
        return pts, affordance
    
    def get_camera_pose(self, path, dataset):
        with open(path) as f:
            lines = f.readlines()
        if dataset == 'GIMO':
            camera_pose = np.zeros((len(lines)-1, 4, 4))
            first_frame = lines[1].split(',')
            first_frame_pose = np.array(first_frame[3:20]).astype(float).reshape((4, 4))
        else:
            camera_pose = np.zeros((len(lines), 4, 4))
            first_frame = lines[0].split(',')
            first_frame_pose = np.array(first_frame).astype(float).reshape((4, 4))
        camera_pose[0] = np.eye(4)
        first_rotate, first_transl = first_frame_pose[0:3,0:3], first_frame_pose[0:3, -1]
        try:
            rotate_inv = np.linalg.inv(first_rotate)
        except:
            rotate_inv = np.linalg.pinv(first_rotate)
        if dataset == 'GIMO':
            for i_frame, frame in enumerate(lines[2:]):
                frame = frame.split(',')
                camera_pose[i_frame] = np.array(frame[3:20]).astype(float).reshape((4, 4))
                camera_pose[i_frame][0:3,0:3] = rotate_inv @ camera_pose[i_frame][0:3,0:3]
                camera_pose[i_frame][0:3, -1] = camera_pose[i_frame][0:3, -1] - first_transl
        else:
            for i_frame, frame in enumerate(lines[1:]):
                frame = frame.split(',')
                camera_pose[i_frame] = np.array(frame).astype(float).reshape((4, 4))
                camera_pose[i_frame][0:3,0:3] = rotate_inv @ camera_pose[i_frame][0:3,0:3]
                camera_pose[i_frame][0:3, -1] = camera_pose[i_frame][0:3, -1] - first_transl
        return camera_pose[:, 0:3, :]
    
    def collate_fn(self, batch):
        frames = [sample['frames'] for sample in batch]
        contact_label = [sample['contact_label'] for sample in batch]
        pts = [sample['pts'] for sample in batch]
        affordance_label = [sample['affordance_label'] for sample in batch]
        aff_logits = [sample['aff_logits'] for sample in batch]
        camera_pose = [sample['camera_pose'] for sample in batch]

        frame_path = [sample['frame_path'] for sample in batch]
        json_key = [sample['json_key'] for sample in batch]
        obj_path = [sample['obj_path'] for sample in batch]
        return {'frames':torch.stack(frames, dim=0), 
                'contact_label':torch.stack(contact_label, dim=0),
                'pts':torch.stack(pts, dim=0),
                'affordance_label':torch.stack(affordance_label, dim=0), 
                'aff_logits': torch.stack(aff_logits, dim=0), 
                'camera_pose': torch.stack(camera_pose, dim=0),
                'frame_path':frame_path, 'json_key':json_key, 
                'obj_path':obj_path
                }

class EgoData_infer(Dataset):
    def __init__(self, args, split):
        super(EgoData_infer).__init__()

        self.args = args
        self.split = split
        self.data_json = json.load(open(args.DATA.json_file, 'r'))
        self.seqs_file = args.DATA.seq_file_test
        self.seq_keys = self.read_file(self.seqs_file)
        self.obj_pts_path = self.read_file(args.DATA.obj_pts_test)
        self.affordance_list = args.DATA.affordances
        all_clips = []
        self.all_clip, self.json_key = self.get_clips_folder(args, self.seq_keys, all_clips)
        self.transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms_video.NormalizeVideo(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_clips_folder(self, args, seq_keys, folders):
        json_key = []
        for seq_key in seq_keys:
            scene_id = seq_key.split('-')[0]
            if "ego" in seq_key.split('-')[2]:
                scene_folder = os.path.join(args.DATA.root_folder, 'GIMO', scene_id)
                seq_ids = os.listdir(scene_folder)
                seq_id_last = seq_key.split('-')[1]
                seq_id = list(filter(lambda seq: seq.split('-')[-1] == seq_id_last, seq_ids))[0]
                ego_frame = seq_key.split('-')[2]
                clip = seq_key.split('-')[3]
                clip_folder = os.path.join(scene_folder, seq_id, ego_frame, clip)
            else:
                clip = seq_key.split('-')[1]
                clip_folder = os.path.join(args.DATA.root_folder, 'Ego-EXO', scene_id, clip)
            folders.append(clip_folder)
            json_key.append(seq_key)
        return folders, json_key

    def read_file(self, path, number_dict=None):
        file_list = []
        with open(path,'r') as f:
            files = f.readlines()
            for file in files:
                file = file.strip('\n')
                if number_dict != None:
                    object_ = file.split('/')[2]
                    affordance = file.split('/')[4]
                    key = object_ + '_' + affordance
                    number_dict[key] += 1
                file_list.append(file)
            f.close()
        if number_dict != None:
            return file_list, number_dict
        else:
            return file_list
          
    def __len__(self):
        return len(self.all_clip)
    
    def __getitem__(self, index):
        
        clip_folder = self.all_clip[index]
        json_key = self.json_key[index]
        img_folder = os.path.join(clip_folder, 'img')
        camera_pose_path = os.path.join(clip_folder, 'camera_pose.txt')
        if 'ego' in json_key:
            camera_pose = self.get_camera_pose(camera_pose_path, 'GIMO')
        else:
            camera_pose = self.get_camera_pose(camera_pose_path, 'EgoEXO')
        camera_pose = torch.from_numpy(camera_pose)
        img_paths = os.listdir(img_folder)
        if "ego" in json_key:
            img_paths = sorted(img_paths, key=lambda x: int((x.split('/')[-1]).split('.')[0]))
            contact_path = os.path.join(clip_folder, 'smpl_contact.npy')
            contact_gt = torch.from_numpy(np.load(contact_path, allow_pickle=True))
            frame_paths = [os.path.join(img_folder, img_path) for img_path in img_paths]
            frame_num = len(frame_paths)
            if (frame_num // self.args.DATA.num_frames) != 0:
                add_num = (frame_num // self.args.DATA.num_frames + 1) * self.args.DATA.num_frames - frame_num
                add_file = frame_paths[-1]
                add_pose_num = (camera_pose.shape[0] // self.args.DATA.num_frames + 1) * self.args.DATA.num_frames - camera_pose.shape[0]
                add_camera_pose = camera_pose[-1,:,:].unsqueeze(dim=0).expand(add_pose_num, -1, -1)
                camera_pose = torch.cat((camera_pose, add_camera_pose), dim=0)
                add_contact = contact_gt[-1,:].unsqueeze(dim=0).expand(add_num, -1)
                contact_gt = torch.cat((contact_gt, add_contact),dim=0)
                for _ in range(add_num):
                    frame_paths.append(add_file)
        else:
            img_paths = sorted(img_paths, key=lambda x: int((x.split('-')[-1]).split('.')[0]))
            contact_path = os.path.join(clip_folder, 'smpl_contact_'+ json_key.split('-')[-1].split('_')[0] + '.npy')
            contact_gt = torch.from_numpy(np.load(contact_path, allow_pickle=True))
            frame_paths = [os.path.join(img_folder, img_path) for img_path in img_paths]
            frame_num = len(frame_paths)
            if (frame_num // self.args.DATA.num_frames) != 0:
                add_num = (frame_num // self.args.DATA.num_frames + 1) * self.args.DATA.num_frames - frame_num
                add_file = frame_paths[-1]
                add_pose_num = (camera_pose.shape[0] // self.args.DATA.num_frames + 1) * self.args.DATA.num_frames - camera_pose.shape[0]
                add_camera_pose = camera_pose[-1,:,:].unsqueeze(dim=0).expand(add_pose_num, -1, -1)
                camera_pose = torch.cat((camera_pose, add_camera_pose), dim=0)
                add_contact = contact_gt[-1,:].unsqueeze(dim=0).expand(add_num, -1)
                contact_gt = torch.cat((contact_gt, add_contact),dim=0)
                for _ in range(add_num):
                    frame_paths.append(add_file)

        frames = retry_load_images(frame_paths)
        frames = frames.permute(3, 0, 1, 2).float()
        frames = self.transform(frames)

        obj_path = self.obj_pts_path[index]
        point_clous, affordance_label, affordance_logits = self.get_interaction(self.data_json, json_key, obj_path)

        data_info = {}
        data_info['frames'] = frames
        data_info['contact_label'] = contact_gt.unsqueeze(dim=-1)
        data_info['pts'] = point_clous
        data_info['affordance_label'] = affordance_label
        data_info['aff_logits'] = torch.tensor(affordance_logits)
        data_info['camera_pose'] = camera_pose
        if self.split == 'infer':
            data_info['frame_path'] = frame_paths
            data_info['obj_path'] = obj_path
            data_info['json_key'] = json_key
        return data_info
    
    def get_interaction(self, data_dict, key, obj_path):
        obj_affordance = data_dict[key]["interaction"]
        pts, aff_label = self.get_object(obj_affordance, obj_path)
        affordance_semantic = obj_affordance.split('_')[-1]
        aff_logits = self.affordance_list.index(affordance_semantic)

        return pts, aff_label, aff_logits
    
    def get_object(self, object, obj_path):
        if obj_path == None:
            obj_range = self.pts_split[object]
            point_sample_idx = random.sample(range(obj_range[0], obj_range[1]), 1)
            obj_data = np.load(self.obj_pts_path[point_sample_idx[0]], allow_pickle=True)
        else:
            obj_data = np.load(obj_path, allow_pickle=True)
        pts = obj_data[:, :3]
        affordance = obj_data[:, -1][:, None]
        pts = pc_normalize(pts)
        pts = pts.transpose()
        pts = torch.from_numpy(pts)
        affordance = torch.from_numpy(affordance)
        return pts, affordance
    
    def get_camera_pose(self, path, dataset):
        with open(path) as f:
            lines = f.readlines()
        if dataset == 'GIMO':
            camera_pose = np.zeros((len(lines)-1, 4, 4))
            first_frame = lines[1].split(',')
            first_frame_pose = np.array(first_frame[3:20]).astype(float).reshape((4, 4))
        else:
            camera_pose = np.zeros((len(lines), 4, 4))
            first_frame = lines[0].split(',')
            first_frame_pose = np.array(first_frame).astype(float).reshape((4, 4))
        camera_pose[0] = np.eye(4)
        first_rotate, first_transl = first_frame_pose[0:3,0:3], first_frame_pose[0:3, -1]
        try:
            rotate_inv = np.linalg.inv(first_rotate)
        except:
            rotate_inv = np.linalg.pinv(first_rotate)
        if dataset == 'GIMO':
            for i_frame, frame in enumerate(lines[2:]):
                frame = frame.split(',')
                camera_pose[i_frame] = np.array(frame[3:20]).astype(float).reshape((4, 4))
                camera_pose[i_frame][0:3,0:3] = rotate_inv @ camera_pose[i_frame][0:3,0:3]
                camera_pose[i_frame][0:3, -1] = camera_pose[i_frame][0:3, -1] - first_transl
        else:
            for i_frame, frame in enumerate(lines[1:]):
                frame = frame.split(',')
                camera_pose[i_frame] = np.array(frame).astype(float).reshape((4, 4))
                camera_pose[i_frame][0:3,0:3] = rotate_inv @ camera_pose[i_frame][0:3,0:3]
                camera_pose[i_frame][0:3, -1] = camera_pose[i_frame][0:3, -1] - first_transl
        return camera_pose[:, 0:3, :]

    def collate_fn(self, batch):
        frames = [sample['frames'] for sample in batch]
        contact_label = [sample['contact_label'] for sample in batch]
        pts = [sample['pts'] for sample in batch]
        affordance_label = [sample['affordance_label'] for sample in batch]
        aff_logits = [sample['aff_logits'] for sample in batch]
        camera_pose = [sample['camera_pose'] for sample in batch]

        frame_path = [sample['frame_path'] for sample in batch]
        json_key = [sample['json_key'] for sample in batch]
        obj_path = [sample['obj_path'] for sample in batch]
        return {'frames':torch.stack(frames, dim=0), 'contact_label':torch.stack(contact_label, dim=0),'pts':torch.stack(pts, dim=0),
                'affordance_label':torch.stack(affordance_label, dim=0), 'aff_logits': torch.stack(aff_logits, dim=0), 'camera_pose': torch.stack(camera_pose, dim=0),
                'frame_path':frame_path, 'json_key':json_key, 'obj_path':obj_path}