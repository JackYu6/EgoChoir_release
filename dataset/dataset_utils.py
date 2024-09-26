import torch
import cv2
import time
import os
import numpy as np
import random

def retry_load_images(image_paths, retry=10, backend="pytorch"):
    """
    This function is to load images with support of retrying for failed load.

    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.

    Returns:
        imgs (list): list of loaded images.
    """
    for i in range(retry):
        imgs = [cv2.imread(image_path) for image_path in image_paths]

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            print("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))

def temporal_sampling(num_frames, start_idx, end_idx, num_samples, oversize, start_frame=0):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        num_frames (int): number of frames of the trimmed action clip
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
        start_frame (int): starting frame of the action clip in the untrimmed video
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    if oversize > 0 and oversize <= num_frames:
        index = torch.arange(0, num_frames)
        repeat_index = torch.arange(0, oversize)
        index = torch.cat((index, repeat_index)).long()
    elif oversize > num_frames:
        index = torch.arange(0, num_frames)
        repeat_time = oversize // num_frames
        for i in range(repeat_time):
            repeat_index = torch.arange(0, num_frames)
            index = torch.cat((index, repeat_index))
        last_index = oversize - repeat_time*num_frames
        last_repeat = torch.arange(0, last_index)
        index = torch.cat((index, last_repeat)).long()
    else:
        index = torch.floor(torch.linspace(start_idx, end_idx, num_samples))
        index = torch.clamp(index, 0, num_frames - 1).long()
    return start_frame + index


def pack_frames_to_video_clip(cfg, imgs, img_folder, split, temporal_sample_index=-1, target_fps=30):
    # Load video by loading its extracted frames
    frames = len(imgs)
    fps, sampling_rate, num_samples = cfg.DATA.fps, cfg.DATA.sampling_rate, cfg.DATA.num_frames
    if split == 'train':
        start_idx, end_idx, oversize = get_start_end_idx(
            frames,
            num_samples,
            temporal_sample_index,
            num_clips=1,
        )
    else:
        start_idx = 0
        end_idx = frames
        oversize = num_samples - frames
    start_idx, end_idx = start_idx + 1, end_idx
    frame_idx = temporal_sampling(frames, start_idx, end_idx, num_samples, oversize)
    frame_paths = [os.path.join(img_folder, imgs[idx]) for idx in frame_idx]
    frames = retry_load_images(frame_paths)

    return  frame_paths, frames, frame_idx

def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    overframes = 0
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = video_size - 1
    return start_idx, end_idx, overframes

