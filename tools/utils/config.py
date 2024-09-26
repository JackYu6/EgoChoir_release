
"""Configs."""
from fvcore.common.config import CfgNode

_C = CfgNode()

_C.SPLIT = ""
_C.USE_GPU = True
_C.train_device = 'ddp'
_C.camera_motion = False
_C.finetune = False

_C.DATA = CfgNode()
_C.DATA.json_file = ""
_C.DATA.seq_file_train = ""
_C.DATA.seq_file_test = ""
_C.DATA.root_folder = ""
_C.DATA.obj_pts_train = ""
_C.DATA.obj_pts_test = ""
_C.DATA.fps = 30
_C.DATA.sampling_rate = 2.0
_C.DATA.num_frames = 32
_C.DATA.num_clips = 5
_C.DATA.mean = []
_C.DATA.std = []
_C.DATA.slow_alapha = 4
_C.DATA.affordances = []
_C.DATA.max_interaction = 3
_C.DATA.stride = 16
_C.DATA.num_workers = 8
_C.DATA.prefetch_factor = None
_C.DATA.feature_file = ''

_C.HYPER = CfgNode()
_C.HYPER.batch = 16
_C.HYPER.epoch = 20
_C.HYPER.lr = 0.0001
_C.HYPER.weight_decay = 0.001

_C.MODEL = CfgNode()
_C.MODEL.emb_dim = 1024
_C.MODEL.video_ckpt = ""
_C.MODEL.slowfast_config = ""
_C.MODEL.motion_ckpt = ""

_C.TRAIN = CfgNode()
_C.TRAIN.save_ckpt_path = ""
_C.TRAIN.tune_checkpoint = ""

_C.Eval.eval_ckpt_path = ""

_C.INFER = CfgNode()
_C.INFER.save_visual_path = ""

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
