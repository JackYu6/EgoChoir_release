from .config import get_cfg
import pdb

def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    if hasattr(args, "use_gpu"):
        cfg.USE_GPU = args.use_gpu
    if hasattr(args, "train_device"):
        cfg.train_device = args.train_device
    if hasattr(args, "finetune"):
        cfg.finetune = args.finetune
    return cfg