from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.vipt.config import cfg, update_config_from_file


def parameters(yaml_name: str, epoch=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/vipt/%s.yaml' % yaml_name)
    # yaml_file = os.path.join(prj_dir, 'experiments/METrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    # coding
    # params.checkpoint = os.path.join(save_dir, "checkpoints/train/vipt/%s/ViPTrack_ep0060.pth.tar" % (yaml_name))

    # params.checkpoint = os.path.join(save_dir, "checkpoints/train/vipt/%s/ViPTrack_ep%04d.pth.tar" % (yaml_name, cfg.TEST.EPOCH))
    # params.checkpoint = os.path.join(save_dir, "checkpoints/train/METrack/%s/METrack_ep%04d.pth.tar" % (yaml_name, cfg.TEST.EPOCH))
    params.checkpoint = os.path.join(prj_dir, "tracking/output/checkpoints/train/METrack/METracker_0pme_1pb_fovea_ep80/METrack_ep0070.pth.tar")

    # params.checkpoint = os.path.join(prj_dir, "./models/ViPT_%s.pth"%yaml_name)
    print("checkpoint path:", params.checkpoint)
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
