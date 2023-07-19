from detectron2.config import CfgNode as CN


def add_ultrasound_config(cfg):
    """
    Add config for ultrasound videos.
    """

    # We use n_fold cross validation for evaluation.
    cfg.DATASETS.NUM_FOLDS = 4
    cfg.DATASETS.TEST_FOLDS = (0,)
    cfg.DATASETS.FRAMESAMPLER = "FrameSampler"
    cfg.DATASETS.SPLIT = "trainval"
    cfg.DATASETS.SUFFIX = "_cva"

    # Segments per batch for training
    cfg.SOLVER.SEGS_PER_BATCH = 16
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.TEST.EVAL_PERIOD = 10000
    cfg.TEST.SKIP_EXISTS = False

    # Frame sampler for training
    cfg.INPUT.TRAIN_FRAME_SAMPLER = CN()
    # Sample interval
    cfg.INPUT.TRAIN_FRAME_SAMPLER.INTERVAL = 30
    cfg.INPUT.TRAIN_FRAME_SAMPLER.MEMORY_INTERVAL = 30
    cfg.INPUT.TRAIN_FRAME_SAMPLER.NUM_OUT_FRAMES = 16
    cfg.INPUT.TRAIN_FRAME_SAMPLER.MEMORY_FRAMES = 16
    cfg.INPUT.TRAIN_FRAME_SAMPLER.SAVE_SAMPLES = False
    
    # Fixed area
    cfg.INPUT.FIXED_AREA_TRAIN = 900000
    cfg.INPUT.FIXED_AREA_TEST = 900000

    # Parameters for backbone
    cfg.MODEL.POSITION_EMBEDDING_TEMPERATURE = 10000
    cfg.MODEL.RESNETS.HALF_CHANNEL = False
    cfg.MODEL.RESNETS.RES5_OUT_CHANNEL = 512
    cfg.MODEL.RESNETS.DEPTH = 34
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    cfg.MODEL.RESNETS.HALF_CHANNEL = True

    # Reset NMS parameters
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 256
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 256
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 128
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 128
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 0.66, 1.0, 2.0, 3.0]]

    # Config for temporal relation
    cfg.MODEL.ROI_BOX_HEAD.RELATION_HEAD_NUMS = 8
    cfg.MODEL.ROI_BOX_HEAD.RELATION_LAYER_NUMS = 2
    cfg.MODEL.ROI_BOX_HEAD.SELSA_LAYER_NUMS = 2
    cfg.MODEL.ROI_BOX_HEAD.INTERVAL_PRE_TEST = 15
    cfg.MODEL.ROI_BOX_HEAD.INTERVAL_AFTER_TEST = 0
    cfg.MODEL.ROI_BOX_HEAD.CAUSAL_RELATION = False
    cfg.MODEL.ROI_BOX_HEAD.ROI_REL_POSITION = False

    # To evaluate mAP
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # ROI heads sampler
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5

    cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
    cfg.MODEL.MASK_ON = False

    cfg.INPUT.SCALE_TRAIN = (0.3, 1.5)
    cfg.INPUT.SCALE_TEST = 1.0

    # context attention
    cfg.MODEL.CONTEXT_FEATURE = ["p5"]
    cfg.MODEL.CONTEXT_STEP_LEN = 10
    cfg.MODEL.CONTEXT_FLOW_FRAMES = 2
    cfg.MODEL.CONTEXT_TEMP_AGGREGATION = True
    cfg.MODEL.CONTEXT_IOF_ALIGN = True
    cfg.MODEL.FLOWNET_WEIGHTS = "pretrained_models/flownet.ckpt"
    cfg.MODEL.FLOWNET_POOL_STRIDE = 4
    cfg.MODEL.FLOWNET_METHOD = "DFF"

    # FPN
    cfg.MODEL.FPN.OUT_FEATURES = ["p4", "p5"]

    # misc
    cfg.AUTO_DIR = False

    # solver
    cfg.SOLVER.ADAM_BETA = (0.9, 0.999)
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.LR_MULTIPLIER_NAME = ()
    cfg.SOLVER.LR_MULTIPLIER_VALUE = ()

