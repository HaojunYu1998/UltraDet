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
    cfg.MODEL.RPN_ONLY_TEST = False
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

    # Config for temporal retinanet
    cfg.MODEL.RETINANET.INTERVAL_PRE_TEST = 15
    cfg.MODEL.RETINANET.INTERVAL_AFTER_TEST = 0

    # Config for temporal relation
    cfg.MODEL.ROI_BOX_HEAD.RELATION_HEAD_NUMS = 8
    cfg.MODEL.ROI_BOX_HEAD.RELATION_LAYER_NUMS = 2
    cfg.MODEL.ROI_BOX_HEAD.SELSA_LAYER_NUMS = 2
    cfg.MODEL.ROI_BOX_HEAD.INTERVAL_PRE_TEST = 15
    cfg.MODEL.ROI_BOX_HEAD.INTERVAL_AFTER_TEST = 0
    cfg.MODEL.ROI_BOX_HEAD.CAUSAL_RELATION = False
    cfg.MODEL.ROI_BOX_HEAD.ROI_REL_POSITION = False
    cfg.MODEL.ROI_BOX_HEAD.USE_ATTENTION = False
    cfg.MODEL.ROI_BOX_HEAD.USE_ONE_LAYER = False
    cfg.MODEL.ROI_BOX_HEAD.NTCA_LAYER_INDEX = 0
    cfg.MODEL.ROI_BOX_HEAD.NO_AUX_LOSS = False

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

    # organ-specific
    cfg.MODEL.ORGAN_SPECIFIC = CN()
    cfg.MODEL.ORGAN_SPECIFIC.ENABLE = ()  # 'cls', 'reg'
    cfg.MODEL.ORGAN_SPECIFIC.BREAST_LOSS_WEIGHT = 1.0
    cfg.MODEL.ORGAN_SPECIFIC.THYROID_LOSS_WEIGHT = 1.0

    # context attention
    cfg.MODEL.CONTEXT_FEATURE = ["p5"]
    cfg.MODEL.CONTEXT_STEP_LEN = 8
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


def add_defcn_config(cfg):
    """
    Add config for DeFCN_RPN
    """
    cfg.MODEL.DeFCN = CN()

    # DeFCN.
    cfg.MODEL.DeFCN.IN_FEATURES = ["res4"]
    cfg.MODEL.DeFCN.FPN_STRIDES = [16]
    cfg.MODEL.DeFCN.NUM_CONVS = 4
    cfg.MODEL.DeFCN.NORM_REG_TARGETS = True
    cfg.MODEL.DeFCN.NMS_THRESH_TEST = 0.6
    cfg.MODEL.DeFCN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    cfg.MODEL.DeFCN.FOCAL_LOSS_GAMMA = 2.0
    cfg.MODEL.DeFCN.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.DeFCN.IOU_LOSS_TYPE = "giou"
    cfg.MODEL.DeFCN.REG_WEIGHT = 2.0
    cfg.MODEL.DeFCN.NMS_TYPE = "normal" # "hindex"
    cfg.MODEL.DeFCN.PRE_NMS_TOPK = 256
    cfg.MODEL.DeFCN.NUM_PROPOSALS = 16

    # Focal loss.
    cfg.MODEL.DeFCN.PRIOR_PROB = 0.01

    # Shift generator.
    cfg.MODEL.SHIFT_GENERATOR = CN()
    cfg.MODEL.SHIFT_GENERATOR.NUM_SHIFTS = 1
    cfg.MODEL.SHIFT_GENERATOR.OFFSET = 0.5

    # POTO.
    cfg.MODEL.POTO = CN()
    cfg.MODEL.POTO.ALPHA = 0.8
    cfg.MODEL.POTO.CENTER_SAMPLING_RADIUS = 1.5
    cfg.MODEL.POTO.FILTER_KERNEL_SIZE = 3
    cfg.MODEL.POTO.FILTER_TAU = 2
    cfg.MODEL.POTO.AUX_TOPK = 9

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.WEIGHT_DECAY = 0.05


def add_deformable_detr_config(cfg):
    """
    Add config for DeformableDETR
    """
    cfg.MODEL.DeformableDETR = CN()

    # DeformableDETR.
    cfg.MODEL.DeformableDETR.NUM_CLASSES = 1
    cfg.MODEL.DeformableDETR.POS_EMBED_TYPE = "sine"
    cfg.MODEL.DeformableDETR.IN_FEATURES = ["res4"]
    cfg.MODEL.DeformableDETR.TWO_STAGE = False
    cfg.MODEL.DeformableDETR.NUM_FEATURE_LEVELS = 1
    cfg.MODEL.DeformableDETR.HIDDEN_DIM = 256
    cfg.MODEL.DeformableDETR.NHEADS = 8
    cfg.MODEL.DeformableDETR.ENCODER_LAYERS = 6
    cfg.MODEL.DeformableDETR.DECODER_LAYERS = 6
    cfg.MODEL.DeformableDETR.DIM_FEEDFORWARD = 1024
    cfg.MODEL.DeformableDETR.DROPOUT = 0.1
    cfg.MODEL.DeformableDETR.DECODER_N_POINTS = 4
    cfg.MODEL.DeformableDETR.ENCODER_N_POINTS = 4
    cfg.MODEL.DeformableDETR.NUM_QUERIES = 16
    cfg.MODEL.DeformableDETR.WITH_BOX_REFINE = False
    cfg.MODEL.DeformableDETR.TEMPORAL_ATTN = False

    # TransVOD.
    cfg.MODEL.DeformableDETR.USE_TEMPORAL_ENCODER = False
    cfg.MODEL.DeformableDETR.QUERY_ENCODER_LAYERS = 3
    cfg.MODEL.DeformableDETR.TEMPORAL_DECODER_LAYERS = 1

    # Flow Context.
    cfg.MODEL.DeformableDETR.USE_FLOW_CONTEXT = False

    # Loss.
    cfg.MODEL.DeformableDETR.CLASS_WEIGHT = 2.0
    cfg.MODEL.DeformableDETR.GIOU_WEIGHT = 2.0
    cfg.MODEL.DeformableDETR.L1_WEIGHT = 5.0
    cfg.MODEL.DeformableDETR.AUX_LOSS = True
    cfg.MODEL.DeformableDETR.NMS_THRESH = 0.7
    cfg.MODEL.DeformableDETR.USE_NMS = True

    cfg.SOLVER.LR_MULTIPLIER_NAME = ()
    cfg.SOLVER.LR_MULTIPLIER_VALUE = ()


def add_fcos_config(cfg):
    cfg.MODEL.FCOS = CN()

    # This is the number of foreground classes.
    # HACK: used for objectness score
    cfg.MODEL.FCOS.NUM_CLASSES = 1
    cfg.MODEL.FCOS.IN_FEATURES = ["res4"]
    cfg.MODEL.FCOS.FPN_STRIDES = [16]
    cfg.MODEL.FCOS.PRIOR_PROB = 0.01
    cfg.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
    cfg.MODEL.FCOS.NMS_TH = 0.6
    cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 256
    cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST = 256
    cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 16
    cfg.MODEL.FCOS.POST_NMS_TOPK_TEST = 16
    cfg.MODEL.FCOS.NORM = "GN"  # Support GN or none
    cfg.MODEL.FCOS.USE_SCALE = True

    # Multiply centerness before threshold
    # This will affect the final performance by about 0.05 AP but save some time
    cfg.MODEL.FCOS.THRESH_WITH_CTR = False

    # Focal loss parameters
    cfg.MODEL.FCOS.LOSS_ALPHA = 0.5
    cfg.MODEL.FCOS.LOSS_GAMMA = 2
    cfg.MODEL.FCOS.SIZES_OF_INTEREST = [32, 64, 128, 256, 512]
    cfg.MODEL.FCOS.USE_DEFORMABLE = False

    # the number of convolutions used in the cls and bbox tower
    cfg.MODEL.FCOS.NUM_CLS_CONVS = 4
    cfg.MODEL.FCOS.NUM_BOX_CONVS = 4
    cfg.MODEL.FCOS.NUM_SHARE_CONVS = 0
    cfg.MODEL.FCOS.CENTER_SAMPLE = True
    cfg.MODEL.FCOS.POS_RADIUS = 1.5
    cfg.MODEL.FCOS.LOC_LOSS_TYPE = "giou"
    cfg.MODEL.FCOS.YIELD_PROPOSAL = False


def add_yolox_config(cfg):
    """
    Add config for YOLOX_RPN
    """
    cfg.MODEL.YOLOX = CN()
    cfg.MODEL.YOLOX.IN_FEATURES = [ "res4" ]
    cfg.MODEL.YOLOX.STRIDES = [ 16 ]
    cfg.MODEL.YOLOX.NMS_THRESH_TEST = 0.7
    cfg.MODEL.YOLOX.NMS_TYPE = "normal"
    cfg.MODEL.YOLOX.PRE_NMS_TOPK = 256
    cfg.MODEL.YOLOX.NUM_PROPOSALS = 16


def add_swin_config(cfg):
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.BACKBONE.FREEZE_AT = -1


def add_tracker_config(cfg):
    """
    Add config for DeFCN
    """
    cfg.MODEL.TRACKER = CN()
    
    cfg.MODEL.TRACKER.SPARSE_NMS_THRESH = 0.5
    cfg.MODEL.TRACKER.NEW_SCORE_THRESH = 0.2
    cfg.MODEL.TRACKER.KALMAN_IOU_THRESH = 0.5
    cfg.MODEL.TRACKER.VALID_SCORE_THRESH = 0.05
    cfg.MODEL.TRACKER.DUPLICATE_IOU_THRESH = 0.5
    cfg.MODEL.TRACKER.TRACK_BUFFER_SIZE = 20
    cfg.MODEL.TRACKER.DURATION_THRESH = 15
