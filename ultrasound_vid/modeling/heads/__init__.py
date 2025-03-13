from .temporal_head import TemporalROIHeads, Res5TemporalROIBoxHeads
from .roi_head import SingleFrameRes5ROIHeads
from .deformable_transformer import DeformableTransformer
from .transvod_head import TransVODHead
from .context_head import Res5FlowContextHeads
from .mega_head import MEGAHeads
from .selsa_head import Res5SELSAHeads
from .context_transformer import ContextDeformableTransformer
from .ptseformer_head import PTSEHead


def build_deforamble_transformer(cfg):
    if cfg.MODEL.META_ARCHITECTURE == "TransVOD":
        return TransVODHead(
            d_model=cfg.MODEL.DeformableDETR.HIDDEN_DIM,
            nhead=cfg.MODEL.DeformableDETR.NHEADS,
            num_encoder_layers=cfg.MODEL.DeformableDETR.ENCODER_LAYERS,
            num_decoder_layers=cfg.MODEL.DeformableDETR.DECODER_LAYERS,
            dim_feedforward=cfg.MODEL.DeformableDETR.DIM_FEEDFORWARD,
            dropout=cfg.MODEL.DeformableDETR.DROPOUT,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=cfg.MODEL.DeformableDETR.NUM_FEATURE_LEVELS,
            dec_n_points=cfg.MODEL.DeformableDETR.DECODER_N_POINTS,
            enc_n_points=cfg.MODEL.DeformableDETR.ENCODER_N_POINTS,
            two_stage=cfg.MODEL.DeformableDETR.TWO_STAGE,
            two_stage_num_proposals=cfg.MODEL.DeformableDETR.NUM_QUERIES,
            use_temporal_encoder=cfg.MODEL.DeformableDETR.USE_TEMPORAL_ENCODER, 
            num_query_encoder_layers=cfg.MODEL.DeformableDETR.QUERY_ENCODER_LAYERS,
            num_temporal_decoder_layers=cfg.MODEL.DeformableDETR.TEMPORAL_DECODER_LAYERS, 
            num_queries=cfg.MODEL.DeformableDETR.NUM_QUERIES,
            buffer_length=cfg.INPUT.TRAIN_FRAME_SAMPLER.NUM_OUT_FRAMES,
        )
    elif cfg.MODEL.META_ARCHITECTURE in ["DeformableDETR", "CVANet"]:
        if cfg.MODEL.DeformableDETR.USE_FLOW_CONTEXT:
            from ultrasound_vid.modeling.layers import FlowNetS
            flownet = FlowNetS(cfg)
            return ContextDeformableTransformer(
                d_model=cfg.MODEL.DeformableDETR.HIDDEN_DIM,
                nhead=cfg.MODEL.DeformableDETR.NHEADS,
                num_encoder_layers=cfg.MODEL.DeformableDETR.ENCODER_LAYERS,
                num_decoder_layers=cfg.MODEL.DeformableDETR.DECODER_LAYERS,
                dim_feedforward=cfg.MODEL.DeformableDETR.DIM_FEEDFORWARD,
                dropout=cfg.MODEL.DeformableDETR.DROPOUT,
                activation="relu",
                return_intermediate_dec=True,
                num_feature_levels=cfg.MODEL.DeformableDETR.NUM_FEATURE_LEVELS,
                dec_n_points=cfg.MODEL.DeformableDETR.DECODER_N_POINTS,
                enc_n_points=cfg.MODEL.DeformableDETR.ENCODER_N_POINTS,
                two_stage=cfg.MODEL.DeformableDETR.TWO_STAGE,
                two_stage_num_proposals=cfg.MODEL.DeformableDETR.NUM_QUERIES,
                temporal_attn=cfg.MODEL.DeformableDETR.TEMPORAL_ATTN,
                buffer_length=cfg.INPUT.TRAIN_FRAME_SAMPLER.NUM_OUT_FRAMES,
                context_feature=cfg.MODEL.CONTEXT_FEATURE,
                context_frames=cfg.MODEL.CONTEXT_FLOW_FRAMES,
                context_step=cfg.MODEL.CONTEXT_STEP_LEN,
                flownet=flownet,
                flownet_weights=cfg.MODEL.FLOWNET_WEIGHTS,
            )
        else:
            return DeformableTransformer(
                d_model=cfg.MODEL.DeformableDETR.HIDDEN_DIM,
                nhead=cfg.MODEL.DeformableDETR.NHEADS,
                num_encoder_layers=cfg.MODEL.DeformableDETR.ENCODER_LAYERS,
                num_decoder_layers=cfg.MODEL.DeformableDETR.DECODER_LAYERS,
                dim_feedforward=cfg.MODEL.DeformableDETR.DIM_FEEDFORWARD,
                dropout=cfg.MODEL.DeformableDETR.DROPOUT,
                activation="relu",
                return_intermediate_dec=True,
                num_feature_levels=cfg.MODEL.DeformableDETR.NUM_FEATURE_LEVELS,
                dec_n_points=cfg.MODEL.DeformableDETR.DECODER_N_POINTS,
                enc_n_points=cfg.MODEL.DeformableDETR.ENCODER_N_POINTS,
                two_stage=cfg.MODEL.DeformableDETR.TWO_STAGE,
                two_stage_num_proposals=cfg.MODEL.DeformableDETR.NUM_QUERIES,
                temporal_attn=cfg.MODEL.DeformableDETR.TEMPORAL_ATTN,
                buffer_length=cfg.INPUT.TRAIN_FRAME_SAMPLER.NUM_OUT_FRAMES,
            )
    elif cfg.MODEL.META_ARCHITECTURE == "PTSEFormer":
        return PTSEHead(
            d_model=cfg.MODEL.DeformableDETR.HIDDEN_DIM,
            nhead=cfg.MODEL.DeformableDETR.NHEADS,
            num_encoder_layers=cfg.MODEL.DeformableDETR.ENCODER_LAYERS,
            num_decoder_layers=cfg.MODEL.DeformableDETR.DECODER_LAYERS,
            dim_feedforward=cfg.MODEL.DeformableDETR.DIM_FEEDFORWARD,
            dropout=cfg.MODEL.DeformableDETR.DROPOUT,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=cfg.MODEL.DeformableDETR.NUM_FEATURE_LEVELS,
            dec_n_points=cfg.MODEL.DeformableDETR.DECODER_N_POINTS,
            enc_n_points=cfg.MODEL.DeformableDETR.ENCODER_N_POINTS,
            two_stage=cfg.MODEL.DeformableDETR.TWO_STAGE,
            two_stage_num_proposals=cfg.MODEL.DeformableDETR.NUM_QUERIES,
            buffer_length=cfg.INPUT.TRAIN_FRAME_SAMPLER.NUM_OUT_FRAMES,
        )
    raise NotImplementedError(
        f"Only support DeformableDETR, TransVOD, CVANet and PTSEFormer now, but reveive {cfg.MODEL.META_ARCHITECTURE}"
    )