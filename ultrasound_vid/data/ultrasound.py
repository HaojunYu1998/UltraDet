# import json
# import logging
# import os
# import pickle
# import pandas as pd
# from glob import glob
# from path import Path

# from detectron2.structures import BoxMode
# from detectron2.utils.comm import is_main_process, get_rank, get_world_size, synchronize
# from ultrasound_vid.data.utils import hash_idx

# logger = logging.getLogger(__name__)


# def map_frame_annos_to_d2type(anno):
#     video_folder = anno["video_folder"]
#     assert not video_folder.endswith("/")
#     relpath = anno["relpath"]
#     frame_annos_ori = anno["frame_anno"]
#     frame_annos_new = dict()

#     for key, fanno in enumerate(frame_annos_ori):
#         fanno_new = dict()
#         fanno_new["dataset"] = anno["dataset"]
#         fanno_new["file_name"] = video_folder + f"/{key}.jpg"
#         fanno_new["height"] = anno["video_info"]["height"]
#         fanno_new["width"] = anno["video_info"]["width"]
#         fanno_new["frame_idx"] = key
#         fanno_new["video_folder"] = video_folder
#         fanno_new["relpath"] = relpath
#         fanno_new["annotations"] = []
#         for box_info in fanno["box"]:
#             box_new = dict(
#                 bbox=[
#                     box_info["xtl"], box_info["ytl"], box_info["xbr"], box_info["ybr"],
#                 ],
#                 bbox_mode=BoxMode.XYXY_ABS,
#                 track_id=box_info["lesion_index"],
#                 category_id=0,
#             )
#             fanno_new["annotations"].append(box_new)
#         frame_annos_new[key] = fanno_new

#     return frame_annos_new


# class DumpManager:
#     def __init__(self, rank, world_size):
#         self.rank = rank
#         self.world_size = world_size
#         self.need_update = True

#     def dump_anno(self, file_name, anno) -> Path:
#         annos_path = Path(file_name)
#         if self.need_update and hash_idx(file_name, self.world_size) == self.rank:
#             # We only do dump work at main process
#             frame_annos = map_frame_annos_to_d2type(anno)
#             with open(annos_path, "wb") as fp:
#                 pickle.dump(frame_annos, fp)
#         return annos_path

#     def check_data_update(self, anno_stats_path, num_annos):
#         try:
#             with open(anno_stats_path, "r") as fp:
#                 anno_stats = json.load(fp)
#                 old_num_annos = anno_stats["num_annos"]
#                 if old_num_annos == num_annos:
#                     self.need_update = False
#                     logger.info(">>> >>> >>> 数据集未更新，无需序列化 <<< <<< <<<")
#         except:
#             logger.info(">>> >>> >>> 开始序列化标注... <<< <<< <<<")


# def load_ultrasound_annotations(
#     dataset_name, csv_file, anno_temp_path, jpg_root, us_processed_data, 
# ):
#     logger.info(f">> >> >> Getting annotations start.")
#     logger.info(f"dataset name: {dataset_name}")

#     dataset_dicts = []
#     world_size = get_world_size()
#     rank = get_rank()
#     dump_manager = DumpManager(rank, world_size)
#     anno_stats_path = os.path.join(anno_temp_path, dataset_name + ".json")
#     df = pd.read_csv(csv_file)

#     num_annos = len(df)

#     dump_manager.check_data_update(anno_stats_path, num_annos)
#     for _, row in df.iterrows():
#         jpg_folder = Path(us_processed_data) / row["jpg_folder"]
#         json_file = Path(us_processed_data) / row["json_file"]
#         relpath = os.path.relpath(jpg_folder, jpg_root)
#         file_name = relpath.replace("/", "_")

#         anno = dict()
#         with open(json_file, "r") as f:
#             anno = json.load(f)
#         assert "frame_anno" in anno
#         anno["num_frames"] = len(anno["frame_anno"])
#         anno["anno_path"] = json_file
#         anno["dataset"] = dataset_name
#         anno["relpath"] = relpath
#         anno["video_key"] = jpg_folder.split("/")[-1] + ".mp4@md5"
#         anno["video_folder"] = jpg_folder
#         # anno["height"] = video_info["cropped_shape"][0]
#         # anno["width"] = video_info["cropped_shape"][1]
#         anno["frame_annos_path"] = dump_manager.dump_anno(
#             os.path.join(anno_temp_path, file_name), anno
#         )
        
#         anno.pop("frame_anno")
#         dataset_dicts.append(anno)

#     anno_stats = {"num_annos": num_annos}
#     if is_main_process():
#         try: # in case the user have no write permission
#             with open(anno_stats_path, "w") as fp:
#                 json.dump(anno_stats, fp)
#         except:
#             pass

#     logger.info(
#         f"<< << << Getting annotations done. {dataset_name} {len(dataset_dicts)}"
#     )
#     synchronize()
#     return dataset_dicts
