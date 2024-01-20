import pandas as pd
import os 
import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import random
import jsonlines

orientation = "front_middle_camera"
parquet_file_path = '/mnt/ve_share/chenminghua/dataset/icu30_2d_3d_key_point_clip.parquet'
df = pd.read_parquet(parquet_file_path)
usable_df = df[df["sensor_name"] == orientation]
print(usable_df)
# annotations = random.sample(["/" + _ for _ in set(usable_df["annotation_path"])], 10000)
annotations = ["/" + _ for _ in set(usable_df["annotation_path"])]
cantread_path = "./demos/data_gn6/cantread.txt"
result_path = "./demos/data_gn6/gn_all.jsonl"

def worker(json_path):
    try:
        result = dict()
        with open(os.path.join(json_path), 'r') as file:
            json_info = json.load(file)
            img_paths = ["1","2","3","4","5","6"]
            for cam in json_info["camera"]:
                if cam['name'] in ["front_middle_camera"]:
                        img_paths[0] = ("/" + cam['oss_path'])
                elif cam['name'] in ["front_right_camera", "rf_wide_camera"]:
                        img_paths[1] = ("/" + cam['oss_path'])
                elif cam['name'] in ["rear_right_camera", "rr_wide_camera"]:
                        img_paths[2] = ("/" + cam['oss_path'])
                elif cam['name'] in ["front_left_camera", "lf_wide_camera"]:
                        img_paths[3] = ("/" + cam['oss_path'])
                elif cam['name'] in ["rear_left_camera", "lr_wide_camera"]:
                        img_paths[4] = ("/" + cam['oss_path'])
                elif cam['name'] in ["rear_middle_camera"]:
                        img_paths[5] = ("/" + cam['oss_path'])
            
            result['images'] = img_paths
            result['json_path'] = json_path
            result['hardware_config'] = "/" + json_info['hardware_config_path']
            
            for lid in json_info["lidar"]:
                if lid['name'] == "center_128_lidar_scan_data":
                    result['point_clouds'] = ["/" + lid['oss_path_pcd_txt']]
            
        #     result['annotation'] = json_info['objects']
            return result
    except FileNotFoundError:
        with open(cantread_path, "a") as cantread_f:
            cantread_f.writelines(json_path + "\n")
    except json.decoder.JSONDecodeError:
            print(json_path )

with ThreadPool(processes = 40) as pool:
    results = list(tqdm(pool.imap(worker, annotations), total=len(annotations), desc='JSON Loading'))
    pool.terminate()
    
# with open (result_path, "w") as result_f:
#     for res in tqdm(results, desc="Data Saving"):
#         result_f.write(json.dumps(res) + '\n')
        
with jsonlines.open(result_path, mode='w') as writer:
    writer.write_all(results)
    