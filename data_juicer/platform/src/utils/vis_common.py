import os
import glob
import click
import json
import open3d as o3d
from loguru import logger
import numpy as np
import sys
sys.path.append(os.path.dirname(__file__))
print(sys.path)
from lidar2bev import Lidar2BEV
import cv2

ToBev = Lidar2BEV()

model_palette = [[128, 64, 128],  # road
                 [244, 35, 232],  # sidewalk
                 [70, 70, 70],  # building
                 [102, 102, 156],  # wall
                 [190, 153, 153],  # fence
                 [153, 153, 153],  # pole
                 [250, 170, 30],  # traffic light
                 [220, 220, 0],  # traffic sign
                 [107, 142, 35],  # vegetation
                 [152, 251, 152],  # terrain
                 [70, 130, 180],  # sky
                 [220, 20, 60],  # person
                 [0, 0, 155],  # rider
                 [0, 0, 155],  # car
                 [220, 20, 60],  # truck
                 [220, 20, 60],  # bus
                 [0, 80, 100],  # train
                 [0, 0, 155],  # motorcycle
                 [0, 0, 155]]  # bicycle


def center_box_to_corners(box):
    pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, yaw = box
    half_dim_x, half_dim_y, half_dim_z = dim_x / 2.0, dim_y / 2.0, dim_z / 2.0
    corners = np.array([[half_dim_x, half_dim_y, -half_dim_z],
                        [half_dim_x, -half_dim_y, -half_dim_z],
                        [-half_dim_x, -half_dim_y, -half_dim_z],
                        [-half_dim_x, half_dim_y, -half_dim_z],
                        [half_dim_x, half_dim_y, half_dim_z],
                        [half_dim_x, -half_dim_y, half_dim_z],
                        [-half_dim_x, -half_dim_y, half_dim_z],
                        [-half_dim_x, half_dim_y, half_dim_z]])
    # 这个时候corners还只是平行于坐标轴且以坐标原点为中心来算的.
    transform_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, pos_x],
        [np.sin(yaw), np.cos(yaw), 0, pos_y],
        [0, 0, 1.0, pos_z],
        [0, 0, 0, 1.0],
    ])
    # 然后根据pose,算出真实的,即RX+T
    corners = (transform_matrix[:3, :3] @
               corners.T + transform_matrix[:3, [3]]).T
    return corners


def parse_obj_get_box(obj):
    """
    haomo's obj
    """
    if '3D_attributes' in obj:
        obj = obj["3D_attributes"]
    if len(obj) == 0:
        return None
    length, width, height = obj["dimension"]["length"], obj["dimension"]["width"], obj["dimension"]["height"]
    cx, cy, cz = obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]
    rot_y = obj["rotation"]["yaw"]
    return (cx, cy, cz, length, width, height, rot_y)


def get_lidar_p(json_data):
    try:
        if 'point_cloud_Url' in json_data:
            return '/' + json_data["point_cloud_Url"]
        if 'lidar' in json_data:
            return '/' + json_data["lidar"][0]["oss_path_pcd_txt"]
        elif 'relative_sensors_data' in json_data:
            return '/' + json_data['relative_sensors_data'][0]['point_cloud_url']
    except:
        return None


# @logger.catch
# @click.command()
# @click.option('--hds', default="/root/dataset/test_vis.hds", type=str)
# @click.option('--out', default="./", type=str)

def vis_annotation(category, annotation_path, out='./'):
    if category == 'icu30_2d_3d_key_point_clip':
        return vis_icu30_2d_3d_box_1001(annotation_path, out)
    elif category == 'icu30_2d_3d_box_1001':
        return vis_icu30_2d_3d_box_1001(annotation_path, out)
    elif category == 'hd_3d_box_1002' or category == 'hd_3d_box_1001':
        return vis_hd_3d_box_1002(annotation_path, out)
    else:
        raise ValueError("Not implemented yet.")

def vis_icu30_2d_3d_box_1001(json_annotation_path, out='./image_vis'):
    # logger.info(hds, out)
    prev_bevimg = None
    json_annotation_path = os.path.join('/', json_annotation_path)
    json_data = json.loads(open(json_annotation_path).read())
    p = get_lidar_p(json_data)
    if p is not None:
        pcd = o3d.io.read_point_cloud(p)
        points = np.array(pcd.points)
        o3d_point_cloud = o3d.geometry.PointCloud()
        o3d_point_cloud.points = o3d.utility.Vector3dVector(points)
        o3d_point_cloud = o3d_point_cloud.paint_uniform_color([0, 0.206, 0])
        pixel = ToBev.points_to_pixel(points, ToBev.resolution)

        points = np.concatenate(
            (points, np.asarray(o3d_point_cloud.colors).reshape(-1, 3)), axis=1)
        bevimg = ToBev.get_bev_img(points, pixel)
        bevimg = bevimg[..., ::-1].astype("uint8")
        prev_bevimg = bevimg
    else:
        bevimg = np.zeros_like(prev_bevimg)
    print("AAAAA %s" % str(bevimg.shape))
    if "objects" in json_data:
        for obj in json_data["objects"]:
            box = parse_obj_get_box(obj)
            if box is None:
                continue
            corner = center_box_to_corners(box)
            corner = corner[:4, :2]
            corner = ToBev.points_to_pixel(corner, ToBev.resolution)
            cv2.polylines(bevimg, [corner], True, (0, 255, 255), 2)
            # 打上id.
            cx = int(corner[:, 0].mean())
            cy = int(corner[:, 1].mean())
            # cv2.putText(bevimg, str(obj["id"]), corner[0],
            cv2.putText(bevimg, str(obj["id"]), (cx, cy),
                        cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
            # 画上朝向箭头.
            tx = int(corner[:2, 0].mean())
            ty = int(corner[:2, 1].mean())
            cv2.arrowedLine(bevimg, (cx, cy), (tx, ty),
                            (255, 0, 0), 2, 0, 0, 0.2)

    keys = ['imgUrl', 'imageUrl', 'image']
    orientations = ['image_orientation', 'camera_orientation', 'camere_orientation', 'name']
    relative_sensors = ['relative_sensors_data', 'relative_images_data', 'images']
    relative_sensor = next((key for img_info in json_data for key in relative_sensors if key in img_info), '')
    image_path = next((key for img_info in json_data[relative_sensor] for key in keys if key in img_info), '')
    orientation = next((key for img_info in json_data[relative_sensor] for key in orientations if key in img_info), '')
    # 左前
    for img_info in json_data[relative_sensor]:
        if img_info.get(orientation, None) == "front_left_camera":
            break
    fm_img_front_left = cv2.imread("/" + img_info[image_path])
    for obj in img_info.get("objects", []):
        x1, y1, w, h = obj["bbox"]
        fm_img_front_left = cv2.rectangle(fm_img_front_left, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                                            model_palette[int(obj["id"]) % len(model_palette)], 10)
        fm_img_front_left = cv2.putText(fm_img_front_left, str(obj["id"]), (int(x1), int(y1)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # 右前
    for img_info in json_data[relative_sensor]:
        if img_info.get(orientation, None) == "front_right_camera":
            break
    fm_img_front_right = cv2.imread("/" + img_info[image_path])
    for obj in img_info.get("objects", []):
        x1, y1, w, h = obj["bbox"]
        fm_img_front_right = cv2.rectangle(fm_img_front_right, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                                            model_palette[int(obj["id"]) % len(model_palette)], 10)
        fm_img_front_right = cv2.putText(fm_img_front_right, str(obj["id"]), (int(x1), int(y1)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # 左后
    for img_info in json_data[relative_sensor]:
        if img_info.get(orientation, None) == "rear_left_camera":
            break
    fm_img_rear_left = cv2.imread("/" + img_info[image_path])
    for obj in img_info.get("objects", []):
        x1, y1, w, h = obj["bbox"]
        fm_img_rear_left = cv2.rectangle(fm_img_rear_left, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                                            model_palette[int(obj["id"]) % len(model_palette)], 10)
        fm_img_rear_left = cv2.putText(fm_img_rear_left, str(obj["id"]), (int(x1), int(y1)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # 右后
    for img_info in json_data[relative_sensor]:
        if img_info.get(orientation, None) == "rear_right_camera":
            break
    fm_img_rear_right = cv2.imread("/" + img_info[image_path])
    for obj in img_info.get("objects", []):
        x1, y1, w, h = obj["bbox"]
        fm_img_rear_right = cv2.rectangle(fm_img_rear_right, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                                            model_palette[int(obj["id"]) % len(model_palette)], 10)
        fm_img_rear_right = cv2.putText(fm_img_rear_right, str(obj["id"]), (int(x1), int(y1)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # 正前
    for img_info in json_data[relative_sensor]:
        if img_info.get(orientation, None) == "front_middle_camera":
            break
    fm_img_front_middle = cv2.imread("/" + img_info[image_path])
    for obj in img_info.get("objects", []):
        x1, y1, w, h = obj["bbox"]
        fm_img_front_middle = cv2.rectangle(fm_img_front_middle, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                                            model_palette[int(obj["id"]) % len(model_palette)], 10)
        fm_img_front_middle = cv2.putText(fm_img_front_middle, str(obj["id"]), (int(x1), int(y1)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # 正后
    for img_info in json_data[relative_sensor]:
        if img_info.get(orientation, None) == "rear_middle_camera":
            break
    fm_img_rear_middle = cv2.imread("/" + img_info[image_path])
    for obj in img_info.get("objects", []):
        x1, y1, w, h = obj["bbox"]
        fm_img_rear_middle = cv2.rectangle(fm_img_rear_middle, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                                            model_palette[int(obj["id"]) % len(model_palette)], 10)
        fm_img_rear_middle = cv2.putText(fm_img_rear_middle, str(obj["id"]), (int(x1), int(y1)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # fm_img_front_middle = cv2.resize(fm_img_front_middle,
    #                                  (fm_img_front_middle.shape[1] // 2, fm_img_front_middle.shape[0] // 2))
    # fm_img_rear_middle = cv2.resize(fm_img_rear_middle,
    #                                 (fm_img_rear_middle.shape[1] // 2, fm_img_rear_middle.shape[0] // 2))
    image_1 = np.concatenate((fm_img_front_middle, fm_img_rear_middle), axis=1)
    image_2 = np.concatenate((fm_img_front_left, fm_img_front_right), axis=1)
    image_3 = np.concatenate((fm_img_rear_left, fm_img_rear_right), axis=1)
    fm_img = np.concatenate((image_1, image_2, image_3), axis=0)

    img_path = os.path.abspath(os.path.join(out, os.path.basename(json_annotation_path).replace("json", "jpg")))
    if bevimg.shape[0] - fm_img.shape[0] > 0:
        fill_x = np.zeros((int((bevimg.shape[0] - fm_img.shape[0]) / 2), fm_img.shape[1], 3))
        fm_img = np.concatenate((fm_img, fill_x), axis=0)
        fm_img = np.concatenate((fill_x, fm_img), axis=0)
        final = np.concatenate((fm_img, bevimg), axis=1)
    else:
        fill_x = np.zeros((int((fm_img.shape[0] - bevimg.shape[0]) / 2), bevimg.shape[1], 3))
        bevimg = np.concatenate((bevimg, fill_x), axis=0)
        bevimg = np.concatenate((fill_x, bevimg), axis=0)
        final = np.concatenate((fm_img, bevimg), axis=1)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    cv2.imwrite(img_path, final)
    return img_path

def vis_hd_3d_box_1002(json_annotation_path, out='./image_vis'):
    # logger.info(hds, out)
    prev_bevimg = None
    json_annotation_path = os.path.join('/', json_annotation_path)
    json_data = json.loads(open(json_annotation_path).read())
    p = get_lidar_p(json_data)
    print(json_annotation_path)
    if p is not None:
        pcd = o3d.io.read_point_cloud(p)
        points = np.array(pcd.points)
        o3d_point_cloud = o3d.geometry.PointCloud()
        o3d_point_cloud.points = o3d.utility.Vector3dVector(points)
        o3d_point_cloud = o3d_point_cloud.paint_uniform_color([0, 0.206, 0])
        pixel = ToBev.points_to_pixel(points, ToBev.resolution)

        points = np.concatenate(
            (points, np.asarray(o3d_point_cloud.colors).reshape(-1, 3)), axis=1)
        bevimg = ToBev.get_bev_img(points, pixel)
        bevimg = bevimg[..., ::-1].astype("uint8")
        prev_bevimg = bevimg
    else:
        bevimg = np.zeros_like(prev_bevimg)
    print("AAAAA %s" % str(bevimg.shape))
    if "objects" in json_data:
        for obj in json_data["objects"]:
            box = parse_obj_get_box(obj)
            if box is None:
                continue
            corner = center_box_to_corners(box)
            corner = corner[:4, :2]
            corner = ToBev.points_to_pixel(corner, ToBev.resolution)
            cv2.polylines(bevimg, [corner], True, (0, 255, 255), 2)
            # 打上id.
            cx = int(corner[:, 0].mean())
            cy = int(corner[:, 1].mean())
            # cv2.putText(bevimg, str(obj["id"]), corner[0],
            cv2.putText(bevimg, str(obj["id"]), (cx, cy),
                        cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
            # 画上朝向箭头.
            tx = int(corner[:2, 0].mean())
            ty = int(corner[:2, 1].mean())
            cv2.arrowedLine(bevimg, (cx, cy), (tx, ty),
                            (255, 0, 0), 2, 0, 0, 0.2)

    keys = ['imgUrl', 'imageUrl', 'image', 'oss_path']
    orientations = ['image_orientation', 'camera_orientation', 'camere_orientation', 'name']
    relative_sensors = ['relative_sensors_data', 'relative_images_data', 'images', 'camera']
    relative_sensor = next((key for img_info in json_data for key in relative_sensors if key in img_info), '')
    image_path = next((key for img_info in json_data[relative_sensor] for key in keys if key in img_info), '')
    orientation = next((key for img_info in json_data[relative_sensor] for key in orientations if key in img_info), '')
    # 左前
    for img_info in json_data[relative_sensor]:
        if img_info.get(orientation, None) == "left_fisheye_camera_record" or img_info.get(orientation, None) == "left_fisheye_camera":
            break
    fm_img_front_left = cv2.imread("/" + img_info[image_path])
    for obj in img_info.get("objects", []):
        x1, y1, w, h = obj["bbox"]
        fm_img_front_left = cv2.rectangle(fm_img_front_left, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                                            model_palette[int(obj["id"]) % len(model_palette)], 10)
        fm_img_front_left = cv2.putText(fm_img_front_left, str(obj["id"]), (int(x1), int(y1)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # 右前
    for img_info in json_data[relative_sensor]:
        if img_info.get(orientation, None) == "right_fisheye_camera_record" or img_info.get(orientation, None) == "right_fisheye_camera":
            break
    fm_img_front_right = cv2.imread("/" + img_info[image_path])
    for obj in img_info.get("objects", []):
        x1, y1, w, h = obj["bbox"]
        fm_img_front_right = cv2.rectangle(fm_img_front_right, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                                            model_palette[int(obj["id"]) % len(model_palette)], 10)
        fm_img_front_right = cv2.putText(fm_img_front_right, str(obj["id"]), (int(x1), int(y1)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # 左后
    for img_info in json_data[relative_sensor]:
        if img_info.get(orientation, None) == "front_long_camera_record" or img_info.get(orientation, None) == "front_long_camera":
            break
    fm_img_rear_left = cv2.imread("/" + img_info[image_path])
    for obj in img_info.get("objects", []):
        x1, y1, w, h = obj["bbox"]
        fm_img_rear_left = cv2.rectangle(fm_img_rear_left, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                                            model_palette[int(obj["id"]) % len(model_palette)], 10)
        fm_img_rear_left = cv2.putText(fm_img_rear_left, str(obj["id"]), (int(x1), int(y1)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # 右后
    for img_info in json_data[relative_sensor]:
        if img_info.get(orientation, None) == "front_fisheye_camera_record" or img_info.get(orientation, None) == "front_fisheye_camera":
            break
    fm_img_rear_right = cv2.imread("/" + img_info[image_path])
    for obj in img_info.get("objects", []):
        x1, y1, w, h = obj["bbox"]
        fm_img_rear_right = cv2.rectangle(fm_img_rear_right, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                                            model_palette[int(obj["id"]) % len(model_palette)], 10)
        fm_img_rear_right = cv2.putText(fm_img_rear_right, str(obj["id"]), (int(x1), int(y1)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # 正前
    for img_info in json_data[relative_sensor]:
        if img_info.get(orientation, None) == "front_wide_camera_record" or img_info.get(orientation, None) == "front_wide_camera":
            break
    fm_img_front_middle = cv2.imread("/" + img_info[image_path])
    for obj in img_info.get("objects", []):
        x1, y1, w, h = obj["bbox"]
        fm_img_front_middle = cv2.rectangle(fm_img_front_middle, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                                            model_palette[int(obj["id"]) % len(model_palette)], 10)
        fm_img_front_middle = cv2.putText(fm_img_front_middle, str(obj["id"]), (int(x1), int(y1)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # 正后
    for img_info in json_data[relative_sensor]:
        if img_info.get(orientation, None) == "rear_fisheye_camera_record" or img_info.get(orientation, None) == "rear_fisheye_camera":
            break
    fm_img_rear_middle = cv2.imread("/" + img_info[image_path])
    for obj in img_info.get("objects", []):
        x1, y1, w, h = obj["bbox"]
        fm_img_rear_middle = cv2.rectangle(fm_img_rear_middle, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)),
                                            model_palette[int(obj["id"]) % len(model_palette)], 10)
        fm_img_rear_middle = cv2.putText(fm_img_rear_middle, str(obj["id"]), (int(x1), int(y1)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # fm_img_front_middle = cv2.resize(fm_img_front_middle,
    #                                  (fm_img_front_middle.shape[1] // 2, fm_img_front_middle.shape[0] // 2))
    # fm_img_rear_middle = cv2.resize(fm_img_rear_middle,
    #                                 (fm_img_rear_middle.shape[1] // 2, fm_img_rear_middle.shape[0] // 2))

    fm_img_rear_middle = cv2.resize(fm_img_rear_middle, (fm_img_front_middle.shape[1], fm_img_front_middle.shape[0]))
    fm_img_front_right = cv2.resize(fm_img_front_right, (fm_img_front_left.shape[1], fm_img_front_left.shape[0]))
    fm_img_rear_right = cv2.resize(fm_img_rear_right, (fm_img_rear_left.shape[1], fm_img_rear_left.shape[0]))

    image_1 = np.concatenate((fm_img_front_middle, fm_img_rear_middle), axis=1)
    image_2 = np.concatenate((fm_img_front_left, fm_img_front_right), axis=1)
    image_3 = np.concatenate((fm_img_rear_left, fm_img_rear_right), axis=1)
    fm_img = np.concatenate((image_1, image_2, image_3), axis=0)

    img_path = os.path.abspath(os.path.join(out, os.path.basename(json_annotation_path).replace("json", "jpg")))
    if bevimg.shape[0] - fm_img.shape[0] > 0:
        fill_x = np.zeros((int((bevimg.shape[0] - fm_img.shape[0]) / 2), fm_img.shape[1], 3))
        fm_img = np.concatenate((fm_img, fill_x), axis=0)
        fm_img = np.concatenate((fill_x, fm_img), axis=0)
        final = np.concatenate((fm_img, bevimg), axis=1)
    else:
        fill_x = np.zeros((int((fm_img.shape[0] - bevimg.shape[0]) / 2), bevimg.shape[1], 3))
        bevimg = np.concatenate((bevimg, fill_x), axis=0)
        bevimg = np.concatenate((fill_x, bevimg), axis=0)
        final = np.concatenate((fm_img, bevimg), axis=1)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    cv2.imwrite(img_path, final)
    return img_path

if __name__ == '__main__':
    main()