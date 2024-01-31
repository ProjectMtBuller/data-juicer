# ===========================================
# --coding:UTF-8 --
# file: lidar2bev.py
# author: baobo
# date: 2022-09-25
# ===========================================

import click
import numpy as np
from loguru import logger


class Lidar2BEV(object):
    def __init__(self, resolution=0.04, vis_range=[40, -40, 100, -100]):
    # def __init__(self, resolution=0.10, vis_range=[-60, 60, -70, 70]):
    # def __init__(self, resolution=0.10, vis_range=[-60, 60, -100, 100]):
        """
        vis_range: [左, 右, 前，后]
        resolution: 单位m.
        """
        self.resolution = resolution
        self.vis_range = vis_range
        self.image_width = int((abs(self.vis_range[1]-self.vis_range[0])) // resolution) + 1
        self.image_height = int((abs(self.vis_range[3]-self.vis_range[2])) // resolution) + 1
        print("W,H: ", self.image_width, self.image_height)
        # 标注范围.
        self.range1 = np.asarray([-10, 10, -135, 66], "int32") * int(1/resolution)
        self.range2 = np.asarray([-30, 30, -135, 66], "int32") * int(1/resolution)
        # 拥挤场景.
        self.range3 = np.asarray([-10, 10, -40, 20], "int32") * int(1/resolution)

    def points_to_pixel(self, data, resolution):
        """
        即lidar坐标系转到图像坐标系.
        """
        # 自己车的数据是x向右. y向上.
        # u = (-data[:, 0] / resolution).astype("int32")
        # v = (data[:, 1] / resolution).astype("int32")
        u = (-data[:, 1] / resolution).astype("int32")
        v = (-data[:, 0] / resolution).astype("int32")
        # 再将左上角转化为0,0
        umin = int(abs(self.vis_range[0] / resolution))
        vmin = int(abs(self.vis_range[2] / resolution))
        u += umin
        v += vmin
        pixel = np.stack((u, v)).T
        return pixel

    def get_bev_img(self, lidar_data, pixel):
        """
        lidar_data: nx6
        """
        image_height = self.image_height
        image_width = self.image_width
        image = np.zeros((image_height, image_width, 3))
        mask_u = np.logical_and(pixel[:, 0] >= 0, pixel[:, 0] < image_width)
        mask_v = np.logical_and(pixel[:, 1] >= 0, pixel[:, 1] < image_height)
        mask = np.logical_and(mask_u, mask_v)

        pixel = pixel[mask]
        color = lidar_data[mask][:, -3:]
        # TODO 这里应该有快速的办法.
        vs = pixel[:, 1]
        us = pixel[:, 0]
        # image[vs, us] = [255, 255, 0]
        image[vs, us] = [255, 0, 0]
        # image[vs, us] = color * 255
        # for i, (u, v) in enumerate(pixel):
        #     image[v, u] = color[i] * 255  # 因为lidar里面存的是0,1之间.
        image = image.astype("uint8")
        return image

    def get_all_imgs(self):
        images = []
        for lidar_data in self.lidar_data:
            pixel = self.points_to_pixel(lidar_data, self.resolution)
            image = self.get_bev_img(lidar_data, pixel)
            images.append(image)
        return images


@logger.catch
@click.command()
@click.option('--root', default=None, type=str)
def main(root):
    logger.info(root)


if __name__ == '__main__':
    main()