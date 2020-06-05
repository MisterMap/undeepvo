import cv2
import os
import wget
import zipfile
import numpy as np


class Groundtruth_data():
    def __init__(self, download=False, main_folder='depth_selection'):
        if download:
            wget.download('https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip')
            with zipfile.ZipFile('data_depth_selection.zip', 'r') as zip_ref:
                zip_ref.extractall()

        self.names = [el.split(".")[0] for el in os.listdir(os.path.join(os.curdir, main_folder, 'val_selection_cropped', 'image'))]
        self.images_folder = os.path.join(os.curdir, main_folder, 'val_selection_cropped', 'image')
        self.intrinsics_folder = os.path.join(os.curdir, main_folder, 'val_selection_cropped', 'intrinsics')
        self.groundtruth_depth_folder = os.path.join(os.curdir, main_folder, 'val_selection_cropped', 'groundtruth_depth')

    def get_item(self, id):

        name_for_depth = f'{self.names[id].split("image")[0]}groundtruth_depth{self.names[id].split("image")[1]}image{self.names[id].split("image")[2]}'
        image_path = os.path.join(self.images_folder, f'{self.names[id]}.png')
        intrinsic_path = os.path.join(self.intrinsics_folder, f'{self.names[id]}.txt')
        groundtruth_depth_path = os.path.join(self.groundtruth_depth_folder, f'{name_for_depth}.png')
        groundtruth_dict = {'image': cv2.imread(image_path), 'groundtruth_depth': cv2.imread(groundtruth_depth_path), 'intrinsic': np.loadtxt(intrinsic_path)}
        return groundtruth_dict

if __name__ == "__main__":
    a = Groundtruth_data()
    a.get_item(6)