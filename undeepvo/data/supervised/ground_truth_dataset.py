import os
import wget
import zipfile
import cv2


# from PIL import Image


class GroundTruthDataset(object):
    def __init__(self, main_folder='depth_selection', length=600, velodyne=False):
        self._path_to_dataset = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip'
        self._main_folder = main_folder
        self._velodyne = velodyne
        self._names = None
        self._images_folder = None
        self._intrinsics_folder = None
        self._ground_truth_depth_folder = None
        self._ground_truth_velodyne_folder = None
        self._length = None
        self.download_dataset()
        assert length <= self.get_length(), "Final length is more then available"
        self._length = length


    def _dataset_exists(self):
        return os.path.exists("./{}/".format(self._main_folder))

    def get_names(self):
        return self._names[-3:-1]

    def download_dataset(self):
        if not self._dataset_exists():
            print("Download dataset")
            wget.download(self._path_to_dataset)
            with zipfile.ZipFile(self._path_to_dataset.split('/')[-1], 'r') as zip_ref:
                zip_ref.extractall()
        self._names = [el.split(".")[0] for el in
                       os.listdir(os.path.join(os.curdir, self._main_folder, 'val_selection_cropped', 'image'))]
        self._images_folder = os.path.join(os.curdir, self._main_folder, 'val_selection_cropped', 'image')
        self._ground_truth_depth_folder = os.path.join(os.curdir, self._main_folder, 'val_selection_cropped',
                                                       'groundtruth_depth')
        self._ground_truth_velodyne_folder = os.path.join(os.curdir, self._main_folder, 'val_selection_cropped',
                                                          'velodyne_raw')
        self._length = len(self._names)

    def get_length(self):
        return self._length

    def get_image(self, idx):
        assert idx < self._length, "Out of dataset"
        image_path = os.path.join(self._images_folder, f'{self._names[idx]}.png')
        img = cv2.imread(image_path)
        return img

    def get_image_size(self):
        depth = self.get_depth(0)
        return depth.shape

    def get_depth(self, idx):
        assert idx < self._length, "Out of dataset"
        if self._velodyne:
            name_for_depth = '{}velodyne_raw{}image{}'.format(self._names[idx].split("image")[0],
                                                              self._names[idx].split("image")[1],
                                                              self._names[idx].split("image")[2])
            depth_path = os.path.join(self._ground_truth_velodyne_folder, f'{name_for_depth}.png')
        else:
            name_for_depth = '{}groundtruth_depth{}image{}'.format(self._names[idx].split("image")[0],
                                                                   self._names[idx].split("image")[1],
                                                                   self._names[idx].split("image")[2])
            depth_path = os.path.join(self._ground_truth_depth_folder, f'{name_for_depth}.png')
        img = cv2.imread(depth_path, 0)
        return img
