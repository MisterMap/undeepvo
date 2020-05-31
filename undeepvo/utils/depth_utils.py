import cv2
import numpy as np


class DepthManager:
    @staticmethod
    def get_depth(frame):
        """
        Get depth map from frame (left and right image) with StereoBM (OpenCV)
        :param frame: frame from kitti dataset (consists left and right image)
        :return: depth map
        """
        left, right = frame
        imgL = cv2.cvtColor(np.array(left), cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(np.array(right), cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=32*4, blockSize=5)
        disparity = stereo.compute(imgL, imgR)
        return disparity