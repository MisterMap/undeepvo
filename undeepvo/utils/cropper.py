import torch


class Cropper:
    @staticmethod
    def make_crop(img, d=64):
        '''
        Make dimensions divisible by `d`
        :param img: source image
        :param d: divide on parameter
        :return: cropped image
        '''
        new_size = (img.shape[0] - img.shape[0] % d,
                    img.shape[1] - img.shape[1] % d)

        bbox = [
            int((img.shape[0] - new_size[0]) / 2),
            int((img.shape[1] - new_size[1]) / 2),
            int((img.shape[0] + new_size[0]) / 2),
            int((img.shape[1] + new_size[1]) / 2),
        ]

        img_cropped = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        return img_cropped