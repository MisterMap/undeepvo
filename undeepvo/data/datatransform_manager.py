import albumentations


class DataTransformManager:
    @staticmethod
    def get_train_transform():
        return albumentations.Compose([
            albumentations.Flip(),
            albumentations.Rotate(7),
            albumentations.RandomBrightnessContrast(limit=(-0.4, 0.4)),
            albumentations.Downscale(), # todo
            albumentations.RandomSizedCrop(min_max_height=(512, 512), width=512, height=512),
            albumentations.RandomGamma(),
            albumentations.ChannelShuffle(),
            albumentations.Normalize()
        ])
    @staticmethod
    def get_validation_transform():
        return albumentations.Compose([
            # TODO resize and downscale
            albumentations.Normalize
        ])

    @staticmethod
    def get_test_transform():
        return albumentations.Compose([
            # TODO resize and downscale
            albumentations.Normalize
        ])
