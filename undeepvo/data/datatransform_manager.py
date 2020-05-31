import albumentations


class DataTransformManager:
    @staticmethod
    def get_train_transform():
        return albumentations.Compose([
            albumentations.Flip(p=0.2),
            albumentations.Rotate(10, p=0.3),
            albumentations.RandomSizedCrop(min_max_height=(112, 144), height=128, width=384),
            albumentations.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.2),
            albumentations.RandomGamma(gamma_limit=(90, 110), p=0.2),
            albumentations.ISONoise(color_shift=(-0.1, 0.1), p=0.2),
            albumentations.ChannelShuffle(),
            albumentations.Normalize(),
            albumentations.Resize(height=128, width=384),
        ])

    @staticmethod
    def get_validation_transform():
        return albumentations.Compose([
            albumentations.Normalize(),
            albumentations.Resize(height=128, width=384),
        ])

    @staticmethod
    def get_test_transform():
        return albumentations.Compose([
            albumentations.Normalize(),
            albumentations.Resize(height=128, width=384),
        ])
