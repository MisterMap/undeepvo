import albumentations


class DataTransformManager:
    @staticmethod
    def get_train_transform():
        return albumentations.Compose([
            albumentations.Flip(p=0.2),
            albumentations.Rotate(7, p=0.3),
            albumentations.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.4),
            albumentations.RandomSizedCrop(min_max_height=(112, 144), height=128, width=384),
            albumentations.RandomGamma(),
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
