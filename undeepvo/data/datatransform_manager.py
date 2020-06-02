import albumentations

custom_additional_targets = {"image2": "image", "image3": "image", "image4": "image"}


class DataTransformManager:
    def __init__(self, used_img_size, final_img_size):
        self._ratio = max(float(final_img_size[0]) / used_img_size[0], float(final_img_size[0]) / used_img_size[0])
        self._final_img_size = final_img_size

    def get_train_transform(self):
        return albumentations.Compose([
            albumentations.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.2),
            albumentations.RandomGamma(gamma_limit=(90, 110), p=0.2),
            albumentations.ChannelShuffle(p=0.2),
            albumentations.Normalize(),
            albumentations.Downscale(scale_min=self._ratio, scale_max=self._ratio, p=1.0, always_apply=True),
            albumentations.CenterCrop(height=self._final_img_size[0], width=self._final_img_size[1], always_apply=True,
                                      p=1),
        ], additional_targets=custom_additional_targets)

    def get_validation_transform(self):
        return albumentations.Compose([
            albumentations.Normalize(),
            albumentations.Downscale(scale_min=self._ratio, scale_max=self._ratio, p=1.0, always_apply=True),
            albumentations.CenterCrop(height=self._final_img_size[0], width=self._final_img_size[1], always_apply=True,
                                      p=1),
        ], additional_targets=custom_additional_targets)

    def get_test_transform(self):
        return albumentations.Compose([
            albumentations.Normalize(),
            albumentations.Downscale(scale_min=self._ratio, scale_max=self._ratio, p=1.0, always_apply=True),
            albumentations.CenterCrop(height=self._final_img_size[0], width=self._final_img_size[1], always_apply=True,
                                      p=1),
        ], additional_targets=custom_additional_targets)
