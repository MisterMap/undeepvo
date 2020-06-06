import albumentations


class DataTransformManager:
    custom_additional_targets = {"image2": "image", "image3": "image", "image4": "image"}

    def __init__(self, used_img_size, final_img_size, transform_params):
        self._ratio = max(float(final_img_size[0]) / used_img_size[0], float(final_img_size[0]) / used_img_size[0])
        self._final_img_size = final_img_size
        self._scale_compose = [
            albumentations.Resize(height=int(used_img_size[0] * self._ratio), width=int(used_img_size[1] * self._ratio),
                                  always_apply=True),
            albumentations.CenterCrop(height=self._final_img_size[0], width=self._final_img_size[1], always_apply=True,
                                      p=1),
        ]
        self._normalize_transform = albumentations.Normalize()
        self._normalize_no_transform = albumentations.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
        self._train_compose = self._scale_compose
        if "flip" in transform_params:
            if transform_params["flip"]:
                self._train_compose = [
                                          albumentations.HorizontalFlip()
                                      ] + self._train_compose
        if "filters" in transform_params:
            if transform_params["filters"]:
                self._train_compose = [
                                          albumentations.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2),
                                                                                  contrast_limit=(-0.2, 0.2), p=0.5),
                                          albumentations.RandomGamma(gamma_limit=(90, 110), p=0.5),
                                          albumentations.ChannelShuffle(p=0.5),
                                      ] + self._train_compose

        if transform_params["normalize"]:
            self._train_compose.append(albumentations.Normalize())
        else:
            self._train_compose.append(albumentations.Normalize(mean=(0, 0, 0), std=(1, 1, 1)))

    def get_train_transform(self):
        return albumentations.Compose(self._train_compose, additional_targets=self.custom_additional_targets)

    def get_validation_transform(self, with_resize=True, with_normalize=True):
        scale_compose = self._scale_compose if with_resize else []
        return albumentations.Compose(scale_compose + self.get_normalize(with_normalize),
                                      additional_targets=self.custom_additional_targets)

    def get_test_transform(self, with_normalize=True):
        return albumentations.Compose(self._scale_compose + self.get_normalize(with_normalize),
                                      additional_targets=self.custom_additional_targets)

    def get_normalize(self, with_normalize=True):
        if with_normalize:
            return [self._normalize_transform]
        else:
            return [self._normalize_no_transform]

    def get_normalize_transform(self, with_normalize=True):
        return albumentations.Compose(self.get_normalize(with_normalize))
