import torch


class ResultDataPoint(object):
    def __init__(self, input_image, reference_image):
        self.input_image = input_image
        self.reference_image = reference_image
        self.depth = None
        self.rotation = None
        self.translation = None

    def apply_model(self, model):
        depth, pose = model(self.normalize(self.input_image), self.normalize(self.reference_image))
        self.depth = depth
        self.rotation = pose[0]
        self.translation = pose[1]
        return self

    @staticmethod
    def normalize(tensor):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.as_tensor(mean, device=tensor.device)[None, :, None, None]
        std = torch.as_tensor(std, device=tensor.device)[None, :, None, None]
        tensor = tensor.sub(mean).div(std)
        return tensor

    def update_pose(self, translation, rotation):
        self.translation = translation
        self.rotation = rotation

