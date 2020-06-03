import torch
import kornia


def generate_transformation(translation, rotation):
    rotation_matrix = kornia.geometry.angle_axis_to_rotation_matrix(rotation)  # current transformation matrix
    translation = translation.unsqueeze(dim=2)
    transform_matrix = torch.cat((rotation_matrix, translation), dim=2)
    tmp = torch.tensor([[0, 0, 0, 1]] * translation.shape[0], dtype=torch.float).to(translation.device)
    tmp = tmp.unsqueeze(dim=1)
    transform_matrix = torch.cat((transform_matrix, tmp), dim=1)
    return transform_matrix


def generate_relative_transformation(src_translation, src_rotation, dst_translation, dst_rotation):
    src_transform = generate_transformation(src_translation, src_rotation)
    dst_transform = generate_transformation(dst_translation, dst_rotation)
    return kornia.geometry.relative_transformation(src_transform, dst_transform)
