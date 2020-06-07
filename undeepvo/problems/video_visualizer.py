import cv2
import torch
import numpy as np
from undeepvo.data.supervised import DataTransformManager
from undeepvo.problems import DepthModelEvaluator


class VideoVisualizer:
    def __init__(self, model, input_video_name, output_video_name, depth_video_name):
        self._model = model
        self._input_video_name = input_video_name
        self._output_video_name = output_video_name
        self._depth_video_name = depth_video_name

    def render(self):
        cap = cv2.VideoCapture(self._input_video_name)
        depth_out = cv2.VideoWriter(self._depth_video_name,
                                    cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 60,
                                    (384, 128))
        img_out = cv2.VideoWriter(self._output_video_name,
                                  cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 60,
                                  (384, 128))
        while cap.isOpened():
            ret, img_origin = cap.read()
            if ret:
                gray_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)

                transforms = DataTransformManager(img_origin.shape[:-1], final_img_size=(128, 384),
                                                  transform_params={"filters": True, "normalize": False})
                transforms_norm = transforms.get_test_transform(with_normalize=True)
                transforms_no_norm = transforms.get_test_transform(with_normalize=False)
                transformed = transforms_norm(image=img_origin, mask=gray_origin)
                img = torch.from_numpy(transformed["image"]).permute(2, 0, 1)

                gray = torch.from_numpy(transformed["mask"]).unsqueeze(0)

                evaluator = DepthModelEvaluator(self._model)
                depth = evaluator.get_depth_from_image(img)

                transformed = transforms_no_norm(image=img_origin, mask=gray_origin)
                img = torch.from_numpy(transformed["image"]).permute(2, 0, 1)
                gray = torch.from_numpy(transformed["mask"]).unsqueeze(0)

                depth, gray = evaluator.convert_to_numpy(depth, gray)
                depth = np.clip(1 / depth, 0, 1)
                depth = (depth / (depth.max()) * 255).astype('uint8')
                colored = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)

                cv2.imshow('depth', colored)
                cv2.imshow('img', img.cpu().permute(1, 2, 0).detach().numpy())
                cv2.waitKey(1)
                depth_out.write(colored)
                img_out.write((img.cpu().permute(1, 2, 0).detach().numpy() * 255).astype('uint8'))
        cap.release()
        depth_out.release()
        img_out.release()
        cv2.destroyAllWindows()
