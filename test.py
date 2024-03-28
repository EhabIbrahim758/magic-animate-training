from animate import MagicAnimate
import cv2
import torch 

model = MagicAnimate()


source_image = cv2.resize(cv2.imread('data/images_data/0.jpeg'), (512, 512))
control = cv2.resize(cv2.imread('data/pose_data/0.jpeg'), (512, 512))
control = torch.from_numpy(control)

model.infer_for_image(source_image, control)
