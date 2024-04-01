from animate import MagicAnimate
import cv2
import torch 

model = MagicAnimate()


source_image = cv2.imread('data/images_data/0.jpeg')
control = cv2.imread('data/pose_data/0.jpeg')
# control = torch.from_numpy(control)
control = control[None, :]

out = model.infer_for_image(source_image, control)

cv2.imwrite('./data/out.jpeg', out*255)