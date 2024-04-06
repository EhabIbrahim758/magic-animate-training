# from animate import MagicAnimate
# import cv2
# import torch 

# model = MagicAnimate()


# source_image = cv2.imread('data/images_data/10.jpeg')
# control = cv2.imread('data/pose_data/1.jpeg')
# # control = torch.from_numpy(control)
# control = control[None, :]

# out = model.infer_for_image(source_image, control)

# cv2.imwrite('./data/out.jpeg', out*255)

from animate import MagicAnimate
import cv2
import torch 
import os 



model = MagicAnimate()

def pred(src_image, control):
    control = control[None, :]
    out = model.infer_for_image(src_image, control)
    return out

def read_image(path):
    return cv2.imread(path)

image_folder_path = './data/images_data'
pose_folder_path = './data/pose_data'
gen_folder_path = './data/generated_data'

if __name__ == '__main__':
    images = sorted(os.listdir(image_folder_path))
    poses = sorted(os.listdir(pose_folder_path))
    
    for i in range(50):
        image_path = os.path.join(image_folder_path, images[i]) 
        pose_path = os.path.join(pose_folder_path, poses[i])
        src_image = read_image(image_path)
        control = read_image(pose_path)
        
        generated = pred(src_image, control)
        cv2.imwrite(os.path.join(f'./data/generated_data/{i}.jpeg'), generated*255)