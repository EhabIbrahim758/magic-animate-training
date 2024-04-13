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

def pred(src_image, control, src_image_latent=True):
    control = control[None, :]
    src_image_copy = torch.from_numpy(src_image).permute(2, 0, 1).to('cuda', dtype=torch.float16)
    src_image_copy = src_image_copy.unsqueeze(dim = 0)
    
    if src_image_latent:
        with torch.no_grad():
            latents = model.vae.encode(src_image_copy).latent_dist
            latents = latents.sample()
            latents = latents * 0.18215
    out = model.infer_for_image(src_image, control, latents)
    return out


# video_length = pixel_values.shape[1]




def read_image(path):
    return cv2.imread(path)

image_folder_path = './data/ronaldo_images'
pose_folder_path = './data/ronaldo_poses'
gen_folder_path = './data/generated_data'

if __name__ == '__main__':
    images = sorted(os.listdir(image_folder_path))
    poses = sorted(os.listdir(pose_folder_path))
    
    for i in range(50):
        image_path = os.path.join(image_folder_path, images[0]) 
        pose_path = os.path.join(pose_folder_path, poses[i])
        src_image = read_image(image_path)
        control = read_image(pose_path)
        
        generated = pred(src_image, control)
        cv2.imwrite(os.path.join(f'./data/generated_data/{i}.jpeg'), generated*255)
        

# UNet3DConditionModel(
#   (conv_in): InflatedConv3d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

#   (time_proj): Timesteps()
#   (time_embedding): TimestepEmbedding(
#     (linear_1): Linear(in_features=320, out_features=1280, bias=True)
#     (act): SiLU()
#     (linear_2): Linear(in_features=1280, out_features=1280, bias=True)
#   )

#   (down_blocks): ModuleList(
#     (0): CrossAttnDownBlock3D(
#       (attentions): ModuleList(
#         (0-1): 2 x Transformer3DModel(
#           (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
#           (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): Attention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=320, out_features=320, bias=False)
#                 (to_v): Linear(in_features=320, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=768, out_features=320, bias=False)
#                 (to_v): Linear(in_features=768, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=320, out_features=2560, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=1280, out_features=320, bias=True)
#                 )
#               )
#               (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0-1): 2 x ResnetBlock3D(
#           (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (conv1): InflatedConv3d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
#           (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): InflatedConv3d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#       (motion_modules): ModuleList(
#         (0-1): 2 x None
#       )
#       (downsamplers): ModuleList(
#         (0): Downsample3D(
#           (conv): InflatedConv3d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         )
#       )
#     )
#     (1): CrossAttnDownBlock3D(
#       (attentions): ModuleList(
#         (0-1): 2 x Transformer3DModel(
#           (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#           (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): Attention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=640, out_features=640, bias=False)
#                 (to_v): Linear(in_features=640, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=768, out_features=640, bias=False)
#                 (to_v): Linear(in_features=768, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=640, out_features=5120, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=2560, out_features=640, bias=True)
#                 )
#               )
#               (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock3D(
#           (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (conv1): InflatedConv3d(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): InflatedConv3d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): InflatedConv3d(320, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock3D(
#           (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (conv1): InflatedConv3d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): InflatedConv3d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#       (motion_modules): ModuleList(
#         (0-1): 2 x None
#       )
#       (downsamplers): ModuleList(
#         (0): Downsample3D(
#           (conv): InflatedConv3d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         )
#       )
#     )
#     (2): CrossAttnDownBlock3D(
#       (attentions): ModuleList(
#         (0-1): 2 x Transformer3DModel(
#           (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#           (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): Attention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=1280, out_features=10240, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=5120, out_features=1280, bias=True)
#                 )
#               )
#               (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock3D(
#           (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (conv1): InflatedConv3d(640, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): InflatedConv3d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): InflatedConv3d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock3D(
#           (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (conv1): InflatedConv3d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): InflatedConv3d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#       (motion_modules): ModuleList(
#         (0-1): 2 x None
#       )
#       (downsamplers): ModuleList(
#         (0): Downsample3D(
#           (conv): InflatedConv3d(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         )
#       )
#     )
#     (3): DownBlock3D(
#       (resnets): ModuleList(
#         (0-1): 2 x ResnetBlock3D(
#           (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (conv1): InflatedConv3d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): InflatedConv3d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#       (motion_modules): ModuleList(
#         (0-1): 2 x None
#       )
#     )
#   )
#   (up_blocks): ModuleList(
#     (0): UpBlock3D(
#       (resnets): ModuleList(
#         (0-2): 3 x ResnetBlock3D(
#           (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
#           (conv1): InflatedConv3d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): InflatedConv3d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): InflatedConv3d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (motion_modules): ModuleList(
#         (0-2): 3 x None
#       )
#       (upsamplers): ModuleList(
#         (0): Upsample3D(
#           (conv): InflatedConv3d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#     )
#     (1): CrossAttnUpBlock3D(
#       (attentions): ModuleList(
#         (0-2): 3 x Transformer3DModel(
#           (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#           (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): Attention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=1280, out_features=10240, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=5120, out_features=1280, bias=True)
#                 )
#               )
#               (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0-1): 2 x ResnetBlock3D(
#           (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
#           (conv1): InflatedConv3d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): InflatedConv3d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): InflatedConv3d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): ResnetBlock3D(
#           (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
#           (conv1): InflatedConv3d(1920, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): InflatedConv3d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): InflatedConv3d(1920, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (motion_modules): ModuleList(
#         (0-2): 3 x None
#       )
#       (upsamplers): ModuleList(
#         (0): Upsample3D(
#           (conv): InflatedConv3d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#     )
#     (2): CrossAttnUpBlock3D(
#       (attentions): ModuleList(
#         (0-2): 3 x Transformer3DModel(
#           (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#           (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): Attention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=640, out_features=640, bias=False)
#                 (to_v): Linear(in_features=640, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=768, out_features=640, bias=False)
#                 (to_v): Linear(in_features=768, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=640, out_features=5120, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=2560, out_features=640, bias=True)
#                 )
#               )
#               (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock3D(
#           (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
#           (conv1): InflatedConv3d(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): InflatedConv3d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): InflatedConv3d(1920, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock3D(
#           (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (conv1): InflatedConv3d(1280, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): InflatedConv3d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): InflatedConv3d(1280, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): ResnetBlock3D(
#           (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
#           (conv1): InflatedConv3d(960, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): InflatedConv3d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): InflatedConv3d(960, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (motion_modules): ModuleList(
#         (0-2): 3 x None
#       )
#       (upsamplers): ModuleList(
#         (0): Upsample3D(
#           (conv): InflatedConv3d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#     )
#     (3): CrossAttnUpBlock3D(
#       (attentions): ModuleList(
#         (0-2): 3 x Transformer3DModel(
#           (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
#           (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (attn1): Attention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=320, out_features=320, bias=False)
#                 (to_v): Linear(in_features=320, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=768, out_features=320, bias=False)
#                 (to_v): Linear(in_features=768, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=320, out_features=2560, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=1280, out_features=320, bias=True)
#                 )
#               )
#               (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#             )
#           )
#           (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock3D(
#           (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
#           (conv1): InflatedConv3d(960, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
#           (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): InflatedConv3d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): InflatedConv3d(960, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1-2): 2 x ResnetBlock3D(
#           (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (conv1): InflatedConv3d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
#           (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): InflatedConv3d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): InflatedConv3d(640, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (motion_modules): ModuleList(
#         (0-2): 3 x None
#       )
#     )
#   )
#   (mid_block): UNetMidBlock3DCrossAttn(
#     (attentions): ModuleList(
#       (0): Transformer3DModel(
#         (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#         (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#         (transformer_blocks): ModuleList(
#           (0): BasicTransformerBlock(
#             (attn1): Attention(
#               (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_out): ModuleList(
#                 (0): Linear(in_features=1280, out_features=1280, bias=True)
#                 (1): Dropout(p=0.0, inplace=False)
#               )
#             )
#             (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             (attn2): Attention(
#               (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_k): Linear(in_features=768, out_features=1280, bias=False)
#               (to_v): Linear(in_features=768, out_features=1280, bias=False)
#               (to_out): ModuleList(
#                 (0): Linear(in_features=1280, out_features=1280, bias=True)
#                 (1): Dropout(p=0.0, inplace=False)
#               )
#             )
#             (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             (ff): FeedForward(
#               (net): ModuleList(
#                 (0): GEGLU(
#                   (proj): Linear(in_features=1280, out_features=10240, bias=True)
#                 )
#                 (1): Dropout(p=0.0, inplace=False)
#                 (2): Linear(in_features=5120, out_features=1280, bias=True)
#               )
#             )
#             (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#           )
#         )
#         (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#       )
#     )
#     (resnets): ModuleList(
#       (0-1): 2 x ResnetBlock3D(
#         (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#         (conv1): InflatedConv3d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#         (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#         (dropout): Dropout(p=0.0, inplace=False)
#         (conv2): InflatedConv3d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (nonlinearity): SiLU()
#       )
#     )
#     (motion_modules): ModuleList(
#       (0): None
#     )
#   )
#   (conv_norm_out): GroupNorm(32, 320, eps=1e-05, affine=True)
#   (conv_act): SiLU()
#   (conv_out): InflatedConv3d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# )



# ====================================Appearance Encoder================================
# AppearanceEncoderModel(
#   (conv_in): Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (time_proj): Timesteps()
#   (time_embedding): TimestepEmbedding(
#     (linear_1): Linear(in_features=320, out_features=1280, bias=True)
#     (act): SiLU()
#     (linear_2): Linear(in_features=1280, out_features=1280, bias=True)
#   )
#   (down_blocks): ModuleList(
#     (0): CrossAttnDownBlock2D(
#       (attentions): ModuleList(
#         (0-1): 2 x Transformer2DModel(
#           (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
#           (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (attn1): Attention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=320, out_features=320, bias=False)
#                 (to_v): Linear(in_features=320, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=768, out_features=320, bias=False)
#                 (to_v): Linear(in_features=768, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=320, out_features=2560, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=1280, out_features=320, bias=True)
#                 )
#               )
#             )
#           )
#           (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0-1): 2 x ResnetBlock2D(
#           (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (conv1): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
#           (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#       (downsamplers): ModuleList(
#         (0): Downsample2D(
#           (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         )
#       )
#     )
#     (1): CrossAttnDownBlock2D(
#       (attentions): ModuleList(
#         (0-1): 2 x Transformer2DModel(
#           (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#           (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (attn1): Attention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=640, out_features=640, bias=False)
#                 (to_v): Linear(in_features=640, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=768, out_features=640, bias=False)
#                 (to_v): Linear(in_features=768, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=640, out_features=5120, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=2560, out_features=640, bias=True)
#                 )
#               )
#             )
#           )
#           (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (conv1): Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (conv1): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#       (downsamplers): ModuleList(
#         (0): Downsample2D(
#           (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         )
#       )
#     )
#     (2): CrossAttnDownBlock2D(
#       (attentions): ModuleList(
#         (0-1): 2 x Transformer2DModel(
#           (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#           (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (attn1): Attention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=1280, out_features=10240, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=5120, out_features=1280, bias=True)
#                 )
#               )
#             )
#           )
#           (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (conv1): Conv2d(640, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#       (downsamplers): ModuleList(
#         (0): Downsample2D(
#           (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         )
#       )
#     )
#     (3): DownBlock2D(
#       (resnets): ModuleList(
#         (0-1): 2 x ResnetBlock2D(
#           (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#         )
#       )
#     )
#   )
#   (up_blocks): ModuleList(
#     (0): UpBlock2D(
#       (resnets): ModuleList(
#         (0-2): 3 x ResnetBlock2D(
#           (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
#           (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (upsamplers): ModuleList(
#         (0): Upsample2D(
#           (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#     )
#     (1): CrossAttnUpBlock2D(
#       (attentions): ModuleList(
#         (0-2): 3 x Transformer2DModel(
#           (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#           (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (attn1): Attention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#                 (to_k): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_v): Linear(in_features=768, out_features=1280, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=1280, out_features=1280, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=1280, out_features=10240, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=5120, out_features=1280, bias=True)
#                 )
#               )
#             )
#           )
#           (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0-1): 2 x ResnetBlock2D(
#           (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
#           (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): ResnetBlock2D(
#           (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
#           (conv1): Conv2d(1920, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#           (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(1920, 1280, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (upsamplers): ModuleList(
#         (0): Upsample2D(
#           (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#     )
#     (2): CrossAttnUpBlock2D(
#       (attentions): ModuleList(
#         (0-2): 3 x Transformer2DModel(
#           (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
#           (proj_in): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (attn1): Attention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=640, out_features=640, bias=False)
#                 (to_v): Linear(in_features=640, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=640, out_features=640, bias=False)
#                 (to_k): Linear(in_features=768, out_features=640, bias=False)
#                 (to_v): Linear(in_features=768, out_features=640, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=640, out_features=640, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=640, out_features=5120, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=2560, out_features=640, bias=True)
#                 )
#               )
#             )
#           )
#           (proj_out): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
#           (conv1): Conv2d(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(1920, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1): ResnetBlock2D(
#           (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#           (conv1): Conv2d(1280, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): ResnetBlock2D(
#           (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
#           (conv1): Conv2d(960, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
#           (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(960, 640, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#       (upsamplers): ModuleList(
#         (0): Upsample2D(
#           (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         )
#       )
#     )
#     (3): CrossAttnUpBlock2D(
#       (attentions): ModuleList(
#         (0-1): 2 x Transformer2DModel(
#           (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
#           (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (attn1): Attention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=320, out_features=320, bias=False)
#                 (to_v): Linear(in_features=320, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (attn2): Attention(
#                 (to_q): Linear(in_features=320, out_features=320, bias=False)
#                 (to_k): Linear(in_features=768, out_features=320, bias=False)
#                 (to_v): Linear(in_features=768, out_features=320, bias=False)
#                 (to_out): ModuleList(
#                   (0): Linear(in_features=320, out_features=320, bias=True)
#                   (1): Dropout(p=0.0, inplace=False)
#                 )
#               )
#               (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (ff): FeedForward(
#                 (net): ModuleList(
#                   (0): GEGLU(
#                     (proj): Linear(in_features=320, out_features=2560, bias=True)
#                   )
#                   (1): Dropout(p=0.0, inplace=False)
#                   (2): Linear(in_features=1280, out_features=320, bias=True)
#                 )
#               )
#             )
#           )
#           (proj_out): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (2): Transformer2DModel(
#           (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
#           (proj_in): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
#           (transformer_blocks): ModuleList(
#             (0): BasicTransformerBlock(
#               (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
#               (attn1): Attention(
#                 (to_q): _LoRACompatibleLinear()
#                 (to_k): _LoRACompatibleLinear()
#                 (to_v): _LoRACompatibleLinear()
#                 (to_out): ModuleList(
#                   (0-1): 2 x Identity()
#                 )
#               )
#               (norm2): Identity()
#               (attn2): None
#               (norm3): Identity()
#               (ff): Identity()
#             )
#           )
#           (proj_out): Identity()
#         )
#       )
#       (resnets): ModuleList(
#         (0): ResnetBlock2D(
#           (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
#           (conv1): Conv2d(960, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
#           (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#         (1-2): 2 x ResnetBlock2D(
#           (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
#           (conv1): Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
#           (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
#           (dropout): Dropout(p=0.0, inplace=False)
#           (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#           (nonlinearity): SiLU()
#           (conv_shortcut): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1))
#         )
#       )
#     )
#   )
#   (mid_block): UNetMidBlock2DCrossAttn(
#     (attentions): ModuleList(
#       (0): Transformer2DModel(
#         (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
#         (proj_in): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#         (transformer_blocks): ModuleList(
#           (0): BasicTransformerBlock(
#             (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             (attn1): Attention(
#               (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_k): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_v): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_out): ModuleList(
#                 (0): Linear(in_features=1280, out_features=1280, bias=True)
#                 (1): Dropout(p=0.0, inplace=False)
#               )
#             )
#             (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             (attn2): Attention(
#               (to_q): Linear(in_features=1280, out_features=1280, bias=False)
#               (to_k): Linear(in_features=768, out_features=1280, bias=False)
#               (to_v): Linear(in_features=768, out_features=1280, bias=False)
#               (to_out): ModuleList(
#                 (0): Linear(in_features=1280, out_features=1280, bias=True)
#                 (1): Dropout(p=0.0, inplace=False)
#               )
#             )
#             (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
#             (ff): FeedForward(
#               (net): ModuleList(
#                 (0): GEGLU(
#                   (proj): Linear(in_features=1280, out_features=10240, bias=True)
#                 )
#                 (1): Dropout(p=0.0, inplace=False)
#                 (2): Linear(in_features=5120, out_features=1280, bias=True)
#               )
#             )
#           )
#         )
#         (proj_out): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
#       )
#     )
#     (resnets): ModuleList(
#       (0-1): 2 x ResnetBlock2D(
#         (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
#         (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
#         (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
#         (dropout): Dropout(p=0.0, inplace=False)
#         (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         (nonlinearity): SiLU()
#       )
#     )
#   )
# )
