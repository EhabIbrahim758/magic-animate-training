import inspect
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict

import torch
import random
import cv2
from diffusers import AutoencoderKL, DDIMScheduler, UniPCMultistepScheduler
from peft import LoraConfig
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from MagicAnimate.magic_animate.unet_controlnet import UNet3DConditionModel
from MagicAnimate.magic_animate.controlnet import ControlNetModel
from MagicAnimate.magic_animate.appearance_encoder import AppearanceEncoderModel
from MagicAnimate.magic_animate.mutual_self_attention import ReferenceAttentionControl
from diffusers.models import UNet2DConditionModel
from MagicAnimate.magic_animate.pipeline import AnimationPipeline 
from MagicAnimate.utils.util import save_videos_grid, resize_and_crop

from accelerate.utils import set_seed
from MagicAnimate.utils.videoreader import VideoReader
from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path


class MagicAnimate(torch.nn.Module):
    def __init__(self,
                 config="configs/training/animation.yaml",
                 device=torch.device("cuda"),
                 unet_additional_kwargs=None,
                 L=None, 
                 unet2d = False, 
                ):
        super().__init__()

        print("Initializing MagicAnimate Pipeline...")
        *_, func_args = inspect.getargvalues(inspect.currentframe())
        func_args = dict(func_args)

        config = OmegaConf.load(config)
        self.device = device

        inference_config = OmegaConf.load(config.inference_config)
        motion_module = config.motion_module
        
        if unet_additional_kwargs is None:
            unet_additional_kwargs = OmegaConf.to_container(inference_config.unet_additional_kwargs)

        ### >>> create animation pipeline >>> ###
        self.tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder")
        
        if not unet2d:
            self.unet = UNet3DConditionModel.from_pretrained_2d(
                config.pretrained_model_path, subfolder="unet", 
                unet_additional_kwargs=unet_additional_kwargs
            )
        else:
            self.unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_path, subfolder="unet")
            
        self.appearance_encoder = AppearanceEncoderModel.from_pretrained(config.pretrained_appearance_encoder_path,
                                                                         subfolder="appearance_encoder")

        # self.unet = UNet2DConditionModel.from_pretrained(config.pretrained_appearance_encoder_path, subfolder="appearance_encoder")
        
        
        if config.pretrained_vae_path is not None:
            self.vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path)
        else:
            self.vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae")

        ### Load controlnet
        self.controlnet = ControlNetModel.from_pretrained(config.pretrained_controlnet_path)

        self.vae.to(device=self.device, dtype=torch.float16)
        self.unet.to(device=self.device, dtype=torch.float16)
        self.text_encoder.to(device=self.device, dtype=torch.float16)
        self.controlnet.to(device=self.device, dtype=torch.float16)
        self.appearance_encoder.to(device=self.device, dtype=torch.float16)


        # 1. unet ckpt
        # 1.1 motion module
        if unet_additional_kwargs['use_motion_module']:
            motion_module_state_dict = torch.load(motion_module, map_location="cpu")
            if "global_step" in motion_module_state_dict: func_args.update(
                {"global_step": motion_module_state_dict["global_step"]})
            motion_module_state_dict = motion_module_state_dict[
                'state_dict'] if 'state_dict' in motion_module_state_dict else motion_module_state_dict
            try:
                # extra steps for self-trained models
                state_dict = OrderedDict()
                for key in motion_module_state_dict.keys():
                    if key.startswith("module."):
                        _key = key.split("module.")[-1]
                        state_dict[_key] = motion_module_state_dict[key]
                    else:
                        state_dict[key] = motion_module_state_dict[key]
                motion_module_state_dict = state_dict
                del state_dict
                missing, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)
                assert len(unexpected) == 0
            except:
                _tmp_ = OrderedDict()
                for key in motion_module_state_dict.keys():
                    if "motion_modules" in key:
                        if key.startswith("unet."):
                            _key = key.split('unet.')[-1]
                            _tmp_[_key] = motion_module_state_dict[key]
                        else:
                            _tmp_[key] = motion_module_state_dict[key]
                missing, unexpected = self.unet.load_state_dict(_tmp_, strict=False)
                assert len(unexpected) == 0
                del _tmp_
            del motion_module_state_dict

        self.pipeline = AnimationPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            controlnet=self.controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            # NOTE: UniPCMultistepScheduler
        ).to(device)

        self.L = config.L if L is None else L
        print("Initialization Done!")


    def infer_for_image(self, source_image, motion_sequence, init_latents=None, random_seed=-1, step=50, guidance_scale=7.5, size=(512, 512)):
        prompt = ""
        n_prompt = ""
        random_seed = int(random_seed)
        step = int(step)
        guidance_scale = float(guidance_scale)
        
        # manually set random seed for reproduction
        if random_seed != -1:
            torch.manual_seed(random_seed)
            set_seed(random_seed)
        else:
            torch.seed()

        control = motion_sequence

        if source_image.shape != size:
            source_image = np.array(Image.fromarray(source_image).resize(size))

        H, W, C = source_image.shape

        generator = torch.Generator(device=self.device)
        generator.manual_seed(torch.initial_seed())

        print('before pipeline code ....................')
        sample = self.pipeline(
            prompt,
            negative_prompt=n_prompt,
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            width=W,
            height=H,
            video_length=control.shape[0],
            controlnet_condition=control,
            init_latents=None,  # inference, start from white noise.
            generator=generator,
            appearance_encoder=self.appearance_encoder,
            unet=self.unet,
            source_image=source_image,
            context_frames=self.L,
            context_stride=1,
            context_overlap=4,
            context_batch_size=1,
            context_schedule="uniform",
        ).videos
        
        # return samples_per_video
        sample = np.array(sample.squeeze())
        sample = np.transpose(sample, (1, 2, 0))
        # print(sample.shape)
        # cv2.imwrite('./data/res.jpeg', sample*255)
        return sample


    def forward(self, init_latents, image_prompts, timestep, source_image, motion_sequence, random_seed):
        """
        :param init_latents: the most important input during training
        :param timestep: another important input during training
        :param source_image: an image in np.array
        :param motion_sequence: np array, (f, h, w, c) (0, 255)
        :param random_seed:
        :param size: width=512, height=768 by default
        :return:
        """
        prompt = n_prompt = ""
        random_seed = int(random_seed)

        # manually set random seed for reproduction
        if random_seed != -1:
            torch.manual_seed(random_seed)
            set_seed(random_seed)
        else:
            torch.seed()

        # control = np.array(motion_sequence)
        control = motion_sequence
        H, W, C = source_image.shape
        
        generator = torch.Generator(device=self.device)
        generator.manual_seed(torch.initial_seed())

        
        noise_pred = self.pipeline.train(
            prompt,
            prompt_embeddings=image_prompts,
            negative_prompt=n_prompt,
            timestep=timestep,
            width=W,
            height=H,
            video_length=control.shape[0],
            controlnet_condition=control,
            init_latents=init_latents,  # add noise to latents
            generator=generator,
            appearance_encoder=self.appearance_encoder,
            unet=self.unet,
            source_image=source_image,
        )

        return noise_pred
