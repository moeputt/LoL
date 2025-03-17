import argparse
import gc
import hashlib
import itertools
import logging
import math
import os
import threading
import warnings
from pathlib import Path
from typing import Optional
import psutil
import json

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers import DDPMScheduler, PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPFeatureExtractor
from peft import PeftModel, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.10.0.dev0")

# logger = get_logger(__name__)


MODEL_NAME = "CompVis/stable-diffusion-v1-4"  # "stabilityai/stable-diffusion-2-1-base"
# INSTANCE_PROMPT = "a photo of sks dog"
# base_path = "."
from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images) :
    return images, [False for i in images]

# edit StableDiffusionSafetyChecker class so that, when called, it just returns the images and an array of True values
safety_checker.StableDiffusionSafetyChecker.forward = sc
device = torch.device('cuda')
from safetensors.numpy import save_file, load_file
import numpy as np

from torchmetrics.functional.multimodal import clip_score
from functools import partial
import time
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2).to(torch.device('cuda')), prompts).detach()
    return round(float(clip_score), 4)
sample_prompts = [
    'a red sphere on top of a yellow box',
    'a chimpanzee sitting on a wooden bench',
    'a clock tower',
    'toy cars',
    'a white rabbit in blue jogging clothes doubled over in pain while a turtle wearing a red tank top dashes confidently through the finish line',
    'a train going to the moon',
    'Four cats surrounding a dog',
    'the Eiffel Tower in a desert',
    'The Millennium Wheel next to the Statue of Liberty. The Sagrada Familia church is also visible.',
    'A punk rock squirrel in a studded leather jacket shouting into a microphone while standing on a stump and holding a beer on dark stage.',
    'The Statue of Liberty surrounded by helicopters',
    'A television made of water that displays an image of a cityscape at night.',
    'a family on a road trip',
    'the mona lisa wearing a cowboy hat and screaming a punk song into a microphone',
    'a family of four posing at Mount Rushmore',
    'force',
    'an oil surrealist painting of a dreamworld on a seashore where clocks and watches appear to be inexplicably limp and melting in the desolate landscape. a table on the left, with a golden watch swarmed by ants. a strange fleshy creature in the center of the painting',
    'Downtown Austin at sunrise. detailed ink wash.',
    'A helicopter flies over Yosemite.',
    'A giraffe walking through a green grass covered field',
    'a corgi’s head',
    'portrait of a well-dressed raccoon, oil painting in the style of Rembrandt',
    'a volcano',
    'happiness',
    "the words 'KEEP OFF THE GRASS' on a black sticker",
    'A heart made of wood',
    'a pixel art corgi pizza',
    'two wine bottles',
    'A funny Rube Goldberg machine made out of metal',
    'a horned owl with a graduation cap and diploma',
    'A celebrity in a park',
    'A person on the beach',
    'A thing in a city',
]


import torch
scores = {}
for model in os.listdir('./models_4'):
    if 'model' not in model:
        continue
    model_path = "./models_4/" + model
    print(model_path)
    
    if os.path.exists(os.path.join(model_path, 'score')):
        print('continuing')
        continue
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, model_path + "/unet", adapter_name='celebrity')
    generator = torch.Generator()
    generator.manual_seed(42)
    images = pipe(sample_prompts, num_images_per_prompt=1, output_type='np', generator = generator, num_inference_steps=20).images
    sd_clip_score = calculate_clip_score(images, sample_prompts)
    scores[model] = sd_clip_score
    print(model_path + '/score.pt')
    torch.save(sd_clip_score, model_path + '/score')
    print(f"CLIP score: {sd_clip_score}")