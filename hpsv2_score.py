# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from typing import Union
import json

from clint.textui import progress
import hpsv2
from hpsv2.utils import root_path, hps_version_map
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import huggingface_hub
from PIL import Image
import requests
import torch


def initialize_model(pretrained_path, device):
    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14", pretrained=pretrained_path, precision='amp',
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )
    return model, preprocess_val


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_info",
        type=str,
        default="./image_info.json",
        help="Image_info.json file.",
    )
    parser.add_argument(
        "--HPSv2_checkpoint",
        type=str,
        default="./HPS_v2_compressed.pt",
        help="HPS_v2 model weights",
    )
    parser.add_argument(
        "--clip_checkpoint",
        type=str,
        default="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
        help="open clip model weights",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, preprocess_val = initialize_model(args.clip_checkpoint, device)

    checkpoint = torch.load(args.HPSv2_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()
    
    with os.fdopen(os.open(args.image_info, os.O_RDONLY), "r") as f:
        image_info = json.load(f)

    result = []
    for i, info in enumerate(image_info):
        image_file = info['images'][0]
        prompt = info['prompt']
        
        # Load your image and prompt
        with torch.no_grad():
            # Process the image
            if isinstance(image_file, str):
                image = preprocess_val(Image.open(image_file))
            elif isinstance(image_file, Image.Image):
                image = preprocess_val(image_file)
            else:
                raise TypeError('The type of parameter img_path is illegal.')
            image = image.unsqueeze(0).to(device=device, non_blocking=True)
            # Process the prompt
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            # Calculate the HPS
            with torch.cuda.amp.autocast():
                outputs = model(image, text)
                image_features = outputs["image_features"]
                text_features = outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
                print(f"image {i} hps_score: ", hps_score[0])

        result.append(hps_score[0])

    print('avg HPSv2 score:', sum(result) / len(result))


if __name__ == '__main__':
    main()