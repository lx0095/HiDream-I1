#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import time
import os
import csv
import json

import torch
import torch_npu

from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.utils import PromptLoader, parse_resolution
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate an image using the HiDream-I1 model.")

    # Define arguments for prompt, model path, etc.
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="./prompts/example_prompts.txt",
        help="A text file of prompts for generating images.",
    )
    parser.add_argument(
        "--prompt_file_type",
        choices=["plain", "parti", "hpsv2"],
        default="plain",
        help="Type of prompt file.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Path to save result images.",
    )
    parser.add_argument(
        "--info_file_save_path",
        type=str,
        default="./image_info.json",
        help="Path to save image information file.",
    )
    parser.add_argument(
        "--save_dir_prof",
        type=str,
        default="../prof",
        help="Path to save profiling.",
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/mnt/mindie_data/sd_weights/HiDream-I1-Full", 
        help="Path to the pre-trained HiDream-I1 model.",
    )
    parser.add_argument(
        "--model_path_extra", 
        type=str, 
        default="/mnt/mindie_data/sd_weights/Llama-3.1-8B-Instruct", 
        help="Path to the pre-trained Llama3.1 model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size."
    )
    parser.add_argument(
        "--num_images_per_prompt", 
        type=int, 
        default=1, 
        help="Number of images to generate per prompt.",
    )
    parser.add_argument(
        "--num_inference_steps", 
        type=int, 
        default=50, 
        help="Number of denoising steps for inference."
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float, 
        default=5.0, 
        help="The guidance scale for classifier-free guidance.",
    )
    parser.add_argument(
        "--max_num_prompts",
        default=0,
        type=int,
        help="Limit the number of prompts (0: no limit).",
    )
    parser.add_argument(
        "--resolution", 
        type=str, 
        default="1024 x 1024", 
        help="Resolution of the generated image."
    )
    parser.add_argument(
        "--shift", 
        type=float, 
        default=3.0, 
        help="Shift of scheduler."
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="bf16", 
        help="data type"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Random seed"
    )
    parser.add_argument(
        "--device_id", 
        type=int, 
        default=0, 
        help="NPU device id"
    )
    parser.add_argument(
        "--infer_type", 
        type=str, 
        default="Default", 
        help="Default, Profiling or Accuracy."
    )

    return parser.parse_args()

def load_pipe(args):
    torch.npu.set_device(args.device_id)
    device = f"npu:{args.device_id}"
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Only support bf16. Don't support {args.dtype}.")

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, 
        shift=args.shift, 
        use_dynamic_shifting=False
    )

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        args.model_path_extra,
        use_fast=False,
        local_files_only=True
    )

    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        args.model_path_extra,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=dtype,
        local_files_only=True
    ).to(device)

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        args.model_path, 
        subfolder="transformer", 
        torch_dtype=dtype
    ).to(device)

    pipe = HiDreamImagePipeline.from_pretrained(
        args.model_path, 
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=dtype
    ).to(device, dtype)
    pipe.transformer = transformer
    pipe.enable_model_cpu_offload(device=device)

    seed = args.seed if args.seed else torch.randint(0, 1000000, (1,)).item()
    generator = torch.Generator(device).manual_seed(seed)

    return pipe, generator

def infer(args, pipe, prompt_loader, generator=None, loops=5):
    height, width = parse_resolution(args.resolution)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for _, input_info in enumerate(prompt_loader):
        prompts = input_info['prompts']
        save_names = input_info['save_names']
        n_prompts = input_info['n_prompts']
        break

    use_time = 0
    for i in range(loops):
        start_time = time.time()
        images = pipe(
            prompts,
            height=height,
            width=width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator
        ).images
        # do not count the time spent inferring the first 0 to 1 batches
        if i > 1:
            use_time += time.time() - start_time
            logger.info("current_time is %.3f" % (time.time() - start_time))

        for j in range(n_prompts):
            image_save_path = os.path.join(args.save_dir, f"{save_names[j]}.png")
            image = images[j]
            image.save(image_save_path)

    logger.info("use_time is %.3f" % (use_time / (loops - 2)))

def infer_profiling(args, pipe, prompt_loader, generator=None):
    height, width = parse_resolution(args.resolution)

    for _, input_info in enumerate(prompt_loader):
        prompts = input_info['prompts']
        break

    loops=5
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=True,
        record_shapes=True,
        profile_memory=True,
        schedule=torch_npu.profiler.schedule(wait=2, warmup=2, active=1, repeat=1),
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.save_dir_prof)
    ) as prof:
        for _ in range(loops):
            _ = pipe(
                prompts,
                height=height,
                width=width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator
            ).images
            prof.step()

def infer_accuracy(args, pipe, prompt_loader, generator=None):
    height, width = parse_resolution(args.resolution)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    infer_num = 0
    image_info = []
    current_prompt = None
    for _, input_info in enumerate(prompt_loader):
        prompts = input_info['prompts']
        catagories = input_info['catagories']
        save_names = input_info['save_names']
        n_prompts = input_info['n_prompts']

        infer_num += n_prompts
        logger.info(f"[{infer_num}/{len(prompt_loader)}]: {prompts}")

        images = pipe(
            prompts,
            height=height,
            width=width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator
        ).images
        
        for j in range(n_prompts):
            image_save_path = os.path.join(args.save_dir, f"{save_names[j]}.png")
            image = images[j]
            image.save(image_save_path)

            if current_prompt != prompts[j]:
                current_prompt = prompts[j]
                image_info.append({'images': [], 'prompt': current_prompt, 'category': catagories[j]})

            image_info[-1]['images'].append(image_save_path)

    # Save image information to a json file
    if os.path.exists(args.info_file_save_path):
        os.remove(args.info_file_save_path)

    with os.fdopen(os.open(args.info_file_save_path, os.O_RDWR | os.O_CREAT, 0o640), "w") as f:
        json.dump(image_info, f)


if __name__ == "__main__":
    args = parse_arguments()
    pipe, generator = load_pipe(args)
    prompt_loader = PromptLoader(
        args.prompt_file,
        args.prompt_file_type,
        args.batch_size,
        args.num_images_per_prompt,
        args.max_num_prompts
    )
    if args.infer_type == "Default":
        infer(args, pipe, prompt_loader, generator)
    elif args.infer_type == "Profiling":
        infer_profiling(args, pipe, prompt_loader, generator)
    elif args.infer_type == "Accuracy":
        infer_accuracy(args, pipe, prompt_loader, generator)
    else:
        raise ValueError(f"Not support infer type {args.infer_type}.")
