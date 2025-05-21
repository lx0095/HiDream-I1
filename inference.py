import os
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
torch_npu.npu.config.allow_internal_format = False
torch_npu.npu.set_compile_mode(jit_compile=False)
import argparse
from hi_diffusers import HiDreamImagePipeline
from hi_diffusers import HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
import os

MODEL_PREFIX = "./HiDream-ai"
# LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLAMA_MODEL_NAME = "./Meta-Llama-3.1-8B-Instruct"
# Model configurations
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}

# Resolution options
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]

# Load models
def load_models(model_type):
    config = MODEL_CONFIGS[model_type]
    pretrained_model_name_or_path = config["path"]
    scheduler = MODEL_CONFIGS[model_type]["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)
    
    
    # Load tokenizer (doesn't need to be on GPU)
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        LLAMA_MODEL_NAME,
        use_fast=False)
    
    # Use device_map to distribute text encoder across multiple GPUs
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        output_hidden_states=True,
        output_attentions=False,
        torch_dtype=torch.bfloat16).to('cuda')

    # Use device_map to distribute transformer model across multiple GPUs
    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="transformer",
        torch_dtype=torch.bfloat16).to('cuda')

    # Load pipeline and configure device_map
    pipe = HiDreamImagePipeline.from_pretrained(
        pretrained_model_name_or_path, 
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16
    ).to('cuda', torch.bfloat16)
    pipe.transformer = transformer

    from diffusers.hooks import apply_group_offloading
    onload_device = torch.device("cuda")
    apply_group_offloading(pipe.text_encoder_4, 
                           onload_device=onload_device,
                           offload_type="leaf_level",
                           num_blocks_per_group=2,
                           non_blocking=True,
                           )

    
    return pipe, config

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    if "1024 × 1024" in resolution_str:
        return 1024, 1024
    elif "768 × 1360" in resolution_str:
        return 768, 1360
    elif "1360 × 768" in resolution_str:
        return 1360, 768
    elif "880 × 1168" in resolution_str:
        return 880, 1168
    elif "1168 × 880" in resolution_str:
        return 1168, 880
    elif "1248 × 832" in resolution_str:
        return 1248, 832
    elif "832 × 1248" in resolution_str:
        return 832, 1248
    else:
        return 1024, 1024  # Default fallback

# Generate image function
def generate_image(pipe, model_type, prompt, resolution, seed):
    # Get current model configuration
    config = MODEL_CONFIGS[model_type]
    guidance_scale = config["guidance_scale"]
    num_inference_steps = config["num_inference_steps"]
    
    # Parse resolution
    height, width = parse_resolution(resolution)
    
    # Handle random seed
    if seed == -1:
        seed = torch.randint(0, 1000000, (1,)).item()
    
    # All available GPUs should already be used by the model, no need to manually specify generator's device
    generator = torch.Generator().manual_seed(seed)
    
    # Execute inference
    with torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=generator
        ).images
    
    return images[0], seed

if __name__ == "__main__":
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # os.environ['NCCL_P2P_DISABLE'] = '1' # for old NVIDIA driver
    # os.environ['NCCL_IB_DISABLE'] = '1' # for old NVIDIA driver

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="dev")
    args = parser.parse_args()
    model_type = args.model_type

    # Initialize default model
    pipe, _ = load_models(model_type)
    print("Model loaded successfully!")

    prompt = "A cat holding a sign that says \"Hi-Dreams.ai\"."
    resolution = "1024 × 1024 (Square)"
    seed = -1
    print(f"Generating image, prompt: '{prompt}'")
    image, used_seed = generate_image(pipe, model_type, prompt, resolution, seed)
    print(f"Image generation completed! Seed used: {used_seed}")
    output_path = "output.png"
    image.save(output_path)
    print(f"Image saved to: {output_path}")