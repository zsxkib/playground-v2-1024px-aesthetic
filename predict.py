# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import time
import torch
import subprocess
import numpy as np
from typing import List
from cog import BasePredictor, Input, Path
from transformers import CLIPImageProcessor
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)


MODEL_NAME = "playgroundai/playground-v2-1024px-aesthetic"
FEATURE_EXTRACTOR = "./feature-extractor"

PGV2_MODEL_CACHE = "./sdxl-cache"
PGV2_MODEL_512_CACHE = "./sdxl-cache-512"
PGV2_MODEL_256_CACHE = "./sdxl-cache-256"

PGV2_URL = "https://weights.replicate.delivery/default/playgroundai/sdxl-cache.tar"
PGV2_URL_512 = (
    "https://weights.replicate.delivery/default/playgroundai/sdxl-cache-512.tar"
)
PGV2_URL_256 = (
    "https://weights.replicate.delivery/default/playgroundai/sdxl-cache-256.tar"
)

SAFETY_CACHE = "./safety-cache"
SAFETY_URL = "https://weights.replicate.delivery/default/playgroundai/safety-cache.tar"


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.load_model(PGV2_URL, PGV2_MODEL_CACHE)
        self.playground_model_loaded = "playground-v2-1024px-aesthetic"

        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE,
            torch_dtype=torch.float16,
        )
        self.safety_checker.to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

    def load_model(self, url_to_model, cache_to_model):
        if not os.path.exists(cache_to_model):
            download_weights(url_to_model, cache_to_model)
        self.pipe = DiffusionPipeline.from_pretrained(
            cache_to_model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.pipe.to("cuda")

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        playground_model: str = Input(
            description="Select the size of the Playground V2 model you'd like to use",
            default="playground-v2-1024px-aesthetic",
            choices=[
                "playground-v2-256px-base",
                "playground-v2-512px-base",
                "playground-v2-1024px-aesthetic",
            ],
        ),
        width: int = Input(
            description="Width of output image, remember to change depending on `playground_model` chosen",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image, remember to change depending on `playground_model` chosen",
            default=1024,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER_ANCESTRAL",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=3.0
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        if self.playground_model_loaded != playground_model:
            if playground_model == "playground-v2-1024px-aesthetic":
                self.load_model(PGV2_URL, PGV2_MODEL_CACHE)
            elif playground_model == "playground-v2-512px-base":
                self.load_model(PGV2_URL_512, PGV2_MODEL_512_CACHE)
            else:  # if playground_model == "playground-v2-256px-base"
                self.load_model(PGV2_URL_256, PGV2_MODEL_256_CACHE)
            self.playground_model_loaded = playground_model

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        PGV2_kwargs = {}
        PGV2_kwargs["width"] = width
        PGV2_kwargs["height"] = height
        pipe = self.pipe

        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        output = pipe(**common_args, **PGV2_kwargs)

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        output_paths = []
        for i, image in enumerate(output.images):
            if not disable_safety_checker:
                if has_nsfw_content[i]:
                    print(f"NSFW content detected in image {i}")
                    continue
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
