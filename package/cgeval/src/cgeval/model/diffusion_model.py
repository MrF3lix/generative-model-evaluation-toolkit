import torch
from diffusers import StableDiffusion3Pipeline
from pathlib import Path

from cgeval import Model

class DiffusionModel(Model):
    def __init__(self, cfg, report_path):
        super().__init__()
        self.cfg = cfg

        self.report_path = f'{report_path}/img'
        Path(self.report_path).mkdir(parents=True, exist_ok=True)

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            cfg.model.name
        ).to(cfg.env.device)

        self.pipe = pipeline


    def generate(self, id, inputs):
        output = self.pipe(
            prompt=inputs,
            height=512,
            width=512,
            num_inference_steps=40,
            guidance_scale=4.5,
            max_sequence_length=512,
        )
        output_names = []
        for idx, image in enumerate(output.images):

            image_path = f'{self.report_path}/{id}_{idx}.png'
            image.save(image_path)

            output_names.append(image_path)

        return output_names