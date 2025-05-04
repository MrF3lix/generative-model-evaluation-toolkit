import torch
from diffusers import FluxPipeline

from pathlib import Path

from cgeval import Model

class FluxModel(Model):
    def __init__(self, cfg, report_path):
        super().__init__()
        self.cfg = cfg

        self.report_path = f'{report_path}/img'
        Path(self.report_path).mkdir(parents=True, exist_ok=True)

        self.pipe = FluxPipeline.from_pretrained(cfg.model.name, torch_dtype=torch.bfloat16).to(cfg.env.device)


    def generate(self, id, inputs):
        output = self.pipe(
            inputs,
            height=512,
            width=512,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator(self.cfg.env.device).manual_seed(0)
        )
        output_names = []
        for idx, image in enumerate(output.images):

            image_path = f'{self.report_path}/{id}_{idx}.png'
            image.save(image_path)

            output_names.append(image_path)

        return output_names