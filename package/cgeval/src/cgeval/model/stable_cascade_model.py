import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

from pathlib import Path

from cgeval import Model

negative_prompt = "text"

class StableCascadeModel(Model):
    def __init__(self, cfg, report_path):
        super().__init__()
        self.cfg = cfg

        self.report_path = f'{report_path}/img'
        Path(self.report_path).mkdir(parents=True, exist_ok=True)

        self.prior = StableCascadePriorPipeline.from_pretrained(cfg.model.prior)
        self.decoder = StableCascadeDecoderPipeline.from_pretrained(cfg.model.name)

    def generate(self, id, inputs):
        prompt = inputs[0]

        prior_output = self.prior(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=512,
            width=512,
            guidance_scale=4.0,
            num_images_per_prompt=1,
            num_inference_steps=20
        )

        output = self.decoder(
            image_embeddings=prior_output.image_embeddings,
            negative_prompt=negative_prompt,
            prompt=prompt,
            guidance_scale=0.0,
            output_type="pil",
            num_inference_steps=10
        )

        output_names = []
        for idx, image in enumerate(output.images):

            image_path = f'{self.report_path}/{id}_{idx}.png'
            image.save(image_path)

            output_names.append(image_path)

        return output_names