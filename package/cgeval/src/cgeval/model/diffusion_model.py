from diffusers import DiffusionPipeline

from cgeval import Model

class DiffusionModel(Model):
    def __init__(self, cfg):
        super().__init__()

        self.pipe = DiffusionPipeline.from_pretrained(cfg.model.name).to(cfg.env.device)

    def generate(self, inputs):
        return self.pipe(inputs).images[0]