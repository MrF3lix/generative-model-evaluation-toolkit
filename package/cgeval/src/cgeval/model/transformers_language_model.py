from transformers import AutoModelForCausalLM, AutoTokenizer

from cgeval import Model

class TransformersLanguageModel(Model):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

        self.model.to(cfg.env.device)


    # TODO: This implementation is untested
    def generate(self, id, inputs):
        instruction = f"<s>[INST]{inputs} [/INST]"

        # model_inputs = self.tokenizer.encode(instruction)
        model_inputs = self.tokenizer(instruction).to(self.cfg.env.device)

        # model_inputs = encodeds.to(self.cfg.env.device)
        # self.model.to(self.cfg.env.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=1000, do_sample=True)
        result = self.tokenizer.decode(generated_ids[0].tolist())

        return result
