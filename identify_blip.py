from io import BytesIO

from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class IdentifyBlip:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        self.model.to(device)

    def run(self, b64_image):
        img = Image.open(BytesIO(b64_image))
        img.thumbnail((768, 768))

        name = self.prompt("the main object of this image is ")
        plastic = self.prompt("Question: how likely is the main object made of plastic? Answer:")
        return {
            name: name,
            plastic: plastic
        }

    def prompt(self, image, prompt):
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
