from io import BytesIO

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class_list = ["plastic bottle", "plastic food container", "iphone", "other"]

class Identify:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_name(self, b64_image):
        img = Image.open(BytesIO(b64_image))
        print(img.size)
        img.thumbnail((512, 512))
        print(img.size)
        inputs = self.processor(text=class_list, images=img, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        print(logits_per_image)
        probs = logits_per_image.softmax(dim=1)
        index = probs.argmax(dim=1).item()
        name = class_list[index]
        if name == 'iphone':
            return 'plastic phone case'
        return name
