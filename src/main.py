"""Main module for the src package."""
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

print("[o] Loading processor and model from pretrained... \n")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

print("[o] Loading image ... \n")
image = Image.open("./img/cameraman.png")
print("Image loaded successfully." + str(image.size) + "\n")
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)

print("[o] Output from model:")
print(processor.decode(out[0], skip_special_tokens=True))
