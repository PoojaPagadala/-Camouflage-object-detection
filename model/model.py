import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
from PIL import Image
import numpy as np

def load_model():
    # Load processor for B2
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")

    # Customize config for 1 class (binary segmentation)
    config = SegformerConfig.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    config.num_labels = 1  # binary: object or background

    # Initialize model (random weights but with B2 architecture)
    model = SegformerForSemanticSegmentation(config)

    # Load fine-tuned weights trained on COD10K (Segformer-B2)
    checkpoint = torch.load("bestsegformerb2final.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    
    model.eval()
    return processor, model

def predict_image(image, processor, model):
    # Preprocess input image
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # shape: [1, 1, H, W]

    # Resize prediction to original image dimensions
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )

    # Binary thresholding after sigmoid activation
    predicted = (upsampled_logits.sigmoid() > 0.5).squeeze().byte().cpu().numpy()

    # Convert mask to PIL Image for visualization/saving
    return Image.fromarray(predicted * 255)
