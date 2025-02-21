import os
import torch
from transformers import CLIPProcessor, CLIPModel

def export_clip_to_onnx():
    """
    Exports the CLIP model from Hugging Face's Transformers library to ONNX format.

    This function performs the following steps:
    1. Creates a directory structure for storing the ONNX model.
    2. Loads the CLIP model and processor using the model name 'openai/clip-vit-base-patch16'.
    3. Prepares a dummy input consisting of a sample text and a random image tensor.
    4. Exports the CLIP model to the specified ONNX file path using PyTorch's ONNX export functionality.

    The exported model will include the following:
    - Input names: 'input_ids' and 'pixel_values' for the text and image inputs, respectively.
    - Output names: 'logits_per_image' and 'logits_per_text' for the model's outputs.
    - Dynamic axes for batching.

    Raises:
        RuntimeError: If the export fails during the ONNX export process.
    """
    os.makedirs("models/clip_model/1", exist_ok=True)

    # Load CLIP
    clip_model_name = "openai/clip-vit-base-patch16"
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    # Dummy input for CLIP (text and image)
    dummy_text = clip_processor(text=["This is a sample text."], return_tensors="pt", padding=True)
    dummy_image = torch.randn(1, 3, 224, 224)  # Dummy image (3, 224, 224)

    # ONNX path for CLIP
    onnx_clip_path = "models/clip_model/1/model.onnx"

    # Export CLIP to ONNX
    torch.onnx.export(
        clip_model,
        (dummy_text["input_ids"], dummy_image),
        onnx_clip_path,
        input_names=["input_ids", "pixel_values"],
        output_names=["logits_per_image", "logits_per_text"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "pixel_values": {0: "batch_size"}},
        opset_version=14
    )
    print(f"CLIP model exported to {onnx_clip_path}")

if __name__ == "__main__":
    export_clip_to_onnx()
