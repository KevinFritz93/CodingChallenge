import os
import torch
from transformers import CLIPProcessor, CLIPModel


def export_clip_to_onnx():
    os.makedirs("models/clip_model/1", exist_ok=True)

    # Lade CLIP
    clip_model_name = "openai/clip-vit-base-patch16"
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    # Dummy-Input für CLIP (Text und Bild)
    dummy_text = clip_processor(text=["This is a sample text."], return_tensors="pt", padding=True)
    dummy_image = torch.randn(1, 3, 224, 224)  # Dummy-Bild (3, 224, 224)

    # ONNX-Pfad für CLIP
    onnx_clip_path = "models/clip_model/1/model.onnx"

    # Exportiere CLIP in ONNX
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
