def create_triton_clip_config(config_path="models/clip_model/config.pbtxt"):
    config_content = """
name: "clip_model"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: "pixel_values"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "logits_per_image"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  },
  {
    name: "logits_per_text"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  }
]
instance_group [{ kind: KIND_CPU }]
"""
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"Triton config for CLIP created at {config_path}")

if __name__ == "__main__":
    create_triton_clip_config()