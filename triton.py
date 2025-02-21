def create_triton_clip_config(config_path="models/clip_model/1/config.pbtxt"):
    """
    Creates a Triton Inference Server configuration file for the CLIP model.

    This function generates a configuration file in the specified path for deploying
    the CLIP model in Triton Inference Server using the ONNX Runtime. The configuration
    includes model input and output specifications, maximum batch size, and instance
    grouping.

    The generated configuration contains the following:
    - Model name: "clip_model"
    - Platform: "onnxruntime_onnx"
    - Maximum batch size: 8
    - Input definitions for 'input_ids' and 'pixel_values' with their respective data types and dimensions.
    - Output definitions for 'logits_per_image' and 'logits_per_text' with their respective data types and dimensions.
    - Instance group configuration specifying that the model will run on the CPU.

    Args:
        config_path (str): The path where the configuration file will be saved.
                           Default is "models/clip_model/config.pbtxt".

    Raises:
        IOError: If there is an error writing to the specified config path.
    """
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
