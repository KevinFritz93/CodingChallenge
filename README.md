# AI Infrastructure Coding Challenge

## Requirements

1. **Mock a Vector Database** (e.g., Qdrant, FAISS, Weaviate, Chroma, or Pinecone)
   - Store product embeddings (textual and visual) in the database.
   - Allow efficient nearest neighbor search for retrieval.

2. **Mock a MongoDB Database to Store Product Metadata**
   - Store product images and metadata (e.g., name, category, price).
   - Implement necessary query functions.

3. **Quantize a Visual Language Model (VLM) using TensorRT**
   - Select a VLM (preferably popular ones like LLaVA, QwenVL, InternVL, Idefics, etc.).
   - Select a text encoder model (e.g., BERT, SentenceEncoder, GPT).
   - Select a vision encoder model (e.g., DINOv2, CLIP).
   - Perform model quantization/compilation using TensorRT.

4. **Deploy the Quantized Model Using NVIDIA Triton Inference Server**
   - Serve the quantized model via an HTTP endpoint.
   - Ensure it supports both image and text inputs.

5. **Build the Product Matching Pipeline**
   - Given an input image, extract text+visual embeddings using the deployed model and DB info.
   - Perform a nearest neighbor search in the vector database.
   - Retrieve the best match along with metadata from MongoDB.

6. **Mock a MongoDB for Logging**
   - Store logs, errors, and execution results.
   - Ensure error handling and tracking.
  
  ## Running the Application

To set up and run the application, follow these steps in order:

1. **Start the CLIP to ONNX Conversion**
   - Run the script to convert the CLIP model to ONNX format:
   ```bash
   python clip_to_onnx.py
   ```
2. **Run the Triton Inference Server with the specified model repository**
   ```bash
   python triton.py
   ```
3. **Run the Triton Server in Docker**
   ```bash
   sudo docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /home/kevin/PycharmProjects/Coding_Challange/models:/models \
    nvcr.io/nvidia/tritonserver:23.02-py3 \
    tritonserver --model-repository=/models
   ```
4. **Start a MongoDB container**
   ```bash
   docker run --name mongodb -d -p 27017:27017 mongo:latest
   ```
5. **Run your matching pipeline script**
   ```bash
   python matching_pipeline.py
   ```




