import requests
import json
import numpy as np
import faiss
from PIL import Image
from transformers import CLIPProcessor
from mongo_pipeline import MongoDBHandler

class ProductMatchingPipeline:
    def __init__(self, triton_url, embedding_dim=512):
        """
        Initializes the ProductMatchingPipeline with FAISS for nearest neighbor search and MongoDB for metadata.

        Args:
            triton_url (str): The URL of the Triton Inference Server for model inference.
            embedding_dim (int): The dimensionality of the embedding vectors. Default is 512.
        """
        self.triton_url = triton_url
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(self.embedding_dim)  # L2-based nearest neighbor search
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.mongo_handler = MongoDBHandler()  # MongoDB connection

    def get_image_embedding(self, image_path, description):
        """
        Generates an embedding for an image and its description using the Triton Inference Server.

        Args:
            image_path (str): The file path to the image.
            description (str): The description of the image.

        Returns:
            numpy.ndarray: The normalized embedding vector for the image.

        Raises:
            Exception: If there is an error during the inference request or response parsing.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=[description], images=image, return_tensors="pt", padding=True)

        input_ids = inputs['input_ids'].numpy().flatten().tolist()
        if len(input_ids) > 8:
            input_ids = input_ids[:8]
        elif len(input_ids) < 8:
            input_ids += [0] * (8 - len(input_ids))

        pixel_values = inputs['pixel_values'].numpy().flatten().tolist()

        input_data = {
            "inputs": [
                {"name": "input_ids", "shape": [1, len(input_ids)], "datatype": "INT64", "data": input_ids},
                {"name": "pixel_values", "shape": [1, 3, 224, 224], "datatype": "FP32", "data": pixel_values}
            ]
        }

        response = requests.post(self.triton_url, json=input_data)
        result = response.json()

        embedding = np.array(result["outputs"][4]["data"], dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding)  # FAISS requires normalized vectors
        return embedding

    def add_embedding_to_index(self, image_path, description):
        """
        Stores the embedding in the FAISS index and in MongoDB.

        Args:
            image_path (str): The file path to the image.
            description (str): The description of the product.

        Raises:
            Exception: If there is an error during embedding generation or database operations.
        """
        embedding = self.get_image_embedding(image_path, description)
        embedding_index = self.index.ntotal  # Current position in the FAISS index

        faiss.normalize_L2(embedding)
        self.index.add(embedding)  # Store embedding in FAISS index

        # Save metadata in MongoDB
        self.mongo_handler.insert_product(
            embedding_index, description, image_path, embedding.tolist()
        )
        print(f"Product added: {description} (Index: {embedding_index})")

    def search_embedding(self, image_path, description, k=1):
        """
        Performs a similarity search using FAISS and returns metadata from MongoDB.

        Args:
            image_path (str): The file path to the image.
            description (str): The description to use for the query.
            k (int): The number of nearest neighbors to retrieve. Default is 1.

        Returns:
            list: A list of product metadata that are the nearest neighbors.

        Raises:
            Exception: If there is an error during embedding retrieval or database operations.
        """
        embedding = self.get_image_embedding(image_path, description)
        faiss.normalize_L2(embedding)

        D, I = self.index.search(embedding, k)  # FAISS: Nearest neighbor search

        results = []
        for idx in I[0]:  # FAISS returns the best k indices
            product = self.mongo_handler.get_product_by_index(idx)
            if product:
                product["distance"] = float(D[0][idx])  # Store similarity score
                results.append(product)

        # Log search results
        self.mongo_handler.log_action(f"Search results for '{description}' (k={k})", "info", extra_info={"query": description, "results": len(results), "k": k})

        return results

    def get_logs(self):
        """
        Retrieves the most recent log entries from MongoDB.

        Returns:
            pymongo.cursor.Cursor: A cursor to the log entries sorted by timestamp in descending order.
        """
        logs = self.mongo_handler.log_collection.find().sort("timestamp", -1).limit(10)
        return logs


# **TEST**
if __name__ == "__main__":
    triton_url = "http://localhost:8000/v2/models/clip_model/infer"
    pipeline = ProductMatchingPipeline(triton_url)

    # Test images and descriptions
    image_paths = ["sneakers.jpg", "tshirt.jpg", "rucksack.jpg"]
    descriptions = ["Sneaker", "T-Shirt", "Rucksack"]

    # Generate and store embeddings
    for image_path, description in zip(image_paths, descriptions):
        pipeline.add_embedding_to_index(image_path, description)

    # Test search
    search_results = pipeline.search_embedding("sneakers.jpg", "seakers", k=3)
    print("Search results:", search_results)
