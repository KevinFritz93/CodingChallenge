import requests
import json
import numpy as np
import faiss
from PIL import Image
from transformers import CLIPProcessor
from mongo_pipeline import MongoDBHandler

class ProductMatchingPipeline:
    def __init__(self, triton_url, embedding_dim=512):
        """Initialisiert FAISS für die Nearest-Neighbor-Suche & MongoDB für Metadaten."""
        self.triton_url = triton_url
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(self.embedding_dim)  # L2-basierte NN-Suche
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.mongo_handler = MongoDBHandler()  # MongoDB Verbindung

    def get_image_embedding(self, image_path, description):
        """Erzeugt ein Embedding für ein Bild + Beschreibung über Triton."""
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
        faiss.normalize_L2(embedding)  # FAISS benötigt normalisierte Vektoren
        return embedding

    def add_embedding_to_index(self, image_path, description):
        """Speichert das Embedding im FAISS-Index & in MongoDB."""
        embedding = self.get_image_embedding(image_path, description)
        embedding_index = self.index.ntotal  # Aktuelle Position im FAISS-Index

        faiss.normalize_L2(embedding)
        self.index.add(embedding)  # Embedding in FAISS-Index speichern

        # Metadaten in MongoDB speichern
        self.mongo_handler.insert_product(
            embedding_index, description, image_path, embedding.tolist()
        )
        print(f"Produkt hinzugefügt: {description} (Index: {embedding_index})")

    def search_embedding(self, image_path, description, k=1):
        """Führt eine Ähnlichkeitssuche mit FAISS durch und gibt Metadaten aus MongoDB zurück."""
        embedding = self.get_image_embedding(image_path, description)
        faiss.normalize_L2(embedding)

        D, I = self.index.search(embedding, k)  # FAISS: Nearest Neighbor Suche

        results = []
        for idx in I[0]:  # FAISS gibt die besten k Indizes zurück
            product = self.mongo_handler.get_product_by_index(idx)
            if product:
                product["distance"] = float(D[0][idx])  # Ähnlichkeitswert speichern
                results.append(product)

        # Logge die Suchergebnisse
        self.mongo_handler.log_action(f"Suchergebnisse für '{description}' (k={k})", "info", extra_info={"query": description, "results": len(results), "k": k})

        return results

    def get_logs(self):
        logs = self.mongo_handler.log_collection.find().sort("timestamp", -1).limit(10)
        return logs


# **TEST**
if __name__ == "__main__":
    triton_url = "http://localhost:8000/v2/models/clip_model/infer"
    pipeline = ProductMatchingPipeline(triton_url)

    # Testbilder und Beschreibungen
    image_paths = ["sneakers.jpg", "tshirt.jpg", "rucksack.jpg"]
    descriptions = ["Sneaker", "T-Shirt", "Rucksack"]

    # Embeddings generieren & speichern
    for image_path, description in zip(image_paths, descriptions):
        pipeline.add_embedding_to_index(image_path, description)

    # Suche testen
    search_results = pipeline.search_embedding("sneakers.jpg", "seakers", k=3)
    print("Suchergebnisse:", search_results)
