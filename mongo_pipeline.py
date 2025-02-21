from pymongo import MongoClient
import datetime

class MongoDBHandler:
    def __init__(self, db_url="mongodb://localhost:27017/", db_name="product_db", product_collection="products", log_collection="logs"):
        """Verbindet sich mit MongoDB und initialisiert die Collections für Produkte und Logs."""
        self.client = MongoClient(db_url)
        self.db = self.client[db_name]
        self.product_collection = self.db[product_collection]
        self.log_collection = self.db[log_collection]

    def insert_product(self, product_id, description, image_path, embedding_index):
        """Speichert ein Produkt mit Metadaten in der MongoDB und loggt den Vorgang."""
        document = {
            "product_id": product_id,
            "description": description,
            "image_path": image_path,
            "embedding_index": embedding_index
        }
        try:
            self.product_collection.insert_one(document)
            self.log_action(f"Produkt {product_id} gespeichert", "info", extra_info={"product_id": product_id, "action": "insert"})
            print(f"Produkt {product_id} gespeichert!")
        except Exception as e:
            self.log_action(f"Fehler beim Speichern des Produkts {product_id}: {e}", "error", extra_info={"product_id": product_id, "action": "insert", "error": str(e)})
            print(f"Fehler beim Speichern des Produkts {product_id}: {e}")

    def get_product_by_index(self, embedding_index):
        """Ruft das Produkt basierend auf dem FAISS-Index ab und loggt den Vorgang."""
        try:
            result = self.product_collection.find_one({"embedding_index": int(embedding_index)})
            if result:
                self.log_action(f"Produkt mit Index {embedding_index} gefunden", "info", extra_info={"embedding_index": embedding_index, "action": "search"})
            else:
                self.log_action(f"Kein Produkt mit Index {embedding_index} gefunden", "warning", extra_info={"embedding_index": embedding_index, "action": "search"})
            return result
        except Exception as e:
            self.log_action(f"Fehler beim Abrufen des Produkts mit Index {embedding_index}: {e}", "error", extra_info={"embedding_index": embedding_index, "action": "search", "error": str(e)})
            return None

    def log_action(self, message, level="info", extra_info=None):
        """Speichert Logeinträge in der MongoDB-Datenbank."""
        log_entry = {
            "message": message,
            "level": level,
            "timestamp": datetime.datetime.utcnow(),
            "extra_info": extra_info if extra_info else {}
        }
        try:
            self.log_collection.insert_one(log_entry)
        except Exception as e:
            print(f"Fehler beim Speichern des Logs: {e}")
