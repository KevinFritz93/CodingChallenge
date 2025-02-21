from pymongo import MongoClient
import datetime

class MongoDBHandler:
    def __init__(self, db_url="mongodb://localhost:27017/", db_name="product_db", product_collection="products", log_collection="logs"):
        """
        Initializes the MongoDBHandler and connects to the MongoDB database.

        Args:
            db_url (str): The MongoDB connection URL. Default is "mongodb://localhost:27017/".
            db_name (str): The name of the database to connect to. Default is "product_db".
            product_collection (str): The name of the collection for storing product data. Default is "products".
            log_collection (str): The name of the collection for storing log entries. Default is "logs".
        """
        self.client = MongoClient(db_url)
        self.db = self.client[db_name]
        self.product_collection = self.db[product_collection]
        self.log_collection = self.db[log_collection]

    def insert_product(self, product_id, description, image_path, embedding_index):
        """
        Inserts a product document into the MongoDB collection and logs the operation.

        Args:
            product_id (str): The unique identifier for the product.
            description (str): A description of the product.
            image_path (str): The file path to the product image.
            embedding_index (int): The index of the product embedding for retrieval.

        Raises:
            Exception: If there is an error during the database operation.
        """
        document = {
            "product_id": product_id,
            "description": description,
            "image_path": image_path,
            "embedding_index": embedding_index
        }
        try:
            self.product_collection.insert_one(document)
            self.log_action(f"Product {product_id} saved", "info", extra_info={"product_id": product_id, "action": "insert"})
            print(f"Product {product_id} saved!")
        except Exception as e:
            self.log_action(f"Error saving product {product_id}: {e}", "error", extra_info={"product_id": product_id, "action": "insert", "error": str(e)})
            print(f"Error saving product {product_id}: {e}")

    def get_product_by_index(self, embedding_index):
        """
        Retrieves a product document from the MongoDB collection based on the embedding index and logs the operation.

        Args:
            embedding_index (int): The index of the product embedding to search for.

        Returns:
            dict or None: The product document if found, or None if not found.

        Raises:
            Exception: If there is an error during the database operation.
        """
        try:
            result = self.product_collection.find_one({"embedding_index": int(embedding_index)})
            if result:
                self.log_action(f"Product with index {embedding_index} found", "info", extra_info={"embedding_index": embedding_index, "action": "search"})
            else:
                self.log_action(f"No product found with index {embedding_index}", "warning", extra_info={"embedding_index": embedding_index, "action": "search"})
            return result
        except Exception as e:
            self.log_action(f"Error retrieving product with index {embedding_index}: {e}", "error", extra_info={"embedding_index": embedding_index, "action": "search", "error": str(e)})
            return None

    def log_action(self, message, level="info", extra_info=None):
        """
        Logs an action to the MongoDB logs collection.

        Args:
            message (str): The log message.
            level (str): The severity level of the log (e.g., "info", "error"). Default is "info".
            extra_info (dict): Additional information to include in the log entry. Default is None.

        Raises:
            Exception: If there is an error during the logging operation.
        """
        log_entry = {
            "message": message,
            "level": level,
            "timestamp": datetime.datetime.utcnow(),
            "extra_info": extra_info if extra_info else {}
        }
        try:
            self.log_collection.insert_one(log_entry)
        except Exception as e:
            print(f"Error saving log: {e}")
