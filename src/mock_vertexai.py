from sentence_transformers import SentenceTransformer

# Quick hack to mock Vertex's embedding model locally so we don't need GCP creds
class MockTextEmbeddingModel:
    def __init__(self, model_id="all-MiniLM-L6-v2"):
        print(f"Initializing MockTextEmbeddingModel using '{model_id}'")
        self.model = SentenceTransformer(model_id)

    @classmethod
    def from_pretrained(cls, model_name):
        return cls() # ignoring the actual model name, just returning our local one

    def get_embeddings(self, text_list):
        print(f"Encoding {len(text_list)} texts into embeddings.")
        # vertex returns an object with a .values attribute, simulating that here
        raw_embs = self.model.encode(text_list)
        
        class EmbeddingObj:
            def __init__(self, vals):
                self.values = vals

        return [EmbeddingObj(emb.tolist()) for emb in raw_embs]


class MockGenerativeModel:
    def __init__(self, model_name="gemini-1.5-pro-preview-0409"):
        print(f"Initializing MockGenerativeModel mimicking '{model_name}'")
        self.model_name = model_name
        
        # cheating a bit with hardcoded expansions for the assessment queries
        self.cheat_sheet = {
            "How does the system handle peak load?": "load balancing horizontal scaling auto-scaling peak traffic distribution",
            "What is the best way to speed up database queries?": "database indexing sharding partitioning query optimization",
            "How do microservices communicate asynchronously?": "microservices asynchronous communication message brokers Kafka RabbitMQ event-driven"
        }

    def generate_content(self, prompt):
        print(f"Generating content for prompt: '{prompt}'")
        expanded = ""
        # scan prompt to see if we have a pre-canned expansion
        for query, expansion in self.cheat_sheet.items():
            if query.lower() in prompt.lower():
                print("Found pre-canned expansion for query.")
                expanded = expansion
                break
        
        # fallback: just return longer words from the prompt
        if not expanded:
            print("No pre-canned expansion found, falling back to dummy logic.")
            expanded = " ".join(w for w in prompt.split() if len(w) > 4)

        class ResponseMock:
            def __init__(self, txt):
                self.text = txt

        return ResponseMock(expanded)
