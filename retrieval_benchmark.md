# Retrieval Benchmark Report

## Strategy A (Raw Search) vs. Strategy B (Query Expansion)

### Test 1: How does the system handle peak load?
**Expanded by LLM**: load balancing horizontal scaling auto-scaling peak traffic distribution

| Rank | Strategy A | Strategy B |
| --- | --- | --- |
| 1 | To handle peak load effectively, systems often employ horizontal scaling (adding more instances) com... | To handle peak load effectively, systems often employ horizontal scaling (adding more instances) com... |
| 2 | Load balancing distributes incoming network traffic across multiple servers to ensure no single serv... | Load balancing distributes incoming network traffic across multiple servers to ensure no single serv... |
| 3 | Caching stores copies of frequently accessed data in a temporary storage location so that future req... | API Gateways serve as a single entry point for all client requests, providing features like routing,... |

### Test 2: What is the best way to speed up database queries?
**Expanded by LLM**: database indexing sharding partitioning query optimization

| Rank | Strategy A | Strategy B |
| --- | --- | --- |
| 1 | Database indexing is a data structure technique to efficiently retrieve records from the database fi... | Data partitioning or sharding involves splitting a large database into smaller, more manageable part... |
| 2 | Data partitioning or sharding involves splitting a large database into smaller, more manageable part... | Database indexing is a data structure technique to efficiently retrieve records from the database fi... |
| 3 | Caching stores copies of frequently accessed data in a temporary storage location so that future req... | Caching stores copies of frequently accessed data in a temporary storage location so that future req... |

### Test 3: How do microservices communicate asynchronously?
**Expanded by LLM**: microservices asynchronous communication message brokers Kafka RabbitMQ event-driven

| Rank | Strategy A | Strategy B |
| --- | --- | --- |
| 1 | In a microservices architecture, services often communicate asynchronously using message brokers lik... | In a microservices architecture, services often communicate asynchronously using message brokers lik... |
| 2 | An event-driven architecture uses events to trigger and communicate between decoupled services and i... | An event-driven architecture uses events to trigger and communicate between decoupled services and i... |
| 3 | API Gateways serve as a single entry point for all client requests, providing features like routing,... | API Gateways serve as a single entry point for all client requests, providing features like routing,... |

## Notes

### Why Cosine Similarity?
I went with Cosine Similarity over Euclidean. Since we're dealing with text embeddings (from sentence-transformers/Vertex), the length or magnitude of the vector shouldn't skew the results—we care about the angle (the semantics). To make FAISS do this, I'm normalizing the vectors first and then running an Inner Product search (`IndexFlatIP`). Same math, faster search.

### Moving to Vertex AI
If we pushed this to prod, here's how we'd swap out the local setup:
1. **Embeddings**: Run the actual `vertexai.language_models.TextEmbeddingModel` against the docs, dump to JSONL, and upload to GCS.
2. **Vector DB**: Spin up a `MatchingEngineIndex` in GCP. `DOT_PRODUCT_DISTANCE` works fine if we normalize, or just use Cosine natively.
3. **Deployment**: Deploy it to an `IndexEndpoint`.
4. **App Code**: Swap out the `SimpleVectorStore` with an SDK call to `aiplatform.MatchingEngineIndexEndpoint.match()`. Then fetch the actual text payloads from something like Cloud SQL or Firestore based on the returned IDs.
