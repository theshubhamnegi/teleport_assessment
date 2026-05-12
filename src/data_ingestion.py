from src.utils.logger import get_logger

logger = get_logger(__name__)

# just some dummy data to play with
DUMMY_DATA = [
    "Load balancing distributes incoming network traffic across multiple servers to ensure no single server bears too much demand. This improves responsiveness and availability of applications.",
    "In a microservices architecture, services often communicate asynchronously using message brokers like RabbitMQ or Apache Kafka. This decouples the sender and receiver, enhancing system resilience.",
    "Database indexing is a data structure technique to efficiently retrieve records from the database files based on some attributes on which the indexing has been done.",
    "Caching stores copies of frequently accessed data in a temporary storage location so that future requests for that data can be served faster, significantly reducing latency.",
    "An event-driven architecture uses events to trigger and communicate between decoupled services and is common in modern applications built with microservices.",
    "To handle peak load effectively, systems often employ horizontal scaling (adding more instances) combined with robust load balancing and auto-scaling groups.",
    "A Content Delivery Network (CDN) refers to a geographically distributed group of servers which work together to provide fast delivery of Internet content.",
    "Data partitioning or sharding involves splitting a large database into smaller, more manageable parts, which can be hosted on multiple servers to improve query performance.",
    "Continuous Integration and Continuous Deployment (CI/CD) pipelines automate the software delivery process, allowing for frequent and reliable updates to applications.",
    "API Gateways serve as a single entry point for all client requests, providing features like routing, rate limiting, and authentication for backend microservices."
]

class DataIngester:
    def __init__(self):
        self.docs = DUMMY_DATA

    def load_data(self):
        # normally we'd pull from a db or file here, but keeping it simple for now
        logger.info(f"Loaded {len(self.docs)} dummy documents for ingestion.")
        return self.docs
