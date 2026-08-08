# demo.py
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import time

# 1. Initialize sentence transformer model (384 dimensions)
print("Loading embedding model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# 2. Database Connection Helper
def get_db_connection():
    max_retries = 10
    for i in range(max_retries):
        try:
            conn = psycopg2.connect(
                dbname="graphrag_db",
                user="postgres",
                password="postgrespassword",
                host="localhost",
                port="5432"
            )
            conn.autocommit = True
            register_vector(conn)
            return conn
        except Exception as e:
            if i == max_retries - 1:
                raise e
            print(f"Waiting for database connection... ({i+1}/{max_retries})")
            time.sleep(2)

def setup_graph_and_vector_data(conn):
    cur = conn.cursor()
    
    # Enable AGE extension in session
    cur.execute("LOAD 'age';")
    cur.execute("SET search_path = ag_catalog, '$user', public;")
    
    print("\n--- 1. Populating Graph Data via Apache AGE (Cypher) ---")
    
    # Clear existing graph data if re-running
    try:
        cur.execute("SELECT drop_graph('tech_graph', true);")
    except Exception:
        pass
    cur.execute("SELECT create_graph('tech_graph');")
    
    # Create Entities (Nodes)
    entities = [
        {"name": "AuthService", "type": "Microservice", "desc": "Handles JWT authentication, password verification, and token issuance."},
        {"name": "UserService", "type": "Microservice", "desc": "Manages user profiles, preferences, and account metadata."},
        {"name": "PaymentService", "type": "Microservice", "desc": "Processes customer credit card payments and subscription billing."},
        {"name": "UserDB", "type": "Database", "desc": "PostgreSQL database storing encrypted user credentials and account profiles."},
        {"name": "PaymentGateway", "type": "ExternalAPI", "desc": "Third-party Stripe REST API integration for credit card processing."},
        {"name": "KafkaBroker", "type": "EventBus", "desc": "Apache Kafka message broker for asynchronous event processing."}
    ]
    
    for entity in entities:
        cypher = f"""
        SELECT * FROM cypher('tech_graph', $$
            CREATE (n:{entity['type']} {{name: '{entity['name']}', description: '{entity['desc']}'}})
            RETURN n
        $$) as (n agtype);
        """
        cur.execute(cypher)
        print(f"Created Node: {entity['name']} ({entity['type']})")

    # Create Relationships (Edges)
    relationships = [
        ("UserService", "DEPENDS_ON", "UserDB"),
        ("AuthService", "READS_FROM", "UserDB"),
        ("AuthService", "COMMUNICATES_WITH", "UserService"),
        ("PaymentService", "CALLS", "PaymentGateway"),
        ("PaymentService", "PUBLISHES_TO", "KafkaBroker"),
        ("UserService", "SUBSCRIBES_TO", "KafkaBroker")
    ]
    
    for src, rel, target in relationships:
        cypher = f"""
        SELECT * FROM cypher('tech_graph', $$
            MATCH (a {{name: '{src}'}}), (b {{name: '{target}'}})
            CREATE (a)-[r:{rel}]->(b)
            RETURN r
        $$) as (r agtype);
        """
        cur.execute(cypher)
        print(f"Created Relationship: ({src}) -[:{rel}]-> ({target})")

    print("\n--- 2. Populating Vector Embeddings via pgvector ---")
    cur.execute("TRUNCATE TABLE entity_embeddings;")
    
    for entity in entities:
        vector = embedder.encode(entity['desc']).tolist()
        
        cur.execute(
            """
            INSERT INTO entity_embeddings (entity_name, entity_type, description, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (entity_name) DO UPDATE SET embedding = EXCLUDED.embedding;
            """,
            (entity['name'], entity['type'], entity['desc'], vector)
        )
        print(f"Stored Vector Embedding for: {entity['name']}")

def run_graphrag_pipeline(conn, user_query):
    cur = conn.cursor()
    cur.execute("LOAD 'age';")
    cur.execute("SET search_path = ag_catalog, '$user', public;")

    print(f"\n=======================================================")
    print(f"USER QUERY: \"{user_query}\"")
    print(f"=======================================================")

    # Step 1: Vector Retrieval (Find Top-K Seed Entities)
    query_vector = embedder.encode(user_query).tolist()
    cur.execute(
        """
        SELECT entity_name, entity_type, description, (embedding <-> %s::vector) AS distance
        FROM entity_embeddings
        ORDER BY distance ASC
        LIMIT 2;
        """,
        (query_vector,)
    )
    seed_entities = cur.fetchall()

    print("\n[STEP 1: Vector Search via pgvector]")
    for name, etype, desc, dist in seed_entities:
        print(f" -> Found Seed Entity: {name} (Type: {etype}) | Distance: {dist:.4f}")

    # Step 2: Graph Traversal via Apache AGE (Explore 1-hop & 2-hop connections)
    print("\n[STEP 2: Graph Traversal via Apache AGE (Cypher)]")
    retrieved_graph_context = []

    for name, etype, desc, dist in seed_entities:
        cypher_query = f"""
        SELECT * FROM cypher('tech_graph', $$
            MATCH (start {{name: '{name}'}})-[r]-(connected)
            RETURN start.name, type(r), connected.name
        $$) as (start agtype, rel agtype, connected agtype);
        """
        cur.execute(cypher_query)
        edges = cur.fetchall()
        
        for start_node, edge_rel, end_node in edges:
            # Strip quotes from agtype strings
            start_clean = str(start_node).replace('"', '')
            rel_clean = str(edge_rel).replace('"', '')
            end_clean = str(end_node).replace('"', '')
            edge_str = f"({start_clean}) -[{rel_clean}]- ({end_clean})"
            if edge_str not in retrieved_graph_context:
                retrieved_graph_context.append(edge_str)
                print(f" -> Discovered Graph Connection: {edge_str}")

    # Step 3: Synthesize GraphRAG Context for LLM
    print("\n[STEP 3: Augmented GraphRAG Context for LLM Prompt]")
    prompt_context = "=== GRAPH-RAG AUGMENTED CONTEXT ===\n"
    prompt_context += "Identified Key Entities:\n"
    for name, etype, desc, dist in seed_entities:
        prompt_context += f"- {name} ({etype}): {desc}\n"
    
    prompt_context += "\nKnowledge Graph Topology & Relationships:\n"
    for edge in retrieved_graph_context:
        prompt_context += f"- {edge}\n"
    prompt_context += "==================================="

    print(prompt_context)

if __name__ == "__main__":
    connection = get_db_connection()
    setup_graph_and_vector_data(connection)
    
    # Run example query
    user_question = "Where are user credentials stored and which services access or manage user data?"
    run_graphrag_pipeline(connection, user_question)
    
    connection.close()
