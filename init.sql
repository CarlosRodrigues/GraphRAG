-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS vector;

-- Set database-level search_path so ag_catalog is available globally across client sessions
ALTER DATABASE graphrag_db SET search_path = ag_catalog, "$user", public;
SET search_path = ag_catalog, "$user", public;

-- Load Apache AGE module into initialization session
LOAD 'age';

-- Initialize Knowledge Graph
SELECT create_graph('tech_graph');

-- Table storing vector embeddings for entities (using 384 dimensions for all-MiniLM-L6-v2)
CREATE TABLE IF NOT EXISTS entity_embeddings (
    id SERIAL PRIMARY KEY,
    entity_name VARCHAR(255) UNIQUE NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    embedding vector(384)
);

-- Create HNSW Index for fast Cosine Distance vector queries
CREATE INDEX IF NOT EXISTS idx_entity_embeddings_cosine 
ON entity_embeddings USING hnsw (embedding vector_cosine_ops);