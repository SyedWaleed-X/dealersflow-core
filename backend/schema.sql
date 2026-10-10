-- 1. Ensure pgvector extension is active in this database
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Clean enum types if re-running
DO $$ BEGIN
    CREATE TYPE property_intent AS ENUM ('BUY', 'SELL');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE property_type AS ENUM ('PLOT', 'HOUSE', 'COMMERCIAL', 'AGRICULTURAL');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 3. Master Properties Table
CREATE TABLE IF NOT EXISTS properties (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Transaction Categorization
    intent property_intent NOT NULL,
    prop_type property_type NOT NULL DEFAULT 'PLOT',
    
    -- Localized Location Hierarchy
    city VARCHAR(64) NOT NULL,
    society VARCHAR(128) NOT NULL,
    phase VARCHAR(64),
    sector_block VARCHAR(64),
    
    -- Standardized Numeric Values
    price_pkr BIGINT NOT NULL,          -- Integer PKR avoids float rounding errors on Crores/Arabs
    area_sqft DOUBLE PRECISION NOT NULL, -- Standardized internal area in Sq Ft (LDA standard)
    bedrooms SMALLINT DEFAULT 0,
    bathrooms SMALLINT DEFAULT 0,
    
    -- Qualitative & Raw Unstructured Data
    raw_transcript TEXT,                -- Original Roman-Urdu voice note text
    features JSONB DEFAULT '{}'::jsonb, -- Flexible tags: {"corner": true, "park_facing": true}
    
    -- Client Contact Details
    contact_name VARCHAR(128),
    contact_phone VARCHAR(32),
    
    -- Vector Embedding (384 dimensions matching sentence-transformers/all-MiniLM-L6-v2)
    embedding vector(384),
    
    -- Audit Timestamps
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 4. Stage 1: Composite B-Tree Indexes for Fast Hard Filtering
-- Queries will filter by: city + intent + price_pkr together
CREATE INDEX IF NOT EXISTS idx_properties_hybrid_filter 
ON properties (city, intent, price_pkr);

CREATE INDEX IF NOT EXISTS idx_properties_society_phase 
ON properties (society, phase);

-- 5. Stage 2: HNSW Vector Index for Semantic Cosine Distance Ranking
-- m = 16: Links per vector in the graph (standard balanced setting)
-- ef_construction = 64: Build exploration depth
CREATE INDEX IF NOT EXISTS idx_properties_hnsw_embedding 
ON properties 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);