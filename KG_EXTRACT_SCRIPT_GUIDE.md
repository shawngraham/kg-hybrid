# Knowledge Graph Extraction Script - Usage Guide

## Overview

Command-line version of the hybrid spaCy + LLM knowledge graph extraction pipeline. Runs without Jupyter overhead for better memory management and batch processing.

## Installation

```bash
# Make script executable
chmod +x kg_extract.py

# Install dependencies
pip install spacy networkx requests tqdm

# Download spaCy model
python -m spacy download en_core_web_trf
```

## Basic Usage

### Single Document

```bash
# With LLM refinement (highest quality)
python kg_extract.py \
    --input document.txt \
    --output results/

# Without LLM refinement (faster, uses spaCy only)
python kg_extract.py \
    --input document.txt \
    --output results/ \
    --no-llm
```

### Multiple Documents (Directory)

```bash
# Process all .txt and .md files in a directory
python kg_extract.py \
    --input-dir documents/ \
    --output results/

# With custom settings
python kg_extract.py \
    --input-dir documents/ \
    --output results/ \
    --no-llm \
    --chunk-size 8000
```

### Large Documents

```bash
# Automatically chunks large documents
python kg_extract.py \
    --input large_book.txt \
    --output results/ \
    --chunk-size 4000
```

## Command-Line Options

```
--input FILE           Single input document
--input-dir DIR        Directory with multiple documents
--output DIR           Output directory (required)
--no-llm              Skip LLM refinement (faster, less accurate)
--chunk-size N        Characters per chunk (default: 4000)
--spacy-model NAME    spaCy model (default: en_core_web_trf)
--api-url URL         Local LLM API URL
--api-key KEY         API key for local LLM
--export-format       json csv neo4j (default: json csv)
```

## Output Files

The script creates:

```
results/
├── knowledge_graph.json       # Full KG in JSON format
├── nodes_table.csv           # Nodes for analysis
├── edges_table.csv           # Edges for analysis
└── neo4j_import.cypher       # Neo4j import script (optional)
```


```bash
# Process your full corpus
python kg_extract.py \
    --input-dir /path/to/corpus \
    --output production_kg \
    --export-format json csv neo4j
```

## Common Use Cases

### 1. Quick Test Without LLM

```bash
# Fastest way to test extraction
python kg_extract.py \
    --input test_doc.txt \
    --output test_results \
    --no-llm
```

**Use when:**
- Testing spaCy extraction patterns
- Don't need entity canonicalization
- Want quick results
- Memory is limited

### 2. High-Quality Extraction with LLM

```bash
# Best quality results
python kg_extract.py \
    --input document.txt \
    --output results
```

**Use when:**
- Need entity deduplication
- Want canonical IDs
- Have local LLM running (jan.ai)
- Quality > speed

### 3. Batch Processing

```bash
# Process entire corpus
for file in corpus/*.txt; do
    name=$(basename "$file" .txt)
    python kg_extract.py \
        --input "$file" \
        --output "results/$name" \
        --no-llm
done

# Then merge results
python merge_kgs.py results/*/knowledge_graph.json \
    --output merged_kg.json
```

### 4. Large Document Processing

```bash
# Handles documents of any size
python kg_extract.py \
    --input 500_page_book.txt \
    --output book_kg \
    --chunk-size 8000 \
    --no-llm  # Recommended for very large docs
```

## Configuration

### Local LLM Setup

Edit script or use command-line args:

```bash
python kg_extract.py \
    --input document.txt \
    --output results \
    --api-url "http://localhost:1337/v1/chat/completions" \
    --api-key "your_key"
```

### Custom spaCy Model

```bash
# Use smaller/faster model
python kg_extract.py \
    --input document.txt \
    --output results \
    --spacy-model en_core_web_sm \
    --no-llm
```

**Model options:**
- `en_core_web_sm` - Fast, less accurate (50 MB)
- `en_core_web_md` - Balanced (91 MB)
- `en_core_web_lg` - Better accuracy (560 MB)
- `en_core_web_trf` - Best accuracy, transformer-based (438 MB)

## Monitoring Progress

The script provides detailed progress output:

```
Loading spaCy model: en_core_web_trf...
spaCy model loaded

Reading documents from: documents/
  Loaded file1.txt: 5234 characters
  Loaded file2.txt: 8912 characters
Found 2 documents

============================================================
Processing document 1/2: file1.txt
============================================================

============================================================
STAGE 1: spaCy Fast Extraction
============================================================
Extracted 25 entities and 12 relationships

============================================================
STAGE 2: Local LLM Refinement
============================================================
LLM refined 18 entities
LLM enhanced 10 relationships

...
```

## Error Handling

The script handles common errors gracefully:

**LLM unavailable:**
```
Warning: LLM refinement failed, using spaCy-only results
```

**File not found:**
```
Error: File not found: document.txt
```

**Out of memory:**
- Automatically falls back to smaller chunks
- Processes one document at a time
- Forces garbage collection

## Advanced Features

### Custom Export Formats

```bash
# Only JSON
python kg_extract.py \
    --input doc.txt \
    --output results \
    --export-format json

# All formats
python kg_extract.py \
    --input doc.txt \
    --output results \
    --export-format json csv neo4j
```

### Processing Pipeline

The script implements a 3-stage pipeline:

**Stage 1: spaCy Extraction**
- Fast NLP extraction
- Entity recognition
- Dependency parsing
- Relationship extraction

**Stage 2: LLM Refinement (optional)**
- Entity canonicalization
- Coreference resolution
- Attribute enhancement
- Relationship validation

**Stage 3: Export**
- JSON format
- CSV tables (for analysis)
- Neo4j Cypher (for graph database)

## Integration with Other Tools

### Feed to PyKEEN

```bash
# Extract KG
python kg_extract.py \
    --input corpus/ \
    --output kg_results \
    --export-format csv

# Train embeddings
python kg_embedding_analysis.py \
    --input kg_results/edges_table.csv \
    --output embeddings
```

### Import to Neo4j

```bash
# Generate Cypher
python kg_extract.py \
    --input doc.txt \
    --output results \
    --export-format neo4j

# Import to Neo4j
cat results/neo4j_import.cypher | cypher-shell
```

### Analyze with NetworkX

```python
import json
import networkx as nx

# Load KG
with open('results/knowledge_graph.json') as f:
    kg = json.load(f)

# Convert to NetworkX
G = nx.DiGraph()
for node_id, node in kg['nodes'].items():
    G.add_node(node_id, **node)
for edge in kg['edges']:
    G.add_edge(edge['source'], edge['target'], **edge)

# Analyze
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.3f}")
```

## Troubleshooting

### Issue: spaCy model not found

```bash
# Download model
python -m spacy download en_core_web_trf

# Or use smaller model
python kg_extract.py --spacy-model en_core_web_sm ...
```

### Issue: LLM not responding

```bash
# Check if jan.ai is running
curl http://localhost:1337/v1/models

# Or skip LLM
python kg_extract.py --no-llm ...
```

### Issue: Out of memory

```bash
# Reduce chunk size
python kg_extract.py --chunk-size 2000 ...

# Skip LLM
python kg_extract.py --no-llm ...

# Process one file at a time
for file in *.txt; do
    python kg_extract.py --input "$file" ...
done
```

### Issue: Slow processing

**Solutions:**
1. Use `--no-llm` (10x faster)
2. Use smaller spaCy model (`en_core_web_sm`)
3. Increase chunk size for fewer LLM calls
4. Process in parallel (see below)

## Parallel Processing

For large corpora, process multiple documents in parallel:

```bash
# Create process script
cat > process_parallel.sh << 'EOF'
#!/bin/bash
find corpus/ -name "*.txt" | \
    parallel -j 4 \
    "python kg_extract.py \
        --input {} \
        --output results/{/.} \
        --no-llm"
EOF

chmod +x process_parallel.sh
./process_parallel.sh
```

Requires GNU Parallel: `brew install parallel`

## Comparison with Notebook

### When to Use Script

- Processing > 10 documents
- Documents > 50,000 characters
- Memory is constrained
- Need batch processing
- Production pipeline
- Scheduled/automated runs

### When to Use Notebook

- Interactive exploration
- Testing parameters
- Visualizing results
- Learning the system
- Single small document
- Need visual feedback

## Complete Example

```bash
# 1. Extract KG from corpus
python kg_extract.py \
    --input-dir antiquities_corpus/ \
    --output antiquities_kg \
    --export-format json csv neo4j

# 2. Train embeddings
python kg_embedding_analysis.py \
    --input antiquities_kg/edges_table.csv \
    --output antiquities_embeddings \
    --models TransE DistMult

# 3. Analyze results
python analyze_results.py \
    --kg antiquities_kg/knowledge_graph.json \
    --embeddings antiquities_embeddings/entity_embeddings.csv
```

