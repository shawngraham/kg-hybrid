#!/usr/bin/env python3
"""
Knowledge Graph Extraction - Command Line Script
Hybrid spaCy + Local LLM pipeline for extracting knowledge graphs from documents.

Usage:
    python kg_extract.py --input document.txt --output results/
    python kg_extract.py --input-dir documents/ --output results/
    python kg_extract.py --input document.txt --no-llm  # Skip LLM refinement
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import gc

import spacy
import networkx as nx
import requests
from tqdm import tqdm

# Configuration
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200
LOCAL_API_URL = "http://localhost:1337/v1/chat/completions"
LOCAL_MODEL_NAME = "jan-nano-128k-Q4_K_M"
LOCAL_API_KEY = ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Split text into overlapping chunks at sentence boundaries."""
    if len(text) <= chunk_size:
        return [{'text': text, 'start': 0, 'end': len(text), 'chunk_id': 0}]
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            search_start = max(end - 200, start)
            sentence_endings = [m.end() for m in re.finditer(r'[.!?]\s+', text[search_start:end])]
            
            if sentence_endings:
                end = search_start + sentence_endings[-1]
        
        chunk_text = text[start:end]
        chunks.append({
            'text': chunk_text,
            'start': start,
            'end': end,
            'chunk_id': chunk_id
        })
        
        start = end - overlap
        chunk_id += 1
    
    return chunks


def read_document(file_path: str) -> str:
    """Read document from file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text


def read_documents_from_directory(directory: str, extensions: List[str] = ['.txt', '.md']) -> Dict[str, str]:
    """Read all documents from a directory."""
    dir_path = Path(directory)
    documents = {}
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    for file_path in dir_path.iterdir():
        if file_path.suffix in extensions and file_path.is_file():
            try:
                text = read_document(str(file_path))
                documents[file_path.name] = text
                print(f"  Loaded {file_path.name}: {len(text)} characters")
            except Exception as e:
                print(f"  Warning: Could not read {file_path.name}: {e}")
    
    return documents


def find_entity_for_span(token, entities):
    """Find which entity contains or matches this token."""
    token_start = token.idx
    token_end = token.idx + len(token.text)
    
    # Exact match
    for ent in entities:
        if token_start >= ent['start'] and token_end <= ent['end']:
            return ent['text']
    
    # Partial match
    for ent in entities:
        if (token_start >= ent['start'] and token_start < ent['end']) or \
           (token_end > ent['start'] and token_end <= ent['end']):
            return ent['text']
    
    # Text match
    token_text_lower = token.text.lower()
    for ent in entities:
        if token_text_lower in ent['text'].lower() or ent['text'].lower() in token_text_lower:
            return ent['text']
    
    return None


def get_entity_head(span, entities):
    """Get the head token of a span and find its corresponding entity."""
    if hasattr(span, 'root'):
        return find_entity_for_span(span.root, entities)
    return find_entity_for_span(span, entities)


def extract_entities_with_spacy(text, nlp):
    """Stage 1: Enhanced entity and relationship extraction with spaCy"""
    doc = nlp(text)
    
    # Extract entities
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })
    
    # Extract relationships
    relationships = []
    transaction_verbs = {
        'sell', 'buy', 'acquire', 'purchase', 'transfer', 'export', 'import', 
        'smuggle', 'consign', 'operate', 'deal', 'trade', 'traffic', 'supply',
        'provide', 'convict', 'charge', 'sentence', 'arrest', 'raid', 'open',
        'close', 'found', 'establish', 'meet', 'return', 'transport'
    }
    
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma not in transaction_verbs:
            continue
        
        subject = None
        obj = None
        prep_obj = None
        
        for child in token.children:
            if child.dep_ in ['nsubj', 'nsubjpass']:
                subject = child
            elif child.dep_ in ['dobj', 'obj']:
                obj = child
            elif child.dep_ == 'prep':
                for pchild in child.children:
                    if pchild.dep_ == 'pobj':
                        prep_obj = pchild
                        break
            elif child.dep_ == 'dative':
                if not obj:
                    obj = child
        
        relationships_to_add = []
        
        if subject and obj:
            subj_entity = get_entity_head(subject, entities)
            obj_entity = get_entity_head(obj, entities)
            
            if subj_entity and obj_entity:
                relationships_to_add.append({
                    'verb': token.text,
                    'lemma': lemma,
                    'subject': subj_entity,
                    'object': obj_entity,
                    'pattern': 'subj-verb-obj'
                })
        
        if subject and prep_obj:
            subj_entity = get_entity_head(subject, entities)
            prep_obj_entity = get_entity_head(prep_obj, entities)
            
            if subj_entity and prep_obj_entity:
                relationships_to_add.append({
                    'verb': token.text,
                    'lemma': lemma,
                    'subject': subj_entity,
                    'object': prep_obj_entity,
                    'pattern': 'subj-verb-prep-obj'
                })
        
        if obj and prep_obj and not subject:
            obj_entity = get_entity_head(obj, entities)
            prep_obj_entity = get_entity_head(prep_obj, entities)
            
            if obj_entity and prep_obj_entity:
                relationships_to_add.append({
                    'verb': token.text,
                    'lemma': lemma,
                    'subject': obj_entity,
                    'object': prep_obj_entity,
                    'pattern': 'obj-verb-prep-obj'
                })
        
        relationships.extend(relationships_to_add)
    
    # Deduplicate
    unique_rels = []
    seen = set()
    for rel in relationships:
        key = (rel['subject'], rel['lemma'], rel['object'])
        if key not in seen:
            seen.add(key)
            unique_rels.append(rel)
    
    return {
        "entities": entities,
        "relationships": unique_rels,
        "text": text
    }


REFINEMENT_SCHEMA = """
## Task: Refine and Enhance Extracted Entities and Relationships

You are given entities and relationships extracted by spaCy from an antiquities trafficking document.
Your task is to:

1. **Resolve Coreferences**: Identify which entity mentions refer to the same real-world entity
2. **Canonicalize**: Assign canonical IDs (e.g., "giacomo_medici", "j_paul_getty_museum")
3. **Classify**: Map spaCy labels to domain-specific types
4. **Enhance Relationships**: Link to canonical entity IDs, extract dates, amounts
5. **Add Context**: Include roles, dates, legal status when available

## Output Format

Return JSON with two sections:

```json
{
  "entities": [
    {
      "canonical_id": "giacomo_medici",
      "full_name": "Giacomo Medici",
      "type": "PERSON",
      "mentions": ["Giacomo Medici", "Medici"],
      "attributes": {
        "role": "dealer",
        "nationality": "Italian"
      }
    }
  ],
  "relationships": [
    {
      "source_id": "giacomo_medici",
      "target_id": "robert_hecht",
      "relation_type": "sold_to",
      "attributes": {
        "date": "1967"
      }
    }
  ]
}
```
"""


def call_local_llm(prompt, timeout=300):
    """Call local jan.ai model"""
    try:
        response = requests.post(
            LOCAL_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LOCAL_API_KEY}"
            },
            json={
                "messages": [
                    {"role": "system", "content": "You are an expert in knowledge graph construction and antiquities trafficking domain analysis."},
                    {"role": "user", "content": prompt}
                ],
                "model": LOCAL_MODEL_NAME,
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 8000
            },
            timeout=timeout
        )
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    except Exception as e:
        print(f"Error calling local LLM: {e}")
        return None


def llm_refine_entities(spacy_extraction):
    """Stage 2: Use local LLM to refine spaCy extractions"""
    entities = spacy_extraction["entities"]
    relationships = spacy_extraction["relationships"]
    text = spacy_extraction["text"]
    
    entity_summary = "\n".join([f"- {e['text']} ({e['label']})" for e in entities[:100]])
    rel_summary = "\n".join([
        f"- {r['subject']} --[{r['verb']}]--> {r['object']}" 
        for r in relationships
    ])
    
    prompt = f"""{REFINEMENT_SCHEMA}

## Input Data

### Document Text (first 8000 chars):
{text[:8000]}...

### Entities Extracted by spaCy:
{entity_summary}

### Relationships Extracted by spaCy:
{rel_summary}

## Instructions

Analyze the above data and return ONLY a JSON object with "entities" and "relationships" arrays.
"""
    
    result_text = call_local_llm(prompt)
    
    if not result_text:
        return None
    
    try:
        result_text = result_text.strip()
        if result_text.startswith('```'):
            result_text = re.sub(r'^```json\s*', '', result_text)
            result_text = re.sub(r'^```\s*', '', result_text)
            result_text = re.sub(r'```\s*$', '', result_text)
        
        refined_data = json.loads(result_text)
        return refined_data
        
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return None


def merge_knowledge_graphs(kg_list: List[Dict]) -> Dict:
    """Merge multiple knowledge graphs"""
    merged_nodes = {}
    merged_edges = []
    edge_set = set()
    
    for kg in kg_list:
        if not kg:
            continue
        
        # Merge nodes
        for node_id, node_data in kg.get('nodes', {}).items():
            if node_id not in merged_nodes:
                merged_nodes[node_id] = node_data
            else:
                existing_mentions = set(merged_nodes[node_id].get('mentions', []))
                new_mentions = set(node_data.get('mentions', []))
                merged_nodes[node_id]['mentions'] = list(existing_mentions | new_mentions)
                merged_nodes[node_id]['attributes'].update(node_data.get('attributes', {}))
        
        # Merge edges
        for edge in kg.get('edges', []):
            edge_key = (edge['source'], edge['relation'], edge['target'])
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                merged_edges.append(edge)
    
    return {
        'nodes': merged_nodes,
        'edges': merged_edges
    }


def build_kg_from_refined_data(refined_data):
    """Build knowledge graph from LLM-refined entities and relationships"""
    if not refined_data:
        return None
    
    nodes = {}
    edges = []
    
    for entity in refined_data.get('entities', []):
        canonical_id = entity.get('canonical_id')
        if not canonical_id:
            continue
        
        nodes[canonical_id] = {
            'id': canonical_id,
            'type': entity.get('type', 'UNKNOWN'),
            'label': entity.get('full_name', canonical_id),
            'mentions': entity.get('mentions', []),
            'attributes': entity.get('attributes', {})
        }
    
    for rel in refined_data.get('relationships', []):
        source_id = rel.get('source_id')
        target_id = rel.get('target_id')
        
        if not source_id or not target_id:
            continue
        
        edges.append({
            'source': source_id,
            'target': target_id,
            'relation': rel.get('relation_type', 'related_to'),
            'attributes': rel.get('attributes', {})
        })
    
    return {
        'nodes': nodes,
        'edges': edges
    }


def build_kg_from_spacy_data(spacy_extraction):
    """Build simpler KG directly from spaCy extraction (when skipping LLM)"""
    entities = spacy_extraction["entities"]
    relationships = spacy_extraction["relationships"]
    
    nodes = {}
    edges = []
    
    # Create nodes from entities
    for ent in entities:
        node_id = ent['text'].lower().replace(' ', '_')
        if node_id not in nodes:
            nodes[node_id] = {
                'id': node_id,
                'type': ent['label'],
                'label': ent['text'],
                'mentions': [ent['text']],
                'attributes': {}
            }
    
    # Create edges from relationships
    for rel in relationships:
        source_id = rel['subject'].lower().replace(' ', '_')
        target_id = rel['object'].lower().replace(' ', '_')
        
        edges.append({
            'source': source_id,
            'target': target_id,
            'relation': rel['lemma'],
            'attributes': {'verb': rel['verb']}
        })
    
    return {
        'nodes': nodes,
        'edges': edges
    }


def save_kg_json(kg, filename):
    """Save knowledge graph as JSON"""
    if not kg:
        return
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)


def export_kg_to_csv(kg, output_dir):
    """Export knowledge graph to CSV files"""
    import csv
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Export nodes
    nodes_path = output_dir / 'nodes_table.csv'
    with open(nodes_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'id', 'label', 'mentions', 'attributes'])
        
        for node_id, node_data in kg['nodes'].items():
            writer.writerow([
                node_data['type'],
                node_data['id'],
                node_data['label'],
                ', '.join(node_data['mentions']) if node_data['mentions'] else '',
                str(node_data.get('attributes', {}))
            ])
    
    # Export edges
    edges_path = output_dir / 'edges_table.csv'
    with open(edges_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target', 'relation', 'attributes'])
        
        for edge in kg['edges']:
            writer.writerow([
                edge['source'],
                edge['target'],
                edge['relation'],
                str(edge.get('attributes', {}))
            ])


def export_to_neo4j_cypher(kg, filename):
    """Export as Neo4j Cypher statements"""
    if not kg:
        return
    
    cypher = []
    
    for node_id, node in kg['nodes'].items():
        label = node['type']
        props = {
            'id': node_id,
            'label': node['label'],
            'mentions': node.get('mentions', [])
        }
        props.update(node.get('attributes', {}))
        
        props_str = ', '.join([f"{k}: {json.dumps(v)}" for k, v in props.items()])
        cypher.append(f"CREATE (:{label} {{{props_str}}})")
    
    for edge in kg['edges']:
        rel_type = edge['relation'].replace(' ', '_').replace('(', '').replace(')', '').upper()
        props_str = ', '.join([f"{k}: {json.dumps(v)}" 
                               for k, v in edge.get('attributes', {}).items()])
        
        cypher.append(
            f"MATCH (a {{id: {json.dumps(edge['source'])}}}), "
            f"(b {{id: {json.dumps(edge['target'])}}}) "
            f"CREATE (a)-[:{rel_type} {{{props_str}}}]->(b)"
        )
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(cypher))


def display_kg_summary(kg):
    """Display text summary of knowledge graph"""
    if not kg:
        return
    
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH SUMMARY")
    print("="*60)
    
    by_type = defaultdict(list)
    for node_id, node in kg['nodes'].items():
        by_type[node['type']].append((node_id, node))
    
    for entity_type, nodes_list in sorted(by_type.items()):
        print(f"\n{entity_type} ({len(nodes_list)}):")
        for node_id, node in sorted(nodes_list)[:10]:
            label = node['label']
            mentions = len(node.get('mentions', []))
            print(f"  • {label} (id: {node_id}, {mentions} mentions)")
    
    print(f"\nRELATIONSHIPS ({len(kg['edges'])}):")
    for edge in kg['edges'][:15]:
        source_label = kg['nodes'].get(edge['source'], {}).get('label', edge['source'])
        target_label = kg['nodes'].get(edge['target'], {}).get('label', edge['target'])
        print(f"  • {source_label} --[{edge['relation']}]--> {target_label}")
    
    if len(kg['edges']) > 15:
        print(f"  ... and {len(kg['edges']) - 15} more")
    
    print("\n" + "="*60)


def run_hybrid_pipeline(document_text, nlp, use_llm=True):
    """Run the complete two-stage hybrid pipeline"""
    print("\n" + "="*60)
    print("STAGE 1: spaCy Fast Extraction")
    print("="*60)
    
    spacy_extraction = extract_entities_with_spacy(document_text, nlp)
    print(f"Extracted {len(spacy_extraction['entities'])} entities and {len(spacy_extraction['relationships'])} relationships")
    
    if not use_llm:
        print("\nSkipping LLM refinement")
        kg = build_kg_from_spacy_data(spacy_extraction)
        return kg, None
    
    print("\n" + "="*60)
    print("STAGE 2: Local LLM Refinement")
    print("="*60)
    
    refined_data = llm_refine_entities(spacy_extraction)
    
    if not refined_data:
        print("LLM refinement failed, using spaCy-only results")
        kg = build_kg_from_spacy_data(spacy_extraction)
        return kg, None
    
    print(f"LLM refined {len(refined_data.get('entities', []))} entities")
    print(f"LLM enhanced {len(refined_data.get('relationships', []))} relationships")
    
    print("\n" + "="*60)
    print("STAGE 3: Knowledge Graph Construction")
    print("="*60)
    
    kg = build_kg_from_refined_data(refined_data)
    
    return kg, refined_data


def process_document_with_chunks(text, nlp, use_llm=True):
    """Process a large document by chunking"""
    chunks = chunk_text(text)
    print(f"\nSplit document into {len(chunks)} chunks")
    
    all_kgs = []
    
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        print(f"\n--- Chunk {i+1}/{len(chunks)} ---")
        kg, _ = run_hybrid_pipeline(chunk['text'], nlp, use_llm=use_llm)
        
        if kg:
            all_kgs.append(kg)
        
        # Force garbage collection
        gc.collect()
    
    if all_kgs:
        print("\nMerging chunks...")
        merged_kg = merge_knowledge_graphs(all_kgs)
        return merged_kg
    
    return None


def process_multiple_documents(documents: Dict[str, str], nlp, use_llm=True):
    """Process multiple documents"""
    all_kgs = []
    
    for i, (filename, text) in enumerate(documents.items()):
        print(f"\n{'='*70}")
        print(f"Processing document {i+1}/{len(documents)}: {filename}")
        print(f"{'='*70}")
        
        if len(text) > CHUNK_SIZE:
            kg = process_document_with_chunks(text, nlp, use_llm=use_llm)
        else:
            kg, _ = run_hybrid_pipeline(text, nlp, use_llm=use_llm)
        
        if kg:
            all_kgs.append(kg)
        
        gc.collect()
    
    if all_kgs:
        print("\nMerging all documents...")
        merged_kg = merge_knowledge_graphs(all_kgs)
        return merged_kg
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Knowledge Graph Extraction from Documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single document with LLM refinement
  python kg_extract.py --input document.txt --output results/
  
  # Directory of documents without LLM
  python kg_extract.py --input-dir documents/ --output results/ --no-llm
  
  # Custom chunk size
  python kg_extract.py --input large_doc.txt --chunk-size 8000 --output results/
        """
    )
    
    parser.add_argument('--input', help='Input document file')
    parser.add_argument('--input-dir', help='Input directory with documents')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM refinement (faster)')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help='Chunk size for large documents')
    parser.add_argument('--spacy-model', default='en_core_web_trf', help='spaCy model to use')
    parser.add_argument('--api-url', default=LOCAL_API_URL, help='Local LLM API URL')
    parser.add_argument('--api-key', default=LOCAL_API_KEY, help='API key for local LLM')
    parser.add_argument('--export-format', nargs='+', 
                       choices=['json', 'csv', 'neo4j'], 
                       default=['json', 'csv'],
                       help='Export formats')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir must be specified")
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    global CHUNK_SIZE, LOCAL_API_URL, LOCAL_API_KEY
    CHUNK_SIZE = args.chunk_size
    LOCAL_API_URL = args.api_url
    LOCAL_API_KEY = args.api_key
    
    # Load spaCy model
    print(f"Loading spaCy model: {args.spacy_model}...")
    try:
        nlp = spacy.load(args.spacy_model)
        print("spaCy model loaded")
    except OSError:
        print(f"Model {args.spacy_model} not found. Downloading...")
        import subprocess
        subprocess.run(['python', '-m', 'spacy', 'download', args.spacy_model])
        nlp = spacy.load(args.spacy_model)
    
    # Process documents
    if args.input:
        print(f"\nReading document: {args.input}")
        text = read_document(args.input)
        print(f"Document size: {len(text)} characters")
        
        if len(text) > args.chunk_size:
            kg = process_document_with_chunks(text, nlp, use_llm=not args.no_llm)
        else:
            kg, _ = run_hybrid_pipeline(text, nlp, use_llm=not args.no_llm)
    
    elif args.input_dir:
        print(f"\nReading documents from: {args.input_dir}")
        documents = read_documents_from_directory(args.input_dir)
        print(f"Found {len(documents)} documents")
        
        kg = process_multiple_documents(documents, nlp, use_llm=not args.no_llm)
    
    # Export results
    if kg:
        print("\n" + "="*60)
        print("EXPORTING RESULTS")
        print("="*60)
        
        display_kg_summary(kg)
        
        if 'json' in args.export_format:
            json_path = output_dir / 'knowledge_graph.json'
            save_kg_json(kg, json_path)
            print(f"\nSaved JSON: {json_path}")
        
        if 'csv' in args.export_format:
            export_kg_to_csv(kg, output_dir)
            print(f"Saved CSV tables: {output_dir}/nodes_table.csv, {output_dir}/edges_table.csv")
        
        if 'neo4j' in args.export_format:
            cypher_path = output_dir / 'neo4j_import.cypher'
            export_to_neo4j_cypher(kg, cypher_path)
            print(f"Saved Neo4j Cypher: {cypher_path}")
        
        print(f"\n{'='*60}")
        print(f"Extraction complete!")
        print(f"  Nodes: {len(kg['nodes'])}")
        print(f"  Edges: {len(kg['edges'])}")
        print(f"  Output directory: {output_dir.absolute()}")
        print(f"{'='*60}")
    else:
        print("\nNo knowledge graph generated")
        sys.exit(1)


if __name__ == '__main__':
    main()
