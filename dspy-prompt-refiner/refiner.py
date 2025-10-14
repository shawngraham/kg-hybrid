"""
DSPy-based Optimization Framework for Hybrid Knowledge Graph Extraction

This script optimizes the LLM-based refinement stage using DSPy with multiple local models.
"""

import dspy
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import spacy
import requests
from collections import defaultdict
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LocalLLMConfig:
    """Configuration for a local LLM"""
    name: str
    api_url: str = "http://localhost:1337/v1/chat/completions"
    api_key: str = ""
    max_tokens: int = 4096
    temperature: float = 0.1
    
    def __str__(self):
        return self.name


def load_models_from_config():
    """Load enabled models from config.py"""
    try:
        from config import MODELS_CONFIG
        
        models = []
        for model_id, config in MODELS_CONFIG.items():
            if config.get('enabled', True):
                models.append(LocalLLMConfig(
                    name=config['name'],
                    api_url=config.get('api_url', 'http://localhost:1337/v1/chat/completions'),
                    api_key=config.get('api_key', ''),
                    max_tokens=config.get('max_tokens', 4096),
                    temperature=config.get('temperature', 0.1)
                ))
        
        if not models:
            print("⚠️  No enabled models found in config.py")
            print("   Enable at least one model by setting 'enabled': True")
        
        return models
    
    except ImportError:
        print("⚠️  Could not import config.py, using default models")
        return [
            LocalLLMConfig(name="openhermes-neural-7b", max_tokens=32768),
            LocalLLMConfig(name="llama2-chat-7b-q4", max_tokens=4096),
        ]


# Load models from config
LOCAL_MODELS = load_models_from_config()


# ============================================================================
# JSON Extraction Utility (FIXED - moved outside of class)
# ============================================================================

def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Aggressively extract JSON from text that might contain:
    - Conversational preamble/postamble
    - Markdown code blocks
    - Mixed content
    """
    if not text or not text.strip():
        return None
    
    # Method 1: Try to find JSON in markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    json_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    matches = re.findall(json_block_pattern, text, re.IGNORECASE)
    
    if matches:
        # Try each matched block
        for match in matches:
            cleaned = match.strip()
            if cleaned.startswith('{') or cleaned.startswith('['):
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue
    
    # Method 2: Look for JSON object {...} or array [...]
    # Find the longest valid JSON structure
    for pattern in [r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]']:
        matches = re.finditer(pattern, text, re.DOTALL)
        candidates = []
        
        for match in matches:
            json_str = match.group(0)
            try:
                parsed = json.loads(json_str)
                candidates.append((len(json_str), parsed))
            except json.JSONDecodeError:
                continue
        
        # Return the longest valid JSON found
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
    
    # Method 3: Try to parse the entire text after cleaning
    cleaned_text = text.strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        r'^here\s+is\s+the\s+json.*?[:\n]',
        r'^here\s+are\s+the\s+.*?[:\n]',
        r'^the\s+json\s+is.*?[:\n]',
        r'^response.*?[:\n]',
        r'^output.*?[:\n]',
        r'^sure[,!.\s]+',
        r'^okay[,!.\s]+',
    ]
    
    for prefix in prefixes_to_remove:
        cleaned_text = re.sub(prefix, '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
    
    cleaned_text = cleaned_text.strip()
    
    # Try direct parse
    if cleaned_text.startswith('{') or cleaned_text.startswith('['):
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass
    
    return None


# ============================================================================
# DSPy Signatures for KG Extraction Tasks
# ============================================================================

class CustomEntityRefiner(dspy.Module):
    """Custom refiner that bypasses DSPy's JSON parsing and handles raw LLM output."""
    
    def __init__(self):
        super().__init__()
        # Don't use dspy.Predict - we'll call the LM directly
        
    def forward(self, document_text, spacy_entities, spacy_relationships):
        try:
            # Get the configured LM from DSPy settings
            lm = dspy.settings.lm
            
            # Build the prompt manually
            prompt = f"""You are an expert in knowledge graph construction and entity extraction.

Task: Refine and canonicalize the entities and relationships extracted by spaCy.

Document Text:
{document_text}

Entities extracted by spaCy:
{spacy_entities}

Relationships extracted by spaCy:
{spacy_relationships}

Instructions:
1. Create canonical IDs for entities (lowercase, underscores, e.g., "giacomo_medici")
2. Resolve coreferences (e.g., "he" → "giacomo_medici")
3. Add domain-specific attributes to entities
4. Enhance relationships with the canonical entity IDs

Output a SINGLE JSON object with this exact structure:
{{
  "entities": [
    {{
      "canonical_id": "entity_id",
      "type": "PERSON|ORG|GPE|etc",
      "mentions": ["mention1", "mention2"],
      "attributes": {{"key": "value"}}
    }}
  ],
  "relationships": [
    {{
      "subject": "canonical_id_1",
      "relation_type": "SOLD_TO|ACQUIRED|etc",
      "object": "canonical_id_2",
      "attributes": {{"date": "1972", "location": "Geneva"}}
    }}
  ]
}}

Return ONLY the JSON object, no other text."""

            print("\n" + "="*60)
            print("CALLING LLM")
            print("="*60)
            print(f"Prompt length: {len(prompt)} chars")
            
            # Call LM directly
            raw_response = lm(prompt=prompt)
            
            print("\n" + "="*60)
            print("RAW LLM RESPONSE")
            print("="*60)
            print(f"Response length: {len(raw_response)} chars")
            print(f"First 500 chars: {raw_response[:500]}")
            
            # Extract JSON using our robust extractor
            parsed_json = extract_json_from_text(raw_response)
            
            if parsed_json is None:
                print("❌ Failed to extract JSON from response")
                print(f"Full response:\n{raw_response}")
                # Return empty structure
                parsed_json = {"entities": [], "relationships": []}
            
            # Return a simple object that mimics DSPy's output
            class Result:
                def __init__(self, data):
                    self.refined_kg = data
            
            return Result(parsed_json)
            
        except Exception as e:
            print(f"Error in CustomEntityRefiner.forward: {e}")
            import traceback
            traceback.print_exc()
            raise


class EntityRefinement(dspy.Signature):
    """Refine and canonicalize entities extracted by spaCy.
    
    Given entities and text, produce canonical IDs, resolve coreferences,
    and add domain-specific attributes. Return both entities and relationships in a single JSON object.
    
    NOTE: This signature is currently NOT USED because we bypass DSPy's automatic parsing
    (which fails on conversational LLM responses). See CustomEntityRefiner for manual approach.
    """
    
    document_text = dspy.InputField(desc="The source document text")
    spacy_entities = dspy.InputField(desc="Entities extracted by spaCy with labels")
    spacy_relationships = dspy.InputField(desc="Relationships extracted by spaCy")
    
    refined_kg = dspy.OutputField(desc='JSON object with two keys: "entities" (array of refined entities with canonical_id, type, mentions, attributes) and "relationships" (array of relationships with canonical entity IDs). Format: {"entities": [...], "relationships": [...]}')


class CoreferenceResolution(dspy.Signature):
    """Resolve which entity mentions refer to the same real-world entity."""
    
    entities = dspy.InputField(desc="List of entity mentions with context")
    document_text = dspy.InputField(desc="Full document text for context")
    
    coreference_clusters = dspy.OutputField(desc="JSON mapping of canonical IDs to mention clusters")


class RelationshipEnhancement(dspy.Signature):
    """Enhance relationships with attributes like dates, amounts, and context."""
    
    relationship = dspy.InputField(desc="Basic relationship (subject, verb, object)")
    document_text = dspy.InputField(desc="Document text for context")
    entities_info = dspy.InputField(desc="Information about the entities involved")
    
    enhanced_relationship = dspy.OutputField(desc="Relationship with type, attributes, and metadata")


# ============================================================================
# DSPy Modules for KG Pipeline
# ============================================================================

class KGRefiner(dspy.Module):
    """Main module for refining knowledge graph extraction."""
    
    def __init__(self):
        super().__init__()
        self.entity_refiner = CustomEntityRefiner()
    
    def forward(self, document_text, spacy_entities, spacy_relationships):
        print("document_text:", document_text) 
        print("spacy_entities:", spacy_entities) 
        print("spacy_relationships:", spacy_relationships) 
        # Format inputs
        entities_str = "\n".join([
            f"- {e['text']} ({e['label']})" 
            for e in spacy_entities[:100]
        ])
        
        if not spacy_relationships:
            rels_str = ""
        else:
            rels_str = "\n".join([
                f"- {r['subject']} --[{r['verb']}]--> {r['object']}"
                for r in spacy_relationships[:50]
            ])
        
        # Get refinement (now returns single combined output)
        result = self.entity_refiner(
            document_text=document_text,
            spacy_entities=entities_str,
            spacy_relationships=rels_str
        )
        print("Result from refiner:", result)
        return result


class CoreferenceResolver(dspy.Module):
    """Resolve coreferences in entity mentions."""
    
    def __init__(self):
        super().__init__()
        self.resolver = dspy.ChainOfThought(CoreferenceResolution)
    
    def forward(self, entities, document_text):
        entities_str = json.dumps(entities, indent=2)
        result = self.resolver(
            entities=entities_str,
            document_text=document_text[:4000]
        )
        return result


# ============================================================================
# Custom DSPy LM for Local Models (FIXED)
# ============================================================================

class LocalLLM(dspy.LM):
    """DSPy Language Model wrapper for local LLMs via OpenAI-compatible API."""

    def __init__(self, config: LocalLLMConfig, **kwargs):
        super().__init__(model=config.name)
        self.config = config
        self.history = []
        self.kwargs = {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            **kwargs
        }

    def _make_request(self, prompt: str, **kwargs) -> str:
        """Internal method to make the actual API request."""
        print("LocalLLM._make_request called!")
        print("  Prompt:", prompt[:200], "...")
        
        # Merge instance kwargs with call-time kwargs
        merged_kwargs = {**self.kwargs, **kwargs}
        response = None
        
        try:
            response = requests.post(
                self.config.api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.config.api_key}"
                },
                json={
                    "messages": [
                        {"role": "system", "content": "You are an expert in knowledge graph construction and entity extraction. Always return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "model": self.config.name,
                    "stream": False,
                    "temperature": merged_kwargs.get('temperature', self.config.temperature),
                    "max_tokens": merged_kwargs.get('max_tokens', self.config.max_tokens)
                },
                timeout=300
            )

            response.raise_for_status()
            
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
                print(f"Raw response text: {response.text}")
                raise

            # Parse response
            if "choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    
                    # Store in history
                    self.history.append({
                        'prompt': prompt,
                        'response': content,
                        'model': self.config.name
                    })
                    
                    return content
                else:
                    print(f"Unexpected 'choice' format: {choice}")
                    raise ValueError(f"Unexpected 'choice' format: {choice}")
            else:
                print(f"Unexpected response format: {result}")
                raise ValueError(f"Unexpected response format: {result}")

        except requests.exceptions.RequestException as e:
            print(f"Error calling {self.config.name}: {e}")
            return ""
        except (KeyError, IndexError, ValueError) as e:
            print(f"Error parsing response from {self.config.name}: {e}")
            if response is not None:
                print("Response content:", response.text)
            return ""

    def __call__(self, prompt=None, messages=None, **kwargs):
        """DSPy-compatible __call__ interface."""
        # Handle both prompt-based and messages-based calls
        if prompt is not None:
            return self._make_request(prompt, **kwargs)
        elif messages is not None:
            # Convert messages to a single prompt
            prompt_parts = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                prompt_parts.append(f"{role}: {content}")
            combined_prompt = "\n".join(prompt_parts)
            return self._make_request(combined_prompt, **kwargs)
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

    def basic_request(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """DSPy compatibility method."""
        response_text = self._make_request(prompt, **kwargs)
        return {"choices": [{"text": response_text}]}


# ============================================================================
# Evaluation Metrics
# ============================================================================

@dataclass
class KGEvaluationMetrics:
    """Metrics for evaluating knowledge graph quality."""
    
    # Entity metrics
    num_entities: int
    num_canonical_ids: int
    avg_mentions_per_entity: float
    entity_types_coverage: Dict[str, int]
    
    # Relationship metrics
    num_relationships: int
    num_typed_relationships: int
    avg_attributes_per_relationship: float
    
    # Quality metrics
    coreference_resolution_score: float
    canonical_id_quality: float
    attribute_completeness: float
    
    # Performance metrics
    extraction_time: float
    
    def overall_score(self) -> float:
        """Compute overall quality score (0-100)."""
        weights = {
            'coreference': 0.3,
            'canonical_id': 0.2,
            'attributes': 0.2,
            'coverage': 0.15,
            'relationships': 0.15
        }
        
        coverage_score = min(self.num_entities / 50, 1.0)
        rel_score = min(self.num_relationships / 30, 1.0)
        
        score = (
            weights['coreference'] * self.coreference_resolution_score +
            weights['canonical_id'] * self.canonical_id_quality +
            weights['attributes'] * self.attribute_completeness +
            weights['coverage'] * coverage_score +
            weights['relationships'] * rel_score
        ) * 100
        
        return round(score, 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'num_entities': self.num_entities,
            'num_canonical_ids': self.num_canonical_ids,
            'avg_mentions_per_entity': round(self.avg_mentions_per_entity, 2),
            'entity_types_coverage': self.entity_types_coverage,
            'num_relationships': self.num_relationships,
            'num_typed_relationships': self.num_typed_relationships,
            'avg_attributes_per_relationship': round(self.avg_attributes_per_relationship, 2),
            'coreference_resolution_score': round(self.coreference_resolution_score, 2),
            'canonical_id_quality': round(self.canonical_id_quality, 2),
            'attribute_completeness': round(self.attribute_completeness, 2),
            'extraction_time': round(self.extraction_time, 2),
            'overall_score': self.overall_score()
        }


def evaluate_kg_quality(refined_data: Dict, ground_truth: Optional[Dict] = None) -> KGEvaluationMetrics:
    """Evaluate the quality of extracted knowledge graph."""
    
    entities = refined_data.get('entities', [])
    relationships = refined_data.get('relationships', [])
    
    # Entity metrics
    num_entities = len(entities)
    canonical_ids = [e.get('canonical_id') for e in entities if e.get('canonical_id')]
    num_canonical_ids = len(canonical_ids)
    
    total_mentions = sum(len(e.get('mentions', [])) for e in entities)
    avg_mentions = total_mentions / num_entities if num_entities > 0 else 0
    
    entity_types = defaultdict(int)
    for e in entities:
        entity_types[e.get('type', 'UNKNOWN')] += 1
    
    # Relationship metrics
    num_relationships = len(relationships)
    typed_rels = [r for r in relationships if r.get('relation_type')]
    num_typed_relationships = len(typed_rels)
    
    total_rel_attrs = sum(len(r.get('attributes', {})) for r in relationships)
    avg_rel_attrs = total_rel_attrs / num_relationships if num_relationships > 0 else 0
    
    # Quality metrics
    valid_canonical_ids = sum(
        1 for cid in canonical_ids 
        if cid and re.match(r'^[a-z0-9_]+$', cid)
    )
    canonical_id_quality = valid_canonical_ids / len(canonical_ids) if canonical_ids else 0
    
    entities_with_attrs = sum(
        1 for e in entities 
        if e.get('attributes') and len(e.get('attributes', {})) > 0
    )
    attribute_completeness = entities_with_attrs / num_entities if num_entities > 0 else 0
    
    coreference_score = min(avg_mentions / 2.0, 1.0)
    
    return KGEvaluationMetrics(
        num_entities=num_entities,
        num_canonical_ids=num_canonical_ids,
        avg_mentions_per_entity=avg_mentions,
        entity_types_coverage=dict(entity_types),
        num_relationships=num_relationships,
        num_typed_relationships=num_typed_relationships,
        avg_attributes_per_relationship=avg_rel_attrs,
        coreference_resolution_score=coreference_score,
        canonical_id_quality=canonical_id_quality,
        attribute_completeness=attribute_completeness,
        extraction_time=0.0
    )


# ============================================================================
# Dataset Creation
# ============================================================================

class KGDataset:
    """Dataset for training and evaluating KG extraction."""
    
    def __init__(self):
        self.examples = []
    
    def add_example(self, document_text: str, spacy_extraction: Dict, 
                   ground_truth: Optional[Dict] = None):
        """Add a training/test example."""
        self.examples.append({
            'document_text': document_text,
            'spacy_extraction': spacy_extraction,
            'ground_truth': ground_truth
        })
    
    def load_from_directory(self, directory: str):
        """Load examples from directory of annotated documents."""
        pass


# ============================================================================
# Benchmarking and Optimization (FIXED)
# ============================================================================

class KGOptimizer:
    """Optimize KG extraction across different local LLMs."""
    
    def __init__(self, models: List[LocalLLMConfig], dataset: KGDataset):
        self.models = models
        self.dataset = dataset
        self.results = []
        
        # Load spaCy model
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_trf")
        print("✅ spaCy loaded")
    
    def find_entity_for_span(self, token, entities):
        """Find which entity (if any) contains or matches this token/span."""
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

    def get_entity_head(self, span, entities):
        """Get the head token of a span and find its corresponding entity."""
        if hasattr(span, 'root'):
            return self.find_entity_for_span(span.root, entities)
        return self.find_entity_for_span(span, entities)

    def extract_with_spacy(self, text: str) -> Dict:
        """Extract entities and relationships with spaCy (Stage 1)."""
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
        
        # Relationship extraction
        relationships = []
        
        transaction_verbs = {
            'sell', 'buy', 'acquire', 'purchase', 'transfer', 'export', 'import', 
            'smuggle', 'consign', 'operate', 'deal', 'trade', 'traffic', 'supply',
            'provide', 'convict', 'charge', 'sentence', 'arrest', 'raid', 'open',
            'close', 'found', 'establish', 'meet', 'return', 'transport', 'loot', 'steal'
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
                subj_entity = self.get_entity_head(subject, entities)
                obj_entity = self.get_entity_head(obj, entities)
                
                if subj_entity and obj_entity:
                    relationships_to_add.append({
                        'verb': token.text,
                        'lemma': lemma,
                        'subject': subj_entity,
                        'object': obj_entity,
                        'pattern': 'subj-verb-obj'
                    })
            
            if subject and prep_obj:
                subj_entity = self.get_entity_head(subject, entities)
                prep_obj_entity = self.get_entity_head(prep_obj, entities)
                
                if subj_entity and prep_obj_entity:
                    relationships_to_add.append({
                        'verb': token.text,
                        'lemma': lemma,
                        'subject': subj_entity,
                        'object': prep_obj_entity,
                        'pattern': 'subj-verb-prep-obj'
                    })
            
            if obj and prep_obj and not subject:
                obj_entity = self.get_entity_head(obj, entities)
                prep_obj_entity = self.get_entity_head(prep_obj, entities)
                
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
        
        print(f"✅ spaCy extracted {len(entities)} entities and {len(unique_rels)} relationships")
        
        return {
            "entities": entities,
            "relationships": unique_rels,
            "text": text
        }
    
    def benchmark_model(self, config: LocalLLMConfig, example: Dict) -> Dict[str, Any]:
        """Benchmark a single model on one example."""
        import time
        
        print(f"\n{'='*60}")
        print(f"Testing: {config.name}")
        print(f"{'='*60}")
        
        # Configure DSPy with this model
        lm = LocalLLM(config)
        dspy.settings.configure(lm=lm)
        
        # Create module
        refiner = KGRefiner()
        
        # Run extraction
        start_time = time.time()
        
        spacy_extraction = example['spacy_extraction']
        
        try:
            result = refiner(
                document_text=example['document_text'],
                spacy_entities=spacy_extraction['entities'],
                spacy_relationships=spacy_extraction['relationships']
            )
            
            # Get the refined data - it's already parsed JSON now, not a string
            refined_data = result.refined_kg
            
            # Ensure it's a dict with the right keys
            if not isinstance(refined_data, dict):
                print(f"⚠️  refined_kg is not a dict: {type(refined_data)}")
                if isinstance(refined_data, str):
                    # Fallback: try to parse it
                    refined_data = extract_json_from_text(refined_data)
                    if refined_data is None:
                        refined_data = {'entities': [], 'relationships': []}
                else:
                    refined_data = {'entities': [], 'relationships': []}
            
            # Ensure we have both keys
            if 'entities' not in refined_data:
                refined_data['entities'] = []
            if 'relationships' not in refined_data:
                refined_data['relationships'] = []
            
            extraction_time = time.time() - start_time
            
            print(f"\n{'='*60}")
            print(f"EXTRACTION RESULTS")
            print(f"{'='*60}")
            print(f"Entities extracted: {len(refined_data['entities'])}")
            print(f"Relationships extracted: {len(refined_data['relationships'])}")
            print(f"Time taken: {extraction_time:.2f}s")
            
            # Evaluate
            if refined_data and (refined_data['entities'] or refined_data['relationships']):
                metrics = evaluate_kg_quality(refined_data, example.get('ground_truth'))
                metrics.extraction_time = extraction_time
                
                print(f"✅ Success! Overall Score: {metrics.overall_score()}")
                print(f"   Entities: {metrics.num_entities}, Relationships: {metrics.num_relationships}")
                
                return {
                    'model': config.name,
                    'success': True,
                    'metrics': metrics.to_dict(),
                    'refined_data': refined_data,
                    'extraction_time': extraction_time
                }
            else:
                print(f"❌ No entities or relationships extracted")
                return {
                    'model': config.name,
                    'success': False,
                    'error': 'Empty extraction'
                }
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'model': config.name,
                'success': False,
                'error': str(e)
            }
    
    def run_full_benchmark(self) -> pd.DataFrame:
        """Run complete benchmark across all models and examples."""
        
        results = []
        
        for example_idx, example in enumerate(self.dataset.examples):
            print(f"\n{'#'*60}")
            print(f"Example {example_idx + 1}/{len(self.dataset.examples)}")
            print(f"{'#'*60}")
            
            for model_config in self.models:
                result = self.benchmark_model(model_config, example)
                result['example_idx'] = example_idx
                results.append(result)
                
                self.results.append(result)
        
        # Create summary DataFrame
        df = pd.DataFrame([
            {
                'model': r['model'],
                'example': r['example_idx'],
                'success': r['success'],
                'overall_score': r.get('metrics', {}).get('overall_score', 0) if r['success'] else 0,
                'num_entities': r.get('metrics', {}).get('num_entities', 0) if r['success'] else 0,
                'num_relationships': r.get('metrics', {}).get('num_relationships', 0) if r['success'] else 0,
                'extraction_time': r.get('extraction_time', 0)
            }
            for r in results
        ])
        
        return df
    
    def generate_report(self, output_file: str = "kg_optimization_report.md"):
        """Generate a detailed markdown report of results."""
        
        df = pd.DataFrame([
            {
                'model': r['model'],
                'success': r['success'],
                **r.get('metrics', {})
            }
            for r in self.results if r['success']
        ])
        
        # Group by model
        summary = df.groupby('model').agg({
            'overall_score': ['mean', 'std'],
            'num_entities': 'mean',
            'num_relationships': 'mean',
            'extraction_time': 'mean'
        }).round(2)
        
        # Generate report
        with open(output_file, 'w') as f:
            f.write("# Knowledge Graph Extraction Optimization Report\n\n")
            f.write("## Summary Statistics by Model\n\n")
            f.write(summary.to_markdown())
            f.write("\n\n")
            
            f.write("## Top Performing Models\n\n")
            top_models = df.groupby('model')['overall_score'].mean().sort_values(ascending=False)
            for i, (model, score) in enumerate(top_models.head(3).items(), 1):
                f.write(f"{i}. **{model}**: {score:.2f}/100\n")
            
            f.write("\n## Detailed Results\n\n")
            for result in self.results:
                if result['success']:
                    f.write(f"### {result['model']}\n\n")
                    metrics = result['metrics']
                    f.write(f"- Overall Score: **{metrics['overall_score']}/100**\n")
                    f.write(f"- Entities: {metrics['num_entities']} ")
                    f.write(f"(Canonical IDs: {metrics['num_canonical_ids']})\n")
                    f.write(f"- Relationships: {metrics['num_relationships']}\n")
                    f.write(f"- Extraction Time: {metrics['extraction_time']:.2f}s\n")
                    f.write(f"- Quality Scores:\n")
                    f.write(f"  - Coreference: {metrics['coreference_resolution_score']:.2f}\n")
                    f.write(f"  - Canonical ID: {metrics['canonical_id_quality']:.2f}\n")
                    f.write(f"  - Attributes: {metrics['attribute_completeness']:.2f}\n")
                    f.write("\n")
        
        print(f"\n✅ Report saved to {output_file}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    
    print("="*60)
    print("DSPy Knowledge Graph Extraction Optimizer")
    print("="*60)
    
    # Display loaded models
    print(f"\n📋 Loaded {len(LOCAL_MODELS)} enabled model(s) from config.py:")
    for i, model in enumerate(LOCAL_MODELS, 1):
        print(f"   {i}. {model.name} (max_tokens: {model.max_tokens})")
    
    if not LOCAL_MODELS:
        print("\n❌ No enabled models found!")
        print("   Edit config.py and set 'enabled': True for at least one model")
        return
    
    # Create dataset
    dataset = KGDataset()
    
    # Add example
    example_text = """
    Giacomo Medici was convicted in 2004 for trafficking looted artifacts. 
    He sold numerous antiquities to Robert Hecht, who then supplied them to museums.
    The J. Paul Getty Museum acquired the Euphronios Sarpedon Krater from Hecht in 1972.
    Italian authorities raided Medici's warehouse in Geneva in 1995, finding thousands
    of photographs of looted artifacts.
    """
    
    optimizer = KGOptimizer(LOCAL_MODELS, dataset)
    spacy_extraction = optimizer.extract_with_spacy(example_text)
    
    dataset.add_example(
        document_text=example_text,
        spacy_extraction=spacy_extraction
    )
    
    # Run benchmark
    print("\nStarting benchmark...")
    results_df = optimizer.run_full_benchmark()
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(results_df.to_string())
    
    # Generate detailed report
    optimizer.generate_report()
    
    # Save results
    results_df.to_csv("kg_optimization_results.csv", index=False)
    print("\n✅ Results saved to kg_optimization_results.csv")


if __name__ == "__main__":
    main()
