import dspy
import spacy
import json
import re
import requests
from typing import List, Dict, Any
from dataclasses import dataclass

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LocalLLMConfig:
    """Configuration for a local LLM"""
    name: str
    api_url: str = "http://localhost:1337/v1/chat/completions"
    api_key: str = ""
    max_tokens: int = 8192
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
                
        return models
        
    except ImportError:
        return [LocalLLMConfig(name="openhermes-neural-7b")]

# ============================================================================
# LocalLLM
# ============================================================================

class LocalLLM(dspy.LM):
    """Simple LLM wrapper that returns raw text."""

    def __init__(self, config: LocalLLMConfig, **kwargs):
        super().__init__(model=config.name)
        self.config = config
        self.history = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        if prompt is None and messages is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        
        if messages is not None:
            prompt_parts = []
            for msg in messages:
                prompt_parts.append(f"{msg['role']}: {msg['content']}")
            prompt = "\n".join(prompt_parts)
        
        try:
            response = requests.post(
                self.config.api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.config.api_key}"
                },
                json={
                    "messages": [
                        {"role": "system", "content": "You are an expert in knowledge graph refinement. Always return valid JSON arrays."},
                        {"role": "user", "content": prompt}
                    ],
                    "model": self.config.name,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                },
                timeout=300
            )
            
            content = response.json()["choices"][0]["message"]["content"]
            print(f"Raw LLM response: {content}")
            self.history.append({'prompt': prompt, 'response': content})
            return content

        except Exception as e:
            print(f"❌ Error: {e}")
            return "[]"

# ============================================================================
# JSON Extractor
# ============================================================================

class JSONExtractor:
    """Extract JSON from LLM responses."""
    
    @staticmethod
    def extract(text: str) -> list:
        if isinstance(text, list):
            return text
        
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        match = re.search(r'(\[.*?\])', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                pass
        
        return []

# ============================================================================
# SpaCy Extractor
# ============================================================================

class SpacyExtractor:
    """Extract entities and relationships with spaCy."""
    
    def __init__(self):
        print("Loading spaCy...")
        self.nlp = spacy.load("en_core_web_trf")
        print("✅ spaCy loaded")
    
    def find_entity_for_span(self, token, entities):
        token_start = token.idx
        token_end = token.idx + len(token.text)
        
        for ent in entities:
            if token_start >= ent['start'] and token_end <= ent['end']:
                return ent['text']
        
        for ent in entities:
            if (token_start >= ent['start'] and token_start < ent['end']) or \
               (token_end > ent['start'] and token_end <= ent['end']):
                return ent['text']
        
        token_text_lower = token.text.lower()
        for ent in entities:
            if token_text_lower in ent['text'].lower() or ent['text'].lower() in token_text_lower:
                return ent['text']
        
        return None

    def get_entity_head(self, span, entities):
        if hasattr(span, 'root'):
            return self.find_entity_for_span(span.root, entities)
        return self.find_entity_for_span(span, entities)

    def extract(self, text: str) -> dict:
        """Run spaCy NER and relationship extraction."""
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
        
        relationships = []
        transaction_verbs = {
    'sell', 'buy', 'acquire', 'purchase', 'transfer', 'export', 'import',
    'smuggle', 'consign', 'operate', 'deal', 'trade', 'traffic', 'supply',
    'provide', 'convict', 'charge', 'sentence', 'arrest', 'raid', 'open',
    'close', 'found', 'establish', 'meet', 'return', 'transport', 'loot', 'steal',
    'auction', 'donate', 'bequeath', 'inherit', 'recover', 'possess', 'appropriate',
    'confiscate', 'retrieve', 'excavate', 'plunder', 'pillage', 'defraud', 'forge',
    'conceal', 'ship', 'move', 'dispatch', 'escort', 'relocate', 'investigate', 'seize',
    'repatriate', 'expropriate', 'display', 'restore', 'store', 'own', 'handle',
    'authenticate', 'value', 'appraise', 'attribute', 'exchange'
}
        
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma not in transaction_verbs:
                continue
            
            subject = None
            obj = None
            prep_obj = None  # for prepositional objects
            
            # Extract subject and objects from dependencies
            for child in token.children:
                # Subject (active or passive)
                if child.dep_ in ['nsubj', 'nsubjpass']:
                    subject = child
                
                # Direct object
                elif child.dep_ in ['dobj', 'obj']:
                    obj = child
                
                # Prepositional phrases ("sold to X", "bought from Y")
                elif child.dep_ == 'prep':
                    # Find the object of the preposition
                    for pchild in child.children:
                        if pchild.dep_ == 'pobj':
                            prep_obj = pchild
                            break
                
                # Dative ("gave him", "told her")
                elif child.dep_ == 'dative':
                    if not obj:  # Use dative as object if no direct object
                        obj = child
            
            # Try to create relationships
            relationships_to_add = []
            
            # Subject -> Object relationship
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
            
            # Subject -> Prepositional Object relationship
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
            
            # Object -> Prepositional Object relationship (passive constructions)
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
        
        # Deduplicate relationships
        unique_rels = []
        seen = set()
        for rel in relationships:
            key = (rel['subject'], rel['lemma'], rel['object'])
            if key not in seen:
                seen.add(key)
                unique_rels.append(rel)
        
        print(f"✅ spaCy: {len(entities)} entities, {len(relationships)} relationships")
        
        return {
            "entities": entities,
            "relationships": unique_rels
        }

# ============================================================================
# Custom Refinement Module (bypasses DSPy response parsing)
# ============================================================================

class CustomRefiner(dspy.Module):
    """
    Custom module that builds prompts manually and handles raw responses.
    This is optimizable by DSPy because it has a demos attribute.
    """
    
    def __init__(self, task_description: str, output_field_name: str):
        super().__init__()
        self.task_description = task_description
        self.output_field_name = output_field_name
        self.demos = []  # DSPy optimizer will populate this
        self.json_extractor = JSONExtractor()
    
    def _build_prompt(self, **kwargs):
        """Build prompt with optional few-shot examples."""
        parts = [self.task_description, ""]
        
        # Add few-shot examples if available
        if self.demos:
            parts.append(f"Here are {len(self.demos)} example(s):\n")
            for i, demo in enumerate(self.demos, 1):
                parts.append(f"=== Example {i} ===")
                
                # Show inputs
                for key, value in kwargs.items():
                    if hasattr(demo, key):
                        demo_value = str(getattr(demo, key))[:200]
                        parts.append(f"{key}: {demo_value}...")
                
                # Show expected output
                if hasattr(demo, self.output_field_name):
                    demo_output = getattr(demo, self.output_field_name)
                    if isinstance(demo_output, list):
                        demo_output = json.dumps(demo_output)[:200]
                    parts.append(f"{self.output_field_name}: {demo_output}...")
                
                parts.append("")
        
        # Add current task
        parts.append("=== Your Task ===")
        for key, value in kwargs.items():
            parts.append(f"{key}: {value}")
        
        parts.append("")
        parts.append(f"Return ONLY a valid JSON array. No other text or formatting.")
        
        return "\n".join(parts)
    
    def forward(self, **kwargs):
        """Execute the module."""
        try:
            prompt = self._build_prompt(**kwargs)
            raw_response = dspy.settings.lm(prompt)
            
            # Parse JSON from raw response
            parsed_output = self.json_extractor.extract(raw_response)
            
            # Return prediction
            result = dspy.Prediction(**{self.output_field_name: parsed_output})
            
            # Store inputs for potential use as demos
            for key, value in kwargs.items():
                setattr(result, key, value)
            
            return result
            
        except Exception as e:
            print(f"⚠️ Refiner error: {e}")
            return dspy.Prediction(**{self.output_field_name: []})

# ============================================================================
# Create Training Data
# ============================================================================

def create_training_data():
    """Create training data matching the sophisticated refinement schema."""
    
    spacy_extractor = SpacyExtractor()
    
    entity_examples = []
    relationship_examples = []
    
    # ========================================================================
    # Example 1: Coreference resolution + Role classification
    # ========================================================================
    text1 = "Giacomo Medici started dealing in antiquities in Rome during the 1960s. In 1967, Medici was convicted in Italy of receiving looted artefacts. He met Robert Hecht the same year."
    spacy_result1 = spacy_extractor.extract(text1)
    
    # GOLD STANDARD: Shows coreference (Medici, He → giacomo_medici)
    gold_entities1 = [
        {
            "canonical_id": "giacomo_medici",
            "full_name": "Giacomo Medici",
            "type": "PERSON",
            "mentions": ["Giacomo Medici", "Medici", "He"],
            "attributes": {
                "role": "dealer",
                "nationality": "Italian"
            }
        },
        {
            "canonical_id": "robert_hecht",
            "full_name": "Robert Hecht",
            "type": "PERSON",
            "mentions": ["Robert Hecht"],
            "attributes": {
                "role": "dealer",
                "nationality": "American"
            }
        },
        {
            "canonical_id": "rome",
            "type": "LOCATION",
            "mentions": ["Rome"],
            "attributes": {
                "location_type": "city",
                "country": "Italy",
                "significance": "dealing_location"
            }
        },
        {
            "canonical_id": "italy",
            "type": "LOCATION",
            "mentions": ["Italy"],
            "attributes": {
                "location_type": "country",
                "significance": "jurisdiction"
            }
        }
    ]
    
    gold_relationships1 = [
        {
            "source_id": "giacomo_medici",
            "target_id": "rome",
            "relation_type": "operated_in",
            "attributes": {
                "date": "1960s",
                "activity": "dealing in antiquities"
            }
        },
        {
            "source_id": "giacomo_medici",
            "target_id": "giacomo_medici",
            "relation_type": "convicted",
            "attributes": {
                "date": "1967",
                "location": "italy",
                "charge": "receiving looted artefacts",
                "legal_status": "convicted"
            }
        },
        {
            "source_id": "giacomo_medici",
            "target_id": "robert_hecht",
            "relation_type": "met",
            "attributes": {
                "date": "1967"
            }
        }
    ]
    
    entity_examples.append(dspy.Example(
        text=text1,
        spacy_entities=json.dumps(spacy_result1["entities"]),
        expected_entities=gold_entities1
    ).with_inputs("text", "spacy_entities"))
    
    relationship_examples.append(dspy.Example(
        text=text1,
        spacy_entities=json.dumps(spacy_result1["entities"]),
        spacy_relationships=json.dumps(spacy_result1["relationships"]),
        expected_relationships=gold_relationships1
    ).with_inputs("text", "spacy_entities", "spacy_relationships"))
    
    # ========================================================================
    # Example 2: Artifact tracking + Transaction details
    # ========================================================================
    text2 = "In December 1971, Medici bought the illegally-excavated Euphronios krater from tombaroli for $100,000. He transported it to Switzerland and sold it to Hecht."
    spacy_result2 = spacy_extractor.extract(text2)
    
    gold_entities2 = [
        {
            "canonical_id": "giacomo_medici",
            "full_name": "Giacomo Medici",
            "type": "PERSON",
            "mentions": ["Medici", "He"],
            "attributes": {
                "role": "dealer"
            }
        },
        {
            "canonical_id": "euphronios_sarpedon_krater",
            "full_name": "Euphronios (Sarpedon) krater",
            "type": "ARTIFACT",
            "mentions": ["Euphronios krater"],
            "attributes": {
                "object_type": "krater",
                "artist": "Euphronios",
                "legal_status": "looted",
                "condition": "illegally-excavated"
            }
        },
        {
            "canonical_id": "tombaroli",
            "type": "PERSON",
            "mentions": ["tombaroli"],
            "attributes": {
                "role": "looter",
                "description": "grave robbers"
            }
        },
        {
            "canonical_id": "robert_hecht",
            "full_name": "Robert Hecht",
            "type": "PERSON",
            "mentions": ["Hecht"],
            "attributes": {
                "role": "dealer"
            }
        },
        {
            "canonical_id": "switzerland",
            "type": "LOCATION",
            "mentions": ["Switzerland"],
            "attributes": {
                "location_type": "country",
                "significance": "transit_location"
            }
        }
    ]
    
    gold_relationships2 = [
        {
            "source_id": "giacomo_medici",
            "target_id": "euphronios_sarpedon_krater",
            "relation_type": "purchased",
            "attributes": {
                "date": "December 1971",
                "amount": "$100,000",
                "seller": "tombaroli",
                "legal_status": "illegal_transaction"
            }
        },
        {
            "source_id": "giacomo_medici",
            "target_id": "euphronios_sarpedon_krater",
            "relation_type": "transported",
            "attributes": {
                "destination": "switzerland",
                "date": "1971"
            }
        },
        {
            "source_id": "giacomo_medici",
            "target_id": "robert_hecht",
            "relation_type": "sold_to",
            "attributes": {
                "date": "1971",
                "artifact": "euphronios_sarpedon_krater",
                "location": "switzerland"
            }
        }
    ]
    
    entity_examples.append(dspy.Example(
        text=text2,
        spacy_entities=json.dumps(spacy_result2["entities"]),
        expected_entities=gold_entities2
    ).with_inputs("text", "spacy_entities"))
    
    relationship_examples.append(dspy.Example(
        text=text2,
        spacy_entities=json.dumps(spacy_result2["entities"]),
        spacy_relationships=json.dumps(spacy_result2["relationships"]),
        expected_relationships=gold_relationships2
    ).with_inputs("text", "spacy_entities", "spacy_relationships"))
    
    # ========================================================================
    # Example 3: Museum acquisition + Network relationships
    # ========================================================================
    text3 = "The J. Paul Getty Museum acquired the krater through Hecht in 1972 for $1 million. The museum's curator was Marion True."
    spacy_result3 = spacy_extractor.extract(text3)
    
    gold_entities3 = [
        {
            "canonical_id": "j_paul_getty_museum",
            "full_name": "J. Paul Getty Museum",
            "type": "ORGANIZATION",
            "mentions": ["J. Paul Getty Museum", "The museum"],
            "attributes": {
                "entity_type": "museum",
                "location": "Los Angeles"
            }
        },
        {
            "canonical_id": "euphronios_sarpedon_krater",
            "full_name": "Euphronios (Sarpedon) krater",
            "type": "ARTIFACT",
            "mentions": ["the krater"],
            "attributes": {
                "object_type": "krater"
            }
        },
        {
            "canonical_id": "robert_hecht",
            "full_name": "Robert Hecht",
            "type": "PERSON",
            "mentions": ["Hecht"],
            "attributes": {
                "role": "dealer"
            }
        },
        {
            "canonical_id": "marion_true",
            "full_name": "Marion True",
            "type": "PERSON",
            "mentions": ["Marion True"],
            "attributes": {
                "role": "curator",
                "affiliation": "j_paul_getty_museum"
            }
        }
    ]
    
    gold_relationships3 = [
        {
            "source_id": "j_paul_getty_museum",
            "target_id": "euphronios_sarpedon_krater",
            "relation_type": "acquired",
            "attributes": {
                "date": "1972",
                "amount": "$1 million",
                "intermediary": "robert_hecht",
                "transaction_type": "purchase"
            }
        },
        {
            "source_id": "robert_hecht",
            "target_id": "j_paul_getty_museum",
            "relation_type": "sold_to",
            "attributes": {
                "date": "1972",
                "artifact": "euphronios_sarpedon_krater",
                "amount": "$1 million"
            }
        },
        {
            "source_id": "marion_true",
            "target_id": "j_paul_getty_museum",
            "relation_type": "employed_by",
            "attributes": {
                "role": "curator",
                "date": "1972"
            }
        }
    ]
    
    entity_examples.append(dspy.Example(
        text=text3,
        spacy_entities=json.dumps(spacy_result3["entities"]),
        expected_entities=gold_entities3
    ).with_inputs("text", "spacy_entities"))
    
    relationship_examples.append(dspy.Example(
        text=text3,
        spacy_entities=json.dumps(spacy_result3["entities"]),
        spacy_relationships=json.dumps(spacy_result3["relationships"]),
        expected_relationships=gold_relationships3
    ).with_inputs("text", "spacy_entities", "spacy_relationships"))
    
    # ========================================================================
    # Example 4: Law enforcement action + Evidence
    # ========================================================================
    text4 = "In 1995, Italian authorities raided Medici's warehouse in the Geneva Freeport. They seized 3,800 photographs of looted artifacts."
    spacy_result4 = spacy_extractor.extract(text4)
    
    gold_entities4 = [
        {
            "canonical_id": "italian_authorities",
            "type": "ORGANIZATION",
            "mentions": ["Italian authorities", "They"],
            "attributes": {
                "entity_type": "law_enforcement",
                "jurisdiction": "Italy"
            }
        },
        {
            "canonical_id": "giacomo_medici",
            "full_name": "Giacomo Medici",
            "type": "PERSON",
            "mentions": ["Medici"],
            "attributes": {
                "role": "dealer"
            }
        },
        {
            "canonical_id": "geneva_freeport",
            "full_name": "Geneva Freeport",
            "type": "LOCATION",
            "mentions": ["Geneva Freeport"],
            "attributes": {
                "location_type": "storage_facility",
                "country": "Switzerland",
                "significance": "storage_of_looted_artifacts"
            }
        }
    ]
    
    gold_relationships4 = [
        {
            "source_id": "italian_authorities",
            "target_id": "geneva_freeport",
            "relation_type": "raided",
            "attributes": {
                "date": "1995",
                "target": "giacomo_medici",
                "legal_action": "search_and_seizure"
            }
        },
        {
            "source_id": "giacomo_medici",
            "target_id": "geneva_freeport",
            "relation_type": "operated",
            "attributes": {
                "facility_type": "warehouse",
                "purpose": "storage"
            }
        },
        {
            "source_id": "italian_authorities",
            "target_id": "giacomo_medici",
            "relation_type": "seized_evidence_from",
            "attributes": {
                "date": "1995",
                "items": "3,800 photographs of looted artifacts",
                "location": "geneva_freeport"
            }
        }
    ]
    
    entity_examples.append(dspy.Example(
        text=text4,
        spacy_entities=json.dumps(spacy_result4["entities"]),
        expected_entities=gold_entities4
    ).with_inputs("text", "spacy_entities"))
    
    relationship_examples.append(dspy.Example(
        text=text4,
        spacy_entities=json.dumps(spacy_result4["entities"]),
        spacy_relationships=json.dumps(spacy_result4["relationships"]),
        expected_relationships=gold_relationships4
    ).with_inputs("text", "spacy_entities", "spacy_relationships"))
    
    return entity_examples, relationship_examples

# ============================================================================
# Metrics
# ============================================================================

# ============================================================================
# Updated Metrics (Match new gold standard schema)
# ============================================================================

def entity_refinement_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    Evaluate entity refinement with the new canonical_id schema.
    Compare based on canonical_ids instead of text strings.
    """
    if not hasattr(example, 'expected_entities') or not hasattr(prediction, 'refined_entities'):
        return 0.0
    
    expected = example.expected_entities
    predicted = prediction.refined_entities
    
    if not predicted or not expected:
        return 0.0
    
    # Extract canonical_ids (or fall back to full_name)
    expected_ids = set()
    for e in expected:
        if isinstance(e, dict):
            if 'canonical_id' in e:
                expected_ids.add(e['canonical_id'])
            elif 'full_name' in e:
                expected_ids.add(e['full_name'].lower().replace(' ', '_'))
    
    predicted_ids = set()
    for e in predicted:
        if isinstance(e, dict):
            if 'canonical_id' in e:
                predicted_ids.add(e['canonical_id'])
            elif 'full_name' in e:
                predicted_ids.add(e['full_name'].lower().replace(' ', '_'))
    
    if not expected_ids:
        return 0.0
    
    # Calculate F1
    true_positives = len(expected_ids & predicted_ids)
    precision = true_positives / len(predicted_ids) if predicted_ids else 0.0
    recall = true_positives / len(expected_ids)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def relationship_refinement_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    Evaluate relationship refinement with the new schema.
    Compare based on (source_id, relation_type, target_id) tuples.
    """
    if not hasattr(example, 'expected_relationships') or not hasattr(prediction, 'refined_relationships'):
        return 0.0
    
    expected = example.expected_relationships
    predicted = prediction.refined_relationships
    
    if not predicted or not expected:
        return 0.0
    
    # Create relationship tuples using new schema (source_id, relation_type, target_id)
    expected_rels = set()
    for r in expected:
        if isinstance(r, dict):
            if all(k in r for k in ['source_id', 'relation_type', 'target_id']):
                expected_rels.add((
                    r['source_id'].lower(),
                    r['relation_type'].lower(),
                    r['target_id'].lower()
                ))
    
    predicted_rels = set()
    for r in predicted:
        if isinstance(r, dict):
            if all(k in r for k in ['source_id', 'relation_type', 'target_id']):
                predicted_rels.add((
                    r['source_id'].lower(),
                    r['relation_type'].lower(),
                    r['target_id'].lower()
                ))
            # Also support old schema (subject, predicate, object) for backward compatibility
            elif all(k in r for k in ['subject', 'predicate', 'object']):
                predicted_rels.add((
                    r['subject'].lower(),
                    r['predicate'].lower(),
                    r['object'].lower()
                ))
    
    if not expected_rels:
        return 0.0
    
    # Calculate F1
    true_positives = len(expected_rels & predicted_rels)
    precision = true_positives / len(predicted_rels) if predicted_rels else 0.0
    recall = true_positives / len(expected_rels)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# ============================================================================
# Custom Optimizer
# ============================================================================

class SimpleBootstrap:
    """Simple optimizer that adds successful examples as demos."""
    
    def __init__(self, metric, max_demos=3, threshold=0.5, input_fields=None):
        self.metric = metric
        self.max_demos = max_demos
        self.threshold = threshold
        self.input_fields = input_fields  # Explicitly specify which fields are inputs
    
    def compile(self, student, trainset):
        """Add successful examples as demos."""
        print(f"  Running bootstrap on {len(trainset)} examples...")
        
        # If input_fields not specified, try to infer
        if self.input_fields is None:
            print(f"  WARNING: input_fields not specified, will try to infer...")
            if trainset:
                # Get all non-expected fields from first example
                self.input_fields = [
                    k for k in trainset[0].__dict__.keys() 
                    if not k.startswith('_') and not k.startswith('expected_')
                ]
                print(f"  Inferred input fields: {self.input_fields}")
        
        good_demos = []
        
        for i, example in enumerate(trainset):
            try:
                # Build kwargs using specified input fields
                kwargs = {field: getattr(example, field) for field in self.input_fields}
                
                print(f"    Example {i+1}: ", end="")
                
                # Run the student module
                pred = student(**kwargs)
                
                # Check if it's a good example
                score = self.metric(example, pred)
                print(f"score = {score:.2f}")
                
                if score >= self.threshold:
                    # Add output field to example for use as demo
                    output_field = student.output_field_name
                    if hasattr(pred, output_field):
                        setattr(example, output_field, getattr(pred, output_field))
                        good_demos.append(example)
                    
            except Exception as e:
                print(f"error = {e}")
        
        print(f"  Found {len(good_demos)} good examples (threshold={self.threshold})")
        
        # Add demos to student
        student.demos = good_demos[:self.max_demos]
        print(f"  Added {len(student.demos)} demos to module")
        
        return student

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Two-Stage KG: spaCy → LLM Refinement (Custom Modules)")
    print("=" * 80)
    
    # Load models
    LOCAL_MODELS = load_models_from_config()
    if not LOCAL_MODELS:
        print("❌ No models!")
        return
    
    print(f"\n📦 Models: {[m.name for m in LOCAL_MODELS]}")
    
    # Create training data
    print("\n📚 Creating training data...")
    entity_dataset, relationship_dataset = create_training_data()
    print(f"✅ {len(entity_dataset)} entity examples, {len(relationship_dataset)} relationship examples")
    
    # Optimize for each model
    for model_config in LOCAL_MODELS:
        print("\n" + "=" * 80)
        print(f"MODEL: {model_config.name}")
        print("=" * 80)
        
        lm = LocalLLM(model_config)
        dspy.settings.configure(lm=lm)
        
        # ENTITY REFINEMENT
        print("\n🔧 Entity refinement optimization...")
        entity_refiner = CustomRefiner(
            task_description=    
"""
Review spaCy entities and refine them into canonical form.

For each entity:
1. Assign a canonical_id (lowercase_underscore format).
2. Provide full_name if known.
3. Set type.  The 'type' value *must* be one of these: PERSON, ORGANIZATION, ARTIFACT, LOCATION.
4. List all mentions found in text.
5. Add attributes (role, entity_type, etc.) that describe important properties.
   For example, for a PERSON, include 'role' (e.g., "dealer"), 'nationality'.  For a LOCATION, include 'location_type' (e.g., "city", "country").

Return *only* a valid JSON array. Each object in the array should represent an entity. 
The JSON objects must have the keys: "canonical_id", "full_name", "type", "mentions", "attributes".
The "attributes" value should be a JSON object (a dictionary).
Do *not* include any other text or tokens (like <|end_of_turn|>).

Here's an example of the *expected output format*:
[
  {
    "canonical_id": "giacomo_medici",
    "full_name": "Giacomo Medici",
    "type": "PERSON",
    "mentions": ["Giacomo Medici", "Medici"],
    "attributes": {"role": "dealer", "nationality": "Italian"}
  },
  {
    "canonical_id": "italy",
    "full_name": "Italy",
    "type": "LOCATION",
    "mentions": ["Italy"],
    "attributes": {"location_type": "country", "significance": "jurisdiction"}
  }
]

Follow this format *exactly*.

""",
            output_field_name="refined_entities"
)
        
        optimizer = SimpleBootstrap(
            metric=entity_refinement_metric,
            max_demos=2,
            threshold=0.3,
            input_fields=['text', 'spacy_entities']  
            )
        
        optimized_entity_refiner = optimizer.compile(entity_refiner, entity_dataset)

        
        # Test
        print("\n  Testing optimized entity refiner...")
        test_ex = entity_dataset[0]
        test_pred = optimized_entity_refiner(
            text=test_ex.text,
            spacy_entities=test_ex.spacy_entities
        )
        test_score = entity_refinement_metric(test_ex, test_pred)
        print(f"  ✅ Test F1: {test_score:.3f}")
        
        # RELATIONSHIP REFINEMENT
        print("\n🔧 Relationship refinement optimization...")
        rel_refiner = CustomRefiner(
            task_description="""Review spaCy relationships and refine them into canonical form.

For each relationship:
1. Use source_id and target_id (canonical entity IDs)
2. Set relation_type (met, sold_to, acquired, etc.)
3. Add attributes (date, amount, location, etc.)

Return JSON array with structure:
{
  "source_id": "giacomo_medici",
  "target_id": "robert_hecht",
  "relation_type": "met",
  "attributes": {"date": "1967"}
}""",
             output_field_name="refined_relationships"
)
        
        optimizer = SimpleBootstrap(
            metric=relationship_refinement_metric,
            max_demos=2,
            threshold=0.3,
            input_fields=['text', 'spacy_entities', 'spacy_relationships'] 
            )

        optimized_rel_refiner = optimizer.compile(rel_refiner, relationship_dataset)
        
        # Test
        print("\n  Testing optimized relationship refiner...")
        test_ex = relationship_dataset[0]
        test_pred = optimized_rel_refiner(
            text=test_ex.text,
            spacy_entities=test_ex.spacy_entities,
            spacy_relationships=test_ex.spacy_relationships
        )
        test_score = relationship_refinement_metric(test_ex, test_pred)
        print(f"  ✅ Test F1: {test_score:.3f}")
        
        print(f"\n✅ {model_config.name} optimization complete!")
        print(f"   Entity refiner has {len(optimized_entity_refiner.demos)} demos")
        print(f"   Relationship refiner has {len(optimized_rel_refiner.demos)} demos")

           # After optimization completes:
    print("\n" + "=" * 80)
    print("PROMPT INSPECTION")
    print("=" * 80)
    
    # Inspect entity refiner
    print(f"\n✨ Entity Refiner: {len(optimized_entity_refiner.demos)} demos added")
    test_entity_prompt = optimized_entity_refiner._build_prompt(
        text=entity_dataset[0].text,
        spacy_entities=entity_dataset[0].spacy_entities
    )
    print(f"\nSample prompt length: {len(test_entity_prompt)} chars")
    print("\nFirst 1000 characters of prompt:")
    print("-" * 80)
    print(test_entity_prompt[:1000])
    print("...")
    
    # Save full prompts
    with open(f"optimized_prompts_{model_config.name}.txt", "w") as f:
        f.write("ENTITY REFINEMENT PROMPT\n")
        f.write("=" * 80 + "\n")
        f.write(test_entity_prompt)
        f.write("\n\n")
        
        test_rel_prompt = optimized_rel_refiner._build_prompt(
            text=relationship_dataset[0].text,
            spacy_entities=relationship_dataset[0].spacy_entities,
            spacy_relationships=relationship_dataset[0].spacy_relationships
        )
        
        f.write("\n\nRELATIONSHIP REFINEMENT PROMPT\n")
        f.write("=" * 80 + "\n")
        f.write(test_rel_prompt)
    
    print(f"\n✅ Full prompts saved to: optimized_prompts_{model_config.name}.txt")

if __name__ == "__main__":
    main()
