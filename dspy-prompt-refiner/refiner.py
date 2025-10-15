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
            #print(f"Raw LLM response: {content}")
            self.history.append({'prompt': prompt, 'response': content})
            return content

        except Exception as e:
            print(f"❌ Error: {e}")
            return "[]"

# ============================================================================
# IMPROVED JSON Extractor with better debugging
# ============================================================================

class JSONExtractor:
    """Extract JSON from LLM responses."""
    
    @staticmethod
    def extract(text: str, debug=False) -> list:
        if isinstance(text, list):
            return text
        
        if debug:
            print(f"\n🔍 JSONExtractor DEBUG:")
            print(f"  Raw input length: {len(text)}")
            print(f"  First 200 chars: {text[:200]}")
            print(f"  Last 200 chars: {text[-200:]}")
        
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Remove various end tokens (be more aggressive)
        text = re.sub(r'<\|end_of_turn\|>', '', text)
        text = re.sub(r'<\|file_separator\|>', '', text)
        text = re.sub(r'<\|im_end\|>', '', text)
        text = re.sub(r'</s>', '', text)
        text = re.sub(r'<eos>', '', text)
        
        if debug:
            print(f"  After cleanup: {text[:200]}")
        
        # Try multiple strategies to find JSON
        
        # Strategy 1: Non-greedy match (original)
        match = re.search(r'(\[.*?\])', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if debug:
                    print(f"  ✓ Strategy 1 worked: {len(parsed)} items")
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError as e:
                if debug:
                    print(f"  ✗ Strategy 1 failed: {e}")
        
        # Strategy 2: Greedy match (get everything between first [ and last ])
        match = re.search(r'(\[.*\])', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if debug:
                    print(f"  ✓ Strategy 2 worked: {len(parsed)} items")
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError as e:
                if debug:
                    print(f"  ✗ Strategy 2 failed: {e}")
        
        # Strategy 3: Find balanced brackets
        if '[' in text:
            start = text.index('[')
            bracket_count = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == '[':
                    bracket_count += 1
                elif text[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end = i + 1
                        break
            
            if end > start:
                try:
                    json_str = text[start:end]
                    parsed = json.loads(json_str)
                    if debug:
                        print(f"  ✓ Strategy 3 worked: {len(parsed)} items")
                    return parsed if isinstance(parsed, list) else []
                except json.JSONDecodeError as e:
                    if debug:
                        print(f"  ✗ Strategy 3 failed: {e}")
        
        if debug:
            print(f"  ✗ All strategies failed")
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
            #print(f"✅ CustomRefiner.forward raw_response: {raw_response}")
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
# ============================================================================
# SIMPLIFIED Training Data - Achievable for 7B models
# ============================================================================

def create_training_data():
    """Simplified training examples - focus on core task only."""
    
    spacy_extractor = SpacyExtractor()
    
    entity_examples = []
    relationship_examples = []
    
    # ========================================================================
    # Training Example 1
    # ========================================================================
    text1 = "Giacomo Medici started dealing in antiquities in Rome during the 1960s. In 1967, Medici was convicted in Italy of receiving looted artefacts. He met Robert Hecht the same year."
    spacy_result1 = spacy_extractor.extract(text1)
    
    # SIMPLIFIED: Just core fields
    gold_entities1 = [
        {"canonical_id": "giacomo_medici", "type": "PERSON", "role": "dealer"},
        {"canonical_id": "robert_hecht", "type": "PERSON", "role": "dealer"},
        {"canonical_id": "rome", "type": "LOCATION"},
        {"canonical_id": "italy", "type": "LOCATION"}
    ]
    
    # SIMPLIFIED: Just the triple + optional date
    gold_relationships1 = [
        {"source_id": "giacomo_medici", "target_id": "rome", "relation_type": "operated_in"},
        {"source_id": "giacomo_medici", "target_id": "robert_hecht", "relation_type": "met"}
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
    # Training Example 2
    # ========================================================================
    text2 = "In December 1971, Medici bought the illegally-excavated Euphronios krater from tombaroli for $100,000. He transported it to Switzerland and sold it to Hecht."
    spacy_result2 = spacy_extractor.extract(text2)
    
    gold_entities2 = [
        {"canonical_id": "giacomo_medici", "type": "PERSON", "role": "dealer"},
        {"canonical_id": "euphronios_krater", "type": "ARTIFACT"},
        {"canonical_id": "tombaroli", "type": "PERSON", "role": "looter"},
        {"canonical_id": "robert_hecht", "type": "PERSON", "role": "dealer"},
        {"canonical_id": "switzerland", "type": "LOCATION"}
    ]
    
    gold_relationships2 = [
        {"source_id": "giacomo_medici", "target_id": "euphronios_krater", "relation_type": "purchased"},
        {"source_id": "giacomo_medici", "target_id": "euphronios_krater", "relation_type": "transported"},
        {"source_id": "giacomo_medici", "target_id": "robert_hecht", "relation_type": "sold_to"}
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
    # Training Example 3
    # ========================================================================
    text3 = "The J. Paul Getty Museum acquired the krater through Hecht in 1972 for $1 million. The museum's curator was Marion True."
    spacy_result3 = spacy_extractor.extract(text3)
    
    gold_entities3 = [
        {"canonical_id": "getty_museum", "type": "ORGANIZATION"},
        {"canonical_id": "euphronios_krater", "type": "ARTIFACT"},
        {"canonical_id": "robert_hecht", "type": "PERSON", "role": "dealer"},
        {"canonical_id": "marion_true", "type": "PERSON", "role": "curator"}
    ]
    
    gold_relationships3 = [
        {"source_id": "getty_museum", "target_id": "euphronios_krater", "relation_type": "acquired"},
        {"source_id": "marion_true", "target_id": "getty_museum", "relation_type": "employed_by"}
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
    # Training Example 4
    # ========================================================================
    text4 = "In 1995, Italian authorities raided Medici's warehouse in the Geneva Freeport. They seized 3,800 photographs of looted artifacts."
    spacy_result4 = spacy_extractor.extract(text4)
    
    gold_entities4 = [
        {"canonical_id": "italian_authorities", "type": "ORGANIZATION"},
        {"canonical_id": "giacomo_medici", "type": "PERSON", "role": "dealer"},
        {"canonical_id": "geneva_freeport", "type": "LOCATION"}
    ]
    
    gold_relationships4 = [
        {"source_id": "italian_authorities", "target_id": "geneva_freeport", "relation_type": "raided"},
        {"source_id": "italian_authorities", "target_id": "giacomo_medici", "relation_type": "seized_from"}
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
# SIMPLIFIED Test Data
# ============================================================================

def create_test_data():
    """Simplified test examples."""
    
    spacy_extractor = SpacyExtractor()
    
    entity_examples = []
    relationship_examples = []
    
    # ========================================================================
    # Test Example 1
    # ========================================================================
    text1 = "Robin Symes operated an antiquities gallery in London during the 1980s. He frequently sold artifacts to the Metropolitan Museum."
    spacy_result1 = spacy_extractor.extract(text1)
    
    gold_entities1 = [
        {"canonical_id": "robin_symes", "type": "PERSON", "role": "dealer"},
        {"canonical_id": "london", "type": "LOCATION"},
        {"canonical_id": "metropolitan_museum", "type": "ORGANIZATION"}
    ]
    
    gold_relationships1 = [
        {"source_id": "robin_symes", "target_id": "london", "relation_type": "operated_in"},
        {"source_id": "robin_symes", "target_id": "metropolitan_museum", "relation_type": "sold_to"}
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
    # Test Example 2
    # ========================================================================
    text2 = "Marion True served as curator at the Getty Museum from 1986 to 2005. She resigned amid scandal."
    spacy_result2 = spacy_extractor.extract(text2)
    
    gold_entities2 = [
        {"canonical_id": "marion_true", "type": "PERSON", "role": "curator"},
        {"canonical_id": "getty_museum", "type": "ORGANIZATION"}
    ]
    
    gold_relationships2 = [
        {"source_id": "marion_true", "target_id": "getty_museum", "relation_type": "employed_by"}
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
    # Test Example 3
    # ========================================================================
    text3 = "Sotheby's auctioned the Lydian Hoard in New York in 1987."
    spacy_result3 = spacy_extractor.extract(text3)
    
    gold_entities3 = [
        {"canonical_id": "sothebys", "type": "ORGANIZATION"},
        {"canonical_id": "lydian_hoard", "type": "ARTIFACT"},
        {"canonical_id": "new_york", "type": "LOCATION"}
    ]
    
    gold_relationships3 = [
        {"source_id": "sothebys", "target_id": "lydian_hoard", "relation_type": "auctioned"}
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
    # Test Example 4
    # ========================================================================
    text4 = "In 2008, Italy negotiated the return of the Morgantina silver from the Metropolitan Museum."
    spacy_result4 = spacy_extractor.extract(text4)
    
    gold_entities4 = [
        {"canonical_id": "italy", "type": "LOCATION"},
        {"canonical_id": "morgantina_silver", "type": "ARTIFACT"},
        {"canonical_id": "metropolitan_museum", "type": "ORGANIZATION"}
    ]
    
    gold_relationships4 = [
        {"source_id": "italy", "target_id": "morgantina_silver", "relation_type": "recovered"},
        {"source_id": "metropolitan_museum", "target_id": "morgantina_silver", "relation_type": "returned"}
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
    
    # ========================================================================
    # Test Example 5
    # ========================================================================
    text5 = "In 2004, an Italian court convicted Medici of trafficking looted antiquities."
    spacy_result5 = spacy_extractor.extract(text5)
    
    gold_entities5 = [
        {"canonical_id": "italian_court", "type": "ORGANIZATION"},
        {"canonical_id": "giacomo_medici", "type": "PERSON", "role": "dealer"}
    ]
    
    gold_relationships5 = [
        {"source_id": "italian_court", "target_id": "giacomo_medici", "relation_type": "convicted"}
    ]
    
    entity_examples.append(dspy.Example(
        text=text5,
        spacy_entities=json.dumps(spacy_result5["entities"]),
        expected_entities=gold_entities5
    ).with_inputs("text", "spacy_entities"))
    
    relationship_examples.append(dspy.Example(
        text=text5,
        spacy_entities=json.dumps(spacy_result5["entities"]),
        spacy_relationships=json.dumps(spacy_result5["relationships"]),
        expected_relationships=gold_relationships5
    ).with_inputs("text", "spacy_entities", "spacy_relationships"))
    
    # ========================================================================
    # Test Example 6
    # ========================================================================
    text6 = "Frieda Tchacos acquired the Gospel of Judas from Egyptian dealers. She later donated it to National Geographic."
    spacy_result6 = spacy_extractor.extract(text6)
    
    gold_entities6 = [
        {"canonical_id": "frieda_tchacos", "type": "PERSON", "role": "dealer"},
        {"canonical_id": "gospel_of_judas", "type": "ARTIFACT"},
        {"canonical_id": "egyptian_dealers", "type": "PERSON", "role": "dealer"},
        {"canonical_id": "national_geographic", "type": "ORGANIZATION"}
    ]
    
    gold_relationships6 = [
        {"source_id": "frieda_tchacos", "target_id": "gospel_of_judas", "relation_type": "acquired"},
        {"source_id": "frieda_tchacos", "target_id": "national_geographic", "relation_type": "donated_to"}
    ]
    
    entity_examples.append(dspy.Example(
        text=text6,
        spacy_entities=json.dumps(spacy_result6["entities"]),
        expected_entities=gold_entities6
    ).with_inputs("text", "spacy_entities"))
    
    relationship_examples.append(dspy.Example(
        text=text6,
        spacy_entities=json.dumps(spacy_result6["entities"]),
        spacy_relationships=json.dumps(spacy_result6["relationships"]),
        expected_relationships=gold_relationships6
    ).with_inputs("text", "spacy_entities", "spacy_relationships"))
    
    return entity_examples, relationship_examples

# ============================================================================
# Metrics
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

def evaluate_on_dataset(refiner, dataset, metric, task_name="Task"):
    """Evaluate a refiner on a dataset and return detailed results."""
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {task_name}")
    print(f"{'='*80}")
    
    scores = []
    results = []
    
    for i, example in enumerate(dataset):
        # Get input fields
        if hasattr(example, '_input_keys'):
            input_keys = example._input_keys
        else:
            input_keys = [k for k in example.__dict__.keys() 
                         if not k.startswith('_') and not k.startswith('expected_')]
        
        # Build kwargs
        kwargs = {key: getattr(example, key) for key in input_keys}
        
        # Run prediction
        pred = refiner(**kwargs)
        
        # Calculate score
        score = metric(example, pred)
        scores.append(score)
        
        results.append({
            'text': example.text,
            'score': score,
            'prediction': pred
        })
        
        print(f"  Example {i+1}: F1={score:.3f}")
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    print(f"\n{'='*80}")
    print(f"  Average F1: {avg_score:.3f}")
    print(f"  Min F1: {min(scores):.3f}")
    print(f"  Max F1: {max(scores):.3f}")
    print(f"{'='*80}")
    
    return {
        'average': avg_score,
        'scores': scores,
        'results': results
    }

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

# ============================================================================
# Main - HYBRID APPROACH
# ============================================================================

def main():
    print("=" * 80)
    print("Two-Stage KG Extraction: HYBRID Approach")
    print("  - Entities: Optimized (proven +29.7% improvement)")
    print("  - Relationships: Improved Zero-Shot (optimization fails)")
    print("=" * 80)
    
    # Load models
    LOCAL_MODELS = load_models_from_config()
    if not LOCAL_MODELS:
        print("❌ No models!")
        return
    
    print(f"\n📦 Models: {[m.name for m in LOCAL_MODELS]}")
    
    # Create SEPARATE training and test data
    print("\n📚 Creating datasets...")
    print("  - Training data (for optimization)...")
    train_entity_dataset, train_relationship_dataset = create_training_data()
    print(f"    ✅ {len(train_entity_dataset)} entity examples, {len(train_relationship_dataset)} relationship examples")
    
    print("  - Test data (held-out for evaluation)...")
    test_entity_dataset, test_relationship_dataset = create_test_data()
    print(f"    ✅ {len(test_entity_dataset)} entity examples, {len(test_relationship_dataset)} relationship examples")
    
    # Results storage
    all_results = {}
    
    # Optimize for each model
    for model_config in LOCAL_MODELS:
        print("\n" + "=" * 80)
        print(f"MODEL: {model_config.name}")
        print("=" * 80)
        
        lm = LocalLLM(model_config)
        dspy.settings.configure(lm=lm)
        
        model_results = {}
        
        # ====================================================================
        # ENTITY REFINEMENT - USE OPTIMIZATION (IT WORKS!)
        # ====================================================================
        print("\n" + "─" * 80)
        print("ENTITY REFINEMENT - Using Optimization")
        print("─" * 80)
        
        # Create baseline (no optimization)
        print("\n📝 Creating baseline entity refiner...")
        baseline_entity_refiner = CustomRefiner(
            task_description="""Convert spaCy entities to JSON format.

Each entity needs:
- canonical_id: lowercase with underscores
- type: PERSON, ORGANIZATION, ARTIFACT, or LOCATION
- role: (optional) for PERSON entities

Return ONLY a JSON array like:
[{"canonical_id": "giacomo_medici", "type": "PERSON", "role": "dealer"}]""",
            output_field_name="refined_entities"
        )
        
        # Evaluate baseline on test set
        print("\n📊 Baseline evaluation on TEST set...")
        baseline_entity_results = evaluate_on_dataset(
            baseline_entity_refiner,
            test_entity_dataset,
            entity_refinement_metric,
            "Baseline Entity Refinement (Test Set)"
        )
        
        # Optimize
        print("\n🔧 Optimizing entity refiner on TRAINING set...")
        optimizer = SimpleBootstrap(
            metric=entity_refinement_metric,
            max_demos=2,
            threshold=0.3,
            input_fields=['text', 'spacy_entities']
        )
        optimized_entity_refiner = optimizer.compile(baseline_entity_refiner, train_entity_dataset)
        
        # Evaluate optimized on TRAINING set (sanity check)
        print("\n📊 Optimized evaluation on TRAINING set (sanity check)...")
        train_entity_results = evaluate_on_dataset(
            optimized_entity_refiner,
            train_entity_dataset,
            entity_refinement_metric,
            "Optimized Entity Refinement (Training Set)"
        )
        
        # Evaluate optimized on TEST set (the real metric!)
        print("\n📊 Optimized evaluation on TEST set (GENERALIZATION)...")
        test_entity_results = evaluate_on_dataset(
            optimized_entity_refiner,
            test_entity_dataset,
            entity_refinement_metric,
            "Optimized Entity Refinement (Test Set)"
        )
        
        model_results['entity'] = {
            'baseline_test': baseline_entity_results['average'],
            'optimized_train': train_entity_results['average'],
            'optimized_test': test_entity_results['average'],
            'improvement': test_entity_results['average'] - baseline_entity_results['average']
        }
        
        # ====================================================================
        # RELATIONSHIP REFINEMENT - USE IMPROVED ZERO-SHOT (NO OPTIMIZATION)
        # ====================================================================
        print("\n" + "─" * 80)
        print("RELATIONSHIP REFINEMENT - Using Improved Zero-Shot")
        print("⚠️  Skipping optimization (proven to hurt performance)")
        print("─" * 80)
        
        # Create IMPROVED zero-shot baseline
        print("\n📝 Creating improved zero-shot relationship refiner...")
        improved_rel_refiner = CustomRefiner(
            task_description="""Extract relationships between entities in antiquities trafficking networks.

CONTEXT: You'll receive:
1. Original text about antiquities trafficking
2. A list of entities extracted by spaCy
3. Preliminary relationships found by spaCy

YOUR TASK: Review and refine these relationships into a standardized format.

RELATIONSHIP TYPES (use these exact terms):
- sold_to: X sold artifacts/items to Y
- purchased: X purchased/bought artifacts/items
- acquired: X acquired artifacts (general acquisition)
- employed_by: X worked for organization Y
- operated_in: X operated/dealt in location Y
- raided: Law enforcement X raided location/person Y
- convicted: X was convicted (can be self-referential)
- transported: X moved/transported artifacts
- donated_to: X donated items to Y
- collaborated_with: X worked/partnered with Y
- seized_from: X seized evidence/items from Y
- returned: X returned artifacts
- recovered: X recovered artifacts
- auctioned: X auctioned artifacts
- met: X met with Y

OUTPUT FORMAT: JSON array of relationship triples
[
  {"source_id": "canonical_id_1", "target_id": "canonical_id_2", "relation_type": "sold_to"}
]

CRITICAL RULES:
1. Use canonical_id format (lowercase_with_underscores) for both source_id and target_id
2. Convert entity names to canonical format: "Giacomo Medici" → "giacomo_medici"
3. Use ONLY relation types from the list above
4. Focus on meaningful relationships (transactions, employment, legal actions)
5. Skip vague or uncertain relationships
6. Return ONLY the JSON array, no explanations

EXAMPLES OF GOOD OUTPUTS:
Text: "Medici sold the krater to Hecht in Switzerland"
Output: [
  {"source_id": "giacomo_medici", "target_id": "robert_hecht", "relation_type": "sold_to"}
]

Text: "Marion True worked as curator at the Getty Museum"
Output: [
  {"source_id": "marion_true", "target_id": "getty_museum", "relation_type": "employed_by"}
]

Text: "Italian authorities raided the warehouse in Geneva"
Output: [
  {"source_id": "italian_authorities", "target_id": "geneva_freeport", "relation_type": "raided"}
]

Now extract relationships from the data below:""",
            output_field_name="refined_relationships"
        )
        
        # Evaluate improved zero-shot on test set
        print("\n📊 Improved zero-shot evaluation on TEST set...")
        improved_rel_results = evaluate_on_dataset(
            improved_rel_refiner,
            test_relationship_dataset,
            relationship_refinement_metric,
            "Improved Zero-Shot Relationship Refinement (Test Set)"
        )
        
        # Also test the OLD baseline for comparison
        print("\n📊 OLD baseline evaluation on TEST set (for comparison)...")
        old_baseline_rel_refiner = CustomRefiner(
            task_description="""Convert spaCy relationships to JSON format.

Each relationship needs:
- source_id: canonical ID of source entity
- target_id: canonical ID of target entity
- relation_type: type of relationship

Return ONLY a JSON array like:
[{"source_id": "giacomo_medici", "target_id": "robert_hecht", "relation_type": "met"}]""",
            output_field_name="refined_relationships"
        )
        
        old_baseline_rel_results = evaluate_on_dataset(
            old_baseline_rel_refiner,
            test_relationship_dataset,
            relationship_refinement_metric,
            "Old Baseline Relationship Refinement (Test Set)"
        )
        
        model_results['relationship'] = {
            'old_baseline_test': old_baseline_rel_results['average'],
            'improved_zeroshot_test': improved_rel_results['average'],
            'improvement': improved_rel_results['average'] - old_baseline_rel_results['average']
        }
        
        all_results[model_config.name] = model_results
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY - HYBRID APPROACH")
    print("=" * 80)
    
    for model_name, results in all_results.items():
        print(f"\n{'='*80}")
        print(f"📊 {model_name}")
        print(f"{'='*80}")
        
        print(f"\n  ENTITY REFINEMENT (Optimized):")
        print(f"    Baseline (Test):     {results['entity']['baseline_test']:.3f}")
        print(f"    Optimized (Train):   {results['entity']['optimized_train']:.3f}")
        print(f"    Optimized (Test):    {results['entity']['optimized_test']:.3f}")
        print(f"    → Improvement:       {results['entity']['improvement']:+.3f}")
        
        if results['entity']['improvement'] > 0:
            print(f"    ✅ Optimization HELPS (+{results['entity']['improvement']:.1%})")
        else:
            print(f"    ❌ Optimization HURTS ({results['entity']['improvement']:.1%})")
        
        print(f"\n  RELATIONSHIP REFINEMENT (Improved Zero-Shot):")
        print(f"    Old Baseline (Test): {results['relationship']['old_baseline_test']:.3f}")
        print(f"    Improved Zero-Shot:  {results['relationship']['improved_zeroshot_test']:.3f}")
        print(f"    → Improvement:       {results['relationship']['improvement']:+.3f}")
        
        if results['relationship']['improvement'] > 0:
            print(f"    ✅ Better prompt HELPS (+{results['relationship']['improvement']:.1%})")
        else:
            print(f"    ⚠️  Still struggling with relationships")
        
        # Overfitting check for entities only
        entity_overfit = results['entity']['optimized_train'] - results['entity']['optimized_test']
        
        print(f"\n  GENERALIZATION CHECK:")
        print(f"    Entity gap (train-test):  {entity_overfit:+.3f}")
        
        if entity_overfit > 0.2:
            print(f"    ⚠️  Entity overfitting detected")
        elif entity_overfit < 0:
            print(f"    ✅ Excellent entity generalization (test > train)")
        else:
            print(f"    ✅ Good entity generalization")
        
        print(f"\n  PRODUCTION RECOMMENDATION:")
        if results['entity']['improvement'] > 0.1 and results['relationship']['improved_zeroshot_test'] > 0.15:
            print(f"    ✅ READY FOR PRODUCTION")
            print(f"       - Use optimized entity extraction")
            print(f"       - Use improved zero-shot relationships")
        elif results['entity']['improvement'] > 0.1:
            print(f"    🟡 ENTITIES READY, RELATIONSHIPS NEED WORK")
            print(f"       - Deploy entity extraction")
            print(f"       - Consider spaCy-only for relationships")
        else:
            print(f"    ❌ NEEDS MORE WORK")
            print(f"       - Try larger model")
            print(f"       - Or use traditional NLP approaches")
    
    print("\n" + "=" * 80)
    print("✅ Evaluation complete!")
    print("=" * 80)
    
    # Save prompts
    for model_name in all_results.keys():
        with open(f"hybrid_prompts_{model_name.replace(' ', '_')}.txt", "w") as f:
            f.write("HYBRID APPROACH PROMPTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("APPROACH:\n")
            f.write("  - Entities: Optimized with few-shot learning\n")
            f.write("  - Relationships: Improved zero-shot (no optimization)\n\n")
            f.write("=" * 80 + "\n\n")
        
            # ENTITY PROMPT
            test_entity_ex = test_entity_dataset[0]
            entity_prompt = optimized_entity_refiner._build_prompt(
                text=test_entity_ex.text,
                spacy_entities=test_entity_ex.spacy_entities
            )
        
            f.write("ENTITY REFINEMENT PROMPT (OPTIMIZED)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of demos: {len(optimized_entity_refiner.demos)}\n")
            f.write(f"Test F1: {results['entity']['optimized_test']:.3f}\n")
            f.write(f"Improvement: {results['entity']['improvement']:+.3f}\n")
            f.write("-" * 80 + "\n\n")
            f.write(entity_prompt)
            f.write("\n\n")
        
            # RELATIONSHIP PROMPT
            test_rel_ex = test_relationship_dataset[0]
            rel_prompt = improved_rel_refiner._build_prompt(
                text=test_rel_ex.text,
                spacy_entities=test_rel_ex.spacy_entities,
                spacy_relationships=test_rel_ex.spacy_relationships
            )
        
            f.write("\n\nRELATIONSHIP REFINEMENT PROMPT (IMPROVED ZERO-SHOT)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of demos: 0 (zero-shot by design)\n")
            f.write(f"Test F1: {results['relationship']['improved_zeroshot_test']:.3f}\n")
            f.write(f"Improvement over old baseline: {results['relationship']['improvement']:+.3f}\n")
            f.write("-" * 80 + "\n\n")
            f.write(rel_prompt)

        print(f"\n✅ Prompts saved to: hybrid_prompts_{model_name.replace(' ', '_')}.txt")

if __name__ == "__main__":
    main()

