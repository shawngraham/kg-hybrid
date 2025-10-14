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

# ============================================================================
# LocalLLM DSPy Wrapper with Better Response Handling
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

    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON from text that might contain markdown or extra content."""
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Try to find raw JSON object or array
        json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        return text

    def _make_request(self, prompt: str, **kwargs) -> str:
        """Internal method to make the actual API request."""
        print("LocalLLM._make_request called!")
        print("  Prompt:", prompt[:200], "...")
        
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

            if "choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    
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
            return '{"error": "request_failed"}'
        except (KeyError, IndexError, ValueError) as e:
            print(f"Error parsing response from {self.config.name}: {e}")
            if response is not None:
                print("Response content:", response.text)
            return '{"error": "parse_failed"}'

    def __call__(self, prompt=None, messages=None, **kwargs):
        """DSPy-compatible __call__ interface."""
        if prompt is not None:
            return self._make_request(prompt, **kwargs)
        elif messages is not None:
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
# Dataset Class
# ============================================================================

class KGDataset:
    """Simple dataset class for knowledge graph extraction."""
    
    def __init__(self, examples: List[Dict]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# ============================================================================
# Stage 1: spaCy Extraction
# ============================================================================

class SpacyExtractor:
    """Stage 1: Extract entities and relationships with spaCy."""
    
    def __init__(self):
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_trf")
        print("✅ spaCy loaded")
    
    def find_entity_for_span(self, token, entities):
        """Find which entity (if any) contains or matches this token/span."""
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

# ============================================================================
# Stage 2: Custom DSPy Modules with Better Error Handling
# ============================================================================

class EntityRefiner(dspy.Module):
    """Custom module to refine entities with robust error handling."""
    
    def __init__(self):
        super().__init__()
        self.prompt_template = """Given the following text and entities extracted by spaCy, refine and enhance the entity list.

Text: {text}

SpaCy Entities: {spacy_entities}

Instructions:
1. Review the entities and improve their categorization
2. Add any missing important entities
3. Return ONLY a JSON array with this exact format:
[
  {{"text": "entity name", "label": "PERSON/ORG/GPE/etc", "category": "description"}}
]

Return only the JSON array, no other text:"""

    def forward(self, text: str, spacy_entities: str):
        prompt = self.prompt_template.format(text=text, spacy_entities=spacy_entities)
        
        try:
            response = dspy.settings.lm(prompt)
            
            # Extract JSON from response
            json_str = self._extract_json(response)
            
            # Parse JSON
            refined = json.loads(json_str)
            
            if not isinstance(refined, list):
                refined = [refined] if isinstance(refined, dict) else []
            
            return dspy.Prediction(refined_entities=refined)
            
        except Exception as e:
            print(f"⚠️  Entity refinement error: {e}")
            # Fallback to original entities
            try:
                original = json.loads(spacy_entities)
                return dspy.Prediction(refined_entities=original)
            except:
                return dspy.Prediction(refined_entities=[])
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text response."""
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Find JSON array or object
        match = re.search(r'(\[.*?\]|\{.*?\})', text, re.DOTALL)
        if match:
            return match.group(1)
        
        return text.strip()

class RelationshipRefiner(dspy.Module):
    """Custom module to refine relationships with robust error handling."""
    
    def __init__(self):
        super().__init__()
        self.prompt_template = """Given the following text, entities, and relationships extracted by spaCy, refine and enhance the relationship list.

Text: {text}

Entities: {entities}

SpaCy Relationships: {spacy_relationships}

Instructions:
1. Review the relationships and improve them
2. Add any missing important relationships between entities
3. Return ONLY a JSON array with this exact format:
[
  {{"subject": "entity1", "predicate": "relationship_type", "object": "entity2", "confidence": 0.9}}
]

Return only the JSON array, no other text:"""

    def forward(self, text: str, entities: str, spacy_relationships: str):
        prompt = self.prompt_template.format(
            text=text,
            entities=entities,
            spacy_relationships=spacy_relationships
        )
        
        try:
            response = dspy.settings.lm(prompt)
            
            # Extract JSON from response
            json_str = self._extract_json(response)
            
            # Parse JSON
            refined = json.loads(json_str)
            
            if not isinstance(refined, list):
                refined = [refined] if isinstance(refined, dict) else []
            
            return dspy.Prediction(refined_relationships=refined)
            
        except Exception as e:
            print(f"⚠️  Relationship refinement error: {e}")
            # Fallback to original relationships
            try:
                original = json.loads(spacy_relationships)
                return dspy.Prediction(refined_relationships=original)
            except:
                return dspy.Prediction(refined_relationships=[])
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text response."""
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Find JSON array or object
        match = re.search(r'(\[.*?\]|\{.*?\})', text, re.DOTALL)
        if match:
            return match.group(1)
        
        return text.strip()

class KGExtractionPipeline(dspy.Module):
    """Two-stage knowledge graph extraction pipeline."""
    
    def __init__(self):
        super().__init__()
        self.spacy_extractor = SpacyExtractor()
        self.refine_entities = EntityRefiner()
        self.refine_relationships = RelationshipRefiner()
    
    def forward(self, text: str):
        # Stage 1: spaCy extraction
        spacy_result = self.spacy_extractor.extract_with_spacy(text)
        
        # Stage 2a: LLM refines entities
        entity_result = self.refine_entities(
            text=text,
            spacy_entities=json.dumps(spacy_result['entities'])
        )
        
        refined_entities = entity_result.refined_entities if hasattr(entity_result, 'refined_entities') else spacy_result['entities']
        
        # Stage 2b: LLM refines relationships
        rel_result = self.refine_relationships(
            text=text,
            entities=json.dumps(refined_entities),
            spacy_relationships=json.dumps(spacy_result['relationships'])
        )
        
        refined_relationships = rel_result.refined_relationships if hasattr(rel_result, 'refined_relationships') else spacy_result['relationships']
        
        return dspy.Prediction(
            entities=refined_entities,
            relationships=refined_relationships,
            spacy_entities=spacy_result['entities'],
            spacy_relationships=spacy_result['relationships']
        )

# ============================================================================
# Evaluation Metric
# ============================================================================

def kg_metric(gold, pred, trace=None):
    """Evaluate knowledge graph extraction quality."""
    score = 0.0
    
    # Check if entities exist
    if 'entities' in pred and isinstance(pred.entities, list) and len(pred.entities) > 0:
        score += 0.3
    
    # Check if relationships exist
    if 'relationships' in pred and isinstance(pred.relationships, list) and len(pred.relationships) > 0:
        score += 0.3
    
    # Check if refined entities have expected fields
    if 'entities' in pred and isinstance(pred.entities, list):
        for ent in pred.entities:
            if isinstance(ent, dict) and 'text' in ent:
                score += 0.2 / max(len(pred.entities), 1)
                break
    
    # Check if refined relationships have expected fields
    if 'relationships' in pred and isinstance(pred.relationships, list):
        for rel in pred.relationships:
            if isinstance(rel, dict) and 'subject' in rel and 'object' in rel:
                score += 0.2 / max(len(pred.relationships), 1)
                break
    
    return score

# ============================================================================
# Main Optimization Pipeline
# ============================================================================

def main():
    print("=" * 80)
    print("DSPy Knowledge Graph Extraction Pipeline Optimizer")
    print("=" * 80)
    
    # Load models
    LOCAL_MODELS = load_models_from_config()
    if not LOCAL_MODELS:
        print("❌ No models available!")
        return
    
    print(f"\n📦 Loaded {len(LOCAL_MODELS)} model(s):")
    for model in LOCAL_MODELS:
        print(f"   - {model.name}")
    
    # Set up DSPy with the first model
    lm = LocalLLM(LOCAL_MODELS[0])
    dspy.settings.configure(lm=lm)
    
    # Create sample dataset
    print("\n📊 Creating sample dataset...")
    sample_texts = [
        "Giacomo Medici started dealing in antiquities in Rome during the 1960s (Silver 2009: 25). In July 1967, he was convicted in Italy of receiving looted artefacts, though in the same year he met and became an important supplier of antiquities to US dealer Robert Hecht (Silver 2009: 27-9). In 1968, Medici opened the gallery Antiquaria Romana in Rome and began to explore business opportunities in Switzerland (Silver 2009: 34). It is widely believed that in December 1971 he bought the illegally-excavated Euphronios (Sarpedon) krater from tombaroli before transporting it to Switzerland and selling it to Hecht (Silver 2009: 50).",
        "By the late 1980s, Medici had developed commercial relations with other major antiquities dealers including Robin Symes, Frieda Tchacos, Nikolas Koutoulakis, Robert Hecht, and the brothers Ali and Hicham Aboutaam (Watson and Todeschini 2007: 73-4). He was the ultimate source of artefacts that would subsequently be sold through dealers or auction houses to private collectors, including Lawrence and Barbara Fleischman, Maurice Tempelsman, Shelby White and Leon Levy, the Hunt brothers, George Ortiz, and José Luis Várez Fisa (Watson and Todeschini 2007: 112-34; Isman 2010), and to museums including the J. Paul Getty, the Metropolitan Museum of Art, the Cleveland Museum of Art, and the Boston Museum of Fine Arts.",
        "Robert Hecht claimed to be acting on ten percent commission as agent for the krater’s owner, whom he identified as Lebanese collector and dealer Dikran Sarrafian (Hoving 2001b). Hecht supplied two documents of provenance for the acquisitions committee meeting that approved the purchase. First was a letter dated 10 July 1971, written by Sarrafian to Hecht, in which Sarrafian declared that he would deliver the vase to Hecht in expectation of a final sale price of $1 million. Second was another letter from Sarrafian to Hecht, dated 9 September 1972, stating that Sarrafian’s father had obtained the krater in 1920 in London and that because it was in fragments it had been sent [to Switzerland] for restoration three years before the writing of the letter (Hoving 1993: 319; Hoving 2001c).",
        "Italian authorities raided Medici's warehouse in Geneva in 1995, finding thousands of photographs of looted artifacts.",
        "Leonardo Patterson donated fake antiquities to the Brooklyn Museum.",
    ]
    
    dataset_examples = []
    for text in sample_texts:
        dataset_examples.append({
            'text': text,
            'entities': [],
            'relationships': []
        })
    
    dataset = KGDataset(dataset_examples)
    
    # Create training examples for DSPy
    trainset = []
    for ex in dataset_examples:
        trainset.append(dspy.Example(text=ex['text']).with_inputs('text'))
    
    print(f"✅ Created dataset with {len(trainset)} examples")
    
    # Initialize pipeline
    print("\n🔧 Initializing extraction pipeline...")
    pipeline = KGExtractionPipeline()
    
    # Run baseline evaluation
    print("\n📊 Running baseline evaluation...")
    baseline_scores = []
    for i, example in enumerate(trainset[:2]):
        try:
            print(f"\n   Processing example {i+1}...")
            pred = pipeline(text=example.text)
            score = kg_metric(None, pred)
            baseline_scores.append(score)
            print(f"   ✅ Example {i+1} score: {score:.2f}")
            print(f"      Entities: {len(pred.entities)}, Relationships: {len(pred.relationships)}")
        except Exception as e:
            print(f"   ⚠️  Error on example {i+1}: {e}")
            baseline_scores.append(0.0)
    
    baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    print(f"\n✅ Baseline average score: {baseline_avg:.2f}")
    
    # Optimize with DSPy
    print("\n🚀 Starting DSPy optimization...")
    print("   (This may take a while...)")
    
    try:
        optimizer = dspy.BootstrapFewShot(
            metric=kg_metric,
            max_bootstrapped_demos=2,
            max_labeled_demos=2
        )
        
        optimized_pipeline = optimizer.compile(
            pipeline,
            trainset=trainset[:3]
        )
        
        print("✅ Optimization complete!")
        
        # Evaluate optimized pipeline
        print("\n📊 Evaluating optimized pipeline...")
        optimized_scores = []
        for i, example in enumerate(trainset[:2]):
            try:
                print(f"\n   Processing example {i+1}...")
                pred = optimized_pipeline(text=example.text)
                score = kg_metric(None, pred)
                optimized_scores.append(score)
                print(f"   ✅ Example {i+1} score: {score:.2f}")
                print(f"      Entities: {len(pred.entities)}, Relationships: {len(pred.relationships)}")
            except Exception as e:
                print(f"   ⚠️  Error on example {i+1}: {e}")
                optimized_scores.append(0.0)
        
        optimized_avg = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0.0
        print(f"\n✅ Optimized average score: {optimized_avg:.2f}")
        print(f"   Improvement: {(optimized_avg - baseline_avg):.2f}")
        
    except Exception as e:
        print(f"⚠️  Optimization error: {e}")
        import traceback
        traceback.print_exc()
        optimized_pipeline = pipeline
    
    # Extract and display optimized prompts
    print("\n" + "=" * 80)
    print("OPTIMIZED PROMPTS")
    print("=" * 80)
    
    # Entity refinement prompt
    print("\n📝 ENTITY REFINEMENT PROMPT:")
    print("-" * 80)
    if hasattr(pipeline, 'refine_entities') and hasattr(pipeline.refine_entities, 'prompt_template'):
        print(pipeline.refine_entities.prompt_template)
    
    # Relationship refinement prompt
    print("\n📝 RELATIONSHIP REFINEMENT PROMPT:")
    print("-" * 80)
    if hasattr(pipeline, 'refine_relationships') and hasattr(pipeline.refine_relationships, 'prompt_template'):
        print(pipeline.refine_relationships.prompt_template)
    
    # Display LLM interaction history
    print("\n" + "=" * 80)
    print("LLM INTERACTION HISTORY (Last 3 interactions)")
    print("=" * 80)
    if hasattr(lm, 'history') and lm.history:
        for i, interaction in enumerate(lm.history[-3:]):
            print(f"\n🔷 Interaction {i+1}:")
            print(f"Model: {interaction['model']}")
            print(f"\nPrompt:\n{interaction['prompt'][:800]}...")
            print(f"\nResponse:\n{interaction['response'][:800]}...")
    
    print("\n" + "=" * 80)
    print("✅ Pipeline optimization complete!")
    print("=" * 80)
    
    return optimized_pipeline

if __name__ == "__main__":
    main()
