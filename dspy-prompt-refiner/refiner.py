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

    def _make_request(self, prompt: str, **kwargs) -> str:
        """Internal method to make the actual API request."""
        print(f"LocalLLM._make_request called for {self.config.name}!")
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
# DSPy Signatures (Let DSPy handle prompt optimization!)
# ============================================================================

class RefineEntities(dspy.Signature):
    """Review spaCy entities and refine them. Add missing entities. Return valid JSON array."""
    
    text = dspy.InputField(desc="The original text to analyze")
    spacy_entities = dspy.InputField(desc="JSON array of entities from spaCy with fields: text, label, start, end")
    refined_entities = dspy.OutputField(desc='JSON array of refined entities with fields: text, label, category. Format: [{"text": "...", "label": "PERSON/ORG/GPE/etc", "category": "..."}]')

class RefineRelationships(dspy.Signature):
    """Review spaCy relationships and refine them. Add missing relationships. Return valid JSON array."""
    
    text = dspy.InputField(desc="The original text to analyze")
    entities = dspy.InputField(desc="JSON array of all entities")
    spacy_relationships = dspy.InputField(desc="JSON array of relationships from spaCy with fields: verb, lemma, subject, object, pattern")
    refined_relationships = dspy.OutputField(desc='JSON array of refined relationships with fields: subject, predicate, object, confidence. Format: [{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}]')

# ============================================================================
# Custom Predictor with JSON Parsing
# ============================================================================

class JSONPredict(dspy.Module):
    """Wrapper around dspy.Predict that handles JSON parsing for local LLMs."""
    
    def __init__(self, signature):
        super().__init__()
        self.predict = dspy.Predict(signature)
        self.signature = signature
    
    def forward(self, **kwargs):
        # Call the underlying predictor
        try:
            result = self.predict(**kwargs)
            
            # Get the output field name (should be the last field)
            output_fields = [k for k, v in self.signature.output_fields.items()]
            if not output_fields:
                return result
            
            output_field = output_fields[0]
            raw_output = getattr(result, output_field, None)
            
            if raw_output is None:
                return result
            
            # Try to parse as JSON
            parsed_json = self._extract_and_parse_json(raw_output)
            
            # Update the result
            setattr(result, output_field, parsed_json)
            return result
            
        except Exception as e:
            print(f"⚠️  JSONPredict error: {e}")
            # Return empty structure
            output_fields = [k for k, v in self.signature.output_fields.items()]
            if output_fields:
                result = dspy.Prediction()
                setattr(result, output_fields[0], [])
                return result
            raise
    
    def _extract_and_parse_json(self, text: str):
        """Extract and parse JSON from text response."""
        if isinstance(text, (list, dict)):
            return text
        
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Find JSON array or object
        match = re.search(r'(\[.*?\]|\{.*?\})', text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                parsed = json.loads(json_str)
                if not isinstance(parsed, list):
                    parsed = [parsed] if isinstance(parsed, dict) else []
                return parsed
            except json.JSONDecodeError:
                pass
        
        # Fallback: return empty list
        return []

# ============================================================================
# Pipeline with Proper DSPy Integration
# ============================================================================

class KGExtractionPipeline(dspy.Module):
    """Two-stage knowledge graph extraction pipeline using DSPy optimization."""
    
    def __init__(self):
        super().__init__()
        self.spacy_extractor = SpacyExtractor()
        # Use DSPy's Predict with our signatures - DSPy will optimize these!
        self.refine_entities = JSONPredict(RefineEntities)
        self.refine_relationships = JSONPredict(RefineRelationships)
    
    def forward(self, text: str):
        # Stage 1: spaCy extraction
        spacy_result = self.spacy_extractor.extract_with_spacy(text)
        
        # Stage 2a: LLM refines entities (DSPy will optimize this prompt!)
        entity_result = self.refine_entities(
            text=text,
            spacy_entities=json.dumps(spacy_result['entities'])
        )
        
        refined_entities = entity_result.refined_entities if hasattr(entity_result, 'refined_entities') else []
        if not refined_entities:
            refined_entities = spacy_result['entities']
        
        # Stage 2b: LLM refines relationships (DSPy will optimize this prompt!)
        rel_result = self.refine_relationships(
            text=text,
            entities=json.dumps(refined_entities),
            spacy_relationships=json.dumps(spacy_result['relationships'])
        )
        
        refined_relationships = rel_result.refined_relationships if hasattr(rel_result, 'refined_relationships') else []
        if not refined_relationships:
            refined_relationships = spacy_result['relationships']
        
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
# Prompt Display Functions
# ============================================================================

def display_dspy_prompt(predictor, name: str):
    """Display the actual DSPy-generated prompt."""
    print(f"\n📝 {name}:")
    print("-" * 80)
    
    if hasattr(predictor, 'predict'):
        inner_predictor = predictor.predict
    else:
        inner_predictor = predictor
    
    # Display signature
    if hasattr(inner_predictor, 'signature'):
        sig = inner_predictor.signature
        print(f"Signature: {sig.__name__ if hasattr(sig, '__name__') else sig}")
        print(f"Docstring: {sig.__doc__ if hasattr(sig, '__doc__') else 'None'}")
        
        if hasattr(sig, 'instructions'):
            print(f"\nInstructions: {sig.instructions}")
        
        print("\nInput Fields:")
        for name, field in sig.input_fields.items():
            desc = field.json_schema_extra.get('desc', '') if hasattr(field, 'json_schema_extra') else ''
            print(f"  - {name}: {desc}")
        
        print("\nOutput Fields:")
        for name, field in sig.output_fields.items():
            desc = field.json_schema_extra.get('desc', '') if hasattr(field, 'json_schema_extra') else ''
            print(f"  - {name}: {desc}")
    
    # Display demos (few-shot examples)
    if hasattr(inner_predictor, 'demos') and inner_predictor.demos:
        print(f"\n✨ FEW-SHOT EXAMPLES ({len(inner_predictor.demos)} demos):")
        for i, demo in enumerate(inner_predictor.demos, 1):
            print(f"\n  Example {i}:")
            if hasattr(demo, '__dict__'):
                for key, value in demo.__dict__.items():
                    if not key.startswith('_'):
                        value_str = str(value)[:150]
                        print(f"    {key}: {value_str}...")
    else:
        print("\n⚠️  No few-shot examples (baseline prompt)")

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
        "Giacomo Medici started dealing in antiquities in Rome during the 1960s. In July 1967, he was convicted in Italy of receiving looted artefacts, though in the same year he met and became an important supplier of antiquities to US dealer Robert Hecht. In 1968, Medici opened the gallery Antiquaria Romana in Rome and began to explore business opportunities in Switzerland. It is widely believed that in December 1971 he bought the illegally-excavated Euphronios (Sarpedon) krater from tombaroli before transporting it to Switzerland and selling it to Hecht.",
        "By the late 1980s, Medici had developed commercial relations with other major antiquities dealers including Robin Symes, Frieda Tchacos, Nikolas Koutoulakis, Robert Hecht, and the brothers Ali and Hicham Aboutaam. He was the ultimate source of artefacts that would subsequently be sold through dealers or auction houses to private collectors, including Lawrence and Barbara Fleischman, Maurice Tempelsman, Shelby White and Leon Levy, the Hunt brothers, George Ortiz, and José Luis Várez Fisa, and to museums including the J. Paul Getty, the Metropolitan Museum of Art, the Cleveland Museum of Art, and the Boston Museum of Fine Arts.",
        "Robert Hecht claimed to be acting on ten percent commission as agent for the krater's owner, whom he identified as Lebanese collector and dealer Dikran Sarrafian. Hecht supplied two documents of provenance for the acquisitions committee meeting that approved the purchase. First was a letter dated 10 July 1971, written by Sarrafian to Hecht, in which Sarrafian declared that he would deliver the vase to Hecht in expectation of a final sale price of $1 million. Second was another letter from Sarrafian to Hecht, dated 9 September 1972, stating that Sarrafian's father had obtained the krater in 1920 in London and that because it was in fragments it had been sent to Switzerland for restoration three years before the writing of the letter.",
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
    
    # Display baseline prompts
    print("\n" + "=" * 80)
    print("BASELINE PROMPTS (Before Optimization)")
    print("=" * 80)
    display_dspy_prompt(pipeline.refine_entities, "ENTITY REFINEMENT")
    display_dspy_prompt(pipeline.refine_relationships, "RELATIONSHIP REFINEMENT")
    
    # Run baseline evaluation
    print("\n" + "=" * 80)
    print("BASELINE EVALUATION")
    print("=" * 80)
    baseline_scores = []
    for i, example in enumerate(trainset[:2]):
        try:
            print(f"\nProcessing example {i+1}...")
            pred = pipeline(text=example.text)
            score = kg_metric(None, pred)
            baseline_scores.append(score)
            print(f"✅ Score: {score:.2f} | Entities: {len(pred.entities)}, Relationships: {len(pred.relationships)}")
        except Exception as e:
            print(f"⚠️  Error: {e}")
            baseline_scores.append(0.0)
    
    baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    print(f"\n✅ Baseline average: {baseline_avg:.2f}")
    
    # Optimize with DSPy
    print("\n" + "=" * 80)
    print("DSPY OPTIMIZATION")
    print("=" * 80)
    print("DSPy will now:")
    print("  1. Run the pipeline on training examples")
    print("  2. Identify successful examples")
    print("  3. Add them as few-shot demonstrations to the prompts")
    print("  4. Create optimized prompts with examples")
    print("\nThis may take a few minutes...")
    
    try:
        optimizer = dspy.BootstrapFewShot(
            metric=kg_metric,
            max_bootstrapped_demos=3,
            max_labeled_demos=3
        )
        
        optimized_pipeline = optimizer.compile(
            pipeline,
            trainset=trainset[:4]
        )
        
        print("\n✅ Optimization complete!")
        
        # Display optimized prompts
        print("\n" + "=" * 80)
        print("OPTIMIZED PROMPTS (After DSPy Optimization)")
        print("=" * 80)
        display_dspy_prompt(optimized_pipeline.refine_entities, "ENTITY REFINEMENT")
        display_dspy_prompt(optimized_pipeline.refine_relationships, "RELATIONSHIP REFINEMENT")
        
        # Evaluate optimized pipeline
        print("\n" + "=" * 80)
        print("OPTIMIZED EVALUATION")
        print("=" * 80)
        optimized_scores = []
        for i, example in enumerate(trainset[:2]):
            try:
                print(f"\nProcessing example {i+1}...")
                pred = optimized_pipeline(text=example.text)
                score = kg_metric(None, pred)
                optimized_scores.append(score)
                print(f"✅ Score: {score:.2f} | Entities: {len(pred.entities)}, Relationships: {len(pred.relationships)}")
            except Exception as e:
                print(f"⚠️  Error: {e}")
                optimized_scores.append(0.0)
        
        optimized_avg = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0.0
        print(f"\n✅ Optimized average: {optimized_avg:.2f}")
        print(f"   Improvement: {(optimized_avg - baseline_avg):+.2f}")
        
    except Exception as e:
        print(f"⚠️  Optimization error: {e}")
        import traceback
        traceback.print_exc()
        optimized_pipeline = pipeline
    
    # Display LLM interaction history
    print("\n" + "=" * 80)
    print("LLM INTERACTION HISTORY")
    print("=" * 80)
    if hasattr(lm, 'history') and lm.history:
        print(f"\nTotal interactions: {len(lm.history)}")
        print("\nLast 3 interactions:")
        for i, interaction in enumerate(lm.history[-3:]):
            print(f"\n🔷 Interaction {len(lm.history) - 3 + i + 1}:")
            print(f"Model: {interaction['model']}")
            print(f"\nPrompt (first 500 chars):\n{interaction['prompt'][:500]}...")
            print(f"\nResponse (first 500 chars):\n{interaction['response'][:500]}...")
    
    print("\n" + "=" * 80)
    print("✅ Pipeline optimization complete!")
    print("=" * 80)
    print("\nKey takeaway:")
    print("The 'OPTIMIZED PROMPTS' section above shows how DSPy added few-shot")
    print("examples to improve the prompts. These are the actual prompts being")
    print("sent to your model after optimization.")
    
    return optimized_pipeline

if __name__ == "__main__":
    main()
