# Configuration file for DSPy KG Optimization
# Edit this file to customize your models and settings

import json
from pathlib import Path

# =============================================================================
# IMPORTANT: API Authentication
# =============================================================================
# If your local LLM server requires authentication (returns 401 errors),
# add your API key here. This will be used for all models unless overridden.
#
# Example:
#   GLOBAL_API_KEY = "your-api-key-here"
#
# For jan.ai: Check Settings → Advanced → API Key
# For LM Studio: Check Server → Authentication
# For Ollama: Usually no auth required (leave empty)

GLOBAL_API_KEY = ""  # Add API key here if server requires auth

# =============================================================================
# Local LLM Models Configuration
# =============================================================================

# Add your local models here. These should be running via jan.ai or similar
# OpenAI-compatible API servers
MODELS_CONFIG = {
    "trinity": {
        "name": "trinity-v1.2-7b",
        "api_url": "http://localhost:1337/v1/chat/completions",
        "api_key": GLOBAL_API_KEY,
        "max_tokens": 32768,
        "temperature": 0.1,
        "enabled": False  
    },
    "phi-2": {
        "name": "phi-2-3b",
        "api_url": "http://localhost:1337/v1/chat/completions",
        "api_key": GLOBAL_API_KEY,
        "max_tokens": 2048,
        "temperature": 0.1,
        "enabled": False
    },
    "openchat": {
      "name": "openchat-3.5-7b",
      "api_url": "http://127.0.0.1:1337/v1/chat/completions",
      "api_key": GLOBAL_API_KEY,
      "max_tokens": 8192,
      "temperature": 0.1,
      "enabled": True
    }
}


# =============================================================================
# Optimization Settings
# =============================================================================

OPTIMIZATION_CONFIG = {
    # Number of optimization iterations for DSPy
    "num_iterations": 10,
    
    # Whether to use DSPy's MIPRO optimizer
    "use_mipro": True,
    
    # Whether to use DSPy's BootstrapFewShot
    "use_bootstrap": True,
    
    # Number of few-shot examples to use
    "num_fewshot": 3,
    
    # Evaluation metrics weights
    "metric_weights": {
        "coreference": 0.3,
        "canonical_id": 0.2,
        "attributes": 0.2,
        "coverage": 0.15,
        "relationships": 0.15
    },
    
    # Timeout for each model call (seconds)
    "timeout": 300,
    
    # Whether to save intermediate results
    "save_intermediate": True
}


# =============================================================================
# Data Paths
# =============================================================================

DATA_PATHS = {
    # Directory containing your document corpus
    "documents_dir": "./documents",
    
    # Directory for ground truth annotations (optional)
    "ground_truth_dir": "./ground_truth",
    
    # Output directory for results
    "output_dir": "./results",
    
    # Trained model artifacts
    "artifacts_dir": "./artifacts"
}


# =============================================================================
# spaCy Settings
# =============================================================================

SPACY_CONFIG = {
    # spaCy model to use for Stage 1 extraction
    "model": "en_core_web_trf",
    
    # Entity types to extract
    "entity_types": [
        "PERSON", "ORG", "GPE", "LOC", 
        "WORK_OF_ART", "PRODUCT", "DATE", "MONEY"
    ],
    
    # Transaction verbs for relationship extraction
    "transaction_verbs": [
        'sell', 'buy', 'acquire', 'purchase', 'transfer', 
        'export', 'import', 'smuggle', 'consign', 'operate', 
        'deal', 'trade', 'traffic', 'supply', 'provide', 
        'convict', 'charge', 'sentence', 'arrest', 'raid', 
        'open', 'close', 'found', 'establish', 'meet', 
        'return', 'transport', 'loot', 'steal'
    ]
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_enabled_models():
    """Get list of enabled model configurations."""
    return [
        config for config in MODELS_CONFIG.values() 
        if config.get('enabled', True)
    ]


def save_config(filepath: str = "config.json"):
    """Save current configuration to JSON file."""
    config_dict = {
        "models": MODELS_CONFIG,
        "optimization": OPTIMIZATION_CONFIG,
        "data_paths": DATA_PATHS,
        "spacy": SPACY_CONFIG
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration saved to {filepath}")


def load_config(filepath: str = "config.json"):
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    global MODELS_CONFIG, OPTIMIZATION_CONFIG, DATA_PATHS, SPACY_CONFIG
    MODELS_CONFIG = config_dict.get("models", MODELS_CONFIG)
    OPTIMIZATION_CONFIG = config_dict.get("optimization", OPTIMIZATION_CONFIG)
    DATA_PATHS = config_dict.get("data_paths", DATA_PATHS)
    SPACY_CONFIG = config_dict.get("spacy", SPACY_CONFIG)
    
    print(f"Configuration loaded from {filepath}")


if __name__ == "__main__":
    # Save default configuration
    save_config()
