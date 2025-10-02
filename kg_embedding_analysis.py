#!/usr/bin/env python3
"""
Knowledge Graph Embedding Analysis - Command Line Script
Runs the full PyKEEN pipeline without Jupyter overhead.

Usage:
    python kg_embedding_analysis.py --input edges_table.csv --output results/
"""

import argparse
import json
import gc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
from pykeen import predict

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score


def setup_device():
    """Configure device for M3 optimization"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using M3 GPU (MPS backend)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    torch.set_num_threads(8)
    return device


def load_data(input_file):
    """Load and prepare triples data"""
    print(f"\nLoading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    if 'attributes' in df.columns:
        df = df.drop(columns='attributes')
    
    df = df.rename(columns={
        'source': 'subject',
        'target': 'object', 
        'relation': 'predicate'
    })
    
    print(f"Loaded {len(df)} triples")
    return df


def create_triples_factory(df):
    """Create PyKEEN triples factory"""
    triples_factory = TriplesFactory.from_labeled_triples(
        triples=df[['subject', 'predicate', 'object']].values,
    )
    
    training, validation, testing = triples_factory.split([0.7, 0.15, 0.15], random_state=42)
    
    id_to_entity = {v: k for k, v in triples_factory.entity_to_id.items()}
    id_to_relation = {v: k for k, v in triples_factory.relation_to_id.items()}
    
    return triples_factory, training, validation, testing, id_to_entity, id_to_relation


def train_model(model_name, config, training, validation, testing, device):
    """Train a single model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Handle complex models on MPS
    complex_models = {'RotatE', 'ComplEx'}
    use_device = 'cpu' if (str(device) == 'mps' and model_name in complex_models) else str(device)
    
    if use_device == 'cpu' and str(device) == 'mps':
        print(f"Warning: {model_name} using CPU (MPS limitation)")
    
    result = pipeline(
        training=training,
        validation=validation,
        testing=testing,
        **config,
        training_kwargs={
            'num_epochs': 300,
            'batch_size': 256,
            'use_tqdm': True,  # Progress bar works in CLI
            'use_tqdm_batch': False,
        },
        device=use_device,
        random_seed=42,
    )
    
    print(f"Training complete for {model_name}")
    
    # Force garbage collection
    gc.collect()
    
    return result


def evaluate_model(model_name, result, testing, training, validation):
    """Evaluate model performance"""
    evaluator = RankBasedEvaluator()
    
    metrics = evaluator.evaluate(
        result.model,
        testing.mapped_triples,
        additional_filter_triples=[training.mapped_triples, validation.mapped_triples]
    )
    
    return {
        'Hits@1': metrics.get_metric('hits@1'),
        'Hits@3': metrics.get_metric('hits@3'),
        'Hits@5': metrics.get_metric('hits@5'),
        'Hits@10': metrics.get_metric('hits@10'),
        'MRR': metrics.get_metric('mean_reciprocal_rank'),
    }


def extract_embeddings(model):
    """Extract and convert embeddings to real values"""
    entity_embeddings_raw = model.entity_representations[0](indices=None).detach().cpu()
    relation_embeddings_raw = model.relation_representations[0](indices=None).detach().cpu()
    
    # Convert complex to real if necessary
    if entity_embeddings_raw.is_complex():
        print("Converting complex embeddings to magnitude...")
        entity_embeddings = torch.abs(entity_embeddings_raw).numpy()
        relation_embeddings = torch.abs(relation_embeddings_raw).numpy()
    else:
        entity_embeddings = entity_embeddings_raw.numpy()
        relation_embeddings = relation_embeddings_raw.numpy()
    
    return entity_embeddings, relation_embeddings


def visualize_embeddings(entity_embeddings, entity_labels, output_dir, model_name):
    """Create PCA visualization"""
    print("\nCreating PCA visualization...")
    
    pca = PCA(n_components=2, random_state=42)
    entity_pca = pca.fit_transform(entity_embeddings)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(entity_pca[:, 0], entity_pca[:, 1], c='steelblue',
               s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    for i, label in enumerate(entity_labels):
        plt.annotate(
            text=label,
            xy=(entity_pca[i, 0], entity_pca[i, 1]),
            fontsize=8,
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    plt.title(f'PCA Projection - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_embeddings.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved PCA visualization")
    return pca


def compute_similarity(entity_embeddings, entity_labels, output_dir):
    """Compute and save similarity matrix"""
    print("\nComputing similarity matrix...")
    
    similarity_matrix = cosine_similarity(entity_embeddings)
    similarity_df = pd.DataFrame(similarity_matrix, index=entity_labels, columns=entity_labels)
    
    # Save CSV
    similarity_df.to_csv(output_dir / 'entity_similarity.csv')
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_df, cmap='coolwarm', center=0,
                annot=len(entity_labels) < 15, fmt='.2f',
                square=True, linewidths=0.5)
    plt.title('Entity Similarity Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved similarity analysis")
    return similarity_df


def cluster_entities(entity_embeddings, entity_labels, entity_pca, output_dir):
    """Perform clustering analysis"""
    print("\nClustering entities...")
    
    # Use heuristic k value
    optimal_k = min(5, max(2, len(entity_embeddings) // 5))
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(entity_embeddings)
    
    # Visualize clusters
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(entity_pca[:, 0], entity_pca[:, 1],
                         c=cluster_labels, cmap='viridis',
                         s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    for i, label in enumerate(entity_labels):
        plt.annotate(
            text=label,
            xy=(entity_pca[i, 0], entity_pca[i, 1]),
            fontsize=8,
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Entity Clusters (k={optimal_k})', fontsize=16, fontweight='bold')
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'entity_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save cluster assignments
    cluster_df = pd.DataFrame({
        'entity': entity_labels,
        'cluster': cluster_labels
    })
    cluster_df.to_csv(output_dir / 'cluster_assignments.csv', index=False)
    
    print(f"Saved clustering results (k={optimal_k})")
    return cluster_labels


def save_embeddings(entity_embeddings, relation_embeddings, entity_labels, relation_labels, output_dir):
    """Save embedding matrices"""
    print("\nSaving embeddings...")
    
    entity_df = pd.DataFrame(entity_embeddings, index=entity_labels)
    entity_df.to_csv(output_dir / 'entity_embeddings.csv')
    
    relation_df = pd.DataFrame(relation_embeddings, index=relation_labels)
    relation_df.to_csv(output_dir / 'relation_embeddings.csv')
    
    print("Saved embedding matrices")


def main():
    parser = argparse.ArgumentParser(description='Knowledge Graph Embedding Analysis')
    parser.add_argument('--input', default='output/edges_table.csv', help='Input CSV file')
    parser.add_argument('--output', default='embedding_results', help='Output directory')
    parser.add_argument('--models', nargs='+', default=['TransE', 'DistMult'],
                       help='Models to train (TransE, RotatE, ComplEx, DistMult)')
    parser.add_argument('--skip-viz', action='store_true', help='Skip visualizations')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = setup_device()
    
    # Load data
    df = load_data(args.input)
    triples_factory, training, validation, testing, id_to_entity, id_to_relation = create_triples_factory(df)
    
    entity_labels = [id_to_entity[i] for i in range(len(id_to_entity))]
    relation_labels = [id_to_relation[i] for i in range(len(id_to_relation))]
    
    # Model configurations
    models_config = {
        'TransE': {
            'model': 'TransE',
            'loss': 'softplus',
            'model_kwargs': {'embedding_dim': 64},
            'optimizer_kwargs': {'lr': 0.001},
        },
        'DistMult': {
            'model': 'DistMult',
            'model_kwargs': {'embedding_dim': 64},
            'optimizer_kwargs': {'lr': 0.001},
        },
        'RotatE': {
            'model': 'RotatE',
            'model_kwargs': {'embedding_dim': 64},
            'optimizer_kwargs': {'lr': 0.001},
        },
        'ComplEx': {
            'model': 'ComplEx',
            'model_kwargs': {'embedding_dim': 64},
            'optimizer_kwargs': {'lr': 0.001},
        },
    }
    
    # Filter requested models
    models_to_train = {k: v for k, v in models_config.items() if k in args.models}
    
    # Train models
    results = {}
    all_metrics = {}
    
    for model_name, config in models_to_train.items():
        try:
            result = train_model(model_name, config, training, validation, testing, device)
            results[model_name] = result
            
            metrics = evaluate_model(model_name, result, testing, training, validation)
            all_metrics[model_name] = metrics
            
            print(f"\n{model_name} Metrics:")
            print(f"  MRR: {metrics['MRR']:.4f}")
            print(f"  Hits@10: {metrics['Hits@10']:.4f}")
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    # Save metrics
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(output_dir / 'model_metrics.csv')
    print(f"\nSaved metrics for {len(results)} models")
    
    # Use best model for analysis
    if results:
        best_model_name = metrics_df['MRR'].idxmax()
        best_model = results[best_model_name].model
        
        print(f"\n{'='*60}")
        print(f"Best model: {best_model_name}")
        print(f"{'='*60}")
        
        # Extract embeddings
        entity_embeddings, relation_embeddings = extract_embeddings(best_model)
        
        # Save embeddings
        save_embeddings(entity_embeddings, relation_embeddings, entity_labels, relation_labels, output_dir)
        
        if not args.skip_viz:
            # Visualizations
            pca = visualize_embeddings(entity_embeddings, entity_labels, output_dir, best_model_name)
            entity_pca = pca.transform(entity_embeddings)
            
            # Similarity analysis
            similarity_df = compute_similarity(entity_embeddings, entity_labels, output_dir)
            
            # Clustering
            cluster_labels = cluster_entities(entity_embeddings, entity_labels, entity_pca, output_dir)
        
        # Save best model
        results[best_model_name].save_to_directory(str(output_dir / 'best_model'))
        
        print(f"\n{'='*60}")
        print(f"Analysis complete! Results in: {output_dir.absolute()}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
