"""
Global Twin — Model Registry

Serializes trained models and metadata to disk for persistence
and provides a loading interface for the dashboard and simulation.
"""

import os
import json
import pickle
from datetime import datetime


REGISTRY_DIR = "models/registry"


def save_registry(results, registry_dir=REGISTRY_DIR):
    """
    Save trained models and metadata to disk.
    
    Creates:
        models/registry/
            {target_name}/
                model.pkl       — serialized best sklearn model
                metadata.json   — metrics, model name, feature list, timestamps
    
    Args:
        results: Output from train_models().
        registry_dir: Root directory for the registry.
    """
    os.makedirs(registry_dir, exist_ok=True)
    
    manifest = {}
    
    for target, data in results.items():
        target_dir = os.path.join(registry_dir, target)
        os.makedirs(target_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(target_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(data['model'], f)
        
        # Save metadata
        metadata = {
            "target": target,
            "model_name": data['model_name'],
            "metrics": data['metrics'],
            "feature_names": data['feature_names'],
            "trained_at": data.get('trained_at', datetime.now().isoformat()),
            "train_size": data.get('train_size', 0),
            "test_size": data.get('test_size', 0),
            "top_features": data['feature_importances'].head(10).to_dict('records'),
            "all_candidates": [
                {"model_name": c['model_name'], "metrics": c['metrics']}
                for c in data.get('all_candidates', [])
            ],
        }
        
        meta_path = os.path.join(target_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        manifest[target] = {
            "model_path": model_path,
            "metadata_path": meta_path,
            "model_name": data['model_name'],
            "rmse": data['metrics']['RMSE'],
        }
        
        print(f"  ✓ Saved {target} → {data['model_name']} (RMSE={data['metrics']['RMSE']:.4f})")
    
    # Save manifest
    manifest_path = os.path.join(registry_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n  Registry saved to {registry_dir}/ ({len(manifest)} models)")
    return manifest


def load_registry(registry_dir=REGISTRY_DIR):
    """
    Load all models and metadata from the registry.
    
    Returns:
        dict mapping target_name → {
            'model': sklearn model,
            'model_name': str,
            'feature_names': list,
            'metrics': dict,
        }
    """
    manifest_path = os.path.join(registry_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No registry found at {registry_dir}/. Train models first.")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    results = {}
    for target, info in manifest.items():
        # Load model
        with open(info['model_path'], 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        with open(info['metadata_path'], 'r') as f:
            metadata = json.load(f)
        
        results[target] = {
            'model': model,
            'model_name': metadata['model_name'],
            'feature_names': metadata['feature_names'],
            'metrics': metadata['metrics'],
        }
        
        print(f"  ✓ Loaded {target} → {metadata['model_name']}")
    
    return results


def list_registry(registry_dir=REGISTRY_DIR):
    """Print a summary of the model registry."""
    manifest_path = os.path.join(registry_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print("  No registry found.")
        return
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"\n  Model Registry ({len(manifest)} models):")
    print(f"  {'Target':30s} {'Model':20s} {'RMSE':>10s}")
    print("  " + "-" * 62)
    for target, info in manifest.items():
        print(f"  {target:30s} {info['model_name']:20s} {info['rmse']:10.4f}")
