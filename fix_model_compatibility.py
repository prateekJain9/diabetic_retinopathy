#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix TensorFlow model compatibility issues with newer versions.
This script handles h5 model loading issues with newer TensorFlow/Keras.
"""

import os
import h5py
import json
import tempfile
import shutil
from pathlib import Path

def fix_h5_model_config(model_path):
    """
    Fix h5 model config that has incompatible parameters with newer TensorFlow.
    Creates a backup and fixes the original file.
    """
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return False
    
    print(f"[INFO] Fixing model: {model_path}")
    
    # Create backup
    backup_path = model_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(model_path, backup_path)
        print(f"[OK] Backup created: {backup_path}")
    
    try:
        # Read the h5 file
        with h5py.File(model_path, 'r+') as f:
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                
                # Handle bytes
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                
                # Parse JSON
                config = json.loads(config_str)
                
                # Remove problematic keys recursively
                def remove_problematic_keys(obj):
                    if isinstance(obj, dict):
                        # Remove quantization_config which causes issues in TF 2.18+
                        if 'quantization_config' in obj:
                            del obj['quantization_config']
                        
                        # Recursively process nested objects
                        for key, value in obj.items():
                            if isinstance(value, (dict, list)):
                                remove_problematic_keys(value)
                    
                    elif isinstance(obj, list):
                        for item in obj:
                            if isinstance(item, (dict, list)):
                                remove_problematic_keys(item)
                
                print("[INFO] Removing incompatible parameters...")
                remove_problematic_keys(config)
                
                # Write back
                new_config_str = json.dumps(config)
                f.attrs['model_config'] = new_config_str
                print("[OK] Model config updated")
        
        print("[SUCCESS] Model fixed successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to fix model: {e}")
        
        # Restore backup if fixing failed
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, model_path)
            print(f"[INFO] Restored from backup")
        
        return False


def try_load_fixed_model(model_path):
    """Try to load the model after fixing."""
    try:
        import tensorflow as tf
        print("\n[INFO] Attempting to load fixed model...")
        model = tf.keras.models.load_model(model_path)
        print("[SUCCESS] Model loaded successfully!")
        return model
    except Exception as e:
        print(f"[ERROR] Still cannot load model: {e}")
        return None


if __name__ == '__main__':
    # Paths
    project_root = r"C:\Users\Prateek Jain\Downloads\CUS00036_Pranjal Tomar_CompleteBackup_09-04-26"
    model_path = os.path.join(project_root, "dr_fedavg_model.h5")
    
    print("="*70)
    print("  TensorFlow Model Compatibility Fixer")
    print("="*70)
    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Model path: {model_path}")
    print()
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        exit(1)
    
    # Fix the model
    if fix_h5_model_config(model_path):
        # Try loading
        model = try_load_fixed_model(model_path)
        if model:
            print("\n[SUCCESS] Model is now compatible and can be loaded!")
            print("[INFO] You can now run: python test_model_metrics.py")
        else:
            print("\n[WARNING] Model was fixed but still has loading issues")
            print("[INFO] This might require manual intervention")
    else:
        print("\n[ERROR] Could not fix model automatically")
    
    print("\n" + "="*70)
