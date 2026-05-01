#!/usr/bin/env python3
"""
===============================================================================
SETUP SCRIPT - Prepare Test Data for DR Model Evaluation
===============================================================================

This script:
1. Extracts dataset.zip to get DR and No_DR images
2. Creates proper folder structure for testing
3. Validates model file exists
4. Prepares environment for running evaluation scripts

Usage:
    python setup_test_data.py

===============================================================================
"""

import os
import zipfile
import shutil
from pathlib import Path

# ============================================================================
#  CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ZIP = os.path.join(PROJECT_ROOT, 'dataset.zip')
DATASET_TEST_DIR = os.path.join(PROJECT_ROOT, 'dataset_test')
MODEL_FILE = os.path.join(PROJECT_ROOT, 'dr_fedavg_model.h5')

print("\n" + "="*70)
print("   🩺 DR Model - Test Data Setup")
print("="*70)

# ============================================================================
#  STEP 1: CHECK PREREQUISITES
# ============================================================================

print("\n✓ Step 1: Checking prerequisites...")
print(f"  Project root    : {PROJECT_ROOT}")
print(f"  Model file      : {MODEL_FILE}")
print(f"  Dataset zip     : {DATASET_ZIP}")

# Check model exists
if os.path.exists(MODEL_FILE):
    print(f"  ✅ Model file found ({os.path.getsize(MODEL_FILE) / (1024*1024):.1f} MB)")
else:
    print(f"  ❌ Model file NOT found: {MODEL_FILE}")
    print("     The file should be at the project root")

# Check dataset.zip exists
if os.path.exists(DATASET_ZIP):
    print(f"  ✅ Dataset zip found ({os.path.getsize(DATASET_ZIP) / (1024*1024):.1f} MB)")
else:
    print(f"  ❌ Dataset zip NOT found: {DATASET_ZIP}")
    print("     Cannot proceed without dataset.zip")
    exit(1)

# ============================================================================
#  STEP 2: EXTRACT DATASET
# ============================================================================

print("\n✓ Step 2: Extracting dataset...")

try:
    # Create extraction directory
    extraction_dir = os.path.join(PROJECT_ROOT, 'dataset_extracted')
    os.makedirs(extraction_dir, exist_ok=True)
    
    # Extract zip file
    print(f"  Extracting {os.path.basename(DATASET_ZIP)}...")
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(extraction_dir)
    
    print(f"  ✅ Extracted to: {extraction_dir}")
    
    # List what was extracted
    extracted_items = os.listdir(extraction_dir)
    print(f"  Contents: {', '.join(extracted_items)}")
    
except Exception as e:
    print(f"  ❌ Error extracting zip: {e}")
    exit(1)

# ============================================================================
#  STEP 3: ORGANIZE INTO TEST STRUCTURE
# ============================================================================

print("\n✓ Step 3: Organizing into test structure...")

try:
    # Create dataset_test directory
    os.makedirs(DATASET_TEST_DIR, exist_ok=True)
    print(f"  Created: {DATASET_TEST_DIR}")
    
    # Find DR and No_DR folders in extracted content
    extracted_dirs = []
    for root, dirs, files in os.walk(extraction_dir):
        for d in dirs:
            extracted_dirs.append(os.path.join(root, d))
    
    dr_source = None
    no_dr_source = None
    
    # Search for DR and No_DR folders
    for directory in extracted_dirs:
        dir_name = os.path.basename(directory)
        if dir_name == 'DR':
            dr_source = directory
        elif dir_name == 'No_DR':
            no_dr_source = directory
    
    # Also check at extraction_dir level
    if os.path.exists(os.path.join(extraction_dir, 'DR')):
        dr_source = os.path.join(extraction_dir, 'DR')
    if os.path.exists(os.path.join(extraction_dir, 'No_DR')):
        no_dr_source = os.path.join(extraction_dir, 'No_DR')
    
    # Copy to test structure
    if dr_source and os.path.exists(dr_source):
        dr_dest = os.path.join(DATASET_TEST_DIR, 'DR')
        if os.path.exists(dr_dest):
            shutil.rmtree(dr_dest)
        shutil.copytree(dr_source, dr_dest)
        dr_count = len(os.listdir(dr_dest))
        print(f"  ✅ Copied {dr_count} DR images")
    else:
        print(f"  ⚠️  Could not find DR folder in extracted data")
    
    if no_dr_source and os.path.exists(no_dr_source):
        no_dr_dest = os.path.join(DATASET_TEST_DIR, 'No_DR')
        if os.path.exists(no_dr_dest):
            shutil.rmtree(no_dr_dest)
        shutil.copytree(no_dr_source, no_dr_dest)
        no_dr_count = len(os.listdir(no_dr_dest))
        print(f"  ✅ Copied {no_dr_count} No_DR images")
    else:
        print(f"  ⚠️  Could not find No_DR folder in extracted data")
    
except Exception as e:
    print(f"  ❌ Error organizing data: {e}")
    exit(1)

# ============================================================================
#  STEP 4: VALIDATE STRUCTURE
# ============================================================================

print("\n✓ Step 4: Validating structure...")

dr_dir = os.path.join(DATASET_TEST_DIR, 'DR')
no_dr_dir = os.path.join(DATASET_TEST_DIR, 'No_DR')

dr_count = len(os.listdir(dr_dir)) if os.path.exists(dr_dir) else 0
no_dr_count = len(os.listdir(no_dr_dir)) if os.path.exists(no_dr_dir) else 0

if dr_count > 0 and no_dr_count > 0:
    print(f"  ✅ DR images      : {dr_count}")
    print(f"  ✅ No_DR images   : {no_dr_count}")
    print(f"  ✅ Total images   : {dr_count + no_dr_count}")
else:
    print(f"  ❌ Dataset structure incomplete")
    print(f"     DR images    : {dr_count}")
    print(f"     No_DR images : {no_dr_count}")

# ============================================================================
#  STEP 5: DISPLAY SUMMARY
# ============================================================================

print("\n" + "="*70)
print("   ✅ SETUP COMPLETE!")
print("="*70)

print("\n📊 Test Data Ready:")
print(f"  Location: {DATASET_TEST_DIR}")
print(f"  Structure:")
print(f"    ├── DR/        ({dr_count} images)")
print(f"    └── No_DR/     ({no_dr_count} images)")

print("\n🚀 Next Steps:")
print("  1. Navigate to code directory:")
print("     cd code")
print()
print("  2. Run evaluation script:")
print("     python test_model_metrics.py")
print()
print("  3. Or open Jupyter notebook:")
print("     jupyter notebook Model_Evaluation_Metrics.ipynb")

print("\n💾 Cleanup:")
print(f"  To free space, remove: {extraction_dir}")
print(f"  The test data is in: {DATASET_TEST_DIR}")

print("\n" + "="*70)
