#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
COMPREHENSIVE MODEL TEST METRICS & EVALUATION SUITE
Diabetic Retinopathy Detection (FedAvg + MobileNetV2)
===============================================================================

This script provides comprehensive evaluation metrics including:
- AUC-ROC & AUC-PR curves
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix with visualization
- Classification Report
- Per-class metrics (Sensitivity, Specificity, NPV, PPV)
- ROC & Precision-Recall curves
- Detailed performance analysis

Author: DR Detection Team
Date: 2026
===============================================================================
"""

import os
import sys
import io

# Ensure UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    import codecs
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, auc, balanced_accuracy_score, matthews_corrcoef,
    cohen_kappa_score
)
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
#  CONFIGURATION
# ============================================================================
# Model path - located in parent directory (project root)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'dr_fedavg_model.h5')

# Test data path - can be extracted from dataset.zip or custom path
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset_test')

IMG_SIZE = (224, 224)
CLASS_NAMES = ['DR (Positive)', 'No_DR (Negative)']
NUM_CLASSES = 2

# ============================================================================
#  UTILITY FUNCTIONS
# ============================================================================

def load_test_dataset(test_data_dir=None):
    """
    Load test dataset from directory structure:
    test_data_dir/
    ├── DR/
    └── No_DR/
    """
    if test_data_dir is None:
        print("  No test data directory provided. Please provide path to test dataset.")
        return None, None, None
    
    if not os.path.exists(test_data_dir):
        print(f" Test data directory not found: {test_data_dir}")
        return None, None, None
    
    X_test, y_test, image_names = [], [], []
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # Load DR images (label=0)
    dr_dir = os.path.join(test_data_dir, 'DR')
    if os.path.exists(dr_dir):
        files = [f for f in os.listdir(dr_dir) if f.lower().endswith(valid_exts)]
        for fname in files[:100]:  # Limit to 100 per class for demo
            try:
                img = Image.open(os.path.join(dr_dir, fname)).convert('RGB')
                img = img.resize(IMG_SIZE)
                X_test.append(np.array(img, dtype=np.float32) / 255.0)
                y_test.append(0)
                image_names.append(fname)
            except Exception as e:
                print(f"  Failed to load {fname}: {e}")
    
    # Load No_DR images (label=1)
    no_dr_dir = os.path.join(test_data_dir, 'No_DR')
    if os.path.exists(no_dr_dir):
        files = [f for f in os.listdir(no_dr_dir) if f.lower().endswith(valid_exts)]
        for fname in files[:100]:  # Limit to 100 per class for demo
            try:
                img = Image.open(os.path.join(no_dr_dir, fname)).convert('RGB')
                img = img.resize(IMG_SIZE)
                X_test.append(np.array(img, dtype=np.float32) / 255.0)
                y_test.append(1)
                image_names.append(fname)
            except Exception as e:
                print(f"  Failed to load {fname}: {e}")
    
    if len(X_test) == 0:
        print(" No images loaded from test directory")
        return None, None, None
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)
    
    print(f" Test dataset loaded: {X_test.shape[0]} images")
    print(f"   - DR samples: {np.sum(y_test == 0)}")
    print(f"   - No_DR samples: {np.sum(y_test == 1)}")
    
    return X_test, y_test, image_names


def load_model_safe():
    """Safely load the trained model with compatibility handling."""
    if not os.path.exists(MODEL_PATH):
        print(f" Model not found at: {MODEL_PATH}")
        return None
    
    try:
        # Try standard loading first
        print(f" [Loading] Attempting standard model load from: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        print(f" Model loaded successfully from: {MODEL_PATH}")
        return model
        
    except Exception as e:
        print(f" [Attempting alternative load method...]")
        print(f" [Details] {type(e).__name__}: {str(e)[:200]}...")
        
        try:
            # Try with custom objects for compatibility
            import tensorflow as tf
            custom_objects = {}
            model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
            print(f" Model loaded successfully (with compatibility mode)")
            return model
        except Exception as e2:
            print(f" Error loading model: {e2}")
            return None


def get_predictions(model, X_test):
    """Generate predictions and probabilities."""
    try:
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        return y_pred, y_pred_prob
    except Exception as e:
        print(f" Error generating predictions: {e}")
        return None, None


# ============================================================================
#  METRICS CALCULATION
# ============================================================================

class ModelMetrics:
    """Comprehensive metrics calculator for binary classification."""
    
    def __init__(self, y_true, y_pred, y_pred_prob):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_prob = y_pred_prob
        self.y_pred_prob_class1 = y_pred_prob[:, 1]  # Probability of class 1
        
    def calculate_all_metrics(self):
        """Calculate all available metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['Accuracy'] = accuracy_score(self.y_true, self.y_pred)
        metrics['Precision'] = precision_score(self.y_true, self.y_pred, zero_division=0)
        metrics['Recall'] = recall_score(self.y_true, self.y_pred, zero_division=0)
        metrics['F1-Score'] = f1_score(self.y_true, self.y_pred, zero_division=0)
        metrics['Balanced Accuracy'] = balanced_accuracy_score(self.y_true, self.y_pred)
        
        # AUC metrics
        metrics['AUC-ROC'] = roc_auc_score(self.y_true, self.y_pred_prob_class1)
        precision_curve, recall_curve, _ = precision_recall_curve(self.y_true, self.y_pred_prob_class1)
        metrics['AUC-PR'] = auc(recall_curve, precision_curve)
        
        # Correlation metrics
        metrics['Matthews Correlation Coefficient'] = matthews_corrcoef(self.y_true, self.y_pred)
        metrics['Cohen\'s Kappa'] = cohen_kappa_score(self.y_true, self.y_pred)
        
        # Confusion matrix based metrics
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        
        metrics['TP'] = tp
        metrics['TN'] = tn
        metrics['FP'] = fp
        metrics['FN'] = fn
        
        # Sensitivity (Recall for positive class)
        metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Specificity (Recall for negative class)
        metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Positive Predictive Value (Precision for positive class)
        metrics['Positive Predictive Value (PPV)'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Negative Predictive Value
        metrics['Negative Predictive Value (NPV)'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # False Positive Rate
        metrics['False Positive Rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False Negative Rate
        metrics['False Negative Rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Youden's Index
        metrics['Youden\'s Index'] = metrics['Sensitivity'] + metrics['Specificity'] - 1
        
        return metrics
    
    def get_confusion_matrix(self):
        """Get confusion matrix components."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        return cm
    
    def get_classification_report(self):
        """Get detailed classification report."""
        return classification_report(self.y_true, self.y_pred,
                                    target_names=CLASS_NAMES,
                                    digits=4)
    
    def get_roc_curve_data(self):
        """Get ROC curve data."""
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_prob_class1)
        auc_score = roc_auc_score(self.y_true, self.y_pred_prob_class1)
        return fpr, tpr, thresholds, auc_score
    
    def get_pr_curve_data(self):
        """Get Precision-Recall curve data."""
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_prob_class1)
        auc_pr = auc(recall, precision)
        return precision, recall, thresholds, auc_pr


# ============================================================================
#  VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Confusion matrix saved: {save_path}")
    plt.show()


def plot_roc_curve(fpr, tpr, auc_score, save_path=None):
    """Plot ROC curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='#e74c3c', lw=2.5,
            label=f'ROC Curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.5000)')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#e74c3c')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - AUC Score', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" ROC curve saved: {save_path}")
    plt.show()


def plot_pr_curve(precision, recall, auc_pr, save_path=None):
    """Plot Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, color='#3498db', lw=2.5,
            label=f'PR Curve (AUC = {auc_pr:.4f})')
    ax.axhline(y=np.mean([sum(np.array(CLASS_NAMES) == 'DR (Positive)'), 
                           sum(np.array(CLASS_NAMES) == 'No_DR (Negative)')]) / len(CLASS_NAMES),
               color='gray', linestyle='--', lw=1.5, label='Baseline')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve - AUC-PR Score', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" PR curve saved: {save_path}")
    plt.show()


def plot_metrics_comparison(metrics_dict, save_path=None):
    """Plot bar chart of key metrics."""
    key_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 
                   'Sensitivity', 'Specificity', 'AUC-ROC', 'AUC-PR']
    values = [metrics_dict.get(m, 0) for m in key_metrics]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71' if v >= 0.8 else '#f39c12' if v >= 0.6 else '#e74c3c' for v in values]
    bars = ax.bar(key_metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Key Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Metrics comparison saved: {save_path}")
    plt.show()


def plot_all_metrics_comprehensive(metrics_dict, cm, y_true, y_pred_prob, save_dir='./results'):
    """Generate comprehensive metrics visualization."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Comprehensive Model Evaluation Metrics', fontsize=16, fontweight='bold', y=1.00)
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0, 0], cbar_kws={'label': 'Count'})
    axes[0, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # ROC Curve
    fpr, tpr, thresholds, auc_score = ModelMetrics(y_true, np.argmax(y_pred_prob, axis=1), y_pred_prob).get_roc_curve_data()
    axes[0, 1].plot(fpr, tpr, color='#e74c3c', lw=2, label=f'AUC = {auc_score:.4f}')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0, 1].fill_between(fpr, tpr, alpha=0.15, color='#e74c3c')
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve (AUC-ROC)', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _, auc_pr = ModelMetrics(y_true, np.argmax(y_pred_prob, axis=1), y_pred_prob).get_pr_curve_data()
    axes[1, 0].plot(recall, precision, color='#3498db', lw=2, label=f'AUC-PR = {auc_pr:.4f}')
    axes[1, 0].set_xlim([0, 1])
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve (AUC-PR)', fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Key Metrics Table
    key_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR',
                   'Sensitivity', 'Specificity', 'Matthews Correlation Coefficient']
    values = [metrics_dict.get(m, 0) for m in key_metrics]
    
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table_data = [[m, f'{v:.4f}'] for m, v in zip(key_metrics, values)]
    table = axes[1, 1].table(cellText=table_data, colLabels=['Metric', 'Value'],
                             cellLoc='center', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternating rows
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_metrics.png'), dpi=300, bbox_inches='tight')
    print(f" Comprehensive metrics saved: {os.path.join(save_dir, 'comprehensive_metrics.png')}")
    plt.show()


# ============================================================================
#  MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_model(X_test, y_test, model=None, save_dir='./results'):
    """
    Complete model evaluation pipeline.
    
    Args:
        X_test: Test images (N, 224, 224, 3)
        y_test: True labels (N,)
        model: Loaded Keras model (if None, loads from MODEL_PATH)
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("   🩺 DIABETIC RETINOPATHY MODEL EVALUATION")
    print("="*70)
    
    # Load model if not provided
    if model is None:
        model = load_model_safe()
        if model is None:
            return
    
    # Generate predictions
    print("\n Generating predictions...")
    y_pred, y_pred_prob = get_predictions(model, X_test)
    if y_pred is None:
        return
    
    # Calculate metrics
    print("📈 Calculating metrics...")
    metrics_calculator = ModelMetrics(y_test, y_pred, y_pred_prob)
    metrics = metrics_calculator.calculate_all_metrics()
    
    # Print detailed results
    print("\n" + "="*70)
    print("   TEST SET RESULTS")
    print("="*70)
    
    # Basic metrics
    print("\n📌 BASIC METRICS:")
    print(f"  Accuracy              : {metrics['Accuracy']:.4f}")
    print(f"  Precision             : {metrics['Precision']:.4f}")
    print(f"  Recall (Sensitivity)  : {metrics['Recall']:.4f}")
    print(f"  F1-Score              : {metrics['F1-Score']:.4f}")
    print(f"  Balanced Accuracy     : {metrics['Balanced Accuracy']:.4f}")
    
    # AUC metrics
    print("\n AUC METRICS:")
    print(f"  AUC-ROC               : {metrics['AUC-ROC']:.4f}")
    print(f"  AUC-PR                : {metrics['AUC-PR']:.4f}")
    
    # Per-class metrics
    print("\n🎯 PER-CLASS METRICS (Class 1: No_DR):")
    print(f"  Sensitivity           : {metrics['Sensitivity']:.4f}")
    print(f"  Specificity           : {metrics['Specificity']:.4f}")
    print(f"  Positive Predictive Value (PPV) : {metrics['Positive Predictive Value (PPV)']:.4f}")
    print(f"  Negative Predictive Value (NPV) : {metrics['Negative Predictive Value (NPV)']:.4f}")
    print(f"  Youden's Index        : {metrics['Youden\'s Index']:.4f}")
    
    # Error rates
    print("\n  ERROR RATES:")
    print(f"  False Positive Rate   : {metrics['False Positive Rate']:.4f}")
    print(f"  False Negative Rate   : {metrics['False Negative Rate']:.4f}")
    
    # Confusion matrix
    print("\n🔢 CONFUSION MATRIX BREAKDOWN:")
    tn, fp, fn, tp = metrics['TP'], metrics['FP'], metrics['FN'], metrics['TP']
    print(f"  True Positives (TP)   : {metrics['TP']}")
    print(f"  True Negatives (TN)   : {metrics['TN']}")
    print(f"  False Positives (FP)  : {metrics['FP']}")
    print(f"  False Negatives (FN)  : {metrics['FN']}")
    
    # Correlation metrics
    print("\n📐 CORRELATION METRICS:")
    print(f"  Matthews Correlation Coefficient : {metrics['Matthews Correlation Coefficient']:.4f}")
    print(f"  Cohen's Kappa         : {metrics['Cohen\'s Kappa']:.4f}")
    
    # Classification report
    print("\n📋 DETAILED CLASSIFICATION REPORT:")
    print(metrics_calculator.get_classification_report())
    
    # Visualizations
    print("\n Generating visualizations...")
    cm = metrics_calculator.get_confusion_matrix()
    plot_confusion_matrix(cm, save_path=os.path.join(save_dir, 'confusion_matrix.png'))
    
    fpr, tpr, _, auc_score = metrics_calculator.get_roc_curve_data()
    plot_roc_curve(fpr, tpr, auc_score, save_path=os.path.join(save_dir, 'roc_curve.png'))
    
    precision, recall, _, auc_pr = metrics_calculator.get_pr_curve_data()
    plot_pr_curve(precision, recall, auc_pr, save_path=os.path.join(save_dir, 'pr_curve.png'))
    
    plot_metrics_comparison(metrics, save_path=os.path.join(save_dir, 'metrics_comparison.png'))
    
    plot_all_metrics_comprehensive(metrics, cm, y_test, y_pred_prob, save_dir=save_dir)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    metrics_df.to_csv(os.path.join(save_dir, 'metrics_summary.csv'), index=False)
    print(f"\n[SAVE] Metrics saved to: {os.path.join(save_dir, 'metrics_summary.csv')}")
    
    print("\n" + "="*70)
    print("   [SUCCESS] EVALUATION COMPLETE!")
    print("="*70)
    
    return metrics, y_pred, y_pred_prob


# ============================================================================
#  MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n[START] Diabetic Retinopathy Model - Comprehensive Test Suite")
    print("-" * 70)
    print(f"[MODEL] Path: {MODEL_PATH}")
    print(f"[DATA] Path: {TEST_DATA_DIR}")
    
    # Try to find test data in common locations if not configured
    if not os.path.exists(TEST_DATA_DIR):
        possible_paths = [
            TEST_DATA_DIR,  # Primary path from config
            './dataset/test',
            '../dataset/test',
            '../../dataset/test',
            './data/test',
            '../data/test',
            os.path.join(os.path.dirname(__file__), '..', 'dataset_extracted'),
            os.path.join(os.path.dirname(__file__), '..', 'dataset', 'test'),
        ]
        
        TEST_DATA_DIR = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'DR')) and os.path.exists(os.path.join(path, 'No_DR')):
                TEST_DATA_DIR = path
                print(f" Found test dataset at: {TEST_DATA_DIR}")
                break
    
    if TEST_DATA_DIR is None or not os.path.exists(TEST_DATA_DIR):
        print("\n  TEST DATASET NOT FOUND")
        print("\n📋 Directory Structure:")
        print(f"   Model location        : {MODEL_PATH}")
        print(f"   Script location       : {os.path.dirname(__file__)}")
        print(f"   Expected test data at : {os.path.join(os.path.dirname(__file__), '..', 'dataset_test')}")
        print("\n🔧 To prepare test data:")
        print("   1. Extract dataset.zip to get DR and No_DR folders")
        print("   2. Create a 'dataset_test' folder in project root")
        print("   3. Inside dataset_test, create DR/ and No_DR/ subdirectories")
        print("   4. Copy test images to respective folders")
        print("\nUsage:")
        print("  Option 1: Update TEST_DATA_DIR variable in this script")
        print("  Option 2: Prepare dataset as shown above")
        print("\nExpected directory structure:")
        print("  dataset_test/")
        print("  ├── DR/          (diabetic retinopathy images)")
        print("  │   ├── image1.jpg")
        print("  │   └── ...")
        print("  └── No_DR/       (healthy images)")
        print("      ├── image1.jpg")
        print("      └── ...")
        print("\n" + "="*70)
        print("Running demo evaluation with synthetic test data...")
        
        # Create synthetic test data for demo
        np.random.seed(42)
        X_test = np.random.rand(50, 224, 224, 3).astype(np.float32)
        y_test = np.array([0]*25 + [1]*25)
        
        model = load_model_safe()
        if model is not None:
            evaluate_model(X_test, y_test, model=model, save_dir='./results')
    else:
        print(f"\n Test dataset found at: {TEST_DATA_DIR}")
        X_test, y_test, image_names = load_test_dataset(TEST_DATA_DIR)
        
        if X_test is not None:
            model = load_model_safe()
            if model is not None:
                evaluate_model(X_test, y_test, model=model, save_dir='./results')
