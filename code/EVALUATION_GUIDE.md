# 🩺 Diabetic Retinopathy Model - Test Metrics & Evaluation Suite

## Overview

This suite provides comprehensive test metrics and evaluation tools for the Diabetic Retinopathy (DR) detection model trained using Federated Learning (FedAvg) and MobileNetV2.

## 📊 Metrics Included

### Classification Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives among predicted positives
- **Recall (Sensitivity)**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Balanced Accuracy**: Average of sensitivity and specificity

### AUC Metrics
- **AUC-ROC**: Area Under Receiver Operating Characteristic Curve
- **AUC-PR**: Area Under Precision-Recall Curve

### Confusion Matrix Components
- **True Positives (TP)**: DR cases correctly identified
- **True Negatives (TN)**: Healthy cases correctly identified
- **False Positives (FP)**: Healthy cases wrongly marked as DR
- **False Negatives (FN)**: DR cases missed

### Per-Class Metrics
- **Sensitivity**: True Positive Rate (TP / (TP + FN))
- **Specificity**: True Negative Rate (TN / (TN + FP))
- **Positive Predictive Value (PPV)**: Precision (TP / (TP + FP))
- **Negative Predictive Value (NPV)**: TN / (TN + FN)
- **Youden's Index**: Sensitivity + Specificity - 1

### Error Rates
- **False Positive Rate (FPR)**: False alarm rate (FP / (FP + TN))
- **False Negative Rate (FNR)**: Missed detection rate (FN / (FN + TP))

### Correlation Metrics
- **Matthews Correlation Coefficient (MCC)**: Overall correlation between predictions and labels
- **Cohen's Kappa**: Agreement measure accounting for chance

---

## 🚀 Quick Start

### 1. Jupyter Notebook (Recommended for Interactive Analysis)

```bash
# Navigate to the code directory
cd code

# Open the notebook
jupyter notebook Model_Evaluation_Metrics.ipynb
```

**Features:**
- Interactive cell-by-cell execution
- Built-in visualizations (Confusion Matrix, ROC, PR curves)
- Live metrics calculation
- Confidence distribution analysis
- Comprehensive dashboard

**Usage:**
1. Update `MODEL_PATH` to your model location
2. Update `TEST_DATA_DIR` to your test dataset
3. Run cells sequentially

Expected test data structure:
```
test_data/
├── DR/          # Diabetic Retinopathy images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── No_DR/       # Healthy images
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

---

### 2. Python Script (For Batch Processing)

```bash
python test_model_metrics.py
```

**Features:**
- Complete end-to-end evaluation
- Generates all visualizations automatically
- Saves metrics to CSV
- Detailed terminal output
- Configurable test data path

**Usage:**
```python
# Basic usage
python test_model_metrics.py

# Or modify MODEL_PATH and TEST_DATA_DIR in the script
MODEL_PATH = './dr_fedavg_model.h5'
TEST_DATA_DIR = './dataset/test'
```

**Output Files:**
- `confusion_matrix.png` - Confusion matrix heatmap
- `roc_curve.png` - ROC curve with AUC score
- `pr_curve.png` - Precision-Recall curve with AUC-PR
- `metrics_comparison.png` - Bar chart of key metrics
- `comprehensive_metrics.png` - Full dashboard
- `metrics_summary.csv` - All metrics in CSV format

---

### 3. Unit Tests (For Validation)

```bash
# Run all tests
python test_model_evaluation.py

# Or use pytest for more detailed output
pytest test_model_evaluation.py -v
```

**Test Categories:**
- `TestMetricsCalculation`: Validates metric calculations
- `TestMetricsRelationships`: Tests mathematical relationships between metrics
- `TestEdgeCases`: Tests boundary conditions
- `TestPerformanceInterpretation`: Tests performance level interpretation

---

## 📈 Interpreting Results

### Performance Levels

| Metric | Excellent | Very Good | Good | Fair | Poor |
|--------|-----------|-----------|------|------|------|
| Accuracy | ≥ 0.95 | ≥ 0.90 | ≥ 0.85 | ≥ 0.75 | < 0.75 |
| AUC-ROC | ≥ 0.95 | ≥ 0.90 | ≥ 0.80 | ≥ 0.70 | < 0.70 |
| F1-Score | ≥ 0.95 | ≥ 0.90 | ≥ 0.85 | ≥ 0.75 | < 0.75 |

### Confusion Matrix Interpretation

```
                    Predicted
              DR (Positive)    Healthy (Negative)
Actual DR     TP ✓             FN ✗ (missed cases)
      Healthy FP ✗ (false)     TN ✓
```

**For Medical Applications:**
- Minimize **False Negatives (FN)**: Missing DR cases is dangerous
- Minimize **False Positives (FP)**: False alarms cause unnecessary treatment
- **Sensitivity** (Recall): How many actual DR cases are caught (higher is better)
- **Specificity**: How many actual healthy cases are identified (higher is better)

### Example Results Interpretation

```
Accuracy:        0.92    →  92% of predictions are correct
Sensitivity:     0.95    →  95% of DR cases are detected ✅ Good for screening
Specificity:     0.89    →  89% of healthy cases correctly identified ✅
AUC-ROC:         0.97    →  Excellent discrimination ability
F1-Score:        0.91    →  Good balance between precision and recall
```

---

## 🔧 Customization

### Modify Test Metrics Script

```python
# In test_model_metrics.py, customize:

# 1. Model path
MODEL_PATH = '/path/to/your/model.h5'

# 2. Test data directory
TEST_DATA_DIR = '/path/to/test/data'

# 3. Results save directory
save_dir = './results'

# 4. Image size (must match model input)
IMG_SIZE = (224, 224)

# 5. Class names
CLASS_NAMES = ['DR (Positive)', 'No_DR (Negative)']
```

### Modify Notebook

Update these cells:
```python
# Cell 3: Configuration
MODEL_PATH = 'your_model_path.h5'
TEST_DATA_DIR = 'path/to/test/data'
```

---

## 📊 Output Examples

### 1. Confusion Matrix
Shows:
- True Positives (correctly identified DR)
- True Negatives (correctly identified healthy)
- False Positives (false alarms)
- False Negatives (missed DR cases)

### 2. ROC Curve
- X-axis: False Positive Rate (1 - Specificity)
- Y-axis: True Positive Rate (Sensitivity)
- AUC = Area Under Curve (0.5 = random, 1.0 = perfect)

### 3. Precision-Recall Curve
- X-axis: Recall (Sensitivity)
- Y-axis: Precision
- Useful for imbalanced datasets

### 4. Metrics Comparison
- Bar chart of key metrics
- Color-coded (green ≥ 0.8, yellow ≥ 0.6, red < 0.6)

### 5. Confidence Distribution
- Histogram of prediction probabilities
- Boxplot by true class
- Shows model confidence levels

---

## 🎯 Medical Significance

### For DR Screening:

| Metric | Why It Matters |
|--------|----------------|
| **Sensitivity** | High sensitivity = catching most DR cases (minimize false negatives) |
| **Specificity** | High specificity = avoiding unnecessary referrals (minimize false positives) |
| **NPV** | If test is negative, how sure are we patient is healthy? |
| **PPV** | If test is positive, how sure are we patient has DR? |

### Target Benchmarks:
- **Sensitivity ≥ 95%**: Catch most DR cases (critical for screening)
- **Specificity ≥ 90%**: Reduce unnecessary referrals
- **AUC-ROC ≥ 0.95**: Excellent discrimination ability

---

## 📁 File Structure

```
code/
├── test_model_metrics.py          # Standalone Python script
├── Model_Evaluation_Metrics.ipynb  # Interactive Jupyter notebook
├── test_model_evaluation.py        # Unit tests
├── dr_fedavg_model.h5             # Trained model
└── results/                        # Output directory
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── pr_curve.png
    ├── metrics_comparison.png
    ├── comprehensive_metrics.png
    └── metrics_summary.csv
```

---

## 🐛 Troubleshooting

### Issue: Model not found
```
Solution: Update MODEL_PATH to correct location
```

### Issue: Test data not found
```
Solution: 
1. Update TEST_DATA_DIR path
2. Ensure directory structure is:
   test_data/
   ├── DR/
   └── No_DR/
```

### Issue: Import errors
```
Solution: Install required packages
pip install tensorflow scikit-learn matplotlib seaborn pandas pillow numpy
```

### Issue: Out of memory
```
Solution: Reduce batch size or test set size in configuration
```

---

## 📚 References

### Metric Definitions:
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall/Sensitivity**: TP / (TP + FN)
- **Specificity**: TN / (TN + FP)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve

### Medical Terminology:
- **True Positive (TP)**: Correctly identified as having DR
- **True Negative (TN)**: Correctly identified as healthy
- **False Positive (FP)**: Healthy patient identified as having DR
- **False Negative (FN)**: DR patient identified as healthy (most critical)

---

## 💡 Best Practices

1. **Always validate on held-out test set**
2. **Monitor both Sensitivity and Specificity** for medical applications
3. **Use AUC-ROC for imbalanced datasets**
4. **Check confusion matrix for class-specific errors**
5. **Document baseline metrics for comparison**
6. **Test on diverse, representative data**
7. **Consider clinical significance, not just statistical significance**

---

## 🚀 Next Steps

After evaluation:

1. **If performance is good (AUC-ROC > 0.95)**:
   - Deploy model to production
   - Implement continuous monitoring
   - Set up performance tracking

2. **If performance needs improvement**:
   - Analyze confusion matrix for patterns
   - Collect more training data
   - Try different model architectures
   - Adjust class weights for imbalance
   - Tune decision threshold

3. **For production deployment**:
   - Implement real-time monitoring
   - Set up alerts for performance degradation
   - Create audit logs
   - Plan retraining pipeline

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review metric interpretations
3. Verify file paths and data format
4. Check TensorFlow/scikit-learn documentation

---

**Created**: 2026  
**Model**: Diabetic Retinopathy Detection (FedAvg + MobileNetV2)  
**Purpose**: Comprehensive evaluation and testing suite
