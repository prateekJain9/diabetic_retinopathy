# 📊 DR Model Metrics - Quick Reference Guide

## One-Page Metrics Summary

### 📌 Core Metrics (Most Important)

| Metric | Formula | Range | What It Means |
|--------|---------|-------|---------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 0-1 | % correct predictions |
| **Precision** | TP/(TP+FP) | 0-1 | When we say DR, how right? |
| **Recall** | TP/(TP+FN) | 0-1 | Of all DR cases, how many found? |
| **F1-Score** | 2×(P×R)/(P+R) | 0-1 | Balance of Precision & Recall |
| **AUC-ROC** | Area under curve | 0-1 | Overall discriminative ability |

---

## 🎯 For Medical Screening (Most Relevant)

```
Goal: Catch ALL DR cases while minimizing false alarms

Key Metrics:
  ✅ Sensitivity (Recall) ≥ 0.95  → Catch 95%+ of DR cases
  ✅ Specificity ≥ 0.90            → Avoid 90%+ of false alarms
  ✅ NPV ≥ 0.98                    → If negative, 98% sure they're healthy
  ⚠️  FNR < 0.05                   → Miss < 5% of DR cases
```

---

## 📋 Confusion Matrix Explained

```
                      Predicted Class
                    DR (1)      Healthy (0)
Actual  DR (1)        TP            FN ✗
        Healthy (0)   FP ✗          TN

Key Point: FN is most critical for medical screening!
           (Missing a DR patient is dangerous)
```

---

## 📊 Metric Relationships

### Precision vs Recall Trade-off
```
HIGH PRECISION (Few false alarms):
  → Good when: False alarms are expensive
  → E.g., Don't want to scare healthy patients

HIGH RECALL (Catch all cases):
  → Good when: Missing cases is expensive
  → E.g., DR screening (don't miss any DR)
  
F1-SCORE balances both
```

### AUC-ROC vs Accuracy
```
When to use AUC-ROC:
  ✓ Imbalanced datasets
  ✓ Need full threshold analysis
  ✓ Medical applications
  
When to use Accuracy:
  ✓ Balanced datasets only
  ✓ Simple performance metric
```

---

## 🎓 Interpretation Guide

### Performance Levels

```
Excellent   ⭐⭐⭐  AUC-ROC ≥ 0.95
Very Good   ⭐⭐    AUC-ROC ≥ 0.90
Good        ⭐      AUC-ROC ≥ 0.80
Fair        ⚠️     AUC-ROC ≥ 0.70
Poor        ❌      AUC-ROC < 0.70
```

### What Each Metric Tells You

```
📊 Accuracy (Overall)
   - Simple overall performance
   - Misleading on imbalanced data
   
🎯 Precision (Avoid False Alarms)
   - Of predicted positives, how many correct?
   - Important: Don't want to scare healthy people
   
🔍 Recall/Sensitivity (Catch All Cases)
   - Of actual positives, how many found?
   - Critical: Don't miss DR patients!
   
⚖️  F1-Score (Balance)
   - Harmonic mean of Precision & Recall
   - Use when both matter equally
   
📈 AUC-ROC (Discrimination)
   - Full picture of performance
   - 0.5 = random, 1.0 = perfect
   - Best for medical applications
   
🔄 AUC-PR (Imbalanced Data)
   - Precision-Recall curve
   - Better for imbalanced datasets
   
🌍 Sensitivity/Specificity (Medical Context)
   - Sensitivity = Recall (True Positive Rate)
   - Specificity = True Negative Rate
   - Need BOTH high for good screening test
```

---

## ✅ Ideal Values for DR Screening

| Metric | Ideal | Acceptable | Poor |
|--------|-------|-----------|------|
| Accuracy | > 0.95 | > 0.85 | < 0.75 |
| Sensitivity | > 0.95 | > 0.90 | < 0.80 |
| Specificity | > 0.90 | > 0.85 | < 0.75 |
| Precision | > 0.90 | > 0.80 | < 0.70 |
| F1-Score | > 0.92 | > 0.85 | < 0.75 |
| AUC-ROC | > 0.95 | > 0.90 | < 0.80 |
| AUC-PR | > 0.95 | > 0.90 | < 0.80 |

---

## 🔍 Interpreting Confusion Matrix

```
        Predicted
        DR  |  Healthy
    DR  TP  |  FN      ← Missed cases (critical!)
    Healthy FP |  TN   ← False alarms

TP = ✅ Correct DR detection
TN = ✅ Correct healthy detection  
FP = ❌ False alarm (said DR but healthy)
FN = ❌ Missed case (said healthy but has DR)

For DR screening:
  Minimize FN at all costs!
  Moderate FP acceptable (just needs follow-up)
```

---

## 🚨 Red Flags to Watch

```
⚠️ High FP rate (> 15%)
   → Many false alarms
   → Patients unnecessarily worried
   → Unnecessary follow-up tests
   
🚨 High FN rate (> 5%)
   → Missed DR cases
   → Patient doesn't get treatment
   → Disease progresses unchecked
   
⚠️ Sensitivity < 85%
   → Too many missed DR cases
   → Not suitable for screening
   
⚠️ AUC-ROC < 0.80
   → Poor discrimination
   → Model can't reliably separate classes
```

---

## 📌 Special Metrics for Medical Use

| Metric | Formula | Medical Meaning |
|--------|---------|-----------------|
| **Sensitivity** | TP/(TP+FN) | If person has DR, prob. test positive |
| **Specificity** | TN/(TN+FP) | If person healthy, prob. test negative |
| **NPV** | TN/(TN+FN) | If test negative, prob. actually healthy |
| **PPV** | TP/(TP+FP) | If test positive, prob. actually has DR |
| **Youden's J** | Sen + Spe - 1 | Optimality index |

---

## 🔧 Quick Troubleshooting

### "Accuracy is high but Recall is low"
```
Problem: High FN (missing DR cases)
→ Sensitivity too low
→ Need to lower decision threshold
→ Sacrifice some precision for recall
```

### "High Precision but Low Recall"
```
Problem: Too conservative (high threshold)
→ Few false positives but many false negatives
→ For screening, this is BAD
→ Lower threshold to catch more DR cases
```

### "AUC-ROC low but Accuracy high"
```
Problem: Likely imbalanced dataset
→ Accuracy can be misleading
→ Trust AUC-ROC more
→ Balance the classes or use AUC-PR
```

### "Metrics differ between validation and test"
```
Problem: Test set might be different
→ Check test set distribution
→ Ensure similar class balance
→ Look for distribution shift
```

---

## 💡 Decision-Making Guide

### Choose Metric Based on Task

```
IF imbalanced data
   → Use AUC-ROC or AUC-PR, NOT accuracy

IF medical screening
   → Prioritize Sensitivity ≥ 0.95
   → Acceptable Specificity ≥ 0.85

IF balanced dataset
   → Accuracy and F1-Score are good

IF want overall goodness
   → Use AUC-ROC (0.5 = random, 1.0 = perfect)

IF need best threshold
   → Analyze ROC curve
   → Choose point with best Sensitivity-Specificity
```

---

## 📊 Interpretation Examples

### ✅ EXCELLENT MODEL
```
Accuracy:    0.96  ✓
Precision:   0.94  ✓
Recall:      0.95  ✓
F1-Score:    0.945 ✓
Sensitivity: 0.95  ✓
Specificity: 0.96  ✓
AUC-ROC:     0.98  ✓

→ Ready for deployment
```

### ⚠️ NEEDS IMPROVEMENT
```
Accuracy:    0.92  ✓
Precision:   0.88  ✓
Recall:      0.75  ✗ TOO LOW!
Sensitivity: 0.75  ✗ MISSING DR CASES
Specificity: 0.99  (too strict)
AUC-ROC:     0.82  ⚠️  Could be better

→ Lower decision threshold
→ Allow more FP to catch DR cases
```

### ❌ POOR MODEL
```
Accuracy:    0.75  ✗
Precision:   0.65  ✗
Recall:      0.60  ✗
F1-Score:    0.62  ✗
Sensitivity: 0.60  ✗
Specificity: 0.90  (but won't help with low recall)
AUC-ROC:     0.68  ✗

→ Needs significant improvement
→ Retrain with more data
→ Try different architecture
```

---

## 🎯 Key Takeaways

1. **For DR Screening: Prioritize Sensitivity (Recall)**
   - Missing DR is worse than false alarm
   - Target Sensitivity ≥ 95%

2. **Use AUC-ROC for Medical Applications**
   - Better than accuracy for imbalanced data
   - Threshold-independent

3. **Interpret Confusion Matrix First**
   - Understand where model fails
   - Identify patterns in errors

4. **Balance Sensitivity and Specificity**
   - Need both for good screening test
   - Plot ROC curve to find optimal threshold

5. **Monitor All Key Metrics**
   - Don't rely on single metric
   - Look at full picture

---

## 📚 Formula Reference

```python
# Basic Metrics
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)

# Medical Metrics
Sensitivity = TP / (TP + FN)      # = Recall
Specificity = TN / (TN + FP)
PPV = TP / (TP + FP)             # = Precision
NPV = TN / (TN + FN)

# Error Rates
FPR = FP / (FP + TN)             # False Positive Rate
FNR = FN / (FN + TP)             # False Negative Rate

# Correlation
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

---

**Last Updated**: 2026  
**Model**: DR Detection (FedAvg + MobileNetV2)  
**Purpose**: Quick reference for metrics interpretation
