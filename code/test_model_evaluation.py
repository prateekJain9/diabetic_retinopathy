"""
===============================================================================
UNIT TESTS FOR MODEL EVALUATION METRICS
Diabetic Retinopathy Detection Model
===============================================================================

Run tests with:
    pytest test_model_evaluation.py -v
    
Or run directly:
    python test_model_evaluation.py

===============================================================================
"""

import unittest
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score
)


class TestMetricsCalculation(unittest.TestCase):
    """Test suite for metrics calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic test data
        np.random.seed(42)
        
        # Perfect predictions
        self.y_true_perfect = np.array([0, 0, 0, 1, 1, 1])
        self.y_pred_perfect = np.array([0, 0, 0, 1, 1, 1])
        
        # Realistic predictions
        self.y_true_realistic = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        self.y_pred_realistic = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        
        # Poor predictions
        self.y_true_poor = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        self.y_pred_poor = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
        
        # Probabilities for AUC calculation
        self.y_prob_perfect = np.array([
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]
        ])
        
        self.y_prob_realistic = np.array([
            [0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8],
            [0.85, 0.15], [0.4, 0.6], [0.88, 0.12], [0.45, 0.55],
            [0.25, 0.75], [0.92, 0.08]
        ])

    def test_accuracy_perfect_predictions(self):
        """Test accuracy with perfect predictions."""
        acc = accuracy_score(self.y_true_perfect, self.y_pred_perfect)
        self.assertEqual(acc, 1.0, "Perfect predictions should have accuracy 1.0")

    def test_accuracy_poor_predictions(self):
        """Test accuracy with poor predictions."""
        acc = accuracy_score(self.y_true_poor, self.y_pred_poor)
        self.assertEqual(acc, 0.0, "All wrong predictions should have accuracy 0.0")

    def test_precision_calculation(self):
        """Test precision calculation."""
        precision = precision_score(self.y_true_realistic, self.y_pred_realistic, zero_division=0)
        self.assertGreaterEqual(precision, 0, "Precision should be >= 0")
        self.assertLessEqual(precision, 1, "Precision should be <= 1")

    def test_recall_calculation(self):
        """Test recall calculation."""
        recall = recall_score(self.y_true_realistic, self.y_pred_realistic, zero_division=0)
        self.assertGreaterEqual(recall, 0, "Recall should be >= 0")
        self.assertLessEqual(recall, 1, "Recall should be <= 1")

    def test_f1_score_calculation(self):
        """Test F1-Score calculation."""
        f1 = f1_score(self.y_true_realistic, self.y_pred_realistic, zero_division=0)
        self.assertGreaterEqual(f1, 0, "F1-Score should be >= 0")
        self.assertLessEqual(f1, 1, "F1-Score should be <= 1")

    def test_f1_perfect_is_one(self):
        """Test that perfect predictions give F1 = 1.0."""
        f1 = f1_score(self.y_true_perfect, self.y_pred_perfect)
        self.assertEqual(f1, 1.0, "Perfect predictions should have F1 = 1.0")

    def test_confusion_matrix_shape(self):
        """Test confusion matrix shape."""
        cm = confusion_matrix(self.y_true_realistic, self.y_pred_realistic)
        self.assertEqual(cm.shape, (2, 2), "Confusion matrix should be 2x2 for binary classification")

    def test_confusion_matrix_values_sum(self):
        """Test that confusion matrix elements sum to total predictions."""
        cm = confusion_matrix(self.y_true_realistic, self.y_pred_realistic)
        total = np.sum(cm)
        self.assertEqual(total, len(self.y_true_realistic),
                        "Confusion matrix elements should sum to number of samples")

    def test_balanced_accuracy(self):
        """Test balanced accuracy calculation."""
        bal_acc = balanced_accuracy_score(self.y_true_realistic, self.y_pred_realistic)
        self.assertGreaterEqual(bal_acc, 0, "Balanced accuracy should be >= 0")
        self.assertLessEqual(bal_acc, 1, "Balanced accuracy should be <= 1")

    def test_roc_auc_perfect(self):
        """Test AUC-ROC with perfect probabilities."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = self.y_prob_perfect[:, 1]  # Probability of class 1
        auc = roc_auc_score(y_true, y_prob)
        self.assertEqual(auc, 1.0, "Perfect probabilities should have AUC = 1.0")

    def test_roc_auc_range(self):
        """Test that AUC-ROC is in valid range."""
        y_true = self.y_true_realistic
        y_prob = self.y_prob_realistic[:, 1]
        auc = roc_auc_score(y_true, y_prob)
        self.assertGreaterEqual(auc, 0.5, "AUC-ROC should be >= 0.5 (better than random)")
        self.assertLessEqual(auc, 1.0, "AUC-ROC should be <= 1.0")

    def test_matthews_correlation_coefficient(self):
        """Test Matthews Correlation Coefficient calculation."""
        mcc = matthews_corrcoef(self.y_true_realistic, self.y_pred_realistic)
        self.assertGreaterEqual(mcc, -1, "MCC should be >= -1")
        self.assertLessEqual(mcc, 1, "MCC should be <= 1")

    def test_mcc_perfect_is_one(self):
        """Test that perfect predictions give MCC = 1.0."""
        mcc = matthews_corrcoef(self.y_true_perfect, self.y_pred_perfect)
        self.assertEqual(mcc, 1.0, "Perfect predictions should have MCC = 1.0")

    def test_cohen_kappa(self):
        """Test Cohen's Kappa calculation."""
        kappa = cohen_kappa_score(self.y_true_realistic, self.y_pred_realistic)
        self.assertGreaterEqual(kappa, -1, "Kappa should be >= -1")
        self.assertLessEqual(kappa, 1, "Kappa should be <= 1")

    def test_sensitivity_calculation(self):
        """Test sensitivity (recall for positive class) calculation."""
        from sklearn.metrics import recall_score
        sensitivity = recall_score(self.y_true_realistic, self.y_pred_realistic, pos_label=1)
        self.assertGreaterEqual(sensitivity, 0, "Sensitivity should be >= 0")
        self.assertLessEqual(sensitivity, 1, "Sensitivity should be <= 1")

    def test_specificity_calculation(self):
        """Test specificity (recall for negative class) calculation."""
        from sklearn.metrics import recall_score
        specificity = recall_score(self.y_true_realistic, self.y_pred_realistic, pos_label=0)
        self.assertGreaterEqual(specificity, 0, "Specificity should be >= 0")
        self.assertLessEqual(specificity, 1, "Specificity should be <= 1")


class TestMetricsRelationships(unittest.TestCase):
    """Test relationships between metrics."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1])
        self.y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1])

    def test_f1_between_precision_recall(self):
        """Test that F1 is between precision and recall."""
        precision = precision_score(self.y_true, self.y_pred)
        recall = recall_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred)
        
        if precision > 0 or recall > 0:
            self.assertLessEqual(f1, max(precision, recall),
                                "F1 should be <= max(precision, recall)")
            self.assertGreaterEqual(f1, min(precision, recall),
                                   "F1 should be >= min(precision, recall)")

    def test_confusion_matrix_relationship_with_metrics(self):
        """Test relationships between confusion matrix and metrics."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Test sensitivity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall = recall_score(self.y_true, self.y_pred, pos_label=1)
        self.assertAlmostEqual(sensitivity, recall, places=5,
                               msg="Sensitivity should equal recall for positive class")

    def test_accuracy_from_confusion_matrix(self):
        """Test that accuracy can be calculated from confusion matrix."""
        acc_direct = accuracy_score(self.y_true, self.y_pred)
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        acc_from_cm = (tp + tn) / (tp + tn + fp + fn)
        
        self.assertAlmostEqual(acc_direct, acc_from_cm, places=5,
                               msg="Accuracy from confusion matrix should match direct calculation")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def test_all_zeros_prediction(self):
        """Test when all predictions are class 0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        self.assertEqual(recall, 0, "Recall should be 0 when no positives predicted")

    def test_all_ones_prediction(self):
        """Test when all predictions are class 1."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        self.assertEqual(precision, 0.5, "Precision should be 0.5 (2 correct out of 4 predicted positive)")

    def test_balanced_dataset(self):
        """Test on balanced dataset."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1])
        
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        
        # On balanced dataset, accuracy and balanced accuracy should be similar
        self.assertAlmostEqual(acc, bal_acc, places=1,
                               msg="On balanced dataset, accuracy should be close to balanced accuracy")

    def test_imbalanced_dataset(self):
        """Test on imbalanced dataset."""
        # 9 negatives, 1 positive
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Predicts all negative
        
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        
        # Accuracy will be high (0.9) but balanced accuracy will be lower
        self.assertGreater(acc, bal_acc,
                           msg="On imbalanced dataset, balanced accuracy should be lower than accuracy")


class TestPerformanceInterpretation(unittest.TestCase):
    """Test interpretation of performance levels."""

    def test_excellent_model_metrics(self):
        """Test metrics of an excellent model."""
        # Excellent predictions (99% correct)
        y_true = np.array([0]*100 + [1]*100)
        # 2 errors out of 200
        y_pred = np.array([0]*99 + [1] + [1]*99 + [0])
        
        acc = accuracy_score(y_true, y_pred)
        self.assertGreater(acc, 0.98, "Excellent model should have accuracy > 0.98")

    def test_good_model_metrics(self):
        """Test metrics of a good model."""
        # Good predictions (85% correct)
        y_true = np.array([0]*100 + [1]*100)
        errors = 30  # 15% error rate
        y_pred = np.array([0]*85 + [1]*15 + [1]*85 + [0]*15)
        
        acc = accuracy_score(y_true, y_pred)
        self.assertGreater(acc, 0.8, "Good model should have accuracy > 0.8")
        self.assertLess(acc, 0.95, "Good model should have accuracy < 0.95")


def run_comprehensive_test_suite():
    """Run all tests and print summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsRelationships))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceInterpretation))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print('\n' + '='*70)
    print('   TEST SUMMARY')
    print('='*70)
    print(f'Tests run    : {result.testsRun}')
    print(f'Passed       : {result.testsRun - len(result.failures) - len(result.errors)}')
    print(f'Failed       : {len(result.failures)}')
    print(f'Errors       : {len(result.errors)}')
    print('='*70)
    
    return result


if __name__ == '__main__':
    result = run_comprehensive_test_suite()
    exit(0 if result.wasSuccessful() else 1)
