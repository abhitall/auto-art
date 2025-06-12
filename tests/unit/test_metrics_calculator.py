import pytest
import numpy as np
from auto_art.core.evaluation.metrics_calculator import MetricsCalculator # Assuming this is the correct path

@pytest.fixture
def metrics_calculator():
    return MetricsCalculator()

# Test data for metrics calculation
@pytest.fixture
def sample_labels_and_predictions():
    # y_true: True labels (e.g., class indices)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1]) # 10 samples

    # y_pred_benign: Predictions on benign data (e.g., class indices)
    # 80% accuracy: 8 correct, 2 incorrect
    y_pred_benign = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0]) # Last two are wrong

    # y_pred_adversarial: Predictions on adversarial data
    # Assume attack is successful if benign prediction was correct AND adversarial prediction is different.
    # Benign correct: [0,1,0,1,0,1,0,0] (first 8 samples)
    # Adversarial predictions for these 8:
    # - Samples 0, 2, 4, 6 (true 0) -> let's say attack changes 2 of them to 1
    # - Samples 1, 3, 5, 7 (true 1) -> let's say attack changes 2 of them to 0
    # Adversarial for all 10 samples:
    # Original: [0, 1, 0, 1, 0, 1, 0, 0, 1, 1] (y_true)
    # Benign:   [0, 1, 0, 1, 0, 1, 0, 0, 0, 0] (y_pred_benign)
    # Adv:      [1, 0, 1, 0, 0, 1, 0, 0, 1, 1]
    #   - For sample 0 (true 0, benign 0): adv 1 (success)
    #   - For sample 1 (true 1, benign 1): adv 0 (success)
    #   - For sample 2 (true 0, benign 0): adv 1 (success)
    #   - For sample 3 (true 1, benign 1): adv 0 (success)
    #   - For sample 4 (true 0, benign 0): adv 0 (fail, still correct class but could be diff if targeted)
    #   - For sample 5 (true 1, benign 1): adv 1 (fail, still correct class)
    #   - For sample 6 (true 0, benign 0): adv 0 (fail)
    #   - For sample 7 (true 1, benign 0): adv 0 (benign was already wrong, ASR not applicable for this definition)
    #   - For sample 8 (true 0, benign 0): adv 1 (success, assuming benign was 0) -> y_pred_benign[8] is 0, y_true[8] is 1. Benign was wrong.
    #   - For sample 9 (true 1, benign 0): adv 1 (success, assuming benign was 0) -> y_pred_benign[9] is 0, y_true[9] is 1. Benign was wrong.

    # Let's simplify y_pred_adversarial for clarity of ASR:
    # Focus on cases where benign prediction was correct.
    # y_true:        [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
    # y_pred_benign: [0, 1, 0, 1, 0, 1, 0, 0, 0, 0] (Correct for first 8 if y_true for first 8 is [0,1,0,1,0,1,0,0])
    # Let's use y_true_subset for the first 8 where benign was correct.
    y_true_first_8 = np.array([0, 1, 0, 1, 0, 1, 0, 0])
    y_pred_benign_first_8 = np.array([0, 1, 0, 1, 0, 1, 0, 0]) # All correct

    # Adversarial predictions for these 8 samples:
    # Let's say 4 of them are now misclassified due to attack
    y_pred_adv_first_8 = np.array([1, 0, 1, 0, 0, 1, 0, 0]) # 4 changed, 4 same
    # Attack success for these 4: 4 / 8 = 50%

    # For overall accuracy on adversarial data (all 10 samples):
    # y_true:             [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
    # y_pred_adversarial: [1, 0, 1, 0, 0, 1, 0, 0, 0, 1] # 4 correct, 6 incorrect => 40% acc
    y_pred_adversarial_all_10 = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1])


    return {
        "y_true": y_true,
        "y_pred_benign": y_pred_benign,
        "y_pred_adversarial": y_pred_adversarial_all_10,
        "y_true_first_8": y_true_first_8, # Subset where benign was correct
        "y_pred_benign_first_8": y_pred_benign_first_8, # Benign predictions for this subset (all correct)
        "y_pred_adv_first_8": y_pred_adv_first_8 # Adversarial predictions for this subset
    }

# Test accuracy calculation
def test_calculate_accuracy(metrics_calculator, sample_labels_and_predictions):
    y_true = sample_labels_and_predictions["y_true"]
    y_pred_benign = sample_labels_and_predictions["y_pred_benign"]
    y_pred_adversarial = sample_labels_and_predictions["y_pred_adversarial"]

    # Accuracy on benign data: 8/10 = 0.8
    accuracy_benign = metrics_calculator.calculate_accuracy(y_true, y_pred_benign)
    assert accuracy_benign == pytest.approx(0.8)

    # Accuracy on adversarial data: 4/10 = 0.4
    accuracy_adversarial = metrics_calculator.calculate_accuracy(y_true, y_pred_adversarial)
    assert accuracy_adversarial == pytest.approx(0.4)

# Test Attack Success Rate (ASR)
# ASR is often defined as:
# % of samples correctly classified by benign model that are misclassified by attacked model.
def test_calculate_attack_success_rate(metrics_calculator, sample_labels_and_predictions):
    y_true = sample_labels_and_predictions["y_true_first_8"] # True labels for subset
    y_pred_benign = sample_labels_and_predictions["y_pred_benign_first_8"] # Benign preds (all correct)
    y_pred_adversarial = sample_labels_and_predictions["y_pred_adv_first_8"] # Adv preds for this subset

    # In y_pred_adv_first_8: [1,0,1,0,0,1,0,0]
    # Compared to y_true:      [0,1,0,1,0,1,0,0]
    # Misclassified by adv: 4 out of 8
    # Benign accuracy on this subset is 100%.
    # So, ASR = 4/8 = 0.5

    # The MetricsCalculator.calculate_attack_success_rate might take all three arrays
    # or expect pre-filtered data. Let's assume it takes all three for the general case.
    # y_true_all = sample_labels_and_predictions["y_true"]
    # y_pred_benign_all = sample_labels_and_predictions["y_pred_benign"]
    # y_pred_adv_all = sample_labels_and_predictions["y_pred_adversarial"]
    # asr = metrics_calculator.calculate_attack_success_rate(y_true_all, y_pred_benign_all, y_pred_adv_all)

    # Let's test with the subset directly if the calculator supports it or implies this filtering.
    # If the calculator's method is `calculate_attack_success_rate(y_true, y_pred_benign, y_pred_adversarial)`
    # it needs to internally filter for samples correctly classified by benign model.

    # Based on the definition:
    # 1. Identify samples where y_true == y_pred_benign
    # 2. For these samples, check if y_true != y_pred_adversarial
    # ASR = (count from step 2) / (count from step 1)

    # Using all data:
    # y_true:             [0, 1, 0, 1, 0, 1, 0, 0, 1, 1] (10 samples)
    # y_pred_benign:      [0, 1, 0, 1, 0, 1, 0, 0, 0, 0] (Benign correct for first 8) -> 8 samples
    # y_pred_adversarial: [1, 0, 1, 0, 0, 1, 0, 0, 0, 1]

    # Samples correctly classified by benign model (indices): 0, 1, 2, 3, 4, 5, 6, 7 (8 samples)
    #   y_true[0-7]:             [0, 1, 0, 1, 0, 1, 0, 0]
    #   y_pred_adversarial[0-7]: [1, 0, 1, 0, 0, 1, 0, 0]
    #   Number misclassified by adv among these: 4 (indices 0,1,2,3)
    # ASR = 4 / 8 = 0.5

    asr = metrics_calculator.calculate_attack_success_rate(
        sample_labels_and_predictions["y_true"],
        sample_labels_and_predictions["y_pred_benign"],
        sample_labels_and_predictions["y_pred_adversarial"]
    )
    assert asr == pytest.approx(0.5)


# Test for targeted ASR (if supported)
# Targeted ASR: % of samples (correctly classified by benign model AND not already target class)
# that are misclassified as the target class by the attacked model.
def test_calculate_targeted_attack_success_rate(metrics_calculator):
    y_true           = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]) # 3 classes
    y_pred_benign    = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]) # All correct benign
    # Attack targets class 1
    target_class = 1
    # Adversarial predictions:
    # True 0 -> Adv wants 1
    # True 2 -> Adv wants 1
    y_pred_adversarial = np.array([1, 1, 0, 1, 1, 1, 1, 1, 2])
    # Sample 0 (true 0): Benign 0, Adv 1 (Success for target 1)
    # Sample 1 (true 0): Benign 0, Adv 1 (Success for target 1)
    # Sample 2 (true 0): Benign 0, Adv 0 (Fail)
    # Sample 3 (true 1): Benign 1 (already target, usually excluded from T-ASR numerator/denominator)
    # Sample 6 (true 2): Benign 2, Adv 1 (Success for target 1)
    # Sample 7 (true 2): Benign 2, Adv 1 (Success for target 1)
    # Sample 8 (true 2): Benign 2, Adv 2 (Fail)

    # Eligible for T-ASR: Samples not already of target_class (indices 0,1,2, 6,7,8) -> 6 samples
    #   y_true[eligible]:           [0,0,0, 2,2,2]
    #   y_pred_adv[eligible]:       [1,1,0, 1,1,2]
    #   Successes (became target 1): 4 (samples 0,1,6,7)
    # T-ASR = 4 / 6 = 0.666...

    if hasattr(metrics_calculator, 'calculate_targeted_attack_success_rate'):
        tasr = metrics_calculator.calculate_targeted_attack_success_rate(
            y_true, y_pred_benign, y_pred_adversarial, target_class
        )
        assert tasr == pytest.approx(4/6)
    else:
        pytest.skip("Targeted ASR method not implemented in MetricsCalculator")

# Test compute_all_metrics method
def test_compute_all_metrics(metrics_calculator, sample_labels_and_predictions):
    y_true = sample_labels_and_predictions["y_true"]
    y_pred_benign = sample_labels_and_predictions["y_pred_benign"]
    y_pred_adversarial = sample_labels_and_predictions["y_pred_adversarial"]

    # Mock the individual calculation methods if compute_all_metrics calls them
    # Or, if compute_all_metrics does calculations directly, test its output based on inputs.

    # For this test, let's assume compute_all_metrics calls the others that we've already tested
    # (or performs similar logic).
    # We can patch `self.calculate_accuracy` and `self.calculate_attack_success_rate`
    # if MetricsCalculator instance calls its own methods.

    with patch.object(MetricsCalculator, 'calculate_accuracy', side_effect=[0.8, 0.4]) as mock_acc, \
         patch.object(MetricsCalculator, 'calculate_attack_success_rate', return_value=0.5) as mock_asr:

        # Need to instantiate calculator *inside* with block if patching its methods directly on class
        # OR patch the instance's methods. Let's patch instance methods for simplicity if possible.
        # However, `metrics_calculator` fixture is already an instance.
        # Patching on the class and then calling on instance works if methods are not static/class methods.

        # Re-instantiate for patching to take effect cleanly if methods are called as self.method()
        calc_for_test = MetricsCalculator()

        all_metrics = calc_for_test.compute_all_metrics(y_true, y_pred_benign, y_pred_adversarial)

        assert "accuracy_benign" in all_metrics
        assert "accuracy_adversarial" in all_metrics
        assert "attack_success_rate" in all_metrics

        assert all_metrics["accuracy_benign"] == pytest.approx(0.8)
        assert all_metrics["accuracy_adversarial"] == pytest.approx(0.4)
        assert all_metrics["attack_success_rate"] == pytest.approx(0.5)

        # Check calls to patched methods
        mock_acc.assert_any_call(y_true, y_pred_benign)
        mock_acc.assert_any_call(y_true, y_pred_adversarial)
        mock_asr.assert_called_once_with(y_true, y_pred_benign, y_pred_adversarial)

# Test edge cases for accuracy
def test_accuracy_edge_cases(metrics_calculator):
    assert metrics_calculator.calculate_accuracy(np.array([]), np.array([])) == 1.0 # Or 0.0 or NaN, depends on impl.
    assert metrics_calculator.calculate_accuracy(np.array([1]), np.array([1])) == 1.0
    assert metrics_calculator.calculate_accuracy(np.array([1]), np.array([0])) == 0.0
    with pytest.raises(ValueError): # Mismatched lengths
        metrics_calculator.calculate_accuracy(np.array([1,2]), np.array([1]))

# Test edge cases for ASR
def test_asr_edge_cases(metrics_calculator):
    # No correctly classified benign samples
    y_true = np.array([0, 1])
    y_pred_benign = np.array([1, 0]) # All wrong
    y_pred_adv = np.array([0, 0])
    assert metrics_calculator.calculate_attack_success_rate(y_true, y_pred_benign, y_pred_adv) == 0.0 # Or NaN

    # All benign correct, no successful attacks
    y_true = np.array([0, 1])
    y_pred_benign = np.array([0, 1]) # All correct
    y_pred_adv = np.array([0, 1])    # Same as benign
    assert metrics_calculator.calculate_attack_success_rate(y_true, y_pred_benign, y_pred_adv) == 0.0

    # All benign correct, all successful attacks
    y_true = np.array([0, 1])
    y_pred_benign = np.array([0, 1]) # All correct
    y_pred_adv = np.array([1, 0])    # All flipped
    assert metrics_calculator.calculate_attack_success_rate(y_true, y_pred_benign, y_pred_adv) == 1.0

    # Empty inputs
    assert metrics_calculator.calculate_attack_success_rate(np.array([]), np.array([]), np.array([])) == 0.0 # Or NaN
