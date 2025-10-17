# 📚 Methodology - ML Cutoff Optimizer

## 🎯 Introduction

**ML Cutoff Optimizer** is a tool for optimizing decision thresholds in binary classification models. Unlike traditional approaches that use a single threshold (typically 0.5), this library implements a **three-zone strategy** that:

1. Automates high-confidence decisions
2. Flags uncertain predictions for human review
3. Reduces critical errors in production systems

---

## ❓ The Problem with Single Thresholds

### Traditional Approach (Single Threshold = 0.5)

Most binary classifiers output probabilities between 0 and 1. The standard approach is:

```python
if probability >= 0.5:
    predict class 1 (positive)
else:
    predict class 0 (negative)
```

**Problems with this approach:**

1. **One-size-fits-all**: Treats all predictions equally, ignoring confidence levels
2. **Arbitrary cutoff**: 0.5 is not optimal for most real-world problems
3. **No uncertainty handling**: Doesn't distinguish between 51% and 99% confidence
4. **Risk-blind**: Doesn't account for different costs of false positives vs false negatives

### Example: Credit Risk Assessment

Consider a loan approval system:

| Probability | Traditional (0.5) | Risk Reality |
|-------------|-------------------|--------------|
| 0.51 | ✅ Approve | 49% chance of default! |
| 0.49 | ❌ Reject | 49% chance of repayment! |
| 0.95 | ✅ Approve | High confidence ✓ |
| 0.05 | ❌ Reject | High confidence ✓ |

The system treats 0.51 (barely positive) the same as 0.95 (very confident), which is risky!

---

## 💡 Three-Zone Strategy

### Concept

Instead of one threshold, use **two thresholds** to create **three decision zones**:

```
0%              T1              T2              100%
|---------------|---------------|---------------|
  Negative Zone   Manual Zone    Positive Zone
  (Auto-Reject)   (Human Review) (Auto-Accept)
```

Where:
- **T1** (Negative Cutoff): Upper bound for negative zone
- **T2** (Positive Cutoff): Lower bound for positive zone

### Decision Logic

```python
if probability < T1:
    # Negative Zone: High confidence it's class 0
    decision = "AUTO_REJECT"

elif probability >= T2:
    # Positive Zone: High confidence it's class 1
    decision = "AUTO_ACCEPT"

else:  # T1 <= probability < T2
    # Manual Zone: Uncertain - requires human review
    decision = "MANUAL_REVIEW"
```

### Benefits

1. **Higher confidence on automated decisions**: Only automate when model is confident
2. **Reduced critical errors**: Flag uncertain cases for expert review
3. **Flexible risk management**: Adjust zone widths based on cost-benefit analysis
4. **Better resource allocation**: Focus human expertise where it matters most

---

## 🔬 Optimization Algorithm

### Goal

Find thresholds T1 and T2 that:
- Maximize **specificity** (correct negatives) in the negative zone
- Maximize **recall** (correct positives) in the positive zone
- Minimize manual review zone width (reduce human workload)

### Algorithm Steps

#### Step 1: Calculate Metrics for All Thresholds

For each threshold from 0.00 to 1.00 (101 values):
1. Convert probabilities to predictions using that threshold
2. Calculate confusion matrix: TP, TN, FP, FN
3. Calculate metrics: precision, recall, F1, specificity, etc.
4. Store in DataFrame for analysis

#### Step 2: Find Negative Cutoff (T1)

**Objective**: Identify cases we can confidently classify as negative

**Method**:
```
1. Filter thresholds where specificity >= min_threshold (e.g., 0.80)
2. Among valid thresholds, choose the HIGHEST one
3. This creates a restrictive negative zone with high confidence
```

**Intuition**:
- Specificity = TN / (TN + FP) = ability to identify true negatives
- High threshold → fewer samples in negative zone → but higher confidence
- Example: If T1=0.35, we're saying "anything below 35% is confidently negative"

#### Step 3: Find Positive Cutoff (T2)

**Objective**: Identify cases we can confidently classify as positive

**Method**:
```
1. Filter thresholds where recall >= min_threshold (e.g., 0.80)
2. Among valid thresholds, choose the LOWEST one
3. This creates a restrictive positive zone with high confidence
```

**Intuition**:
- Recall = TP / (TP + FN) = ability to identify true positives
- Low threshold → fewer samples in positive zone → but higher confidence
- Example: If T2=0.72, we're saying "anything above 72% is confidently positive"

#### Step 4: Validate Manual Zone

**Constraints**:
```python
# T1 must be <= T2 (zones don't overlap)
if T1 > T2:
    midpoint = (T1 + T2) / 2
    T1 = midpoint - 0.05
    T2 = midpoint + 0.05

# Manual zone can't be too wide
manual_width = T2 - T1
if manual_width > max_manual_zone_width:
    center = (T1 + T2) / 2
    half_width = max_manual_zone_width / 2
    T1 = center - half_width
    T2 = center + half_width
```

#### Step 5: Generate Justification

Calculate and report:
- Population distribution (% in each zone)
- Performance metrics for each zone
- Expected error rates
- Recommendations for implementation

---

## 📊 Key Metrics Explained

### Confusion Matrix Components

```
                    Predicted
                    0       1
Actual    0       [TN]    [FP]
          1       [FN]    [TP]
```

- **TP (True Positive)**: Correctly predicted positive
- **TN (True Negative)**: Correctly predicted negative
- **FP (False Positive)**: Incorrectly predicted positive (Type I error)
- **FN (False Negative)**: Incorrectly predicted negative (Type II error)

### Primary Metrics

#### 1. Precision (Positive Predictive Value)
```
Precision = TP / (TP + FP)
```
**Question**: "Of all positive predictions, how many were correct?"

**Use case**: Important when false positives are costly
- Example: Spam detection (don't want to mark real emails as spam)

#### 2. Recall (Sensitivity, True Positive Rate)
```
Recall = TP / (TP + FN)
```
**Question**: "Of all actual positives, how many did we catch?"

**Use case**: Important when false negatives are costly
- Example: Cancer detection (don't want to miss actual cases)

#### 3. Specificity (True Negative Rate)
```
Specificity = TN / (TN + FP)
```
**Question**: "Of all actual negatives, how many did we correctly identify?"

**Use case**: Important for negative zone optimization
- Example: Credit risk (correctly identify reliable borrowers)

#### 4. F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
**Question**: "What's the harmonic mean of precision and recall?"

**Use case**: When you need balanced performance

#### 5. Accuracy
```
Accuracy = (TP + TN) / Total
```
**Question**: "What percentage of all predictions were correct?"

**Warning**: Can be misleading with imbalanced datasets!

### Error Rates

#### False Positive Rate (FPR)
```
FPR = FP / (FP + TN) = 1 - Specificity
```
Lower is better for negative zone

#### False Negative Rate (FNR)
```
FNR = FN / (FN + TP) = 1 - Recall
```
Lower is better for positive zone

---

## 🎯 Practical Examples

### Example 1: Spam Detection

**Scenario**: Email classification system

**Goal**:
- Auto-block obvious spam (high confidence)
- Auto-allow obvious ham (high confidence)
- Flag borderline emails for user review

**Three-Zone Configuration**:
```
Negative Zone (0% - 25%): Auto-allow (inbox)
Manual Zone  (25% - 85%): User review (questionable)
Positive Zone (85% - 100%): Auto-block (spam folder)
```

**Why this works**:
- Emails with <25% spam probability are clearly legitimate
- Emails with >85% spam probability are clearly spam
- Only 15-20% of emails need user decision

**Metrics to optimize**:
- Negative zone: High specificity (don't block legitimate emails)
- Positive zone: High recall (catch most spam)

### Example 2: Credit Risk Assessment

**Scenario**: Automated loan approval system

**Goal**:
- Auto-approve low-risk applicants
- Auto-reject high-risk applicants
- Send medium-risk to underwriters

**Three-Zone Configuration**:
```
Negative Zone (0% - 30%): Auto-approve loan
Manual Zone  (30% - 75%): Manual underwriting
Positive Zone (75% - 100%): Auto-reject loan
```

**Business impact**:
- 40% of applications auto-approved → faster service
- 25% of applications auto-rejected → save underwriter time
- 35% require manual review → focused expert attention

**Cost-benefit**:
- False positive (reject good applicant): Lost customer + revenue
- False negative (approve bad applicant): Default + loss
- Manual review: Underwriter cost vs accuracy gain

### Example 3: Medical Diagnosis Support

**Scenario**: Cancer screening tool

**Goal**:
- Reduce unnecessary biopsies (false positives)
- Never miss actual cancer (false negatives)
- Flag uncertain cases for specialist review

**Three-Zone Configuration**:
```
Negative Zone (0% - 15%): No further action needed
Manual Zone  (15% - 60%): Refer to specialist
Positive Zone (60% - 100%): Immediate biopsy recommended
```

**Critical considerations**:
- Recall in positive zone should be very high (>95%)
- Specificity in negative zone should be high (>90%)
- Manual zone might be large (better safe than sorry)

---

## 🔧 Parameters Guide

### `min_metric_value` (default: 0.80)

**What it controls**: Minimum acceptable performance for automated zones

**Higher values (0.90-0.99)**:
- ✅ Higher confidence in automated decisions
- ✅ Lower error rates
- ❌ Smaller automated zones
- ❌ More manual reviews

**Lower values (0.60-0.75)**:
- ✅ Larger automated zones
- ✅ Less manual workload
- ❌ Higher error rates
- ❌ Lower confidence

**Recommendation**: Start with 0.80, adjust based on error cost

### `max_manual_zone_width` (default: 0.40)

**What it controls**: Maximum percentage of data in manual review zone

**Lower values (0.10-0.25)**:
- ✅ Most decisions automated
- ✅ Less human workload
- ❌ May sacrifice accuracy
- ❌ Zones might be too aggressive

**Higher values (0.40-0.60)**:
- ✅ More conservative automation
- ✅ Lower risk of errors
- ❌ More manual reviews needed
- ❌ Higher operational cost

**Recommendation**: Balance between automation efficiency and accuracy needs

### `step` (visualization parameter, default: 0.05)

**What it controls**: Histogram bin size for visualization

**Smaller values (0.01-0.03)**:
- ✅ More detailed histogram
- ✅ Better for large datasets
- ❌ Can be visually cluttered

**Larger values (0.10-0.20)**:
- ✅ Cleaner visualization
- ✅ Better for small datasets
- ❌ Less detail

---

## 📖 References

### Academic Papers

1. **Threshold Optimization in Binary Classification**
   - Provost, F., & Fawcett, T. (2001). Robust classification for imprecise environments. *Machine Learning*, 42(3), 203-231.

2. **Cost-Sensitive Learning**
   - Elkan, C. (2001). The foundations of cost-sensitive learning. In *International joint conference on artificial intelligence* (Vol. 17, No. 1, pp. 973-978).

3. **ROC Analysis**
   - Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861-874.

### Related Concepts

- **Precision-Recall Trade-off**: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
- **ROC Curves**: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
- **Confusion Matrix**: https://en.wikipedia.org/wiki/Confusion_matrix
- **Cost-Sensitive Learning**: https://machinelearningmastery.com/cost-sensitive-learning-for-imbalanced-classification/

### Tools & Libraries

- **scikit-learn**: https://scikit-learn.org/ (metrics calculation)
- **matplotlib**: https://matplotlib.org/ (visualizations)
- **pandas**: https://pandas.pydata.org/ (data manipulation)
- **numpy**: https://numpy.org/ (numerical operations)

---

## 🤝 Contributing

If you'd like to contribute to improving the methodology:

1. Suggest alternative optimization algorithms
2. Propose new metrics to consider
3. Share real-world use cases
4. Report edge cases or bugs

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

---

**Last Updated**: 2025
**Author**: Luan Drulla
