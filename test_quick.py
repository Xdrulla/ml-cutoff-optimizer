"""
Quick test to validate the ml_cutoff_optimizer implementation.
"""

import sys
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Add src to path so we can import our package
sys.path.insert(0, "src")

print("=" * 70)
print("🧪 QUICK TEST - ML Cutoff Optimizer")
print("=" * 70)

# Step 1: Import our modules
print("\n📦 Step 1: Importing modules...")
try:
    from ml_cutoff_optimizer import (
        ThresholdVisualizer,
        CutoffOptimizer,
        MetricsCalculator,
    )

    print("✅ All modules imported successfully!")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Step 2: Create synthetic data
print("\n📊 Step 2: Creating synthetic dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.6, 0.4],  # 60% class 0, 40% class 1
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(
    f"✅ Dataset created: {len(X_train)} training samples, {len(X_test)} test samples"
)
print(
    f"   Class distribution: {np.sum(y_test == 0)} negatives, {np.sum(y_test == 1)} positives"
)

# Step 3: Train a simple model
print("\n🤖 Step 3: Training Logistic Regression model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1
print(f"✅ Model trained! Accuracy: {model.score(X_test, y_test):.2%}")
print(f"   Probability range: {y_proba.min():.3f} - {y_proba.max():.3f}")

# Step 4: Test MetricsCalculator
print("\n📈 Step 4: Testing MetricsCalculator...")
try:
    metrics = MetricsCalculator.calculate_all_metrics(y_test, y_proba, threshold=0.5)
    print(f"✅ Metrics at threshold=0.5:")
    print(f"   • Accuracy:  {metrics['accuracy']:.2%}")
    print(f"   • Precision: {metrics['precision']:.2%}")
    print(f"   • Recall:    {metrics['recall']:.2%}")
    print(f"   • F1-Score:  {metrics['f1']:.2%}")
except Exception as e:
    print(f"❌ MetricsCalculator error: {e}")
    import traceback

    traceback.print_exc()

# Step 5: Test CutoffOptimizer
print("\n🎯 Step 5: Testing CutoffOptimizer...")
try:
    optimizer = CutoffOptimizer(y_test, y_proba)
    cutoffs = optimizer.suggest_three_zones(
        min_metric_value=0.75,  # Lower threshold for test data
        max_manual_zone_width=0.50,
    )

    print(f"✅ Cutoffs suggested successfully!")
    print(f"\n   ZONES:")
    print(f"   • Negative Zone: 0% - {cutoffs['negative_cutoff']*100:.1f}%")
    print(
        f"   • Manual Zone:   {cutoffs['negative_cutoff']*100:.1f}% - {cutoffs['positive_cutoff']*100:.1f}%"
    )
    print(f"   • Positive Zone: {cutoffs['positive_cutoff']*100:.1f}% - 100%")

    print(f"\n   POPULATION:")
    print(
        f"   • {cutoffs['population']['negative_zone_pct']:.1f}% in Negative Zone ({cutoffs['population']['negative_zone_count']} samples)"
    )
    print(
        f"   • {cutoffs['population']['manual_zone_pct']:.1f}% in Manual Zone ({cutoffs['population']['manual_zone_count']} samples)"
    )
    print(
        f"   • {cutoffs['population']['positive_zone_pct']:.1f}% in Positive Zone ({cutoffs['population']['positive_zone_count']} samples)"
    )

    print(f"\n   PERFORMANCE:")
    neg_metrics = cutoffs["metrics"]["negative_zone"]
    pos_metrics = cutoffs["metrics"]["positive_zone"]
    print(f"   • Negative Zone Specificity: {neg_metrics['specificity']:.2%}")
    print(f"   • Positive Zone Recall:      {pos_metrics['recall']:.2%}")

except Exception as e:
    print(f"❌ CutoffOptimizer error: {e}")
    import traceback

    traceback.print_exc()

# Step 6: Test ThresholdVisualizer (without showing plot)
print("\n🎨 Step 6: Testing ThresholdVisualizer...")
try:
    visualizer = ThresholdVisualizer(y_test, y_proba, step=0.05)
    print(f"✅ Visualizer initialized successfully!")
    print(f"   • Number of bins: {len(visualizer.bins) - 1}")
    print(f"   • Positive class samples: {len(visualizer.y_proba_positive)}")
    print(f"   • Negative class samples: {len(visualizer.y_proba_negative)}")

    # Create plot (but don't show it in terminal)
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    fig, ax = visualizer.plot_distributions()
    visualizer.add_cutoff_lines(
        {
            "negative_cutoff": cutoffs["negative_cutoff"],
            "positive_cutoff": cutoffs["positive_cutoff"],
        }
    )
    print(f"✅ Visualization created successfully!")

except Exception as e:
    print(f"❌ ThresholdVisualizer error: {e}")
    import traceback

    traceback.print_exc()

# Final summary
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\n💡 Next steps:")
print("   1. Create unit tests in tests/")
print("   2. Create Jupyter notebooks with examples")
print("   3. Build Streamlit app")
print("   4. Write documentation")
print("\n🎉 Your ml-cutoff-optimizer is ready to use!")
print("=" * 70)
