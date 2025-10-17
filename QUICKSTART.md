# 🚀 Quick Start Guide

Get started with **ML Cutoff Optimizer** in 5 minutes!

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Xdrulla/ml-cutoff-optimizer.git
cd ml-cutoff-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 🎯 Basic Usage

### Python Script

```python
from ml_cutoff_optimizer import ThresholdVisualizer, CutoffOptimizer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Create/load your data
X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 2. Train any binary classifier
model = LogisticRegression()
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

# 3. Visualize distributions
visualizer = ThresholdVisualizer(y_test, y_proba, step=0.05)
visualizer.plot_distributions()

# 4. Find optimal cutoffs
optimizer = CutoffOptimizer(y_test, y_proba)
cutoffs = optimizer.suggest_three_zones()

# 5. View results
print(f"Negative Zone: 0% - {cutoffs['negative_cutoff']*100:.1f}%")
print(f"Manual Zone:   {cutoffs['negative_cutoff']*100:.1f}% - {cutoffs['positive_cutoff']*100:.1f}%")
print(f"Positive Zone: {cutoffs['positive_cutoff']*100:.1f}% - 100%")
```

## 📓 Jupyter Notebook

```bash
# Launch Jupyter
jupyter notebook

# Open the example notebook
# examples/notebooks/01_basic_usage.ipynb
```

## 🌐 Web Interface

```bash
# Run Streamlit app
streamlit run app/streamlit_app.py

# App opens at http://localhost:8501
```

## 🧪 Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ml_cutoff_optimizer --cov-report=html

# View coverage report
open htmlcov/index.html  # On macOS/Linux
start htmlcov/index.html  # On Windows
```

## 📊 Example Output

```
🎯 SUGGESTED CUTOFFS:
======================================================================
   Negative Zone: 0% - 35.0%
   Manual Zone:   35.0% - 72.0%
   Positive Zone: 72.0% - 100%
======================================================================

👥 POPULATION DISTRIBUTION:
======================================================================
   Negative Zone: 178 samples (59.3%)
   Manual Zone:    19 samples (6.3%)
   Positive Zone: 103 samples (34.3%)
======================================================================

📈 ZONE PERFORMANCE:
   • Negative Zone Specificity: 81.35%
   • Positive Zone Recall:      71.03%
```

## 🎨 Customization

### Adjust Parameters

```python
cutoffs = optimizer.suggest_three_zones(
    negative_zone_metric='specificity',  # or 'precision', 'accuracy'
    positive_zone_metric='recall',       # or 'precision', 'f1'
    min_metric_value=0.85,              # higher = more conservative
    max_manual_zone_width=0.30          # lower = less manual review
)
```

### Change Visualization

```python
visualizer = ThresholdVisualizer(y_test, y_proba, step=0.01)  # Smaller bins
fig, ax = visualizer.plot_distributions(
    figsize=(16, 8),
    title="My Custom Title"
)
visualizer.save_plot("output.png", dpi=300)
```

## 📖 Next Steps

- 📚 Read [Methodology](docs/methodology.md) for deep dive
- 📓 Explore [Example Notebooks](examples/notebooks/)
- 🌐 Try [Streamlit App](app/streamlit_app.py)
- 🧪 Check [Test Coverage](tests/)
- 🤝 Contribute via [CONTRIBUTING.md](CONTRIBUTING.md)

## ❓ Common Issues

### Import Error

```bash
# Make sure you installed the package
pip install -e .

# Or add src to path manually
import sys
sys.path.insert(0, 'src')
```

### Matplotlib Not Showing Plots

```python
import matplotlib.pyplot as plt
plt.show()  # Add this after visualizer.plot_distributions()
```

### Tests Failing

```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Clear pytest cache
rm -rf .pytest_cache __pycache__
```

## 🆘 Getting Help

- 📫 Open an [Issue](https://github.com/Xdrulla/ml-cutoff-optimizer/issues)
- 💬 Discussion: GitHub Discussions
- 📧 Email: [Your Email]

---

**Happy Optimizing! 🎯**
