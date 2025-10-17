# 📊 ML Cutoff Optimizer

> Professional toolkit for binary classification threshold optimization with intelligent three-zone analysis

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 What is this?

**ML Cutoff Optimizer** is a Python library that helps you find the **optimal probability thresholds** for binary classification models. Instead of using a single threshold (like 0.5), it intelligently suggests **three decision zones**:

- 🔴 **Negative Zone**: High confidence predictions for class 0
- 🟡 **Manual Zone**: Uncertain predictions requiring human review
- 🟢 **Positive Zone**: High confidence predictions for class 1

### Why 3 zones instead of 1 threshold?

In real-world applications, **not all predictions are equally confident**. By identifying three zones, you can:

- ✅ Automate high-confidence decisions
- ✅ Flag uncertain cases for manual review
- ✅ Reduce errors and improve business outcomes
- ✅ Balance precision, recall, and operational costs

---

## 🚀 Features

- 📈 **Visual Distribution Analysis**: Overlay population histograms to see how predictions distribute
- 🎯 **Smart Cutoff Suggestions**: Automatically find optimal thresholds based on your data
- 📊 **Comprehensive Metrics**: Calculate precision, recall, F1, accuracy at any threshold
- 🎨 **Beautiful Visualizations**: Publication-ready plots with customizable styling
- 🔧 **Model Agnostic**: Works with any binary classifier (sklearn, XGBoost, neural networks, etc.)
- 📓 **Interactive Examples**: Jupyter notebooks with real-world use cases
- 🌐 **Web Interface**: Streamlit app for non-technical users

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Xdrulla/ml-cutoff-optimizer.git
cd ml-cutoff-optimizer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

---

## 🎬 Quick Start

```python
from ml_cutoff_optimizer import ThresholdVisualizer, CutoffOptimizer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Train any binary classification model
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# 2. Visualize distributions
visualizer = ThresholdVisualizer(y_test, y_proba, step=0.05)
visualizer.plot_distributions()

# 3. Get optimal cutoffs
optimizer = CutoffOptimizer(y_test, y_proba)
cutoffs = optimizer.suggest_three_zones()

print(f"Negative Zone: 0% - {cutoffs['negative_cutoff']*100:.1f}%")
print(f"Manual Zone: {cutoffs['negative_cutoff']*100:.1f}% - {cutoffs['positive_cutoff']*100:.1f}%")
print(f"Positive Zone: {cutoffs['positive_cutoff']*100:.1f}% - 100%")
print(f"\nJustification:\n{cutoffs['justification']}")
```

**Output example:**

```
Negative Zone: 0% - 35.0%
Manual Zone: 35.0% - 72.0%
Positive Zone: 72.0% - 100%

Justification:
The negative cutoff (35%) captures 89% of true negatives with only 5% error rate.
The positive cutoff (72%) captures 91% of true positives with 8% error rate.
The manual zone (37% of total population) requires human review.
```

---

## 📚 Examples

Explore real-world use cases in the `examples/notebooks/` directory:

1. **[Basic Usage](examples/notebooks/01_basic_usage.ipynb)** - Simple synthetic example
2. **[Spam Detection](examples/notebooks/02_spam_detection.ipynb)** - Email classification
3. **[Credit Risk](examples/notebooks/03_credit_risk.ipynb)** - Loan approval automation

---

## 🎨 Visualization Gallery

*(Images will be added here after implementation)*

---

## 📖 Documentation

- [Methodology](docs/methodology.md) - How the optimization works
- [API Reference](docs/api_reference.md) - Complete class/method documentation
- [Quick Start Guide](QUICKSTART.md) - Get started in 5 minutes

---

## 🌐 Web Interface

Launch the interactive Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

Upload your model predictions and explore cutoffs visually!

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ml_cutoff_optimizer --cov-report=html
```

**Test Results:**
- ✅ 69 tests passing (100%)
- ✅ 96% code coverage

---

## 🤝 Contributing

Contributions are welcome! Please check [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Luan Drulla**

- GitHub: [@Xdrulla](https://github.com/Xdrulla)
- LinkedIn: [Luan Drulla](https://www.linkedin.com/in/luan-drulla-822a24189/)

---

## 🙏 Acknowledgments

This project was developed as part of a machine learning course and serves as a practical tool for real-world classification problems.

---

**⭐ If you find this useful, please star the repository!**
