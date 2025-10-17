# 🌐 ML Cutoff Optimizer - Streamlit App

Interactive web application for finding optimal probability thresholds in binary classification.

## 🚀 How to Run

### Option 1: From project root
```bash
# Activate virtual environment
source venv/bin/activate

# Run the app
streamlit run app/streamlit_app.py
```

### Option 2: From app directory
```bash
# Activate virtual environment
source ../venv/bin/activate

# Run the app
streamlit run streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 📋 Features

### 1. Data Input
- **Upload CSV**: Upload your own binary classification results
  - Required: Two columns (true labels 0/1, predicted probabilities 0-1)
- **Use Example Data**: Auto-generated synthetic dataset for testing

### 2. Configuration
- **Histogram Bin Size**: Adjust visualization granularity (0.01 - 0.20)
- **Minimum Metric Value**: Set minimum performance for automated zones (0.50 - 0.99)
- **Max Manual Zone Width**: Control maximum percentage requiring human review (0.10 - 0.60)

### 3. Visualizations
- **Probability Distribution**: Overlapping histograms showing overall vs positive class
- **Cutoff Lines**: Visual representation of suggested thresholds

### 4. Results
- **Three-Zone Cutoffs**: Negative, Manual, and Positive zones
- **Population Distribution**: Percentage and count in each zone
- **Performance Metrics**: Detailed metrics for each zone
- **Justification Report**: Comprehensive explanation of suggestions

### 5. Export
- **JSON Format**: Complete results with all metrics
- **CSV Format**: Summary table with key metrics

## 🎯 Use Cases

### Spam Detection
```
Negative Zone → Auto-block
Manual Zone → User review
Positive Zone → Auto-allow
```

### Credit Risk Assessment
```
Negative Zone → Auto-reject loan
Manual Zone → Manual underwriting
Positive Zone → Auto-approve loan
```

### Medical Diagnosis
```
Negative Zone → No further testing
Manual Zone → Additional screening
Positive Zone → Refer to specialist
```

## 📸 Screenshots

*(Screenshots would be added here after running the app)*

## 🛠️ Troubleshooting

### Port already in use
If port 8501 is already in use:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Import errors
Make sure you're in the virtual environment:
```bash
source venv/bin/activate  # or ../venv/bin/activate from app/
```

### Module not found
Install dependencies:
```bash
pip install -r requirements.txt
```

## 📝 CSV Format Example

Your CSV should look like this:

```csv
y_true,y_proba
0,0.23
1,0.87
0,0.12
1,0.94
...
```

- `y_true`: Actual labels (must be 0 or 1)
- `y_proba`: Predicted probabilities (must be between 0 and 1)

## 🎨 Customization

To customize the app, edit `streamlit_app.py`:

- **Colors**: Modify CSS in the `st.markdown()` section
- **Default parameters**: Change default values in sliders
- **Layout**: Adjust column widths and arrangements

---

**Built with ❤️ by Luan Drulla**
