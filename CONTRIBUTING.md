# Contributing to ML Cutoff Optimizer

First off, thank you for considering contributing to ML Cutoff Optimizer! 🎉

## 🐛 How to Report Bugs

If you find a bug, please open an issue on GitHub with:

- **Clear title** describing the problem
- **Steps to reproduce** the bug
- **Expected behavior** vs actual behavior
- **Environment details** (Python version, OS, library versions)
- **Code snippet** that reproduces the issue (if possible)

## 💡 How to Suggest Features

We welcome feature suggestions! Please open an issue with:

- **Clear description** of the feature
- **Use case** - Why is this useful?
- **Examples** - How would it work?

## 🔧 Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ml-cutoff-optimizer.git
   cd ml-cutoff-optimizer
Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
pip install -e ".[dev]"  # Install with dev dependencies
Create a branch for your changes:
git checkout -b feature/your-feature-name
📝 Code Standards
We follow strict code quality standards:
Style Guide
PEP 8 compliance (enforced by flake8)
Black for code formatting
Type hints for all function parameters and returns
Before Committing
Run these commands to ensure code quality:
# Format code with Black
black src/ tests/

# Check code style
flake8 src/ tests/

# Type checking
mypy src/

# Run tests
pytest

# Check test coverage
pytest --cov=src/ml_cutoff_optimizer --cov-report=html
Docstring Format
We use NumPy-style docstrings:
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Examples
    --------
    >>> example_function("test", 42)
    True
    """
    pass
🧪 Testing Guidelines
Write tests for all new features
Maintain coverage - Aim for >80%
Test edge cases - Empty arrays, all zeros, all ones, etc.
Use descriptive test names - test_visualizer_handles_empty_array
Example test structure:
import pytest
from ml_cutoff_optimizer import ThresholdVisualizer

class TestThresholdVisualizer:
    def test_initialization_with_valid_data(self):
        """Test that visualizer initializes correctly with valid inputs"""
        # Arrange
        y_true = [0, 0, 1, 1]
        y_proba = [0.2, 0.3, 0.7, 0.8]
        
        # Act
        viz = ThresholdVisualizer(y_true, y_proba)
        
        # Assert
        assert viz is not None
📤 Submitting a Pull Request
Commit your changes:
git add .
git commit -m "Add feature: description of your changes"
Push to your fork:
git push origin feature/your-feature-name
Open a Pull Request on GitHub with:
Clear description of changes
Link to related issues (if any)
Screenshots (for visual changes)
Test results
Wait for review - We'll review and provide feedback
📋 Commit Message Guidelines
Use clear, descriptive commit messages:
Add feature: three-zone cutoff optimization
Fix bug: handle empty probability arrays
Docs: update README with new examples
Test: add edge case tests for visualizer
Refactor: simplify metrics calculation logic
✅ Pull Request Checklist
Before submitting, make sure:
 Code follows style guidelines (Black, Flake8, MyPy pass)
 All tests pass (pytest)
 New tests added for new features
 Docstrings added/updated
 README updated (if needed)
 No breaking changes (or clearly documented)
🙏 Thank You!
Your contributions make this project better for everyone. We appreciate your time and effort! ❤️
📞 Questions?
Feel free to open an issue with the label "question" if you need help or clarification.

---

Pronto! ✅

Agora vamos começar a criar os **arquivos Python principais** - o coração do projeto! 🚀

---