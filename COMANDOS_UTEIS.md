# 🔧 Comandos Úteis - ML Cutoff Optimizer

Guia rápido de comandos para desenvolvimento e apresentação.

---

## 📦 Setup Inicial

```bash
# Navegar para o projeto
cd /home/luan/ml-cutoff-optimizer

# Ativar ambiente virtual
source venv/bin/activate

# Instalar dependências (se necessário)
pip install -r requirements.txt

# Instalar o pacote em modo desenvolvimento
pip install -e .
```

---

## 🧪 Testes

```bash
# Rodar todos os testes
pytest

# Com output verboso
pytest -v

# Com cobertura
pytest --cov=src/ml_cutoff_optimizer --cov-report=term-missing

# Gerar relatório HTML de cobertura
pytest --cov=src/ml_cutoff_optimizer --cov-report=html

# Ver relatório no navegador
firefox htmlcov/index.html  # ou chrome/brave

# Rodar testes de um arquivo específico
pytest tests/test_optimizer.py

# Rodar um teste específico
pytest tests/test_optimizer.py::TestSuggestThreeZones::test_cutoffs_are_valid
```

---

## 🚀 Executar Exemplos

### Script Python Rápido

```bash
# Rodar teste rápido
python test_quick.py
```

### Jupyter Notebook

```bash
# Iniciar Jupyter
jupyter notebook

# Ou diretamente o notebook específico
jupyter notebook examples/notebooks/01_basic_usage.ipynb
```

### Streamlit App

```bash
# Rodar app (abre automaticamente no navegador)
streamlit run app/streamlit_app.py

# Em porta específica
streamlit run app/streamlit_app.py --server.port 8502

# Modo de desenvolvimento (recarrega ao salvar)
streamlit run app/streamlit_app.py --server.runOnSave true
```

---

## 📊 Demonstração para Apresentação

### Opção 1: Demo Rápida (Terminal)

```bash
python test_quick.py
```

**Mostra**:
- Importações funcionando
- Dataset criado
- Modelo treinado
- Visualizações geradas
- Cutoffs sugeridos
- Métricas calculadas

### Opção 2: Demo Interativa (Jupyter)

```bash
jupyter notebook examples/notebooks/01_basic_usage.ipynb
```

**Mostra**:
- Passo a passo explicado
- Gráficos visualizados
- Comparação com threshold padrão

### Opção 3: Demo Visual (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

**Mostra**:
- Interface profissional
- Upload de dados
- Configuração interativa
- Resultados em tempo real
- Download de resultados

---

## 🔍 Verificar Qualidade do Código

```bash
# Formatar código com Black
black src/ tests/

# Verificar style com Flake8
flake8 src/ tests/

# Type checking com MyPy
mypy src/

# Contar linhas de código
find src -name "*.py" -exec wc -l {} + | tail -1
```

---

## 📝 Git Commands

```bash
# Ver status
git status

# Adicionar todos os arquivos
git add .

# Commit
git commit -m "feat: implementação completa do ml-cutoff-optimizer"

# Ver log
git log --oneline

# Criar branch
git checkout -b feature/nome-da-feature

# Push para GitHub
git push origin master
```

---

## 📚 Gerar Documentação

```bash
# Converter notebooks para HTML
jupyter nbconvert --to html examples/notebooks/01_basic_usage.ipynb

# Gerar documentação da API (futuro)
# pdoc --html --output-dir docs/api src/ml_cutoff_optimizer
```

---

## 🐍 Python Interativo

```bash
# Abrir Python
python

# Dentro do Python:
```

```python
import sys
sys.path.insert(0, 'src')

from ml_cutoff_optimizer import ThresholdVisualizer, CutoffOptimizer

# Exemplo rápido
y_true = [0, 0, 1, 1]
y_proba = [0.2, 0.4, 0.6, 0.8]

optimizer = CutoffOptimizer(y_true, y_proba)
cutoffs = optimizer.suggest_three_zones()

print(f"Negative: 0-{cutoffs['negative_cutoff']:.2%}")
print(f"Positive: {cutoffs['positive_cutoff']:.2%}-100%")
```

---

## 📊 Criar Visualizações

```bash
# Script Python para gerar gráficos
python << 'EOF'
import sys
sys.path.insert(0, 'src')
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo

from ml_cutoff_optimizer import ThresholdVisualizer, CutoffOptimizer
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Dados
X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Modelo
model = LogisticRegression()
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

# Otimizar
optimizer = CutoffOptimizer(y_test, y_proba)
cutoffs = optimizer.suggest_three_zones()

# Visualizar
viz = ThresholdVisualizer(y_test, y_proba, step=0.05)
viz.plot_distributions()
viz.add_cutoff_lines({
    'negative_cutoff': cutoffs['negative_cutoff'],
    'positive_cutoff': cutoffs['positive_cutoff']
})
viz.save_plot('output_visualization.png', dpi=300)

print("✅ Gráfico salvo em output_visualization.png")
EOF
```

---

## 🎯 Preparação para Apresentação

### Checklist Antes de Apresentar

```bash
# 1. Verificar que tudo funciona
pytest

# 2. Rodar demonstração rápida
python test_quick.py

# 3. Testar app Streamlit
streamlit run app/streamlit_app.py

# 4. Verificar cobertura de testes
pytest --cov=src/ml_cutoff_optimizer

# 5. Verificar que repositório está atualizado
git status
```

### Durante a Apresentação

**Terminal 1** (sempre aberto):
```bash
cd /home/luan/ml-cutoff-optimizer
source venv/bin/activate
```

**Terminal 2** (backup):
```bash
cd /home/luan/ml-cutoff-optimizer
source venv/bin/activate
streamlit run app/streamlit_app.py
```

---

## 🆘 Troubleshooting

### Erro: "ModuleNotFoundError"

```bash
# Reinstalar pacote
pip install -e .

# Ou adicionar ao path manualmente
export PYTHONPATH="${PYTHONPATH}:/home/luan/ml-cutoff-optimizer/src"
```

### Erro: "Port 8501 already in use"

```bash
# Usar porta diferente
streamlit run app/streamlit_app.py --server.port 8502

# Ou matar processo na porta 8501
lsof -ti:8501 | xargs kill -9
```

### Erro: "Permission denied"

```bash
# Dar permissão de execução
chmod +x test_quick.py
```

### Jupyter não abre

```bash
# Reinstalar Jupyter
pip install --upgrade jupyter notebook

# Ou especificar navegador
jupyter notebook --browser=firefox
```

---

## 📦 Limpar Cache

```bash
# Limpar cache do pytest
rm -rf .pytest_cache __pycache__ src/ml_cutoff_optimizer/__pycache__

# Limpar arquivos temporários
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Limpar coverage
rm -rf .coverage htmlcov/
```

---

## 🎓 Comandos para o Professor Testar

**Teste 1: Verificar instalação**
```bash
cd /home/luan/ml-cutoff-optimizer
source venv/bin/activate
python -c "from ml_cutoff_optimizer import CutoffOptimizer; print('✅ Funciona!')"
```

**Teste 2: Rodar testes**
```bash
pytest -v
```

**Teste 3: Ver demonstração**
```bash
python test_quick.py
```

**Teste 4: Ver app web**
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Estatísticas do Projeto

```bash
# Contar linhas de código
find src -name "*.py" | xargs wc -l

# Contar linhas de testes
find tests -name "*.py" | xargs wc -l

# Contar arquivos
find . -name "*.py" | wc -l

# Ver tamanho do projeto
du -sh .
```

---

**💡 Dica**: Mantenha este arquivo aberto durante a apresentação!
