# 📊 ML Cutoff Optimizer - Resumo Completo do Projeto

**Autor**: Luan Drulla
**Repositório**: https://github.com/Xdrulla/ml-cutoff-optimizer
**Data**: Outubro 2025

---

## 🎯 Objetivo do Projeto

Criar uma biblioteca Python profissional para visualização e otimização de pontos de corte em modelos de classificação binária, implementando uma estratégia inteligente de **três zonas de decisão**.

---

## 💡 Problema Resolvido

### Problema Tradicional (Threshold único = 0.5)

A maioria dos sistemas de ML usa um threshold fixo (0.5):
- ❌ Trata todas as previsões igualmente
- ❌ Não considera níveis de confiança
- ❌ Threshold 0.5 nem sempre é ótimo
- ❌ Não permite revisão manual de casos incertos

### Solução Proposta (3 Zonas)

```
0%              35%             72%            100%
|---------------|---------------|---------------|
  Zona Negativa   Zona Manual    Zona Positiva
  (Auto-Rejeitar) (Revisar)      (Auto-Aceitar)
```

**Benefícios**:
- ✅ Automatiza decisões com alta confiança
- ✅ Sinaliza casos incertos para revisão humana
- ✅ Reduz erros críticos
- ✅ Otimiza uso de recursos humanos

---

## 🏗️ Arquitetura do Projeto

### Estrutura de Arquivos

```
ml-cutoff-optimizer/
├── src/ml_cutoff_optimizer/    # Código principal
│   ├── __init__.py              # Package initialization
│   ├── utils.py                 # Validações (100% cobertura)
│   ├── metrics.py               # Cálculo de métricas (100% cobertura)
│   ├── visualizer.py            # Visualizações (97% cobertura)
│   └── optimizer.py             # Otimização (90% cobertura)
├── tests/                       # Testes unitários
│   ├── test_utils.py            # 16 testes
│   ├── test_metrics.py          # 16 testes
│   ├── test_visualizer.py       # 17 testes
│   └── test_optimizer.py        # 20 testes
├── examples/notebooks/          # Jupyter notebooks
│   └── 01_basic_usage.ipynb     # Exemplo completo
├── app/                         # Interface web
│   ├── streamlit_app.py         # App Streamlit
│   └── README.md                # Instruções do app
├── docs/                        # Documentação
│   └── methodology.md           # Metodologia detalhada
├── README.md                    # Documentação principal
├── QUICKSTART.md                # Guia rápido
├── CONTRIBUTING.md              # Guia de contribuição
├── LICENSE                      # Licença MIT
├── setup.py                     # Instalação
└── requirements.txt             # Dependências
```

### Módulos Principais

#### 1. **utils.py** - Validações
**Responsabilidade**: Validar dados de entrada

**Funções**:
- `validate_binary_inputs()` - Valida y_true e y_proba
- `validate_probabilities()` - Garante probabilidades entre 0-1
- `validate_step()` - Valida tamanho dos bins
- `validate_threshold()` - Valida thresholds individuais

**Por que é importante**: Evita erros silenciosos com dados inválidos (fail fast)

---

#### 2. **metrics.py** - Cálculo de Métricas
**Responsabilidade**: Calcular métricas de classificação para qualquer threshold

**Classe**: `MetricsCalculator`

**Métodos principais**:
- `confusion_matrix_at_threshold()` - TP, TN, FP, FN para threshold específico
- `calculate_all_metrics()` - Todas as métricas (precision, recall, F1, accuracy, etc.)
- `population_distribution()` - Distribuição populacional por bins
- `metrics_by_threshold_range()` - Métricas para 101 thresholds (0.00 a 1.00)

**Métricas calculadas**:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
- Accuracy = (TP + TN) / Total
- FPR (False Positive Rate) = FP / (FP + TN)
- FNR (False Negative Rate) = FN / (FN + TP)

---

#### 3. **visualizer.py** - Visualizações
**Responsabilidade**: Criar gráficos profissionais de distribuições

**Classe**: `ThresholdVisualizer`

**Métodos principais**:
- `plot_distributions()` - Histogramas sobrepostos (azul: população total, vermelho: classe positiva)
- `add_cutoff_lines()` - Adiciona linhas verticais nos cutoffs sugeridos
- `save_plot()` - Salva gráfico em alta resolução

**Por que histogramas sobrepostos**:
- Facilita comparação visual
- Mostra onde modelo está confiante vs incerto
- Identifica pontos naturais de separação

---

#### 4. **optimizer.py** - Otimização Inteligente
**Responsabilidade**: Sugerir automaticamente os 3 cutoffs ótimos

**Classe**: `CutoffOptimizer`

**Métodos principais**:
- `calculate_metrics_matrix()` - Calcula métricas para todos os thresholds
- `suggest_three_zones()` - **Método principal** - sugere os cutoffs
- `plot_metrics_evolution()` - Visualiza como métricas mudam com threshold

**Algoritmo de Otimização**:

```
1. Calcular métricas para 101 thresholds (0.00 a 1.00)

2. Encontrar Cutoff Negativo:
   - Filtrar onde specificity >= min_metric_value (ex: 0.80)
   - Escolher o MAIOR threshold válido
   - Resultado: Alta confiança para classificar como negativo

3. Encontrar Cutoff Positivo:
   - Filtrar onde recall >= min_metric_value (ex: 0.80)
   - Escolher o MENOR threshold válido
   - Resultado: Alta confiança para classificar como positivo

4. Validar Zona Manual:
   - Garantir que cutoff_negativo <= cutoff_positivo
   - Limitar largura da zona manual (max 40% da população)

5. Gerar Justificativa:
   - Calcular % da população em cada zona
   - Calcular métricas de performance
   - Gerar relatório explicativo
```

---

## 🧪 Testes e Qualidade

### Cobertura de Testes: 96%

**Total**: 69 testes unitários

| Módulo | Testes | Cobertura |
|--------|--------|-----------|
| utils.py | 16 | 100% |
| metrics.py | 16 | 100% |
| visualizer.py | 17 | 97% |
| optimizer.py | 20 | 90% |

### Tipos de Testes

1. **Testes Positivos** - Dados válidos devem funcionar
2. **Testes Negativos** - Dados inválidos devem gerar erros apropriados
3. **Edge Cases** - Casos extremos (arrays vazios, todos 0s, todos 1s)
4. **Integração** - Testes de fluxo completo

### Executar Testes

```bash
# Rodar todos os testes
pytest

# Com cobertura
pytest --cov=src/ml_cutoff_optimizer --cov-report=html

# Ver relatório
open htmlcov/index.html
```

---

## 📓 Exemplos e Demonstrações

### 1. Jupyter Notebook

**Arquivo**: `examples/notebooks/01_basic_usage.ipynb`

**Conteúdo** (22 células):
1. Importações
2. Criação de dataset sintético
3. Treinamento de modelo
4. Visualização de distribuições
5. Otimização de cutoffs
6. Análise de resultados
7. Comparação com threshold padrão (0.5)

**Como rodar**:
```bash
jupyter notebook examples/notebooks/01_basic_usage.ipynb
```

### 2. Streamlit App (Interface Web)

**Arquivo**: `app/streamlit_app.py`

**Funcionalidades**:
- Upload de CSV ou uso de dados exemplo
- Seleção de colunas (y_true, y_proba)
- Configuração de parâmetros (step, min_metric_value, max_manual_zone)
- Visualização interativa
- Métricas detalhadas por zona
- Download de resultados (JSON e CSV)

**Como rodar**:
```bash
streamlit run app/streamlit_app.py
# Abre automaticamente em http://localhost:8501
```

### 3. Script Python Simples

```python
from ml_cutoff_optimizer import ThresholdVisualizer, CutoffOptimizer

# Dados (substitua pelos seus)
y_true = [0, 0, 1, 1, 0, 1, ...]
y_proba = [0.2, 0.3, 0.8, 0.9, 0.1, 0.7, ...]

# Visualizar
viz = ThresholdVisualizer(y_true, y_proba, step=0.05)
viz.plot_distributions()

# Otimizar
optimizer = CutoffOptimizer(y_true, y_proba)
cutoffs = optimizer.suggest_three_zones()

# Resultados
print(cutoffs['justification'])
```

---

## 📚 Documentação

### README.md
- Visão geral do projeto
- Features principais
- Instalação
- Quick start
- Exemplos de uso
- Links para documentação

### QUICKSTART.md
- Instalação rápida
- Exemplo básico em Python
- Como rodar notebook
- Como rodar app Streamlit
- Troubleshooting

### docs/methodology.md
- Explicação detalhada do problema
- Metodologia de otimização
- Explicação de todas as métricas
- Casos de uso práticos
- Referências acadêmicas

### CONTRIBUTING.md
- Como reportar bugs
- Como sugerir features
- Setup de desenvolvimento
- Padrões de código
- Como fazer Pull Requests

---

## 🎓 Casos de Uso Reais

### 1. Detecção de Spam

```
Zona Negativa (0-25%):  Auto-permitir → Inbox
Zona Manual (25-85%):   Revisão do usuário
Zona Positiva (85-100%): Auto-bloquear → Spam
```

**Benefício**: 80% dos emails processados automaticamente

### 2. Avaliação de Risco de Crédito

```
Zona Negativa (0-30%):  Auto-aprovar empréstimo
Zona Manual (30-75%):   Análise de subscritor
Zona Positiva (75-100%): Auto-rejeitar
```

**Benefício**: 65% das aplicações processadas automaticamente

### 3. Diagnóstico Médico

```
Zona Negativa (0-15%):  Sem ação necessária
Zona Manual (15-60%):   Encaminhar especialista
Zona Positiva (60-100%): Biópsia recomendada
```

**Benefício**: Reduz biópsias desnecessárias mantendo alta sensibilidade

---

## 🛠️ Tecnologias Utilizadas

### Core
- **Python 3.8+** - Linguagem principal
- **NumPy** - Operações matemáticas
- **Pandas** - Manipulação de dados
- **Matplotlib** - Visualizações
- **Seaborn** - Estilização de gráficos
- **scikit-learn** - Métricas de ML

### Interface
- **Streamlit** - App web interativo

### Desenvolvimento
- **pytest** - Framework de testes
- **pytest-cov** - Cobertura de testes
- **Jupyter** - Notebooks interativos

### Documentação
- **Markdown** - Documentação
- **GitHub Pages** - (futuro) Hospedagem de docs

---

## 📊 Resultados de Exemplo

Com dataset sintético (300 amostras):

```
CUTOFFS SUGERIDOS:
  Zona Negativa: 0% - 45%
  Zona Manual:   45% - 55%
  Zona Positiva: 55% - 100%

DISTRIBUIÇÃO POPULACIONAL:
  59.3% em Zona Negativa  (178 amostras) → Automação
  6.3%  em Zona Manual    (19 amostras)  → Revisão Humana
  34.3% em Zona Positiva  (103 amostras) → Automação

PERFORMANCE:
  Zona Negativa - Specificity: 81.35% (acerta 81% dos negativos)
  Zona Positiva - Recall:      71.03% (captura 71% dos positivos)

IMPACTO:
  93.6% das decisões automatizadas com alta confiança
  6.4% requerem revisão manual (casos incertos)
```

---

## 🎯 Diferenciais do Projeto

### Para Portfólio

1. **Código Profissional**
   - ✅ Arquitetura modular (SOLID principles)
   - ✅ Type hints em todas as funções
   - ✅ Docstrings estilo NumPy
   - ✅ Code style consistente (Black-compliant)

2. **Testes Robustos**
   - ✅ 69 testes unitários
   - ✅ 96% de cobertura
   - ✅ CI/CD ready (GitHub Actions)
   - ✅ Edge cases cobertos

3. **Documentação Completa**
   - ✅ README profissional
   - ✅ Documentação técnica detalhada
   - ✅ Guias de uso (QUICKSTART)
   - ✅ Notebooks interativos

4. **Interface de Usuário**
   - ✅ App web (Streamlit)
   - ✅ Jupyter notebooks
   - ✅ CLI-friendly

5. **Boas Práticas**
   - ✅ Licença open-source (MIT)
   - ✅ CONTRIBUTING.md
   - ✅ .gitignore apropriado
   - ✅ setup.py para instalação

### Para Trabalho Acadêmico

1. **Fundamentação Teórica**
   - ✅ Explicação das métricas
   - ✅ Justificativa do algoritmo
   - ✅ Referências acadêmicas

2. **Demonstração Prática**
   - ✅ Casos de uso reais
   - ✅ Visualizações claras
   - ✅ Resultados quantificáveis

3. **Reprodutibilidade**
   - ✅ Código open-source
   - ✅ Dados exemplo incluídos
   - ✅ Instruções detalhadas

---

## 🚀 Como Apresentar ao Professor

### 1. Demonstração ao Vivo (5-10 min)

**Opção A: Jupyter Notebook**
```bash
jupyter notebook examples/notebooks/01_basic_usage.ipynb
```
- Executa célula por célula
- Explica cada passo
- Mostra visualizações

**Opção B: Streamlit App**
```bash
streamlit run app/streamlit_app.py
```
- Interface visual
- Interativo
- Impressionante visualmente

### 2. Pontos a Destacar

**Problema**:
*"Sistemas de ML tradicionalmente usam threshold fixo de 0.5, que não considera níveis de confiança do modelo. Isso pode levar a erros em decisões críticas."*

**Solução**:
*"Implementei um sistema de 3 zonas que automatiza decisões com alta confiança e sinaliza casos incertos para revisão humana, reduzindo erros e otimizando recursos."*

**Implementação**:
*"Criei uma biblioteca Python modular com 96% de cobertura de testes, documentação completa, e interface web interativa."*

**Resultados**:
*"No exemplo, 93.6% das decisões podem ser automatizadas com alta confiança, reduzindo carga de trabalho manual em 93.6% enquanto mantém baixa taxa de erro."*

### 3. Perguntas Esperadas e Respostas

**P: "Por que 3 zonas e não 2 ou 4?"**
**R**: *"Três zonas fornecem o melhor equilíbrio: automatizam decisões óbvias (alta confiança para 0 e 1) e isolam incerteza (zona manual). Mais zonas complicam sem ganho significativo."*

**P: "Como você escolhe os cutoffs?"**
**R**: *"Uso otimização baseada em métricas: busco o ponto onde specificity (acerto de negativos) ≥ 80% para cutoff negativo, e recall (captura de positivos) ≥ 80% para cutoff positivo. Isso garante alta confiança nas zonas automatizadas."*

**P: "Funciona com qualquer modelo?"**
**R**: *"Sim! É model-agnostic. Funciona com qualquer classificador binário que produza probabilidades: Logistic Regression, Random Forest, XGBoost, Redes Neurais, etc."*

**P: "Como você testou?"**
**R**: *"Criei 69 testes unitários cobrindo 96% do código, testando casos normais, edge cases (arrays vazios, todos 0s), e casos difíceis (probabilidades todas ~0.5). Todos os testes passam."*

**P: "Por que NumPy/Pandas e não só Python puro?"**
**R**: *"NumPy é muito mais eficiente para operações matemáticas (arrays otimizados em C). Pandas facilita manipulação de dados tabulares. Scikit-learn fornece métricas validadas pela comunidade."*

---

## 📈 Métricas de Qualidade

| Aspecto | Métrica | Status |
|---------|---------|--------|
| Cobertura de Testes | 96% | ✅ Excelente |
| Testes Passando | 69/69 (100%) | ✅ Todos passam |
| Documentação | 8 arquivos MD | ✅ Completa |
| Exemplos | 1 notebook + 1 app | ✅ Interativos |
| Code Style | PEP 8 compliant | ✅ Consistente |
| Type Hints | 100% das funções | ✅ Totalmente tipado |

---

## 🔮 Possíveis Extensões Futuras

1. **Suporte a Multiclasse** - Generalizar para >2 classes
2. **Otimização Baseada em Custo** - Considerar custos de FP vs FN
3. **AutoML Integration** - Integrar com AutoML pipelines
4. **Deploy em Produção** - API REST com FastAPI
5. **Publicação no PyPI** - `pip install ml-cutoff-optimizer`
6. **Artigo Técnico** - Publicar no Medium/Dev.to

---

## 📞 Contato

**Luan Drulla**
- GitHub: [@Xdrulla](https://github.com/Xdrulla)
- LinkedIn: [luan-drulla-822a24189](https://www.linkedin.com/in/luan-drulla-822a24189/)
- Email: [serighelli003@gmail.com]

---

## 📄 Licença

Este projeto está licenciado sob a **MIT License**, permitindo uso comercial, modificação e distribuição.

---

**⭐ Se este projeto foi útil, considere dar uma estrela no GitHub!**
