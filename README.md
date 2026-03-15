# Análise de Risco de Crédito com Regressão Logística

Projeto desenvolvido durante a Residência em Ciência de Dados do CEPEDI, 
aplicando regressão logística para prever inadimplência em uma carteira de crédito.

## Sobre o projeto
O dataset utilizado é o [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) do Kaggle.

O notebook cobre desde limpeza e análise exploratória até modelagem com pipeline, interpretabilidade com SHAP e
análise de limiar de decisão, com foco em métricas adequadas para datasets desbalanceados (AUC-ROC, Precision-Recall)
e conclusões orientadas ao negócio.

## Estrutura
1. Carregamento e limpeza dos dados
2. Análise exploratória (EDA)
3. Modelagem — base, SMOTE, GridSearch, Pipeline
4. Avaliação — AUC-ROC, limiar, Precision-Recall
5. Interpretabilidade — coeficientes e SHAP
6. Conclusões de negócio

## Principais resultados
- AUC-ROC: 0.8761 (modelo final com pipeline)
- Limiar ótimo: 0.62 — F1 de 0.68 para classe de inadimplentes
- Principal preditor: renda percentual comprometida e grau do empréstimo

## Como executar

**Opção 1 — Google Colab (recomendado):**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iIdmccf0pvXC6cY9veSc-BoxCm69NSZu#scrollTo=_exWpe_lZpLO)

Todas as dependências são instaladas automaticamente nas duas primeiras células,
basta descomentar a célula 2 e executar.

**Opção 2 — Ambiente local:**
```bash
git clone https://github.com/meurii/credit-risk-logistic-regression
cd credit-risk-logistic-regression
pip install -r requirements.txt
jupyter notebook regressao_logistica.ipynb
```

## Stack
Python · Pandas · NumPy · Scikit-learn · Imbalanced-learn · SHAP · Statsmodels · Matplotlib · Seaborn · Google Colab · KaggleHub
