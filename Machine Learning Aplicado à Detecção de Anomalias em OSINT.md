# 🤖 Machine Learning Aplicado à Detecção de Anomalias em OSINT

*Este guia apresenta modelos de aprendizado de máquina para classificar e prever comportamentos suspeitos em dados de fontes abertas, permitindo automatizar a identificação de padrões fraudulentos ou anômalos.*

## Índice
- [Abordagem Supervisionada](#-abordagem-supervisionada)
- [Modelos Recomendados](#-modelos-recomendados)
- [Engenharia de Features](#-engenharia-de-features-variáveis-preditivas)
- [Avaliação de Desempenho](#-avaliação-de-desempenho-métricas-críticas)
- [Fluxo de Implementação](#-fluxo-de-implementação)

---

## 🎯 Abordagem Supervisionada

A classificação supervisionada é a técnica mais adequada quando se possui um conjunto de dados histórico com exemplos rotulados (ex: casos confirmados como fraudulentos ou legítimos).

**Premissa:** O modelo aprende a partir de exemplos passados para classificar novos casos como "suspeitos" ou "normais".

---

## 🌲 Modelos Recomendados

### Random Forest

**O que é:** Algoritmo baseado em múltiplas árvores de decisão que trabalham em conjunto para classificar dados.

**Vantagens para OSINT:**
- ✅ Lida bem com dados heterogêneos (numéricos e categóricos)
- ✅ Robusto contra overfitting
- ✅ Fornece importância das variáveis (explica o que mais influencia a decisão)
- ✅ Suporta dados desbalanceados (comum em fraude, onde casos legítimos são maioria)

**Aplicação em OSINT:**
- Classificar empresas como "alto risco" ou "baixo risco" com base em características cadastrais e comportamentais
- Identificar perfis de redes sociais com probabilidade de serem bots
- Detectar candidatos com maior chance de envolvimento em irregularidades eleitorais

### XGBoost (Extreme Gradient Boosting)

**O que é:** Algoritmo avançado de boosting que combina múltiplos modelos fracos sequencialmente, onde cada novo modelo corrige os erros do anterior.

**Vantagens para OSINT:**
- ✅ Alta performance e precisão
- ✅ Excelente para dados estruturados
- ✅ Lida bem com valores missing
- ✅ Regularização integrada para evitar overfitting
- ✅ Velocidade de treinamento e predição

**Aplicação em OSINT:**
- Modelos preditivos para identificar padrões de corrupção em licitações
- Detecção de fraudes documentais baseada em metadados
- Classificação de ameaças em investigações de cibersegurança

```mermaid
flowchart TD
    A[Dados Históricos<br/>Rotulados] --> B[Treinamento do Modelo]
    
    B --> C[Random Forest<br/>Múltiplas Árvores]
    B --> D[XGBoost<br/>Boosting Sequencial]
    
    C --> E[Modelo Treinado]
    D --> E
    
    E --> F[Novos Dados<br/>Não Rotulados]
    F --> G[Predição:<br/>Fraudulento ou Legítimo?]
    G --> H[🚩 ALERTA:<br/>Casos Suspeitos]
    
    style A fill:#e1f5e1
    style H fill:#ffe1e1
    style E fill:#e3f2fd
