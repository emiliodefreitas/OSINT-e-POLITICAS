Técnicas Estatísticas para OSINT 
*Abordagem quantitativa para detectar anomalias*

+------------------------+-------------------------------------------+-----------------------------------------------+
| Técnica                | Aplicação em OSINT                        | O que Detecta                                 |
+========================+===========================================+===============================================+
| Z-score                | Contratos públicos, patrimônio,            | Outliers numéricos (valores muito acima/      |
|                        | menções na mídia                           | abaixo da média)                              |
+------------------------+-------------------------------------------+-----------------------------------------------+
| Clusterização          | Empresas, perfis sociais, doações          | Grupos anômalos e comportamentos isolados     |
| (K-means/DBSCAN)       |                                           |                                               |
+------------------------+-------------------------------------------+-----------------------------------------------+
| Regras de Associação   | Relacionamentos suspeitos, sociedade,      | Combinações ilógicas e padrões coordenados    |
| (Apriori)              | desinformação                              |                                               |
+------------------------+-------------------------------------------+-----------------------------------------------+

**Fluxo Integrado:**
Coleta -> EDA (análise exploratória) -> Detecção (Z-score/Clusterização/Regras) -> Alertas -> Validação Humana

---

### 2. Machine Learning para OSINT (Pergunta 5)
*Modelos supervisionados para classificação de fraudes*

#### Modelos Recomendados:

+----------------+---------------------------------------------------------------------------------+
| Modelo         | Características                                                                |
+================+=================================================================================+
| Random Forest  | - Robustez a outliers                                                          |
|                | - Fornece importância das features                                             |
|                | - Lida bem com dados heterogêneos (numéricos e categóricos)                    |
|                | - Menos propenso a overfitting                                                 |
+----------------+---------------------------------------------------------------------------------+
| XGBoost        | - Alta performance e precisão                                                  |
|                | - Excelente para dados estruturados e tabulares                                |
|                | - Velocidade de processamento otimizada                                        |
|                | - Lida bem com desbalanceamento de classes                                     |
+----------------+---------------------------------------------------------------------------------+

#### Métricas de Avaliação (foco em recall):

+---------------------+---------------------------------------------------+---------------------------------------------+
| Métrica             | Fórmula                                           | Aplicação em OSINT                          |
+=====================+===================================================+=============================================+
| Recall              | TP / (TP + FN)                                    | Prioridade: minimizar falsos negativos      |
| (Sensibilidade)     |                                                   | (fraudes não detectadas)                    |
+---------------------+---------------------------------------------------+---------------------------------------------+
| AUC-ROC             | Área sob a curva ROC                              | Capacidade geral de distinguir classes      |
+---------------------+---------------------------------------------------+---------------------------------------------+
| Precision           | TP / (TP + FP)                                    | Importante para não sobrecarregar           |
|                     |                                                   | investigadores com falsos alertas           |
+---------------------+---------------------------------------------------+---------------------------------------------+
| Matthews            | (TP*TN - FP*FN) /                                 | Melhor para dados desbalanceados            |
| Correlation (MCC)   | √((TP+FP)(TP+FN)(TN+FP)(TN+FN))                   |                                             |
+---------------------+---------------------------------------------------+---------------------------------------------+

**Observação:** Em investigações, priorizamos **recall** (encontrar o máximo de fraudes possíveis) mesmo que isso signifique alguns falsos positivos que serão depois validados manualmente.

---

### 3. Feature Engineering para OSINT

+---------------------+---------------------------------------------------+-----------------------------------------------+
| Categoria           | Descrição                                         | Exemplos de Features                          |
+=====================+===================================================+===============================================+
| Temporais           | Padrões relacionados a tempo                      | - Tempo entre eventos                         |
|                     |                                                   | - Recorrência em datas específicas            |
|                     |                                                   | - Sazonalidade                                |
+---------------------+---------------------------------------------------+-----------------------------------------------+
| Comportamentais     | Padrões de ação da entidade                       | - Frequência de atividades                    |
|                     |                                                   | - Horários atípicos                           |
|                     |                                                   | - Mudanças bruscas de comportamento           |
+---------------------+---------------------------------------------------+-----------------------------------------------+
| Rede/Relacionamento | Conexões entre entidades                          | - Grau de separação de nós suspeitos          |
|                     |                                                   | - Densidade de conexões                       |
|                     |                                                   | - Centralidade em rede                        |
+---------------------+---------------------------------------------------+-----------------------------------------------+
| Anomalias           | Desvios da norma                                  | - Z-score de valores                          |
| Estatísticas        |                                                   | - Distância em clusterização                  |
|                     |                                                   | - Frequência incomum de co-ocorrências        |
+---------------------+---------------------------------------------------+-----------------------------------------------+

#### Exemplos Práticos de Features:

**Para Análise de Empresas:**
- tempo_media_abertura_fechamento: tempo médio que empresas do mesmo sócio permanecem ativas
- proporcao_contratos_suspeitos: percentual de contratos da empresa com indícios de irregularidade
- rede_socios_tamanho: número total de conexões societárias do grupo econômico
- reincidencia_cnpj: quantidade de CNPJs vinculados ao mesmo CPF

**Para Análise de Perfis em Redes Sociais:**
- horario_medio_post: horário médio das postagens
- desvio_padrao_horario: variabilidade nos horários de postagem
- razao_seguidores_seguindo: proporção entre seguidores e contas seguidas
- frequencia_hashtags: padrões de uso de hashtags específicas

**Para Análise de Licitações:**
- vencedor_unico: se a mesma empresa venceu licitações com concorrentes diferentes
- proximidade_edital_resultado: tempo entre publicação do edital e resultado
- concentracao_geografica: se a empresa vence em regiões muito específicas

---

### 4. Exemplos de Código

#### Exemplo em Python (Random Forest):

```python
# ============================================
# EXEMPLO PYTHON - DETECÇÃO DE FRAUDES EM OSINT
# ============================================

# Importação das bibliotecas essenciais
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score
from sklearn.preprocessing import LabelEncoder

# 1. CARREGAR DADOS
dados = pd.read_csv('dados_osint.csv')  # Substitua pelo seu arquivo

# 2. PRÉ-PROCESSAMENTO
# Tratar valores faltantes
dados = dados.fillna(dados.median())

# Codificar variáveis categóricas
le = LabelEncoder()
for col in dados.select_dtypes(include=['object']).columns:
    if col != 'target':
        dados[col] = le.fit_transform(dados[col])

# Separar features (X) e target (y)
X = dados.drop('target', axis=1)
y = dados['target']

# 3. DIVIDIR EM TREINO E TESTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. TREINAR MODELO RANDOM FOREST
modelo_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'  # importante para dados desbalanceados
)

modelo_rf.fit(X_train, y_train)

# 5. FAZER PREDIÇÕES
y_pred = modelo_rf.predict(X_test)
y_proba = modelo_rf.predict_proba(X_test)[:, 1]

# 6. AVALIAR O MODELO
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraude']))

print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")

# 7. FEATURE IMPORTANCE
importancias = pd.DataFrame({
    'feature': X.columns,
    'importancia': modelo_rf.feature_importances_
}).sort_values('importancia', ascending=False)

print("\nTop 5 Features Mais Importantes:")
print(importancias.head(5))
Exemplo em R (Random Forest):
r
# ============================================
# EXEMPLO R - DETECÇÃO DE FRAUDES EM OSINT
# ============================================

# Carregar bibliotecas
library(randomForest)
library(caret)
library(pROC)

# 1. CARREGAR DADOS
dados <- read.csv('dados_osint.csv')  # Substitua pelo seu arquivo

# 2. PRÉ-PROCESSAMENTO
# Tratar valores faltantes (substituir pela mediana)
for (col in names(dados)) {
  if (col != 'target' && is.numeric(dados[[col]])) {
    dados[[col]][is.na(dados[[col]])] <- median(dados[[col]], na.rm = TRUE)
  }
}

# Garantir que target seja fator
dados$target <- as.factor(dados$target)
levels(dados$target) <- c("Normal", "Fraude")

# 3. DIVIDIR EM TREINO E TESTE
set.seed(42)
indices <- createDataPartition(dados$target, p = 0.8, list = FALSE)
treino <- dados[indices, ]
teste <- dados[-indices, ]

# 4. TREINAR MODELO RANDOM FOREST
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

modelo_rf <- train(
  target ~ .,
  data = treino,
  method = "rf",
  trControl = ctrl,
  metric = "ROC",
  importance = TRUE
)

# 5. FAZER PREDIÇÕES
predicoes <- predict(modelo_rf, newdata = teste)
probabilidades <- predict(modelo_rf, newdata = teste, type = "prob")

# 6. AVALIAR O MODELO
cm <- confusionMatrix(predicoes, teste$target)
print("Matriz de Confusão:")
print(cm$table)

print("\nMétricas por classe:")
print(cm$byClass)

roc_curve <- roc(teste$target, probabilidades$Fraude)
print(paste("AUC-ROC:", round(auc(roc_curve), 4)))

# 7. FEATURE IMPORTANCE
importancia <- varImp(modelo_rf)
print("\nImportância das Features:")
print(importancia)
5. Objetivos da Análise OSINT
+------------------------+---------------------------------------------------------------------------------+
| Objetivo | Descrição |
+========================+=================================================================================+
| Prevenir Perdas | Identificar irregularidades em contratos, licitações, transações |
+------------------------+---------------------------------------------------------------------------------+
| Detectar Fraudes | Revelar esquemas coordenados, empresas de fachada, perfis falsos |
+------------------------+---------------------------------------------------------------------------------+
| Proteger Integridade | Garantir transparência em processos públicos e privados |
+------------------------+---------------------------------------------------------------------------------+
| Otimizar Recursos | Focar esforços investigativos onde há maior probabilidade de achados |
+------------------------+---------------------------------------------------------------------------------+

6. Fluxo de Trabalho Completo
text
    [Coleta de Dados] 
           |
           v
    [Feature Engineering]
           |
           v
    [Modelagem ML]
           |
           v
    [Avaliação com Foco em Recall]
           |
           v
    [Alertas Investigativos]
           |
           v
    [Validação Humana]
           |
           v
    [Relatório Final]
7. Principais Alertas
+------------------------+---------------------------------------------------------------------------------+
| Alerta | Descrição |
+========================+=================================================================================+
| Z-score > 3 | Valor muito acima da média (contratos, patrimônio, etc.) |
+------------------------+---------------------------------------------------------------------------------+
| Pontos isolados em | Comportamento que não se encaixa em nenhum grupo |
| clusterização | |
+------------------------+---------------------------------------------------------------------------------+
| Regras com lift > 1 | Associações suspeitas entre entidades |
+------------------------+---------------------------------------------------------------------------------+
| Recall baixo | Modelo está deixando fraudes passarem (falsos negativos) |
+------------------------+---------------------------------------------------------------------------------+

8. Checklist de Implementação
+---------------------------------------------------------------------------------+
| [ ] Coletar dados históricos com rótulos confiáveis |
| [ ] Criar features relevantes para o contexto |
| [ ] Dividir dados (treino/validação/teste) |
| [ ] Treinar Random Forest e XGBoost |
| [ ] Avaliar com foco em recall |
| [ ] Interpretar feature importance |
| [ ] Validar alertas com investigação humana |
| [ ] Documentar todo o processo |
+---------------------------------------------------------------------------------+

9. Exemplos de Aplicação
+---------------------+------------------------+---------------------------------------------------+
| Contexto | Técnica ML Recomendada | Features Relevantes |
+=====================+========================+===================================================+
| Licitações Públicas | Random Forest | - Concentração de contratos |
| | | - Tempo entre edital e resultado |
| | | - Rede de sócios |
| | | - Proximidade geográfica |
+---------------------+------------------------+---------------------------------------------------+
| Redes Sociais | XGBoost | - Horário médio de posts |
| | | - Razão seguidores/seguindo |
| | | - Frequência de hashtags |
| | | - Padrões de interação |
+---------------------+------------------------+---------------------------------------------------+
| Doações Eleitorais | Random Forest | - Valor médio por doador |
| | | - Reincidência de CPFs |
| | | - Padrões geográficos |
| | | - Concentração em datas específicas |
+---------------------+------------------------+---------------------------------------------------+

📌 Resumo dos Conceitos Chave
Z-score: Detecta valores extremos (|Z| > 3 = alerta)

Clusterização: Encontra grupos e isolados (DBSCAN é ótimo para ruído)

Regras de Associação: Descobre padrões de co-ocorrência (lift > 1 = associação relevante)

Random Forest: Robusto, interpretável, bom para dados heterogêneos

XGBoost: Alta performance, excelente para dados desbalanceados

Recall: Métrica crítica (não deixar fraudes passarem)

Feature Engineering: Etapa mais importante do processo
