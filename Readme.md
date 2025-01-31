# Análise de Segurança e Previsão de Eventos

## Descrição do Projeto
Este projeto utiliza Machine Learning para prever a quantidade de eventos de segurança em operações de transporte, com base em dados históricos. A partir de um modelo NeuralProphet, realizamos análises e previsões para identificar os motoristas mais propensos a eventos em um dia específico.

---

## 📁 Estrutura do Projeto

- **Importação das bibliotecas:** Utilização de bibliotecas populares como `pandas`, `numpy`, `matplotlib`, `seaborn`, e `neuralprophet`.
- **Análise exploratória:** Verificação e tratamento de dados ausentes, análise estatística e remoção de duplicatas.
- **Modelo de Previsão:** Treinamento do modelo NeuralProphet para previsões diárias.
- **Análise por Motorista:** Identificação dos motoristas mais propensos a eventos com base nas previsões.

---

## 🚀 Etapas do Desenvolvimento

### **1. Importação das Bibliotecas**
As principais bibliotecas utilizadas incluem:
- `pandas`, `numpy`: Manipulação de dados.
- `matplotlib`, `seaborn`: Visualização de dados.
- `neuralprophet`: Previsão com redes neurais.

### **2. Carregamento e Verificação Inicial dos Dados**
Leitura e inspeção dos dados armazenados em `basedadosseguranca.csv`.

### **3. Limpeza e Preparação dos Dados**
Tratamento de valores ausentes, conversão de datas e remoção de duplicatas.

### **4. Treinamento do Modelo**
Treinamento do modelo NeuralProphet para prever eventos em operações de transporte.

### **5. Análise de Resultados**
- Previsão para datas específicas
- Identificação dos motoristas mais propensos a eventos.
- Gráficos de visualização dos principais resultados.

---

## 📊 Análises e Visualizações
- Gráfico de barras com os motoristas mais propensos a eventos.
- Estatísticas descritivas para análise dos dados históricos.

---

## 🔍 Previsões Realizadas
- Probabilidades para os 10 motoristas mais propensos a eventos.
- Probabilidades detalhadas para eventos específicos como "Excesso de Velocidade", "Fadiga" e "Curva Brusca".

---

## 💡 Tecnologias Utilizadas
- Python
- Pandas
- Matplotlib
- Seaborn
- NeuralProphet

---

## 📈 Principais Contribuições
- Automação da previsão de eventos em segurança de transporte.
- Identificação de motoristas com alto risco em períodos futuros.
- Visualizações claras para suporte à tomada de decisões.

---

## 📜 Como Executar o Projeto
1. Clone o repositório
2. Instale as dependências:
   ```bash
   pip install pandas numpy matplotlib seaborn neuralprophet termcolor
3. Execute o script principal
python nome_do_arquivo.py
