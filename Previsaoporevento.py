# 1. Importação das Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neuralprophet import NeuralProphet
from termcolor import colored

# 2. Carregar dados
dados = pd.read_csv('C:/Users/Operação/Desktop/Projeto/basedadosseguranca.csv', delimiter=';')

# 3. Verificação Inicial
print(colored("Dados iniciais:", 'cyan', attrs=['bold']))
print(dados.head())
print(f"\n{colored('Shape do dataset:', 'yellow')} {dados.shape}")
print(colored("\nInformações do dataset:", 'cyan', attrs=['bold']))
print(dados.info())

# 4. Verificar valores ausentes
print(colored("\nValores ausentes por coluna:", 'yellow'))
print(dados.isnull().sum())

# 5. Preencher valores ausentes
dados['Motorista'] = dados['Motorista'].fillna(dados['Motorista'].mode()[0])

for column in dados.columns:
    if dados[column].dtype == 'object' and column != 'Motorista':
        dados[column] = dados[column].fillna(dados[column].mode()[0])
    elif dados[column].dtype in ['int64', 'float64']:
        dados[column] = dados[column].fillna(dados[column].mean())

# 6. Verificar os tipos de dados
print(colored("\nTipos de dados por coluna:", 'cyan', attrs=['bold']))
print(dados.dtypes)

# Converter colunas de data para datetime
dados['Data'] = pd.to_datetime(dados['Data'], dayfirst=True)

# 7. Estatísticas Descritivas
print(colored("\nEstatísticas descritivas do dataset:", 'cyan', attrs=['bold']))
print(dados.describe())

# 8. Remover duplicatas agregando a quantidade de eventos por data
df_aggregated = dados.groupby('Data', as_index=False)['QUANTIDADE'].sum()

# 9. Preparar os dados para o NeuralProphet
df = df_aggregated.rename(columns={'Data': 'ds', 'QUANTIDADE': 'y'})

# 10. Criar o modelo NeuralProphet
modelo = NeuralProphet()

# 11. Ajustar o modelo aos dados
modelo.fit(df, freq='D')

# 12. Criar um DataFrame para as futuras previsões
futuro = modelo.make_future_dataframe(df, periods=30)

# 13. Fazer a previsão
previsao = modelo.predict(futuro)

# 14. Filtrar a previsão para a data específica
data_especifica = '2024-09-25'
previsao_hoje = previsao[previsao['ds'] == data_especifica]
print(f"\n{colored('Previsão para', 'yellow')} {colored(data_especifica, 'yellow', attrs=['bold'])}:")
print(previsao_hoje)

# 15. Analisando os 10 motoristas mais propensos a eventos
total_eventos = dados['QUANTIDADE'].sum()  # Total de eventos históricos

# Total de eventos previstos para o dia
total_previsto_hoje = previsao_hoje['yhat1'].values[0]  

eventos_por_motorista = dados.groupby('Motorista')['QUANTIDADE'].sum().reset_index()
top_10_motoristas = eventos_por_motorista.sort_values(by='QUANTIDADE', ascending=False).head(10)

# Calcular a porcentagem de probabilidade de cada motorista ter eventos no dia específico
top_10_motoristas['Probabilidade'] = (top_10_motoristas['QUANTIDADE'] / total_eventos) * total_previsto_hoje

# Exibir os 10 motoristas mais propensos com a probabilidade de eventos previstos para o dia
print(colored(f"\nProbabilidade dos 10 motoristas mais propensos a eventos em {data_especifica}:", 'yellow', attrs=['bold']))
for index, row in top_10_motoristas.iterrows():
    print(colored(f"Motorista: {row['Motorista']}, Probabilidade de Eventos: {row['Probabilidade']:.2f}", 'green'))

# 16. Criar gráfico de barras para os motoristas mais propensos
plt.figure(figsize=(10, 6))
sns.barplot(x='Motorista', y='Probabilidade', data=top_10_motoristas)

plt.title(f'Probabilidade dos Top 10 Motoristas para Eventos em {data_especifica}', fontsize=16)
plt.xlabel('Motorista', fontsize=12)
plt.ylabel('Probabilidade de Eventos', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()  # Exibir o gráfico

# 17. Analisando os 5 motoristas mais propensos a um tipo específico de evento
def motoristas_mais_propensos_por_evento(dados, nome_evento, total_previsto_hoje):
    if 'Nome' not in dados.columns:
        print(colored(f"Erro: A coluna 'Nome' não foi encontrada no DataFrame.", 'red'))
        return
    
    dados_evento = dados[dados['Nome'] == nome_evento]

    if dados_evento.empty:
        print(colored(f"Nenhum dado encontrado para o evento: {nome_evento}.", 'red'))
        return
    
    total_eventos_tipo = dados_evento['QUANTIDADE'].sum()
    eventos_por_motorista = dados_evento.groupby('Motorista')['QUANTIDADE'].sum().reset_index()
    top_5_motoristas_evento = eventos_por_motorista.sort_values(by='QUANTIDADE', ascending=False).head(5)

    if top_5_motoristas_evento.empty:
        print(colored(f"Nenhum motorista encontrado para o evento: {nome_evento}.", 'red'))
        return

    top_5_motoristas_evento['Probabilidade'] = (top_5_motoristas_evento['QUANTIDADE'] / total_eventos_tipo) * total_previsto_hoje

    print(colored(f"\nProbabilidade dos 5 motoristas mais propensos a eventos de {nome_evento}:", 'yellow', attrs=['bold']))
    for index, row in top_5_motoristas_evento.iterrows():
        print(colored(f"Motorista: {row['Motorista']}, Probabilidade de Eventos de {nome_evento}: {row['Probabilidade']:.2f}", 'green'))

# Exemplo de uso para calcular os motoristas mais propensos a cada evento
eventos = ['Excesso de Velocidade', 'Fadiga', 'Curva Brusca']
for evento in eventos:
    motoristas_mais_propensos_por_evento(dados, evento, total_previsto_hoje)
