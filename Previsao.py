import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Previsão com Prophet
eventos_por_dia.columns = ['ds', 'y']  # Renomeando colunas
eventos_por_dia['ds'] = pd.to_datetime(eventos_por_dia['ds'])  # Garantindo que 'ds' seja datetime

# Inicializando e treinando o modelo Prophet
modelo = Prophet()
modelo.fit(eventos_por_dia)

# Fazendo previsões
futuro = modelo.make_future_dataframe(periods=30)
previsoes = modelo.predict(futuro)

# Visualizando as previsões
modelo.plot(previsoes)
plt.show()

# Classificação com Random Forest
# Transformando colunas categóricas em numéricas
le_evento = LabelEncoder()
dados['Nome Evento'] = le_evento.fit_transform(dados['Nome Evento'])
dados['Motorista'] = LabelEncoder().fit_transform(dados['Motorista'])
dados['Frota'] = LabelEncoder().fit_transform(dados['Frota'])

# Garantindo que a data esteja no formato adequado
dados['Data'] = pd.to_datetime(dados['Data'])
dados['Ano'] = dados['Data'].dt.year
dados['Mês'] = dados['Data'].dt.month
dados['Dia'] = dados['Data'].dt.day

# Variáveis independentes (X) e dependentes (y)
X = dados[['Ano', 'Mês', 'Dia', 'Frota', 'Motorista']]
y = dados['Nome Evento']

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializando e treinando o modelo Random Forest
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Fazendo previsões e avaliando o modelo
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
