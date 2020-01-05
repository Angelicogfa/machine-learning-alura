## Import
import pandas as pd

## Carregando dataset
uri_filme = "https://raw.githubusercontent.com/alura-cursos/machine-learning-algoritmos-nao-supervisionados/master/movies.csv"

filmes = pd.read_csv(uri_filme)
filmes.columns  = ["filme_id", "titulo", "generos"]
filmes.head()

## Obtendo os generos dos filmes
generos = filmes["generos"].str.get_dummies()
filmes = pd.concat([filmes, generos], axis=1)
filmes.head()

## Escalando os generos dos filmes para manter esses dados em normalização
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
generos_escalados = scaler.fit_transform(generos)

## Usando o algoritmo de agrupamento para agrupar os dados dos generos
from sklearn.cluster import KMeans
kmn = KMeans(n_clusters=3)
kmn.fit(generos_escalados)

print(f'Grupos {kmn.labels_}')
print(f"Generos¨:\n {generos}")
print(f"Centros:\n {kmn.cluster_centers_}")

## Relacionando labels de grupos com centros
grupos = pd.DataFrame(kmn.cluster_centers_, columns=generos.columns)
grupos.head()

## Plotando os dados em grafico para melhorvisualizar os clusters e os generos de cada grupo
grupos.T.plot.bar(subplots=True,
                figsize= (28,28), 
                sharex=False)