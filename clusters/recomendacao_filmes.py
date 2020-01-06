# Import
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Carregando dataset
uri_filme = "https://raw.githubusercontent.com/alura-cursos/machine-learning-algoritmos-nao-supervisionados/master/movies.csv"

filmes = pd.read_csv(uri_filme)
filmes.columns = ["filme_id", "titulo", "generos"]
filmes.head()

# Obtendo os generos dos filmes
generos = filmes["generos"].str.get_dummies()
filmes = pd.concat([filmes, generos], axis=1)
filmes.head()

# Escalando os generos dos filmes para manter esses dados em normalização

scaler = StandardScaler()
generos_escalados = scaler.fit_transform(generos)

# Usando o algoritmo de agrupamento para agrupar os dados dos generos
kmn = KMeans(n_clusters=3)
kmn.fit(generos_escalados)

print(f'Grupos {kmn.labels_}')
print(f"Generos¨:\n {generos}")
print(f"Centros:\n {kmn.cluster_centers_}")

# Relacionando labels de grupos com centros
grupos = pd.DataFrame(kmn.cluster_centers_, columns=generos.columns)
grupos.head()


# Plotando os dados em grafico para melhor visualizar os clusters e os generos de cada grupo
grupos.T.plot.bar(subplots=True,
                  figsize=(28, 28),
                  sharex=False)

# Visualizando os filmes por grupo
grupo = 0
filtro = kmn.labels_ == grupo
filmes[filtro].sample(10)

# Visualizando os grupos apos com a redução de dimensionalidade
tsne = TSNE()
visualizacao = tsne.fit_transform(generos_escalados)
visualizacao

sns.set(rc={"figure.figsize": (13, 13)})
sns.scatterplot(x=visualizacao[:, 0],
                y=visualizacao[:, 1],
                hue=kmn.labels_,
                palette=sns.color_palette('Set1', 3))

plt.scatter(x=visualizacao[:, 0], y=visualizacao[:, 1])

# Agrupando os dados com base nos generos
kmn = KMeans(n_clusters=20)
kmn.fit(generos_escalados)

grupos = pd.DataFrame(kmn.cluster_centers_, columns=generos.columns)
grupos.head()

# Visualizando os dados do grafico para melhor visualizar os cluster e os generos
grupos.T.plot.bar(subplots=True, figsize=(25, 50), sharex=False, rot=0)
