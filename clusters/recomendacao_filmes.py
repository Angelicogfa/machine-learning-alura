# Import
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

np.random.seed(42)

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
grupos.T.plot.bar(subplots=True,  figsize=(28, 28),  sharex=False)

# Visualizando os filmes por grupo


def visualizar_filmes_grupo(grupo):
    filtro = kmn.labels_ == grupo
    return filmes[filtro].sample(10)


visualizar_filmes_grupo(0)

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
kmn = KMeans(n_clusters=3)
kmn.fit(generos_escalados)

grupos = pd.DataFrame(kmn.cluster_centers_, columns=generos.columns)
grupos.head()

# Visualizando os dados do grafico para melhor visualizar os cluster e os generos
grupos.T.plot.bar(subplots=True, figsize=(25, 50), sharex=False,  rot=0)

# Visualizando o filmes de um dado grupo
visualizar_filmes_grupo(2)

# Validando o erro para a quantidade de grupos gerados


def kmeans(n_clusters, generos):
    modelo = KMeans(n_clusters=n_clusters)
    modelo.fit(generos)
    return {"n_cluster": n_clusters,
            "erro": modelo.inertia_}


kmeans(17, generos_escalados)

# Buscando o melhor erro para a quantidade de grupos
kmn = KMeans(n_clusters=17)
kmn.fit(generos_escalados)

resultado = [kmeans(n, generos) for n in range(1, 41)]
frame = pd.DataFrame(resultado)
frame["erro"].plot(xticks=frame["n_cluster"])

visualizar_filmes_grupo(16)

# grupos hierarquicos
modelo = AgglomerativeClustering(n_clusters=17)
grupos = modelo.fit_predict(generos_escalados)
grupos

tnse = TSNE()
visualizacao = tnse.fit_transform(generos_escalados)
visualizacao

sns.scatterplot(x=visualizacao[:, 0], y=visualizacao[:, 1], hue=grupos)

modelo = KMeans(n_clusters=17)
modelo.fit(generos_escalados)

grupos = pd.DataFrame(modelo.cluster_centers_, columns=generos.columns)
grupos.head()
grupos.T.plot.bar(subplots=True, figsize=(25, 25), sharex=False, rot=0)

matriz_distancia = linkage(grupos)
matriz_distancia

dendograma = dendrogram(matriz_distancia)

# Visualização de grupos gerados automaticamente
dbscan = DBSCAN()
grupos = dbscan.fit_predict(generos_escalados)

tnse = TSNE()
visualizacao = tnse.fit_transform(generos_escalados)
visualizacao

sns.scatterplot(x=visualizacao[:,0], y=visualizacao[:,1], hue=grupos, palette=sns.color_palette("Set1", len(np.unique(grupos))))

filmes.head()
agrupados = pd.concat([filmes[["filme_id", "titulo", "generos"]], pd.DataFrame(dbscan.labels_, columns=["grupo"])], axis=1)
agrupados[agrupados["grupo"]==1]