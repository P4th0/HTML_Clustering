import os 
import sklearn
import numpy as np
import codecs
from bs4 import BeautifulSoup
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

class Clustering:
    def __init__(self, eps=1, min_samples=2):
        self.eps = eps
        self.min_samples = min_samples
        self.vectors = []
        self.matrix = None
        self.labels = None
        self.clusters = {}

    def validate_html(self, path):
        return path.lower().endswith(".html")

    def load_html(self, path):
        with open(path, "r", encoding="utf-8") as f:
            continut_html = f.read()
        soup = BeautifulSoup(continut_html, "lxml")
        return soup

    def vector_features(self, path_html):
        soup = self.load_html(path_html)
        return {
            "path": path_html,
            "num_tags": len(soup.find_all(True)),
            "num_headings": sum(len(soup.find_all(f"h{i}")) for i in range(1, 7)),
            "num_imgs": len(soup.find_all("img")),
            "title_len": len(soup.title.string.strip()) if soup.title and soup.title.string else 0,
            "has_nav": 1 if soup.find("nav") else 0,
            "inline_styles": len([tag for tag in soup.find_all(style=True)]),
            "style_blocks": len(soup.find_all("style")),
            "num_links": len(soup.find_all("a")),
            "num_scripts": len(soup.find_all("script")),
            "has_inputs": 1 if soup.find("input") else 0,
            
        }

    def create_vectorsF_htmls(self, path):
        vectors = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not self.validate_html(file_path):
                continue
            try:
                vector = self.vector_features(file_path)
                if vector:
                    vectors.append(vector)
            except Exception as e:
                print(f"Eroare la fisierul {file_path}: {e}")
        return vectors

    def vectors_to_matrix(self, vectors):
        matrix = []
        for v in vectors:
            matrix.append([
                v["num_tags"],
                v["num_headings"],
                v["num_imgs"],
                v["title_len"],
                v["has_nav"],
                v["inline_styles"],
                v["style_blocks"],
                v["num_links"],
                v["num_scripts"],
                v["has_inputs"],
            ])
        return np.array(matrix)

    def cluster_features(self, matrix):
        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(matrix)
        model = sklearn.cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = model.fit_predict(scaled_matrix)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if num_clusters > 1:
            score = silhouette_score(scaled_matrix, labels)
            print(f"DBSCAN : {num_clusters} clustere, silhouette score: {score:.3f}")
        else:
            print("DBSCAN : doar 1 cluster sau zgomot, silhouette score nerelevant.")

        return labels

    def assign_clusters(self, vectors, labels):
        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(vectors[i]["path"])
        return clusters

    def classify(self, path):
        vectors = self.create_vectorsF_htmls(path)
        matrix = self.vectors_to_matrix(vectors)
        labels = self.cluster_features(matrix)
        clusters = self.assign_clusters(vectors, labels)
        return clusters

    def plot_k_distance_graph(self, X, k):
        print(f"Dimensiunile matricei: {X.shape}")
        num_dimensions = X.shape[1]
        print(f"Numărul de dimensiuni (caracteristici): {num_dimensions}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(X_scaled)
        distances, indices = neigh.kneighbors(X_scaled)
        distances = np.sort(distances[:, k-1])

        plt.figure(figsize=(10, 6))
        plt.plot(distances, label=f'{k}-th nearest neighbor distance')
        diffs = np.diff(distances)
        threshold = np.percentile(diffs, 95)

        change_points = np.where(diffs > threshold)[0]
        if len(change_points) > 0:
            elbow_index = change_points[0]
            elbow_value = distances[elbow_index]
            plt.axhline(y=elbow_value, color='red', linestyle='--', label='Elbow Point')
            print(f"Punctul de elbow detectat la index {elbow_index}, cu valoarea {elbow_value:.3f}.")

        y_min = np.floor(min(distances) * 10) / 10
        y_max = np.ceil(max(distances) * 10) / 10
        plt.yticks(np.arange(y_min, y_max + 0.1, 0.1))

        plt.xlabel('Points')
        plt.ylabel(f'{k}-th nearest neighbor distance')
        plt.title('K-distance Graph')
        plt.legend()
        plt.show()



   


