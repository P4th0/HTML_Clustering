from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import time
import numpy as np
import torch
import clip
from PIL import Image
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import umap
from html_DBSCAN import Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



class ImageComp:
    def __init__(self):
        self.A = []                         # matricea de antrenament
        self.file_paths = []

    def validate_html(self,path):  # verifică dacă fișierul are extensie .html
        return path.lower().endswith(".html")

    def ss_html(self,file_path,file_name,tier_nr): #functie screenshot pagina html

        options = Options()
        options.add_argument("--no-sandbox") 
        options.add_argument("--disable-dev-shm-usage")  
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.set_window_size(1920, 1080) 

        try:
            abs_path = os.path.abspath(file_path)
            file_url = "file://" + abs_path
            driver.get(file_url)
            time.sleep(1)
            output_dir = os.path.join("html_ss", f"tier{tier_nr}")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{file_name}.png")
            driver.save_screenshot(output_path)
            
        finally:
            driver.quit()

    def ss_all_htmls(self,path,tier_nr): #ss la toate html din folder tier
        for file in os.listdir(path):
            file_path = os.path.join(path, file)

            if not self.validate_html(file_path):  # dacă NU e html skip
                continue
            try:
                self.ss_html(file_path,file,tier_nr)
            except Exception as e:
                print(f"Eroare la fisierul {file_path}: {e}")
    
    def clip_clustering(self, A,folder_path,num_clusters): 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        image_paths = []
        embeddings = []

        for filename in A:
            file_path = os.path.join(folder_path, filename+".png")
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            image_paths.append(file_path)
            
        for img in image_paths:
            image = preprocess(Image.open(img)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image).cpu().numpy()[0]
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        
        return labels,embeddings, image_paths
    
    def show_clusters(self, image_paths, labels,embeddings, num_clusters):
        reducer = umap.UMAP(random_state=42)
        reduced = reducer.fit_transform(embeddings)

        plt.figure(figsize=(10, 6))
        for i in range(num_clusters):
            cluster_points = reduced[labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}", s=100)

        for i, path in enumerate(image_paths):
            plt.text(reduced[i, 0], reduced[i, 1], os.path.basename(path), fontsize=8)

        plt.title("K-Means Clustering pentru Paginile HTML")
        plt.legend()
        plt.show()

    def ImageClustering(self, html_path, num_clusters):
        html_ss = "html_ss/"+(html_path.split("/")[-1])
        A = []
        for file in os.listdir(html_ss):
            fn_name = os.path.basename(file).split("'\'")[-1]
            A.append(fn_name.split(".png")[0])

        labels,embeddings, image_paths = self.clip_clustering(A,html_ss,num_clusters)
        #self.optimal_clusters(embeddings, max_clusters=10)
        clusters = self.assign_clusters(A, labels)
        return clusters

    def assign_clusters(self, vectors, labels):
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(vectors[i])
        return clusters



