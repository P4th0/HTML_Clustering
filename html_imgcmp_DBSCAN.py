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
import matplotlib.pyplot as plt
import umap
from html_DBSCAN import Clustering


class ImageCompplusDBSCAN:
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
    
    def clip_clustering(self, A,folder_path): 
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
        return embeddings, image_paths
    
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

    def assign_clusters(self, vectors, labels):
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(vectors[i])
        return clusters
        
    def ImageClustering(self, html_path): #clustetring trecand prin fiecare label
        clustering = Clustering()
        clusters = clustering.classify(html_path)
        html_ss = "html_ss/"+(html_path.split("/")[-1])
        labels = [int(label) for label in clusters.keys()]
        new_labels =[]
        new_outliers =[]

        for l in labels:
            print(f"Cluster {l}:")
            A = []
            for label, paths in clusters.items():
                if l == int(label) and l != -1:
                    for fn in paths:
                        fn_name = os.path.basename(fn).split("'\'")[-1]
                        A.append(fn_name)
                    embeddings,image_paths = self.clip_clustering(A,html_ss)
                    outliers = self.check_clusters(embeddings,image_paths)
                    if len(outliers) > 1:
                        new_labels.append(outliers)
                    elif len(outliers) == 1:
                        new_outliers.append(outliers[0])

        curr_label = np.max(labels)+1

        new_labels = [
            [
                filename.replace("html_ss/", "clones/")[:-4] if filename.endswith('.png') else filename.replace("html_ss/", "clones/")
                for filename in group
            ]
            for group in new_labels
        ]

        new_outliers = [
            filename.replace("html_ss/", "clones/")[:-4] if filename.endswith('.png') else filename.replace("html_ss/", "clones/")
            for filename in new_outliers
        ]

    
        print(f" new labels: {new_labels}")
        print(f"new outliers {new_outliers}")
        
        if new_labels:
            for group in new_labels:
                for g in group:
                    for label, paths in clusters.items():
                        if g in paths:
                            l = int(label)
                            break
                    clusters[l].remove(g)
                clusters[curr_label] = group
                curr_label += 1

        if new_outliers:
            for label, paths in clusters.items():
                if new_outliers[0] in paths:
                    l = int(label)
                    break
            
            clusters[l].remove(new_outliers[0])
            clusters[-1].append(new_outliers[0])

        return clusters
    
    def check_clusters(self,embeddings,image_paths): #verifica fiecare cluster, un restart la fiecare outlier detectat pentru recalculare medie embeddings
        confidence = 0.87     
        inliers = list(image_paths)
        outliers = []
        idx_to_inliers=[]
        idx_to_outliers=[]
        current_indices = list(range(len(embeddings)))
        similarity_scores = []

        while True:
            restart_loop = False
            mean_embedding = np.mean(embeddings[current_indices], axis=0)
            mean_embedding /= np.linalg.norm(mean_embedding)
            for idx in current_indices.copy():
                emb = embeddings[idx]
                norm_embedding = emb / np.linalg.norm(emb)
                similarity = np.dot(norm_embedding, mean_embedding)
                similarity_scores.append(similarity)
                if similarity < confidence:
                    current_indices.remove(idx)
                    outliers.append(image_paths[idx])
                    print(f"Imagine outlier detectata: {os.path.basename(image_paths[idx])}, similaritate: {similarity:.4f}")
                    idx_to_outliers.append(idx)
                    restart_loop = True
                    break
            
            if not restart_loop:
                break
          
        idx_to_inliers = list(set(range(len(embeddings))) - set(idx_to_outliers))
        inliers = [image_paths[i] for i in idx_to_inliers]
        avg_similarity = np.mean(similarity_scores)
        min_similarity = np.min(similarity_scores)
        max_similarity = np.max(similarity_scores)
        
        print(f"Statistici cluster:")
        print(f"  - Similaritate medie: {avg_similarity:.4f}")
        print(f"  - Similaritate minima: {min_similarity:.4f}")
        print(f"  - Similaritate maxima: {max_similarity:.4f}")
        print(f"  - Imagini inlier: {len(inliers)}")
        print(f"  - Imagini outlier: {len(outliers)}")
        return outliers
       

# def main():
#     imageComp = ImageCompplusDBSCAN()
#     imageComp.ImageClustering("clones/tier3")
    
# if __name__ == "__main__":
#     main()

