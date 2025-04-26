# HTML Clustering - Classifying HTML Pages Using DBSCAN and CLIP

## Description

A Python application for **clustering HTML pages** based on their **layout**, using three different methods:
- **CLIP** (the weakest, based only on image similarity)
- **DBSCAN** (clustering based on distance between points)
- **DBSCAN + CLIP** (the strongest and most accurate method)

The graphical interface is built simply and intuitively using **Tkinter**.

---

## Project Structure

- **4 tiers** of HTML pages, each classified separately.
- **3 available clustering methods**:
  1. **DBSCAN**
  2. **CLIP** (layout-based)
  3. **DBSCAN + CLIP** (refining clusters with additional filtering)

---

## Details About the Methods

### 1. Hierarchical Clustering with **DBSCAN**
- **Epsilon** (`eps`) was chosen by:
  - Manually inspecting the *k-th nearest neighbor distance* graph and identifying a sudden jump.
  - Final value: **eps = 1**
- **min_samples**:
  - Set to **2**, because two similar HTML pages are considered enough to form a cluster (based on the user perspective, as required).

### 2. Layout Similarity Clustering Using **CLIP**
- **A. CLIP only**:
  - Generate **embeddings** for the page screenshots using the CLIP model.
  - Apply **K-Means** clustering on the embeddings.
- **B. DBSCAN + CLIP**:
  - First, apply **DBSCAN** for a rough clustering.
  - Then, for each cluster, verify the similarity between each screenshot and the **average layout** using a custom `check_clusters()` function.
  - This greatly improves precision and filters out outliers.

---

## Additional Features

- **A. Automatic screenshot generation** for all HTML pages using **Selenium** and **ChromeDriver**.
- **B. Exporting clusters into **grid images** for easy visual validation.

---

## References

- [DBSCAN Clustering Algorithm - Datacamp](https://www.datacamp.com/tutorial/dbscan-clustering-algorithm)
- [How to Choose Epsilon and MinPts for DBSCAN](https://www.sefidian.com/2022/12/18/how-to-determine-epsilon-and-minpts-parameters-of-dbscan-clustering/)
- [How to Take Website Screenshots in Python](https://screenshotone.com/blog/how-to-take-website-screenshots-in-python/)
- [OpenAI CLIP GitHub Repository](https://github.com/openai/CLIP)

---

## How to Run the Application

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
