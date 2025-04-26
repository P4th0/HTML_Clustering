import tkinter as tk
from tkinter import ttk
import os
from PIL import Image, ImageDraw
import math
from html_DBSCAN import Clustering
from html_imgcmp import ImageComp
from html_imgcmp_DBSCAN import ImageCompplusDBSCAN

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HTML Clustering")
        self.root.geometry("1000x600")
        self.root.configure(bg="white")
        self.tier_var = tk.StringVar(value="tier1")
        self.main_frame = tk.Frame(self.root, bg="white")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.clusters = {}   
        self.setup_ui()
        self.last_pressed = None

    def setup_ui(self):
        ctrl_frame = tk.Frame(self.main_frame, bg="white")
        ctrl_frame.pack(fill=tk.X, pady=10)
        label = tk.Label(ctrl_frame, text="Alege tier:", bg="white")
        label.pack(side=tk.LEFT, padx=(0, 10))
        tier_options = ["tier1", "tier2", "tier3", "tier4"]
        dropdown = ttk.Combobox(ctrl_frame, textvariable=self.tier_var, values=tier_options, state="readonly", width=15)
        dropdown.pack(side=tk.LEFT, padx=(0, 20))

        b1 = tk.Button(ctrl_frame, text="Clasifică (CLIP)",command=self.clasiffy_imagecomp)
        b1.pack(side=tk.LEFT, padx=(10, 0))
        self.cluster_label = tk.Label(ctrl_frame, text="Nr. Clustere (CLIP):", bg="white")
        self.cluster_label.pack(side=tk.LEFT, padx=(0, 5))
        self.cluster_entry = tk.Entry(ctrl_frame, width=5)
        self.cluster_entry.insert(0, "4") 
        self.cluster_entry.pack(side=tk.LEFT)

        button = tk.Button(ctrl_frame, text="Clasifică(DBSCAN)", command=self.classify_selected)
        button.pack(side=tk.LEFT)
        b2 = tk.Button(ctrl_frame, text="Clasifica(DBSCAN + CLIP)",command = self.classify_selected_CDB)
        b2.pack(side=tk.LEFT, padx=(10, 0))
        b3 = tk.Button(ctrl_frame, text="Salveaza Grid cu clustere curente",command=self.export_grid_clusters)
        b3.pack(side=tk.RIGHT, padx=(10, 0))
        b4 = tk.Button(ctrl_frame, text="ss htmls",command=self.ss_htmls_gui)
        b4.pack(side=tk.RIGHT, padx=(10, 0))
        self.result_frame = tk.Frame(self.main_frame, bg="gray")
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        y_scrollbar = tk.Scrollbar(self.result_frame)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        x_scrollbar = tk.Scrollbar(self.result_frame, orient=tk.HORIZONTAL)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.result_text = tk.Text(self.result_frame, wrap=tk.WORD, bg="lightgray", fg="black")
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.config(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        y_scrollbar.config(command=self.result_text.yview)
        x_scrollbar.config(command=self.result_text.xview)
        self.result_text.config(state=tk.NORMAL)
        
        self.result_text.bind("<Key>", lambda e: "break")
        self.result_text.bind("<Control-c>", lambda e: None)
        self.result_text.bind("<Button-1>", lambda e: None)
        self.result_text.bind("<MouseWheel>", lambda e: None)

    def classify_selected(self):      #clasificare folosind DBSCAN
        self.last_pressed = "DBSCAN"
        self.result_text.delete(1.0, tk.END)
        clustering = Clustering()
        tier = self.tier_var.get()
        folder_path = f"clones/{tier}"
        self.result_text.insert(tk.END, f"Clusterizare pentru {tier}:\n")
        
        try: 
            self.clusters = clustering.classify(folder_path)
            for label, paths in self.clusters.items():
                self.result_text.insert(tk.END, f"\nCluster {label}:\n")
                file_list = [os.path.basename(path) for path in paths]
                for file_name in file_list:
                    self.result_text.insert(tk.END, f"    • {file_name}\n")
            
        except Exception as e:
            self.result_text.insert(tk.END, f"Eroare: {str(e)}")

        self.result_text.mark_set("insert", "1.0")
        self.result_text.focus_set()
    
    def clasiffy_imagecomp(self):  #clasificare folosind CLIP (cu nr de clustere de la tastatura)
        self.last_pressed = "CLIP"
        try:
            num_clusters = int(self.cluster_entry.get())
        except ValueError:
            num_clusters = 4  
        self.result_text.delete(1.0, tk.END)
        image_comp = ImageComp()
        tier = self.tier_var.get()
        folder_path = f"clones/{tier}"
        self.result_text.insert(tk.END, f"Clasificare pentru {tier}:\n")
        
        try: 
            self.clusters = image_comp.ImageClustering(folder_path,num_clusters)
            for label, paths in self.clusters.items():
                self.result_text.insert(tk.END, f"\nCluster {label}:\n")
                file_list = [os.path.basename(path) for path in paths]
                for file_name in file_list:
                    self.result_text.insert(tk.END, f"    • {file_name}\n")
        except Exception as e:
            self.result_text.insert(tk.END, f"Eroare: {str(e)}")

        self.result_text.mark_set("insert", "1.0")
        self.result_text.focus_set()
    
    def classify_selected_CDB(self):      #clasificare folosind DBSCAN+CLIP
        self.last_pressed = "DBSCAN+CLIP"
        self.result_text.delete(1.0, tk.END)
        imgcmp = ImageCompplusDBSCAN()
        tier = self.tier_var.get()
        folder_path = f"clones/{tier}"
        self.result_text.insert(tk.END, f"Clusterizare pentru {tier}:\n")
        
        try: 
            self.clusters = imgcmp.ImageClustering(folder_path)
            for label, paths in self.clusters.items():
                self.result_text.insert(tk.END, f"\nCluster {label}:\n")
                file_list = [os.path.basename(path) for path in paths]
                for file_name in file_list:
                    self.result_text.insert(tk.END, f"    • {file_name}\n")
            
        except Exception as e:
            self.result_text.insert(tk.END, f"Eroare: {str(e)}")

        self.result_text.mark_set("insert", "1.0")
        self.result_text.focus_set()


    def export_grid_clusters(self): #export grid clustere curente 
        if not self.clusters:
            print("Trebuie sa rulezi mai întâi clasificarea.")
            return
        
        tier = self.tier_var.get()
        ss_folder = f"html_ss/{tier}"
        output_folder = "grid_clustere"+"/"+tier+"/"+self.last_pressed
        os.makedirs(output_folder, exist_ok=True)
        for label, paths in self.clusters.items():
            images = []
            for path in paths:
                filename = os.path.basename(path)
                img_path = os.path.join(ss_folder, filename + ".png")
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img = img.resize((200, 150))
                        images.append((filename, img))
                    except:
                        print(f"Eroare la deschiderea imaginii {img_path}.")
                        continue

            if not images:
                print(f"Nu am găsit imagini pentru cluster {label}.")
                continue

            cols = 5
            rows = math.ceil(len(images) / cols)
            img_width, img_height = 200, 150
            padding = 20
            text_height = 30

            grid_img = Image.new("RGB", (
                cols * (img_width + padding) + padding,
                rows * (img_height + text_height + padding) + padding
            ), "white")

            draw = ImageDraw.Draw(grid_img)

            for idx, (filename, img) in enumerate(images):
                row = idx // cols
                col = idx % cols
                x = padding + col * (img_width + padding)
                y = padding + row * (img_height + text_height + padding)

                grid_img.paste(img, (x, y))
                draw.text((x, y + img_height + 5), filename, fill="black")

            grid_path = os.path.join(output_folder, f"cluster_{label}.png")
            grid_img.save(grid_path)
            print(f"Salvat: {grid_path}")
    
    def ss_htmls_gui(self): #ss html clones (takes a while)
        img_cmp = ImageComp()
        for i in range (1,5):
            path = f"clones/tier{i}"
            img_cmp.ss_all_htmls(path,i)
            print(f"Screenshot-uri salvate pentru tier {i}.")

    
    def run(self):
        self.root.mainloop()