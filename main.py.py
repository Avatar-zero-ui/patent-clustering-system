# ============================================
# DMA PROJECT - GUI VERSION (ADVANCED)
# Agglomerative Clustering with Tkinter UI
# ============================================

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sns.set()

# Global variables
df = None
X_scaled = None
clusters = None

# ============================================
# LOAD DATA
# ============================================

def load_data():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        messagebox.showinfo("Success", "Dataset Loaded Successfully!")

# ============================================
# PREPROCESS
# ============================================

def preprocess():
    global X_scaled
    if df is None:
        messagebox.showerror("Error", "Load dataset first!")
        return

    X = df.iloc[:, 1:].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    messagebox.showinfo("Success", "Data Standardized!")

# ============================================
# HEATMAP
# ============================================

def show_heatmap():
    if df is None:
        messagebox.showerror("Error", "Load dataset first!")
        return

    plt.figure()
    sns.heatmap(df.iloc[:, 1:], annot=True)
    plt.title("Heatmap")
    plt.show()

# ============================================
# DENDROGRAM
# ============================================

def show_dendrogram():
    if X_scaled is None:
        messagebox.showerror("Error", "Preprocess data first!")
        return

    linked = linkage(X_scaled, method='ward')

    plt.figure()
    dendrogram(linked, labels=list(df['Patent_ID']))
    plt.title("Dendrogram")
    plt.show()

# ============================================
# CLUSTERING
# ============================================

def run_clustering():
    global clusters

    if X_scaled is None:
        messagebox.showerror("Error", "Preprocess data first!")
        return

    model = AgglomerativeClustering(n_clusters=3)
    clusters = model.fit_predict(X_scaled)

    df['Cluster'] = clusters

    score = silhouette_score(X_scaled, clusters)

    messagebox.showinfo("Done", f"Clustering Completed!\nSilhouette Score: {score:.3f}")

# ============================================
# PCA PLOT
# ============================================

def show_pca():
    if clusters is None:
        messagebox.showerror("Error", "Run clustering first!")
        return

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)

    for i, txt in enumerate(df['Patent_ID']):
        plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]))

    plt.title("PCA Visualization")
    plt.show()

# ============================================
# CLUSTER DISTRIBUTION
# ============================================

def show_distribution():
    if 'Cluster' not in df:
        messagebox.showerror("Error", "Run clustering first!")
        return

    sns.countplot(x='Cluster', data=df)
    plt.title("Cluster Distribution")
    plt.show()

# ============================================
# EXPORT RESULTS
# ============================================

def export_data():
    if 'Cluster' not in df:
        messagebox.showerror("Error", "Run clustering first!")
        return

    df.to_csv("clustered_output.csv", index=False)
    messagebox.showinfo("Saved", "Results exported as clustered_output.csv")

# ============================================
# GUI DESIGN
# ============================================

root = tk.Tk()
root.title("DMA Project - Patent Clustering System")
root.geometry("500x500")

title = tk.Label(root, text="Patent Clustering System", font=("Arial", 18, "bold"))
title.pack(pady=20)

tk.Button(root, text="Load Dataset", width=25, command=load_data).pack(pady=5)
tk.Button(root, text="Preprocess Data", width=25, command=preprocess).pack(pady=5)
tk.Button(root, text="Show Heatmap", width=25, command=show_heatmap).pack(pady=5)
tk.Button(root, text="Show Dendrogram", width=25, command=show_dendrogram).pack(pady=5)
tk.Button(root, text="Run Clustering", width=25, command=run_clustering).pack(pady=5)
tk.Button(root, text="Show PCA Plot", width=25, command=show_pca).pack(pady=5)
tk.Button(root, text="Cluster Distribution", width=25, command=show_distribution).pack(pady=5)
tk.Button(root, text="Export Results", width=25, command=export_data).pack(pady=5)

root.mainloop()
