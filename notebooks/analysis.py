import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from spark.pca_cluster import run_clustering #python3 -m notebooks.analysis inside sd_clustering folder to run this code
from src.generate_images import generate_images
from src.extract_embeddings import extract_embeddings
from paths.paths import results_dir, prompts_file
from matplotlib.lines import Line2D
from matplotlib.patches import Patch 

spark = SparkSession.builder.appName("SDClustScript").getOrCreate()

generate_images()
extract_embeddings()
run_clustering(spark)

df_clustered = pd.read_parquet(str(results_dir / "clustered"))
df_furthest = pd.read_parquet(str(results_dir / "furthest"))

print(df_furthest)

with prompts_file.open("r") as f:
    prompts = [line.strip() for line in f if line.strip()]

marker_map = {
    prompts[0]: 'o',   
    prompts[1]: 's',   
    prompts[2]: '^'    
}

clusters = sorted(df_clustered['cluster'].unique())
cmap = plt.cm.get_cmap('Set1', len(clusters))

clusters = sorted(df_clustered['cluster'].unique())
cmap = plt.cm.get_cmap('Set1', len(clusters))

plt.figure(figsize=(12,8))
for prompt in prompts:
    subset = df_clustered[df_clustered['prompt'] == prompt]
    plt.scatter(
        subset['pca1'], subset['pca2'],
        c=subset['cluster'],
        cmap=cmap,
        vmin=clusters[0], vmax=clusters[-1],
        marker=marker_map[prompt],
        s=30, alpha=0.7,
        edgecolor='k', linewidth=0.5
    )

plt.scatter(
    df_furthest['pca1'], df_furthest['pca2'],
    marker='D', s=200,
    facecolors='none', edgecolors='black', linewidths=1.5,
    label='Furthest in cluster'
)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Embeddings: Shapes by Prompt, Color by Cluster')

cluster_patches = [Patch(facecolor=cmap(i), edgecolor='k', label=f'Cluster {i}') for i in clusters]
leg1 = plt.legend(handles=cluster_patches, title='Cluster', loc='upper right', bbox_to_anchor=(1, 1))
plt.gca().add_artist(leg1)

prompt_handles = [Line2D([0], [0], marker=marker_map[p], color='black', linestyle='None', markersize=8, label=p) for p in prompts]
prompt_handles.append(Line2D([0], [0], marker='D', color='black', linestyle='None', markersize=12, label='Furthest in cluster'))
plt.legend(handles=prompt_handles, title='Prompt / Furthest', loc='lower right', bbox_to_anchor=(1, -0.3))

plt.tight_layout()
plt.show()











