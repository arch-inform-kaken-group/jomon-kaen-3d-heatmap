import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sentence_transformers import SentenceTransformer
import umap
import japanize_matplotlib # Automatically handles Japanese font display

# --- 1. Define Simplified Labels ---
TARGET_LABELS_JP = ["面白い", "美しい", "不思議", "怖い", "何も感じない"]
TARGET_LABELS_EN = ["Interesting", "Beautiful", "Strange", "Scary", "Feel nothing"]

# Combine both lists for embedding
# all_labels = TARGET_LABELS_JP + TARGET_LABELS_EN
all_labels = TARGET_LABELS_JP

# --- 2. Create a Color Map ---
# Assign a unique color to each semantic concept.
# The color for "面白い" will be the same as for "Interesting", and so on.
num_unique_labels = len(TARGET_LABELS_EN)
# Use a color palette designed for distinct categories
palette = plt.cm.get_cmap('jet', num_unique_labels)
# Create the color list: the first 5 colors are repeated for the next 5 labels
# colors = [palette(i) for i in range(num_unique_labels)] * 2
colors = [palette(i) for i in range(num_unique_labels)]

# --- 3. Embed Text Labels into Vectors ---
print("Loading multilingual model and embedding labels...")
# This model is specifically trained to understand multiple languages
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
embeddings = model.encode(all_labels)
print(f"Successfully created {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")

# --- 4. Reduce Embedding Dimensions with UMAP ---
print("Reducing embedding dimensions with UMAP...")

# Reduce to 2 dimensions
reducer_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.0)
embedding_2d = reducer_2d.fit_transform(embeddings)

# Reduce to 3 dimensions
reducer_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=5, min_dist=0.0)
embedding_3d = reducer_3d.fit_transform(embeddings)
print("Dimensionality reduction complete.")

# --- 5. Create and Display Plots ---
print("Generating plots...")
fig = plt.figure(figsize=(20, 9))
fig.suptitle('2D & 3D Semantic Embeddings of Simplified Labels', fontsize=18)

# --- 2D Plot ---
ax_2d = fig.add_subplot(1, 2, 1)
ax_2d.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, s=150, alpha=0.9)
ax_2d.set_title('2D UMAP Projection', fontsize=14)
ax_2d.set_xlabel('UMAP Dimension 1')
ax_2d.set_ylabel('UMAP Dimension 2')
ax_2d.grid(True, linestyle='--', alpha=0.6)

# Annotate each point with its label
for i, label in enumerate(all_labels):
    ax_2d.text(embedding_2d[i, 0] + 0.05, embedding_2d[i, 1] + 0.05, label, fontsize=12)

# --- 3D Plot ---
ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], c=colors, s=150, alpha=0.9)
ax_3d.set_title('3D UMAP Projection', fontsize=14)
ax_3d.set_xlabel('UMAP Dimension 1')
ax_3d.set_ylabel('UMAP Dimension 2')
ax_3d.set_zlabel('UMAP Dimension 3')

# Annotate each point in the 3D plot
for i, label in enumerate(all_labels):
     ax_3d.text(embedding_3d[i, 0], embedding_3d[i, 1], embedding_3d[i, 2], label, fontsize=12)


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print("Script finished.")