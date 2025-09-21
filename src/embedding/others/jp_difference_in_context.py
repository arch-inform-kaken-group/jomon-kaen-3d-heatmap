import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sentence_transformers import SentenceTransformer
import umap
import japanize_matplotlib  # Automatically handles Japanese font display

# --- 1. Define Labels and Example Transcripts ---
TARGET_LABELS_JP = ["面白い", "美しい", "不思議", "怖い", "何も感じない"]

EXAMPLE_TRANSCRIPTS = [
    # Example 1
    "人型の土偶で、顔は気持ち悪くて、頭がないみたいな感じで、\n不気味である。黒い似ているものがあって、\nとても不気味な土偶だった。",
    # Example 2
    "面白い。個人的に一番好きな形だと感じた。\n回りのうずまきなどの模様がくっきりとのこっていて面白いと感じた。\n上の方についている突起の部分が危ないと感じた。\n中にひび割れのようなものがあって面白い。",
    # Example 3
    "造形が綺麗で壮大、壮厳な感じがしました。\n大きくて圧倒されるような感覚になりました。\nあと美しさも感じました。"
]
EXAMPLE_NAMES = ["Example 1 (不気味)", "Example 2 (面白い)", "Example 3 (美しい)"]

# Combine labels and transcripts into one list for embedding
all_texts = TARGET_LABELS_JP + EXAMPLE_TRANSCRIPTS

# --- 2. Create Color and Marker Maps ---
num_labels = len(TARGET_LABELS_JP)
num_examples = len(EXAMPLE_TRANSCRIPTS)

palette = plt.cm.get_cmap('jet', num_labels)
label_colors = [palette(i) for i in range(num_labels)]
example_markers = ['X', '*', 'P']

# --- 3. Embed All Text into Vectors ---
print("Loading multilingual model and embedding all texts...")
model = SentenceTransformer(
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
all_embeddings = model.encode(all_texts)
print(f"Successfully created {all_embeddings.shape[0]} embeddings.")

# --- 4. Reduce Embedding Dimensions with UMAP ---
print("Reducing embedding dimensions with UMAP...")
reducer_2d = umap.UMAP(n_components=2,
                       random_state=64,
                       n_neighbors=len(all_texts) - 1,
                       min_dist=0.0,
                       n_jobs=1)
embedding_2d = reducer_2d.fit_transform(all_embeddings)

reducer_3d = umap.UMAP(n_components=3,
                       random_state=64,
                       n_neighbors=len(all_texts) - 1,
                       min_dist=0.0,
                       n_jobs=1)
embedding_3d = reducer_3d.fit_transform(all_embeddings)
print("Dimensionality reduction complete.")

label_embedding_2d = embedding_2d[:num_labels]
example_embedding_2d = embedding_2d[num_labels:]
label_embedding_3d = embedding_3d[:num_labels]
example_embedding_3d = embedding_3d[num_labels:]

# --- 5. Create and Display Plots ---
print("Generating plots...")
fig = plt.figure(figsize=(24, 15))
fig.suptitle('Semantic Embeddings of Japanese Labels and Example Transcripts',
             fontsize=20,
             y=0.98)  # Adjust title y-position

# --- 2D Plot (Square) ---
ax_2d = fig.add_subplot(1, 3, 1)
ax_2d.scatter(label_embedding_2d[:, 0],
              label_embedding_2d[:, 1],
              c=label_colors,
              s=150,
              alpha=0.9)
for i in range(num_examples):
    ax_2d.scatter(example_embedding_2d[i, 0],
                  example_embedding_2d[i, 1],
                  c='black',
                  marker=example_markers[i],
                  s=300,
                  alpha=1.0,
                  label=EXAMPLE_NAMES[i],
                  edgecolors='white',
                  linewidths=1)

ax_2d.set_title('2D UMAP Projection', fontsize=16)
ax_2d.set_xlabel('UMAP Dimension 1')
ax_2d.set_ylabel('UMAP Dimension 2')
ax_2d.grid(True, linestyle='--', alpha=0.6)

# ** NEW CODE TO MAKE THE PLOT SQUARE **
# Get the limits of the axes
x_limits = ax_2d.get_xlim()
y_limits = ax_2d.get_ylim()
# Calculate the range of the axes
x_range = abs(x_limits[1] - x_limits[0])
y_range = abs(y_limits[1] - y_limits[0])
# Find the maximum range
max_range = max(x_range, y_range)
# Calculate the center of the axes
x_center = np.mean(x_limits)
y_center = np.mean(y_limits)
# Set new square limits
ax_2d.set_xlim(x_center - max_range * 0.55, x_center + max_range * 0.55)
ax_2d.set_ylim(y_center - max_range * 0.55, y_center + max_range * 0.55)
ax_2d.set_aspect('equal', adjustable='box')

for i, label in enumerate(TARGET_LABELS_JP):
    ax_2d.text(label_embedding_2d[i, 0] + 0.05,
               label_embedding_2d[i, 1] + 0.05,
               label,
               fontsize=14,
               weight='bold')

# --- 3D Plot (View 1) ---
ax_3d_view1 = fig.add_subplot(1, 3, 2, projection='3d')
ax_3d_view1.scatter(label_embedding_3d[:, 0],
                    label_embedding_3d[:, 1],
                    label_embedding_3d[:, 2],
                    c=label_colors,
                    s=150,
                    alpha=0.9)
for i in range(num_examples):
    ax_3d_view1.scatter(example_embedding_3d[i, 0],
                        example_embedding_3d[i, 1],
                        example_embedding_3d[i, 2],
                        c='black',
                        marker=example_markers[i],
                        s=300,
                        alpha=1.0,
                        label=EXAMPLE_NAMES[i],
                        edgecolors='white',
                        linewidths=1)

ax_3d_view1.set_title('3D UMAP Projection (View 1)', fontsize=16)
ax_3d_view1.set_xlabel('UMAP Dim 1')
ax_3d_view1.set_ylabel('UMAP Dim 2')
ax_3d_view1.set_zlabel('UMAP Dim 3')
ax_3d_view1.view_init(elev=20, azim=-65)
for i, label in enumerate(TARGET_LABELS_JP):
    ax_3d_view1.text(label_embedding_3d[i, 0],
                     label_embedding_3d[i, 1],
                     label_embedding_3d[i, 2],
                     label,
                     fontsize=14,
                     weight='bold')

# --- 3D Plot (View 2) ---
ax_3d_view2 = fig.add_subplot(1, 3, 3, projection='3d')
ax_3d_view2.scatter(label_embedding_3d[:, 0],
                    label_embedding_3d[:, 1],
                    label_embedding_3d[:, 2],
                    c=label_colors,
                    s=150,
                    alpha=0.9)
for i in range(num_examples):
    ax_3d_view2.scatter(example_embedding_3d[i, 0],
                        example_embedding_3d[i, 1],
                        example_embedding_3d[i, 2],
                        c='black',
                        marker=example_markers[i],
                        s=300,
                        alpha=1.0,
                        label=EXAMPLE_NAMES[i],
                        edgecolors='white',
                        linewidths=1)

ax_3d_view2.set_title('3D UMAP Projection (View 2)', fontsize=16)
ax_3d_view2.set_xlabel('UMAP Dim 1')
ax_3d_view2.set_ylabel('UMAP Dim 2')
ax_3d_view2.set_zlabel('UMAP Dim 3')
ax_3d_view2.view_init(elev=20, azim=-25)
# REMOVED ax.legend() from here

for i, label in enumerate(TARGET_LABELS_JP):
    ax_3d_view2.text(label_embedding_3d[i, 0],
                     label_embedding_3d[i, 1],
                     label_embedding_3d[i, 2],
                     label,
                     fontsize=14,
                     weight='bold')

# --- 6. Create and Position the Figure-level Legend ---
# Get handles and labels from one of the plots
handles, labels = ax_2d.get_legend_handles_labels()
# Create the legend for the entire figure, positioned at the top right
fig.legend(handles,
           labels,
           loc='upper right',
           bbox_to_anchor=(1.0, 1.0),
           title="Example Transcripts",
           fontsize=11)

# --- 7. Display Example Sentences in a Table-like Format ---
plt.subplots_adjust(bottom=0.25)

text1 = f"Example 1 (不気味)\n\n{EXAMPLE_TRANSCRIPTS[0]}"
text2 = f"Example 2 (面白い)\n\n{EXAMPLE_TRANSCRIPTS[1]}"
text3 = f"Example 3 (美しい)\n\n{EXAMPLE_TRANSCRIPTS[2]}"

fig.text(0.05,
         0.18,
         text1,
         ha='left',
         va='top',
         fontsize=10,
         wrap=True,
         bbox=dict(boxstyle='round,pad=0.5',
                   fc='aliceblue',
                   ec='grey',
                   lw=1,
                   alpha=0.5))
fig.text(0.38,
         0.18,
         text2,
         ha='left',
         va='top',
         fontsize=10,
         wrap=True,
         bbox=dict(boxstyle='round,pad=0.5',
                   fc='aliceblue',
                   ec='grey',
                   lw=1,
                   alpha=0.5))
fig.text(0.71,
         0.18,
         text3,
         ha='left',
         va='top',
         fontsize=10,
         wrap=True,
         bbox=dict(boxstyle='round,pad=0.5',
                   fc='aliceblue',
                   ec='grey',
                   lw=1,
                   alpha=0.5))

plt.show()

print("Script finished.")
