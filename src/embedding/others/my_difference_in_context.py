import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sentence_transformers import SentenceTransformer
import umap
import numpy as np

# --- 1. Define Labels and Example Transcripts ---
TARGET_LABELS_EN = ["Interesting", "Beautiful", "Strange", "Scary", "Feel nothing"]

EXAMPLE_TRANSCRIPTS = [
    # Example 1 (Scary)
    "At first, when I saw this, I saw this from the back at first. \n"
    "It looks kind of pretty from the back, but when I look in front \n"
    "It looks kind of creepy because it's a shape of a human, \n"
    "and I wouldn't have this in my house. Honestly, because it just looks kind of creepy. \n"
    "Yeah, I guess this is all right. And it is pretty. I mean, it is creepy, honestly. \n"
    "Yeah, it's creepy.",
    # Example 2 (Interesting)
    "I think it looks really cool and I like how asymmetrical it is. \n"
    "And I'm very curious about what it's made out of. \n"
    "Maybe the roughness of the pottery, I feel like it really gives that handmade feel. \n"
    "It just looks really interesting to me. \n"
    "It's really cool. Unfortunately, I really want to touch it, but I can't, \n"
    "because I'm very curious about how it feels, because the material\n"
    "seems to be very rough. "
    "This is really cool.",
    # Example 3 (Beautiful)
    "This one I think is more beautiful than others. \n"
    "It's unique and the color is so beautiful. \n"
    "Something unique. I've never seen something like this. \n"
    "It's like a crystal, is it? "
    "I like the shape. \n"
    "And it's shining like a crystal. "
    "Wow."
]
EXAMPLE_NAMES = ["Example 1 (Scary)", "Example 2 (Interesting)", "Example 3 (Beautiful)"]

# Combine labels and transcripts into one list for embedding
all_texts = TARGET_LABELS_EN + EXAMPLE_TRANSCRIPTS

# --- 2. Create Color and Marker Maps ---
num_labels = len(TARGET_LABELS_EN)
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
# UMAP parameters for reproducible results
umap_params = {
    'n_components': 2,
    'random_state': 64,
    'n_neighbors': len(all_texts) - 1,
    'min_dist': 0.0,
    'n_jobs': 1
}

reducer_2d = umap.UMAP(**umap_params)
embedding_2d = reducer_2d.fit_transform(all_embeddings)

umap_params['n_components'] = 3
reducer_3d = umap.UMAP(**umap_params)
embedding_3d = reducer_3d.fit_transform(all_embeddings)
print("Dimensionality reduction complete.")

# Separate the embeddings back into labels and examples
label_embedding_2d = embedding_2d[:num_labels]
example_embedding_2d = embedding_2d[num_labels:]
label_embedding_3d = embedding_3d[:num_labels]
example_embedding_3d = embedding_3d[num_labels:]

# --- 5. Create and Display Plots ---
print("Generating plots...")
fig = plt.figure(figsize=(24, 15))
fig.suptitle('Semantic Embeddings of English Labels and Example Transcripts',
             fontsize=20,
             y=0.98)

# --- 2D Plot (Square) ---
ax_2d = fig.add_subplot(1, 3, 1)
# Plot label points
ax_2d.scatter(label_embedding_2d[:, 0],
              label_embedding_2d[:, 1],
              c=label_colors,
              s=150,
              alpha=0.9)
# Plot example points
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

# Force the plot to be square
x_limits = ax_2d.get_xlim()
y_limits = ax_2d.get_ylim()
x_range = abs(x_limits[1] - x_limits[0])
y_range = abs(y_limits[1] - y_limits[0])
max_range = max(x_range, y_range)
x_center = np.mean(x_limits)
y_center = np.mean(y_limits)
ax_2d.set_xlim(x_center - max_range * 0.55, x_center + max_range * 0.55)
ax_2d.set_ylim(y_center - max_range * 0.55, y_center + max_range * 0.55)
ax_2d.set_aspect('equal', adjustable='box')

# Add text labels to the 2D plot
for i, label in enumerate(TARGET_LABELS_EN):
    ax_2d.text(label_embedding_2d[i, 0] + 0.05,
               label_embedding_2d[i, 1] + 0.05,
               label,
               fontsize=9,
            )

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
for i, label in enumerate(TARGET_LABELS_EN):
    ax_3d_view1.text(label_embedding_3d[i, 0],
                     label_embedding_3d[i, 1],
                     label_embedding_3d[i, 2],
                     label,
                     fontsize=9,
                    )

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
for i, label in enumerate(TARGET_LABELS_EN):
    ax_3d_view2.text(label_embedding_3d[i, 0],
                     label_embedding_3d[i, 1],
                     label_embedding_3d[i, 2],
                     label,
                     fontsize=9,
                  )

# --- 6. Create and Position the Figure-level Legend ---
handles, labels = ax_2d.get_legend_handles_labels()
for handle in handles:
    handle.set_sizes([150.0]) 
fig.legend(handles,
           labels,
           loc='upper right',
           bbox_to_anchor=(1.0, 1.0),
           title="Example Transcripts",
           fontsize=8,
           title_fontsize=9)

# --- 7. Display Example Sentences in Text Boxes ---
plt.subplots_adjust(bottom=0.25, top=0.9)

text_box_props = dict(boxstyle='round,pad=0.5', fc='aliceblue', ec='grey', lw=1, alpha=0.8)

text1 = f"Example 1 (Scary)\n\n{EXAMPLE_TRANSCRIPTS[0]}"
text2 = f"Example 2 (Interesting)\n\n{EXAMPLE_TRANSCRIPTS[1]}"
text3 = f"Example 3 (Beautiful)\n\n{EXAMPLE_TRANSCRIPTS[2]}"

fig.text(0.05, 0.22, text1, ha='left', va='top', fontsize=8, wrap=True, bbox=text_box_props)
fig.text(0.38, 0.22, text2, ha='left', va='top', fontsize=8, wrap=True, bbox=text_box_props)
fig.text(0.71, 0.22, text3, ha='left', va='top', fontsize=8, wrap=True, bbox=text_box_props)

plt.show()

print("Script finished.")
