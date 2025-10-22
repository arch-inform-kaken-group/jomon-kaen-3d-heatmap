import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
import neologdn
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sudachipy import tokenizer as sudachi_tokenizer, dictionary as sudachi_dictionary

import helper  # Import our helper library

# 1. HEATMAP COMPARISON WORKFLOW


def _calculate_vertex_intensities(gaze_points_np, mesh, gaussian_denominator,
                                  spatial_error):
    """Calculates smoothed vertex hit counts from gaze points."""
    mesh_vertices_np = np.asarray(mesh.vertices)
    n_vertices = mesh_vertices_np.shape[0]

    mesh_scene = o3d.t.geometry.RaycastingScene()
    mesh_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query_points = o3d.core.Tensor(gaze_points_np,
                                   dtype=o3d.core.Dtype.Float32)
    closest_geometry = mesh_scene.compute_closest_points(query_points)
    closest_face_indices = closest_geometry['primitive_ids'].numpy()

    raw_hit_counts = np.zeros(n_vertices, dtype=np.float64)
    for face_idx in closest_face_indices:
        if face_idx != o3d.t.geometry.RaycastingScene.INVALID_ID:
            for v_idx in np.asarray(mesh.triangles)[face_idx]:
                raw_hit_counts[v_idx] += 1

    raw_hit_counts = np.log1p(raw_hit_counts)  # Log scaling

    # Apply Gaussian smoothing
    kdtree = o3d.geometry.KDTreeFlann(mesh)
    interpolated_values = np.copy(raw_hit_counts)
    hit_indices = np.where(raw_hit_counts > 0)[0]

    for v_idx in tqdm(hit_indices, desc="Smoothing", leave=False):
        hit_value = raw_hit_counts[v_idx]
        [_, indices,
         dists] = kdtree.search_radius_vector_3d(mesh_vertices_np[v_idx],
                                                 spatial_error)
        if len(indices) > 1:
            gaussian_weights = np.exp(-np.asarray(dists)**2 /
                                      gaussian_denominator)
            for i, neighbor_idx in enumerate(indices):
                if neighbor_idx != v_idx:
                    interpolated_values[
                        neighbor_idx] += hit_value * gaussian_weights[i]
    return interpolated_values


def run_heatmap_comparison(args):
    """Main function for gaze heatmap comparison analysis."""
    print("--- Running Gaze Heatmap Comparison")
    os.makedirs(args.output_dir, exist_ok=True)

    japan_data = helper.group_data_by_pottery(args.japan_dir)
    malaysia_data = helper.group_data_by_pottery(args.malaysia_dir)
    all_pottery_ids = sorted(
        list(set(japan_data.keys()) | set(malaysia_data.keys())))

    all_scores = []
    gaussian_denominator = 2 * (args.hololens_error**2)

    for pottery_id in tqdm(all_pottery_ids, desc="Processing All Pottery"):
        pottery_output_dir = os.path.join(args.output_dir, pottery_id)
        os.makedirs(pottery_output_dir, exist_ok=True)

        all_intensities = {}
        base_mesh = None

        for country, data in [('japan', japan_data),
                              ('malaysia', malaysia_data)]:
            instances = data.get(pottery_id)
            if not instances:
                all_intensities[country] = None
                continue

            if base_mesh is None:
                base_mesh = o3d.io.read_triangle_mesh(instances[0]['model'])

            num_vertices = len(base_mesh.vertices)
            summed_intensities = np.zeros(num_vertices)

            for inst in tqdm(instances,
                             desc=f"{pottery_id} - {country}",
                             leave=False):
                gaze_points = pd.read_csv(
                    inst['pointcloud'],
                    header=0).iloc[:, :3].to_numpy().astype(np.float64)
                mesh = o3d.io.read_triangle_mesh(inst['model'])
                if len(mesh.vertices) == num_vertices:
                    summed_intensities += _calculate_vertex_intensities(
                        gaze_points, mesh, gaussian_denominator,
                        args.hololens_error)

            all_intensities[country] = summed_intensities / len(
                instances) if instances else np.zeros(num_vertices)

        if base_mesh is None or not base_mesh.has_vertices(): continue

        num_v = len(base_mesh.vertices)
        jp_intensities = all_intensities.get('japan')
        my_intensities = all_intensities.get('malaysia')
        if jp_intensities is None or len(jp_intensities) != num_v:
            jp_intensities = np.zeros(num_v)
        if my_intensities is None or len(my_intensities) != num_v:
            my_intensities = np.zeros(num_v)

        # Normalize for visualization
        max_jp = np.max(jp_intensities)
        norm_jp = jp_intensities / max_jp if max_jp > 0 else jp_intensities
        max_my = np.max(my_intensities)
        norm_my = my_intensities / max_my if max_my > 0 else my_intensities

        jsd_score = helper.calculate_jensen_shannon_distance(
            jp_intensities, my_intensities)
        all_scores.append({'pottery_id': pottery_id, 'js_distance': jsd_score})

        cmap = plt.get_cmap(args.cmap)
        helper.save_colored_mesh(
            base_mesh, norm_jp, cmap,
            os.path.join(pottery_output_dir, "japan_average.ply"))
        helper.save_colored_mesh(
            base_mesh, norm_my, cmap,
            os.path.join(pottery_output_dir, "malaysia_average.ply"))

        diff_colors = helper.create_difference_colors(norm_jp, norm_my)
        diff_mesh = deepcopy(base_mesh)
        diff_mesh.vertex_colors = o3d.utility.Vector3dVector(diff_colors)
        o3d.io.write_triangle_mesh(os.path.join(pottery_output_dir,
                                                "difference.ply"),
                                   diff_mesh,
                                   write_ascii=True)

    # Final Summary
    scores_df = pd.DataFrame(all_scores)
    if not scores_df.empty:
        summary_stats = scores_df['js_distance'].describe()
        helper.create_jsd_bar_chart(scores_df, summary_stats, args.output_dir)
        with open(os.path.join(args.output_dir, "comparison_summary.txt"),
                  'w') as f:
            f.write("JSD Comparison Summary\n" + "=" * 25 + "\n" +
                    str(summary_stats) + "\n\nAll Scores:\n" + "=" * 25 +
                    "\n" + scores_df.to_string(index=False))
        print(f"Summary file and chart saved to '{args.output_dir}'")


# 2. QA-EVENT CLUSTERING WORKFLOW


def run_qa_cluster(args):
    """Performs K-Means clustering on QA event data."""
    print("--- Running QA-Event Based Emotion Clustering")
    combined_df = helper.load_combined_qna_data(args.data_dir)
    if combined_df.empty:
        print("No QA data found. Aborting.")
        return

    session_counts = pd.crosstab(
        [combined_df['pottery_id'], combined_df['session_id']],
        combined_df['answer'])
    session_percentages = session_counts.div(session_counts.sum(axis=1),
                                             axis=0) * 100
    pottery_level_df = session_percentages.groupby('pottery_id').mean().fillna(
        0)

    # Clustering logic from original script
    print("Finding optimal K using Elbow Method...")
    inertia = [
        KMeans(n_clusters=k, random_state=42,
               n_init='auto').fit(pottery_level_df).inertia_
        for k in range(1, min(11, len(pottery_level_df)))
    ]
    plt.figure()
    plt.plot(range(1, len(inertia) + 1), inertia, marker='o')
    plt.title('Elbow Method for Optimal K (QA Data)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.savefig("qa_cluster_elbow_plot.png")
    plt.close()
    print("Elbow plot saved to 'qa_cluster_elbow_plot.png'")

    for k in range(2, args.max_k + 1):
        if k > len(pottery_level_df): break
        output_dir = f'qa_clusters_k{k}'
        os.makedirs(output_dir, exist_ok=True)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(pottery_level_df)
        pottery_level_df[f'cluster_k{k}'] = cluster_labels

        pca = PCA(n_components=2, random_state=42)
        reduced_features = pca.fit_transform(
            pottery_level_df.iloc[:, :-1])  # Exclude cluster col

        plt.figure(figsize=(12, 8))
        cmap = plt.get_cmap('viridis', k)
        scatter = plt.scatter(reduced_features[:, 0],
                              reduced_features[:, 1],
                              c=cluster_labels,
                              cmap=cmap)
        for i, txt in enumerate(pottery_level_df.index):
            plt.annotate(txt, (reduced_features[i, 0], reduced_features[i, 1]))

        for i in range(k):
            points = reduced_features[cluster_labels == i]
            helper.draw_ellipse(points,
                                ax=plt.gca(),
                                edgecolor=cmap(i / (k - 1 if k > 1 else 1)),
                                facecolor='none',
                                lw=2,
                                linestyle='--')

        plt.title(f'K-Means Clustering (K={k}) of Pottery by QA Emotion')
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=[f'Cluster {i}' for i in range(k)])
        plt.savefig(os.path.join(output_dir, f'pca_cluster_plot_k{k}.png'))
        plt.close()

        for i in range(k):
            members = pottery_level_df[pottery_level_df[f'cluster_k{k}'] ==
                                       i].index.tolist()
            if members:
                helper.create_cluster_collage(members, args.pottery_models_dir,
                                              i, output_dir)


# 3. TRANSCRIPT CLUSTERING WORKFLOW


def run_transcript_clustering(args):
    """Performs emotion analysis on transcripts and clusters the results."""
    print("--- Running Transcript-Based Emotion Clustering")
    transcripts_dict, _ = helper.load_transcripts(args.data_dir)
    if not transcripts_dict:
        print("No transcripts found. Aborting.")
        return

    print(f"Loading zero-shot model: {args.model_id}")
    device = 0 if torch.cuda.is_available() and not args.force_cpu else -1
    classifier = pipeline("zero-shot-classification",
                          model=args.model_id,
                          device=device)

    target_labels = helper.EMOTION_MAPS[args.language]['target_labels']
    session_keys, transcript_texts = zip(*transcripts_dict.items())

    print(f"Classifying {len(transcript_texts)} transcripts...")
    results_generator = (classifier(text, target_labels, multi_label=False)
                         for text in transcript_texts)
    all_results = list(tqdm(results_generator, total=len(transcript_texts)))

    records = []
    for i, result in enumerate(all_results):
        pottery_id, session_id = session_keys[i]
        score_dict = {
            label: score
            for label, score in zip(result['labels'], result['scores'])
        }
        records.append({
            'pottery_id': pottery_id,
            'session_id': session_id,
            **score_dict
        })

    session_level_df = pd.DataFrame(records)
    pottery_level_df = session_level_df.drop(
        columns='session_id').groupby('pottery_id').mean()

    print("Finding optimal K using Elbow Method...")
    inertia = [
        KMeans(n_clusters=k, random_state=42,
               n_init='auto').fit(pottery_level_df).inertia_
        for k in range(1, min(11, len(pottery_level_df)))
    ]
    plt.figure()
    plt.plot(range(1, len(inertia) + 1), inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Clusters (K)')
    plt.ylabel('Inertia')
    plt.savefig("transcript_cluster_elbow_plot.png")
    plt.close()
    print("Elbow plot saved to 'transcript_cluster_elbow_plot.png'")

    for k in range(2, args.max_k + 1):
        if k > len(pottery_level_df): break
        output_dir = f'transcript_clusters_k{k}'
        os.makedirs(output_dir, exist_ok=True)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(pottery_level_df)

        pca = PCA(n_components=2, random_state=42)
        reduced_features = pca.fit_transform(pottery_level_df)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced_features[:, 0],
                              reduced_features[:, 1],
                              c=cluster_labels,
                              cmap=plt.get_cmap('viridis', k))
        for i, txt in enumerate(pottery_level_df.index):
            plt.annotate(txt, (reduced_features[i, 0], reduced_features[i, 1]))
        plt.title(
            f'K-Means Clustering (K={k}) of Pottery by Transcript Emotion')
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=[f'Cluster {i}' for i in range(k)])
        plt.savefig(os.path.join(output_dir, f'pca_cluster_plot_k{k}.png'))
        plt.close()

        for i in range(k):
            members = pottery_level_df.index[cluster_labels == i].tolist()
            if members:
                helper.create_cluster_collage(members, args.pottery_models_dir,
                                              i, output_dir)


# 4. QA vs. TRANSCRIPT ALIGNMENT WORKFLOW
def run_qa_alignment(args):
    """Generates a PDF report comparing QA emotions vs. transcript emotions."""
    print("--- Running QA vs. Transcript Alignment Analysis")
    data_paths = helper.load_alignment_data(args.data_dir)
    if not data_paths: return

    data_paths = helper.calculate_qa_emotion_percentages(
        data_paths, args.language)

    print(f"Loading zero-shot model: {args.model_id}")
    device = 0 if torch.cuda.is_available() and not args.force_cpu else -1
    classifier = pipeline("zero-shot-classification",
                          model=args.model_id,
                          device=device)

    target_labels = helper.EMOTION_MAPS[args.language]['target_labels']

    print(f"Classifying {len(data_paths)} transcripts...")
    for item in tqdm(data_paths, desc="Classifying Transcripts"):
        with open(item['TRANSCRIPT'], 'r', encoding='utf-8') as f:
            text = f.read()
            item['transcript_text'] = text  # Store for PDF

        result = classifier(text, target_labels, multi_label=False)
        score_dict = {
            label: score
            for label, score in zip(result['labels'], result['scores'])
        }
        item['embedding_percentages'] = {
            label: score_dict.get(label, 0) * 100
            for label in target_labels
        }

        qa_vec = np.array(list(item['qa_percentages'].values()))
        embed_vec = np.array(list(item['embedding_percentages'].values()))

        norm_qa = np.linalg.norm(qa_vec)
        norm_embed = np.linalg.norm(embed_vec)

        if norm_qa > 0 and norm_embed > 0:
            item['alignment_score'] = np.dot(
                qa_vec, embed_vec) / (norm_qa * norm_embed)
        else:
            item['alignment_score'] = 0.0

    output_filename = f"{args.language.upper()}_QA_Transcript_Alignment_Report.pdf"
    helper.generate_alignment_report(data_paths, args.model_id,
                                     output_filename, args.language,
                                     args.font_path)


# 5. WORD FREQUENCY ANALYSIS
def run_word_frequency_analysis(args):
    """Performs word frequency analysis on transcripts."""
    print("--- Running Word Frequency Analysis")
    transcripts_dict, _ = helper.load_transcripts(args.data_dir)
    if not transcripts_dict: return

    transcripts_by_pottery = defaultdict(list)
    for (pottery_id, session_id), content in transcripts_dict.items():
        transcripts_by_pottery[pottery_id].append((session_id, content))

    all_words = []
    if args.language == 'japan':
        tokenizer = sudachi_dictionary.Dictionary().create()
        mode = sudachi_tokenizer.Tokenizer.SplitMode.A
        pos_to_keep = {"名詞", "動詞", "形容詞"}
        for text in transcripts_dict.values():
            normalized = neologdn.normalize(text)
            tokens = [
                m.normalized_form()
                for m in tokenizer.tokenize(normalized, mode)
                if m.part_of_speech()[0] in pos_to_keep
            ]
            all_words.extend(tokens)
    else:  # English
        for text in transcripts_dict.values():
            words = [word for word in text.lower().split() if word.isalpha()]
            all_words.extend(words)

    output_prefix = f"{args.language}_word_freq"
    helper.generate_word_cloud_and_bar_chart(all_words, args.font_path,
                                             output_prefix)
    helper.generate_transcript_pdf(transcripts_by_pottery,
                                   f"{args.language}_transcripts.pdf",
                                   args.font_path)


# 6. VOXEL ANALYSIS
def run_voxel_analysis(args):
    """Analyzes voxel files and generates a bar chart of counts."""
    print("--- Running Voxel Count Analysis")
    if not os.path.isdir(args.voxel_dir):
        print(f"Error: Directory not found: {args.voxel_dir}")
        return

    ply_files = [f for f in os.listdir(args.voxel_dir) if f.endswith('.ply')]
    if not ply_files:
        print("No .ply files found.")
        return

    counts = {
        os.path.splitext(f)[0]:
        helper.read_ply_vertex_count(os.path.join(args.voxel_dir, f))
        for f in ply_files
    }
    sorted_items = sorted(counts.items(), key=lambda item: item[1])
    ids, values = zip(*sorted_items)

    plt.figure(figsize=(15, 8))
    plt.bar(ids, values)
    plt.title('Voxel Count per Pottery ID')
    plt.ylabel('Number of Voxels')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("voxel_counts_barchart.png")
    plt.close()
    print("Voxel count bar chart saved to 'voxel_counts_barchart.png'")


# 7. LABEL EMBEDDING VISUALIZATION
def run_label_embedding_visualization(args):
    """Visualizes the semantic relationship between emotion labels."""
    print("--- Running Label Embedding Visualization")
    model = SentenceTransformer(args.model_id)
    jp_labels = helper.EMOTION_MAPS['japan']['target_labels']
    en_labels = helper.EMOTION_MAPS['malaysia']['target_labels']
    all_labels = jp_labels + en_labels

    embeddings = model.encode(all_labels)

    embedding_2d = umap.UMAP(n_components=2,
                             random_state=42,
                             min_dist=0.0,
                             metric="cosine",
                             n_jobs=1).fit_transform(embeddings)
    embedding_3d = umap.UMAP(n_components=3,
                             random_state=42,
                             min_dist=0.0,
                             metric="cosine",
                             n_jobs=1).fit_transform(embeddings)

    colors = [
        plt.cm.get_cmap('jet', len(jp_labels))(i)
        for i in range(len(jp_labels))
    ] * 2

    fig = plt.figure(figsize=(18, 8))
    ax_2d = fig.add_subplot(1, 2, 1)
    ax_2d.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors)
    for i, label in enumerate(all_labels):
        ax_2d.text(embedding_2d[i, 0], embedding_2d[i, 1], label)
    ax_2d.set_title('2D UMAP Projection of Emotion Labels')
    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
    ax_3d.scatter(embedding_3d[:, 0],
                  embedding_3d[:, 1],
                  embedding_3d[:, 2],
                  c=colors)
    for i, label in enumerate(all_labels):
        ax_3d.text(embedding_3d[i, 0], embedding_3d[i, 1], embedding_3d[i, 2],
                   label)
    ax_3d.set_title('3D UMAP Projection of Emotion Labels')

    plt.tight_layout()
    plt.savefig("label_embedding_visualization.png")
    plt.show()


def run_individual_bar_charts(args):
    """Generates individual stacked bar charts for each pottery item."""
    print("--- Running Individual Pottery Bar Chart Generation")
    combined_df = helper.load_combined_qna_data(args.data_dir)

    if combined_df.empty:
        print("No QA data found. Aborting.")
        return

    # Map to short labels
    emotion_map = helper.EMOTION_MAPS[args.language]['full_map']
    combined_df['answer'] = combined_df['answer'].str.strip()
    combined_df['short_answer'] = combined_df['answer'].map(emotion_map)

    # Generate the bar charts
    helper.create_individual_pottery_bar_charts(combined_df=combined_df,
                                                language=args.language,
                                                output_dir=args.output_dir)

    print(
        f"\n✓ Individual bar charts generated successfully in '{args.output_dir}'"
    )


# MAIN EXECUTION BLOCK
def main():
    parser = argparse.ArgumentParser(
        description="Run various analyses on Jomon Kaen dataset.")
    subparsers = parser.add_subparsers(dest="command",
                                       required=True,
                                       help="Available analyses")

    # 1. Heatmap Parser
    p1 = subparsers.add_parser(
        'heatmap', help="Compare gaze heatmaps between Japan and Malaysia.")
    p1.add_argument('--japan_dir',
                    type=str,
                    default="./src/jomon_kaen_dataset/japan")
    p1.add_argument('--malaysia_dir',
                    type=str,
                    default="./src/jomon_kaen_dataset/malaysia")
    p1.add_argument('--output_dir',
                    type=str,
                    default="./processed/heatmap_comparison")
    p1.add_argument('--cmap', type=str, default='jet')
    p1.add_argument('--hololens_error', type=float, default=1.5)
    p1.set_defaults(func=run_heatmap_comparison)

    # 2. QA Cluster Parser
    p2 = subparsers.add_parser(
        'qa_cluster', help="Cluster pottery based on QA event emotion data.")
    p2.add_argument('language', choices=['japan', 'malaysia'])
    p2.add_argument('--data_dir', type=str, required=True)
    p2.add_argument('--pottery_models_dir', type=str, default="./src/pottery")
    p2.add_argument('--max_k', type=int, default=8)
    p2.set_defaults(func=run_qa_cluster)

    # 3. Transcript Clustering Parser
    p3 = subparsers.add_parser(
        'transcript_cluster',
        help="Cluster pottery based on transcript emotion analysis.")
    p3.add_argument('language', choices=['japan', 'malaysia'])
    p3.add_argument('--data_dir', type=str, required=True)
    p3.add_argument('--pottery_models_dir', type=str, default="./src/pottery")
    p3.add_argument('--model_id', type=str, required=True)
    p3.add_argument('--max_k', type=int, default=8)
    p3.add_argument('--force_cpu', action='store_true')
    p3.set_defaults(func=run_transcript_clustering)

    # 4. QA Alignment Parser
    p4 = subparsers.add_parser(
        'qa_alignment',
        help="Generate a PDF report comparing QA and transcript emotions.")
    p4.add_argument('language', choices=['japan', 'malaysia'])
    p4.add_argument('--data_dir', type=str, required=True)
    p4.add_argument('--model_id', type=str, required=True)
    p4.add_argument('--font_path', type=str, required=True)
    p4.add_argument('--force_cpu', action='store_true')
    p4.set_defaults(func=run_qa_alignment)

    # 5. Word Frequency Parser
    p5 = subparsers.add_parser('word_freq',
                               help="Analyze word frequencies in transcripts.")
    p5.add_argument('language', choices=['japan', 'malaysia'])
    p5.add_argument('--data_dir', type=str, required=True)
    p5.add_argument('--font_path', type=str, required=True)
    p5.set_defaults(func=run_word_frequency_analysis)

    # 6. Voxel Parser
    p6 = subparsers.add_parser('voxels', help="Analyze and plot voxel counts.")
    p6.add_argument('--voxel_dir',
                    type=str,
                    default="./src/jomon_kaen_dataset/processed/voxel_pottery")
    p6.set_defaults(func=run_voxel_analysis)

    # 7. Label Embedding Parser
    p7 = subparsers.add_parser(
        'label_viz', help="Visualize semantic embeddings of emotion labels.")
    p7.add_argument(
        '--model_id',
        type=str,
        default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    p7.set_defaults(func=run_label_embedding_visualization)

    # 8. Individual Bar Charts Parser
    p8 = subparsers.add_parser(
        'bar_charts',
        help="Generate individual stacked bar charts for each pottery item.")
    p8.add_argument('language',
                    choices=['japan', 'malaysia'],
                    help="Dataset language (affects labels and formatting)")
    p8.add_argument('--data_dir',
                    type=str,
                    required=True,
                    help="Path to the dataset directory")
    p8.add_argument(
        '--output_dir',
        type=str,
        default='output_data',
        help="Directory where bar charts will be saved (default: output_data)")
    p8.set_defaults(func=run_individual_bar_charts)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
