# 3D Heatmap Generation

- Analysis: **[FULL ANALYSIS VISUALIZATIONS](https://drive.google.com/file/d/1kiGFp70P4F-uNYrngHd1EO4dQ7gHZOr4/view?usp=sharing)** | **[ANALYSIS REPORT (*Subject to updates*)](https://drive.google.com/file/d/1ELGBItzYAaonFbO6x9OT3DY3TTNPWBfq/view?usp=sharing)**

- Generative Model: **[PRELIMINARY REPORT ON MODEL DEVELOPMENT](https://drive.google.com/file/d/1kMOmDZE0qLVqI-0uwtS5XCh7JABxCtfI/view?usp=sharing)** | [FULL ARCHITECTURE REPORT (*To be added*)]()

**Process experiment data from**

- PointCloud (.csv)
- QNA (.csv)
- model (.obj) 

into

- Segmented QNA (.ply)
- PointCloud (.ply)
- Heatmap (.ply)

**PyTorch Dataset & DataLoader**

Template to load the processed data into PyTorch for model training.

Functions

- Filter data based on

    - Group

    - Session ID

    - Pottery / Dogu ID

    - Point cloud data size

    - QNA data size

    - Voice quality, 1 - 5

    - Language, JP | EN

- Generate filtered data statistics

- Pre-process OR In-time process data

- TO BE ADDED: voice quality enhancement (normalization, background noise removal, AI to isolate comments)

## Clone the latest version

```
git clone --depth 1 https://github.com/luhouyang/3d-heatmap-generation.git
```

## PyTorch Dataset & DataLoader

[**SCRIPT**](src/dataset/dataset.py)

## Processing & Visualization Scripts

[**SCRIPT**](src/testing_use.py)

1. Create a folder in the `src` directory called `data`

1. Download the cleaned data [Japan](https://drive.google.com/file/d/1OetZFTXpzAPe5ZcNBdGylORAe_s6UxGR/view?usp=sharing) AND / OR [MALAYSIA](https://drive.google.com/file/d/1qNQJ-ipZ3IoATHDDP6gNDjIfT65aogTq/view?usp=sharing)

1. Place both inside the `data` directory

1. Create a folder in the `src` directory called `pottery`

1. Download the downsized pottery and dogu [Google Drive](https://drive.google.com/drive/folders/17zaoAvf2vPFnV8Yj6pCrLF8rSl9DqJyM?usp=drive_link)

1. Paste all pottery and dogu into the `pottery` directory

1. Filter the data by passing in the arguments or by using a [tracking sheet](https://docs.google.com/spreadsheets/d/1FLe6tAEtF5eAC3YXU8YLfOeI-VT83V1C/edit?usp=sharing&ouid=100175822335349725367&rtpof=true&sd=true) to the `filter_data_on_condition` function

1. Run the script, visualizations will be created inside each model folder / use the dataloader function `get_jomon_kaen_dataset` in [**SCRIPT**](src/dataset/dataset.py)

---

Modify the parameters for different results in the `filter_data_on_condition` function

### Example Using `split`

```python
train_dataset, test_dataset = get_jomon_kaen_dataset(
        root="./src/data/japan",
        pottery_path="./src/pottery",
        split=0.25,
        preprocess=True,
        use_cache=True,
        # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
        mode=3,
        generate_pointcloud=False,
        generate_mesh=False,
        generate_transcript=False,
    )
```

### Example using Pottery ID to group

```python
train_dataset, test_dataset = get_jomon_kaen_dataset(
        root="./src/data/japan",
        pottery_path="./src/pottery",
        test_groups=["G17"],
        preprocess=True,
        use_cache=True,
        # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
        mode=1,
        generate_pointcloud=False,
        generate_mesh=False,
        generate_transcript=False,
        generate_voxel=True,
    )
```

```python
root (str): Root directory that contains all groups 
pottery_path (str): Path to pottery files
preprocess (bool): Weather to preprocess and save the data to processed folder. Default: True
split (float): Fraction of test dataset. Default: 0.1,
test_groups (list): Pottery IDs to use as test group, n=excluded from training. Default: []
seed (int): np.random.seed(42),
mode (int): 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
hololens_2_spatial_error (float): Eye tracker spatial error of HoloLens 2. Default: DEFAULT_HOLOLENS_2_SPATIAL_ERROR
target_voxel_resolution (int): Target heatmap voxel resolution. Default: DEFAULT_TARGET_VOXEL_RESOLUTION
qna_answer_color_map (dict): The dictionary containing QNA answers with the rbg & name (color name). Default: DEFAULT_QNA_ANSWER_COLOR_MAP
base_color (list): Background color of all generated data. Default: DEFAULT_BASE_COLOR
cmap (plt.Colormap): Color scheme for intensities. Default: DEFAULT_CMAP
limit (int): Max of each pottery instance. Default: 9
groups (list): The list of groups to include, leave empty for all groups. Default: []
session_ids (list): The list of sessions to include, leave empty for all sessions. Default: []
pottery_ids (list): The list of potteries to include, leave empty for all potteries. Default: []
min_pointcloud_size (float): Minimum pointcoud data size. Default: 0.0
min_qa_size (float): Minimum qa data size. Default: 0.0
min_voice_quality (float): Minimum voice quality 1-5. Requires a tracking sheet to filter. Default: 0.1
min_emotion_count (int): Minimum emotion count. Unique QNA answers. Default: 0
use_cache (bool): Use previous preprocessed data. Default: True
from_tracking_sheet (bool): Use a tracking sheet .csv, downloaded from Google Sheets (You can filter the data at Google Sheets and export the subset). Default: False
tracking_sheet_path (str): Path to the tracking sheet. Default: ""
generate_report (bool): Generate a data report. Default: True
generate_pc_hm_voxel (bool): Generate pointcloud, heatmap & voxel. Default: True
generate_qna (bool): Generate QNA combined meah, segmented mesh, pointcloud. Default: True
generate_voice (bool): Generate voice. Default: True
generate_pottery_dogu_voxel (bool): Generate the input pottery and dogu voxel. Default: True
generate_sanity_check (bool): Generate sanity check png. Default: False
generate_fixation (bool): Generate gaze fixation point cloud and heatmap, with a duration aggregated point cloud, heatmap and legend. Default: False
voxel_color (str): 'gray' or 'rgb'. NOT YET IMPLEMENTED. Default: 'gray'
qna_marker (bool): Generate QNA point cloud as shaped markers. Default: False
generate_voxel (bool): Generate voxel data. Default: True,
generate_mesh (bool): Generate mesh data. Default: True,
generate_pointcloud (bool): Generate point cloud data. Default: True,
generate_transcript (bool): Generate transcript data. Default: True,
```

## Analysis

```bash
cd src/analysis
```

### Heatmap Comparison

This command runs the **gaze heatmap comparison** between the Japanese and Malaysian datasets. It uses the default directory paths and parameters.

```bash
python analysis.py heatmap
```

To specify a different output directory:

```bash
python analysis.py heatmap --output_dir ./results/my_heatmap_analysis
```

-----

### QA Event Clustering

This command clusters pottery based on the emotions recorded in the **QA events**. It generates PCA plots and 3D model collages for each cluster.

For the **Japanese** dataset:

```bash
python analysis.py qa_cluster japan --data_dir ./src/jomon_kaen_dataset/japan
```

For the **Malaysian** dataset:

```bash
python analysis.py qa_cluster malaysia --data_dir ./src/jomon_kaen_dataset/malaysia
```

-----

### Transcript Emotion Clustering

This command runs **transcript-based clustering** on the **Japanese** dataset. You must provide the data directory and a suitable multilingual model from Hugging Face.

```bash
python analysis.py transcript_cluster japan --data_dir ./src/jomon_kaen_dataset/japan --model_id MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
```

Here's the equivalent command for the **Malaysian** (English) dataset, using a model optimized for English.

```bash
python analysis.py transcript_cluster malaysia --data_dir ./src/jomon_kaen_dataset/malaysia --model_id cross-encoder/nli-deberta-v3-large
```

-----

### QA vs. Transcript Alignment Report

This command generates a detailed **PDF report** comparing the emotion distributions from QA events against the emotion distributions derived from classifying the full text of the transcripts.

For the **Japanese** dataset:

```bash
python analysis.py qa_alignment japan --data_dir ./src/jomon_kaen_dataset/japan --model_id MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 --font_path "C:/Windows/Fonts/msgothic.ttc"
```

For the **Malaysian** dataset:

```bash
python analysis.py qa_alignment malaysia --data_dir ./src/jomon_kaen_dataset/malaysia --model_id cross-encoder/nli-deberta-v3-large --font_path "C:/Windows/Fonts/simhei.ttf"
```

-----

### Word Frequency Analysis

This command generates word clouds, frequency charts, and a transcript PDF for the **Japanese** dataset. You must provide a path to a Japanese font file.

```bash
python analysis.py word_freq japan --data_dir ./src/jomon_kaen_dataset/japan --font_path "C:/Windows/Fonts/msgothic.ttc"
```

And for the **Malaysian** (English) dataset. A standard system font like Arial works well here.

```bash
python analysis.py word_freq malaysia --data_dir ./src/jomon_kaen_dataset/malaysia --font_path "C:/Windows/Fonts/simhei.ttf"
```

-----

### Voxel Count Analysis

This command analyzes the `.ply` files in the default processed directory to **plot the voxel counts** for each pottery item.

```bash
python analysis.py voxels
```

-----

### Label Embedding Visualization

This command **visualizes the semantic relationships** between the Japanese and English emotion labels in 2D and 3D space.

```bash
python analysis.py label_viz
```

-----

### Individual Pottery Bar Charts

This command generates **individual stacked percentage bar charts** for each pottery item, showing the viewing time distribution across emotion categories. This is useful for creating presentation materials or detailed per-item analyses.

For the **Japanese** dataset:

```bash
python analysis.py bar_charts japan --data_dir ./src/jomon_kaen_dataset/japan --output_dir ./pottery_charts_jp
```

For the **Malaysian** dataset:

```bash
python analysis.py bar_charts malaysia --data_dir ./src/jomon_kaen_dataset/malaysia --output_dir ./pottery_charts_my
```

**Optional Parameters:**
- `--output_dir`: Directory where charts will be saved (default: `output_data`)
- Charts are saved as `{pottery_id}_viewing_time_bar_chart.png`

**Output:**
- One stacked bar chart per pottery item
- Shows percentage distribution of viewing time across emotions
- Includes proper legends and color coding matching the emotion categories
- Charts are sized for easy embedding in presentations (4x6 inches by default)

-----

## Python API Usage

You can also use the bar chart generation function directly in your Python scripts:

```python
from helper import create_individual_pottery_bar_charts, load_combined_qna_data, EMOTION_MAPS

# Load your data
combined_df = load_combined_qna_data("./src/jomon_kaen_dataset/japan")

# Add short label mapping
language = 'japan'  # or 'malaysia'
emotion_map = EMOTION_MAPS[language]['full_map']
combined_df['short_answer'] = combined_df['answer'].str.strip().map(emotion_map)

# Generate the bar charts
create_individual_pottery_bar_charts(
    combined_df=combined_df,
    language=language,
    output_dir='./my_charts'
)
```

-----

## Helper Functions Reference

The `helper.py` module now includes the following key functions:

### Data Loading
- `get_pottery_id_list()` - Returns formatted pottery IDs
- `group_data_by_pottery()` - Groups pointcloud and model files
- `load_transcripts()` - Loads transcript data
- `load_combined_qna_data()` - Loads and combines QA data
- `load_alignment_data()` - Loads QA and transcript pairs

### Visualization
- `create_individual_pottery_bar_charts()` - **NEW** - Generates individual bar charts
- `create_jsd_bar_chart()` - JSD comparison bar chart
- `create_cluster_collage()` - 3D model collages
- `draw_ellipse()` - Fitted ellipse drawing
- `generate_word_cloud_and_bar_chart()` - Word frequency visualizations

### 3D Rendering
- `render_glb_matplotlib()` - Renders 3D models
- `render_glb_front_view()` - Front view rendering
- `create_simple_pottery_image()` - Placeholder images
- `save_colored_mesh()` - Saves colored meshes

### Analysis
- `calculate_jensen_shannon_distance()` - JSD calculation
- `calculate_qa_emotion_percentages()` - Emotion distribution from QA

### Report Generation
- `generate_alignment_report()` - PDF alignment report
- `generate_transcript_pdf()` - PDF transcript compilation

-----

## Notes

- All visualization functions respect the language setting for proper Japanese/English labels
- Bar charts use consistent color schemes matching the emotion categories
- Charts are optimized for both screen display and print/presentation use
- Progress bars (tqdm) provide feedback during batch processing