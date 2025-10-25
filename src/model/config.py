EMOTION_ORDER = ["面白い・気になる形だ", "美しい・芸術的だ", "不思議・意味不明", "不気味・不安・怖い", "何も感じない"]

RAW_DATA_DIR = r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan"
MESH_DIR = r"D:\storage\jomon_kaen\pottery"
TEST_GROUPS = ['G9']
AUGMENT_COLOR_P = 0.5
COLOR_JITTER_STD = 0.05
JITTER_VOXEL_P = 0.2

BATCH_SIZE = 4
VOXEL_RESOLUTION = 80
MAX_EPOCHS = 1000
NUM_WORKERS = 8
LEARNING_RATE = 1e-4
MAX_COMMENT_LEN = 80
L1_WEIGHT = 0.001
VOXEL_LOSS_WEIGHT = 1.5

CONV_DIMS = [3, 8, 16, 32, 64]
NUM_EXPERTS = 3
TEACHER_FORCING_RATIO = 0.3

SAVE_EVERY_N_EPOCHS = 20
MAX_SAMPLES_TO_SAVE = 100
SAVE_DIR = r"D:\storage\jomon_kaen\validation_predictions_efficient_fixed"

EARLY_STOPPING_PATIENCE = 1000

VISUALIZE_SAMPLES = False
NUM_SAMPLES = 3