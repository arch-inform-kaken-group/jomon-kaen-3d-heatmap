from dataset.utils import *
import time


def main():
    st = time.time_ns()

    data, errors = filter_data_on_condition(
        # root=r"D:\storage\jomon_kaen\jomon_kaen_dataset\malaysia",
        # pottery_path=r"D:\storage\jomon_kaen\pottery",
        root="./src/jomon_kaen_dataset/malaysia",
        pottery_path="./src/pottery",
        hololens_2_spatial_error=1.5,
        # hololens_2_spatial_error=0.1,
        preprocess=True,
        use_cache=False,
        limit=1000,
        # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
        mode=1,
        ####################################################################
        # groups=['GX'],
        # generate_pc_hm_voxel=False,
        # generate_qna=False,
        # generate_voice=False,
        generate_pottery_dogu_voxel=False,
        generate_sanity_check=True,
        qna_marker=True,
        # generate_fixation=True,
        # generate_pointcloud=False,
        # generate_mesh=False,
        # generate_voxel=False,
        # generate_transcript=False,
        ###############
        # pottery_ids=["rembak7", "TK0020"],
        # session_ids=[
        #     "2025_07_02_18_03_06", "2025_07_02_16_54_58",
        #     "2025_07_09_10_26_16", "2025_07_17_09_12_35", "2025_06_30_18_45_20", "2025_09_01_11_45_07",
        #     "2025_07_10_11_03_23", "2025_07_10_08_46_51", "2025_06_25_19_31_00", "2025_07_10_11_11_25"
        # ],
        # pottery_ids=["IN0017"],
        # pottery_ids=["UD0028"],
        pottery_ids=["UD0003"],
        # min_qa_size=100,
        min_emotion_count=2,
    )

    et = time.time_ns()

    print(f"TIME: {(et-st)/1e9}")


if "__main__" == __name__:
    main()
