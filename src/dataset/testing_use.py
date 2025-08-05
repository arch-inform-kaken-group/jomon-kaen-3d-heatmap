from utils import *
import time


def main():
    st = time.time_ns()

    data, errors = filter_data_on_condition(
        root="./src/data",
        # root="c:/Users/luhou/Downloads/raw_my",
        pottery_path="./src/pottery",
        hololens_2_spatial_error=1.5,
        preprocess=True,
        use_cache=False,
        # cmap=plt.get_cmap('gray'),
        session_ids=[
            "2025_07_02_18_03_06", "2025_07_02_16_54_58",
            "2025_07_09_10_26_16", "2025_07_17_09_12_35", "2025_06_30_18_45_20"
        ],
        pottery_ids=["IN0295", "IN0017", "TK0020", "UD0023", "NM0099"],
        # session_ids=["2025_07_09_11_12_59", "2025_07_09_11_23_30", "2025_07_02_16_54_58", "2025_07_02_17_06_29"],
        # pottery_ids=["TJ0005", "TK0020"],
        min_emotion_count=2,
        min_qa_size=20,
        # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
        mode=0,
        # generate_pc_hm_voxel=False,
        # generate_qna=False,
        # generate_voice=True,
        generate_pottery_dogu_voxel=False,
        # generate_sanity_check=True,
        generate_fixation=True,
    )

    # data, errors = filter_data_on_condition(
    #     root=r"D:\storage\jomon_kaen\data",
    #     pottery_path=r"D:\storage\jomon_kaen\pottery",
    #     hololens_2_spatial_error=1.5,
    #     preprocess=True,
    #     use_cache=True,
    #     # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
    #     mode=0,
    #     # generate_pc_hm_voxel=False,
    #     # generate_qna=False,
    #     generate_voice=True,
    #     # generate_pottery_dogu_voxel=False,
    # )

    # print(errors, len(data))

    et = time.time_ns()

    print(f"TIME: {(et-st)/1e9}")


if "__main__" == __name__:
    main()
