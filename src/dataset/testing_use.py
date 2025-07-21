from utils import *
import time

def main():
    st = time.time_ns()

    data, errors = filter_data_on_condition(
        root="./src/data",
        pottery_path="./src/pottery",
        hololens_2_spatial_error=1.5,
        preprocess=True,
        use_cache=False,
        session_ids=["2025_06_25_17_17_56"],
        # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
        mode=0,
        # generate_pc_hm_voxel=False,
        # generate_qna=False,
        # generate_voice=True,
        # generate_pottery_dogu_voxel=False,
        # generate_sanity_check=True,
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
