from dataset.utils import *
import time


def main():
    st = time.time_ns()

    data, errors = filter_data_on_condition(
        root="./src/data/japan",
        # root="c:/Users/luhou/Downloads/raw_my",
        pottery_path="./src/pottery",
        hololens_2_spatial_error=1.5,
        # hololens_2_spatial_error=0.1,
        preprocess=True,
        use_cache=False,
        ####################################################################
        # session_ids=[
        #     "2025_07_02_18_03_06", "2025_07_02_16_54_58",
        #     "2025_07_09_10_26_16", "2025_07_17_09_12_35", "2025_06_30_18_45_20"
        # ],
        # pottery_ids=["IN0295", "IN0017", "TK0020", "UD0023", "NM0099"],
        # pottery_ids=["IN0017", "TK0020"],
        # limit=1000,
        # min_emotion_count=3,
        # min_qa_size=20,
        ####################################################################
        # session_ids=["2025_07_10_11_03_23", "2025_07_10_08_46_51", "2025_06_25_19_31_00", "2025_07_10_11_11_25"],
        # min_emotion_count=1,
        # max_emotion_count=1,
        qna_marker=True,
        # pottery_ids=["MH0037", "SK0001", "TK0020", "UD0308"],
        # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
        mode=1,
        ####################################################################
        # groups=['GX'],
        # generate_pc_hm_voxel=False,
        # generate_qna=False,
        # generate_voice=True,
        generate_pottery_dogu_voxel=False,
        # generate_sanity_check=True,
        # generate_fixation=True,
    )

    # data, errors = filter_data_on_condition(
    #     root=r"D:\storage\jomon_kaen\data",
    #     pottery_path=r"D:\storage\jomon_kaen\pottery",
    #     hololens_2_spatial_error=1.5,
    #     preprocess=True,
    #     use_cache=True,
    #     # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
    #     mode=3,
    #     # generate_pc_hm_voxel=False,
    #     # generate_qna=False,
    #     # generate_voice=True,
    #     # generate_pottery_dogu_voxel=False,
    # )

    # pottery_id_all = [f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()]
    # avg_track = {key: {'count': 0, 'polysum': 0} for key in pottery_id_all}
    # for data_path in data:
    #     voxel = o3d.io.read_point_cloud(data_path[voxel_filename])
    #     avg_track[data_path['ID']]['count'] += 1
    #     avg_track[data_path['ID']]['polysum'] += len(voxel.points)
    # for key, cp in avg_track.items():
    #     print(f"{key}: {cp['polysum']/cp['count']}")

    # print(errors, len(data))

    et = time.time_ns()

    print(f"TIME: {(et-st)/1e9}")


if "__main__" == __name__:
    main()
