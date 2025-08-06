from utils import *
import time

def main():
    st = time.time_ns()

    # data, errors = filter_data_on_condition(
    #     root="./src/data",
    #     pottery_path="./src/pottery",
    #     hololens_2_spatial_error=1.5,
    #     preprocess=True,
    #     use_cache=False,
    #     session_ids=["2025_06_25_17_17_56"],
    #     # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
    #     mode=0,
    #     # generate_pc_hm_voxel=False,
    #     # generate_qna=False,
    #     # generate_voice=True,
    #     # generate_pottery_dogu_voxel=False,
    #     # generate_sanity_check=True,
    # )

    data, errors = filter_data_on_condition(
        root=r"D:\storage\jomon_kaen\data",
        pottery_path=r"D:\storage\jomon_kaen\pottery",
        hololens_2_spatial_error=1.5,
        preprocess=True,
        use_cache=True,
        # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
        mode=3,
        # generate_pc_hm_voxel=False,
        # generate_qna=False,
        # generate_voice=True,
        # generate_pottery_dogu_voxel=False,
    )

    pottery_id_all = [f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()]
    avg_track = {key: {'count': 0, 'polysum': 0} for key in pottery_id_all}
    for data_path in data:
        voxel = o3d.io.read_point_cloud(data_path[voxel_filename])
        avg_track[data_path['ID']]['count'] += 1
        avg_track[data_path['ID']]['polysum'] += len(voxel.points)
    for key, cp in avg_track.items():
        print(f"{key}: {cp['polysum']/cp['count']}")

    # print(errors, len(data))

    et = time.time_ns()

    print(f"TIME: {(et-st)/1e9}")


if "__main__" == __name__:
    main()
