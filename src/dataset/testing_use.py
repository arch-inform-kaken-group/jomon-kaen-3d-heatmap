from utils import *


def main():
    data, errors = filter_data_on_condition(
        root="./src/data",
        preprocess=True,
        use_cache=False,
        pottery_ids=["NM0135"],
        # 'HEATMAP, QNA, VOICE': 0 | 'HEATMAP, QNA': 1 | 'HEATMAP, VOICE': 2 | 'HEATMAP': 3
        mode=2,
        # generate_pc_hm_voxel=False,
        # generate_qna=False,
        # generate_voice=False,
    )

    # print(errors, len(data))


if "__main__" == __name__:
    main()
