from utils import *

def main():
    data, errors = filter_data_on_condition(
        root="./src/data",
        preprocess=False,
        use_cache=False,
        min_emotion_count=3,
        mode=0,
        # generate_pc_hm_voxel=False,
        # generate_qna=False,
        # generate_voice=False,
    )

    # print(errors, len(data))


if "__main__" == __name__:
    main()

