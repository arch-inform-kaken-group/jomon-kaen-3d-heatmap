import time

import numpy as np

from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset.dataset import get_jomon_kaen_dataset, visualize_geometry


def main():
    st = time.time_ns()

    train_dataset, test_dataset = get_jomon_kaen_dataset(
        root="./src/data",
        pottery_path="./src/pottery",
        split=0.25,
        preprocess=True,
        use_cache=True,
        pottery_ids=["IN0017"],
        # 'HEATMAP(VOXEL), QNA, VOICE': 0 | 'HEATMAP(VOXEL), QNA': 1 | 'HEATMAP(VOXEL), VOICE': 2 | 'HEATMAP(VOXEL)': 3
        mode=3,
        # generate_pc_hm_voxel=False,
        generate_qna=False,
        generate_voice=False,
        generate_pottery_dogu_voxel=False,
        generate_sanity_check=False)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
        # pin_memory=True,
        # num_workers=2,
        # prefetch_factor=2,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        shuffle=True,
        # pin_memory=True,
        # num_workers=2,
        # prefetch_factor=2,
    )

    EPOCHS = 50

    for epoch in range(EPOCHS):
        for pottery, voxel in tqdm(iter(train_dataloader), desc=f"EPOCH: {epoch + 1}"):
            print(pottery.shape, voxel.shape)

    et = time.time_ns()

    print(f"TIME: {(et-st)/1e9}")


if __name__ == "__main__":
    main()
