import os
import shutil

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dataset.processing.voice import process_voice_data

CORRECTION_MAP = {
    "面白い・気になる形だ": "面白い・気になる形だ",    # No change
    "不思議・意味不明": "美しい・芸術的だ",
    "何も感じない": "不思議・意味不明",
    "不気味・不安・怖い": "不気味・不安・怖い",    # No change
    "美しい・芸術的だ": "何も感じない",
    "Interesting and attentional shape": "Interesting and attentional shape",
    "Beautiful and artistic": "Feel nothing",
    "Strange and incomprehensible": "Beautiful and artistic",
    "Creepy / unsettling / scary": "Creepy / unsettling / scary",
    "Feel nothing": "Strange and incomprehensible",
}

def increment_error(key, path, errors: dict):
    if errors.get(key) == None:
        errors[key] = {'count': 1, 'paths': set([path])}
    else:
        errors[key]['count'] += 1
        errors[key]['paths'].add(path)

    return errors

def main(root="", output_dir=""):
    errors = {}
    
    # Check if each data instance / file path exists
    data = []
    if not Path(root).exists():
        raise (ValueError(f"Root directory not found: {root}"))
    
    os.makedirs(output_dir, exist_ok=True)

    # Filter based on group, session, model
    print(f"\nCHECKING RAW DATA PATHS")

    group_keys = os.listdir(root)
    for g in group_keys:
        group_path = root / Path(g)

        session_keys = os.listdir(group_path)
        for s in tqdm(session_keys, desc=g):
            session_path = group_path / Path(s)

            pottery_keys = os.listdir(session_path)
            for p in pottery_keys:
                if (p == 'language.txt'):
                    continue

                data_paths = {}
                pottery_path = session_path / Path(p)

                save_path = Path(output_dir) / g / s / p

                pointcloud_path = pottery_path / Path("pointcloud.csv")
                qa_path = pottery_path / Path("qa.csv")
                model_path = pottery_path / Path("model.obj")
                voice_path = pottery_path / Path("session_audio_0.wav")

                pointcloud_save_path = save_path / Path("pointcloud.csv")
                qa_save_path = save_path / Path("qa_corrected.csv")
                model_save_path = save_path / Path("model.obj")
                voice_save_path = save_path / Path("session_audio_45s.mp3")

                # Check if paths exist and increment error
                if Path(model_path).exists():
                    data_paths['model'] = str(model_path)
                    data_paths['model_save'] = str(model_save_path)
                    if Path(pointcloud_path).exists():
                        data_paths['pointcloud'] = str(pointcloud_path)
                        data_paths['pointcloud_save'] = str(pointcloud_save_path)
                        
                    else:
                        errors = increment_error('Point cloud path does not exist', str(pointcloud_path), errors)

                    if Path(qa_path).exists():
                        data_paths['qa'] = str(qa_path)
                        data_paths['qa_save'] = str(qa_save_path)

                    else:
                        errors = increment_error('QNA path does not exist', str(qa_path), errors)
                else:
                    errors = increment_error('Model path does not exist', str(model_path), errors)

                if Path(voice_path).exists():
                    data_paths['voice'] = str(voice_path)
                    data_paths['voice_save'] = str(voice_save_path)
                else:
                    errors = increment_error('Voice path does not exist', str(voice_path), errors)

                data_paths['SAVE'] = str(save_path)
                data_paths['GROUP'] = g
                data_paths['SESSION_ID'] = s
                data_paths['ID'] = p

                data.append(data_paths)

    n_valid_data = len(data)
    print(n_valid_data)

    model_count = 0
    pc_count = 0
    qa_count = 0
    voice_count = 0

    for data_paths in tqdm(data, desc="Making Clean Dataset"):
        os.makedirs(data_paths['SAVE'], exist_ok=True)
        
        if data_paths.get('model'):
            model_count += 1

            shutil.copy(data_paths['model'], data_paths['model_save'])

        if data_paths.get('pointcloud'):
            pc_count += 1

            shutil.copy(data_paths['pointcloud'], data_paths['pointcloud_save'])

        if data_paths.get('qa'):
            qa_count += 1

            df = pd.read_csv(data_paths['qa'])

            df['answer'] = df['answer'].str.strip()

            # Apply the correction using the map. 
            # .fillna(df['answer']) ensures that any value not in the map's keys remains unchanged.
            df['answer'] = df['answer'].map(CORRECTION_MAP).fillna(df['answer'])

            # Save the corrected dataframe to the new file
            # encoding='utf-8-sig' is recommended for CSVs with non-ASCII characters
            df.to_csv(data_paths['qa_save'], index=False, encoding='utf-8-sig')

        if data_paths.get('voice'):
            voice_count += 1

            audio_segment = process_voice_data(data_paths['voice'])
                    
            audio_segment.export(
                data_paths['voice_save'],
                format="mp3",
                bitrate="16k"
            )

    print('MODEL\t|\tPC\t|\tQA\t|\tVOICE')
    print(model_count, '\t|\t', pc_count, '\t|\t', qa_count, '\t|\t', voice_count)


if __name__ == "__main__":
    main(
        root="./src/data",
        output_dir="./src/jomon_kaen_dataset/japan",
    )

    main(
        root="./src/data_my",
        output_dir="./src/jomon_kaen_dataset/malaysia",
    )
