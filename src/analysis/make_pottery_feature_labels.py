import os
from pathlib import Path
import numpy as np
import pandas as pd

# Pottery & Dogu assigned numbers
ASSIGNED_NUMBERS_DICT = {
    'AS0001': '1', 'FH0008': '2', 'IN0003': '3', 'IN0008': '4',
    'IN0009': '5', 'IN0017': '6', 'IN0081': '7', 'IN0104': '8',
    'IN0135': '9', 'IN0148': '10', 'IN0220': '11', 'IN0228': '12',
    'IN0232': '13', 'IN0239': '14', 'IN0277': '15', 'MY0001': '16',
    'MY0002': '17', 'MY0004': '18', 'MY0006': '19', 'MY0007': '20',
    'ND0001': '21', 'NM0001': '22', 'NM0002': '23', 'NM0009': '24',
    'NM0010': '25', 'NM0014': '26', 'NM0015': '27', 'NM0017': '28',
    'NM0041': '29', 'NM0049': '30', 'NM0066': '31', 'NM0070': '32',
    'NM0072': '33', 'NM0073': '34', 'NM0079': '35', 'NM0080': '36',
    'NM0099': '37', 'NM0106': '38', 'NM0133': '39', 'NM0135': '40',
    'NM0144': '41', 'NM0154': '42', 'NM0156': '43', 'NM0159': '44',
    'NM0168': '45', 'NM0173': '46', 'NM0175': '47', 'NM0189': '48',
    'NM0191': '49', 'NM0206': '50', 'SB0002': '51', 'SB0004': '52',
    'SI0001': '53', 'SJ0503': '54', 'SJ0504': '55', 'SK0001': '56',
    'SK0002': '57', 'SK0003': '58', 'SK0004': '59', 'SK0005': '60',
    'SK0013': '61', 'SS0001': '62', 'TJ0004': '63', 'TJ0005': '64',
    'TJ0010': '65', 'TK0002': '66', 'TK0048': '67', 'TK0057': '68',
    'UD0001': '69', 'UD0003': '70', 'UD0005': '71', 'UD0006': '72',
    'UD0011': '73', 'UD0013': '74', 'UD0014': '75', 'UD0016': '76',
    'UD0023': '77', 'UD0302': '78', 'UD0304': '79', 'UD0308': '80',
    'UD0318': '81', 'UD0322': '82', 'UD0411': '83', 'UD0412': '84',
    'UK0001': '85', 'IN0295': '86', 'IN0306': '87', 'MH0037': '88',
    'NM0239': '89', 'NZ0001': '90', 'SK0035': '91', 'TK0020': '92',
    'UD0028': '93',
}

# Mapping of Japanese pottery types to English
POTTERY_TYPE_MAPPING = {
    '朝日式': 'Asahi type',
    '馬高式': 'Umataka type',
    '栃倉式': 'Tochikura type',
    '三十稲場式': 'Sanjuinaba type',
    '大木7b式': 'Daigi-7b-type',
    '大木8a式': 'Daigi-8a-type',
    '三仏生式': 'Sanbushou type',
    '南三十稲場式': 'Minamisanjuinaba type',
    '沖ノ原式': 'Okinohara type',
    '千石原式': 'Sengokuhara type',
    '新保・新崎式': 'Shinbo-Ninzaki type',
    '焼町系（やけまち）': 'Yakemachi line',
    '室谷上層式（むろや じょうそう）': 'Muroyajoso type',
    '貝殻条痕文（かいがらじょうこんもん）': 'Kaigatrajokonmon',
    '室谷下層式（むろや かそう）': 'Muroyakaso type',
    '卯ノ木式（うのき）:押型文（おしがたもん）': 'Unoki type',
    '河童型（かっぱ）': 'Kappa type'
}


def main():
    df = pd.read_csv(
        r"src\analysis\Deeply_Supervised_Labels-POTTERY_DOGU.csv")
    
    # Select and rename columns
    df = df[[
        "CODE", "HAS_FLAME_LIKE_DECORATION", "HAS_CROWN_LIKE_DECORATION",
        "HAS_HANDLES", "HAS_CORD_MARKED_PATTERN", "HAS_NAIL_ENGRAVING",
        "HAS_SPIRAL_PATTERN", "HAS_FLAT_BASE", "NUMBER_OF_PERTRUSIONS",
        "SHAPE_TYPE"
    ]]
    
    # Rename columns to English
    df.rename(columns={
        'CODE': 'Pottery ID',
        'HAS_FLAME_LIKE_DECORATION': 'Flame-like decoration',
        'HAS_CROWN_LIKE_DECORATION': 'Crown-like decoration',
        'HAS_HANDLES': 'Handles',
        'HAS_CORD_MARKED_PATTERN': 'Cord-marked pattern',
        'HAS_NAIL_ENGRAVING': 'Nail engraving',
        'HAS_SPIRAL_PATTERN': 'Spiral pattern',
        'HAS_FLAT_BASE': 'Flat base'
    }, inplace=True)
    
    # Keep only first 85 rows
    df = df[:85]
    
    # Create one-hot encoding for NUMBER_OF_PERTRUSIONS
    protrusion_dummies = pd.get_dummies(df['NUMBER_OF_PERTRUSIONS'], prefix='Number_of_protrusions')
    
    # Ensure all protrusion columns exist (0, 1, 2, 3, 4, 6, 8)
    all_protrusion_cols = [
        'Number_of_protrusions_0.0', 'Number_of_protrusions_1.0',
        'Number_of_protrusions_2.0', 'Number_of_protrusions_3.0',
        'Number_of_protrusions_4.0', 'Number_of_protrusions_6.0',
        'Number_of_protrusions_8.0'
    ]
    
    for col in all_protrusion_cols:
        if col not in protrusion_dummies.columns:
            protrusion_dummies[col] = 0
    
    # Reorder protrusion columns
    protrusion_dummies = protrusion_dummies[all_protrusion_cols]
    
    # Create one-hot encoding for SHAPE_TYPE
    df['SHAPE_TYPE_ENGLISH'] = df['SHAPE_TYPE'].map(POTTERY_TYPE_MAPPING)
    shape_dummies = pd.get_dummies(df['SHAPE_TYPE_ENGLISH'])
    
    # Ensure all shape type columns exist
    all_shape_types = [
        'Asahi type', 'Umataka type', 'Tochikura type', 'Sanjuinaba type',
        'Daigi-7b-type', 'Daigi-8a-type', 'Sanbushou type',
        'Minamisanjuinaba type', 'Okinohara type', 'Sengokuhara type',
        'Shinbo-Ninzaki type', 'Yakemachi line', 'Muroyajoso type',
        'Kaigatrajokonmon', 'Muroyakaso type', 'Unoki type'
    ]
    
    for shape_type in all_shape_types:
        if shape_type not in shape_dummies.columns:
            shape_dummies[shape_type] = 0
    
    shape_dummies = shape_dummies[all_shape_types]
    
    # Drop the original columns
    df = df.drop(['NUMBER_OF_PERTRUSIONS', 'SHAPE_TYPE', 'SHAPE_TYPE_ENGLISH'], axis=1)
    
    # Concatenate all dataframes
    df_encoded = pd.concat([df, protrusion_dummies, shape_dummies], axis=1)
    
    # Update Pottery ID with assigned numbers
    for i, code in enumerate(df_encoded['Pottery ID']):
        code_clean = str(code).replace('.obj', '')
        assigned_num = ASSIGNED_NUMBERS_DICT.get(code_clean)
        if assigned_num:
            df_encoded.iloc[i, 0] = f"{code_clean}({assigned_num})"
    
    # Convert boolean to 0/1
    binary_cols = list(df_encoded.columns[1:])
    df_encoded[binary_cols] = df_encoded[binary_cols].replace({True: 1, False: 0})
    df_encoded[binary_cols] = df_encoded[binary_cols].astype(int)
    
    print(df_encoded.head())
    print(f"\nTotal columns: {len(df_encoded.columns)}")
    print(f"Column names: {list(df_encoded.columns)}")
    
    df_encoded.to_csv("DS_Labels_Cleaned.csv",
                      index=False)
    print("\nFile saved successfully!")


if __name__ == "__main__":
    main()