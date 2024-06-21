from itertools import permutations
import pandas as pd
import json
import os

FLORES_DIR = "/workspace/floresp-v2.0-rc.2"
SUBDIRS = ['dev', 'devtest']
SAVE_DIR = "/workspace/ALMA/flores-eval"

OPUS_LANGUAGE_IDS = ['ban', 'ace', 'bjn', 'bug', 'min', 'su', 'jv', 'id', 'en']
LANG_IDS = ['ban_Latn', 'ace_Latn', 'bjn_Latn', 'bug_Latn', 'min_Latn', 'sun_Latn', 'jav_Latn', 'ind_Latn', 'eng_Latn']

df = pd.DataFrame()

for sd in SUBDIRS:
    ndf = pd.DataFrame()
    for l in LANG_IDS:
        filename = os.path.join(FLORES_DIR,sd,f'{sd}.{l}')
        with open(filename) as f:
            ndf[l] = f.readlines()
    df = pd.concat([df, ndf])

for idx1, idx2 in permutations(range(len(OPUS_LANGUAGE_IDS)), 2):
    combo_dir = os.path.join(SAVE_DIR, f"{OPUS_LANGUAGE_IDS[idx1]}{OPUS_LANGUAGE_IDS[idx2]}")
    os.makedirs(combo_dir, exist_ok=True)

    with open(os.path.join(combo_dir, f"test.{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}.{OPUS_LANGUAGE_IDS[idx1]}"), 'w') as f:
        f.writelines(df[LANG_IDS[idx1]])
    
    with open(os.path.join(combo_dir, f"test.{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}.{OPUS_LANGUAGE_IDS[idx2]}"), 'w') as f:
        f.writelines(df[LANG_IDS[idx2]])
    
    with open(os.path.join(combo_dir, f"test.{OPUS_LANGUAGE_IDS[idx1]}-{OPUS_LANGUAGE_IDS[idx2]}.json"), 'w') as f:
        json.dump([
            {
                "translation": {
                    OPUS_LANGUAGE_IDS[idx1]: row[LANG_IDS[idx1]],
                    OPUS_LANGUAGE_IDS[idx2]: row[LANG_IDS[idx2]]
                }
            }
            for _,row in df.iterrows()
        ], f)
