# Hateful Memes Classification

[![forthebadge](https://forthebadge.com/images/badges/works-on-my-machine.svg)](https://forthebadge.com)

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

[![forthebadge](https://forthebadge.com/images/badges/powered-by-black-magic.svg)](https://forthebadge.com)

## Introduction

This repo is my bachelor thesis project's source code. It's based on and forked from [vladsandulescu/hatefulmemes](README_BASELINE.md). As in the title, it's about my solution to [hateful memes challenge](https://hatefulmemeschallenge.com/). To come up the solution, I have read all [wining solutions](https://hatefulmemeschallenge.com/#leaderboard)'s publications. The first place and the fifth place solutions insfluenced my work most. Besides those influences, I also came up a **novel** mechanism called Multiple Directional Attention (MDA) to support [UNITER](https://github.com/ChenRocks/UNITER) in utilizing different data channels at once. MDA is the generalization of bidirectional cross-attention of the fifth solution. A data channel in this case is a pair of image and text. Text can be: meme text; caption; paraphrased meme text; context. Image is image feature (including detected objects in image). Unlike the fifth solution only using 2 data channels (`[[img, meme text], [img, caption]]`), 3 data channels were used to improve model performance. As a result, UNITER with a MDA variant achieved 0.8026 AUC ROC and 0.7510 Accuracy which is above 5th place in the challenge. 

Please read [my bachelor thesis]() and [fifth place solution](README_BASELINE.md) publication to understand more.

## Enviroment

One should read the installation scripts and know hardware information (like what kind of GPU, which driver that GPU needs, what are packages versions mentioned in scripts, etc). Don't worry when it doesn't work at the first time. 

This project was carried out on a machine has:
- a GPU Tesla K80 12 GiBs 
- a CPU Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz

On [Debian 10.9](https://www.debian.org/News/2021/20210327). (This is just another way to say I use linux.)

And with the following tools:
- [conda](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#install-linux-silent)
- Python 3.6.13 (installed from conda)

This project was divided into subprojects with different conda enviroments. Therefore, one should **not** put everything in 1 conda environment.

## Project workflow and structure

The project is generally structured as follows:
```
root/
├─ data/
├─ model_asset/
├─ notebooks/
│  ├─ graph.ipynb
├─ py-scripts/
│  ├─ README.md
├─ py-bottom-up-attention/
│  ├─ conda/
│  │  ├─ setup_bua.sh
├─ UNITER/
│  ├─ conda/
│  │  ├─ setup_uniter.sh
│  ├─ storage/
│  │  ├─ pretrained/
```

`root/` is where YOU clone this repo into. Therefore, you might want to rename `root/` to whatever you want. Here I use `root/` for convenience. There are 3 folders ignored by git because they contains large files. 
- `data/` - original dataset and generated dataset
- `model_asset/` - models checkpoints and logs
- `UNITER/storage/pretrained/` - pretrained UNITER core models

Therefore, you should create these directories first for later convenience. 

And we have 3 foreign repos as subprojects:
- [dinhanhx/imgcap/hm_inf_out](https://github.com/dinhanhx/imgcap/tree/master/hm_inf_out)
- [dinhanhx/paraphrased_text_for_hm](https://github.com/dinhanhx/paraphrased_text_for_hm)
- [dinhanhx/performance_calculation_tool_for_hm](https://github.com/dinhanhx/performance_calculation_tool_for_hm)

**Note**: You must **read** files shown in the general structure and README files of foreign repos.

## Dataset preparation

**Shortcut**: go to release section of this repos then download `data.zip` extract files to `data/` folder. There is a `README.md`, please read it carefully.

### Hateful memes challenge dataset

Download the original dataset from [this](https://hatefulmemeschallenge.com/#download) then extract files to `data/` folder. 

Get from [dev_seen_unseen.jsonl](https://drive.google.com/file/d/1e1__LhD9fNBIzgQUQQygbHA0BiCl1nBh/view?usp=sharing) file and place it in the same folder as the other `jsonl` files. 

### Image feature extraction

**Setup**
```bash
# start at root folder
cd py-bottom-up-attention/conda
bash setup.sh
```

**Download** Before run this command, [click me](https://github.com/vladsandulescu/hatefulmemes#1-py-bottom-up-attention)
```base
wget --no-check-certificate http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl -P ~/.torch/fvcore_cache/models/
```
It simply download pretrained model for py-bottom-up-attention

**Generate**
```bash
# In conda enviroment: bua
# In folder: py-bottom-up-attention

# This will extract features for 3 hours
python demo/detectron2_mscoco_proposal_maxnms_hm.py --split img --data_path ../data/ --output_path ../data/imgfeat/ --output_type tsv --min_boxes 10 --max_boxes 100

# This will split for under 30 minutes
python demo/hm.py --split img --split_json_file train.jsonl --d2_file_suffix d2_10-100_vg --data_path ../data/ --output_path ../data/imgfeat/
python demo/hm.py --split img --split_json_file dev_seen_unseen.jsonl --d2_file_suffix d2_10-100_vg --data_path ../data/ --output_path ../data/imgfeat/
python demo/hm.py --split img --split_json_file test_seen.jsonl --d2_file_suffix d2_10-100_vg --data_path ../data/ --output_path ../data/imgfeat/ 
python demo/hm.py --split img --split_json_file test_unseen.jsonl --d2_file_suffix d2_10-100_vg --data_path ../data/ --output_path ../data/imgfeat/ 
```

**Split more**
```bash
# In a conda enviroment that has: python 3.6, pandas, tqdm
# In folder: data

# This simply split big tsv files into smaller npy files
# This will split for under 30 minutes
python spliter.py
```
<details>
    <summary>Click to see spilter.py</summary>

    ```python
    import csv
    import numpy as np
    import base64
    import sys
    import os

    from tqdm import tqdm
    from pathlib import Path
    csv.field_size_limit(sys.maxsize)

    def read_ff(feature_file, test_mode=False):
        TRAIN_VAL_FIELDNAMES = ["id", "img", "label", "text", "img_id", "img_h", "img_w", "objects_id", "objects_conf",
                            "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
        TEST_FIELDNAMES = ["id", "img", "text", "img_id", "img_h", "img_w", "objects_id", "objects_conf",
                    "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

        with open(feature_file, mode='r', encoding='utf8') as f:
            tsv_reader = csv.DictReader(f, delimiter='\t',
                                        fieldnames=TRAIN_VAL_FIELDNAMES if not test_mode else TEST_FIELDNAMES)
            data = []
            for item in tsv_reader:
                try:
                    idb = {'img_id': str(item['img_id']),
                            'img': str(item['img']),
                            'text': str(item['text']),
                            'label': int(item['label']) if not test_mode else None,
                            'img_h': int(item['img_h']),
                            'img_w': int(item['img_w']),
                            'num_boxes': int(item['num_boxes']),
                            'boxes': np.frombuffer(base64.decodebytes(item['boxes'].encode()),
                                                    dtype=np.float32).reshape((int(item['num_boxes']), -1)),
                            
                            'features': np.frombuffer(base64.decodebytes(item['features'].encode()),
                                                        dtype=np.float32).reshape((int(item['num_boxes']), -1))}
                    data.append(idb)
                except:
                    print(f"Some error occurred reading img id {item['img_id']}")

            return data

    def split(data, folder_name, test_mode=False):
        with open(f"map_{folder_name}.tsv", mode='w', encoding='utf8') as f:
            TRAIN_VAL_FIELDNAMES = ["img", "label", "text", "img_id", "img_h", "img_w", "num_boxes", "npy"]
            TEST_FIELDNAMES =      ["img",          "text", "img_id", "img_h", "img_w", "num_boxes", "npy"]

            tsv_writer = csv.DictWriter(f, fieldnames=TRAIN_VAL_FIELDNAMES if not test_mode else TEST_FIELDNAMES, delimiter='\t')

            tsv_writer.writeheader()
            for d in tqdm(data):
                if test_mode:
                    tsv_writer.writerow({'img_id': d['img_id'], 
                                            'img': d['img'],
                                            'text': d['text'],
                                            'img_h': d['img_h'],
                                            'img_w': d['img_w'],
                                            'num_boxes': d['num_boxes'], 
                                            'npy': f"{folder_name}/{d['img_id']}.npy"})
                else:
                    tsv_writer.writerow({'img_id': d['img_id'], 
                                            'img': d['img'],
                                            'text': d['text'],
                                            'label': d['label'],
                                            'img_h': d['img_h'],
                                            'img_w': d['img_w'],
                                            'num_boxes': d['num_boxes'], 
                                            'npy': f"{folder_name}/{d['img_id']}.npy"})

                np.save(f"{folder_name}/{d['img_id']}.npy", d)

        return len(os.listdir(folder_name))

    if '__main__' == __name__:
        feature_files = ['data_train_d2_10-100_vg.tsv', 
                            'data_dev_seen_unseen_d2_10-100_vg.tsv', 
                            'data_test_seen_d2_10-100_vg.tsv',
                            'data_test_unseen_d2_10-100_vg.tsv']
        for ff in feature_files:
            print(Path(ff).exists())

        data_ff = []
        for ff in feature_files:
            if ff == 'data_test_unseen_d2_10-100_vg.tsv':
                data_ff.append(read_ff(ff, True))
            else:
                data_ff.append(read_ff(ff))

        folders = ['data_train_d2_10-100_vg', 
                    'data_dev_seen_unseen_d2_10-100_vg', 
                    'data_test_seen_d2_10-100_vg',
                    'data_test_unseen_d2_10-100_vg']

        for data, folder in zip(data_ff, folders):
            if 'data_test_unseen_d2_10-100_vg' == folder or 'data_test_seen_d2_10-100_vg' == folder:
                print(split(data, folder, True))
            else: 
                print(split(data, folder, False))
    ```
</details>

### Image captioning

Download all 3 `csv` files from [dinhanhx/imgcap/hm_inf_out](https://github.com/dinhanhx/imgcap/tree/master/hm_inf_out). If you want to reproduce these files, please read [this](https://github.com/dinhanhx/imgcap#image-captioning-with-visual-attention) and [that](https://github.com/dinhanhx/imgcap/tree/master/hm_inf_out#regenerate-captions). 

Then place those files into `root/data/imgcap/` folder

### Text paraphrasing 

Download all `jsonl` files staring with `data_test_paraphrased_nlpaug_` from [dinhanhx/paraphrased_text_for_hm](https://github.com/dinhanhx/paraphrased_text_for_hm).

Then place those files into `root/data/textaug/`.

### Context addition

Download [annotations](https://drive.google.com/file/d/1NTaDqL2hPFGRZywBqDwkUBVfOq_0DMDy/view?usp=sharing) files from [the first place solution](https://github.com/HimariO/HatefulMemesChallenge/blob/main/data_utils/README.md).

Then place those files `jsonl` files into `root/data/HimariO_annotations/` folder

Then place `preprocess.py` into the same folder and read it then run it (in a conda enviroment that has python 3.6 and pandas)
<details>
    <summary>Click to see preprocess.py</summary>

    ```python
    import pandas as pd
    from pandas.core.common import flatten
    import json

    def load_jsonl(filename):
        data = []
        with open(filename, 'r') as fobj:
            for line in fobj:
                d = json.loads(line)
                data.append({'id': d['id'],
                                'img': d['img'],
                                'partition_description': ' '.join(list(flatten(d['partition_description'])))
                            })
            return pd.DataFrame.from_records(data)


    if '__main__' == __name__:
        train_dev_all_df = load_jsonl('train_dev_all.entity.jsonl')
        test_seen_df = load_jsonl('test_seen.entity.jsonl')
        test_unseen_df = load_jsonl('test_unseen.entity.jsonl')

        data_test_df = train_dev_all_df.merge(test_seen_df, how='outer').merge(test_unseen_df, how='outer')
        data_test_df['id'] = data_test_df['id'].apply(lambda x: str(x).zfill(5))
        data_test_df.to_json('data_test.jsonl', orient='records', lines=True)
    ```
</details>

### Double check

The `data/` folder should look like this:
```
imgfeat/
data_dev_seen_unseen_d2_10-100_vg.tsv
data_test_seen_d2_10-100_vg.tsv
data_test_unseen_d2_10-100_vg.tsv
data_train_d2_10-100_vg.tsv
tiny_data_dev_seen_unseen_d2_10-100_vg.tsv
tiny_data_test_seen_d2_10-100_vg.tsv
tiny_data_test_unseen_d2_10-100_vg.tsv
tiny_data_train_d2_10-100_vg.tsv

spliter.py
map_data_dev_seen_unseen_d2_10-100_vg
map_data_test_seen_d2_10-100_vg
map_data_test_unseen_d2_10-100_vg
map_data_train_d2_10-100_vg
data_dev_seen_unseen_d2_10-100_vg/
data_test_seen_d2_10-100_vg/
data_test_unseen_d2_10-100_vg/
data_train_d2_10-100_vg/

HimariO_annotations/

imgcap/

textaug/

img/                  - the PNG images
train.jsonl           - the training set
dev_seen.jsonl        - the development set for Phase 1
dev_unseen.jsonl      - the development set for Phase 2
dev_seen_unseen.jsonl - the combined development set
test_seen.jsonl       - the test set for Phase 1
test_unseen.jsonl     - the test set for Phase 2
README.md
LICENSE.txt
```

## Result reproduction

