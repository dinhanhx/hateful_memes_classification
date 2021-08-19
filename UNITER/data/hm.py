import os
import base64
import numpy as np
import csv
import sys

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import pandas as pd

from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer

from .data import (pad_tensors, get_gather_index, default_none_dict)

csv.field_size_limit(sys.maxsize)
TRAIN_VAL_FIELDNAMES = ["id", "img", "label", "text", "img_id", "img_h", "img_w", "objects_id", "objects_conf",
                        "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
TEST_FIELDNAMES = ["id", "img", "text", "img_id", "img_h", "img_w", "objects_id", "objects_conf",
                   "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids


class HMDataset(Dataset):
    def __init__(self, image_set, root_path, data_path, use_img_type, boxes="36",
                 test_mode=False, **kwargs):
        super(HMDataset, self).__init__()

        # # Phase 1
        # precomputed_boxes = {
        #     "train": "data_train_d2_36-36_batch.tsv",
        #     "dev": "data_dev_d2_36-36_batch.tsv",
        #     "test": "data_test_d2_36-36_batch.tsv",
        # }

        # # Phase seen
        precomputed_boxes = {
            "train": "data_train_d2_10-100_vg.tsv",
            "dev": "data_dev_seen_unseen_d2_10-100_vg.tsv",
            "test": "data_test_seen_d2_10-100_vg.tsv",
        }
        print(precomputed_boxes)

        self.boxes = boxes
        self.test_mode = test_mode
        self.use_img_type = use_img_type
        self.data_path = data_path
        self.image_sets = [iset.strip() for iset in image_set.split('+')]
        self.precomputed_box_files = [
            os.path.join(data_path, precomputed_boxes[iset])
            for iset in self.image_sets]
        self.box_bank = {}
        self.cache_dir = os.path.join(root_path, 'cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.toker = BertTokenizer.from_pretrained(
            'bert-base-cased', do_lower_case='uncased' in 'bert-base-cased', cache_dir=self.cache_dir)
        self.cls_token_id = self.toker.convert_tokens_to_ids(['[CLS]'])[0]
        self.sep_token_id = self.toker.convert_tokens_to_ids(['[SEP]'])[0]
        self.database = self.load_precomputed_boxes(self.precomputed_box_files)
        self.lens = [len(bert_tokenize(self.toker, item['text'])) + int(item['num_boxes']) for item in self.database]

    def __getitem__(self, index):
        idb = self.database[index]

        # text
        # tokens = self.tokenizer.tokenize('[CLS] ' + idb['text'] + ' [SEP]')
        text_ids = [self.cls_token_id] + bert_tokenize(self.toker, idb['text']) + [self.sep_token_id]
        text_ids = torch.tensor(text_ids)
        # text_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))

        # image
        w0, h0 = idb['img_h'], idb['img_w']
        num_boxes = idb['num_boxes']
        boxes = torch.as_tensor(idb['boxes'])
        img_features = torch.as_tensor(idb['features'])

        # normalize boxes coordinates
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / w0
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / h0

        # 7-d ROI location features
        img_pos_features = torch.cat([boxes,
                                      # box width
                                      boxes[:, 2:3] - boxes[:, 0:1],
                                      # box height
                                      boxes[:, 3:4] - boxes[:, 1:2],
                                      # box area
                                      (boxes[:, 2:3] - boxes[:, 0:1]) *
                                      (boxes[:, 3:4] - boxes[:, 1:2])], dim=-1)

        # attention masks
        attn_masks = torch.tensor([1] * (len(text_ids) + num_boxes))

        # label
        label = torch.tensor(idb['label']) if not self.test_mode else None

        if self.use_img_type:
            img_type_ids = torch.tensor([index + 1] * num_boxes)
        else:
            img_type_ids = None

        return tuple([(text_ids, img_features, img_pos_features,
                      attn_masks, img_type_ids)]), label

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    def load_precomputed_boxes(self, box_files):
        data = []
        for box_file in box_files:
            data.extend(self.load_precomputed_boxes_from_file(box_file))
        return data

    def load_precomputed_boxes_from_file(self, box_file):
        if box_file in self.box_bank:
            return self.box_bank[box_file]
        else:
            in_data = []
            with open(box_file, "r", encoding='utf8') as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t',
                                        fieldnames=TRAIN_VAL_FIELDNAMES if not self.test_mode else TEST_FIELDNAMES)
                for item in reader:
                    try:
                        idb = {'img_id': str(item['img_id']),
                               'img': str(item['img']),
                               'text': str(item['text']),
                               'label': int(item['label']) if not self.test_mode else None,
                               'img_h': int(item['img_h']),
                               'img_w': int(item['img_w']),
                               'num_boxes': int(item['num_boxes']),
                               'boxes': np.frombuffer(base64.decodebytes(item['boxes'].encode()),
                                                      dtype=np.float32).reshape((int(item['num_boxes']), -1)),
                               'features': np.frombuffer(base64.decodebytes(item['features'].encode()),
                                                         dtype=np.float32).reshape((int(item['num_boxes']), -1))
                               }
                        in_data.append(idb)
                    except:
                        print('Some error occured reading img id {}, skipping it.'.format(str(item['img_id'])))
            self.box_bank[box_file] = in_data

            return in_data

    def __len__(self):
        return len(self.database)


def hm_collate(inputs, test_mode=False):
    (input_ids, img_feats, img_pos_feats, attn_masks,
     img_type_ids) = map(list, unzip(concat(outs for outs, _ in inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    if img_type_ids[0] is None:
        img_type_ids = None
    else:
        img_type_ids = pad_sequence(img_type_ids,
                                    batch_first=True, padding_value=0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    if not test_mode:
        targets = torch.Tensor([t for _, t in inputs]).long()

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    if not test_mode:
        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'gather_index': gather_index,
                 'img_type_ids': img_type_ids,
                 'targets': targets}
    else:
        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'gather_index': gather_index,
                 'img_type_ids': img_type_ids}
    batch = default_none_dict(batch)

    return batch


class HMEvalDataset(HMDataset):
    def __getitem__(self, i):
        img_id = self.database[i]['img_id']
        outs, targets = super().__getitem__(i)
        return img_id, outs, targets


def hm_eval_collate(inputs):
    img_ids, batch = [], []
    for id_, *tensors in inputs:
        img_ids.append(id_)
        batch.append(tensors)
    batch = hm_collate(batch)
    batch['img_ids'] = img_ids
    return batch


class HMTestDataset(HMDataset):
    def __getitem__(self, i):
        img_id = self.database[i]['img_id']
        outs, _ = super().__getitem__(i)
        return img_id, outs, _


def hm_test_collate(inputs):
    img_ids, batch = [], []
    for id_, *tensors in inputs:
        img_ids.append(id_)
        batch.append(tensors)
    batch = hm_collate(batch, test_mode=True)
    batch['img_ids'] = img_ids
    return batch


class HMPairedDataset(Dataset):
    def __init__(self, image_set, root_path, data_path, 
                precomputed_boxes, captions_file, paraphrased_file,
                use_img_type, boxes="36",
                test_mode=False, **kwargs):
        super(HMPairedDataset, self).__init__()

        print(f'>>> {precomputed_boxes}; {captions_file}')

        df_captions = pd.read_csv(os.path.join(data_path, captions_file))
        df_captions['id'] = df_captions['id'].str.replace('.png', '')
        self.df_captions = df_captions[['id', 'caption']]

        self.boxes = boxes
        self.test_mode = test_mode
        self.use_img_type = use_img_type
        self.data_path = data_path
        self.image_sets = [iset.strip() for iset in image_set.split('+')]
        self.precomputed_box_files = [
            os.path.join(data_path, precomputed_boxes[iset])
            for iset in self.image_sets]
        self.box_bank = {}
        self.cache_dir = os.path.join(root_path, 'cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.toker = BertTokenizer.from_pretrained(
            'bert-base-cased', do_lower_case='uncased' in 'bert-base-cased', cache_dir=self.cache_dir)
        # self.tokenizer = bert_tokenize(self.toker)
        self.cls_token_id = self.toker.convert_tokens_to_ids(['[CLS]'])[0]
        self.sep_token_id = self.toker.convert_tokens_to_ids(['[SEP]'])[0]
        self.database = self.load_precomputed_boxes(self.precomputed_box_files)
        self.lens = [len(bert_tokenize(self.toker, item['text'])) +
                     len(bert_tokenize(self.toker, self.get_im2txt(item['img_id']))) +
                     2*int(item['num_boxes']) for item in self.database]

    def get_im2txt(self, img_id):
        img_id = str(img_id).zfill(5)
        return self.df_captions.query('id==@img_id')[['caption']].values[0][0]

    def __getitem__(self, index):
        """
        [[txt1, img],
        [txt2, img]]
        """
        idb = self.database[index]

        # label
        label = torch.tensor(idb['label']) if not self.test_mode else None
        outs = []

        # text
        # tokens = self.tokenizer.tokenize('[CLS] ' + idb['text'] + ' [SEP]')
        text_ids = [self.cls_token_id] + bert_tokenize(self.toker, idb['text']) + [self.sep_token_id]
        text_ids = torch.tensor(text_ids)

        # im2txt inferred caption
        im2txt_caption = self.get_im2txt(idb['img_id'])
        im2text_ids = [self.cls_token_id] + \
                      bert_tokenize(self.toker, im2txt_caption) + \
                      [self.sep_token_id]
        im2text_ids = torch.tensor(im2text_ids)

        # image
        w0, h0 = idb['img_h'], idb['img_w']
        num_boxes = idb['num_boxes']
        boxes = torch.as_tensor(idb['boxes'])
        img_features = torch.as_tensor(idb['features'])

        # normalize boxes coordinates
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / w0
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / h0

        # 7-d ROI location features
        img_pos_features = torch.cat([boxes,
                                      # box width
                                      boxes[:, 2:3] - boxes[:, 0:1],
                                      # box height
                                      boxes[:, 3:4] - boxes[:, 1:2],
                                      # box area
                                      (boxes[:, 2:3] - boxes[:, 0:1]) *
                                      (boxes[:, 3:4] - boxes[:, 1:2])], dim=-1)

        # attention masks
        attn_masks = torch.tensor([1] * (len(text_ids) + num_boxes))
        attn_masks_im2txt = torch.tensor([1] * (len(im2text_ids) + num_boxes))

        if self.use_img_type:
            img_type_ids = torch.tensor([index + 1] * num_boxes)
        else:
            img_type_ids = None

        outs.append((text_ids, img_features, img_pos_features,
                      attn_masks, img_type_ids))
        outs.append((im2text_ids, img_features, img_pos_features,
                      attn_masks_im2txt, img_type_ids))
        return tuple(outs), label

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    def load_precomputed_boxes(self, box_files):
        data = []
        for box_file in box_files:
            data.extend(self.load_precomputed_boxes_from_file(box_file))
        return data

    def load_precomputed_boxes_from_file(self, box_file):
        if box_file in self.box_bank:
            return self.box_bank[box_file]
        else:
            in_data = []
            with open(box_file, "r", encoding='utf8') as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t',
                                        fieldnames=TRAIN_VAL_FIELDNAMES if not self.test_mode else TEST_FIELDNAMES)
                for item in reader:
                    try:
                        idb = {'img_id': str(item['img_id']),
                               'img': str(item['img']),
                               'text': str(item['text']),
                               'label': int(item['label']) if not self.test_mode else None,
                               'img_h': int(item['img_h']),
                               'img_w': int(item['img_w']),
                               'num_boxes': int(item['num_boxes']),
                               'boxes': np.frombuffer(base64.decodebytes(item['boxes'].encode()),
                                                      dtype=np.float32).reshape((int(item['num_boxes']), -1)),
                               'features': np.frombuffer(base64.decodebytes(item['features'].encode()),
                                                         dtype=np.float32).reshape((int(item['num_boxes']), -1))
                               }
                        in_data.append(idb)
                    except:
                        print('Some error occured reading img id {}, skipping it.'.format(str(item['img_id'])))
            self.box_bank[box_file] = in_data

            return in_data

    def __len__(self):
        return len(self.database)


def hm_paired_collate(inputs, test_mode=False):
    (input_ids, img_feats, img_pos_feats, attn_masks,
     img_type_ids) = map(list, unzip(concat(outs for outs, _ in inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    if img_type_ids[0] is None:
        img_type_ids = None
    else:
        img_type_ids = pad_sequence(img_type_ids,
                                    batch_first=True, padding_value=0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    if not test_mode:
        targets = torch.Tensor([t for _, t in inputs]).long()

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    if not test_mode:
        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'gather_index': gather_index,
                 'img_type_ids': img_type_ids,
                 'targets': targets}
    else:
        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'gather_index': gather_index,
                 'img_type_ids': img_type_ids}
    batch = default_none_dict(batch)

    return batch


class HMPairedEvalDataset(HMPairedDataset):
    def __getitem__(self, i):
        img_id = self.database[i]['img_id']
        outs, targets = super().__getitem__(i)
        return img_id, outs, targets


def hm_paired_eval_collate(inputs):
    img_ids, batch = [], []
    for id_, *tensors in inputs:
        img_ids.append(id_)
        batch.append(tensors)
    batch = hm_paired_collate(batch)
    batch['img_ids'] = img_ids
    return batch


class HMPairedTestDataset(HMPairedDataset):
    def __getitem__(self, i):
        img_id = self.database[i]['img_id']
        outs, _ = super().__getitem__(i)
        return img_id, outs, _


def hm_paired_test_collate(inputs):
    img_ids, batch = [], []
    for id_, *tensors in inputs:
        img_ids.append(id_)
        batch.append(tensors)
    batch = hm_paired_collate(batch, test_mode=True)
    batch['img_ids'] = img_ids
    return batch


class HMTripleDataset(Dataset):
    """This Dataset class is meant for any model that can take triple format
        Triple format: [(img, label), (img, caption), (img, paraphrased label)]

        Idea by Vu Dinh Anh
    """

    def __init__(self, image_set, root_path, data_path,
                precomputed_boxes, captions_file, paraphrased_file,
                use_img_type, boxes="36",
                test_mode=False, **kwargs):
        super(HMTripleDataset, self).__init__()

        # Data files
        print(f'>>> {precomputed_boxes}; {captions_file}; {paraphrased_file}')

        # Load caption file
        df_captions = pd.read_csv(os.path.join(data_path, captions_file))
        df_captions['id'] = df_captions['id'].str.replace('.png', '')
        self.df_captions = df_captions[['id', 'caption']]

        # Load paraphrased file
        df_paraphrased = pd.read_json(os.path.join(data_path, paraphrased_file), lines=True)
        self.df_paraphrased = df_paraphrased[['id', 'paraphrased_text']]

        # Bunch of things that I don't know nor understand
        self.boxes = boxes
        self.test_mode = test_mode
        self.use_img_type = use_img_type
        self.data_path = data_path
        self.image_sets = [iset.strip() for iset in image_set.split('+')]
        self.precomputed_box_files = [
            os.path.join(data_path, precomputed_boxes[iset])
            for iset in self.image_sets]
        self.box_bank = {}

        self.cache_dir = os.path.join(root_path, 'cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.toker = BertTokenizer.from_pretrained(
            'bert-base-cased', do_lower_case='uncased' in 'bert-base-cased', cache_dir=self.cache_dir)

        self.cls_token_id = self.toker.convert_tokens_to_ids(['[CLS]'])[0]
        self.sep_token_id = self.toker.convert_tokens_to_ids(['[SEP]'])[0]
        self.database = self.load_precomputed_boxes(self.precomputed_box_files)

        self.lens = [len(bert_tokenize(self.toker, item['text'])) +
                     len(bert_tokenize(self.toker, self.get_im2txt(item['img_id']))) + 
                     len(bert_tokenize(self.toker, self.get_paraphrased(item['img_id']))) +
                     2*int(item['num_boxes']) for item in self.database]

    
    def get_im2txt(self, img_id):
        img_id = str(img_id).zfill(5)
        return self.df_captions.query('id==@img_id')[['caption']].values[0][0]


    def get_paraphrased(self, img_id):
        img_id = str(img_id).zfill(5)
        return self.df_paraphrased.query('id==@img_id')[['paraphrased_text']].values[0][0]


    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())


    def load_precomputed_boxes(self, box_files):
        data = []
        for box_file in box_files:
            data.extend(self.load_precomputed_boxes_from_file(box_file))
        return data


    def load_precomputed_boxes_from_file(self, box_file):
        if box_file in self.box_bank:
            return self.box_bank[box_file]
        else:
            in_data = []
            with open(box_file, "r", encoding='utf8') as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t',
                                        fieldnames=TRAIN_VAL_FIELDNAMES if not self.test_mode else TEST_FIELDNAMES)
                for item in reader:
                    try:
                        idb = {'img_id': str(item['img_id']),
                               'img': str(item['img']),
                               'text': str(item['text']),
                               'label': int(item['label']) if not self.test_mode else None,
                               'img_h': int(item['img_h']),
                               'img_w': int(item['img_w']),
                               'num_boxes': int(item['num_boxes']),
                               'boxes': np.frombuffer(base64.decodebytes(item['boxes'].encode()),
                                                      dtype=np.float32).reshape((int(item['num_boxes']), -1)),
                               'features': np.frombuffer(base64.decodebytes(item['features'].encode()),
                                                         dtype=np.float32).reshape((int(item['num_boxes']), -1))
                               }
                        in_data.append(idb)
                    except:
                        print('Some error occured reading img id {}, skipping it.'.format(str(item['img_id'])))
            self.box_bank[box_file] = in_data

            return in_data

        
    def __len__(self):
        return len(self.database)


    def __getitem__(self, index):
        idb = self.database[index]

        # label: 0 ~ non-hateful, 1 ~ hateful
        label = torch.tensor(idb['label']) if not self.test_mode else None

        # text
        text_ids = [self.cls_token_id] + bert_tokenize(self.toker, idb['text']) + [self.sep_token_id]
        text_ids = torch.tensor(text_ids)

        # caption
        im2txt_caption = self.get_im2txt(idb['img_id'])
        im2text_ids = [self.cls_token_id] + \
                      bert_tokenize(self.toker, im2txt_caption) + \
                      [self.sep_token_id]
        im2text_ids = torch.tensor(im2text_ids)

        # paraphrased
        paraphrased = self.get_paraphrased(idb['img_id'])
        paraphrased_ids = [self.cls_token_id] + \
                            bert_tokenize(self.toker, paraphrased) + \
                            [self.sep_token_id]
        paraphrased_ids = torch.tensor(paraphrased_ids)

        # image
        w0, h0 = idb['img_h'], idb['img_w']
        num_boxes = idb['num_boxes']
        boxes = torch.as_tensor(idb['boxes'])
        img_features = torch.as_tensor(idb['features'])

        # normalize boxes coordinates
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / w0
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / h0

        # 7-d ROI location features
        img_pos_features = torch.cat([boxes,
                                      # box width
                                      boxes[:, 2:3] - boxes[:, 0:1],
                                      # box height
                                      boxes[:, 3:4] - boxes[:, 1:2],
                                      # box area
                                      (boxes[:, 2:3] - boxes[:, 0:1]) *
                                      (boxes[:, 3:4] - boxes[:, 1:2])], dim=-1)

        # attention masks
        attn_masks = torch.tensor([1] * (len(text_ids) + num_boxes))
        attn_masks_im2txt = torch.tensor([1] * (len(im2text_ids) + num_boxes))
        attn_masks_paraphrased = torch.tensor([1] * (len(paraphrased_ids) + num_boxes))

        if self.use_img_type:
            img_type_ids = torch.tensor([index + 1] * num_boxes)
        else:
            img_type_ids = None

        outs = ((text_ids, img_features, img_pos_features, attn_masks, img_type_ids),
                (im2text_ids, img_features, img_pos_features, attn_masks_im2txt, img_type_ids),
                (paraphrased_ids, img_features, img_pos_features, attn_masks_paraphrased, img_type_ids))

        return outs, label

def hm_triple_collate(inputs, test_mode=False):
    return hm_paired_collate(inputs=inputs, test_mode=test_mode)


class HMTripleEvalDataset(HMTripleDataset):
    def __getitem__(self, i):
        img_id = self.database[i]['img_id']
        outs, targets = super().__getitem__(i)
        return img_id, outs, targets


def hm_triple_eval_collate(inputs):
    return hm_paired_eval_collate(inputs)


class HMTripleTestDataset(HMTripleDataset):
    def __getitem__(self, i):
        img_id = self.database[i]['img_id']
        outs, _ = super().__getitem__(i)
        return img_id, outs, _


def hm_triple_test_collate(inputs):
    return hm_paired_test_collate(inputs)


class HMQuadDataset(Dataset):
    """This Dataset class is meant for any model that can take quadruple format
        Quadruple format: [(img, label), 
                        (img, caption),
                        (img, knowledge), 
                        (img, paraphrased label)]

        Idea by Vu Dinh Anh
    """

    def __init__(self, image_set, root_path, data_path,
                precomputed_boxes, captions_file, knowledge_file, paraphrased_file,
                use_img_type, boxes="36",
                test_mode=False, **kwargs):
        super(HMQuadDataset, self).__init__()

        # Data files
        print(f'>>> {precomputed_boxes}; {captions_file}; {knowledge_file}; {paraphrased_file}')

        # Load caption file
        df_captions = pd.read_csv(os.path.join(data_path, captions_file))
        df_captions['id'] = df_captions['id'].str.replace('.png', '')
        self.df_captions = df_captions[['id', 'caption']]

        # Load knowledge file
        df_knowledge = pd.read_json(os.path.join(data_path, knowledge_file), lines=True)
        self.df_knowledge = df_knowledge[['id', 'partition_description']]

        # Load paraphrased file
        df_paraphrased = pd.read_json(os.path.join(data_path, paraphrased_file), lines=True)
        self.df_paraphrased = df_paraphrased[['id', 'paraphrased_text']]

        # Bunch of things that I know or understand a little bit
        self.boxes = boxes
        self.test_mode = test_mode
        self.use_img_type = use_img_type
        self.data_path = data_path
        self.image_sets = [iset.strip() for iset in image_set.split('+')]
        self.precomputed_box_files = [
            os.path.join(data_path, precomputed_boxes[iset])
            for iset in self.image_sets]
        self.box_bank = {}

        self.cache_dir = os.path.join(root_path, 'cache')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.toker = BertTokenizer.from_pretrained(
            'bert-base-cased', do_lower_case='uncased' in 'bert-base-cased', cache_dir=self.cache_dir)

        self.cls_token_id = self.toker.convert_tokens_to_ids(['[CLS]'])[0]
        self.sep_token_id = self.toker.convert_tokens_to_ids(['[SEP]'])[0]
        self.database = self.load_precomputed_boxes(self.precomputed_box_files)

        self.lens = [len(bert_tokenize(self.toker, item['text'])) +
                     len(bert_tokenize(self.toker, self.get_im2txt(item['img_id']))) + 
                     len(bert_tokenize(self.toker, self.get_paraphrased(item['img_id']))) +
                     2*int(item['num_boxes']) for item in self.database]


    def get_im2txt(self, img_id):
        img_id = str(img_id).zfill(5)
        return self.df_captions.query('id==@img_id')[['caption']].values[0][0]


    def get_knowledge(self, img_id):
        img_id = str(img_id).zfill(5)
        return self.df_knowledge.query('id==@img_id')[['partition_description']].values[0][0]


    def get_paraphrased(self, img_id):
        img_id = str(img_id).zfill(5)
        return self.df_paraphrased.query('id==@img_id')[['paraphrased_text']].values[0][0]


    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())


    def load_precomputed_boxes(self, box_files):
        data = []
        for box_file in box_files:
            data.extend(self.load_precomputed_boxes_from_file(box_file))
        return data


    def load_precomputed_boxes_from_file(self, box_file):
        if box_file in self.box_bank:
            return self.box_bank[box_file]
        else:
            in_data = []
            with open(box_file, "r", encoding='utf8') as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t',
                                        fieldnames=TRAIN_VAL_FIELDNAMES if not self.test_mode else TEST_FIELDNAMES)
                for item in reader:
                    try:
                        idb = {'img_id': str(item['img_id']),
                               'img': str(item['img']),
                               'text': str(item['text']),
                               'label': int(item['label']) if not self.test_mode else None,
                               'img_h': int(item['img_h']),
                               'img_w': int(item['img_w']),
                               'num_boxes': int(item['num_boxes']),
                               'boxes': np.frombuffer(base64.decodebytes(item['boxes'].encode()),
                                                      dtype=np.float32).reshape((int(item['num_boxes']), -1)),
                               'features': np.frombuffer(base64.decodebytes(item['features'].encode()),
                                                         dtype=np.float32).reshape((int(item['num_boxes']), -1))
                               }
                        in_data.append(idb)
                    except:
                        print('Some error occured reading img id {}, skipping it.'.format(str(item['img_id'])))
            self.box_bank[box_file] = in_data

            return in_data

        
    def __len__(self):
        return len(self.database)


    def __getitem__(self, index):
        idb = self.database[index]

        # label: 0 ~ non-hateful, 1 ~ hateful
        label = torch.tensor(idb['label']) if not self.test_mode else None

        # text
        text_ids = [self.cls_token_id] + bert_tokenize(self.toker, idb['text']) + [self.sep_token_id]
        text_ids = torch.tensor(text_ids)

        # caption
        im2txt_caption = self.get_im2txt(idb['img_id'])
        im2text_ids = [self.cls_token_id] + \
                      bert_tokenize(self.toker, im2txt_caption) + \
                      [self.sep_token_id]
        im2text_ids = torch.tensor(im2text_ids)

        # knowledge
        knowledge = self.get_knowledge(idb['img_id'])
        knowledge_ids = [self.cls_token_id] + \
                        bert_tokenize(self.toker, knowledge) + \
                        [self.sep_token_id]
        knowledge_ids = torch.tensor(knowledge_ids)

        # paraphrased
        paraphrased = self.get_paraphrased(idb['img_id'])
        paraphrased_ids = [self.cls_token_id] + \
                            bert_tokenize(self.toker, paraphrased) + \
                            [self.sep_token_id]
        paraphrased_ids = torch.tensor(paraphrased_ids)

        # image
        w0, h0 = idb['img_h'], idb['img_w']
        num_boxes = idb['num_boxes']
        boxes = torch.as_tensor(idb['boxes'])
        img_features = torch.as_tensor(idb['features'])

        # normalize boxes coordinates
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / w0
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / h0

        # 7-d ROI location features
        img_pos_features = torch.cat([boxes,
                                      # box width
                                      boxes[:, 2:3] - boxes[:, 0:1],
                                      # box height
                                      boxes[:, 3:4] - boxes[:, 1:2],
                                      # box area
                                      (boxes[:, 2:3] - boxes[:, 0:1]) *
                                      (boxes[:, 3:4] - boxes[:, 1:2])], dim=-1)
        
        # attention masks
        attn_masks = torch.tensor([1] * (len(text_ids) + num_boxes))
        attn_masks_im2txt = torch.tensor([1] * (len(im2text_ids) + num_boxes))
        attn_masks_knowledge = torch.tensor([1] * (len(knowledge_ids) + num_boxes))
        attn_masks_paraphrased = torch.tensor([1] * (len(paraphrased_ids) + num_boxes))

        if self.use_img_type:
            img_type_ids = torch.tensor([index + 1] * num_boxes)
        else:
            img_type_ids = None

        outs = ((text_ids, img_features, img_pos_features, attn_masks, img_type_ids),
                (im2text_ids, img_features, img_pos_features, attn_masks_im2txt, img_type_ids),
                (knowledge_ids, img_features, img_pos_features, attn_masks_knowledge, img_type_ids),
                (paraphrased_ids, img_features, img_pos_features, attn_masks_paraphrased, img_type_ids))

        return outs, label

def hm_quad_collate(inputs, test_mode=False):
    return hm_paired_collate(inputs=inputs, test_mode=test_mode)


class HMQuadEvalDataset(HMQuadDataset):
    def __getitem__(self, i):
        img_id = self.database[i]['img_id']
        outs, targets = super().__getitem__(i)
        return img_id, outs, targets


def hm_quad_eval_collate(inputs):
    return hm_paired_eval_collate(inputs)


class HMQuadTestDataset(HMQuadDataset):
    def __getitem__(self, i):
        img_id = self.database[i]['img_id']
        outs, _ = super().__getitem__(i)
        return img_id, outs, _


def hm_quad_test_collate(inputs):
    return hm_paired_test_collate(inputs)
