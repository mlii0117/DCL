import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import json


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.args = args
        if args.dataset_name == 'iu_xray':
            self.ann_path = args.ann_path.split('&')[0]
            self.image_dir = args.image_dir.split('&')[0]
        elif args.dataset_name == 'mimic_cxr':
            self.ann_path = args.ann_path.split('&')[1]
            self.image_dir = args.image_dir.split('&')[1]
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        with open(self.ann_path, encoding='utf-8') as f:
            self.ann = json.load(f)
            f.close()
        # self.ann = pd.read_csv(self.ann_path)
        self.examples = self.ann[self.split]
        # self.examples = self.ann[self.ann['Split'] == self.split]
        # reset the index, and each split has one dataframe
        # self.examples.reset_index(drop=True, inplace=True)
        ## create a dict to store the mask
        self.masks = []
        self.reports = []
        self.prompt = 'a picture of '

        for i in range(len(self.examples)):
            if self.split == 'train':
                caption = tokenizer(self.prompt + self.examples[i]['report'])[:self.max_seq_length]
            else:
                caption = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            # caption = tokenizer(self.examples[i]['report'], padding='longest', max_length=250, return_tensors="pt")
            # caption = caption['input_ids'][:self.max_seq_length]
            self.examples[i]['ids'] = caption
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            # self.masks.append([1] * len(self.reports[i]))

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        # example = self.examples.loc[idx]
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']     #ids?
        report_masks = example['mask']      #mask?
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample

class CHXrayDataSet2(Dataset):
    def __init__(self, opt, split, transform=None):
        self.transform = transform

        self.data_dir = opt.data_dir
        # TODO
        # self.pkl_dir = os.path.join('/media/data1/jiangzixiao/medical_pretrain/data/wingspan', 'processed_new_papers_extra_tag')
        # self.pkl_dir = os.path.join('/media/data1/jiangzixiao/medical_pretrain/data/wingspan', 'processed_new_papers_tag')
        self.pkl_dir = os.path.join('/home/mmvg/Desktop/COVID', 'reports')
        self.img_dir = os.path.join(self.data_dir, 'COVID')

        self.num_medterm = opt.num_medterm

        with open(os.path.join(self.pkl_dir, 'align2.' + split + '.pkl'), 'rb') as f:
            self.findings = pkl.load(f)
            self.findings_labels = pkl.load(f)
            self.image = pkl.load(f)
            self.medterms = pkl.load(f)

        f.close()

        with open(os.path.join(self.pkl_dir, 'word2idw.pkl'), 'rb') as f:
            self.word2idw = pkl.load(f)
        f.close()

        with open(os.path.join(self.pkl_dir, 'idw2word.pkl'), 'rb') as f:
            self.idw2word = pkl.load(f)
        f.close()

        self.ids = list(self.image.keys())
        self.vocab_size = len(self.word2idw)

    def __getitem__(self, index):
        ix = self.ids[index]
        image_id = self.image[ix]
        image_name = os.path.join(self.img_dir, image_id)
        img = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        #print(img.size(), image_id)
        medterm_labels = np.zeros(self.num_medterm)
        medterms = self.medterms[ix]
        for medterm in medterms:
            # medterm_labels[medterm] = 1
            if medterm < self.num_medterm:
                medterm_labels[medterm] = 1

        findings = self.findings[ix]
        findings_labels = self.findings_labels[ix]
        findings = np.array(findings)
        findings_labels = np.array(findings_labels)

        findings = torch.from_numpy(findings).long()
        findings_labels = torch.from_numpy(findings_labels).long()


        return ix, image_id, img, findings, findings_labels, torch.FloatTensor(medterm_labels)

    def __len__(self):
        return len(self.ids)


