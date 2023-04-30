import pandas as pd
import re
from collections import Counter
import jieba
import jieba.posseg
import json
# jieba.load_userdict("/home/mmvg/Desktop/FAIRformer/FAIR/modules/eye_words.txt")


class Tokenizer(object):
    def __init__(self, args):
        if args.dataset_name == 'iu_xray':
            self.ann_path = args.ann_path.split('&')[0]
        elif args.dataset_name == 'mimic_cxr':
            self.ann_path = args.ann_path.split('&')[1]
        # self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        self.clean_report = self.clean_report_fair
        with open(self.ann_path, encoding='utf-8') as f:
            self.ann = json.load(f)
            f.close()

        self.ann_path_all = args.ann_path.split('&')
        self.ann_all = []
        for ann_path in self.ann_path_all:
            with open(ann_path, encoding='utf-8') as f:
                self.ann_all.append(json.load(f))
                f.close()

        # self.ann = pd.read_csv(self.ann_path)
        self.token2idx, self.idx2token = self.create_vocabulary()
        self.special_id = [164, 165, 166, 168, 169]

    def create_vocabulary(self):
        total_tokens = []

        # for i in range(len(self.ann['train'])):
        #     report = self.ann['train'][i]['report']
        #     tokens = self.clean_report_fair(report).split()
        #     tokens.append('picture')
        #     for token in tokens:
        #         total_tokens.append(token)

        for i in range(len(self.ann_all)):
            cur_ann = self.ann_all[i]
            for item in cur_ann['train']:
                report = item['report']
                tokens = self.clean_report_fair(report).split()
                for token in tokens:
                    total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>'] + ['[DEC]'] + ['<pad>'] + ['<cls>'] + ['<sep>'] + ['[ENC]'] + ['picture']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_fair(self, report):
        report_copy = report
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1.', '') \
            .replace('2.', '').replace('3.', '').replace('4.', '').replace('5.', '') \
            .replace('1、', '').replace('2、', '').replace('3、', '').replace('4、', '') \
            .strip().lower()
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        # jieba cut the tokens
        tokens = [sent_cleaner(sent) for sent in jieba.lcut(report_cleaner(report)) if sent_cleaner(sent) != []]
        report = ' '.join(tokens)
        if '[DEC]' in report_copy:
            report = report.replace('dec', '[DEC]')
        if '[ENC]' in report_copy:
            report = report.replace('enc', '[ENC]')
        if '<cls>' in report_copy:
            report = report.replace('cls', '<cls>')
        if '<sep>' in report_copy:
            report = report.replace('sep', '<sep>')
        if '<pad>' in report_copy:
            report = report.replace('pad', '<pad>')
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        # ids = [0] + ids + [0]
        ids = [164] + ids + [166]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx in self.special_id:
                continue
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
