import json
import os
import torch

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .utils import pre_caption
import os

label_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
              'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
              'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

node = [
    'normal','other finding','heart','cardiomegaly','spine','scoliosis','pleural','effusion','thickening','pneumothorax',
    'bone', 'bone fractures','lung','emphysema','pneumonia','edema','atelectasis','clcatrix','opacity','lesion',
    'mediastinum','hernia','calcinosis','foreign object','airspace','airspace disease','hypoinflation'
]
nodes = '-'.join(node)

node_inds = [0,1,2,3,3,4,4,5,5,5,5,6,6,7,7,7,7,7,7,7,7,8,8,8,9,10,10,10]
node_labels = [0,2,3,1,4,1,5,1,6,6,6,1,7,1,8,8,8,8,8,8,8,1,9,9,10,1,11,11]
node_inds = [each+1 for each in node_inds]
node_labels = [each+1 for each in node_labels]

node_relations = list(range(len(node_inds)))
node_relations = [each+1 for each in node_relations]


skg = {'nodes':nodes, 'node_inds':node_inds, 'node_labels':node_labels, 'node_relations': node_relations}

class generation_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=90, prompt='', dataset='', args=None):
        
        self.annotation = json.load(open(os.path.join(ann_root),'r'))
        # self.ann = self.annotation['train']
        self.ann = self.annotation
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        self.dataset = dataset
        self.args = args
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = ann['image_path']
        if self.dataset == 'iu_xray':
            image_1 = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_root, image_path[1])).convert('RGB')
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image = torch.stack((image_1, image_2), 0)

        elif self.dataset == 'mimic_cxr':
            image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image = self.transform(image)

        caption = self.prompt + pre_caption(ann['report'], self.max_words)

        knowledge_skg = skg
        knowledge_tc = ''
        triplet_len = len(ann['triplet'])
        if triplet_len > 30:
            for i in range(30):
                knowledge_tc += ann['triplet'][i]
                if i < 29:
                    knowledge_tc += ' '
        else:
            tri_idx = 0
            for triplet in ann['triplet']:
                knowledge_tc += triplet
                tri_idx += 1
                if tri_idx < triplet_len:
                    knowledge_tc += ' '
        knowledge_tc = pre_caption(knowledge_tc, self.max_words)  #triplet can see each other

        return image, caption, knowledge_skg, knowledge_tc
    
    
class generation_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, dataset, args=None):
        self.annotation = json.load(open(os.path.join(ann_root), 'r'))
        self.ann = self.annotation[split]
        self.transform = transform
        self.image_root = image_root
        self.dataset = dataset
        self.args = args

        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        image_path = ann['image_path']
        if self.dataset == 'iu_xray':
            image_1 = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_root, image_path[1])).convert('RGB')
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image = torch.stack((image_1, image_2), 0)

        elif self.dataset == 'mimic_cxr':
            image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
            image = self.transform(image)

        caption = pre_caption(ann['report'], 90)
        knowledge_skg = skg

        return image, caption, knowledge_skg, image_path
