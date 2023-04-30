import warnings
warnings.filterwarnings("ignore")

from models.vit_blip import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer, AutoTokenizer
import transformers
transformers.logging.set_verbosity_error()

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
from functools import partial
from medical_knowledge.knowledge import create_knowledge
from medical_knowledge.SKG_knowledge import *
from models.tagencoder import TagEncoder, update_skg
        
        
class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 tokenizer = None,
                 embed_dim = 256,     
                 queue_size = 65536,
                 momentum = 0.995,
                 args = None
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        # self.tokenizer = init_tokenizer()
        self.tokenizer = tokenizer
        self.args = args
        self.prompt = prompt
        
        if args.bert == 'base':
            med_config = 'configs/med_config_blip.json'
        elif args.bert == 'sci':
            med_config = 'configs/med_config_sci.json'
        elif args.bert == 'cli':
            med_config = 'configs/med_config_cli.json'
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        # med_config.encoder_width = vision_width
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        text_width = self.text_encoder.config.hidden_size

        self.text_decoder = BertLMHeadModel(config=med_config)
        
        self.vision_proj = nn.Linear(vision_width, 256)
        self.text_proj = nn.Linear(768, 256)

            
        self.itm_head = nn.Linear(text_width, 2) 

        # create momentum encoders  
        self.visual_encoder_m, vision_width = create_vit(vit,image_size)              
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)      
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        c = copy.deepcopy
        attn = MultiHeadedAttention(6, 768)
        ff = PositionwiseFeedForward(768, 1024, 0.1)
        self.cross_attn = Decoder(DecoderLayer(768, c(attn), c(ff), 0.1), 2)

        self.tag_encoder = TagEncoder(0.1, self.args)

        self.cross_attn_m = Decoder(DecoderLayer(768, c(attn), c(ff), 0.1), 2)
        
        if args.dataset_name == 'iu_xray':
            self.iu_proj = nn.Linear(768*2, 768)
            self.iu_proj_m = nn.Linear(768*2, 768)
            queue_size = 1380
            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            [self.cross_attn, self.cross_attn_m],
                            [self.iu_proj,self.iu_proj_m]]
        else:
            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            [self.cross_attn, self.cross_attn_m]
                            ]
        

        self.copy_params()
        
        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))   
        
        # create the decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = vision_width
        if args.bert == 'base':
            self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=decoder_config)
        elif args.bert == 'sci':
            self.text_decoder = BertLMHeadModel.from_pretrained('allenai/scibert_scivocab_uncased',config=decoder_config)
        elif args.bert == 'cli':
            self.text_decoder = BertLMHeadModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT',config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        tie_encoder_decoder_weights(self.text_encoder,self.text_decoder.bert,'','/attention')

        self.create_knowledge = create_knowledge(embed_dim=embed_dim, queue_size=queue_size,
                                                 text_encoder=self.text_encoder, text_proj=self.text_proj,
                                                 tokenizer=self.tokenizer, args=args)

        
        
    def forward(self, image, caption, knowledge_skg, knowledge_tc,alpha, epoch):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        if self.args.dataset_name == 'iu_xray':
            image_embeds0 = self.visual_encoder(image[:, 0])
            image_embeds1 = self.visual_encoder(image[:, 1])
            image_embeds = torch.cat((image_embeds0, image_embeds1), dim=2)
            image_embeds = self.iu_proj(image_embeds)
        else:
            image_embeds = self.visual_encoder(image)

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=90, return_tensors="pt").to(image.device)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
        self.create_knowledge(image.device, text_feat, knowledge_tc)
        
        ###============== Obtain Knowledge ===================###
        if epoch > 0:
            # ---------------------Get Knowledge Words---------------------
            _, knowledge_words = self.create_knowledge.get_image_knowledge(image.device, image_feat, image_embeds, k=3)

            #get each knowledge
            bs = image_feat.shape[0]
            knowledge_tc_re = []
            for i in range(bs):
                knowledge_all = ''
                for j in range(3):
                    knowledge_item = knowledge_words[i+bs*j]
                    knowledge_item = knowledge_item.split('-')
                    knowledge_item.pop()
                    knowledge_item = '-'.join(knowledge_item)
                    if j == 0:
                        knowledge_all += knowledge_item
                    else:
                        knowledge_all = knowledge_all + '-' + knowledge_item
                    # knowledge_all += knowledge_item.replace(knowledge_item.split('-')[-1], ' ')
                knowledge_list = knowledge_all.split('-')
                # knowledge_list.pop()
                knowledge_tc_re.append(knowledge_list)
                
            knowledge_skg['node_inds'] = [each.tolist() for each in knowledge_skg['node_inds']]
            knowledge_skg['node_labels'] = [each.tolist() for each in knowledge_skg['node_labels']]
            knowledge_skg['node_relations'] = [each.tolist() for each in knowledge_skg['node_relations']]
            for i in range(58):
                knowledge_skg['node_inds'].append(torch.zeros(bs).tolist())
                knowledge_skg['node_labels'].append(torch.zeros(bs).tolist())
                knowledge_skg['node_relations'].append(torch.zeros(bs).tolist())
            knowledge_skg['node_inds'] = torch.tensor(knowledge_skg['node_inds'])
            knowledge_skg['node_labels'] = torch.tensor(knowledge_skg['node_labels'])
            knowledge_skg['node_relations'] = torch.tensor(knowledge_skg['node_relations'])

            for idx in range(len(knowledge_tc_re)):
                knowledge = knowledge_tc_re[idx]
                for i in range(len(knowledge)):
                    knowledge[i] = knowledge[i].strip().replace('located _ at', 'located_at').replace('suggestive _ of', 'suggestive_of')
                    triplet_item = knowledge[i].split(' ')
                    if len(triplet_item) != 3:
                        triplet = None
                        knowledge_skg = update_skg(knowledge_skg, triplet, idx, i)
                    else:
                        triplet = [triplet_item[0], triplet_item[1], triplet_item[2]]
                        knowledge_skg = update_skg(knowledge_skg, triplet, idx, i)
            for i in range(bs):
                knowledge_skg['nodes'][i] = knowledge_skg['nodes'][i].replace('-', ' ')
        else:
            bs = image_feat.shape[0]
            knowledge_skg['node_inds'] = [each.tolist() for each in knowledge_skg['node_inds']]
            knowledge_skg['node_labels'] = [each.tolist() for each in knowledge_skg['node_labels']]
            knowledge_skg['node_relations'] = [each.tolist() for each in knowledge_skg['node_relations']]
            for i in range(58):
                knowledge_skg['node_inds'].append(torch.zeros(bs).tolist())
                knowledge_skg['node_labels'].append(torch.zeros(bs).tolist())
                knowledge_skg['node_relations'].append(torch.zeros(bs).tolist())
            knowledge_skg['node_inds'] = torch.tensor(knowledge_skg['node_inds'])
            knowledge_skg['node_labels'] = torch.tensor(knowledge_skg['node_labels'])
            knowledge_skg['node_relations'] = torch.tensor(knowledge_skg['node_relations'])
            for i in range(bs):
                knowledge_skg['nodes'][i] = knowledge_skg['nodes'][i].replace('-', ' ')
                
        tag_output = self.tag_encoder(knowledge_skg, image.device)

        image_embeds, vis_attn2 = self.cross_attn(image_embeds, tag_output)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1) # bs x 768     
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        ###============== Image-report Contrastive Learning ===================###

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            
            if self.args.dataset_name == 'iu_xray':
                image_embeds0_m = self.visual_encoder_m(image[:, 0])
                image_embeds1_m = self.visual_encoder_m(image[:, 1])
                image_embeds_m = torch.cat((image_embeds0_m, image_embeds1_m), dim=2)
                image_embeds_m = self.iu_proj_m(image_embeds_m)
            else:
                image_embeds_m = self.visual_encoder_m(image)
            
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]),dim=-1)

            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)

            ###============== Obtain Knowledge ===================###

            image_embeds_m, _ = self.cross_attn_m(image_embeds_m, tag_output)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)

            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)

            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean()

        loss_irc = (loss_i2t+loss_t2i)/2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###============== Image-report Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:,0] = self.tokenizer.enc_token_id
        
        # forward the positve image-text pair
        bs = image.size(0)
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )

        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)+1e-4
            weights_t2i.fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)+1e-4
            weights_i2t.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg,dim=0)
        text_atts_neg = torch.stack(text_atts_neg,dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg],dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask = text_atts_all,
                                       encoder_hidden_states = image_embeds_all,
                                       encoder_attention_mask = image_atts_all,
                                       return_dict = True,
                                      )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_irm = F.cross_entropy(vl_output, itm_labels)
        
        ##================= LM ========================##
        text.input_ids[:,0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)

        decoder_targets[:,:self.prompt_length] = -100

        decoder_output = self.text_decoder(text.input_ids,
                                           attention_mask = text.attention_mask,
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,
                                           labels = decoder_targets,
                                           return_dict = True,
                                          )


        loss_lm = decoder_output.loss

        return loss_irc, loss_irm, loss_lm
        
    def generate(self, image, knowledge_skg, sample=False, num_beams=3, max_length=90, min_length=10, top_p=0.9, repetition_penalty=1.0):
        if self.args.dataset_name == 'iu_xray':
            image_embeds0 = self.visual_encoder(image[:, 0])
            image_embeds1 = self.visual_encoder(image[:, 1])
            image_embeds = torch.cat((image_embeds0, image_embeds1), dim=2)
            image_embeds = self.iu_proj(image_embeds)
        else:
            image_embeds = self.visual_encoder(image)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        _, knowledge_words = self.create_knowledge.get_image_knowledge(image.device, image_feat, image_embeds, k=3)

        #get each knowledge
        bs = image_feat.shape[0]
        knowledge_tc_re = []
        for i in range(bs):
            knowledge_all = ''
            for j in range(3):
                knowledge_item = knowledge_words[i+bs*j]
                knowledge_item = knowledge_item.split('-')
                knowledge_item.pop()
                knowledge_item = '-'.join(knowledge_item)
                if j == 0:
                    knowledge_all += knowledge_item
                else:
                    knowledge_all = knowledge_all + '-' + knowledge_item
                # knowledge_all += knowledge_item.replace(knowledge_item.split('-')[-1], ' ')
            knowledge_list = knowledge_all.split('-')
            # knowledge_list.pop()
            knowledge_tc_re.append(knowledge_list)

        knowledge_skg['node_inds'] = [each.tolist() for each in knowledge_skg['node_inds']]
        knowledge_skg['node_labels'] = [each.tolist() for each in knowledge_skg['node_labels']]
        knowledge_skg['node_relations'] = [each.tolist() for each in knowledge_skg['node_relations']]
        for i in range(58):
            knowledge_skg['node_inds'].append(torch.zeros(bs).tolist())
            knowledge_skg['node_labels'].append(torch.zeros(bs).tolist())
            knowledge_skg['node_relations'].append(torch.zeros(bs).tolist())
        knowledge_skg['node_inds'] = torch.tensor(knowledge_skg['node_inds'])
        knowledge_skg['node_labels'] = torch.tensor(knowledge_skg['node_labels'])
        knowledge_skg['node_relations'] = torch.tensor(knowledge_skg['node_relations'])

        for idx in range(len(knowledge_tc_re)):

            knowledge = knowledge_tc_re[idx]
            for i in range(len(knowledge)):
                knowledge[i] = knowledge[i].strip().replace('located _ at', 'located_at').replace('suggestive _ of', 'suggestive_of')
                triplet_item = knowledge[i].split(' ')
                if len(triplet_item) != 3:
                    triplet = None
                    knowledge_skg = update_skg(knowledge_skg, triplet, idx, i)
                else:
                    triplet = [triplet_item[0], triplet_item[1], triplet_item[2]]
                    knowledge_skg = update_skg(knowledge_skg, triplet, idx, i)

        for i in range(bs):
            knowledge_skg['nodes'][i] = knowledge_skg['nodes'][i].replace('-', ' ')

        tag_output = self.tag_encoder(knowledge_skg, image.device)

        image_embeds, vis_attn2 = self.cross_attn(image_embeds, tag_output)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1) # bs x 768
        
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
        
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}

        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]
        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,
                                                  repetition_penalty=1.1,
                                                  **model_kwargs)
        else:
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])

        return captions, knowledge_words
    
    
    
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

                        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T

        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
    

def blip_decoder(pretrained='',**kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
    return model    
    
def init_tokenizer(args):
    if args.bert == 'base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.bert == 'sci':
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    elif args.bert == 'cli':
        tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    # create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )
        # visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
        #                                    num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                # print(state_dict[key])
                print(state_dict[key].shape)
                print(model.state_dict()[key].shape)
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg
    

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output     


from typing import List
def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias                
            print(module_name+' is tied')    
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)  
