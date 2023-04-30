from torch import nn
import torch
import torch.nn.functional as F
import json
import math
import copy
from models.med import BertConfig, BertModel, BertLMHeadModel
from medical_knowledge.SKG_knowledge import *

class create_knowledge(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 queue_size=57600,
                 text_encoder=None,
                 text_proj=None,
                 tokenizer=None,
                 args=None
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.args = args
        # create the queue
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        # self.register_buffer("knowledge_queue", torch.randn(embed_dim, queue_size))
        if args.bert == 'base':
            vocab_size = 30522
        elif args.bert == 'sci':
            vocab_size = 31090
        elif args.bert == 'cli':
            vocab_size = 28996
        self.register_buffer("knowledge_input_ids_queue", torch.randint(0,vocab_size,(queue_size, 90)))
        # self.register_buffer("knowledge_attention_mask_queue", torch.randint(0, 2, (queue_size, 90)))
        self.register_buffer("knowledge_attention_mask_queue", torch.ones((queue_size, 90)))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        # self.knowledge_input_ids_queue = nn.functional.normalize(self.knowledge_input_ids_queue, dim=0)
        # self.knowledge_attention_mask_queue = nn.functional.normalize(self.knowledge_attention_mask_queue, dim=0)
        self.knowledge_input_ids_queue = self.knowledge_input_ids_queue
        self.knowledge_attention_mask_queue = self.knowledge_attention_mask_queue

        self.queue_size = queue_size
        # self.text_encoder = text_encoder
        self.text_proj = text_proj
        self.tokenizer = tokenizer
        self.knowledge_encoder = BertModel.from_pretrained('allenai/scibert_scivocab_uncased',
                                                     config=BertConfig.from_json_file('configs/tag_config_sci.json'),
                                                     add_pooling_layer=False)
        self.knowledge_encoder.resize_token_embeddings(len(self.tokenizer))

        # self.enhanced_visual_proj=nn.Linear(1536, 768)

        c = copy.deepcopy
        attn = MultiHeadedAttention(6, 768)
        ff = PositionwiseFeedForward(768, 1024, 0.1)
        self.cross_attn = Decoder(DecoderLayer(768, c(attn), c(ff), 0.1), 2)


    def forward(self, device, text_feat, knowledge):

        knowledge_token = self.tokenizer(knowledge, padding='max_length', truncation=True, max_length=90,
                              return_tensors="pt").to(device)

        knowledge_token.attention_mask[knowledge_token.input_ids == 579] = 0

        self._dequeue_and_enqueue(text_feat, knowledge_token)


    def get_image_knowledge(self, device, image_feat, image_embeds, k):
        bs = image_feat.shape[0]
        queue_size = self.queue_size

        sim_matrix = torch.matmul(image_feat, self.text_queue.clone().detach())

        values, indices = sim_matrix.topk(k, dim=-1)

        weight_value = F.softmax(values, dim=-1)

        topk_knowledge_input_ids = self.knowledge_input_ids_queue[indices]
        topk_knowledge_attention_mask = self.knowledge_attention_mask_queue[indices]

        topk_knowledge = torch.randn(bs, k, 90, 768).to(device)
        knowledge_words = []
        for i in range(0,k):
            input_ids = topk_knowledge_input_ids[:,i,:]
            attention_mask = topk_knowledge_attention_mask[:,i,:]
            knowledge_output = self.knowledge_encoder(input_ids, attention_mask = attention_mask,
                                        return_dict = True, mode = 'text')
            knowledge_output = knowledge_output.last_hidden_state * weight_value.unsqueeze(-1).unsqueeze(-1)[:,i,:]
            topk_knowledge[:,i,:] = knowledge_output
            # knowledge_output = knowledge_output.last_hidden_state * weight_value.unsqueeze(-1).unsqueeze(-1)[:, i, :, :]
            # topk_knowledge[:, i, :, :] = knowledge_output
            for output in input_ids:
                topk_knowledge_word = self.tokenizer.decode(output, skip_special_tokens=True)
                knowledge_words.append(topk_knowledge_word)

        # print('#########topk##########')
        # print(topk_knowledge.size())    #[bs * k * max_len * embed_dim]

        final_knowledge = topk_knowledge.sum(1)   #[bs * max_len * embed_dim]

        enhanced_image_embeds, _ = self.cross_attn(image_embeds, final_knowledge)

        # enhanced_image_embeds = torch.cat((image_embeds, enhanced_image_embeds),dim=2)
        # enhanced_image_embeds = self.enhanced_visual_proj(enhanced_image_embeds)
        return enhanced_image_embeds, knowledge_words

    @torch.no_grad()
    def _dequeue_and_enqueue(self, text_feat, knowledge):
        # gather keys before updating queue
        if self.args.task == 'pretrain' or self.args.task == 'retrieval':
            text_feats = concat_all_gather(text_feat)
            knowledge_input_ids = concat_all_gather(knowledge.input_ids)
            knowledge_attention_mask = concat_all_gather(knowledge.attention_mask)
        else:
            text_feats = text_feat
            knowledge_input_ids = knowledge.input_ids
            knowledge_attention_mask = knowledge.attention_mask

        batch_size = text_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.knowledge_input_ids_queue[ptr:ptr + batch_size] = knowledge_input_ids
        self.knowledge_attention_mask_queue[ptr:ptr + batch_size] = knowledge_attention_mask

        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

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

