import torch
import argparse
import ruamel_yaml as yaml
import numpy as np
from generation_api.metrics import compute_scores
from generation_api.optimizers import build_optimizer_blip, build_lr_scheduler
from generation_api.trainer_blip import Trainer
from generation_api.loss import compute_loss
from transformers import BertTokenizer
from generation_api.tokenizers_blip import Tokenizer
from models.blip import blip_decoder
from blip_original import create_loader, create_dataset
import os
from transformers import AutoTokenizer, AutoModel



def main(args, config):

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer

    if args.bert == 'base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.bert == 'sci':
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    elif args.bert == 'cli':
        tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    # tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    train_dataset, val_dataset, test_dataset = create_dataset('generation_%s'%args.dataset_name, args, config)
    samplers = [None, None, None]
    train_dataloader, val_dataloader, test_dataloader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                            batch_size=[args.batch_size] * 3,
                                                            num_workers=[4, 4, 4],
                                                            is_trains=[True, False, False],
                                                            collate_fns=[None, None, None])

    model = blip_decoder(pretrained=args.pretrained, image_size=config['image_size'], vit=config['vit'],
                         vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                         prompt=config['prompt'], tokenizer=tokenizer, args=args)


    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer_blip(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader, tokenizer)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Generation.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--pretrained', default='')
    parser.add_argument('--output_dir', default='output/generation')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    parser.add_argument('--image_dir', type=str,
                        default='./dataset/iu_xray/images&./dataset/MIMIC-CXR/mimic_cxr/images',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str,
                        default='./annotations/iu-annotation.json&./annotations/mimic_annotation.json',
                        help='the path to the directory containing the data.')
    parser.add_argument('--knowledge_path', type=str,
                        default='./annotations/iu_train_kg_AO.json&./annotations/mimic_train_kg_AO.json',
                        help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=90, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=2, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,
                        help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')


    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=30, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/fair', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/generation/',
                        help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=1e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    # parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--test_visual', type=bool, default=False, help='whether to test the visual encoder.')
    parser.add_argument('--test_text', type=bool, default=False, help='whether to test the text encoder.')
    parser.add_argument('--test_text_cross', type=bool, default=False, help='whether to test the text and cross encoder.')
    parser.add_argument('--test_visual_text', type=bool, default=False, help='whether to test the text and visual encoder.')
    parser.add_argument('--test_best', type=bool, default=False, help='whether to test the best model.')

    parser.add_argument('--add_memory', type=bool, default=False, help='whether to test the best model.')
    parser.add_argument('--tokenizer', type=str, default='blip', choices=['r2gen', 'blip'],
                        help='the dataset to be used.')
    parser.add_argument('--bert', type=str, default='base', choices=['base', 'sci', 'cli'],
                        help='the dataset to be used.')
    parser.add_argument('--concat', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    main(args, config)
