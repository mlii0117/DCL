# DCL
Official code for "Dynamic Graph Enhanced Contrastive Learning for Chest X-ray Report Generation" (CVPR 2023)

In this paper, we present a practical approach to leverage dynamic graph to enhance contrastive learning for radiology report generation. In which the dynamic graph is constructed in a bottom-up manner to integrate retrieved specific knowledge with general knowledge. Then contrastive learning is employed to improve visual and textual representations, which also promises the accuracy of our dynamic graph. Experiments on two popular benchmarks verify the effectiveness of our method in generating accurate and meaningful reports

## Requirements

All the requirements are listed in the requirements.txt file.
Please use this command to create a new environment and activate it.
```
conda create --name DCL --file requirements.txt
conda activate DCL
```  

## Data

Please download the IU and MIMIC datasets, and place them in the `./dataset/` folder.

- IU dataset from [here](https://iuhealth.org/find-medical-services/x-rays)
- MIMIC dataset from [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
- We prvoide the annotation files and knowledge files for both datasets. Please download from [here](https://drive.google.com/drive/folders/1BX_Fbs6FVeCtr6xOLdi-YHAYHAy7lCDv?usp=sharing), and place them in the `./annotations/` folder


## Training and Testing

The source code for training and testing is `main.py`.
To run this code, please use the following command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.run --nproc_per_node=2 --master_port=21073 main.py --config ./configs/BLIP.yaml --output_dir output/Generation --dataset_name iu_xray --distributed True --batch_size 8 --epochs 50 --save_dir results/test --bert sci
```


## Citation
If you find the code useful, please cite our paper:
~~~
@article{li2023dynamic,
  title={Dynamic Graph Enhanced Contrastive Learning for Chest X-ray Report Generation},
  author={Li, Mingjie and Lin, Bingqian and Chen, Zicong and Lin, Haokun and Liang, Xiaodan and Chang, Xiaojun},
  journal={arXiv preprint arXiv:2303.10323},
  year={2023}
}
~~~


## Contact

If you are interested in work or have any questions, please connect us: lmj695@gmail.com

## Acknowledges

We thank [BLIP](https://github.com/salesforce/BLIP) and [R2Gen](https://github.com/cuhksz-nlp/R2Gen) for their open source works.

