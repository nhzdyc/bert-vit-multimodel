import torch
import torch.nn as nn
import torch.optim as optim

from model import *
from dataset import *
from utilis import *

from torchvision import transforms
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

import numpy as np
import argparse
from tqdm import tqdm
        
def test_main():
    parser = argparse.ArgumentParser(description="Val Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)
    tokenizer=BertTokenizer.from_pretrained(args.pretrained_path)
    transform=transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
        )])
    dataset=get_dataset(args,train=False,val=True,transform=transform,tokenizer=tokenizer)
    collate_fn=collater(args)
    if args.model_type=='multimodel':
        model_pretrained=torch.load(args.checkpoint_dir + '/m-best_'+args.concat_type+'.pth.tar')
        model=bert_vit_feature_model(args).to(device)
    elif args.model_type=='image':
        model_pretrained=torch.load(args.checkpoint_dir + '/m-best_'+args.image_model_type+'.pth.tar')
        model=imageModel(args).to(device)
    elif args.model_type=='text':
        model_pretrained=torch.load(args.checkpoint_dir + '/m-best_'+args.text_model_type+'.pth.tar')
        model=textModel(args).to(device)
    elif args.model_type=='multibase':
        model_pretrained=torch.load(args.checkpoint_dir + '/m-best_'+args.multi_base_model_type+'.pth.tar')
        if args.multi_base_model_type.split('+')[0]=='bert':
            model=multi_baseline_bert_model(args).to(device)
        else:
            model=multi_baseline_vit_model(args).to(device)
          
    model.load_state_dict(model_pretrained['state_dict'])
    
    return get_val_accuracy(args,dataset=dataset,model=model,device=device,collate_fn=collate_fn)

if __name__=='__main__':
    val_accuracy=test_main()
    print(val_accuracy)
