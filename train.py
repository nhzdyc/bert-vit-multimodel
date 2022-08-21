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
import pytorch_warmup as warmup

import numpy as np
import argparse
from tqdm import tqdm
#from datetime import datetime
import os
import requests
    
def train(args):
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
    dataset_train=get_dataset(args,train=True,val=False,transform=transform,tokenizer=tokenizer)
    collate_fn=collater(args)
    dataloader_train=DataLoaderX(dataset_train,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn,num_workers=args.num_workers)
    if args.model_type=='multimodel':
        model=bert_vit_feature_model(args).to(device)
    if args.model_type=='image':
        model=imageModel(args).to(device)
    if args.model_type=='text':
        model=textModel(args).to(device)
    if args.model_type=='multibase':
        if args.multi_base_model_type.split('+')[0]=='bert':
            model=multi_baseline_bert_model(args).to(device)
        else:
            model=multi_baseline_vit_model(args).to(device)
    num_steps=len(dataloader_train)*args.maxepoch
    criterion=nn.BCEWithLogitsLoss()
    #optimizer=optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    optimizer=optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.momentum)
    #optimizer=optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    lr_scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    #warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    best_accuracy=0
    n_no_improve=0
    start_epoch=0
    if args.model_type=='multimodel' and os.path.exists(args.checkpoint_dir + '/m-best_'+args.concat_type+'.pth.tar'):
        checkpoint = torch.load(args.checkpoint_dir+'/m-best_'+args.concat_type+'.pth.tar')
    if args.model_type=='image' and os.path.exists(args.checkpoint_dir + '/m-best_'+args.image_model_type+'.pth.tar'):
        checkpoint = torch.load(args.checkpoint_dir + '/m-best_'+args.image_model_type+'.pth.tar')
    if args.model_type=='text' and os.path.exists(args.checkpoint_dir + '/m-best_'+args.text_model_type+'.pth.tar'):
        checkpoint = torch.load(args.checkpoint_dir + '/m-best_'+args.text_model_type+'.pth.tar')
    if args.model_type=='multibase' and os.path.exists(args.checkpoint_dir + '/m-best_'+args.multi_base_model_type+'.pth.tar'):
        checkpoint = torch.load(args.checkpoint_dir + '/m-best_'+args.multi_base_model_type+'.pth.tar')
    if 'checkpoint' in locals().keys():
        start_epoch,n_no_improve,best_accuracy=load_model(checkpoint,model,optimizer,lr_scheduler)
        print('导入模型成功')
        
    for i in tqdm(range(start_epoch,args.maxepoch+1)):
        training_loss=0
        train_accuracy_all=0
        if args.model_type=='multimodel':
            for j,(input_ids,token_type_ids,attention_mask,image,label) in tqdm(enumerate(dataloader_train),total=len(dataloader_train)):
                input_ids,token_type_ids,attention_mask,image=input_ids.to(device),token_type_ids.to(device),attention_mask.to(device),image.to(device)
                label=label.unsqueeze(1).to(torch.float32).to(device)
                model.train()
                output=model(input_ids,token_type_ids,attention_mask,image)
                loss=criterion(output,label)
                optimizer.zero_grad()
                training_loss+=loss.item()
                accuracy_batch=get_accuracy_num(output,label,args)
                train_accuracy_all+=accuracy_batch
                torch.cuda.empty_cache()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
                optimizer.step()
                lr_scheduler.step()
                #warmup_scheduler.dampen()
                if j%1000==0 and j!=0:
                    print(f' batch:{j} loss:{training_loss/((j+1)*args.batch_size)} accuracy:{train_accuracy_all/((j+1)*args.batch_size)}')
                    #torch.cuda.empty_cache()
                                                                                
        if args.model_type=='image':
            for j,(image,label) in tqdm(enumerate(dataloader_train),total=len(dataloader_train)):
                image=image.to(device)
                label=label.unsqueeze(1).to(torch.float32).to(device)
                model.train()
                output=model(image)
                loss=criterion(output,label)
                optimizer.zero_grad()
                training_loss+=loss.item()
                accuracy_batch=get_accuracy_num(output,label,args)
                train_accuracy_all+=accuracy_batch
                torch.cuda.empty_cache()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),args.max_grad)
                optimizer.step()
                lr_scheduler.step()
                #warmup_scheduler.dampen()
                if j%1000==0 and j!=0:
                    print(f' batch:{j} loss:{training_loss/((j+1)*args.batch_size)} accuracy:{train_accuracy_all/((j+1)*args.batch_size)}')    
                    
        if args.model_type=='text':
            for j,(comment,label) in tqdm(enumerate(dataloader_train),total=len(dataloader_train)):
                comment=comment.to(device)
                label=label.unsqueeze(1).to(torch.float32).to(device)
                model.train()
                output=model(comment)
                loss=criterion(output,label)
                optimizer.zero_grad()
                training_loss+=loss.item()
                accuracy_batch=get_accuracy_num(output,label,args)
                train_accuracy_all+=accuracy_batch
                torch.cuda.empty_cache()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),args.max_grad)
                optimizer.step()
                lr_scheduler.step()
                #warmup_scheduler.dampen()
                if j%1000==0 and j!=0:
                    print(f' batch:{j} loss:{training_loss/((j+1)*args.batch_size)} accuracy:{train_accuracy_all/((j+1)*args.batch_size)}')  
                    
        if args.model_type=='multibase':
            if args.multi_base_model_type.split('+')[0]=='bert':
                for j,(input_ids,token_type_ids,attention_mask,image,label) in tqdm(enumerate(dataloader_train),total=len(dataloader_train)):
                    input_ids,token_type_ids,attention_mask,image=input_ids.to(device),token_type_ids.to(device),attention_mask.to(device),image.to(device)
                    label=label.unsqueeze(1).to(torch.float32).to(device)
                    model.train()
                    output=model(input_ids,token_type_ids,attention_mask,image)
                    loss=criterion(output,label)
                    optimizer.zero_grad()
                    training_loss+=loss.item()
                    accuracy_batch=get_accuracy_num(output,label,args)
                    train_accuracy_all+=accuracy_batch
                    torch.cuda.empty_cache()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
                    optimizer.step()
                    lr_scheduler.step()
                    #warmup_scheduler.dampen()
                    if j%1000==0 and j!=0:
                        print(f' batch:{j} loss:{training_loss/((j+1)*args.batch_size)} accuracy:{train_accuracy_all/((j+1)*args.batch_size)}')
            else:
                for j,(comment,image,label) in tqdm(enumerate(dataloader_train),total=len(dataloader_train)):
                    comment,image=comment.to(device),image.to(device)
                    label=label.unsqueeze(1).to(torch.float32).to(device)
                    model.train()
                    output=model(comment,image)
                    loss=criterion(output,label)
                    optimizer.zero_grad()
                    training_loss+=loss.item()
                    accuracy_batch=get_accuracy_num(output,label,args)
                    train_accuracy_all+=accuracy_batch
                    torch.cuda.empty_cache()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad)
                    optimizer.step()
                    lr_scheduler.step()
                    #warmup_scheduler.dampen()
                    if j%1000==0 and j!=0:
                        print(f' batch:{j} loss:{training_loss/((j+1)*args.batch_size)} accuracy:{train_accuracy_all/((j+1)*args.batch_size)}')
                        
        train_accuracy=train_accuracy_all/len(dataset_train)
        dataset_test=get_dataset(args,train=False,val=False,transform=transform,tokenizer=tokenizer)
        test_accuracy=get_val_accuracy(args,dataset_test,model,collate_fn,device)
        
        if test_accuracy>best_accuracy:
            if args.model_type=='multimodel':
                #torch.save({'epoch': i + 1, 'state_dict': model.state_dict(), 'best_accuracy':train_accuracy,'optimizer': optimizer.state_dict(),"lr_scheduler": lr_scheduler.state_dict(),'n_no_improve':n_no_improve,'test_accuracy':test_accuracy,'warmup_scheduler':warmup_scheduler.state_dict()},args.checkpoint_dir + '/m-best_'+args.concat_type+'.pth.tar')
                torch.save({'epoch': i + 1, 'state_dict': model.state_dict(), 'best_accuracy':train_accuracy,'optimizer': optimizer.state_dict(),"lr_scheduler": lr_scheduler.state_dict(),'n_no_improve':n_no_improve,'test_accuracy':test_accuracy},
                           args.checkpoint_dir + '/m-best_'+args.concat_type+'.pth.tar')
            if args.model_type=='image':
                torch.save({'epoch': i + 1, 'state_dict': model.state_dict(), 'best_accuracy':train_accuracy,'optimizer': optimizer.state_dict(),"lr_scheduler": lr_scheduler.state_dict(),'n_no_improve':n_no_improve,'test_accuracy':test_accuracy},
                        args.checkpoint_dir + '/m-best_'+args.image_model_type+'.pth.tar')
            if args.model_type=='text':
                torch.save({'epoch': i + 1, 'state_dict': model.state_dict(), 'best_accuracy':train_accuracy,'optimizer': optimizer.state_dict(),"lr_scheduler": lr_scheduler.state_dict(),'n_no_improve':n_no_improve,'test_accuracy':test_accuracy},
                        args.checkpoint_dir + '/m-best_'+args.text_model_type+'.pth.tar')
            if args.model_type=='multibase':
                torch.save({'epoch': i + 1, 'state_dict': model.state_dict(), 'best_accuracy':train_accuracy,'optimizer': optimizer.state_dict(),"lr_scheduler": lr_scheduler.state_dict(),'n_no_improve':n_no_improve,'test_accuracy':test_accuracy},
                        args.checkpoint_dir + '/m-best_'+args.multi_base_model_type+'.pth.tar')
                
            best_accuracy=test_accuracy
            n_no_improve=0
        else:
            n_no_improve+=1
        print(f'epoch:{i} train_loss"{training_loss/len(dataset_train)} train_accuracy:{train_accuracy} test_accuracy{test_accuracy}')
        if n_no_improve>=args.patience:    
            break

def train_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)
    
if __name__=='__main__':
    train_main()