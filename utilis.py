import torch
import torch.nn as nn
from dataset import *
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import numpy as np
from tqdm import tqdm

def get_args(parser):
    parser.add_argument('--pretrained_path', type=str, default='./pretrained')
    parser.add_argument('--image_feature',type=int,default=1000)
    parser.add_argument('--text_feature',type=int,default=768)
    parser.add_argument('--image_size',type=int,default=224)
    parser.add_argument('--num_classes',type=int,default=1)
    parser.add_argument('--dropout_rate',type=float,default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument('--comment_dir', type=str, default='./data/comment')
    parser.add_argument('--image_dir', type=str, default='./data/image')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--maxepoch", type=int, default=50)
    parser.add_argument('--learning_rate',type=float,default=1e-3)
    parser.add_argument('--weight_decay',type=float,default=1e-5)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument('--concat_type', type=str, default='self-attention')
    parser.add_argument('--model_type', type=str, default='multimodel')
    parser.add_argument('--image_model_type', type=str, default='resnet')
    parser.add_argument('--max_grad', type=float, default=2.0)
    parser.add_argument('--text_model_type', type=str, default='bilstm')
    parser.add_argument('--hidden_size',type=int,default=128)
    parser.add_argument('--sentence_length',type=int,default=1024)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--multi_base_model_type', type=str, default='bert+resnet')
    
class DataLoaderX(DataLoader):
    
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def get_accuracy_num(output,label,args):
    output_predict_label=[]
    for i in output:
        if i>=0.5:
            output_predict_label.append(1)
        else:
            output_predict_label.append(0)
    return np.sum(np.array(output_predict_label)==label.squeeze(1).cpu().detach().numpy())

def get_dataset(args,train,val,transform,tokenizer):
    if args.model_type=='multimodel':
        dataset=multimodelDataset(args=args,transform=transform,tokenizer=tokenizer,train=train,val=val)
    if args.model_type=='image':
        dataset=imageDataset(args=args,transform=transform,train=train,val=val)
    if args.model_type=='text':
        dataset=textDataset(args=args,train=train,val=val)
    if args.model_type=='multibase':
        if args.multi_base_model_type.split('+')[0]=='bert':
            dataset=multimodelDataset(args=args,transform=transform,tokenizer=tokenizer,train=train,val=val)
        else:
            dataset=multi_base_textDataset(args=args,transform=transform,train=train,val=val)
    return dataset

def get_val_accuracy(args,dataset,model,collate_fn,device):
    with torch.no_grad():
        print('测试阶段')
        test_accuracy_all=0
        dataloader_test=DataLoaderX(dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn,num_workers=args.num_workers)
        if args.model_type=='multimodel':
            for k,(input_ids,token_type_ids,attention_mask,image,label) in tqdm(enumerate(dataloader_test),total=len(dataloader_test)):
                input_ids,token_type_ids,attention_mask,image=input_ids.to(device),token_type_ids.to(device),attention_mask.to(device),image.to(device)
                label=label.unsqueeze(1).to(torch.float32).to(device)
                model.eval()
                output=model(input_ids,token_type_ids,attention_mask,image)
                accuracy_batch=get_accuracy_num(output,label,args)
                test_accuracy_all+=accuracy_batch

        if args.model_type=='image':
            for k,(image,label) in tqdm(enumerate(dataloader_test),total=len(dataloader_test)):
                image=image.to(device)
                label=label.unsqueeze(1).to(torch.float32).to(device)
                model.eval()
                output=model(image)
                accuracy_batch=get_accuracy_num(output,label,args)
                test_accuracy_all+=accuracy_batch
                
        if args.model_type=='text':
            for k,(comment,label) in tqdm(enumerate(dataloader_test),total=len(dataloader_test)):
                comment=comment.to(device)
                label=label.unsqueeze(1).to(torch.float32).to(device)
                model.eval()
                output=model(comment)
                accuracy_batch=get_accuracy_num(output,label,args)
                test_accuracy_all+=accuracy_batch
        
        if args.model_type=='multibase':
            if args.multi_base_model_type.split('+')[0]=='bert':
                for j,(input_ids,token_type_ids,attention_mask,image,label) in tqdm(enumerate(dataloader_test),total=len(dataloader_test)):
                    input_ids,token_type_ids,attention_mask,image=input_ids.to(device),token_type_ids.to(device),attention_mask.to(device),image.to(device)
                    label=label.unsqueeze(1).to(torch.float32).to(device)
                    model.eval()
                    output=model(input_ids,token_type_ids,attention_mask,image)
                    accuracy_batch=get_accuracy_num(output,label,args)
                    test_accuracy_all+=accuracy_batch
            else:
                for j,(comment,image,label) in tqdm(enumerate(dataloader_test),total=len(dataloader_test)):
                    comment,image=comment.to(device),image.to(device)
                    label=label.unsqueeze(1).to(torch.float32).to(device)
                    model.eval()
                    output=model(comment,image)
                    accuracy_batch=get_accuracy_num(output,label,args)
                    test_accuracy_all+=accuracy_batch
                
        test_accuracy=test_accuracy_all/len(dataset)
        return test_accuracy

def load_model(checkpoint,model,optimizer,lr_scheduler):
    start_epoch = checkpoint["epoch"]
    n_no_improve = checkpoint["n_no_improve"]
    best_accuracy = checkpoint["best_accuracy"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    #warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler"])
    return start_epoch,n_no_improve,best_accuracy