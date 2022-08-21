import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import os
import pandas as pd
from gensim.models import keyedvectors
import jieba

class multimodelDataset(Dataset):
    def __init__(self,args,tokenizer,transform,train,val):
        self.args=args
        self.train=train
        self.val=val
        if self.train:
            self.df=pd.read_csv(self.args.comment_dir+'/data_train.csv',low_memory=False)
        elif self.val:
            self.df=pd.read_csv(self.args.comment_dir+'/data_val.csv',low_memory=False)
        else:
            self.df=pd.read_csv(self.args.comment_dir+'/data_test.csv',low_memory=False)
        self.image_dir=self.args.image_dir
        self.transform=transform
        self.tokenizer=tokenizer
        
    def __len__(self):
        return len(self.df)  
     
    def __getitem__(self,index):
        image_id_list=eval(self.df['图片id'][index])
        if len(image_id_list)==0:
            image=torch.randn(3,224,224)
        elif len(image_id_list)==1:
            image_path=os.path.join(self.image_dir,image_id_list[0]+'.jpg')
            image=Image.open(image_path).convert('RGB')
            image=self.transform(image)
        elif len(image_id_list)>1:
            image=torch.zeros(3,224,224)
            for i in range(len(image_id_list)):
                image_path=os.path.join(self.image_dir,image_id_list[i]+'.jpg')
                one_image=Image.open(image_path).convert('RGB')
                one_image=self.transform(one_image)
                image+=one_image
            image=image/len(image_id_list)
    
        comment=self.df['评论processed'][index]
        comment=self.tokenizer(comment,max_length=512,return_tensors='pt',truncation=True,padding='max_length')
        input_ids=comment['input_ids']
        token_type_ids=comment['token_type_ids']
        attention_mask=comment['attention_mask']
        label=int(self.df['评价性质'][index])
        return input_ids,token_type_ids,attention_mask,image,label

class multi_base_textDataset(Dataset):
    def __init__(self,args,transform,train,val):
        self.args=args
        self.train=train
        self.val=val
        self.image_dir=self.args.image_dir
        if self.train:
            self.df=pd.read_csv(self.args.comment_dir+'/data_train.csv',low_memory=False)
        elif self.val:
            self.df=pd.read_csv(self.args.comment_dir+'/data_val.csv',low_memory=False)
        else:
            self.df=pd.read_csv(self.args.comment_dir+'/data_test.csv',low_memory=False)
        self.transform=transform
        self.model=keyedvectors.load_word2vec_format(args.pretrained_path+'/sgns.weibo.bigram-char')
        
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self,index):
        comment=self.df['评论processed'][index]
        if len(comment)>self.args.sentence_length:
            comment_list=list(jieba.cut(comment[0:self.args.sentence_length],cut_all=False))
        else:
            comment_list=list(jieba.cut(comment,cut_all=False))
        comment_embed=[]
        for i in comment_list:
            try:
                comment_embed.append(torch.tensor(self.model.get_vector(i)))
            except:
                comment_embed.append(torch.zeros(300))
        comment=torch.stack(comment_embed,0)
        if comment.shape[0]<self.args.sentence_length:
            comment=torch.cat((comment,torch.zeros(1024-comment.shape[0],300)),dim=0)
        label=int(self.df['评价性质'][index])
        
        image_id_list=eval(self.df['图片id'][index])
        if len(image_id_list)==0:
            image=torch.randn(3,224,224)
        elif len(image_id_list)==1:
            image_path=os.path.join(self.image_dir,image_id_list[0]+'.jpg')
            image=Image.open(image_path).convert('RGB')
            image=self.transform(image)
        elif len(image_id_list)>1:
            image=torch.zeros(3,224,224)
            for i in range(len(image_id_list)):
                image_path=os.path.join(self.image_dir,image_id_list[i]+'.jpg')
                one_image=Image.open(image_path).convert('RGB')
                one_image=self.transform(one_image)
                image+=one_image
            image=image/len(image_id_list)
        return comment,image,label
    
class imageDataset(Dataset):
    def __init__(self,args,transform,train,val):
        self.args=args
        self.train=train
        self.val=val
        if self.train:
            self.df=pd.read_csv(self.args.comment_dir+'/data_train.csv',low_memory=False)
        elif self.val:
            self.df=pd.read_csv(self.args.comment_dir+'/data_val.csv',low_memory=False)
        else:
            self.df=pd.read_csv(self.args.comment_dir+'/data_test.csv',low_memory=False)
        self.image_dir=self.args.image_dir
        self.transform=transform
        id_list=[]
        for i in self.df['图片id'].to_list():
            if len(eval(i))!=0:
                id_list.append(eval(i))
        self.id_list=id_list
        
    def __len__(self):
        return len(self.id_list)  
     
    def __getitem__(self,index):
        image_id_list=self.id_list[index]
        if len(image_id_list)==1:
            image_path=os.path.join(self.image_dir,image_id_list[0]+'.jpg')
            image=Image.open(image_path).convert('RGB')
            image=self.transform(image)
        elif len(image_id_list)>1:
            image=torch.zeros(3,224,224)
            for i in range(len(image_id_list)):
                image_path=os.path.join(self.image_dir,image_id_list[i]+'.jpg')
                one_image=Image.open(image_path).convert('RGB')
                one_image=self.transform(one_image)
                image+=one_image
            image=image/len(image_id_list)
        label=int(self.df['评价性质'][index])
        return image,label
    
class textDataset(Dataset):
    def __init__(self,args,train,val):
        self.args=args
        self.train=train
        self.val=val
        if self.train:
            self.df=pd.read_csv(self.args.comment_dir+'/data_train.csv',low_memory=False)
        elif self.val:
            self.df=pd.read_csv(self.args.comment_dir+'/data_val.csv',low_memory=False)
        else:
            self.df=pd.read_csv(self.args.comment_dir+'/data_test.csv',low_memory=False)
        self.model=keyedvectors.load_word2vec_format(args.pretrained_path+'/sgns.weibo.bigram-char')
        
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self,index):
        comment=self.df['评论processed'][index]
        if len(comment)>self.args.sentence_length:
            comment_list=list(jieba.cut(comment[0:self.args.sentence_length],cut_all=False))
        else:
            comment_list=list(jieba.cut(comment,cut_all=False))
        comment_embed=[]
        for i in comment_list:
            try:
                comment_embed.append(torch.tensor(self.model.get_vector(i)))
            except:
                comment_embed.append(torch.zeros(300))
        comment=torch.stack(comment_embed,0)
        if comment.shape[0]<self.args.sentence_length:
            comment=torch.cat((comment,torch.zeros(1024-comment.shape[0],300)),dim=0)
        label=int(self.df['评价性质'][index])
        return comment,label
    
class collater():
    def __init__(self, args):
        self.args=args

    def __call__(self,batch):
        if self.args.model_type=='multimodel' or self.args.model_type=='multibase' and self.args.multi_base_model_type.split('+')[0]=='bert':
            input_ids=[item[0] for item in batch]
            token_type_ids=[item[1] for item in batch]
            attention_mask=[item[2] for item in batch]
            images_list=[item[3] for item in batch]
            labels=[item[4] for item in batch]
            size_max=0
            images=[]
            for i in images_list:
                if i.size()!=torch.Size([]):
                    size_max=max(size_max,i.size()[0])
            for j in images_list:
                if j.size()==torch.Size([]):
                    j=torch.zeros((size_max,224,224))
                elif  j.size()[0]<size_max:
                    j=torch.cat([j,torch.zeros((size_max-j.size()[0],224,224))],0)
                images.append(j)  
            input_ids=torch.stack(input_ids,dim=0).squeeze(1)
            token_type_ids=torch.stack(token_type_ids,dim=0).squeeze(1)
            attention_mask=torch.stack(attention_mask,dim=0).squeeze(1)
            images=torch.stack(images,dim=0)
            labels=torch.LongTensor(labels)
            return (input_ids,token_type_ids,attention_mask,images,labels)  
        
        if self.args.model_type=='image':
            images_list=[item[0] for item in batch]
            labels=[item[1] for item in batch]
            size_max=0
            images=[]
            for i in images_list:
                if i.size()!=torch.Size([]):
                    size_max=max(size_max,i.size()[0])
            for j in images_list:
                if j.size()==torch.Size([]):
                    j=torch.zeros((size_max,224,224))
                elif  j.size()[0]<size_max:
                    j=torch.cat([j,torch.zeros((size_max-j.size()[0],224,224))],0)
                images.append(j) 
            images=torch.stack(images,dim=0)
            labels=torch.LongTensor(labels) 
            return (images,labels)
        
        if self.args.model_type=='text':
            comments=[item[0] for item in batch]
            labels=[item[1] for item in batch]
            comments=torch.stack(comments,dim=0)
            labels=torch.LongTensor(labels)
            return (comments,labels)
        
        if self.args.model_type=='multibase' and self.args.multi_base_model_type.split('+')[0]!='bert':
            comments=[item[0] for item in batch]
            images_list=[item[1] for item in batch]
            labels=[item[2] for item in batch]
            labels=torch.LongTensor(labels)
            size_max=0
            images=[]
            for i in images_list:
                if i.size()!=torch.Size([]):
                    size_max=max(size_max,i.size()[0])
            for j in images_list:
                if j.size()==torch.Size([]):
                    j=torch.zeros((size_max,224,224))
                elif  j.size()[0]<size_max:
                    j=torch.cat([j,torch.zeros((size_max-j.size()[0],224,224))],0)
                images.append(j) 
            images=torch.stack(images,dim=0)
            comments=torch.stack(comments,dim=0)
            return (comments,images,labels)
            
        
        