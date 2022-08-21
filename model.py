import torch
import torch.nn as nn
from transformers import BertModel
from vit_pytorch import ViT
from torchvision import models

class bert_vit_feature_model(nn.Module):
    def __init__(self,args):
        super(bert_vit_feature_model,self).__init__()
        self.args=args
        self.bert=BertModel.from_pretrained(self.args.pretrained_path)
        self.transformencoderlayer=nn.TransformerEncoderLayer(d_model=args.image_feature+768,nhead=8)
        self.transformencoder=nn.TransformerEncoder(self.transformencoderlayer,num_layers=6)
        self.linear=nn.Linear(self.args.image_feature+768,self.args.num_classes)
        self.dropout=nn.Dropout(p=self.args.dropout_rate)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.w1=nn.Parameter(torch.randn(1))
        self.w2=nn.Parameter(torch.randn(1))
        self.v=v=ViT(channels=3,image_size=self.args.image_size,dropout=self.args.dropout_rate,num_classes=self.args.image_feature,
              emb_dropout=self.args.dropout_rate,patch_size=32,dim=1024,depth=6,heads=16,mlp_dim=2048)
        
    def forward(self,input_ids,token_type_ids,attention_mask,images):
        #batch_size,hidden_size(768)
        bert_output=self.bert(input_ids,token_type_ids,attention_mask).pooler_output
        #batch_size,numclasses
        vit_ouptput=self.v(images)
        if self.args.concat_type=='self-attention':
            feature_concated=torch.cat([bert_output.unsqueeze(1),vit_ouptput.unsqueeze(1)],2).transpose(0,1)
            #(1,batch_size,image_feature+768)-(batch_size,image_feature+768)
            output=self.transformencoder(feature_concated).squeeze(0)
            output=self.dropout(self.linear(output))
        elif self.args.concat_type=='attention':
            feature_concated=torch.cat([self.w1*bert_output,self.w2*vit_ouptput],1)
            output=self.dropout(self.linear(feature_concated))
        elif self.args.concat_type=='concat':
            feature_concated=torch.cat([bert_output,vit_ouptput],1)
            output=self.dropout(self.linear(feature_concated))
        return output
    
class multi_baseline_bert_model(nn.Module):
    def __init__(self,args):
        super(multi_baseline_bert_model,self).__init__()
        self.args=args
        self.bert=BertModel.from_pretrained(self.args.pretrained_path)
        self.res152=models.resnet152(pretrained=True)
        self.vgg19=models.vgg19(pretrained=True)
        self.linear=nn.Linear(1768,self.args.num_classes)
        self.dropout=nn.Dropout(p=self.args.dropout_rate)
        
    def forward(self,input_ids,token_type_ids,attention_mask,images):
        bert_output=self.bert(input_ids,token_type_ids,attention_mask).pooler_output 
        if self.args.multi_base_model_type=='bert+resnet':
            resnet_output=self.res152(images)
            feature_concated=torch.cat([bert_output,resnet_output],1)
            output=self.dropout(self.linear(feature_concated))
        if self.args.multi_base_model_type=='bert+vgg':
            vgg_output=self.vgg19(images)
            feature_concated=torch.cat([bert_output,vgg_output],1)
            output=self.dropout(self.linear(feature_concated))
        return output
          
class multi_baseline_vit_model(nn.Module):
    def __init__(self,args):
        super(multi_baseline_vit_model,self).__init__()
        self.args=args
        self.bilstm=nn.LSTM(input_size=300,batch_first=True,bidirectional=True,hidden_size=self.args.hidden_size)
        self.filter_sizes = (2, 3, 4)
        self.relu=nn.LeakyReLU(inplace=True)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.args.sentence_length, (k, 300)) for k in self.filter_sizes])  
        self.v=v=ViT(channels=3,image_size=self.args.image_size,dropout=self.args.dropout_rate,num_classes=self.args.image_feature,
              emb_dropout=self.args.dropout_rate,patch_size=32,dim=1024,depth=6,heads=16,mlp_dim=2048)
        self.linear1=nn.Linear(self.args.sentence_length*self.args.hidden_size*2+self.args.image_feature,self.args.num_classes)
        self.linear2=nn.Linear(self.args.sentence_length*len(self.filter_sizes)+self.args.image_feature,self.args.num_classes)
        self.dropout=nn.Dropout(p=self.args.dropout_rate)
    
    def conv_and_pool(self,comments,conv):
        comments=self.relu(conv(comments)).squeeze(3)
        comments=nn.functional.max_pool1d(comments,comments.shape[2]).squeeze(2)
        return comments
        
    def forward(self,comments,images):
        vit_output=self.v(images)
        if self.args.multi_base_model_type=='bilstm+vit':
            bilstm_output,_=self.bilstm(comments)
            feature_concated=torch.cat([vit_output,bilstm_output.reshape(bilstm_output.shape[0],-1)],1)
            output=self.dropout(self.linear1(feature_concated))
        if self.args.multi_base_model_type=='textcnn+vit':
            textcnn_output = torch.cat([self.conv_and_pool(comments.unsqueeze(1), conv) for conv in self.convs], 1)
            feature_concated=torch.cat([vit_output,textcnn_output],1)
            output=self.dropout(self.linear2(feature_concated))
        return output
        
class imageModel(nn.Module):
    def __init__(self, args):
        super(imageModel,self).__init__()
        self.args=args
        self.res152=models.resnet152(pretrained=True)
        self.vgg19=models.vgg19(pretrained=True)
        self.dropout=nn.Dropout(p=self.args.dropout_rate)
        self.linear=nn.Linear(1000,self.args.num_classes)
    
    def forward(self,images):
        if self.args.image_model_type=='resnet':
            output=self.res152(images)
        if self.args.image_model_type=='vgg':
            output=self.vgg19(images)
        return self.dropout(self.linear(output))
    
class textModel(nn.Module):
    def __init__(self,args):
        super(textModel,self).__init__()
        self.args=args
        self.bilstm=nn.LSTM(input_size=300,batch_first=True,bidirectional=True,hidden_size=self.args.hidden_size)
        self.linear1=nn.Linear(self.args.sentence_length*self.args.hidden_size*2,512)
        self.linear2=nn.Linear(512,self.args.num_classes)
        self.linear3=nn.Linear(self.args.sentence_length*len(self.filter_sizes),self.args.num_classes)
        self.relu=nn.LeakyReLU(inplace=True)
        self.dropout=nn.Dropout(p=self.args.dropout_rate)
        self.filter_sizes = (2, 3, 4)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.args.sentence_length, (k, 300)) for k in self.filter_sizes])  
        
    def conv_and_pool(self,comments,conv):
        comments=self.relu(conv(comments)).squeeze(3)
        comments=nn.functional.max_pool1d(comments,comments.shape[2]).squeeze(2)
        return comments
      
    def forward(self,comments):
        if self.args.text_model_type=='bilstm':
            output,_=self.bilstm(comments)
            output=self.relu(self.dropout(self.linear1(output.reshape(output.shape[0],-1))))
            output=self.dropout(self.linear2(output))
        elif self.args.text_model_type=='textcnn':
            output = torch.cat([self.conv_and_pool(comments.unsqueeze(1), conv) for conv in self.convs], 1)
            output=self.dropout(self.linear3(output))
        return output
       