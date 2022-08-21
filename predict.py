from model import bert_vit_feature_model
from dataset import multimodelDataset,collater
import pandas as pd
from torchvision import transforms
from transformers import BertTokenizer
from utilis import *
import argparse
import torch


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    
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
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset=multimodelDataset(args,tokenizer,transform,predict=True)
    collate_fn=collater(args)
    dataloader_predict=DataLoaderX(dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn,num_workers=args.num_workers)
    model_pretrained=torch.load(args.checkpoint_dir + '/m-best_'+args.concat_type+'.pth.tar')
    model=bert_vit_feature_model(args).to(device)
    model.load_state_dict(model_pretrained['state_dict'])
    result=[]
    for k,(input_ids,token_type_ids,attention_mask,image,label) in tqdm(enumerate(dataloader_predict),total=len(dataloader_predict)):
        input_ids,token_type_ids,attention_mask,image=input_ids.to(device),token_type_ids.to(device),attention_mask.to(device),image.to(device)
        model.eval()
        output=model(input_ids,token_type_ids,attention_mask,image)
        if float(torch.sigmoid(output[0]))>0.5:
            result.append(1)
        else:
            result.append(0)
    df=pd.read_csv(args.comment_dir+'/data_predict.csv',low_memory=False)
    df['追评性质']=result
    df.to_csv(args.comment_dir+'/data_predict_final.csv')
            