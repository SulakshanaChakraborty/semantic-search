from typing_extensions import Required
from sentence_transformers import SentenceTransformer, InputExample, losses,util,evaluation
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import re
import argparse
from datetime import datetime
import os
today_date = datetime.today().strftime('%Y-%m-%d')

def generate_dataset(product_name,target):
    n = len(product_name)
    data = []
    for i in range(n):
        anchor = product_name[i]
        pos = target[i]
        data.append(InputExample(texts=[anchor,pos]))
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path',type = str,required = True)
    parser.add_argument('--pretrained_model_path',type = str,required = True)
    parser.add_argument('--model_save_path',type = str,default = os.path.join('model',f'model-{today_date}'))
    parser.add_argument('--epochs',type = int, default = 1)
    parser.add_argument('--batch_size',type = int, default = 16)
    
    args = parser.parse_args()
    model = SentenceTransformer(args.pretrained_model_path)

    input_path = args.train_data_path

    df = pd.read_excel(input_path)

    df.columns.values[0] = 'Product_Name'
    df.columns.values[1] = 'Target_Product_type'
    
    train_data = generate_dataset(df['Product_Name'].tolist(),df['Target_Product_type'].tolist())

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1) 

    print("training start ..")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs  = args.epochs, output_path  = "log", 
          checkpoint_path = "chekpoints",
          warmup_steps=warmup_steps)

    model.save(args.model_save_path)
    print("model saved!")