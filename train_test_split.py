from operator import index
import pandas as pd
import argparse
import numpy as np
import os

def split_data_save(df,args):
  n_examples = len(df)
  shuffled_idx = np.random.permutation(n_examples)
  
  split_ratio = args.test_split_percentage/100
  if split_ratio > 1:
    raise ValueError('Please enter valid percentage for split')

  train_idx = shuffled_idx[:int((1-split_ratio)*n_examples)]
  test_idx = shuffled_idx[int((1-split_ratio)*n_examples):]

  train_set_df = df.iloc[train_idx]
  test_set_df = df.iloc[test_idx]
  
  output_path = args.output_path
  if not os.path.exists(output_path):
    os.mkdir(output_path)

  train_out_path = os.path.join(output_path,'train.xlsx')
  test_out_path = os.path.join(output_path,'test.xlsx')

  train_set_df.to_excel(train_out_path, index = False)
  test_set_df.to_excel(test_out_path, index = False)
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel_path',type=str,required=True)
    parser.add_argument('--output_path',type=str,default = "train_test_split")
    parser.add_argument('--test_split_percentage',type=int,default = 10)
    ## TO-DO add validation split
    args = parser.parse_args()
    print(args.excel_path)
    df = pd.read_excel(args.excel_path)
    df.columns.values[0] = "Product_Name"
    df.columns.values[1] = "Target_Product_Type"
    split_data_save(df,args)


