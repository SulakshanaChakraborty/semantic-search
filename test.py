from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import re
lemmatizer = WordNetLemmatizer() 
import argparse
from datetime import datetime
import os
today_date = datetime.today().strftime('%Y-%m-%d')

lemmatizer = WordNetLemmatizer() 

# clean data
def clean_data(list_df):
  list_df = [re.sub(r'\d+',' ',x) for x in list_df]

  list_df = [x.lower() for x in list_df]
  list_df = [x.split() for x in list_df]
  list_df = [[lemmatizer.lemmatize(word) for word in x] for x in list_df]
  list_df = [' '.join(x)for x in list_df]
  return list_df



# generate output file
def generate_matched_file(args,model,product_name,product_type,target):

   product_name_embeddings = model.encode(product_name)
   product_type_embeddings = model.encode(product_type)
   cosine_sim_product_bert = util.pytorch_cos_sim(product_name_embeddings,product_type_embeddings).numpy()
  
   cosine_sim_product = cosine_sim_product_bert
   max_ids = (-cosine_sim_product).argsort(axis = 1)[:,:3]

   prod_arr = np.array(gs_prod_type_df['Product_Type'].tolist())

   d = {'product_name':product_name,'Target_Product_Type':target,'1st_Match':prod_arr[max_ids[:,0]],'2nd_Match':prod_arr[max_ids[:,1]],'3rd_Match':prod_arr[max_ids[:,2]]} 
   res = pd.DataFrame(data = d)

   
 
   res.to_csv(os.path.join(args.output_path,f'results-{today_date}.csv'))

   mismatches = res[res['Target_Product_Type'] != res['1st_Match']]
   mismatches.to_csv(os.path.join(args.output_path,f'mismatches-{today_date}.csv'))

   matches = res[res['Target_Product_Type'] == res['1st_Match']]
   matches.to_csv(os.path.join(args.output_path,f'matches-{today_date}.csv'))

# check similairty
def generate_similarity(model,product_name,product_type,target):
  product_name_embeddings = model.encode(product_name)
  product_type_embeddings = model.encode(product_type)
  cosine_sim_product_bert = util.pytorch_cos_sim(product_name_embeddings,product_type_embeddings).numpy()

  cosine_sim_product = cosine_sim_product_bert
  max_ids = (-cosine_sim_product).argsort(axis = 1)
  prod_arr = np.array(gs_prod_type_df['Product_Type'].tolist())
  match_1 = prod_arr[max_ids[:,0]]

  print("---------------------------------------------------------------------------------------------")
  print("accuracy of first match:",np.round(np.mean(match_1 == target),3))
  print("---------------------------------------------------------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str,required = True)
    parser.add_argument('--input_path', type = str,required = True)
    parser.add_argument('--corpus_path', type = str,required = True)
    parser.add_argument('--text_preprocessing', type = str,default = False)
    parser.add_argument('--output_path', type = str,default = 'output')

    args = parser.parse_args()
    path = args.model_path
    # print(path)
    model = SentenceTransformer(path)
    input_df = pd.read_excel(args.input_path)
    gs_prod_type_df = pd.read_excel(args.corpus_path)

    # ensure column 3 has product type
    gs_prod_type_df.columns.values[2] = 'Product_Type'
    
    if args.text_preprocessing:
        print("cleaning text!")
        product_name = clean_data(input_df['Product_Name'].tolist())
        product_type = clean_data(gs_prod_type_df['Product_Type'].tolist())
    else:
        product_name =  input_df['Product_Name'].tolist()
        product_type = gs_prod_type_df['Product_Type'].tolist()


    target = input_df['Target_Product_Type'].tolist()

    
    generate_similarity(model,product_name,product_type,target)
    
    print("generating output files.. ")
    generate_matched_file(args,model,product_name,product_type,target)






    






