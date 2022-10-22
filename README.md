# semantic-search

This folder contains the code to run inference and finetune SBERT.

### To split data into train and test files run the following code:
python train_test_split.py 

Arguments taken are:
'--excel_path' : the input excel file
'--output_path' : output path for saving test and train excel files
'--test_split_percentage' : percentage split, default is 10%

### To finetune the model, run the following code:
python train.py 

Arguments taken are:
'--train_data_path' : the input excel file for finetuning
'--pretrained_model_path' :  path for the model to finetune
'--epochs' : Number of epochs, default 1
'--batch_size' : batch size, default 16

### To test model performance, run the following code:
python test.py

Arguments taken are:
'--model_path' : path for the model
'--input_path' :  path for input data.
'--corpus_path' : The corpus path.
'--text_preprocessing' : Whether to preprocess the data, default False
'--output_path' : output path for saving the result files.