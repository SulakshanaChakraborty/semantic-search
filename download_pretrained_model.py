from sentence_transformers import SentenceTransformer
import os

print("downloading model ..")
model = SentenceTransformer('all-mpnet-base-v2')
dir = os.path.join('model','pretrained_model')

if not os.path.exists(dir):
    os.makedirs(dir)

model.save(dir)