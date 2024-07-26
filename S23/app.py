import s23_openai_clip
from s23_openai_clip import make_train_valid_dfs 
from s23_openai_clip import get_image_embeddings 
from s23_openai_clip import inference_CLIP

import gradio as gr
import zipfile
import os
import pandas as pd
import subprocess

# query_text = "dogs on the grass"
image_path = "./Images"
captions_path = "."
data_source = 'flickr8k.zip'

print("\n\n")
print("Going to unzip dataset")
with zipfile.ZipFile(data_source, 'r') as zip_ref:
    zip_ref.extractall('.')
print("unzip of dataset is done")

#=============================================

cmd = "pwd"
output1 = subprocess.check_output(cmd, shell=True).decode("utf-8")
print("result of pwd command")
print(output1) # result => /home/user/app


# shell command to run
cmd = "ls -l"
output1 = subprocess.check_output(cmd, shell=True).decode("utf-8")
print("result of ls -l command")
print(output1)

#=============================================

print("Going to prepare captions.csv")
df = pd.read_csv("captions.txt")
df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]
df.to_csv("captions.csv", index=False)
df = pd.read_csv("captions.csv")
print("Finished in preparing captions.csv")
print("\n\n")

print("Going to invoke make_train_valid_dfs")
_, valid_df = make_train_valid_dfs()
print("Going to invoke make_train_valid_dfs")
model, image_embeddings = get_image_embeddings(valid_df, "best.pt")


examples1 = ["dogs on the grass",
           "cat and dog",
           "sunny day",
           "raining in forest"]

def greet(query_text):
    print("Going to invoke inference_CLIP")
    return inference_CLIP(query_text)
    
gallery = gr.Gallery(
           label="CLIP result images", show_label=True, elem_id="gallery", 
           columns=[3], rows=[3], object_fit="contain", height="auto")

demo = gr.Interface(fn=greet, 
                    inputs=gr.Dropdown(choices=examples1, label="Pre-defined Prompt"), 
                    outputs=gallery,
                    title="Open AI CLIP")
print("Going to invoke demo.launch")
demo.launch("debug")