# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from tqdm.notebook import tqdm
# from sklearn.metrics import accuracy_score
# import numpy as np
# from datasets import load_dataset, Image
# from transformers import BlipForQuestionAnswering, BlipProcessor, BlipConfig, BlipModel
import os
import logging
import argparse
import openai
from dotenv import load_dotenv
import datetime
import time

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_file", default="./BLIP_Final_Results/BLIP_cifar100_no_text_outputs.txt")
    parser.add_argument("--labels_file", default="./BLIP_Final_Results/BLIP_cifar100_no_text_labels.txt")
    parser.add_argument("--log_dir", default="./logs/")
    parser.add_argument("--evaluator_batch_size", default=10, type=int)
    
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    args = parser.parse_args()

    log_dir = args.log_dir
    evaluator_batch_size = args.evaluator_batch_size
    
    orig_outputs = None
    labels = None
    with open(args.outputs_file, 'r') as f:
        orig_outputs = f.read().splitlines()
    with open(args.labels_file, 'r') as f:
        labels = f.read().splitlines()
    
    
    # Build the base prompt
#     base_prompt = """I will give you a list of reference words and phrases, followed by a list of candidate words and phrases. For each candidate, if the candidate is synonymous with an item from the reference list, please print the item. If not, please print the string “NONE”. Please print just the output, with no explanation. 

# Reference list: \"""
# worm
# tank
# crab
# bee
# orchid
# skunk
# woman
# hamster
# plate
# table
# house
# possum
# lobster
# rocket
# elephant
# oak_tree
# girl
# whale
# bridge
# leopard
# sea
# sweet_pepper
# wardrobe
# crocodile
# pickup_truck
# chimpanzee
# bus
# clock
# castle
# bed
# pear
# snake
# squirrel
# bear
# telephone
# forest
# trout
# seal
# flatfish
# pine_tree
# boy
# cockroach
# apple
# aquarium_fish
# skyscraper
# lamp
# porcupine
# tulip
# butterfly
# couch
# wolf
# man
# cattle
# bottle
# raccoon
# dolphin
# lizard
# train
# dinosaur
# mushroom
# caterpillar
# poppy
# sunflower
# keyboard
# bicycle
# fox
# plain
# turtle
# mountain
# ray
# cloud
# kangaroo
# baby
# cup
# beaver
# shark
# mouse
# television
# snail
# spider
# shrew
# rose
# motorcycle
# camel
# rabbit
# otter
# willow_tree
# orange
# streetcar
# road
# maple_tree
# lion
# palm_tree
# tiger
# beetle
# lawn_mower
# can
# chair
# bowl
# tractor
# \"""

# Example input: \"""
# TV
# rooftop
# ocean
# ice
# \"""

# Example output: \"""
# television
# NONE
# sea
# NONE
# \"""

# Input: \"""
# {}
# \"""
#     """

    base_prompt = """I will give you a list of reference words and phrases, followed by a list of candidate words and phrases. For each candidate, if the candidate is synonymous with an item from the reference list, please print the item. If not, please print the string “NONE”. Please print just the output, with no explanation. 

Reference list: \"""
worm
tank
crab
bee
orchid
skunk
woman
hamster
plate
table
house
possum
lobster
rocket
elephant
oaktree
girl
whale
bridge
leopard
sea
sweetpepper
wardrobe
crocodile
pickup_truck
chimpanzee
bus
clock
castle
bed
pear
snake
squirrel
bear
telephone
forest
trout
seal
flatfish
pinetree
boy
cockroach
apple
aquarium_fish
skyscraper
lamp
porcupine
tulip
butterfly
couch
wolf
man
cattle
bottle
raccoon
dolphin
lizard
train
dinosaur
mushroom
caterpillar
poppy
sunflower
keyboard
bicycle
fox
plain
turtle
mountain
ray
cloud
kangaroo
baby
cup
beaver
shark
mouse
television
snail
spider
shrew
rose
motorcycle
camel
rabbit
otter
willow_tree
orange
streetcar
road
maple_tree
lion
palm_tree
tiger
beetle
lawn_mower
can
chair
bowl
tractor
\"""

Example input: \"""
TV
rooftop
ocean
ice
\"""

Example output: \"""
television
NONE
sea
NONE
\"""

Input: \"""
{}
\"""
    """
    
    print(f"Base prompt: {base_prompt}")
    
    now = datetime.datetime.now()
    translation_fname = args.outputs_file[:-11] + "translations_" + now.strftime("%Y-%m-%d_%H:%M:%S") + ".txt"
    
    # Get gpt-3.5-turbo labels
    translated_outputs = []
    total = 0
    correct = 0
    for i in range(0, len(orig_outputs), evaluator_batch_size):
        print(f"Starting index is {i} out of {len(orig_outputs)} total")
        chunk = orig_outputs[i:i+evaluator_batch_size]
        chunk_str = "\n".join(chunk)
        prompt = base_prompt.format(chunk_str)
        
        response_handled = False
        while not response_handled:
            try:
                response = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages=[{
                      "role": "user",
                      "content": prompt
                  }],
                  temperature=0.6
                )
                response_handled = True
            except openai.error.RateLimitError:
                time.sleep(1)
        
        responses = response["choices"][0]["message"]["content"].split('\n')
        translated_outputs += responses
        
        translation_lines = []
        for j in range(len(responses)):
            translation_lines.append(orig_outputs[i+j] + " -> " + responses[j] + "\n")
            if orig_outputs[i+j] == responses[j]:
                correct += 1
            total += 1
        with open(translation_fname, 'a') as f:
            f.writelines(translation_lines)
        print(f"Running acc: {(correct/total):.2f}")
    
    acc = correct / total
    print(acc)
    
    with open(translation_fname, 'a') as f:
        f.write("Final accuracy: " + str(acc) + "\n")
    
    
    