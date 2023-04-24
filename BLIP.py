import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from datasets import load_dataset, Image
from transformers import BlipForQuestionAnswering, BlipProcessor, BlipConfig, BlipModel
import os
import logging
import argparse

logger = logging.getLogger(__name__)

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, config, image_files, text, processor, num_labels):
        self.config = config
        self.image_files = image_files
        self.text = text
        self.processor = processor
        self.num_labels = num_labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        text = self.text
        image = self.image_files[idx][image_name]
        label = self.image_files[idx][label_name]
        if image.mode != "RGB":
            image = image.convert("RGB")
        encoding = self.processor(image, text, return_tensors="pt")
        label_encoding = self.processor(text=self.config.id2label[str(label)], padding="max_length", return_tensors="pt").input_ids
        label_encoding = label_encoding.squeeze()

        # remove batch dimension
        for k,v in encoding.items():
            encoding[k] = v.squeeze()
        encoding["labels"] = label_encoding
        return encoding
    
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['pixel_values'] = torch.stack(pixel_values)
    batch['labels'] = torch.stack(labels)
    return batch

@torch.no_grad()
def evaluate(blip_model, device, processor, test_dataloader):
    #losses = []  # List of scalar tensors
    total_acc = 0.0
    for batch in tqdm(test_dataloader):
        batch = {k:v.to(device) for k,v in batch.items()}
        outputs = blip_model.generate(**batch)  
        outputs = outputs[:,0:max_label_token_length]
        labels = batch['labels'][:,0:max_label_token_length]
        outputs = processor.batch_decode(outputs, skip_special_tokens=True)
        labels = processor.batch_decode(labels, skip_special_tokens=True)
        #print('outputs:', outputs)
        #print('labels:', labels)
        correct = 0
        for i in range(len(labels)):
            output = outputs[i].replace(" ", "")
            label = labels[i]
            if output == label:
                correct += 1
        acc = (correct * 100) / len(labels)
        total_acc += acc
        #print(acc)

    #stacked_losses = torch.stack(losses)  # (num_batches, ) 
    #total_avg_loss = stacked_losses.mean()  # (num test examples, ) -> scalar
    total_avg_acc = total_acc / len(test_dataloader)
    #logger.info(f"Correct: {correct} / Total: {total}")
    #logger.info(f"Average val loss: {total_avg_loss.item()}")
    logger.info(f"Average val acc: {total_avg_acc}")
    return total_avg_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", default=0, type=int)
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--log_dir", default="./logs/")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"]) 
    parser.add_argument("--adaptation_name", default="no_text", choices=["no_text", "question", "class_names", "task_description"])
    parser.add_argument("--train_batch_size", default=32, type=int) 
    parser.add_argument("--test_batch_size", default=32, type=int) 
    parser.add_argument("--lr", default=5e-05, type=float) 
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--max_patience", default=1, type=int)

    args = parser.parse_args()

    cache_dir=args.cache_dir
    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir
    device = "cuda:"+str(args.device_num)
    dataset = args.dataset
    dataset_name = dataset.split('/')[-1]
    adaptation_name = args.adaptation_name
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    lr = args.lr
    num_epochs = args.num_epochs
    max_patience = args.max_patience

    if dataset_name == 'cifar10':
        image_name = 'img'
        label_name = 'label'
        trainset_name = 'train'
        testset_name = 'test'
    elif dataset_name == 'cifar100':
        image_name = 'img'
        label_name = 'fine_label'
        trainset_name = 'train'
        testset_name = 'test'

    if adaptation_name == 'no_text':
        adaptation = ''
    elif adaptation_name == 'question':
        adaptation = 'What is this image?'
    elif adaptation_name == 'class_names':
        if dataset_name == 'cifar10':
            adaptation = "The image belongs to one of the following classes: 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'."
        elif dataset_name == 'cifar100':
            adaptation = adaptation = "The image belongs to one of the following classes: 'worm', 'tank', 'crab', 'bee', 'orchid', 'skunk', 'woman', 'hamster', 'plate', 'table', 'house', 'possum', 'lobster', 'rocket', 'elephant', 'oak_tree', 'girl', 'whale', 'bridge', 'leopard', 'sea', 'sweet_pepper', 'wardrobe', 'crocodile', 'pickup_truck', 'chimpanzee', 'bus', 'clock', 'castle', 'bed', 'pear', 'snake', 'squirrel', 'bear', 'telephone', 'forest', 'trout', 'seal', 'flatfish', 'pine_tree', 'boy', 'cockroach', 'apple', 'aquarium_fish', 'skyscraper', 'lamp', 'porcupine', 'tulip', 'butterfly', 'couch', 'wolf', 'man', 'cattle', 'bottle', 'raccoon', 'dolphin', 'lizard', 'train', 'dinosaur', 'mushroom', 'caterpillar', 'poppy', 'sunflower', 'keyboard', 'bicycle', 'fox', 'plain', 'turtle', 'mountain', 'ray', 'cloud', 'kangaroo', 'baby', 'cup', 'beaver', 'shark', 'mouse', 'television', 'snail', 'spider', 'shrew', 'rose', 'motorcycle', 'camel', 'rabbit', 'otter', 'willow_tree', 'orange', 'streetcar', 'road', 'maple_tree', 'lion', 'palm_tree', 'tiger', 'beetle', 'lawn_mower', 'can', 'chair', 'bowl', 'tractor'."
    elif adaptation_name == 'task_description':
        if dataset_name == 'cifar10':
            adaptation = "This is a classification problem from the cifar10 dataset. Classifiy the images amongst the classes 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'."
        elif dataset_name == 'cifar100':
            adaptation = adaptation = "This is a classification problem from the cifar100 dataset. Classifiy the images amongst the classes 'worm', 'tank', 'crab', 'bee', 'orchid', 'skunk', 'woman', 'hamster', 'plate', 'table', 'house', 'possum', 'lobster', 'rocket', 'elephant', 'oak_tree', 'girl', 'whale', 'bridge', 'leopard', 'sea', 'sweet_pepper', 'wardrobe', 'crocodile', 'pickup_truck', 'chimpanzee', 'bus', 'clock', 'castle', 'bed', 'pear', 'snake', 'squirrel', 'bear', 'telephone', 'forest', 'trout', 'seal', 'flatfish', 'pine_tree', 'boy', 'cockroach', 'apple', 'aquarium_fish', 'skyscraper', 'lamp', 'porcupine', 'tulip', 'butterfly', 'couch', 'wolf', 'man', 'cattle', 'bottle', 'raccoon', 'dolphin', 'lizard', 'train', 'dinosaur', 'mushroom', 'caterpillar', 'poppy', 'sunflower', 'keyboard', 'bicycle', 'fox', 'plain', 'turtle', 'mountain', 'ray', 'cloud', 'kangaroo', 'baby', 'cup', 'beaver', 'shark', 'mouse', 'television', 'snail', 'spider', 'shrew', 'rose', 'motorcycle', 'camel', 'rabbit', 'otter', 'willow_tree', 'orange', 'streetcar', 'road', 'maple_tree', 'lion', 'palm_tree', 'tiger', 'beetle', 'lawn_mower', 'can', 'chair', 'bowl', 'tractor'."

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=log_dir+'BLIP_'+dataset_name+'_'+adaptation_name+'.txt', 
                        level=logging.INFO,
					    format='%(asctime)s %(message)s', 
					    filemode='w') 
    logger.info(f"train_batch_size: {train_batch_size}")
    logger.info(f"test_batch_size: {test_batch_size}")
    logger.info(f"lr: {lr}")
    logger.info(f"num_epochs: {num_epochs}")
    logger.info(f"max_patience: {max_patience}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    datasets = load_dataset(dataset, cache_dir=cache_dir)
    #datasets = load_dataset('cifar10', cache_dir=cache_dir)
    label_list = datasets["train"].features[label_name].names
    if dataset_name == 'cifar100':
        # replace label 'cra' to 'crab'
        label_list.remove('cra')
        label_list.append('crab')
    num_labels = len(label_list)

    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

    max_label_token_length = 0
    for label in label_list:
        labels = processor(text=label, return_tensors="pt").input_ids
        max_label_token_length = max(max_label_token_length, len(labels[0]))

    config = BlipConfig.from_pretrained("Salesforce/blip-vqa-base")
    config.id2label = {str(i): label for i, label in enumerate(label_list)}
    config.label2id = {label: str(i) for i, label in enumerate(label_list)}
    config.num_labels = num_labels
    config.text_config.max_length = max_label_token_length

    train_dataset = ImageDataset(config=config, image_files=datasets[trainset_name], text=adaptation, processor=processor, num_labels=num_labels)
    test_dataset = ImageDataset(config=config, image_files=datasets[testset_name], text=adaptation, processor=processor, num_labels=num_labels)

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=test_batch_size, shuffle=True)

    blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", config=config)
    blip_model = blip_model.to(device)
    for name, param in blip_model.named_parameters():
        if 'text_decoder.cls' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = optim.AdamW(blip_model.parameters(), lr=lr)

    best_test_acc = -1
    step = -1
    patience = 0
    blip_model.train()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        
        logger.info(f"Train Epoch: {epoch}")
        for batch in tqdm(train_dataloader):
            step += 1
            batch = {k:v.to(device) for k,v in batch.items()}
            optimizer.zero_grad()
            outputs = blip_model(**batch) 
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        blip_model.eval()
        logger.info(f"Evaluate Epoch: {epoch}")
        new_test_acc = evaluate(blip_model, device, processor, test_dataloader)
        # save checkpoint with best test loss
        if new_test_acc > best_test_acc or best_test_acc < 0:
            patience = 0
            if best_test_acc > 0:
                os.remove(checkpoint_dir + '/'+ best_checkpoint_filename)
            best_checkpoint_filename = 'BLIP_'+dataset_name+'_'+adaptation_name+'_'+str(epoch) +".pt"
            torch.save(blip_model.state_dict(), checkpoint_dir + '/' + best_checkpoint_filename)
            best_test_acc = new_test_acc
            logger.info(f"Best test acc: {best_test_acc}")
            logger.info(f"Best model saved at {checkpoint_dir + '/' + best_checkpoint_filename}")
        else:
            patience += 1
            if patience > max_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        blip_model.train()