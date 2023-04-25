from transformers import ViltForQuestionAnswering, ViltConfig, ViltProcessor, BertTokenizer
from torchmultimodal.models.flava.model import flava_model_for_classification
from PIL import Image
import torchvision.transforms as T
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import os
import logging
import argparse

logger = logging.getLogger(__name__)

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, image_files, text, processor, num_labels):
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
        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k,v in encoding.items():
            encoding[k] = v.squeeze()
        targets = torch.zeros(self.num_labels)
        targets[label] = 1
        encoding["labels"] = targets
        return encoding

def collate_fn(batch):
    transform = transform = T.Resize((224,224))
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # create padded pixel values and corresponding pixel mask
    encoding = processor.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")

    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['pixel_values'] = transform(encoding['pixel_values'])
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = torch.stack(labels)

    return batch

@torch.no_grad()
def evaluate(model, device, test_dataloader):
    losses = []  # List of scalar tensors
    correct = 0
    total = 0
    for batch in tqdm(test_dataloader):
        # adapt batch to model
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(text = batch["input_ids"].to(device), image = batch["pixel_values"].to(device), labels = batch["labels"].to(device))
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        target = torch.argmax(batch['labels'], dim=1)
        correct += torch.sum(preds==target).item()
        total += target.size(0)
        losses.append(outputs.loss)
    stacked_losses = torch.stack(losses)  # (num_batches, ) 
    total_avg_loss = stacked_losses.mean()  # (num test examples, ) -> scalar
    total_avg_acc = (100 * correct) / total
    logger.info(f"Correct: {correct} / Total: {total}")
    logger.info(f"Average val loss: {total_avg_loss.item()}")
    logger.info(f"Average val acc: {total_avg_acc}")
    return total_avg_loss.item(), total_avg_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", default=0, type=int)
    parser.add_argument("--cache_dir", default="./cache")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--log_dir", default="./logs/")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"]) 
    parser.add_argument("--adaptation_name", default="no_text", choices=["no_text", "question", "class_names", "task_description"])
    parser.add_argument("--train_batch_size", default=64, type=int) 
    parser.add_argument("--test_batch_size", default=64, type=int) 
    parser.add_argument("--lr", default=5e-5, type=float) 
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
            adaptation = 'The image belongs to one of the following classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.'
        elif dataset_name == 'cifar100':
            adaptation = adaptation = 'The image belongs to one of the following classes: beaver, dolphin, otter, seal, whale ,aquarium fish, flatfish, ray, shark, trout, orchids, poppies, roses, sunflowers, tulips, bottles, bowls, cans, cups, plates, apples, mushrooms, oranges, pears, sweet peppers, clock, computer keyboard, lamp, telephone, television, bed, chair, couch, table, wardrobe, bee, beetle, butterfly, caterpillar, cockroach, bear, leopard, lion, tiger, wolf, bridge, castle, house, road, skyscraper, cloud, forest, mountain, plain, sea, camel, cattle, chimpanzee, elephant, kangaroo, fox, porcupine, possum, raccoon, skunk, crab, lobster, snail, spider, worm, baby, boy, girl, man, woman, crocodile, dinosaur, lizard, snake, turtle, hamster, mouse, rabbit, shrew, squirrel, maple, oak, palm, pine, willow, bicycle, bus, motorcycle, pickup truck, train, lawn-mower, rocket, streetcar, tank, tractor.'
    elif adaptation_name == 'task_description':
        if dataset_name == 'cifar10':
            adaptation = "This is a classification problem from the cifar10 dataset. Classify the images amongst the classes 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', and 'truck'."
        elif dataset_name == 'cifar100':
            adaptation = adaptation = "This is a classification problem from the cifar100 dataset. Classify the images amongst the classes 'beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium fish', 'flatfish', 'ray', 'shark', 'trout', 'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', 'bottles', 'bowls', 'cans', 'cups', 'plates', 'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', 'clock', 'computer keyboard', 'lamp', 'telephone', 'television', 'bed', 'chair', 'couch', 'table', 'wardrobe', 'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'bear', 'leopard', 'lion', 'tiger', 'wolf', 'bridge', 'castle', 'house', 'road', 'skyscraper', 'cloud', 'forest', 'mountain', 'plain', 'sea', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crab', 'lobster', 'snail', 'spider', 'worm', 'baby', 'boy', 'girl', 'man', 'woman', 'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 'maple', 'oak', 'palm', 'pine', 'willow', 'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'."

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=log_dir+'FLAVA_'+dataset_name+'_'+adaptation_name+'.txt', 
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
    label_list = datasets["train"].features[label_name].names
    num_labels = len(label_list)

    config = BertTokenizer.from_pretrained("bert-base-uncased",padding="max_length", cache_dir=cache_dir)
    config.id2label = {str(i): label for i, label in enumerate(label_list)}
    config.label2id = {label: str(i) for i, label in enumerate(label_list)}
    config.num_labels = num_labels

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    train_dataset = ImageDataset(image_files=datasets[trainset_name], text=adaptation, processor=processor, num_labels=num_labels)
    test_dataset = ImageDataset(image_files=datasets[testset_name], text=adaptation, processor=processor, num_labels=num_labels)

    model = flava_model_for_classification(num_classes=num_labels)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'classifier' in name or 'pooler' in name:
            param.requires_grad = True

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=test_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_test_acc = -1
    step = -1
    patience = 0
    model.train()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        logger.info(f"Train Epoch: {epoch}")
        for batch in tqdm(train_dataloader):
            step += 1
            # get the inputs; 
            batch = {k:v.to(device) for k,v in batch.items()}
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(text = batch["input_ids"].to(device), image = batch["pixel_values"].to(device), labels = batch["labels"].to(device))
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            target = torch.argmax(batch['labels'], dim=1)
            correct = torch.sum(preds==target).item()
            acc = (correct * 100) / target.size(0)
            loss = outputs.loss
            logger.info(f"Step {step} - loss: {loss.item()} , train acc: {acc}")
            loss.backward()
            optimizer.step()


        model.eval()
        logger.info(f"Evaluate Epoch: {epoch}")
        new_test_loss, new_test_acc = evaluate(model, device, test_dataloader)
        # save checkpoint with best test loss
        if new_test_acc > best_test_acc or best_test_acc < 0:
            patience = 0
            if best_test_acc > 0:
                os.remove(checkpoint_dir + '/'+ best_checkpoint_filename)
            best_checkpoint_filename = 'FLAVA_'+dataset_name+'_'+adaptation_name+'_'+str(epoch) +".pt"
            torch.save(model.state_dict(), checkpoint_dir + '/' + best_checkpoint_filename)
            best_test_acc = new_test_acc
            logger.info(f"Best test acc: {best_test_acc}")
            logger.info(f"Best model saved at {checkpoint_dir + '/' + best_checkpoint_filename}")
        else:
            patience += 1
            if patience > max_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        model.train()

    logger.info(f"Best model acc {best_test_acc}")
    logger.info("Finished Training!")