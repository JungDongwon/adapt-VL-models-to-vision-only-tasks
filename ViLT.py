from transformers import ViltForQuestionAnswering, ViltConfig, ViltProcessor
from PIL import Image
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import os
import logging

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
        image = self.image_files[idx]['img']
        label = self.image_files[idx]['label']
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
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = torch.stack(labels)
    return batch

@torch.no_grad()
def evaluate(model, device, test_dataloader, step):
    losses = []  # List of scalar tensors
    correct = 0
    total = 0
    for batch in tqdm(test_dataloader):
        # adapt batch to model
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        target = torch.argmax(batch['labels'], dim=1)
        correct += torch.sum(preds==target).item()
        total += target.size(0)
        losses.append(outputs.loss)
    stacked_losses = torch.stack(losses)  # (num_batches, ) 
    total_avg_loss = stacked_losses.mean()  # (num test examples, ) -> scalar
    total_avg_acc = (100 * correct) / total
    logger.info(f"Eval at step {step}")
    logger.info(f"Correct: {correct} / Total: {total}")
    logger.info(f"Average val loss: {total_avg_loss.item()}")
    logger.info(f"Average val acc: {total_avg_acc}")
    return total_avg_loss.item(), total_avg_acc


if __name__ == "__main__":

    cache_dir='./cache'
    checkpoint_dir = './checkpoints'
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logging.basicConfig(filename="log.txt", 
                        level=logging.INFO,
					    format='%(asctime)s %(message)s', 
					    filemode='w') 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #datasets = load_dataset('Maysee/tiny-imagenet', cache_dir=cache_dir)
    datasets = load_dataset('cifar10', cache_dir=cache_dir)
    label_list = datasets["train"].features["label"].names
    num_labels = len(label_list)

    config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir=cache_dir)
    config.id2label = {i: label for i, label in enumerate(label_list)}
    config.label2id = {label: i for i, label in enumerate(label_list)}
    config.num_labels = num_labels

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    train_dataset = ImageDataset(image_files=datasets["train"], text="", processor=processor, num_labels=num_labels)
    test_dataset = ImageDataset(image_files=datasets["test"], text="", processor=processor, num_labels=num_labels)

    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm", config=config)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'classifier' in name or 'pooler' in name:
            param.requires_grad = True

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=512, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    best_test_acc = -1
    step = -1
    patience = 0
    model.train()
    for epoch in range(50):  # loop over the dataset multiple times
        print(f"Epoch: {epoch}")
        for batch in tqdm(train_dataloader):
            step += 1
            # get the inputs; 
            batch = {k:v.to(device) for k,v in batch.items()}
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            target = torch.argmax(batch['labels'], dim=1)
            correct = torch.sum(preds==target).item()
            acc = (correct * 100) / target.size(0)
            loss = outputs.loss
            logger.info(f"Step {step} - loss: {loss.item()} , train acc: {acc}")
            loss.backward()
            optimizer.step()

            # Evaluate
            if step != 0 and step % 20 == 0:
                model.eval()
                logger.info(f"Evaluate step: {step}")
                new_test_loss, new_test_acc = evaluate(model, device, test_dataloader, step)
                # save checkpoint with best test loss
                if new_test_acc > best_test_acc or best_test_acc < 0:
                    patience = 0
                    if best_test_acc > 0:
                        os.remove(checkpoint_dir + '/'+ best_checkpoint_filename)
                    best_checkpoint_filename = "best_model" + str(step) +".pt"
                    torch.save(model.state_dict(), checkpoint_dir + '/' + best_checkpoint_filename)
                    best_test_acc = new_test_acc
                    logger.info(f"Best test acc: {best_test_acc}")
                    logger.info(f"Best model saved at {checkpoint_dir + '/' + best_checkpoint_filename}")
                else:
                    patience += 1
                    if patience > 3:
                        logger.info(f"Early stopping at step {step}")
                        break

                model.train()
    logger.info(f"Best model acc {best_test_acc}")
    logger.shutdown()