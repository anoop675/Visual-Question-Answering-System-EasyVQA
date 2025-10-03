'''
Implementation of a simple VQA System leveraging a VLP Fusion model with pre-trained ViT and BERT as feature backbones.
Dataset used is EasyVQA
The Vision (V) Backbone used in this VLP is ViT (for modern VLP fusion models) which extract features from images and produce embeddings capturing the high level image features, CNNs were used in legacy VLP fusion models
The Language (L) Backbone used in this VLP is BERT (for modern VLP fusion models) which extract features from sentences and produce embeddings capturing the semantics, RNNs were used in legacy VLP fusion models

Vision-Language Pre-training models are AI systems that learn from both visual (images, video) and textual data simultaneously.
They are a foundational step in multimodal AI, designed to understand and relate information across different data types.
'''
#installing VQA library from PyPI & sentence-transformers library from Hugging Face that is used to create vector embeddings from sentences
!pip install -qqq easy-vqa
!pip install -qqq sentence_transformers transformers timm

#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)

import os
import math
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from easy_vqa import get_train_questions, get_test_questions, get_answers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import timm #importing PyTorch's Image Models for many pre-trained CNNs like ResNet, EfficientNet, etc
import torchvision.transforms as T #for image preprocessing tasks
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor #importing HuggingFace's transformer library fro pre-trained transfomers like BERT, ViT, etc
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import get_linear_schedule_with_warmup
import requests
from torch.optim import AdamW
from zipfile import ZipFile
import traceback

# Put this in your notebook/script and run once before training
import torch
from PIL import Image
from tqdm import tqdm

#batch computing multimodal embeddings before fusion model training
def build_and_save_embeddings(df, tokenizer, img_preprocessor, text_encoder, img_encoder, out_path, device="cuda:0", batch_size=64):
    text_encoder.eval()
    img_encoder.eval()
    text_encoder.to(device)
    img_encoder.to(device)
    text_embeddings, img_embeddings, labels = [], [], []

    for i in tqdm(range(0, len(df), batch_size), desc=f"Building {out_path}"):
        batch = df.iloc[i:i+batch_size] #slicing batch
        qs = batch["question"].tolist()
        imgs = [Image.open(this_path).convert("RGB") for this_path in batch["image_path"].tolist()]

        with torch.no_grad():
            tokenized_qs = tokenizer(qs, padding=True, truncation=True, return_tensors="pt")
            tokenized_qs.to(device)
            text_output = text_encoder(**tokenized_qs).pooler_output.cpu() # (64, 768)

            img_inputs = img_preprocessor(images=imgs, return_tensors="pt")
            img_inputs.to(device)
            img_output = img_encoder(**img_inputs).pooler_output.cpu()# (64, 768)

        text_embeddings.append(text_output)
        img_embeddings.append(img_output)
        labels.append(torch.tensor(batch["label"].values, dtype=torch.long))

    text_emb = torch.cat(text_embeddings, dim=0)
    img_emb = torch.cat(img_embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    torch.save({"text": text_emb, "img": img_emb, "labels": labels}, out_path)
    print("Saved:", out_path)

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        d = torch.load(path)
        self.text = d["text"]   # (N, 768)
        self.img  = d["img"]    # (N, 768)
        self.labels = d["labels"]
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {"text_embedding": self.text[idx], "image_embedding": self.img[idx], "label": self.labels[idx]}

"""
class VQADatasetToEmbeddings(Dataset):

    def __init__(self, df, tokenizer, img_preprocessor, text_encoder, img_encoder):
        self.df = df
        self.tokenizer = tokenizer
        self.img_preprocessor = img_preprocessor
        self.text_encoder = text_encoder
        self.img_encoder = img_encoder

    def __len__(self): #returns the number of records in the raw multimodal dataset (EasyVQA)
        return len(self.df)

    def __getitem__(self, idx): #this method fetches a record from the VQA dataset and converts each of the raw multimodal (text and image) data into vector embeddings using the respective feature extractors (backbones), to train the VLP fusion network
        encodings = {}
        #Converting textual data into vector embeddings using the L backbone
        question = self.df['question'][idx]
        text_inputs = self.tokenizer(question, return_tensors="pt") # string -> tokens -> { tensor(int ID values), tensor(attention masks)}
        #print(text_inputs)
        # Assuming 'device' is defined globally in __main__
        global device
        text_inputs = {k:v.to(device) for k,v in text_inputs.items()} #Moving these tensors to the GPU/CPU so they can be fed into BERT.
        #print(text_inputs.items())
        #text_outputs = self.text_encoder(**text_inputs)
        text_outputs = self.text_encoder(
                          input_ids=text_inputs['input_ids'],
                          attention_mask=text_inputs['attention_mask']
                      )
        text_embedding = text_outputs.pooler_output #Can also experiment with raw CLS embedding below
        #text_embedding = text_outputs.last_hidden_state[:,0,:] # Raw CLS embedding
        text_embedding = text_embedding.view(-1)
        text_embedding = text_embedding.detach()
        # print("Text emb", text_embedding.shape)

        img_file = self.df["image_path"][idx]
        img = Image.open(img_file).convert("RGB") #fetches and loads image from local disk located at that path
        label = self.df['label'][idx]

        # Converting image data into vector embeddings using the V backbone (in modern VLP models)
        img_inputs = self.img_preprocessor(img, return_tensors="pt")
        img_inputs = {k:v.to(device) for k,v in img_inputs.items()}
        img_outputs = self.img_encoder(**img_inputs)
        '''
        img_outputs = self.img_encoder(
                          input_ids=img_inputs['input_ids'],
                          attention_mask=img_inputs['attention_mask']
                      )
        '''
        img_embedding = img_outputs.pooler_output
        img_embedding = img_embedding.view(-1)
        img_embedding = img_embedding.detach()
        # print(img_embedding.shape)

        # When CNNs are used for V backbone (in legacy VLP models)
        #img = resize_transform(img)
        #img_inputs = T.ToTensor()(img).unsqueeze_(0)
        #img_inputs = img_inputs.to(device)
        #img_outputs = self.img_encoder(img_inputs)
        #img_embedding = img_outputs[0]
        #img_embedding = img_embedding.detach()
        #print(img_embedding.shape)

        encodings["image_embedding"] = img_embedding
        encodings["text_embedding"] = text_embedding
        encodings["label"] = torch.tensor(label)

        return encodings
"""

class EarlyFusionNetwork(nn.Module):

    def __init__(self, hyperparms=None):
        super(EarlyFusionNetwork, self).__init__()
        self.dropout = nn.Dropout(0.3) #30% of neurons randomly turned off during training for regularization
        # ViT-base and BERT-base pooler_output size is 768.
        self.vision_projection = nn.Linear(768, 768) #fully connected layer that linearly projects the output tensor from the V backbone of size 768 (for ViT) into tensor of size 768 as output from this layer)
        self.text_projection = nn.Linear(768, 768) #fully connected layer that linearly projects the output tensor from the L backbone of size 768 (for BERT) into tensor of size 768 as output from this layer)
        self.fully_connected_layer1 = nn.Linear(768, 256)
        self.batch_normalization_layer1 = nn.BatchNorm1d(256)
        self.classifier = nn.Linear(256, 13)

        W = torch.Tensor(768, 768)
        self.W = nn.Parameter(W)

        self.relu_f = nn.ReLU()

        # initialize weight matrices using He/Kaiming to prevent gradients from vanishing during training
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, img_embedding, text_embedding):
        x1 = img_embedding
        x1 = torch.nn.functional.normalize(x1, p=2, dim=1)
        Xv = self.relu_f(self.vision_projection(x1))

        x2 = text_embedding
        x2 = torch.nn.functional.normalize(x2, p=2, dim=1)
        Xt = self.relu_f(self.text_projection(x2))

        Xvt = Xv * Xt #fusion step
        Xvt = self.relu_f(torch.mm(Xvt, self.W.t()))

        Xvt = self.fully_connected_layer1(Xvt)
        Xvt = self.batch_normalization_layer1(Xvt)
        Xvt = self.dropout(Xvt)
        Xvt = self.classifier(Xvt)

        return Xvt

class MidFusionNetwork(nn.Module):

    def __init__(self, hyperparms=None):
        super(MidFusionNetwork, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.fully_connected_layer1 = nn.Linear(768, 256)
        self.batch_normalization_layer1 = nn.BatchNorm1d(256)
        self.classifier = nn.Linear(256, 13)

        W = torch.Tensor(768, 768)
        self.W = nn.Parameter(W)

        self.relu_f = nn.ReLU()

        # initialize weight matrices uisng He/Kaiming init
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, img_embedding, text_embedding):
        x1 = img_embedding
        Xv = torch.nn.functional.normalize(x1, p=2, dim=1)

        x2 = text_embedding
        Xt = torch.nn.functional.normalize(x2, p=2, dim=1)

        Xvt = Xv * Xt #fusion step
        Xvt = self.relu_f(torch.mm(Xvt, self.W.t()))

        Xvt = self.fully_connected_layer1(Xvt)
        Xvt = self.batch_normalization_layer1(Xvt)
        Xvt = self.dropout(Xvt)
        Xvt = self.classifier(Xvt)

        return Xvt

def create_pd_dataframe(qs, ans, img_ids, type="train"): #function to convert EasyVQA dataset into a pandas dataframe.
    records = []
    #easy_vqa_path = "/content/drive/MyDrive/easy-VQA/easy_vqa/data"
    easy_vqa_zip_path = "easy-VQA.zip"
    dest_path = "easy-VQA"
    with ZipFile(easy_vqa_zip_path) as zf:
      zf.extractall(dest_path)

    for q, a, img_id in zip(qs, ans, img_ids):
        if os.path.exists(dest_path):
          #img_path = f"/usr/local/lib/python3.7/dist-packages/easy_vqa/data/{type}/images/{img_id}.png"
          #img_path = f"/content/drive/MyDrive/easy-VQA/easy_vqa/data/{type}/images/{img_id}.png"
          img_path = f"easy-VQA/easy-VQA/easy_vqa/data/{type}/images/{img_id}.png"

          records.append({"question" : q, "answer": a, "image_path": img_path})
    df = pd.DataFrame(records)
    return df

def get_accuracy_score(preds, labels):
    return accuracy_score(labels, preds)

def train():
    train_history = open("/content/models/train_history.csv", "w")
    log_header  = "Epoch, train_loss, train_acc, val_loss, val_acc"
    train_history.write(log_header  + "\n")
    train_f1_scores = []
    val_f1_scores = []
    train_losses = []
    val_losses = []
    min_val_loss = -1
    max_auc_score = 0
    epochs_no_improve = 0
    early_stopping_epoch = 3
    early_stop = False

    global epochs, model, train_dataloader, criterion, optimizer, scheduler, device

    for epoch in tqdm(range(1, epochs+1)):
        model.train() #switching model to training mode
        total_train_loss = 0
        train_predictions, train_true_vals = [], []

        progress_bar = tqdm(train_dataloader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)

        for batch in progress_bar:
            model.zero_grad() #refreshing model's gradients

            batch = {k: v.to(device) for k, v in batch.items()} #Unpacking dictionary using keys, then move to device

            inputs = {'img_embedding':  batch['image_embedding'],'text_embedding': batch['text_embedding']}
            labels =  batch['label']

            outputs = model(**inputs)
            '''
            outputs = model(
                        img_embedding=batch[0],
                        text_embedding=batch[1]
                      )
            '''
            loss = criterion(outputs.view(-1, 13), labels.view(-1))
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            logits = outputs.argmax(-1)
            logits = logits.detach().cpu().numpy()
            label_ids = labels.cpu().numpy()
            train_predictions.append(logits)
            train_true_vals.append(label_ids)

            optimizer.step()
            scheduler.step()

            #TODO: Early stopping functionality and make sure that accuracy above 50p

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(labels))})

        train_predictions = np.concatenate(train_predictions, axis=0)
        train_true_vals = np.concatenate(train_true_vals, axis=0)

        tqdm.write(f'\nEpoch {epoch}')
        avg_train_loss = total_train_loss / len(train_dataloader)
        tqdm.write(f'Training loss: {avg_train_loss}')
        train_f1_score = get_accuracy_score(train_predictions, train_true_vals)
        tqdm.write(f'Train Acc: {train_f1_score}')

        val_loss, predictions, true_vals, _ = evaluate(val_dataloader)
        val_f1_score = get_accuracy_score(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'Val Acc: {val_f1_score}')

        if val_f1_score >= max_auc_score:
            tqdm.write('\nSaving best model')
            torch.save(model.state_dict(), f'/content/models/best_model_fusion.model') #Save the model with the best performance
            max_auc_score = val_f1_score

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_f1_scores.append(train_f1_score)
        val_f1_scores.append(val_f1_score)
        log_str  = "{}, {}, {}, {}, {}".format(epoch, avg_train_loss, train_f1_score, val_loss, val_f1_score)
        train_history.write(log_str + "\n")

        if min_val_loss < 0:
            min_val_loss = val_loss
        else:
          if val_loss < min_val_loss:
              min_val_loss = val_loss
              epochs_no_improve = 0 #Resets counter on improvement
          else:
              epochs_no_improve += 1
              if epochs_no_improve >= early_stopping_epoch:
                  early_stop = True
                  break
              else:
                  continue


    if early_stop:
      print("Early Stopping activated at epoch -", epoch )
      print("Use the best checkpoint saved at /content/models/best_model_fusion.model")

    train_history.close()
    return train_losses, val_losses

def evaluate(val_dataloader):
    model.eval() #switching model to validation mode

    total_val_loss = 0.0
    predictions = []
    true_vals = []
    conf = []

    global criterion, device

    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        inputs = {'img_embedding': batch['image_embedding'], 'text_embedding': batch['text_embedding']}

        with torch.no_grad():
            outputs = model(**inputs)
            '''
            outputs = model(
                        img_embedding=batch[0],
                        text_embedding=batch[1]
                      )
            '''
        labels = batch['label']
        loss = criterion(outputs.view(-1, 13), labels.view(-1))
        total_val_loss += loss.item()

        probs = torch.max(outputs.softmax(dim=1), dim=-1)[0].detach().cpu().numpy()
        outputs = outputs.argmax(-1)
        logits = outputs.detach().cpu().numpy()
        label_ids = labels.cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
        conf.append(probs)

    loss_val_avg = total_val_loss / len(val_dataloader)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    conf = np.concatenate(conf, axis=0)

    return loss_val_avg, predictions, true_vals, conf


if __name__ == "__main__":

  '''Splitting the VQA dataframe'''
  # Getting textual questions and corresponding answers from Easy VQA dataset
  train_qs, train_ans, train_img_ids = get_train_questions()
  test_qs, test_ans, test_img_ids = get_test_questions()
  ans = get_answers()

  temp_df = create_pd_dataframe(train_qs, train_ans, train_img_ids, type="train")
  temp_df = temp_df.sample(frac=1) # shuffling entire training set
  train_df, val_df = train_test_split(temp_df, test_size=0.25) #75% train & 25% val
  test_df = create_pd_dataframe(test_qs, test_ans, test_img_ids, type="test")
  #print(train_df.shape)
  #print(val_df.shape)
  #print(test_df.shape)
  #print(len(ans))

  '''Encoding each answer string into integer indices (The fusion network must predict an integer index (label) and we map it to the answer string)'''
  ans_to_labels = {a : i for i, a in enumerate(ans)} # dictionary of unique labels (answers) mapping to integer indexes
  #print(label_to_idx)

  # inserting new column "label" containing the corresponding integer indices for the answer strings to each dataframe
  train_df["label"] = train_df["answer"].apply(lambda a: ans_to_labels.get(a))
  val_df["label"] = val_df["answer"].apply(lambda a: ans_to_labels.get(a))
  test_df["label"] = test_df["answer"].apply(lambda a: ans_to_labels.get(a))
  #print(train_df.head())
  #print(val_df.head())
  #print(test_df.head())

  '''Loading V + L feature extraction backbones'''
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  # The BERT and ViT models act as the frozen backbones (pre-trained text and image feature extractors) of the VLP fusion model
  tokenize_text_preprocessor = AutoTokenizer.from_pretrained("bert-base-uncased") #loads BERT's tokenizer to tokenize the question strings into many lowercased tokens and convert them into token IDs that BERT understands
  text_encoder_bert = AutoModel.from_pretrained("bert-base-uncased") #loads a pre-trained BERT model used for converting the token IDs and produces dense vector embeddings for each token
  for p in text_encoder_bert.parameters(): #We are leaving the parameters (tensors) of the BERT model unchanged (frozen) (to avoid re-training it when the entire fusion network learns)
      p.requires_grad = False # No need to compute gradients for each tensor during backpropagation


  img_preprocessor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k") #loads ViT’s feature extractor that normalizes & resizes images to 224×224 patches
  img_encoder_vit = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k") #loads the ViT model
  # img_encoder = timm.create_model("resnet50d", pretrained=True,  num_classes=0) #using ViT as the fusing transformer which is part of the backbone of this  VLP model instead of ResNet50 CNN (which was used as the image feature extractor in legacy VLP fusion models)
  # resize_transform = T.Resize((224, 224))
  for p in img_encoder_vit.parameters(): #We are leaving the parameters (tensors) of the ViT model unchanged (frozen) (to avoid re-training it when the entire fusion network learns)
      p.requires_grad = False

  if not os.path.exists("train_embeddings.pt"):
    build_and_save_embeddings(train_df, tokenize_text_preprocessor, img_preprocessor, text_encoder_bert, img_encoder_vit, "train_embeddings.pt", device=device)

  if not os.path.exists("val_embeddings.pt"):
    build_and_save_embeddings(val_df,   tokenize_text_preprocessor, img_preprocessor, text_encoder_bert, img_encoder_vit, "val_embeddings.pt", device=device)

  if not os.path.exists("test_embeddings.pt"):
    build_and_save_embeddings(test_df,  tokenize_text_preprocessor, img_preprocessor, text_encoder_bert, img_encoder_vit, "test_embeddings.pt", device=device)

  '''Creating train and validation sets for training the VLP fusion model after converting raw V + L data into respective feature vectors (embeddings) from the backbones
  train_df.reset_index(drop=True, inplace=True)
  val_df.reset_index(drop=True, inplace=True)

  train_dataset = VQADatasetToEmbeddings(
                      df=train_df,
                      tokenizer=tokenize_text_preprocessor,
                      img_preprocessor=img_preprocessor,
                      text_encoder=text_encoder_bert,
                      img_encoder=img_encoder_vit
                  )

  val_dataset = VQADatasetToEmbeddings(
                      df=val_df,
                      tokenizer=tokenize_text_preprocessor,
                      img_preprocessor=img_preprocessor,
                      text_encoder=text_encoder_bert,
                      img_encoder = img_encoder_vit
                  )
  test_dataset = VQADatasetToEmbeddings(
                      df=test_df,
                      tokenizer=tokenize_text_preprocessor,
                      img_preprocessor=img_preprocessor,
                      text_encoder=text_encoder_bert,
                      img_encoder=img_encoder_vit
                  )

  train_batch_size = 32
  val_batch_size = 32

  train_dataloader = DataLoader(
                        train_dataset,
                        sampler=RandomSampler(train_dataset),
                        batch_size=train_batch_size
                    )

  val_dataloader = DataLoader(
                            val_dataset,
                            sampler=SequentialSampler(val_dataset),
                            batch_size=val_batch_size
                        )'''

  train_dataset = EmbeddingDataset("train_embeddings.pt")
  val_dataset   = EmbeddingDataset("val_embeddings.pt")

  train_dataloader = DataLoader(
                        train_dataset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True
                    )

  val_dataloader   = DataLoader(
                        val_dataset,
                        batch_size=32,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True
                    )

  '''Defining the training loop for our early fusion and mid fusion networks'''
  try:
    criterion = nn.CrossEntropyLoss() #using cross entropy loss since this is a classification problem

    epochs = 10
    train_steps = len(train_dataloader) * epochs
    #print("train_steps", train_steps)
    warm_steps = train_steps * 0.1
    #print("warm_steps", warm_steps)

    torch.cuda.empty_cache() #refreshing the GPU cache

    #Training the Early Fusion network:
    print("\n--- Training Early Fusion Network ---")
    model = EarlyFusionNetwork()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay = 1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=train_steps)

    !rm -rf /content/models
    !mkdir /content/models
    train_losses, val_losses =  train()
    torch.cuda.empty_cache()

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Early Fusion Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    #Training the mid fusion network
    print("\n--- Training Mid Fusion Network ---")
    model = MidFusionNetwork()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay = 1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=train_steps)

    # !rm -rf /content/models     # Re-initialize models directory for new model training if needed, though unnecessary here as best_model_fusion.model is used as the best checkpoint
    # !mkdir /content/models
    train_losses, val_losses = train()
    torch.cuda.empty_cache()

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Mid Fusion Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
  
    # Testing the last trained model (Mid Fusion) with the best checkpoint saved
    print("\n--- Testing Mid Fusion Network ---")
    device = "cuda:0"

    model.load_state_dict(torch.load('/content/models/best_model_fusion.model'), map_location=device) # Load the best saved model from the last training run (Mid Fusion)
    model.to(device)

    dataloader_test = DataLoader(
                          test_dataset,
                          sampler=SequentialSampler(test_dataset),
                          batch_size=128
                      )

    _, preds, truths, confidence = evaluate(dataloader_test)

    print("Test Accuracy with ViT: " , get_accuracy_score(preds,truths))
    test_results_df = pd.concat([
                        test_df,
                        pd.DataFrame(preds, columns=["preds"]),
                        pd.DataFrame(truths, columns=["gt"]),
                        pd.DataFrame(confidence, columns=["confidence"]
                    )], axis=1)
  except Exception as e:
    print(f"\nError: {e}")
    print(traceback.format_exc())
