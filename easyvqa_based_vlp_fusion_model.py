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

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

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
        text_inputs = {k:v.to(device) for k,v in text_inputs.items()} #Moving these tensors to the GPU/CPU so they can be fed into BERT.
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

class EarlyFusionNetwork(nn.Module):

    def __init__(self, hyperparms=None):
        super(EarlyFusionNetwork, self).__init__()
        self.dropout = nn.Dropout(0.3) #30% of neurons randomly turned off during training for regularization
        self.vision_projection = nn.Linear(2048, 768) #fully connected layer that linearly projects the output tensor from the V backbone of size 2048 (for ViT) into tensor of size 768 as output from this layer)
        self.text_projection = nn.Linear(512, 768) #fully connected layer that linearly projects the output tensor from the L backbone of size 512 (for BERT) into tensor of size 768 as output from this layer)
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
    easy_vqa_path = "/content/drive/MyDrive/easy-VQA/easy_vqa/data"
    for q, a, img_id in zip(qs, ans, img_ids):
        if os.path.exists(easy_vqa_path):
          #img_path = f"/usr/local/lib/python3.7/dist-packages/easy_vqa/data/{type}/images/{img_id}.png"
          img_path = f"/content/drive/MyDrive/easy-VQA/easy_vqa/data/{type}/images/{img_id}.png"
          
          records.append({"question" : q, "answer": a, "image_path": img_path})
    df = pd.DataFrame(records)
    return df

def get_accuracy_score(preds, labels):
    return accuracy_score(labels, preds)

def evaluate(val_dataloader):
    model.eval()

    loss_val_total = 0
    predictions = []
    true_vals = []
    conf = []

    for batch in val_dataloader: #batch = {}

        batch = tuple(b.to(device) for b in batch.values()) #moving the batch of data to the GPU/CPU to feed the VLP fusion model

        inputs = {'img_embedding':batch[0], 'text_embedding':batch[1]}

        with torch.no_grad():
            outputs = model(**inputs)
            '''
            outputs = model(
                        img_embedding=batch[0],
                        text_embedding=batch[1]
                      )
            '''
        labels = batch[2]
        loss = criterion(outputs.view(-1, 13), labels.view(-1))
        loss_val_total += loss.item()

        probs = torch.max(outputs.softmax(dim=1), dim=-1)[0].detach().cpu().numpy()
        outputs = outputs.argmax(-1)
        logits = outputs.detach().cpu().numpy()
        label_ids = labels.cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
        conf.append(probs)

    loss_val_avg = loss_val_total/len(dataloader_val)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    conf = np.concatenate(confidence, axis=0)

    return loss_val_avg, predictions, true_vals, conf

def train():
    train_history = open("/content/models/train_history.csv", "w")
    log_hdr  = "Epoch, train_loss, train_acc, val_loss, val_acc"
    train_history.write(log_hdr  + "\n")
    train_f1s = []
    val_f1s = []
    train_losses = []
    val_losses = []
    min_val_loss = -1
    max_auc_score = 0
    epochs_no_improve = 0
    early_stopping_epoch = 3
    early_stop = False

    for epoch in tqdm(range(1, epochs+1)):

        model.train()
        loss_train_total = 0
        train_predictions, train_true_vals = [], []

        progress_bar = tqdm(train_dataloader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)

        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch.values())

            inputs = {'img_embedding':  batch[0],'text_embedding': batch[1]}
            labels =  batch[2]

            outputs = model(**inputs)
            '''
            outputs = model(
                        img_embedding=batch[0],
                        text_embedding=batch[1]
                      )
            '''
            loss = criterion(outputs.view(-1, 13), labels.view(-1))
            loss_train_total += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            logits = outputs.argmax(-1)
            logits = logits.detach().cpu().numpy()
            label_ids = labels.cpu().numpy()
            train_predictions.append(logits)
            train_true_vals.append(label_ids)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

        train_predictions = np.concatenate(train_predictions, axis=0)
        train_true_vals = np.concatenate(train_true_vals, axis=0)

        tqdm.write(f'\nEpoch {epoch}')
        loss_train_avg = loss_train_total/len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')
        train_f1 = get_accuracy_score(train_predictions, train_true_vals)
        tqdm.write(f'Train Acc: {train_f1}')

        val_loss, predictions, true_vals,_ = evaluate(dataloader_validation)
        val_f1 = get_accuracy_score(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'Val Acc: {val_f1}')

        if val_f1 >= max_auc_score:
            tqdm.write('\nSaving best model')
            torch.save(model.state_dict(), f'/content/models/easyvqa_finetuned_epoch_{epoch}.model')
            max_auc_score = val_f1

        train_losses.append(loss_train_avg)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        log_str  = "{}, {}, {}, {}, {}".format(epoch, loss_train_avg, train_f1, val_loss, val_f1)
        train_history.write(log_str + "\n")

        if min_val_loss < 0:
            min_val_loss = val_loss
        else:
          if val_loss < min_val_loss:
              min_val_loss = val_loss
          else:
              epochs_no_improve += 1
              if epochs_no_improve >= early_stopping_epoch:
                  early_stop = True
                  break
              else:
                  continue


    if early_stop:
      print("Early Stopping activated at epoch -", epoch )
      print("Use the checkpoint at epoch - ", epoch - early_stopping_epoch)

    train_history.close()
    return train_losses, val_losses

if __name__ == "__main__":
  '''Splitting the VQA dataframe'''
  # Getting textual questions and corresponding answers from Easy VQA dataset
  train_qs, train_ans, train_img_ids = get_train_questions()
  test_qs, test_ans, test_img_ids = get_test_questions()
  ans = get_answers()

  temp_df = create_pd_dataframe(train_qs, train_ans, train_img_ids, type="train")
  temp_df = temp_df.sample(frac=1) # shuffling trainign set
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

  text_encoder_bert.to(device)
  img_encoder_vit.to(device)
  #print()

  '''Creating train and validation sets for training the VLP fusion model after converting raw V + L data into respective feature vectors (embeddings) from the backbones'''
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

  print(train_dataset)

  batch_size = 32
  eval_batch_size = 32

  train_dataloader = DataLoader(
                        train_dataset,
                        sampler=RandomSampler(train_dataset),
                        batch_size=batch_size
                    )

  val_dataloader = DataLoader(
                            val_dataset,
                            sampler=SequentialSampler(val_dataset),
                            batch_size=eval_batch_size
                        )

  criterion = nn.CrossEntropyLoss() #using cross entropy loss since this is a classification problem

  torch.cuda.empty_cache()
  model = EarlyFusionNetwork()
  model.to(device)

  model = MidFusionNetwork()
  model.to(device)

  optimizer = AdamW(model.parameters(),
                  lr=5e-5,
                  weight_decay = 1e-5,
                  eps=1e-8
                  )

  epochs = 10
  train_steps=20000
  print("train_steps", train_steps)
  warm_steps = train_steps * 0.1
  print("warm_steps", warm_steps)
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=warm_steps,
                                              num_training_steps=train_steps)
  try:
    !rm -rf /content/models
    !mkdir /content/models
    train_losses, val_losses =  train()
    torch.cuda.empty_cache()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
  except Exception as e:
    print(f"Training loop interrupted! Due to an error: {e}")

  test_dataset = VQADatasetToEmbeddings(
                    df=test_df,
                    tokenizer=tokenize_text_preprocessor,
                    img_preprocessor=img_preprocessor,
                    text_encoder=text_encoder_bert,
                    img_encoder=img_encoder_vit
                )

  device = "cuda:0"
  model.load_state_dict(torch.load('/content/models/easyvqa_finetuned_epoch_9.model'))
  model.to(device)

  dataloader_test = DataLoader(
                        test_dataset,
                        sampler=SequentialSampler(test_dataset),
                        batch_size=128
                    )

_, preds, truths, confidence = evaluate(dataloader_test)

print("Test Acc with ViT: " , get_accuracy_score(preds,truths))
test_results_df = pd.concat([test_df, pd.DataFrame(preds, columns=["preds"]), pd.DataFrame(truths, columns=["gt"]), pd.DataFrame(confidence, columns=["confidence"])], axis=1)
