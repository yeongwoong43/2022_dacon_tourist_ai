# %%
import numpy as np
import pandas as pd
import os, re
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import random
import datetime
import time
from itertools import combinations

from transformers import *
import torch
import torch.nn.init
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# %%
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

# %%
def train_model(args):
  '''
  [1. 전처리 과정에서 생성된 데이터 호출]
  [2. 텐서 생성]
  '''
  set_seed(args)

  train_data = pd.read_csv('../data/train.csv')
  encoder = LabelEncoder()
  encoder.fit(train_data['cat3'].unique())

  N = train_data.shape[0]
  MAX_LEN = 512
  
  train_idx, valid_idx = train_test_split(np.arange(N))

  # training
  N = train_idx.shape[0]
  overview = train_data.loc[train_idx, 'overview'].values
  cat3 = encoder.transform(train_data.loc[train_idx, 'cat3'].values)

  input_ids = np.zeros((N, MAX_LEN), dtype=int)
  attention_masks = np.zeros((N, MAX_LEN), dtype=int)
  labels = np.zeros((N), dtype=int)

  tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)

  for i in tqdm(range(N), position=0, leave=True):
    try:
      cur_ov = str(overview[i])
      encoded_input = tokenizer(cur_ov, return_tensors='pt', max_length=512,
                                padding='max_length', truncation=True)
      input_ids[i, ] = encoded_input['input_ids']
      attention_masks[i, ] = encoded_input['attention_mask']
      labels[i] = cat3[i]
    except Exception as e:
      print(e)
      pass
  
  # validating
  N = valid_idx.shape[0]

  overview = train_data.loc[valid_idx, 'overview'].values
  cat3 = encoder.transform(train_data.loc[valid_idx, 'overview'].values)

  valid_input_ids = np.zeros((N, MAX_LEN), dtype=int)
  valid_attention_masks = np.zeros((N, MAX_LEN), dtype=int)
  valid_labels = np.zeros((N), dtype=int)

  for i in tqdm(range(N), position=0, leave=True):
    try:
        cur_ov = str(overview[i])
        encoded_input = tokenizer(cur_ov, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        valid_input_ids[i, ] = encoded_input['input_ids']
        valid_attention_masks[i, ] = encoded_input['attention_mask']
        valid_labels[i] = cat3[i]
    except Exception as e:
        print(e)
        pass

    if os.path.exists(args.dir_path):
        os.makedirs(args.dir_path, exist_ok=True)
    
    print("\n\nMake tensor\n\n")
    input_ids = torch.tensor(input_ids, dtype=int)
    attention_masks = torch.tensor(attention_masks, dtype=int)
    labels= torch.tensor(labels, dtype=int)

    valid_input_ids = torch.tensor(valid_input_ids, dtype=int)
    valid_attention_masks = torch.tensor(valid_attention_masks, dtype=int)
    valid_labels = torch.tensor(valid_labels, dtype=int)


    if args.save_tensor == True:
        torch.save(input_ids, "./data/"+args.dir_path+"/train_input_ids_1012.pt")
        torch.save(attention_masks, "./data/"+args.dir_path+"train_attention_masks_1012.pt")
        torch.save(labels, "./data/"+args.dir_path+"train_labels_1012.pt")

        torch.save(valid_input_ids, "./data/"+args.dir_path+"/valid_input_ids_1012.pt")
        torch.save(attention_masks, "./data/"+args.dir_path+"/valid_attention_masks_1012.pt")
        torch.save(valid_labels, "./data/"+args.dir_path+"valid_labels_1012.pt")
    


    # Setup training
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat==labels_flat)/len(labels/flat)
    
    def format_time(elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))
    
    train_data = TensorDataset(input_ids, attention_masks, labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    validation_data = TensorDataset(valid_input_ids, valid_attention_masks, valid_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path)
    model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-5)

    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    device = torch.device("cuda")
    loss_f = nn.CrossEntropyLoss()


    # Train
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    model.zero_grad()
    for i in range(args.epochs):
        print("")
        print('======= Epoch {:} / {:} ======'.format(i+1, args.epochs))
        print('Training...')
        t0 = time.time()
        train_loss, train_accuracy = 0, 0
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), desc="Iteration", smoothing=0.05):
            if step%10000 == 0 and not step == 0:
                elapsed = format_time(time.time() -t0)
                print('\tBatch {:>5} of {:>5}. \tElapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                print('\tcurrent average loss = {}'.format(train_loss/step))

                batch = tuple(t.to(device) for t t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                outputs = model(b_input_ids, attention_masks=b_input_mask, labels=b_labels)
                loss = outputs[0]
                logits = outputs[1]
                train_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.detach().cpu().numpy()
                train_accuracy += flat_accuracy(logits, label_ids)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
        
        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_accuracy = train_accuracy / len(train_dataloader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        print("\tAverage training loss: {0:.8f}",format(avg_train_loss))
        print("\tAverage training accuracy: {0:.8f}".format(avg_train_accuracy))

        print("")
        print("Validating...")
        t0 = time.time()
        model.eval()
        val_loss, val_accuracy = 0, 0
        for step, batch in tqdm(enumerate(validation_dataloader), desc="Iteration", smoothing=0.05):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, attention_masks=b_input_mask)
            
            logits = outputs[0]
            logits = logits.detach().cpu()
            labels_ids = b_labels.detach().cpu()

            logits = logits.numpy()
            label_ids = label_ids.numpy()
            val_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = val_accuracy/len(validation_dataloader)
        avg_val_loss = val_loss/len(validation_dataloader)
        val_accuracies.append(avg_val_accuracy)
        val_losses.append(avg_val_loss)
        print("\tAverage validation loss: {0:.8f}".format(avg_val_loss))
        print("\tAverage validation accuracy: {0:.8f}".format(avg_val_accuracy))
        print("\tTraining epoch took: {:}".format(format_time(time.time()-t0)))

        # if np.min(val_losses) == val_losses[-1]:
        print("saving current best checkpoint")
        torch.save(model.state_dict(), "./data/{}/{}th_1012.pt".format(args.dir_path, i+1))
        

# %%
def inference_model(args):
    test_data = pd.read_csv('../data/test.csv')

    overview = test_data['overview'].values

    N = test_data.shape[0]
    MAX_LEN = 512

    test_input_ids = np.zeros((N, MAX_LEN), dtype=int)
    test_attention_masks = np.zeros((N, MAX_LEN), dtype=int)

    tokenizer = AutoTokenizer.form_pretrained(args.checkpoint_path)
    tokenizer.truncation_side = "right"

    for i in tqdm(range(N), position=0, leave=True):
        try:
            cur_ov = str(overview[i])
            encoded_input = tokenizer(cur_ov, return_tensors='pt', max_length=MAX_LEN, padding='max_length', truncation=True)
            test_input_ids[i, ] = encoded_input['input_ids']
            test_attention_masks[i, ] = encoded_input['attention_mask']
        
        except Exception as e:
            print(e)
            pass
    
    test_input_ids = torch.tensor(test_input_ids, dtype=int)
    test_attention_masks = torch.tensor(test_attention_masks, dtype=int)

    if args.save_tensor == True:
        torch.save(test_input_ids, "./data/{}/test_input_ids_1012.pt".format(args.dir_path))
        torch.save(test_attention_masks, "./data/{}/test_attention_masks_1012.pt".format(args.dir_path))
    
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path)
    PATH = "몰루?"

    model.load_state_dict(torch.load(PATH))
    model.cuda()

    test_tensor = TensorDataset(test_input_ids, test_attention_masks)
    test_sampler = SequentialSampler(test_tensor)
    test_dataloader = DataLoader(test_tensor, sampler=test_sampler, batch_size=args.test_batch_size)

    submission = pd.read_csv("../data/sample_submission.csv")
    device = torch.device("cuda")

    preds = np.array([])
    for step, batch in tqdm(enumerate(test_dataloader), desc="Iteration", smoothing=0.05):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
        

        logits = outputs[0]
        logits = logits.detach().cpu()
        _pred = logits.numpy()
        pred = np.argmax(_pred, axis=1).flatten()
        preds = np.append(preds, pred)
    
    submission['cat3'] = preds
    submission.to_csv('../data/submission_1013.csv', index=False)

# %%
def model_ensemble():
    submission = pd.read_csv('../data/sample_submission.csv')

    submission_1 = pd.read_csv(PATH1)
    submission_2 = pd.read_csv(PATH2)
    submission_3 = pd.read_csv(PATH3)

    sub_1 = submission_1['cat3']
    sub_2 = submission_2['cat3']
    sub_3 = submission_3['cat3']

    ensemble_preds = (sub_1+sub_2+sub_3)/3

    preds = np.where(ensemble_preds>0.5, 1, 0)

    submission['cat3'] =preds

    submission.to_csv('../data/submission_ensemble_1013.csv', index=False)

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set arguments.")

    parser.add_argument("--seed", default="42", type=int, help="Random seed for initialization")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--eps", default=1e-5, type=float, help="The initial eps.")
    parser.add_argument("--epochs", default=3, type=int, help="Total number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=None, help="batch_size")
    parser.add_argument("--test_batch_size", type=int, default=None, help="test_batch_size")

    parser.add_argument("--no_cuda", default=False, type=bool, help="Say True if you don't want to use cuda.")
    parser.add_argument("--ensemble", default=False, type=bool, help="Ensemble.")
    parser.add_argument("--save_tensor", default=True, type=str, help="Save tensor.")
    parser.add_argument("--mode", default="train", type=str, help="When you train the model.")
    parser.add_argument("--dir_path", default="klue/roberta-large", type=str, help="Save model path.")
    parser.add_argument("--model_name", default="klue_roberta_large", type=str, help="Model name.")
    parser.add_argument("--process_name", default="tourist_ai", type=str, help="process_name.")
    parser.add_argument("--checkpoint_path", default="klue/roberta-large", type=str, help="Pre-trained Language Model.")

    args = parser.parse_args()

    if args.mode == "train":
        data_preprocess(args) # 수정 필요
        train_model(args)
    else:
        inference_model(args)

    if args.ensemble == True:
        model_ensemble()


