import pandas as pd
import numpy as np
import os
import torchtext
import torch
from torch import nn
import json, re, unicodedata, string, typing, time
import torch.nn.functional as F
import spacy
import argparse
from collections import Counter
import pickle
from nltk import word_tokenize

nlp = spacy.load('en')
from preprocess import *
from metrics import *
from models import *
from torchtext import vocab

ap = argparse.ArgumentParser()
ap.add_argument('-en', '--emb_name', default='6B', help='GloVe embedding name')
ap.add_argument('-ed', '--emb_dim', type=int, default=100, help='Word embedding dimension')
ap.add_argument('-hd', '--hidden_dim', type=int, default=128, help='LSTM hidden dimension')
ap.add_argument('-d', '--dropout', type=float, default=0.3, help='Dropout rate')
ap.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
ap.add_argument('-m', '--margin', type=float, default=1, help='Margin for loss function')
ap.add_argument('-l', '--layers', type=int, default=2, help='number of layers')
ap.add_argument('-e', '--epochs', type=int, default=200, help='Number of epochs')
ap.add_argument('--working_dir', default='train', help='Working directory for checkpoints and logs')
ap.add_argument('--glove_cache', default='glove_cache', help='Word embeddings cache directory')
ap.add_argument('--random_seed', type=int, default=12345, help='Random seed')
ap.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning Rate')
args = ap.parse_args()

train_df = pd.read_pickle('drqatrain.pkl')
valid_df = pd.read_pickle('drqavalid.pkl')
vocab_text = gather_text_for_vocab([train_df, valid_df])
print("Number of sentences in dataset: ", len(vocab_text))
word2idx, idx2word, word_vocab = build_word_vocab(vocab_text)

train_dataset = SquadDataset(train_df[:80000], args.batch_size)
valid_dataset = SquadDataset(valid_df[:30000], args.batch_size)
a = next(iter(train_dataset))


def create_glove_matrix():
    '''
    Parses the glove word vectors text file and returns a dictionary with the words as
    keys and their respective pretrained word vectors as values.

    '''
    glove_dict = {}
    name = args.emb_name
    dim = args.emb_dim
    cache = args.glove_cache
    glove = vocab.GloVe(name=name, dim=dim, cache=cache)
    with open("./glove_cache/glove.%s.%sd.txt" % (name, dim), "r", encoding="utf-8") as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            glove_dict[word] = vector

    f.close()

    return glove_dict


glove_dict = create_glove_matrix()


def create_word_embedding(glove_dict):
    '''
    Creates a weight matrix of the words that are common in the GloVe vocab and
    the dataset's vocab. Initializes OOV words with a zero vector.
    '''
    weights_matrix = np.zeros((len(word_vocab), 100))
    words_found = 0
    for i, word in enumerate(word_vocab):
        try:
            weights_matrix[i] = glove_dict[word]
            words_found += 1
        except:
            pass
    return weights_matrix, words_found


weights_matrix, words_found = create_word_embedding(glove_dict)
print("Total words found in glove vocab: ", words_found)
np.save('drqaglove_vt.npy', weights_matrix)


def weighted_average(x, weights):
    # x = [bs, len, dim]
    # weights = [bs, len]

    weights = weights.unsqueeze(1)
    # weights = [bs, 1, len]

    w = weights.bmm(x).squeeze(1)
    # w = [bs, 1, dim] => [bs, dim]

    return w


def count_parameters(model):
    '''Returns the number of trainable parameters in the model.'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_dataset):
    '''
    Trains the model.
    '''

    print("Starting training ........")

    train_loss = 0.
    batch_count = 0

    # put the model in training mode
    model.train()

    # iterate through training data
    for batch in train_dataset:

        if batch_count % 500 == 0:
            print(f"Starting batch: {batch_count}")
        batch_count += 1

        context, question, context_mask, question_mask, label, ctx, ans, ids = batch

        # place the tensors on GPU
        context, context_mask, question, question_mask, label = context.to(device), context_mask.to(device), \
                                                                question.to(device), question_mask.to(device), label.to(
            device)

        # forward pass, get the predictions
        preds = model(context, question, context_mask, question_mask)

        start_pred, end_pred = preds

        # separate labels for start and end position
        start_label, end_label = label[:, 0], label[:, 1]

        # calculate loss
        loss = F.cross_entropy(start_pred, start_label) + F.cross_entropy(end_pred, end_label)

        # backward pass, calculates the gradients
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        # update the gradients
        optimizer.step()

        # zero the gradients to prevent them from accumulating
        optimizer.zero_grad()

        train_loss += loss.item()

    return train_loss / len(train_dataset)


def valid(model, valid_dataset):
    '''
    Performs validation.
    '''

    print("Starting validation .........")

    valid_loss = 0.

    batch_count = 0

    f1, em = 0., 0.

    # puts the model in eval mode. Turns off dropout
    model.eval()

    predictions = {}

    for batch in valid_dataset:

        if batch_count % 500 == 0:
            print(f"Starting batch {batch_count}")
        batch_count += 1

        context, question, context_mask, question_mask, label, context_text, answers, ids = batch

        context, context_mask, question, question_mask, label = context.to(device), context_mask.to(device), \
                                                                question.to(device), question_mask.to(device), label.to(
            device)

        with torch.no_grad():

            preds = model(context, question, context_mask, question_mask)

            p1, p2 = preds

            y1, y2 = label[:, 0], label[:, 1]

            loss = F.cross_entropy(p1, y1) + F.cross_entropy(p2, y2)

            valid_loss += loss.item()

            # get the start and end index positions from the model preds

            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1,
                                                                                                      -1)

            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

            # stack predictions
            for i in range(batch_size):
                id = ids[i]
                pred = context[i][s_idx[i]:e_idx[i] + 1]
                pred = ' '.join([idx2word[idx.item()] for idx in pred])
                predictions[id] = pred

    em, f1 = evaluate(predictions)
    return valid_loss / len(valid_dataset), em, f1


def evaluate(predictions):
    '''
    Gets a dictionary of predictions with question_id as key
    and prediction as value. The validation dataset has multiple
    answers for a single question. Hence we compare our prediction
    with all the answers and choose the one that gives us
    the maximum metric (em or f1).
    This method first parses the JSON file, gets all the answers
    for a given id and then passes the list of answers and the
    predictions to calculate em, f1.


    :param dict predictions
    Returns
    : exact_match: 1 if the prediction and ground truth
      match exactly, 0 otherwise.
    : f1_score:
    '''
    with open('./data/squad_dev.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    dataset = dataset['data']
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    continue

                ground_truths = list(map(lambda x: x['text'], qa['answers']))

                prediction = predictions[qa['id']]

                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)

                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return exact_match, f1


def load_pretrained_model(curr_model, curr_optimizer, pretrained_path):
    try:
        model_dict = curr_model.module.state_dict()
        optim_dict = curr_optimizer.state_dict()
        load_pretrained = torch.load(pretrained_path)
        pretrained_model_specs = load_pretrained['model_state_dict']
        pretrained_model_specs = {k: v for k, v in pretrained_model_specs.items() if k in model_dict}
        pretrained_optim_specs = load_pretrained['optimizer_state_dict']
        pretrained_optim_specs = {k: v for k, v in pretrained_optim_specs.items() if k in optim_dict}

        # update & load
        model_dict.update(pretrained_model_specs)
        optim_dict.update(pretrained_optim_specs)
        curr_model.module.load_state_dict(model_dict)
        curr_optimizer.load_state_dict(optim_dict)
        print(f"the model loaded successfully")
    except:
        print(f"the pretrained model doesn't exist or it failed to load")
    return curr_model, curr_optimizer


if __name__ == '__main__':
    device = torch.device('cuda')
    # HIDDEN_DIM = 128
    # EMB_DIM = 100
    # NUM_LAYERS = 3
    NUM_DIRECTIONS = 2
    # DROPOUT = 0.3
    device = torch.device('cuda')
    model = DocumentReader(args.hidden_dim,
                           args.emb_dim,
                           args.layers,
                           NUM_DIRECTIONS,
                           args.dropout,
                           device).to(device)
    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    ckpt_dir_name = "%s_%s" % (args.working_dir, args.batch_size)
    model, optimizer = load_pretrained_model(model, optimizer,
                                             "%s/ckpt/%s" % (ckpt_dir_name, "best_weights.pt"))

    print(args)
    ckpt_dir = os.path.join(ckpt_dir_name, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        metrics = pickle.load(open(os.path.join(ckpt_dir, 'metrics.p'), 'rb'))
        print("load metric files successfully!")
    except:
        metrics = {"train_losses": [], "valid_losses": [], "ems": [], "f1s": []}
        print("no metric files exist!")
    valid_loss_prev = 10000
    print("Start training")
    epochs = args.epochs
    lives = 20
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")

        start_time = time.time()
        train_loss = train(model, train_dataset)
        valid_loss, em, f1 = valid(model, valid_dataset)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        metrics['train_losses'].append(train_loss)
        metrics['valid_losses'].append(valid_loss)
        metrics['ems'].append(em)
        metrics['f1s'].append(f1)

        if valid_loss < valid_loss_prev:
            state = {'epoch': epoch, 'model_state_dict': model.module.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()}
            # fname = os.path.join(ckpt_dir, 'weights_{:03d}.pt'.format(epoch))
            # torch.save(state, fname)
            fname = os.path.join(ckpt_dir, 'best_weights.pt'.format(epoch))
            torch.save(state, fname)
        else:
            lives -= 1
            if lives == 0:
                break
        valid_loss_prev = valid_loss
        pickle.dump(metrics, open(os.path.join(ckpt_dir, 'metrics.p'), 'wb'))
        print(f"Epoch train loss : {train_loss}| Time: {epoch_mins}m {epoch_secs}s")
        print(f"Epoch valid loss: {valid_loss}")
        print(f"Epoch EM: {em}")
        print(f"Epoch F1: {f1}")
        print("====================================================================================")