#!/usr/bin/env python
# coding: utf-8

# In[1]:


from io import open
import unicodedata
import string
import re
import random
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator
from collections import Counter 

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker


# In[18]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[2]:


def generate_batch(data_batch):
    '''
    Prepare English and French examples for batch-friendly modeling by appending
    BOS/EOS tokens to each, stacking the tensors, and filling trailing spaces of
    shorter sentences with the <pad> token. To be used as the collate_fn in the
    English-to-Turkish DataLoader.

    Input: 
    - data_batch, an iterable of (English, Turkish) tuples from the datasets 
      created above

    Outputs
    - en_batch: a (max length X batch size) tensor of English token IDs
    - tr_batch: a (max length X batch size) tensor of Turkish token IDs 
    '''
    en_batch, tr_batch = [], []
    for (en_item, tr_item) in data_batch:
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        tr_batch.append(torch.cat([torch.tensor([BOS_IDX]), tr_item, torch.tensor([EOS_IDX])], dim=0))

    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX, batch_first=False)
    tr_batch = pad_sequence(tr_batch, padding_value=PAD_IDX, batch_first=False)

    return en_batch, tr_batch


# In[3]:


import pickle
import os
datadir = os.getcwd()
# Değişkeni geri yükleme
with open(os.path.join(datadir, 'en_vocab.pkl'), 'rb') as f:
    en_vocab = pickle.load(f)
    
with open(os.path.join(datadir,'tr_vocab.pkl'), 'rb') as f:
    tr_vocab = pickle.load(f)
    
PAD_IDX = en_vocab['<pad>']
BOS_IDX = en_vocab['<bos>']
EOS_IDX = en_vocab['<eos>']

SPECIALS = ['<unk>', '<pad>', '<bos>', '<eos>']

for en_id, tr_id in zip(en_vocab.lookup_indices(SPECIALS), tr_vocab.lookup_indices(SPECIALS)):
  assert en_id == tr_id


# In[4]:


import pickle

# Değişkeni geri yükleme
with open(os.path.join(datadir,'train_iter.pkl'), 'rb') as f:
    train_iter = pickle.load(f)

with open(os.path.join(datadir, 'en_vocab.pkl'), 'rb') as f:
    en_vocab = pickle.load(f)    
    # Değişkeni geri yükleme
with open(os.path.join(datadir, 'tr_vocab.pkl'), 'rb') as f:
    tr_vocab = pickle.load(f)
    
with open(os.path.join(datadir, 'test_iter.pkl'), 'rb') as f:
    test_iter = pickle.load(f)
    
with open(os.path.join(datadir, 'valid_iter.pkl'), 'rb') as f:
    valid_iter = pickle.load(f)
    
    
for i, (en_id, tr_id) in enumerate(train_iter):
    print('English:', ' '.join([en_vocab.lookup_token(idx) for idx in en_id[:, 0]]))
    print('Turkish:', ' '.join([tr_vocab.lookup_token(idx) for idx in tr_id[:, 0]]))
    if i == 4: 
        break
    else:
        print()


# In[20]:


class BahdanauAttentionQKV(nn.Module):
    def __init__(self, hidden_size, query_size=None, key_size=None, dropout_p=0.15):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_size = hidden_size if query_size is None else query_size
        
        # assume bidirectional encoder, but can specify otherwise
        self.key_size = 2*hidden_size if key_size is None else key_size
        
        self.query_layer = nn.Linear(self.query_size, hidden_size)
        self.key_layer = nn.Linear(self.key_size, hidden_size)
        self.energy_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, hidden, encoder_outputs, src_mask=None):
        '''
        Calculate attention weights using query and key features, with
        an optional mask for input sequences.
        
        Inputs:
          - hidden: most recent RNN hidden state; (B, Dec)
          - encoder_outputs: RNN outputs at individual time steps with 
            directions concatenated; (L, B, 2*Enc)
          - src_mask: boolean tensor of same size as source tokens (Src, B) 
            where False denotes tokens to be ignored
            
        Outputs:
          - attention weights: (B, src) tensor of softmax attention 
            weights to be applied to downstream values
        '''
        
        # (B, H)
        query_out = self.query_layer(hidden) 
        
        # (Src, B, 2*H) --> (Src, B, H)
        key_out = self.key_layer(encoder_outputs) 
        
        # (B, H) + (Src, B, H) = (Src, B, H)
        energy_input = torch.tanh(query_out + key_out) 
        
        # (Src, B, H) --> (Src, B, 1) --> (Src, B)
        energies = self.energy_layer(energy_input).squeeze(2) 
        
        # if a mask is provided, remove masked tokens from softmax calc
        if src_mask is not None:
            energies.data.masked_fill_(src_mask == 0, float("-inf"))
        
        # softmax over the length dimension
        weights = F.softmax(energies, dim=0)
        
        # return as (B, Src) as expected by later multiplication
        return weights.transpose(0, 1)


# In[21]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_p=0.1, max_len=100):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_p)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_attention_heads, 
                 num_encoder_layers, num_decoder_layers, dim_feedforward, 
                 max_seq_length, pos_dropout, transformer_dropout):
        super().__init__()
        self.d_model = d_model
        self.embed_src = nn.Embedding(input_dim, d_model)
        self.embed_tgt = nn.Embedding(output_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)
        
        self.transformer = nn.Transformer(d_model, num_attention_heads, num_encoder_layers, 
                                          num_decoder_layers, dim_feedforward, transformer_dropout)
        self.output = nn.Linear(d_model, output_dim)
        
    def forward(self,
                src=None, 
                tgt=None,
                src_mask=None,
                tgt_mask=None, 
                src_key_padding_mask=None, 
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                src_embeds=None, 
                tgt_embeds=None):
        
        if (src_embeds is None) and (src is not None):
            if (tgt_embeds is None) and (tgt is not None):
                src_embeds, tgt_embeds = self._embed_tokens(src, tgt)
        elif (src_embeds is not None) and (src is not None):
            raise ValueError("Must specify exactly one of src and src_embeds")
        elif (src_embeds is None) and (src is None):
            raise ValueError("Must specify exactly one of src and src_embeds")
        elif (tgt_embeds is not None) and (tgt is not None):
            raise ValueError("Must specify exactly one of tgt and tgt_embeds")
        elif (tgt_embeds is None) and (tgt is None):
            raise ValueError("Must specify exactly one of tgt and tgt_embeds")
        
        output = self.transformer(src_embeds, 
                                  tgt_embeds, 
                                  tgt_mask=tgt_mask, 
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        
        return self.output(output)
    
    def _embed_tokens(self, src, tgt):
        src_embeds = self.embed_src(src) * np.sqrt(self.d_model)
        tgt_embeds = self.embed_tgt(tgt) * np.sqrt(self.d_model)
        
        src_embeds = self.pos_enc(src_embeds)
        tgt_embeds = self.pos_enc(tgt_embeds)
        return src_embeds, tgt_embeds


# In[22]:


def train_transformer(model, iterator, optimizer, loss_fn, device, clip=None):
    model.train()
        
    epoch_loss = 0
    with tqdm(total=len(iterator), leave=False) as t:
        for i, (src, tgt) in enumerate(iterator):
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)
            tgt_inp, tgt_out = tgt[:-1, :], tgt[1:, :]

            tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_inp.size(0)).to(device)
            src_key_padding_mask = (src == PAD_IDX).transpose(0, 1)
            tgt_key_padding_mask = (tgt_inp == PAD_IDX).transpose(0, 1)
            memory_key_padding_mask = src_key_padding_mask.clone()
            
            optimizer.zero_grad()
            
            output = model(src=src, tgt=tgt_inp, 
                           tgt_mask=tgt_mask,
                           src_key_padding_mask = src_key_padding_mask,
                           tgt_key_padding_mask = tgt_key_padding_mask,
                           memory_key_padding_mask = memory_key_padding_mask)
            
            loss = loss_fn(output.view(-1, output.shape[2]),
                           tgt_out.view(-1))
            
            loss.backward()
            
            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (i+1)
            t.set_postfix(loss='{:05.3f}'.format(avg_loss),
                          ppl='{:05.3f}'.format(np.exp(avg_loss)))
            t.update()
            
    return epoch_loss / len(iterator)
    
def evaluate_transformer(model, iterator, loss_fn, device):
    model.eval()
        
    epoch_loss = 0
    with torch.no_grad():
        with tqdm(total=len(iterator), leave=False) as t:
            for i, (src, tgt) in enumerate(iterator):
                src = src.to(device)
                tgt = tgt.to(device)
                
                # Create tgt_inp and tgt_out (which is tgt_inp but shifted by 1)
                tgt_inp, tgt_out = tgt[:-1, :], tgt[1:, :]
                
                tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_inp.size(0)).to(device)
                src_key_padding_mask = (src == PAD_IDX).transpose(0, 1)
                tgt_key_padding_mask = (tgt_inp == PAD_IDX).transpose(0, 1)
                memory_key_padding_mask = src_key_padding_mask.clone()

                output = model(src=src, tgt=tgt_inp, 
                               tgt_mask=tgt_mask,
                               src_key_padding_mask = src_key_padding_mask,
                               tgt_key_padding_mask = tgt_key_padding_mask,
                               memory_key_padding_mask = memory_key_padding_mask)
                
                loss = loss_fn(output.view(-1, output.shape[2]),
                               tgt_out.view(-1))
                
                epoch_loss += loss.item()
                
                avg_loss = epoch_loss / (i+1)
                t.set_postfix(loss='{:05.3f}'.format(avg_loss),
                              ppl='{:05.3f}'.format(np.exp(avg_loss)))
                t.update()
    
    return epoch_loss / len(iterator)



# In[23]:


transformer = TransformerModel(input_dim=len(en_vocab), 
                             output_dim=len(tr_vocab), 
                             d_model=256, 
                             num_attention_heads=8,
                             num_encoder_layers=6, 
                             num_decoder_layers=6, 
                             dim_feedforward=2048,
                             max_seq_length=32,
                             pos_dropout=0.15,
                             transformer_dropout=0.3)

transformer = transformer.to(device)


# In[27]:


xf_optim = torch.optim.AdamW(transformer.parameters(), lr=1e-4)


# In[28]:


loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


# In[ ]:


#get_ipython().run_cell_magic('time', '', 'N_EPOCHS = 50\nCLIP = 15 # clipping value, or None to prevent gradient clipping\nEARLY_STOPPING_EPOCHS = 5\n    \nmodel_path = os.path.join(SAVE_DIR, \'transformer_en_fr.pt\')\ntransformer_metrics = {}\nbest_valid_loss = float("inf")\nearly_stopping_count = 0\nfor epoch in tqdm(range(N_EPOCHS), desc="Epoch"):\n    train_loss = train_transformer(transformer, train_iter, xf_optim, loss_fn, device, clip=CLIP)\n    valid_loss = evaluate_transformer(transformer, valid_iter, loss_fn, device)\n    \n    if valid_loss < best_valid_loss:\n        tqdm.write(f"Checkpointing at epoch {epoch + 1}")\n        best_valid_loss = valid_loss\n        torch.save(transformer.state_dict(), model_path)\n        early_stopping_count = 0\n    elif epoch > EARLY_STOPPING_EPOCHS:\n        early_stopping_count += 1\n    \n    transformer_metrics[epoch+1] = dict(\n        train_loss = train_loss,\n        train_ppl = np.exp(train_loss),\n        valid_loss = valid_loss,\n        valid_ppl = np.exp(valid_loss)\n    )\n    \n    if early_stopping_count == EARLY_STOPPING_EPOCHS:\n        tqdm.write(f"Early stopping triggered in epoch {epoch + 1}")\n        break\n')
N_EPOCHS = 50
CLIP = 15 # clipping value, or None to prevent gradient clipping
EARLY_STOPPING_EPOCHS = 5
SAVE_DIR = "C://Users//vuralbayrakli//transformerstranslation"   
model_path = os.path.join(SAVE_DIR, 'transformer_en_tr.pt')
transformer_metrics = {}
best_valid_loss = float("inf")
early_stopping_count = 0
for epoch in tqdm(range(N_EPOCHS), desc="Epoch"):
    train_loss = train_transformer(transformer, train_iter, xf_optim, loss_fn, device, clip=CLIP)
    valid_loss = evaluate_transformer(transformer, valid_iter, loss_fn, device)
    
    if valid_loss < best_valid_loss:
        tqdm.write(f"Checkpointing at epoch {epoch + 1}")
        best_valid_loss = valid_loss
        torch.save(transformer.state_dict(), model_path)
        early_stopping_count = 0
    elif epoch > EARLY_STOPPING_EPOCHS:
        early_stopping_count += 1
    
    transformer_metrics[epoch+1] = dict(
        train_loss = train_loss,
        train_ppl = np.exp(train_loss),
        valid_loss = valid_loss,
        valid_ppl = np.exp(valid_loss)
    )
    
    if early_stopping_count == EARLY_STOPPING_EPOCHS:
        tqdm.write(f"Early stopping triggered in epoch {epoch + 1}")
        break

# In[ ]:




