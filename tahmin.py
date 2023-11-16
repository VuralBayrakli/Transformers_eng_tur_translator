
print("Bekleyin...")

# Gerekli kütüphaneleri içe aktar
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
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import pickle
import spacy


# In[2]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# Metni normalize etmek için bir fonksiyon


def unicodeToAscii(s):
    return ''.join(
      c for c in unicodedata.normalize('NFD', s) 
      if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z.!?]+", " ", s)
    return s

def normalizeString(s):
    # Metni düşük harfe dönüştür, Türkçe karakterleri ve belirli noktalama işaretlerini koru
    s = s.lower().strip()
    # Sadece harfler, nokta (.), soru işareti (?), ünlem işareti (!) ve Türkçe karakterleri koru
    s = re.sub(r"[^a-zçğıöşü.!?,'']+", " ", s)
    return s

# Belirli kriterlere göre çiftleri filtrelemek için bir fonksiyon
def filterPair(p, max_length, prefixes):
    # Her iki cümle de belirtilen maksimum uzunluktan daha kısa mı kontrol et
    good_length = (len(p[0].split(' ')) < max_length) and (len(p[1].split(' ')) < max_length)
    # Eğer önekler belirtilmişse, cümlenin önek ile başlayıp başlamadığını kontrol et
    if len(prefixes) == 0:
        return good_length
    else:
        return good_length and p[0].startswith(prefixes)

# Belirli kriterlere göre çiftleri filtrelemek için bir fonksiyon
def filterPairs(pairs, max_length, prefixes=()):
    return [pair for pair in pairs if filterPair(pair, max_length, prefixes)]


# İngilizce dil modelini yükleme
en_nlp = spacy.load("en_core_web_sm")

# İngilizce cümleleri token'lara ayıran fonksiyon
def tokenize_en(text):
    return [tok.text for tok in en_nlp.tokenizer(text)]

print("Bekleyin...")

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

# Değişkeni geri yükleme
with open('en_vocab.pkl', 'rb') as f:
    en_vocab = pickle.load(f)
    
with open('tr_vocab.pkl', 'rb') as f:
    tr_vocab = pickle.load(f)
    
PAD_IDX = en_vocab['<pad>']
BOS_IDX = en_vocab['<bos>']
EOS_IDX = en_vocab['<eos>']

SPECIALS = ['<unk>', '<pad>', '<bos>', '<eos>']

for en_id, tr_id in zip(en_vocab.lookup_indices(SPECIALS), tr_vocab.lookup_indices(SPECIALS)):
    assert en_id == tr_id

# Değişkeni geri yükleme
with open('train_iter.pkl', 'rb') as f:
    train_iter = pickle.load(f)

with open('en_vocab.pkl', 'rb') as f:
    en_vocab = pickle.load(f)    
    # Değişkeni geri yükleme
with open('tr_vocab.pkl', 'rb') as f:
    tr_vocab = pickle.load(f)
    
with open('test_iter.pkl', 'rb') as f:
    test_iter = pickle.load(f)
    
with open('valid_iter.pkl', 'rb') as f:
    valid_iter = pickle.load(f)
    
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



# In[10]:


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

transformer.load_state_dict(torch.load("transformer_en_tr_son2.pt", map_location=device))

MAX_SENTENCE_LENGTH = 32

def predict_transformer(text, model, 
                        src_vocab=en_vocab, 
                        src_tokenizer=tokenize_en, 
                        tgt_vocab=tr_vocab, 
                        device=device):
    
    input_ids = [src_vocab[token.lower()] for token in src_tokenizer(text)]
    input_ids = [BOS_IDX] + input_ids + [EOS_IDX]
    
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_ids).to(device).unsqueeze(1) 
        
        causal_out = torch.ones(MAX_SENTENCE_LENGTH, 1).long().to(device) * BOS_IDX
        for t in range(1, MAX_SENTENCE_LENGTH):
            decoder_output = transformer(input_tensor, causal_out[:t, :])[-1, :, :]
            next_token = decoder_output.data.topk(1)[1].squeeze()
            causal_out[t, :] = next_token
            if next_token.item() == EOS_IDX:
                break
                
        pred_words = [tgt_vocab.lookup_token(tok.item()) for tok in causal_out.squeeze(1)[1:(t)]]
        return " ".join(pred_words)


#Creating tkinter GUI
import tkinter
from tkinter import *

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12 ))
    
        
        trr = predict_transformer(msg, transformer)
        
        ChatBox.insert(END, "translation: " + trr + '\n\n')
            
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)
 

root = Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatBox.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(root, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#f9a602", activebackground="#3c9d9b",fg='#000000',
                    command= send )

#Create the box to enter message
EntryBox = Text(root, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)
EntryBox.insert(END, "Buraya yazın...")

#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatBox.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

root.mainloop()





