from io import open
import unicodedata
import string
import re
import random
import os
import pickle
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

#from torchtext.data.utils import get_tokenizer
#from torchtext.vocab import Vocab, build_vocab_from_iterator
from collections import Counter 

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_batch(data_batch):
    '''
    Veri yığınlarını modelleme için hazırlar. Her bir örneğe BOS/EOS belirteçlerini ekler, tensörleri birleştirir
    ve daha kısa cümlelerin sonundaki boşlukları <pad> belirteci ile doldurur. 
    English-to-Turkish DataLoader'ında collate_fn olarak kullanılması amaçlanmıştır.

    Input:
    - data_batch, yukarıda oluşturulan veri setlerinden alınan (İngilizce, Türkçe) tuple'larını içeren bir iterasyon

    Output:
    - en_batch: İngilizce token ID'leri içeren (maksimum uzunluk X yığın boyutu) bir tensör
    - tr_batch: Türkçe token ID'leri içeren (maksimum uzunluk X yığın boyutu) bir tensör 
    '''
    
    en_batch, tr_batch = [], []
    
    for (en_item, tr_item) in data_batch:
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        tr_batch.append(torch.cat([torch.tensor([BOS_IDX]), tr_item, torch.tensor([EOS_IDX])], dim=0))

    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX, batch_first=False)
    tr_batch = pad_sequence(tr_batch, padding_value=PAD_IDX, batch_first=False)

    return en_batch, tr_batch


# Veri dizini (current working directory)
datadir = os.getcwd()

# İngilizce ve Türkçe kelimelerin bulunduğu sözlükleri yükle
with open(os.path.join(datadir, 'en_vocab.pkl'), 'rb') as f:
    en_vocab = pickle.load(f)
    
with open(os.path.join(datadir,'tr_vocab.pkl'), 'rb') as f:
    tr_vocab = pickle.load(f)

# Padding, başlangıç ve bitiş belirteçlerinin token ID'lerini belirle
PAD_IDX = en_vocab['<pad>']
BOS_IDX = en_vocab['<bos>']
EOS_IDX = en_vocab['<eos>']

# Özel belirteçlerin (unknown, padding, bos, eos) token ID'lerini kontrol et
SPECIALS = ['<unk>', '<pad>', '<bos>', '<eos>']
for en_id, tr_id in zip(en_vocab.lookup_indices(SPECIALS), tr_vocab.lookup_indices(SPECIALS)):
    assert en_id == tr_id

# Eğitim veri setini yükle
with open(os.path.join(datadir,'train_iter.pkl'), 'rb') as f:
    train_iter = pickle.load(f)

# Test veri setini yükle
with open(os.path.join(datadir, 'test_iter.pkl'), 'rb') as f:
    test_iter = pickle.load(f)

# Doğrulama veri setini yükle
with open(os.path.join(datadir, 'valid_iter.pkl'), 'rb') as f:
    valid_iter = pickle.load(f)

    
# Eğitim veri setinden örnek bir batch'i görüntüleme
for i, (en_id, tr_id) in enumerate(train_iter):
    # İngilizce cümleyi token ID'lerini kelimelere dönüştürerek yazdır
    print('English:', ' '.join([en_vocab.lookup_token(idx) for idx in en_id[:, 0]]))
    
    # Türkçe cümleyi token ID'lerini kelimelere dönüştürerek yazdır
    print('Turkish:', ' '.join([tr_vocab.lookup_token(idx) for idx in tr_id[:, 0]]))
    
    # İlk 5 batch'i görüntüledikten sonra döngüyü sonlandır
    if i == 4: 
        break
    else:
        print()

        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_p=0.1, max_len=100):
        super().__init__()
        
        # Dropout katmanı
        self.dropout = nn.Dropout(dropout_p)
        
        # Maksimum uzunluk belirtilen değeri aşarsa, 
        # pozisyonel kodlamayı kesmek için kullanılır
        self.max_len = max_len
        
        # Pozisyonel kodlamayı hesapla ve 'pe' adında bir buffer olarak kaydet
        pe = self.calculate_positional_encoding(max_len, d_model)
        self.register_buffer('pe', pe)
        
    def calculate_positional_encoding(self, max_len, d_model):
        # Pozisyonlar: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Div term: (d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.) / d_model))
        
        # Pozisyonel kodlama matrisi oluştur
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Buffer olarak kaydet ve boyutları (1, max_len, d_model) yap
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        return pe
        
    def forward(self, x):
        # Giriş tensorüne pozisyonel kodlamayı ekleyin
        x = x + self.pe[:x.size(0), :].detach()  # detach added for no backpropagation on positional encodings
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_attention_heads, 
                 num_encoder_layers, num_decoder_layers, dim_feedforward, 
                 max_seq_length, pos_dropout, transformer_dropout):
        super().__init__()
        
        # Model parametreleri
        self.d_model = d_model
        
        # Giriş ve çıkış için gömme katmanları
        self.embed_src = nn.Embedding(input_dim, d_model)
        self.embed_tgt = nn.Embedding(output_dim, d_model)
        
        # Pozisyonel kodlama katmanı
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)
        
        # Transformer katmanı
        self.transformer = nn.Transformer(d_model, num_attention_heads, num_encoder_layers, 
                                          num_decoder_layers, dim_feedforward, transformer_dropout)
        
        # Çıkış katmanı
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
        
        # Giriş ve hedef gömme katmanlarını hazırla
        if (src_embeds is None) and (src is not None):
            if (tgt_embeds is None) and (tgt is not None):
                src_embeds, tgt_embeds = self._embed_tokens(src, tgt)
        elif (src_embeds is not None) and (src is not None):
            raise ValueError("src ve src_embeds' ten yalnızca biri belirtilmelidir.")
        elif (src_embeds is None) and (src is None):
            raise ValueError("src veya src_embeds' den biri belirtilmelidir.")
        elif (tgt_embeds is not None) and (tgt is not None):
            raise ValueError("tgt ve tgt_embeds' ten yalnızca biri belirtilmelidir.")
        elif (tgt_embeds is None) and (tgt is None):
            raise ValueError("tgt veya tgt_embeds' den biri belirtilmelidir.")
        
        # Transformer modelini uygula
        output = self.transformer(src_embeds, 
                                  tgt_embeds, 
                                  tgt_mask=tgt_mask, 
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        
        # Çıkış katmanına geçir
        return self.output(output)
    
    def _embed_tokens(self, src, tgt):
        # Giriş ve hedef gömme katmanlarını uygula
        src_embeds = self.embed_src(src) * np.sqrt(self.d_model)
        tgt_embeds = self.embed_tgt(tgt) * np.sqrt(self.d_model)
        
        # Pozisyonel kodlamayı uygula
        src_embeds = self.pos_enc(src_embeds)
        tgt_embeds = self.pos_enc(tgt_embeds)
        
        return src_embeds, tgt_embeds


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_attention_heads, 
                 num_encoder_layers, num_decoder_layers, dim_feedforward, 
                 max_seq_length, pos_dropout, transformer_dropout):
        super().__init__()
        
        # Model parametreleri
        self.d_model = d_model
        
        # Giriş ve çıkış için gömme katmanları
        self.embed_src = nn.Embedding(input_dim, d_model)
        self.embed_tgt = nn.Embedding(output_dim, d_model)
        
        # Pozisyonel kodlama katmanı
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)
        
        # Transformer katmanı
        self.transformer = nn.Transformer(d_model, num_attention_heads, num_encoder_layers, 
                                          num_decoder_layers, dim_feedforward, transformer_dropout)
        
        # Çıkış katmanı
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
        
        # Giriş ve hedef gömme katmanlarını hazırla
        if (src_embeds is None) and (src is not None):
            if (tgt_embeds is None) and (tgt is not None):
                src_embeds, tgt_embeds = self._embed_tokens(src, tgt)
        elif (src_embeds is not None) and (src is not None):
            raise ValueError("src ve src_embeds' ten yalnızca biri belirtilmelidir.")
        elif (src_embeds is None) and (src is None):
            raise ValueError("src veya src_embeds' den biri belirtilmelidir.")
        elif (tgt_embeds is not None) and (tgt is not None):
            raise ValueError("tgt ve tgt_embeds' ten yalnızca biri belirtilmelidir.")
        elif (tgt_embeds is None) and (tgt is None):
            raise ValueError("tgt veya tgt_embeds' den biri belirtilmelidir.")
        
        # Transformer modelini uygula
        output = self.transformer(src_embeds, 
                                  tgt_embeds, 
                                  tgt_mask=tgt_mask, 
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        
        # Çıkış katmanına geçir
        return self.output(output)
    
    def _embed_tokens(self, src, tgt):
        # Giriş ve hedef gömme katmanlarını uygula
        src_embeds = self.embed_src(src) * np.sqrt(self.d_model)
        tgt_embeds = self.embed_tgt(tgt) * np.sqrt(self.d_model)
        
        # Pozisyonel kodlamayı uygula
        src_embeds = self.pos_enc(src_embeds)
        tgt_embeds = self.pos_enc(tgt_embeds)
        
        return src_embeds, tgt_embeds

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


# TransformerModel sınıfından bir örnek oluştur
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

# Modelin parametrelerini belirtilen cihaza (device) taşı
transformer = transformer.to(device)

# Transformer modelinin parametrelerini optimize etmek için bir AdamW optimizasyonu tanımla
xf_optim = torch.optim.AdamW(transformer.parameters(), lr=1e-4)

# CrossEntropyLoss'u tanımla ve PAD_IDX'yi yoksay
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Eğitim için belirlenen epoch sayısı
N_EPOCHS = 50

# Gradyan sınırlama değeri (clipping value)
CLIP = 15

# Erken durdurma için belirlenen epoch sayısı
EARLY_STOPPING_EPOCHS = 5

# Modelin kaydedileceği dizin
SAVE_DIR = datadir  

# Modelin kaydedileceği dosya yolu
model_path = os.path.join(SAVE_DIR, 'transformer_en_tr.pt')

# Eğitim metriklerini saklamak için bir sözlük
transformer_metrics = {}

# En iyi geçerli kaybı takip etmek için bir değişken
best_valid_loss = float("inf")

# Erken durdurma sayacı
early_stopping_count = 0

# tqdm kullanarak epoch döngüsü oluştur
for epoch in tqdm(range(N_EPOCHS), desc="Epoch"):
    # Modeli eğit ve eğitim kaybını al
    train_loss = train_transformer(transformer, train_iter, xf_optim, loss_fn, device, clip=CLIP)
    
    # Modeli değerlendir ve geçerli kaybı al
    valid_loss = evaluate_transformer(transformer, valid_iter, loss_fn, device)
    
    # Eğer geçerli kayıp daha önce gördüğümüz en iyi kayıptan daha iyiyse
    if valid_loss < best_valid_loss:
        tqdm.write(f"Checkpointing at epoch {epoch + 1}")
        best_valid_loss = valid_loss
        # Modelin parametrelerini kaydet
        torch.save(transformer.state_dict(), model_path)
        # Erken durdurma sayacını sıfırla
        early_stopping_count = 0
    # Eğer geçerli epoch EARLY_STOPPING_EPOCHS'ten büyükse
    elif epoch > EARLY_STOPPING_EPOCHS:
        # Erken durdurma sayacını arttır
        early_stopping_count += 1
    
    # Eğitim ve geçerli kayıpları ve perpleksiteleri sakla
    transformer_metrics[epoch+1] = dict(
        train_loss = train_loss,
        train_ppl = np.exp(train_loss),
        valid_loss = valid_loss,
        valid_ppl = np.exp(valid_loss)
    )
    
    # Eğer erken durdurma sayacı belirlenen eşik değeri aştıysa
    if early_stopping_count == EARLY_STOPPING_EPOCHS:
        tqdm.write(f"Early stopping triggered in epoch {epoch + 1}")
        # Döngüyü sonlandır
        break

def predict_transformer(text, model, 
                        src_vocab=en_vocab, 
                        src_tokenizer=en_tokenizer, 
                        tgt_vocab=fr_vocab, 
                        device=device):
    
    input_ids = [src_vocab[token] for token in src_tokenizer(text)]
    input_ids = [BOS_IDX] + input_ids + [EOS_IDX]
    
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_ids).to(device).unsqueeze(1) # add fake batch dim
        
        causal_out = torch.ones(MAX_SENTENCE_LENGTH, 1).long().to(device) * BOS_IDX
        for t in range(1, MAX_SENTENCE_LENGTH):
            decoder_output = transformer(input_tensor, causal_out[:t, :])[-1, :, :]
            next_token = decoder_output.data.topk(1)[1].squeeze()
            causal_out[t, :] = next_token
            if next_token.item() == EOS_IDX:
                break
                
        pred_words = [tgt_vocab.lookup_token(tok.item()) for tok in causal_out.squeeze(1)[1:(t)]]
        return " ".join(pred_words)
        
        
predict_transformer("she is not my mother .", transformer)