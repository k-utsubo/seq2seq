# %%


# %%
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import sample_model



# %%
def evaluate(device, EOS_token, encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = sample_model.tensorFromSentence(device, EOS_token, input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

# %%
def evaluateRandomly(device, EOS_token,encoder, decoder,pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(device, EOS_token, encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluateAndShowAttention(device, EOS_token, input_sentence):
    output_words, attentions = evaluate(device, EOS_token,encoder, decoder, input_sentence, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))

input_lang, output_lang, pairs = sample_model.prepareData(sample_model.G_MAX_LENGTH, sample_model.G_eng_prefixes,'eng', 'fra', True)
# %%
hidden_size = 128
batch_size = 32
device = torch.device( "cpu")

#input_lang, output_lang, train_dataloader = sample_model.get_dataloader(device, sample_model.G_MAX_LENGTH, sample_model.G_EOS_token,sample_model.G_eng_prefixes,batch_size)

encoder = sample_model.EncoderRNN(device, input_lang.n_words, hidden_size).to(device)
decoder = sample_model.AttnDecoderRNN(device, sample_model.G_SOS_token, sample_model.G_MAX_LENGTH,hidden_size, output_lang.n_words).to(device)

# %%
encoder.load_state_dict(torch.load("model/encoder.pt"))
decoder.load_state_dict(torch.load("model/decoder.pt"))

# %%
encoder.eval()
decoder.eval()


# %%
# predict

# %%
evaluateRandomly(device, sample_model.G_EOS_token, encoder, decoder, pairs)




evaluateAndShowAttention(device,sample_model.G_EOS_token, 'il n est pas aussi grand que son pere')

evaluateAndShowAttention(device,sample_model.G_EOS_token,'je suis trop fatigue pour conduire')

evaluateAndShowAttention(device,sample_model.G_EOS_token,'je suis desole si c est une question idiote')

evaluateAndShowAttention(device,sample_model.G_EOS_token,'je suis reellement fiere de vous')
