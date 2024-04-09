# %%
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

# %%

# %%
from __future__ import unicode_literals, print_function, division
from io import open

import torch
import torch.nn as nn
from torch import optim
import json


import sample_model




# %%
def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# %%
import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# %%

# %%
def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


# %%
#eval

# %%
def evaluate(device,EOS_token, encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = sample_model.tensorFromSentence(device ,EOS_token, input_lang, sentence)

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
# exec

# %%
hidden_size = 128
batch_size = 32
n_epoch=80
n_epoch = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


input_lang, output_lang, train_dataloader = sample_model.get_dataloader(device,sample_model.G_MAX_LENGTH, sample_model.G_EOS_token,sample_model.G_eng_prefixes,  batch_size)

input_lang.save("model")
output_lang.save("model")

encoder = sample_model.EncoderRNN(device, input_lang.n_words, hidden_size).to(device)
decoder = sample_model.AttnDecoderRNN(device, sample_model.G_SOS_token, sample_model.G_MAX_LENGTH,hidden_size, output_lang.n_words).to(device)

params={}
params['hidden_size']=hidden_size
params['batch_size']=batch_size
params['n_epoch']=n_epoch
params["MAX_LENGTH"]=sample_model.G_MAX_LENGTH
params["SOS_token"] = sample_model.G_SOS_token
params["EOS_token"] = sample_model.G_EOS_token

with open('model/params.json','w',encoding='utf-8') as f:
    json.dump(params,f,indent=2,ensure_ascii=False)

train(train_dataloader, encoder, decoder, n_epoch, print_every=5, plot_every=5)

# %%
encoder.eval()
decoder.eval()


# %%
import os
os.makedirs("model",exist_ok=True)
encoder.device=torch.device('cpu')
decoder.device=torch.device('cpu')
encoder.cpu()
decoder.cpu()
print(encoder.device)
print(decoder.device)
print(encoder.embedding.weight.device)
# %%
torch.save(encoder.state_dict(),"model/encoder.pt")
torch.save(decoder.state_dict(),"model/decoder.pt")

# %%
# https://pytorch.org/tutorials/beginner/saving_loading_models.html

encoder_scripted = torch.jit.script(encoder) # Export to TorchScript
encoder_scripted.save('model/encoder_scripted.pt') # Save


decoder_scripted = torch.jit.script(decoder) # Export to TorchScript
decoder_scripted.save('model/decoder_scripted.pt') # Save

# %%






