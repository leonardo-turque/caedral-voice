---
url: https://cdn.jsdelivr.net/gh/huawei-noah/Speech-Backbones@main/Grad-TTS/train.py
title: 
retrieved_at: 2026-08-22T20:07:21Z
api_version: v2
---

\# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
\# This program is free software; you can redistribute it and/or modify
\# it under the terms of the MIT License.
\# This program is distributed in the hope that it will be useful,
\# but WITHOUT ANY WARRANTY; without even the implied warranty of
\# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
\# MIT License for more details.

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from model import GradTTS
from data import TextMelDataset, TextMelBatchCollate
from utils import plot\_tensor, save\_plot
from text.symbols import symbols

train\_filelist\_path = params.train\_filelist\_path
valid\_filelist\_path = params.valid\_filelist\_path
cmudict\_path = params.cmudict\_path
add\_blank = params.add\_blank

log\_dir = params.log\_dir
n\_epochs = params.n\_epochs
batch\_size = params.batch\_size
out\_size = params.out\_size
learning\_rate = params.learning\_rate
random\_seed = params.seed

nsymbols = len(symbols) + 1 if add\_blank else len(symbols)
n\_enc\_channels = params.n\_enc\_channels
filter\_channels = params.filter\_channels
filter\_channels\_dp = params.filter\_channels\_dp
n\_enc\_layers = params.n\_enc\_layers
enc\_kernel = params.enc\_kernel
enc\_dropout = params.enc\_dropout
n\_heads = params.n\_heads
window\_size = params.window\_size

n\_feats = params.n\_feats
n\_fft = params.n\_fft
sample\_rate = params.sample\_rate
hop\_length = params.hop\_length
win\_length = params.win\_length
f\_min = params.f\_min
f\_max = params.f\_max

dec\_dim = params.dec\_dim
beta\_min = params.beta\_min
beta\_max = params.beta\_max
pe\_scale = params.pe\_scale

if \_\_name\_\_ == "\_\_main\_\_":
 torch.manual\_seed(random\_seed)
 np.random.seed(random\_seed)

 print('Initializing logger...')
 logger = SummaryWriter(log\_dir=log\_dir)

 print('Initializing data loaders...')
 train\_dataset = TextMelDataset(train\_filelist\_path, cmudict\_path, add\_blank,
 n\_fft, n\_feats, sample\_rate, hop\_length,
 win\_length, f\_min, f\_max)
 batch\_collate = TextMelBatchCollate()
 loader = DataLoader(dataset=train\_dataset, batch\_size=batch\_size,
 collate\_fn=batch\_collate, drop\_last=True,
 num\_workers=4, shuffle=False)
 test\_dataset = TextMelDataset(valid\_filelist\_path, cmudict\_path, add\_blank,
 n\_fft, n\_feats, sample\_rate, hop\_length,
 win\_length, f\_min, f\_max)

 print('Initializing model...')
 model = GradTTS(nsymbols, 1, None, n\_enc\_channels, filter\_channels, filter\_channels\_dp,
 n\_heads, n\_enc\_layers, enc\_kernel, enc\_dropout, window\_size,
 n\_feats, dec\_dim, beta\_min, beta\_max, pe\_scale).cuda()
 print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams/1e6))
 print('Number of decoder parameters: %.2fm' % (model.decoder.nparams/1e6))
 print('Total parameters: %.2fm' % (model.nparams/1e6))

 print('Initializing optimizer...')
 optimizer = torch.optim.Adam(params=model.parameters(), lr=learning\_rate)

 print('Logging test batch...')
 test\_batch = test\_dataset.sample\_test\_batch(size=params.test\_size)
 for i, item in enumerate(test\_batch):
 mel = item\['y'\]
 logger.add\_image(f'image\_{i}/ground\_truth', plot\_tensor(mel.squeeze()),
 global\_step=0, dataformats='HWC')
 save\_plot(mel.squeeze(), f'{log\_dir}/original\_{i}.png')

 print('Start training...')
 iteration = 0
 for epoch in range(1, n\_epochs + 1):
 model.train()
 dur\_losses = \[\]
 prior\_losses = \[\]
 diff\_losses = \[\]
 with tqdm(loader, total=len(train\_dataset)//batch\_size) as progress\_bar:
 for batch\_idx, batch in enumerate(progress\_bar):
 model.zero\_grad()
 x, x\_lengths = batch\['x'\].cuda(), batch\['x\_lengths'\].cuda()
 y, y\_lengths = batch\['y'\].cuda(), batch\['y\_lengths'\].cuda()
 dur\_loss, prior\_loss, diff\_loss = model.compute\_loss(x, x\_lengths,
 y, y\_lengths,
 out\_size=out\_size)
 loss = sum(\[dur\_loss, prior\_loss, diff\_loss\])
 loss.backward()

 enc\_grad\_norm = torch.nn.utils.clip\_grad\_norm\_(model.encoder.parameters(),
 max\_norm=1)
 dec\_grad\_norm = torch.nn.utils.clip\_grad\_norm\_(model.decoder.parameters(),
 max\_norm=1)
 optimizer.step()

 logger.add\_scalar('training/duration\_loss', dur\_loss.item(),
 global\_step=iteration)
 logger.add\_scalar('training/prior\_loss', prior\_loss.item(),
 global\_step=iteration)
 logger.add\_scalar('training/diffusion\_loss', diff\_loss.item(),
 global\_step=iteration)
 logger.add\_scalar('training/encoder\_grad\_norm', enc\_grad\_norm,
 global\_step=iteration)
 logger.add\_scalar('training/decoder\_grad\_norm', dec\_grad\_norm,
 global\_step=iteration)

 dur\_losses.append(dur\_loss.item())
 prior\_losses.append(prior\_loss.item())
 diff\_losses.append(diff\_loss.item())

 if batch\_idx % 5 == 0:
 msg = f'Epoch: {epoch}, iteration: {iteration} \| dur\_loss: {dur\_loss.item()}, prior\_loss: {prior\_loss.item()}, diff\_loss: {diff\_loss.item()}'
 progress\_bar.set\_description(msg)

 iteration += 1

 log\_msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur\_losses))
 log\_msg += '\| prior loss = %.3f ' % np.mean(prior\_losses)
 log\_msg += '\| diffusion loss = %.3f\\n' % np.mean(diff\_losses)
 with open(f'{log\_dir}/train.log', 'a') as f:
 f.write(log\_msg)

 if epoch % params.save\_every > 0:
 continue

 model.eval()
 print('Synthesis...')
 with torch.no\_grad():
 for i, item in enumerate(test\_batch):
 x = item\['x'\].to(torch.long).unsqueeze(0).cuda()
 x\_lengths = torch.LongTensor(\[x.shape\[-1\]\]).cuda()
 y\_enc, y\_dec, attn = model(x, x\_lengths, n\_timesteps=50)
 logger.add\_image(f'image\_{i}/generated\_enc',
 plot\_tensor(y\_enc.squeeze().cpu()),
 global\_step=iteration, dataformats='HWC')
 logger.add\_image(f'image\_{i}/generated\_dec',
 plot\_tensor(y\_dec.squeeze().cpu()),
 global\_step=iteration, dataformats='HWC')
 logger.add\_image(f'image\_{i}/alignment',
 plot\_tensor(attn.squeeze().cpu()),
 global\_step=iteration, dataformats='HWC')
 save\_plot(y\_enc.squeeze().cpu(),
 f'{log\_dir}/generated\_enc\_{i}.png')
 save\_plot(y\_dec.squeeze().cpu(),
 f'{log\_dir}/generated\_dec\_{i}.png')
 save\_plot(attn.squeeze().cpu(),
 f'{log\_dir}/alignment\_{i}.png')

 ckpt = model.state\_dict()
 torch.save(ckpt, f=f"{log\_dir}/grad\_{epoch}.pt")