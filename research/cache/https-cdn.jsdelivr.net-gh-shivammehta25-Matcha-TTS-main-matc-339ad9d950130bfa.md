---
url: https://cdn.jsdelivr.net/gh/shivammehta25/Matcha-TTS@main/matcha/utils/model.py
title: 
retrieved_at: 2026-08-22T20:06:23Z
api_version: v2
---

""" from https://github.com/jaywalnut310/glow-tts """

import numpy as np
import torch

def sequence\_mask(length, max\_length=None):
 if max\_length is None:
 max\_length = length.max()
 x = torch.arange(max\_length, dtype=length.dtype, device=length.device)
 return x.unsqueeze(0) < length.unsqueeze(1)

def fix\_len\_compatibility(length, num\_downsamplings\_in\_unet=2):
 factor = torch.scalar\_tensor(2).pow(num\_downsamplings\_in\_unet)
 length = (length / factor).ceil() \* factor
 if not torch.onnx.is\_in\_onnx\_export():
 return length.int().item()
 else:
 return length

def convert\_pad\_shape(pad\_shape):
 inverted\_shape = pad\_shape\[::-1\]
 pad\_shape = \[item for sublist in inverted\_shape for item in sublist\]
 return pad\_shape

def generate\_path(duration, mask):
 device = duration.device

 b, t\_x, t\_y = mask.shape
 cum\_duration = torch.cumsum(duration, 1)
 path = torch.zeros(b, t\_x, t\_y, dtype=mask.dtype).to(device=device)

 cum\_duration\_flat = cum\_duration.view(b \* t\_x)
 path = sequence\_mask(cum\_duration\_flat, t\_y).to(mask.dtype)
 path = path.view(b, t\_x, t\_y)
 path = path - torch.nn.functional.pad(path, convert\_pad\_shape(\[\[0, 0\], \[1, 0\], \[0, 0\]\]))\[:, :-1\]
 path = path \* mask
 return path

def duration\_loss(logw, logw\_, lengths):
 loss = torch.sum((logw - logw\_) \*\* 2) / torch.sum(lengths)
 return loss

def normalize(data, mu, std):
 if not isinstance(mu, (float, int)):
 if isinstance(mu, list):
 mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
 elif isinstance(mu, torch.Tensor):
 mu = mu.to(data.device)
 elif isinstance(mu, np.ndarray):
 mu = torch.from\_numpy(mu).to(data.device)
 mu = mu.unsqueeze(-1)

 if not isinstance(std, (float, int)):
 if isinstance(std, list):
 std = torch.tensor(std, dtype=data.dtype, device=data.device)
 elif isinstance(std, torch.Tensor):
 std = std.to(data.device)
 elif isinstance(std, np.ndarray):
 std = torch.from\_numpy(std).to(data.device)
 std = std.unsqueeze(-1)

 return (data - mu) / std

def denormalize(data, mu, std):
 if not isinstance(mu, float):
 if isinstance(mu, list):
 mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
 elif isinstance(mu, torch.Tensor):
 mu = mu.to(data.device)
 elif isinstance(mu, np.ndarray):
 mu = torch.from\_numpy(mu).to(data.device)
 mu = mu.unsqueeze(-1)

 if not isinstance(std, float):
 if isinstance(std, list):
 std = torch.tensor(std, dtype=data.dtype, device=data.device)
 elif isinstance(std, torch.Tensor):
 std = std.to(data.device)
 elif isinstance(std, np.ndarray):
 std = torch.from\_numpy(std).to(data.device)
 std = std.unsqueeze(-1)

 return data \* std + mu