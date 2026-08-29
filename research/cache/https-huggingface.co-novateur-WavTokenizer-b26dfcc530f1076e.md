---
url: https://huggingface.co/novateur/WavTokenizer
title: novateur/WavTokenizer · Hugging Face
retrieved_at: 2026-08-22T18:35:57Z
api_version: v2
---

# WavTokenizer: SOTA Discrete Codec Models With Forty Tokens Per Second for Audio Language Modeling

[![arXiv](https://img.shields.io/badge/arXiv-Paper-%3CCOLOR%3E.svg)](https://arxiv.org/abs/2408.16532)[![demo](https://img.shields.io/badge/WanTokenizer-Demo-red)](https://wavtokenizer.github.io/)[![model](https://img.shields.io/badge/%F0%9F%A4%97%20WavTokenizer-Models-blue)](https://huggingface.co/novateur/WavTokenizer)

### 🎉🎉 with WavTokenizer, you can represent speech, music, and audio with only 40 tokens per second!

### 🎉🎉 with WavTokenizer, You can get strong reconstruction results.

### 🎉🎉 WavTokenizer owns rich semantic information and is build for audio language models such as GPT4-o.

# 🔥 News

- _2024.08_: We release WavTokenizer on arxiv.

[![result](https://huggingface.co/novateur/WavTokenizer/resolve/main/result.png)](https://huggingface.co/novateur/WavTokenizer/blob/main/result.png)

## Installation

To use WavTokenizer, install it using:

```bash
conda create -n wavtokenizer python=3.9
conda activate wavtokenizer
pip install -r requirements.txt
```

## Infer

### Part1: Reconstruct audio from raw wav

```python

from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer

device=torch.device('cpu')

config_path = "./configs/xxx.yaml"
model_path = "./xxx.ckpt"
audio_outpath = "xxx"

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)

wav, sr = torchaudio.load(audio_path)
wav = convert_audio(wav, sr, 24000, 1)
bandwidth_id = torch.tensor([0])
wav=wav.to(device)
features,discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
torchaudio.save(audio_outpath, audio_out, sample_rate=24000, encoding='PCM_S', bits_per_sample=16)
```

### Part2: Generating discrete codecs

```python

from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer

device=torch.device('cpu')

config_path = "./configs/xxx.yaml"
model_path = "./xxx.ckpt"

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)

wav, sr = torchaudio.load(audio_path)
wav = convert_audio(wav, sr, 24000, 1)
bandwidth_id = torch.tensor([0])
wav=wav.to(device)
_,discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
print(discrete_code)
```

### Part3: Audio reconstruction through codecs

```python
# audio_tokens [n_q,1,t]/[n_q,t]
features = wavtokenizer.codes_to_features(audio_tokens)
bandwidth_id = torch.tensor([0])
audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
```

## Available models

🤗 links to the Huggingface model hub.

| Model name | HuggingFace | Corpus | Token/s | Domain | Open-Source |
| :-- | :-: | :-: | :-: | :-: | :-: |
| WavTokenizer-small-600-24k-4096 | [🤗](https://huggingface.co/novateur/WavTokenizer/blob/main/WavTokenizer_small_600_24k_4096.ckpt) | LibriTTS | 40 | Speech | √ |
| WavTokenizer-small-320-24k-4096 | [🤗](https://huggingface.co/novateur/WavTokenizer/blob/main/WavTokenizer_small_320_24k_4096.ckpt) | LibriTTS | 75 | Speech | √ |
| WavTokenizer-medium-600-24k-4096 | [🤗](https://github.com/jishengpeng/wavtokenizer) | 10000 Hours | 40 | Speech, Audio, Music | Coming Soon |
| WavTokenizer-medium-320-24k-4096 | [🤗](https://github.com/jishengpeng/wavtokenizer) | 10000 Hours | 75 | Speech, Audio, Music | Coming Soon |
| WavTokenizer-large-600-24k-4096 | [🤗](https://github.com/jishengpeng/wavtokenizer) | 80000 Hours | 40 | Speech, Audio, Music | Coming Soon |
| WavTokenizer-large-320-24k-4096 | [🤗](https://github.com/jishengpeng/wavtokenizer) | 80000 Hours | 75 | Speech, Audio, Music | Coming Soon |

## Training

### Step1: Prepare train dataset

```python
# Process the data into a form similar to ./data/demo.txt
```

### Step2: Modifying configuration files

```python
# ./configs/xxx.yaml
# Modify the values of parameters such as batch_size, filelist_path, save_dir, device
```

### Step3: Start training process

Refer to [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/) for details about customizing the
training pipeline.

```bash
cd ./WavTokenizer
python train.py fit --config ./configs/xxx.yaml
```

## Citation

If this code contributes to your research, please cite our work, Language-Codec and WavTokenizer:

```
@article{ji2024wavtokenizer,
  title={Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling},
  author={Ji, Shengpeng and Jiang, Ziyue and Wang, Wen and Chen, Yifu and Fang, Minghui and Zuo, Jialong and Yang, Qian and Cheng, Xize and Wang, Zehan and Li, Ruiqi and others},
  journal={arXiv preprint arXiv:2408.16532},
  year={2024}
}

@article{ji2024language,
  title={Language-codec: Reducing the gaps between discrete codec representation and speech language models},
  author={Ji, Shengpeng and Fang, Minghui and Jiang, Ziyue and Huang, Rongjie and Zuo, Jialung and Wang, Shulei and Zhao, Zhou},
  journal={arXiv preprint arXiv:2402.12208},
  year={2024}
}
```

Downloads last month

-

Downloads are not tracked for this model. [How to track](https://huggingface.co/docs/hub/models-download-stats)

Inference Providers [NEW](https://huggingface.co/docs/inference-providers)

[Text-to-Speech](https://huggingface.co/tasks/text-to-speech "Learn more about text-to-speech")

This model isn't deployed by any Inference Provider. [🙋Ask for provider support](https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/new?title=novateur/WavTokenizer&description=React%20to%20this%20comment%20with%20an%20emoji%20to%20vote%20for%20%5Bnovateur%2FWavTokenizer%5D(%2Fnovateur%2FWavTokenizer)%20to%20be%20supported%20by%20Inference%20Providers.%0A%0A(optional)%20Which%20providers%20are%20you%20interested%20in%3F%20(Novita%2C%20Hyperbolic%2C%20Together%E2%80%A6)%0A)

## Space using novateur/WavTokenizer1

## Papers for novateur/WavTokenizer

[Paper • 2408.16532 •Published Aug 29, 2024• 50](https://huggingface.co/papers/2408.16532)[Paper • 2402.12208 •Published Feb 19, 2024](https://huggingface.co/papers/2402.12208)