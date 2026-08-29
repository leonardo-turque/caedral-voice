---
url: https://huggingface.co/hubertsiuzdak/snac_24khz
title: hubertsiuzdak/snac_24khz · Hugging Face
retrieved_at: 2026-08-22T18:35:54Z
api_version: v2
---

# SNAC 🍿

Multi- **S** cale **N** eural **A** udio **C** odec (SNAC) compressess audio into discrete codes at a low bitrate.

👉 This model was primarily trained on speech data, and its recommended use case is speech synthesis. See below for other pretrained models.

🔗 GitHub repository: [https://github.com/hubertsiuzdak/snac/](https://github.com/hubertsiuzdak/snac/)

## Overview

SNAC encodes audio into hierarchical tokens similarly to SoundStream, EnCodec, and DAC. However, SNAC introduces a simple change where coarse tokens are sampled less frequently,
covering a broader time span.

This model compresses 24 kHz audio into discrete codes at a 0.98 kbps bitrate. It uses 3 RVQ levels with token rates of 12, 23, and
47 Hz.

## Pretrained models

Currently, all models support only single audio channel (mono).

| Model | Bitrate | Sample Rate | Params | Recommended use case |
| --- | --- | --- | --- | --- |
| hubertsiuzdak/snac\_24khz (this model) | 0.98 kbps | 24 kHz | 19.8 M | 🗣️ Speech |
| [hubertsiuzdak/snac\_32khz](https://huggingface.co/hubertsiuzdak/snac_32khz) | 1.9 kbps | 32 kHz | 54.5 M | 🎸 Music / Sound Effects |
| [hubertsiuzdak/snac\_44khz](https://huggingface.co/hubertsiuzdak/snac_44khz) | 2.6 kbps | 44 kHz | 54.5 M | 🎸 Music / Sound Effects |

## Usage

Install it using:

```bash
pip install snac
```

To encode (and decode) audio with SNAC in Python, use the following code:

```python
import torch
from snac import SNAC

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
audio = torch.randn(1, 1, 24000).cuda()  # B, 1, T

with torch.inference_mode():
    codes = model.encode(audio)
    audio_hat = model.decode(codes)
```

You can also encode and reconstruct in a single call:

```python
with torch.inference_mode():
    audio_hat, codes = model(audio)
```

⚠️ Note that `codes` is a list of token sequences of variable lengths, each corresponding to a different temporal
resolution.

```
>>> [code.shape[1] for code in codes]
[12, 24, 48]
```

## Acknowledgements

Module definitions are adapted from the [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).

Downloads last month455,157

Inference Providers [NEW](https://huggingface.co/docs/inference-providers)

This model isn't deployed by any Inference Provider. [🙋Ask for provider support](https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/new?title=hubertsiuzdak/snac_24khz&description=React%20to%20this%20comment%20with%20an%20emoji%20to%20vote%20for%20%5Bhubertsiuzdak%2Fsnac_24khz%5D(%2Fhubertsiuzdak%2Fsnac_24khz)%20to%20be%20supported%20by%20Inference%20Providers.%0A%0A(optional)%20Which%20providers%20are%20you%20interested%20in%3F%20(Novita%2C%20Hyperbolic%2C%20Together%E2%80%A6)%0A)

## Model tree for hubertsiuzdak/snac\_24khz

Finetunes

[5 models](https://huggingface.co/models?other=base_model:finetune:hubertsiuzdak/snac_24khz)

Quantizations

[2 models](https://huggingface.co/models?other=base_model:quantized:hubertsiuzdak/snac_24khz)

## Spaces using hubertsiuzdak/snac\_24khz100