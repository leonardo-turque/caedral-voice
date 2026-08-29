---
url: https://huggingface.co/hexgrad/Kokoro-82M
title: hexgrad/Kokoro-82M · Hugging Face
retrieved_at: 2026-08-22T20:08:31Z
api_version: v2
---

**Kokoro** is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.

🐈 **GitHub**: [https://github.com/hexgrad/kokoro](https://github.com/hexgrad/kokoro)

🚀 **Demo**: [https://hf.co/spaces/hexgrad/Kokoro-TTS](https://hf.co/spaces/hexgrad/Kokoro-TTS)

> As of April 2025, the market rate of Kokoro served over API is **under $1 per million characters of text input**, or under $0.06 per hour of audio output. (On average, 1000 characters of input is about 1 minute of output.) Sources: [ArtificialAnalysis/Replicate at 65 cents per M chars](https://artificialanalysis.ai/text-to-speech/model-family/kokoro#price) and [DeepInfra at 80 cents per M chars](https://deepinfra.com/hexgrad/Kokoro-82M).
>
> This is an Apache-licensed model, and Kokoro has been deployed in numerous projects and commercial APIs. We welcome the deployment of the model in real use cases.

> Fake websites like kokorottsai\_com (snapshot: [https://archive.ph/nRRnk](https://archive.ph/nRRnk)) and kokorotts\_net (snapshot: [https://archive.ph/60opa](https://archive.ph/60opa)) are likely scams masquerading under the banner of a popular model.
>
> Any website containing "kokoro" in its root domain (e.g. kokorottsai\_com, kokorotts\_net) is **NOT owned by and NOT affiliated with this model page or its author**, and attempts to imply otherwise are red flags.

- [Releases](https://huggingface.co/hexgrad/Kokoro-82M#releases)
- [Usage](https://huggingface.co/hexgrad/Kokoro-82M#usage)
- [EVAL.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/EVAL.md) ↗️
- [SAMPLES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/SAMPLES.md) ↗️
- [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) ↗️
- [Model Facts](https://huggingface.co/hexgrad/Kokoro-82M#model-facts)
- [Training Details](https://huggingface.co/hexgrad/Kokoro-82M#training-details)
- [Creative Commons Attribution](https://huggingface.co/hexgrad/Kokoro-82M#creative-commons-attribution)
- [Acknowledgements](https://huggingface.co/hexgrad/Kokoro-82M#acknowledgements)

### Releases

| Model | Published | Training Data | Langs & Voices | SHA256 |
| --- | --- | --- | --- | --- |
| **v1.0** | **2025 Jan 27** | **Few hundred hrs** | [**8 & 54**](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) | `496dba11` |
| [v0.19](https://huggingface.co/hexgrad/kLegacy/tree/main/v0.19) | 2024 Dec 25 | <100 hrs | 1 & 10 | `3b0c392f` |

| Training Costs | v0.19 | v1.0 | **Total** |
| --- | --- | --- | --- |
| in A100 80GB GPU hours | 500 | 500 | **1000** |
| average hourly rate | $0.80/h | $1.20/h | **$1/h** |
| in USD | $400 | $600 | **$1000** |

### Usage

You can run this basic cell on [Google Colab](https://colab.research.google.com/). [Listen to samples](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/SAMPLES.md). For more languages and details, see [Advanced Usage](https://github.com/hexgrad/kokoro?tab=readme-ov-file#advanced-usage).

```py
!pip install -q kokoro>=0.9.2 soundfile
!apt-get -qq -y install espeak-ng > /dev/null 2>&1
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
pipeline = KPipeline(lang_code='a')
text = '''
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
'''
generator = pipeline(text, voice='af_heart')
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write(f'{i}.wav', audio, 24000)
```

Under the hood, `kokoro` uses [`misaki`](https://pypi.org/project/misaki/), a G2P library at [https://github.com/hexgrad/misaki](https://github.com/hexgrad/misaki)

### Model Facts

**Architecture:**

- StyleTTS 2: [https://arxiv.org/abs/2306.07691](https://arxiv.org/abs/2306.07691)
- ISTFTNet: [https://arxiv.org/abs/2203.02395](https://arxiv.org/abs/2203.02395)
- Decoder only: no diffusion, no encoder release

**Architected by:** Li et al @ [https://github.com/yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)

**Trained by**: `@rzvzn` on Discord

**Languages:** Multiple

**Model SHA256 Hash:**`496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`

### Training Details

**Data:** Kokoro was trained exclusively on **permissive/non-copyrighted audio data** and IPA phoneme labels. Examples of permissive/non-copyrighted audio include:

- Public domain audio
- Audio licensed under Apache, MIT, etc
- Synthetic audio\[1\] generated by closed\[2\] TTS models from large providers


\[1\] [https://copyright.gov/ai/ai\_policy\_guidance.pdf](https://copyright.gov/ai/ai_policy_guidance.pdf)


\[2\] No synthetic audio from open TTS models or "custom voice clones"

**Total Dataset Size:** A few hundred hours of audio

**Total Training Cost:** About $1000 for 1000 hours of A100 80GB vRAM

### Creative Commons Attribution

The following CC BY audio was part of the dataset used to train Kokoro v1.0.

| Audio Data | Duration Used | License | Added to Training Set After |
| --- | --- | --- | --- |
| [Koniwa](https://github.com/koniwa/koniwa)`tnc` | <1h | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/deed.ja) | v0.19 / 22 Nov 2024 |
| [SIWIS](https://datashare.ed.ac.uk/handle/10283/2353) | <11h | [CC BY 4.0](https://datashare.ed.ac.uk/bitstream/handle/10283/2353/license_text) | v0.19 / 22 Nov 2024 |

### Acknowledgements

- 🛠️ [@yl4579](https://huggingface.co/yl4579) for architecting StyleTTS 2.
- 🏆 [@Pendrokar](https://huggingface.co/Pendrokar) for adding Kokoro as a contender in the TTS Spaces Arena.
- 📊 Thank you to everyone who contributed synthetic training data.
- ❤️ Special thanks to all compute sponsors.
- 👾 Discord server: [https://discord.gg/QuGxSWBfQy](https://discord.gg/QuGxSWBfQy)
- 🪽 Kokoro is a Japanese word that translates to "heart" or "spirit". It is also the name of an [AI in the Terminator franchise](https://terminator.fandom.com/wiki/Kokoro).

![kokoro](https://static0.gamerantimages.com/wordpress/wp-content/uploads/2024/08/terminator-zero-41-1.jpg)

Downloads last month12,795,347

Inference Providers [NEW](https://huggingface.co/docs/inference-providers)

fal

[Text-to-Speech](https://huggingface.co/tasks/text-to-speech "Learn more about text-to-speech")

Generate

View Code Snippets

Maximize

## Model tree for hexgrad/Kokoro-82M

Base model

[yl4579/StyleTTS2-LJSpeech](https://huggingface.co/yl4579/StyleTTS2-LJSpeech)

Finetuned

( [56](https://huggingface.co/models?other=base_model:finetune:yl4579/StyleTTS2-LJSpeech))

this model

Adapters

[9 models](https://huggingface.co/models?other=base_model:adapter:hexgrad/Kokoro-82M)

Finetunes

[52 models](https://huggingface.co/models?other=base_model:finetune:hexgrad/Kokoro-82M)

Quantizations

[61 models](https://huggingface.co/models?other=base_model:quantized:hexgrad/Kokoro-82M)

## Spaces using hexgrad/Kokoro-82M100

## Papers for hexgrad/Kokoro-82M

[Paper • 2306.07691 •Published Jun 13, 2023• 18](https://huggingface.co/papers/2306.07691)[Paper • 2203.02395 •Published Mar 4, 2022• 2](https://huggingface.co/papers/2203.02395)