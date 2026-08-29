---
url: https://huggingface.co/facebook/wav2vec2-base-960h
title: facebook/wav2vec2-base-960h · Hugging Face
retrieved_at: 2026-08-22T16:50:07Z
api_version: v2
---

# Wav2Vec2-Base-960h

[Facebook's Wav2Vec2](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)

The base model pretrained and fine-tuned on 960 hours of Librispeech on 16kHz sampled speech audio. When using the model
make sure that your speech input is also sampled at 16Khz.

[Paper](https://arxiv.org/abs/2006.11477)

Authors: Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli

**Abstract**

We show for the first time that learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler. wav2vec 2.0 masks the speech input in the latent space and solves a contrastive task defined over a quantization of the latent representations which are jointly learned. Experiments using all labeled data of Librispeech achieve 1.8/3.3 WER on the clean/other test sets. When lowering the amount of labeled data to one hour, wav2vec 2.0 outperforms the previous state of the art on the 100 hour subset while using 100 times less labeled data. Using just ten minutes of labeled data and pre-training on 53k hours of unlabeled data still achieves 4.8/8.2 WER. This demonstrates the feasibility of speech recognition with limited amounts of labeled data.

The original model can be found under [https://github.com/pytorch/fairseq/tree/master/examples/wav2vec#wav2vec-20](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec#wav2vec-20).

# Usage

To transcribe audio files the model can be used as a standalone acoustic model as follows:

```python
 from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
 from datasets import load_dataset
 import torch

 # load model and tokenizer
 processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
 model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

 # load dummy dataset and read soundfiles
 ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

 # tokenize
 input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest").input_values  # Batch size 1

 # retrieve logits
 logits = model(input_values).logits

 # take argmax and decode
 predicted_ids = torch.argmax(logits, dim=-1)
 transcription = processor.batch_decode(predicted_ids)
```

## Evaluation

This code snippet shows how to evaluate **facebook/wav2vec2-base-960h** on LibriSpeech's "clean" and "other" test data.

```python
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from jiwer import wer

librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def map_to_pred(batch):
    input_values = processor(batch["audio"]["array"], return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch

result = librispeech_eval.map(map_to_pred, batched=True, batch_size=1, remove_columns=["audio"])

print("WER:", wer(result["text"], result["transcription"]))
```

_Result (WER)_:

| "clean" | "other" |
| --- | --- |
| 3.4 | 8.6 |

Downloads last month1,806,151

Safetensors

Model size

94.4M params

Tensor type

F32

·

Files info

Inference Providers [NEW](https://huggingface.co/docs/inference-providers)

[Automatic Speech Recognition](https://huggingface.co/tasks/automatic-speech-recognition "Learn more about automatic-speech-recognition")

This model isn't deployed by any Inference Provider. [🙋Ask for provider support](https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/new?title=facebook/wav2vec2-base-960h&description=React%20to%20this%20comment%20with%20an%20emoji%20to%20vote%20for%20%5Bfacebook%2Fwav2vec2-base-960h%5D(%2Ffacebook%2Fwav2vec2-base-960h)%20to%20be%20supported%20by%20Inference%20Providers.%0A%0A(optional)%20Which%20providers%20are%20you%20interested%20in%3F%20(Novita%2C%20Hyperbolic%2C%20Together%E2%80%A6)%0A)

## Model tree for facebook/wav2vec2-base-960h

Adapters

[1 model](https://huggingface.co/models?other=base_model:adapter:facebook/wav2vec2-base-960h)

Finetunes

[177 models](https://huggingface.co/models?other=base_model:finetune:facebook/wav2vec2-base-960h)

Quantizations

[7 models](https://huggingface.co/models?other=base_model:quantized:facebook/wav2vec2-base-960h)

## Dataset used to train facebook/wav2vec2-base-960h

## Spaces using facebook/wav2vec2-base-960h100

## Collection including facebook/wav2vec2-base-960h

[A collection for the first release of Wav2Vec 2.0, a speech encoder that learns powerful representations from unlabelled audio data.•8 items•Updated Jan 16, 2024• 31](https://huggingface.co/collections/facebook/wav2vec-20)

## Paper for facebook/wav2vec2-base-960h

[Paper • 2006.11477 •Published Jun 20, 2020• 10](https://huggingface.co/papers/2006.11477)

## Evaluation results

- [hf-audio/open-asr-leaderboard](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard)[leaderboard](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard?eval_result=facebook/wav2vec2-base-960h)
- Mean Wer [View evaluation results](https://huggingface.co/facebook/wav2vec2-base-960h/discussions/19) [![](https://cdn-avatars.huggingface.co/v1/production/uploads/61b85ce86eb1f2c5e6233736/pCka87yd0Yvl2JzI5h2Oq.png)\\
source](https://huggingface.co/hf-audio)


29.4

- Rtfx [View evaluation results](https://huggingface.co/facebook/wav2vec2-base-960h/discussions/19) [![](https://cdn-avatars.huggingface.co/v1/production/uploads/61b85ce86eb1f2c5e6233736/pCka87yd0Yvl2JzI5h2Oq.png)\\
source](https://huggingface.co/hf-audio)


686

- Ami Wer [View evaluation results](https://huggingface.co/facebook/wav2vec2-base-960h/discussions/19) [![](https://cdn-avatars.huggingface.co/v1/production/uploads/61b85ce86eb1f2c5e6233736/pCka87yd0Yvl2JzI5h2Oq.png)\\
source](https://huggingface.co/hf-audio)


45.56

- +7 more

- Test WERon LibriSpeech (clean)
[test set](https://huggingface.co/facebook/wav2vec2-base-960h/blob/main/README.md) [self-reported](https://huggingface.co/facebook/wav2vec2-base-960h/blob/main/README.md)


3.400

- Test WERon LibriSpeech (other)
[test set](https://huggingface.co/facebook/wav2vec2-base-960h/blob/main/README.md) [self-reported](https://huggingface.co/facebook/wav2vec2-base-960h/blob/main/README.md)


8.600


This is not a dataset id from hf.co/datasets