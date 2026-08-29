---
url: https://k2-fsa.github.io/icefall/recipes/TTS/vctk/vits.html
title: VITS-VCTK — icefall 0.1 documentation
retrieved_at: 2026-08-22T19:58:40Z
api_version: v2
---

- [Home](https://k2-fsa.github.io/icefall/index.html)
- [Recipes](https://k2-fsa.github.io/icefall/recipes/index.html)
- [TTS](https://k2-fsa.github.io/icefall/recipes/TTS/index.html)
- VITS-VCTK
- [Edit on GitHub](https://github.com/k2-fsa/icefall/blob/master/docs/source/recipes/TTS/vctk/vits.rst)

* * *

# VITS-VCTK [](https://k2-fsa.github.io/icefall/recipes/TTS/vctk/vits.html\#vits-vctk "Permalink to this heading")

This tutorial shows you how to train an VITS model
with the [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) dataset.

Note

TTS related recipes require packages in `requirements-tts.txt`.

Note

The VITS paper: [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/pdf/2106.06103.pdf)

## Data preparation [](https://k2-fsa.github.io/icefall/recipes/TTS/vctk/vits.html\#data-preparation "Permalink to this heading")

```
$ cd egs/vctk/TTS
$ ./prepare.sh
```

To run stage 1 to stage 6, use

```
$ ./prepare.sh --stage 1 --stop_stage 6
```

## Build Monotonic Alignment Search [](https://k2-fsa.github.io/icefall/recipes/TTS/vctk/vits.html\#build-monotonic-alignment-search "Permalink to this heading")

To build the monotonic alignment search, use the following commands:

```
$ ./prepare.sh --stage -1 --stop_stage -1
```

or

```
$ cd vits/monotonic_align
$ python setup.py build_ext --inplace
$ cd ../../
```

## Training [](https://k2-fsa.github.io/icefall/recipes/TTS/vctk/vits.html\#training "Permalink to this heading")

```
$ export CUDA_VISIBLE_DEVICES="0,1,2,3"
$ ./vits/train.py \
    --world-size 4 \
    --num-epochs 1000 \
    --start-epoch 1 \
    --use-fp16 1 \
    --exp-dir vits/exp \
    --tokens data/tokens.txt
    --max-duration 350
```

Note

You can adjust the hyper-parameters to control the size of the VITS model and
the training configurations. For more details, please run `./vits/train.py --help`.

Note

The training can take a long time (usually a couple of days).

Training logs, checkpoints and tensorboard logs are saved in `vits/exp`.

## Inference [](https://k2-fsa.github.io/icefall/recipes/TTS/vctk/vits.html\#inference "Permalink to this heading")

The inference part uses checkpoints saved by the training part, so you have to run the
training part first. It will save the ground-truth and generated wavs to the directory
`vits/exp/infer/epoch-*/wav`, e.g., `vits/exp/infer/epoch-1000/wav`.

```
$ export CUDA_VISIBLE_DEVICES="0"
$ ./vits/infer.py \
    --epoch 1000 \
    --exp-dir vits/exp \
    --tokens data/tokens.txt \
    --max-duration 500
```

Note

For more details, please run `./vits/infer.py --help`.

## Export models [](https://k2-fsa.github.io/icefall/recipes/TTS/vctk/vits.html\#export-models "Permalink to this heading")

Currently we only support ONNX model exporting. It will generate two files in the given `exp-dir`:
`vits-epoch-*.onnx` and `vits-epoch-*.int8.onnx`.

```
$ ./vits/export-onnx.py \
    --epoch 1000 \
    --exp-dir vits/exp \
    --tokens data/tokens.txt
```

You can test the exported ONNX model with:

```
$ ./vits/test_onnx.py \
    --model-filename vits/exp/vits-epoch-1000.onnx \
    --tokens data/tokens.txt
```

## Download pretrained models [](https://k2-fsa.github.io/icefall/recipes/TTS/vctk/vits.html\#download-pretrained-models "Permalink to this heading")

If you don’t want to train from scratch, you can download the pretrained models
by visiting the following link:

> - [https://huggingface.co/zrjin/icefall-tts-vctk-vits-2023-12-05](https://huggingface.co/zrjin/icefall-tts-vctk-vits-2023-12-05)