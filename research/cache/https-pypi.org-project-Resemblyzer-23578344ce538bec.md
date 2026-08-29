---
url: https://pypi.org/project/Resemblyzer/
title: Resemblyzer · PyPI
retrieved_at: 2026-08-23T00:12:54Z
api_version: v2
---

[Skip to main content](https://pypi.org/project/Resemblyzer/#content) Switch to mobile version

Search PyPISearch

# Resemblyzer 0.1.4

Analyze and compare voices with deep learning

pip install ResemblyzerCopy PIP instructions

Resemblyzer allows you to derive a **high-level representation of a voice** through a deep learning model (referred to as the voice encoder). Given an audio file of speech, it creates a summary vector of 256 values (an embedding, often shortened to "embed" in this repo) that summarizes the characteristics of the voice spoken.

N.B.: this repo holds 100mb of audio data for demonstration purpose. To get [the package](https://pypi.org/project/Resemblyzer/) alone, run `pip install resemblyzer` (python 3.5+ is required).

## Demos

**Speaker diarization**: [\[Demo 02\]](https://github.com/resemble-ai/Resemblyzer/blob/master/demo02_diarization.py) recognize who is talking when with only a few seconds of reference audio per speaker:

_(click the image for a video)_

[![demo_02](https://pypi-camo.freetls.fastly.net/43a6def35dfe2a6286f347deecf0444be73bed3c/68747470733a2f2f692e696d6775722e636f6d2f324d704e6175472e706e67)](https://streamable.com/uef39)

**Fake speech detection**: [\[Demo 05\]](https://github.com/resemble-ai/Resemblyzer/blob/master/demo05_fake_speech_detection.py) modest detection of fake speech by comparing the similarity of 12 unknown utterances (6 real ones, 6 fakes) against ground truth reference audio. Scores above the dashed line are predicted as real, so the model makes one error here.

![demo_05](https://pypi-camo.freetls.fastly.net/84558f11a749c168763867793c4cc9caffd97752/706c6f74732f66616b655f7370656563685f646574656374696f6e2e706e673f7261773d74727565)

For reference, [this](https://www.youtube.com/watch?v=Ho9h0ouemWQ) is the fake video that achieved a high score.

**Visualizing the manifold**:

[\[Demo 03 - left\]](https://github.com/resemble-ai/Resemblyzer/blob/master/demo03_projection.py) projecting the embeddings of 100 utterances (10 each from 10 speakers) in 2D space. The utterances from the same speakers form a tight cluster. With a trivial clustering algorithm, the speaker verification error rate for this example (with data unseen in training) would be 0%.

[\[Demo 04 - right\]](https://github.com/resemble-ai/Resemblyzer/blob/master/demo04_clustering.py) same as demo 03 but with 251 embeddings all from distinct speakers, highlighting that the model has learned on its own to identify the sex of the speaker.

![demo_03_04](https://pypi-camo.freetls.fastly.net/183776ce95584e168c19aa3987c5e44a503fcb97/706c6f74732f616c6c5f636c7573746572696e672e706e673f7261773d74727565)

**Cross-similarity**: [\[Demo 01\]](https://github.com/resemble-ai/Resemblyzer/blob/master/demo01_similarity.py) comparing 10 utterances from 10 speakers against 10 other utterances from the same speakers.

![demo_01](https://pypi-camo.freetls.fastly.net/3f8e1e38c8e00041f6caf6012135b1149e1a6d97/706c6f74732f73696d5f6d61747269785f312e706e673f7261773d74727565)

## What can I do with this package?

Resemblyzer has many uses:

- **Voice similarity metric**: compare different voices and get a value on how similar they sound. This leads to other applications:

  - **Speaker verification**: create a voice profile for a person from a few seconds of speech (5s - 30s) and compare it to that of new audio. Reject similarity scores below a threshold.
  - **Speaker diarization**: figure out who is talking when by comparing voice profiles with the continuous embedding of a multispeaker speech segment.
  - **Fake speech detection**: verify if some speech is legitimate or fake by comparing the similarity of possible fake speech to real speech.
- **High-level feature extraction**: you can use the embeddings generated as feature vectors for machine learning or data analysis. This also leads to other applications:

  - **Voice cloning**: see [this other project](https://github.com/CorentinJ/Real-Time-Voice-Cloning).
  - **Component analysis**: figure out accents, tones, prosody, gender, ... through a component analysis of the embeddings.
  - **Virtual voices**: create entirely new voice embeddings by sampling from a prior distribution.
- **Loss function**: you can backpropagate through the voice encoder model and use it as a perceptual loss for your deep learning model! The voice encoder is written in PyTorch.

Resemblyzer is fast to execute (around 1000x real-time on a GTX 1080, with a minimum of 10ms for I/O operations), and can run both on CPU or GPU. It is robust to noise. It currently works best on English language only, but should still be able to perform somewhat decently on other languages.

## Code example

This is a short example showing how to use Resemblyzer:

```
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

fpath = Path("path_to_an_audio_file")
wav = preprocess_wav(fpath)

encoder = VoiceEncoder()
embed = encoder.embed_utterance(wav)
np.set_printoptions(precision=3, suppress=True)
print(embed)
```

I highly suggest giving a peek to the demos to understand how similarity is computed and to see practical usages of the voice encoder.

## Additional info

Resemblyzer emerged as a side project of the [Real-Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) repository. The pretrained model that comes with Resemblyzer is interchangeable with models trained in that repository, so feel free to finetune a model on new data and possibly new languages! The paper from which the voice encoder was implemented is [Generalized End-To-End Loss for Speaker Verification](https://arxiv.org/pdf/1710.10467.pdf) (in which it is called the _speaker_ encoder).

## Project links

- [Homepage](https://github.com/resemble-ai/Resemblyzer)

## Key dates

PyPI data

Data sourced directly from PyPI's database.

- **Released:** Oct 12, 2023

Latest release

## 1 maintainer

PyPI data

Data sourced directly from PyPI's database.

[![Avatar for CorentinJ from gravatar.com](https://pypi-camo.freetls.fastly.net/7c122a04aea86934f1ce6b0007da909079511c15/68747470733a2f2f7365637572652e67726176617461722e636f6d2f6176617461722f63646531323761646431663136343637386664366330653965636364646131303f73697a653d3335)CorentinJ](https://pypi.org/user/CorentinJ/)

## Credits

**Author:** [Corentin Jemine](mailto:corentin@resemble.ai)

## License

Apache Software License

## Requires

**Python** >=3.5

## Classifiers

- License
  - [OSI Approved :: Apache Software License](https://pypi.org/search/?c=License+%3A%3A+OSI+Approved+%3A%3A+Apache+Software+License)
- Operating System
  - [OS Independent](https://pypi.org/search/?c=Operating+System+%3A%3A+OS+Independent)
- Programming Language
  - [Python :: 3](https://pypi.org/search/?c=Programming+Language+%3A%3A+Python+%3A%3A+3)

[Report project as malware](https://pypi.org/project/Resemblyzer/submit-malware-report/)

## Download files

Download the file for your platform. If you're not sure which to choose, learn more about [installing packages](https://packaging.python.org/tutorials/installing-packages/ "External link").

### Source Distribution

[Resemblyzer-0.1.4.tar.gz](https://files.pythonhosted.org/packages/fb/69/b71d45dca34c51467d0df29154f6eb8c19d4effb95dbbaeef6fc160b29de/Resemblyzer-0.1.4.tar.gz)
(15.7 MB
[view details](https://pypi.org/project/Resemblyzer/#Resemblyzer-0.1.4.tar.gz))


Uploaded Oct 12, 2023`Source`

### Built Distribution

Filter files by name, interpreter, ABI, and platform.

If you're not sure about the file name format, learn more about [wheel file names](https://packaging.python.org/en/latest/specifications/binary-distribution-format/ "External link").

Copy a direct link to the current filters [https://pypi.org/project/Resemblyzer/#files](https://pypi.org/project/Resemblyzer/#files)
Copy

Showing 1 of 1 file.

File name

InterpreterInterpreterpy3

ABIABInone

PlatformPlatformany

[Resemblyzer-0.1.4-py3-none-any.whl](https://files.pythonhosted.org/packages/d9/b5/d897d5de5b2123a1ad5d32e8e007d83f76582f5c32c60992577263bd7c7e/Resemblyzer-0.1.4-py3-none-any.whl)
(15.7 MB
[view details](https://pypi.org/project/Resemblyzer/#Resemblyzer-0.1.4-py3-none-any.whl))


Uploaded Oct 12, 2023`Python 3`

## File details

Details for the file `Resemblyzer-0.1.4.tar.gz`.


### File metadata

- Download URL: [Resemblyzer-0.1.4.tar.gz](https://files.pythonhosted.org/packages/fb/69/b71d45dca34c51467d0df29154f6eb8c19d4effb95dbbaeef6fc160b29de/Resemblyzer-0.1.4.tar.gz)
- Upload date: Oct 12, 2023
- Size: 15.7 MB
- Tags: Source
- Uploaded using Trusted Publishing? No
- Uploaded via: `twine/4.0.2 CPython/3.10.10`

### File hashes

| Algorithm | Hash digest |  |
| --- | --- | --- |
| SHA256 | `ae98d6eaf1def4b91d353347f7d31f10c29e8e109b5196ab29b390293036ad19` | Copy |
| MD5 | `a312bf116f16087f58edbbcc97f45988` | Copy |
| BLAKE2b-256 | `fb69b71d45dca34c51467d0df29154f6eb8c19d4effb95dbbaeef6fc160b29de` | Copy |

Hashes for Resemblyzer-0.1.4.tar.gz

[See more details on using hashes here.](https://pip.pypa.io/en/stable/topics/secure-installs/#hash-checking-mode "External link")

## File details

Details for the file `Resemblyzer-0.1.4-py3-none-any.whl`.


### File metadata

- Download URL: [Resemblyzer-0.1.4-py3-none-any.whl](https://files.pythonhosted.org/packages/d9/b5/d897d5de5b2123a1ad5d32e8e007d83f76582f5c32c60992577263bd7c7e/Resemblyzer-0.1.4-py3-none-any.whl)
- Upload date: Oct 12, 2023
- Size: 15.7 MB
- Tags: Python 3
- Uploaded using Trusted Publishing? No
- Uploaded via: `twine/4.0.2 CPython/3.10.10`

### File hashes

| Algorithm | Hash digest |  |
| --- | --- | --- |
| SHA256 | `8f12eb2f1a9982d32e8db7856de754709b59c93a77bcf0ff536584b619a9dd1f` | Copy |
| MD5 | `48ba5ecfea339f34b46ecdc461f8f063` | Copy |
| BLAKE2b-256 | `d9b5d897d5de5b2123a1ad5d32e8e007d83f76582f5c32c60992577263bd7c7e` | Copy |

Hashes for Resemblyzer-0.1.4-py3-none-any.whl

[See more details on using hashes here.](https://pip.pypa.io/en/stable/topics/secure-installs/#hash-checking-mode "External link")

## Release history[Release notifications](https://pypi.org/help/\#project-release-notifications) \|  [RSS feed](https://pypi.org/rss/project/resemblyzer/releases.xml)

This release

![](https://pypi.org/static/images/blue-cube.572a5bfb.svg)

[0.1.4](https://pypi.org/project/Resemblyzer/0.1.4/) This release

Oct 12, 2023 [2 files](https://pypi.org/project/Resemblyzer/0.1.4/#files)

![](https://pypi.org/static/images/white-cube.2351a86c.svg)

[0.1.3](https://pypi.org/project/Resemblyzer/0.1.3/)

May 8, 2023 [2 files](https://pypi.org/project/Resemblyzer/0.1.3/#files)

![](https://pypi.org/static/images/white-cube.2351a86c.svg)

[Yanked](https://pypi.org/help/#yanked "Release yanked on May 8, 2023")

[0.1.2](https://pypi.org/project/Resemblyzer/0.1.2/)

May 6, 2023 [2 files](https://pypi.org/project/Resemblyzer/0.1.2/#files)

**Yanked reason:** Package files missing

![](https://pypi.org/static/images/white-cube.2351a86c.svg)

[Pre-release](https://pypi.org/help/#pre-release-versioning)

[0.1.1.dev0](https://pypi.org/project/Resemblyzer/0.1.1.dev0/)

Aug 19, 2019 [2 files](https://pypi.org/project/Resemblyzer/0.1.1.dev0/#files)

- English
- español
- français
- 日本語
- português (Brasil)
- українська
- Ελληνικά
- Deutsch
- 中文 (简体)
- 中文 (繁體)
- русский
- עברית
- Esperanto
- 한국어

Supported by

[![](https://pypi-camo.freetls.fastly.net/ed7074cadad1a06f56bc520ad9bd3e00d0704c5b/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f707970692d6173736574732f73706f6e736f726c6f676f732f6177732d77686974652d6c6f676f2d7443615473387a432e706e67)AWS\\
Cloud computing and Security Sponsor](https://aws.amazon.com/) [![](https://pypi-camo.freetls.fastly.net/8855f7c063a3bdb5b0ce8d91bfc50cf851cc5c51/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f707970692d6173736574732f73706f6e736f726c6f676f732f64617461646f672d77686974652d6c6f676f2d6668644c4e666c6f2e706e67)Datadog\\
Monitoring](https://www.datadoghq.com/) [![](https://pypi-camo.freetls.fastly.net/60f709d24f3e4d469f9adc77c65e2f5291a3d165/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f707970692d6173736574732f73706f6e736f726c6f676f732f6465706f742d77686974652d6c6f676f2d7038506f476831302e706e67)Depot\\
Continuous Integration](https://depot.dev/) [![](https://pypi-camo.freetls.fastly.net/df6fe8829cbff2d7f668d98571df1fd011f36192/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f707970692d6173736574732f73706f6e736f726c6f676f732f666173746c792d77686974652d6c6f676f2d65684d3077735f6f2e706e67)Fastly\\
CDN](https://www.fastly.com/) [![](https://pypi-camo.freetls.fastly.net/420cc8cf360bac879e24c923b2f50ba7d1314fb0/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f707970692d6173736574732f73706f6e736f726c6f676f732f676f6f676c652d77686974652d6c6f676f2d616734424e3774332e706e67)Google\\
Download Analytics](https://careers.google.com/) [![](https://pypi-camo.freetls.fastly.net/67af7117035e2345bacb5a82e9aa8b5b3e70701d/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f707970692d6173736574732f73706f6e736f726c6f676f732f73656e7472792d77686974652d6c6f676f2d4a2d6b64742d706e2e706e67)Sentry\\
Error logging](https://sentry.io/for/python/?utm_source=pypi&utm_medium=paid-community&utm_campaign=python-na-evergreen&utm_content=static-ad-pypi-sponsor-learnmore) [![](https://pypi-camo.freetls.fastly.net/b611884ff90435a0575dbab7d9b0d3e60f136466/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f707970692d6173736574732f73706f6e736f726c6f676f732f737461747573706167652d77686974652d6c6f676f2d5467476c6a4a2d502e706e67)StatusPage\\
Status page](https://statuspage.io/)