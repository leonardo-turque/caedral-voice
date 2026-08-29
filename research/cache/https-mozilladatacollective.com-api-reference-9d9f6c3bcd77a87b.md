---
url: https://mozilladatacollective.com/api-reference
title: API Reference | Mozilla Data Collective
retrieved_at: 2026-08-22T19:16:04Z
api_version: v2
---

## Harness community-driven datasets with our API

![](https://mozilladatacollective.com/_next/static/media/explosion-brown.046ia-djylexz.svg)

**Version:** Beta

The Mozilla Data Collective API gives developers access to community-created datasets while empowering contributors to maintain control over their data.

[![](https://mozilladatacollective.com/_next/static/media/api-key.1k-nylukp80gp.svg)\\
Get API Access](https://mozilladatacollective.com/profile/credentials) [Browse API Docs](https://mozilladatacollective.com/api-reference/docs)

### Mozilla Data Collective API at a glance

![](https://mozilladatacollective.com/_next/static/media/bolt.14eirkihck0vm.svg)

###### Create access credentials

Manage your API credentials by going to Profile > API

![](https://mozilladatacollective.com/_next/static/media/shield.413c7dxnbq6s3.svg)

###### Secure your key

Store your access credentials in a secret key

![](https://mozilladatacollective.com/_next/static/media/union.0gvnfqbmqbcpx.svg)

###### Authentication

Provide your API key in your request header to authenticate

![](https://mozilladatacollective.com/_next/static/media/globe.1qhamcbucs5ke.svg)

###### Select your dataset

Choose from over 300 global datasets to use

![](https://mozilladatacollective.com/_next/static/media/check-box.2s1gvgf8iffo2.svg)

###### Agree to dataset terms

You will only be able to download datasets after accepting terms

![](https://mozilladatacollective.com/_next/static/media/arrow.35eiao015-5zr.svg)

###### Download

Use our REST endpoint or the MDC python library to get started

[Create API credentials](https://mozilladatacollective.com/profile/credentials)

### API Overview

Power your projects with diverse, ethically-created datasets that are just one REST call away.

[![](https://mozilladatacollective.com/_next/static/media/api-key.1k-nylukp80gp.svg)\\
Get API Access](https://mozilladatacollective.com/profile/credentials) [Browse API Docs](https://mozilladatacollective.com/api-reference/docs)

![](https://mozilladatacollective.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fagave-plant.0r16-i1xpmvao.png&w=1080&q=75)

copy

```
# Test code for connecting to Mozilla Data Collective API

from datacollective import download_dataset
from datacollective import load_dataset

# Download a dataset to your computer
dataset_path = download_dataset("your-dataset-id")

# Or load it directly in a Pandas DataFrame
# (if supported)
dataset = load_dataset("your-dataset-id")
```

### Give it a try

Get up and running with datacollective-python, a Python library for authenticating and interacting with the MDC API.

![](https://mozilladatacollective.com/_next/static/media/api-key.1k-nylukp80gp.svg)[Python Library](https://github.com/Mozilla-Data-Collective/datacollective-python)

### Links & Docs

[Get API Access](https://mozilladatacollective.com/profile/credentials) [Browse API Docs](https://mozilladatacollective.com/api-reference/docs) [Python Library](https://github.com/Mozilla-Data-Collective/datacollective-python)