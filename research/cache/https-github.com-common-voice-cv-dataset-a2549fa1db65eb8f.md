---
url: https://github.com/common-voice/cv-dataset
title: GitHub - common-voice/cv-dataset: Metadata and versioning details for the Common Voice dataset · GitHub
retrieved_at: 2026-08-22T19:12:18Z
api_version: v2
---

[Skip to content](https://github.com/common-voice/cv-dataset#start-of-content)

You signed in with another tab or window. [Reload](https://github.com/common-voice/cv-dataset) to refresh your session.You signed out in another tab or window. [Reload](https://github.com/common-voice/cv-dataset) to refresh your session.You switched accounts on another tab or window. [Reload](https://github.com/common-voice/cv-dataset) to refresh your session.Dismiss alert

{{ message }}

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/common-voice/cv-dataset).

[common-voice](https://github.com/common-voice)/ **[cv-dataset](https://github.com/common-voice/cv-dataset)** Public

- [Notifications](https://github.com/login?return_to=%2Fcommon-voice%2Fcv-dataset) You must be signed in to change notification settings
- [Fork\\
20](https://github.com/login?return_to=%2Fcommon-voice%2Fcv-dataset)
- [Star\\
174](https://github.com/login?return_to=%2Fcommon-voice%2Fcv-dataset)


main

[**16** Branches](https://github.com/common-voice/cv-dataset/branches) [**0** Tags](https://github.com/common-voice/cv-dataset/tags)

[Go to Branches page](https://github.com/common-voice/cv-dataset/branches)[Go to Tags page](https://github.com/common-voice/cv-dataset/tags)

Go to file

Code

Open more actions menu

## Latest commit

[![moz-bozden](https://avatars.githubusercontent.com/u/207743035?v=4&size=40)](https://github.com/moz-bozden)[moz-bozden](https://github.com/common-voice/cv-dataset/commits?author=moz-bozden)

[Merge pull request](https://github.com/common-voice/cv-dataset/commit/f99d8239d2796131b73ac99f92ee7cb4443bf3ba) [#52](https://github.com/common-voice/cv-dataset/pull/52) [from common-voice/1750-add-scs-v26-and-sps-v4-…](https://github.com/common-voice/cv-dataset/commit/f99d8239d2796131b73ac99f92ee7cb4443bf3ba)

Open commit details

2 months agoJun 16, 2026

[f99d823](https://github.com/common-voice/cv-dataset/commit/f99d8239d2796131b73ac99f92ee7cb4443bf3ba) · 2 months agoJun 16, 2026

## History

[133 Commits](https://github.com/common-voice/cv-dataset/commits/main/)

Open commit details

[View commit history for this file.](https://github.com/common-voice/cv-dataset/commits/main/) 133 Commits

## Folders and files

| Name | Name | Last commit message | Last commit date |
| --- | --- | --- | --- |
| [datasets](https://github.com/common-voice/cv-dataset/tree/main/datasets "datasets") | [datasets](https://github.com/common-voice/cv-dataset/tree/main/datasets "datasets") | [Update Spontaneous Speech documentation for v4.0](https://github.com/common-voice/cv-dataset/commit/3e15d70e551940af171c44c752a5c1264606041a "Update Spontaneous Speech documentation for v4.0") | 2 months agoJun 16, 2026 |
| [helpers](https://github.com/common-voice/cv-dataset/tree/main/helpers "helpers") | [helpers](https://github.com/common-voice/cv-dataset/tree/main/helpers "helpers") | [fix: scripts omitting new locales from totals](https://github.com/common-voice/cv-dataset/commit/a231b5582f818c3fb76b2ff6897c2a0d1fe3b0a7 "fix: scripts omitting new locales from totals") | 5 months agoMar 26, 2026 |
| [CHANGELOG.md](https://github.com/common-voice/cv-dataset/blob/main/CHANGELOG.md "CHANGELOG.md") | [CHANGELOG.md](https://github.com/common-voice/cv-dataset/blob/main/CHANGELOG.md "CHANGELOG.md") | [Update per reviews](https://github.com/common-voice/cv-dataset/commit/be1fbea96ff4ebe2407d43782fda7dfe39d7db2c "Update per reviews") | 5 months agoMar 26, 2026 |
| [LICENSE](https://github.com/common-voice/cv-dataset/blob/main/LICENSE "LICENSE") | [LICENSE](https://github.com/common-voice/cv-dataset/blob/main/LICENSE "LICENSE") | [Initial commit](https://github.com/common-voice/cv-dataset/commit/c6769d9be71458f7d98b869737399f1df6263bb2 "Initial commit") | 6 years agoJul 16, 2020 |
| [README.md](https://github.com/common-voice/cv-dataset/blob/main/README.md "README.md") | [README.md](https://github.com/common-voice/cv-dataset/blob/main/README.md "README.md") | [Update dataset overview for the new releases](https://github.com/common-voice/cv-dataset/commit/502d68980bd2b0a84e170febc99847ef1f5922e6 "Update dataset overview for the new releases") | 2 months agoJun 16, 2026 |
| View all files |

## Repository files navigation

# Common Voice Datasets

[Permalink: Common Voice Datasets](https://github.com/common-voice/cv-dataset#common-voice-datasets)

This repo contains release details and metadata for the [Common Voice](https://commonvoice.mozilla.org/) datasets. Please visit the [Mozilla Data Collective Common Voice section](https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk) to download the latest datasets.

## Dataset Types

[Permalink: Dataset Types](https://github.com/common-voice/cv-dataset#dataset-types)

Common Voice collects voice data through multiple modalities. Each dataset type has its own release information, data structure, and documentation.

| Type | Alias | Status | Releases | Latest (2026-06) | Languages |
| --- | --- | --- | --: | :-: | --: |
| [Scripted Speech](https://github.com/common-voice/cv-dataset/blob/main/datasets/scripted-speech) | SCS | Active | 26 | v26.0 | 294 |
| [Spontaneous Speech](https://github.com/common-voice/cv-dataset/blob/main/datasets/spontaneous-speech) | SPS | Active | 4 | v4.0 | 78 |
| [Code Switching](https://github.com/common-voice/cv-dataset/blob/main/datasets/code-switching) | CS | Alpha | -- | -- | -- |

See each dataset type's documentation for detailed information about data structures, fields in metadata files (`.tsv`), archive contents, and release changelogs. Note that the "date" in releases represents the cut-off date for data collection and validation, not the actual release date of the dataset.

## Data Pipeline

[Permalink: Data Pipeline](https://github.com/common-voice/cv-dataset#data-pipeline)

Render

Loading

```
flowchart LR
    subgraph SCS["Scripted Speech (SCS)"]
        SCS_DB[("DB")]
        SCS_GCS["GCS"]
    end
    subgraph SCS_BUN["SCS Bundler"]
        CC["CorporaCreator"]
    end
    subgraph SCS_BUN2["SCS Bundler"]
      UP["Uploader"]
    end

    DSH["cv-datasheets"]

    subgraph SPS["Spontaneous Speech (SPS)"]
        SPS_DB[("DB")]
        SPS_GCS["GCS"]
    end
    subgraph SPS_BUN["SPS Bundler"]
        QA["QA Pipeline"]
    end

    BUN_GCS["GCS\
    datasets\
    datasheets\
    stats"]

    MDC[["MDC\
    downloads"]]
    CDS[["cv-dataset ◀"]]

    SCS_DB -->|data| SCS_BUN
    SCS_GCS -->|clips| SCS_BUN
    DSH -->|JSON| SCS_BUN
    DSH -->|JSON| SPS_BUN
    SPS_DB -->|data| SPS_BUN
    SPS_GCS -->|audio| SPS_BUN
    SCS_BUN --> BUN_GCS
    SPS_BUN --> BUN_GCS
    BUN_GCS -->|datasets| UP
    BUN_GCS -->|datasheets| UP
    UP -->|API| MDC
    BUN_GCS -->|stats| CDS

    style CDS fill:#1a73e8,color:#ffffff,stroke:#1558b0,stroke-width:2px
```

## Overview

[Permalink: Overview](https://github.com/common-voice/cv-dataset#overview)

### Scripted Speech (SCS)

[Permalink: Scripted Speech (SCS)](https://github.com/common-voice/cv-dataset#scripted-speech-scs)

Render

Scripted Speech: Total & Validated Hours12345.16.178910111213141516.1171819202122232425264000035000300002500020000150001000050000Hours

Loading

```
---
config:
    xyChart:
        width: 900
        height: 400
    themeVariables:
        xyChart:
            plotColorPalette: "#1a73e8, #34a853"
---
xychart-beta
    title "Scripted Speech: Total & Validated Hours"
    x-axis ["1","2","3","4","5.1","6.1","7","8","9","10","11","12","13","14","15","16.1","17","18","19","20","21","22","23","24","25","26"]
    y-axis "Hours" 0 --> 43000
    bar [1368,2366,2454,4257,7226,9283,13905,18243,20217,20817,24231,26119,27141,28117,28750,30328,31175,32121,32584,33154,33534,33815,35921,38932,41792,42388]
    bar [1096,1872,1979,3401,5671,7335,11192,14122,14973,15234,16429,17127,17689,18651,19159,19915,20408,20943,21593,22106,22344,22640,24600,25886,28377,28893]
```

For details see: [Scripted Speech documentation](https://github.com/common-voice/cv-dataset/blob/main/datasets/scripted-speech)

### Spontaneous Speech (SPS)

[Permalink: Spontaneous Speech (SPS)](https://github.com/common-voice/cv-dataset#spontaneous-speech-sps)

Render

Spontaneous Speech: Total vs Validated Hoursv1.0v2.0v3.0v4.0600550500450400350300250200150100500Hours

Loading

```
---
config:
    xyChart:
        width: 600
        height: 350
    themeVariables:
        xyChart:
            plotColorPalette: "#1a73e8, #34a853"
---
xychart-beta
    title "Spontaneous Speech: Total vs Validated Hours"
    x-axis ["v1.0","v2.0","v3.0","v4.0"]
    y-axis "Hours" 0 --> 600
    bar [428,454,508,515]
    bar [263,268,269,273]
```

For details see: [Spontaneous Speech documentation](https://github.com/common-voice/cv-dataset/blob/main/datasets/spontaneous-speech)

## Dataset Access

[Permalink: Dataset Access](https://github.com/common-voice/cv-dataset#dataset-access)

You can download the Common Voice datasets from the [Mozilla Data Collective](https://mozilladatacollective.com/) (MDC) platform:

- [Directly from the browser](https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk)
- [Using the MDC API](https://mozilladatacollective.com/api-reference)
- [Using the MDC Python SDK](https://github.com/Mozilla-Data-Collective/datacollective-python) to directly load the datasets as pandas DataFrame in your Python codebase

## Generating Dataset Statistics

[Permalink: Generating Dataset Statistics](https://github.com/common-voice/cv-dataset#generating-dataset-statistics)

Helper scripts are available in the [helpers/](https://github.com/common-voice/cv-dataset/blob/main/helpers) directory for processing bundler output into dataset statistics. See [helpers/README.md](https://github.com/common-voice/cv-dataset/blob/main/helpers/README.md) for detailed usage and examples.

All helper scripts support multiple dataset types via the first argument:

```
node helpers/createStats.js <dataset-type> <stats-folder>
node helpers/compareReleases.js <dataset-type> <dataset-1> <dataset-2>
node helpers/createDeltaStatistics.js <dataset-type> <dataset-1> <dataset-2>
node helpers/recalculateStats.js <dataset-type> <dataset>
```

## Citation

[Permalink: Citation](https://github.com/common-voice/cv-dataset#citation)

If you use the data in a published academic work we would appreciate if you cite the following article:

- Ardila, R., Branson, M., Davis, K., Henretty, M., Kohler, M., Meyer, J., Morais, R., Saunders, L., Tyers, F. M. and Weber, G. (2020) " [Common Voice: A Massively-Multilingual Speech Corpus](https://arxiv.org/abs/1912.06670)". _Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)._ pp. 4211--4215

```
@inproceedings{commonvoice:2020,
  author = {Ardila, R. and Branson, M. and Davis, K. and Henretty, M. and Kohler, M. and Meyer, J. and Morais, R. and Saunders, L. and Tyers, F. M. and Weber, G.},
  title = {Common Voice: A Massively-Multilingual Speech Corpus},
  booktitle = {Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)},
  pages = {4211--4215},
  year = 2020
}
```

## Feedback

[Permalink: Feedback](https://github.com/common-voice/cv-dataset#feedback)

Please only use this repo to provide feedback on **technical issues** with the dataset, such as file corruptions, problems with the partitions, and so on. For more expansive discussions, please join us in [Discourse](https://discourse.mozilla.org/c/voice) or [Matrix](https://chat.mozilla.org/#/room/#common-voice:mozilla.org).

## About

Metadata and versioning details for the Common Voice dataset

[commonvoice.mozilla.org/datasets](https://commonvoice.mozilla.org/datasets)

### Topics

[asr](https://github.com/topics/asr) [dataset](https://github.com/topics/dataset) [open-data](https://github.com/topics/open-data) [open-datasets](https://github.com/topics/open-datasets) [speech-recognition](https://github.com/topics/speech-recognition) [voice](https://github.com/topics/voice)

### Resources

[Readme](https://github.com/common-voice/cv-dataset#readme-ov-file)

[MPL-2.0 license](https://github.com/common-voice/cv-dataset#MPL-2.0-1-ov-file)

[Activity](https://github.com/common-voice/cv-dataset/activity)

[Custom properties](https://github.com/common-voice/cv-dataset/custom-properties)

### Stars

**174** stars

### Watchers

**14** watching

### Forks

[**20** forks](https://github.com/common-voice/cv-dataset/forks)

[Report repository](https://github.com/contact/report-content?content_url=https%3A%2F%2Fgithub.com%2Fcommon-voice%2Fcv-dataset&report=common-voice+%28user%29)

## [Contributors](https://github.com/common-voice/cv-dataset/graphs/contributors) 6 (6)

- [![@moz-bozden](https://avatars.githubusercontent.com/u/207743035?s=64&v=4)](https://github.com/moz-bozden)
- [![@moz-dfeller](https://avatars.githubusercontent.com/u/119537889?s=64&v=4)](https://github.com/moz-dfeller)
- [![@mozgzh](https://avatars.githubusercontent.com/u/97530463?s=64&v=4)](https://github.com/mozgzh)
- [![@phirework](https://avatars.githubusercontent.com/u/582683?s=64&v=4)](https://github.com/phirework)
- [![@HarikalarKutusu](https://avatars.githubusercontent.com/u/8849617?s=64&v=4)](https://github.com/HarikalarKutusu)
- [![@Kostis-S-Z](https://avatars.githubusercontent.com/u/9952015?s=64&v=4)](https://github.com/Kostis-S-Z)

## Languages

- [JavaScript100%](https://github.com/common-voice/cv-dataset/search?l=javascript)

You can’t perform that action at this time.