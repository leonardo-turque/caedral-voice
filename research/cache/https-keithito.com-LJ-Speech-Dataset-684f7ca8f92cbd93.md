---
url: https://keithito.com/LJ-Speech-Dataset/
title: The LJ Speech Dataset
retrieved_at: 2026-08-22T19:12:12Z
api_version: v2
---

Latest Version

1.1

License

[Public Domain](https://librivox.org/pages/public-domain/)

Download Size

2.6 GB

# The LJ Speech Dataset

This is a public domain speech dataset consisting of 13,100 short audio clips
of a single speaker reading passages from 7 non-fiction books. A transcription is provided for each clip.
Clips vary in length from 1 to 10 seconds and have a total length of approximately
24 hours.


The texts were published between 1884 and 1964, and are in the public domain. The audio was
recorded in 2016-17 by the [LibriVox](https://librivox.org/) project and is also in the
public domain.


[Download Dataset (2.6 GB)](https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2)

## Sample Data

The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired,

Many animals of even complex structure which live parasitically within others are wholly devoid of an alimentary cavity.

## File Format

Metadata is provided in **transcripts.csv**. This file consists of one record
per line, delimited by the pipe character (0x7c). The fields are:


1. **ID**: this is the name of the corresponding .wav file
2. **Transcription**: words spoken by the reader (UTF-8)
3. **Normalized Transcription**: transcription with numbers, ordinals, and monetary units
    expanded into full words (UTF-8).

Each audio file is a single-channel 16-bit PCM WAV with a sample rate of 22050 Hz.


## Statistics

|     |     |
| --- | --- |
| **Total Clips** | 13,100 |
| **Total Words** | 225,715 |
| **Total Characters** | 1,308,678 |
| **Total Duration** | 23:55:17 |
| **Mean Clip Duration** | 6.57 sec |
| **Min Clip Duration** | 1.11 sec |
| **Max Clip Duration** | 10.10 sec |
| **Mean Words per Clip** | 17.23 |
| **Distinct Words** | 13,821 |

## Miscellaneous

- The audio clips range in length from approximately 1 second to 10 seconds.
They were segmented automatically based on silences in the recording.
Clip boundaries generally align with sentence or clause boundaries, but
not always.

- The text was matched to the audio manually, and a QA pass was done to
ensure that the text accurately matched the words spoken in the audio.

- The original LibriVox recordings were distributed as
128 kbps MP3 files. As a result, they may contain artifacts introduced by
the MP3 encoding.

- The following abbreviations appear in the text. They may be expanded as follows:



| Abbreviation | Expansion |
| --- | --- |
| Mr. | Mister |
| Mrs. | Misess (\*) |
| Dr. | Doctor |
| No. | Number |
| St. | Saint |
| Co. | Company |
| Jr. | Junior |
| Maj. | Major |
| Gen. | General |
| Drs. | Doctors |
| Rev. | Reverend |
| Lt. | Lieutenant |
| Hon. | Honorable |
| Sgt. | Sergeant |
| Capt. | Captain |
| Esq. | Esquire |
| Ltd. | Limited |
| Col. | Colonel |
| Ft. | Fort |


(\*) there's no standard expansion for "Mrs."

- 19 of the transcriptions contain non-ASCII characters (for example, LJ016-0257 contains
"raison d'être").


- Example code using this dataset to train a speech synthesis model can be found at:
[github.com/keithito/tacotron](https://github.com/keithito/tacotron).


- For more information or to report errors, please email kito@kito.us.

## Changelog

- [1.1](https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2) (current release)


_Version 1.0 included 30 .wav files without corresponding_
_annotations in metadata.csv. These have been removed in version 1.1. Thanks to Rafael_
_Valle for spotting this._
- [1.0](https://data.keithito.com/data/speech/LJSpeech-1.0.tar.bz2)

_Initial release_

## License

This dataset is in the public domain in the US (and most likely other countries as well).
There are no restrictions on its use. For more information, please see:
[librivox.org/pages/public-domain](https://librivox.org/pages/public-domain).


## Credits

This dataset consists of excerpts from the following works:

- Morris, William, et al. _Arts and Crafts Essays_. 1893.
- Griffiths, Arthur. _The Chronicles of Newgate, Vol. 2_. 1884.
- Roosevelt, Franklin D. _The Fireside Chats of Franklin Delano Roosevelt_. 1933-42.
- Harland, Marion. _Marion Harland's Cookery for Beginners_. 1893.
- Rolt-Wheeler, Francis. _The Science - History of the Universe, Vol. 5: Biology_. 1910.
- Banks, Edgar J. _The Seven Wonders of the Ancient World_. 1916.
- President's Commission on the Assassination of President Kennedy. _Report_
_of the President's Commission on the Assassination of President Kennedy_. 1964.

Recordings by Linda Johnson from [LibriVox](https://librivox.org/).
Alignment and annotation by [Keith Ito](https://keithito.com/).
All text, audio, and annotations are in the public domain.
We request that you use this dataset for good and not evil.


As this work is in the public domain, you may use it without attribution.
However, if you'd like to cite it in a publication, please do so by linking to this page or using the following:


@misc{ljspeech17,
author = {Keith Ito and Linda Johnson},
title = {The LJ Speech Dataset},
howpublished = {\\url{https://keithito.com/LJ-Speech-Dataset/}},
year = 2017
}