---
url: https://github.com/coqui-ai/TTS/discussions/2615
title: Training VITS on clean dataset for 70k creates beautiful...gibberish · coqui-ai/TTS · Discussion #2615 · GitHub
retrieved_at: 2026-08-22T19:53:34Z
api_version: v2
---

[Skip to content](https://github.com/coqui-ai/TTS/discussions/2615#start-of-content)

You signed in with another tab or window. [Reload](https://github.com/coqui-ai/TTS/discussions/2615) to refresh your session.You signed out in another tab or window. [Reload](https://github.com/coqui-ai/TTS/discussions/2615) to refresh your session.You switched accounts on another tab or window. [Reload](https://github.com/coqui-ai/TTS/discussions/2615) to refresh your session.Dismiss alert

{{ message }}

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/2615).

[coqui-ai](https://github.com/coqui-ai)/ **[TTS](https://github.com/coqui-ai/TTS)** Public

- [Notifications](https://github.com/login?return_to=%2Fcoqui-ai%2FTTS) You must be signed in to change notification settings
- [Fork\\
6.1k](https://github.com/login?return_to=%2Fcoqui-ai%2FTTS)
- [Star\\
45.9k](https://github.com/login?return_to=%2Fcoqui-ai%2FTTS)


# Training VITS on clean dataset for 70k creates beautiful...gibberish  \#2615

[Answered](https://github.com/coqui-ai/TTS/discussions/2615#discussioncomment-5937525) by [r-dh](https://github.com/r-dh "r-dh")

[r-dh](https://github.com/r-dh)

asked this question in
[General Q&A](https://github.com/coqui-ai/TTS/discussions/categories/general-q-a)

[Training VITS on clean dataset for 70k creates beautiful...gibberish](https://github.com/coqui-ai/TTS/discussions/2615#top)#2615

[![@r-dh](https://avatars.githubusercontent.com/u/14875684?s=40&v=4)\\
r-dh](https://github.com/r-dh)

on May 15, 2023May 15, 2023·
1 comment


[Answered](https://github.com/coqui-ai/TTS/discussions/2615#discussioncomment-5937525)by [r-dh](https://github.com/r-dh "r-dh")[Return to top](https://github.com/coqui-ai/TTS/discussions/2615#top)

Discussion options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/2615).

# {{title}}

Quote reply

edited

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/2615).

# {{editor}}'s edit

{{actor}} deleted this content
.

# {{editor}}'s edit

## [![](https://avatars.githubusercontent.com/u/14875684?s=64&v=4)\ r-dh](https://github.com/r-dh) [on May 15, 2023May 15, 2023](https://github.com/coqui-ai/TTS/discussions/2615\#discussion-5197072)

|     |
| --- |
| Hi<br>A couple of years ago I released a [Dutch dataset](https://github.com/r-dh/dutch-vl-tts) of 15k samples which I achieved good results with using Tacotron 2.<br>Now I wanted to test the "new" VITS model on the same dataset, and I trained it for 70k steps. The outcome is definitely of good quality, but it sounds like some keyboard smashing is pronounced. Every time I run the following command:<br>```<br> !tts --text "Het is maandag vijftien mei." \<br>      --model_path $test_ckpt \<br>      --config_path $test_config \<br>      --out_path out.wav<br>```<br>It generates different sounding gibberish, so I'm really wondering what could be going on here.<br>I used the following configuration:<br>```<br>config = VitsConfig(<br>    batch_size=32,<br>    eval_batch_size=16,<br>    num_loader_workers=4, <br>    num_eval_loader_workers=4, <br>    run_eval=True,<br>    test_delay_epochs=-1,<br>    epochs=1000,<br>    text_cleaner="phoneme_cleaners",<br>    use_phonemes=True,<br>    phoneme_language="nl", <br>    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),<br>    print_step=25,<br>    print_eval=False,<br>    mixed_precision=True,<br>    output_path=output_path,<br>    datasets=[dataset_config],<br>)<br>```<br>A semi-related [discussion](https://github.com/coqui-ai/TTS/discussions/1998) suggests that Gruut might be iffy for certain foreign languages, and that it's better to either turn off phonemes, or to use espeak as a backend. Neither alternative has been confirmed to improve results. I attempt those solutions and will report back with those results as well, for those struggling with the same issue.<br>In the meantime, any experts' input is much appreciated. |

2You must be logged in to vote

All reactions

Answered by
[r-dh](https://github.com/r-dh)[on May 18, 2023May 18, 2023](https://github.com/coqui-ai/TTS/discussions/2615#discussioncomment-5937525)

Update: It seems dat using the default phonemizer, gruut, was indeed the curlpit. Changing to espeak allowed the model to speak.

However, I trained another model to 180k steps as well (both still ongoing) and I notice that the model _without_ phonemes gives a far more natural result.

* * *

**Graphemes (personally recommended)**

1. Import the CharactersConfig

```
from TTS.tts.models.vits import CharactersConfig
```

2. Update config

```
    ...
    text_cleaner="basic_cleaners", # or multilingual_cleaners
    use_phonemes=False,
    phoneme_language="nl",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        characters="!¡'(),-.:;¿?abcdefghijk…
```

[View full answer](https://github.com/coqui-ai/TTS/discussions/2615#discussioncomment-5937525)

## Replies:   1 comment

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/2615).

# {{title}}

Quote reply

### [![](https://avatars.githubusercontent.com/u/14875684?s=64&v=4)\ r-dh](https://github.com/r-dh) [on May 18, 2023May 18, 2023](https://github.com/coqui-ai/TTS/discussions/2615\#discussioncomment-5937525)   Author

|     |
| --- |
| Update: It seems dat using the default phonemizer, gruut, was indeed the curlpit. Changing to espeak allowed the model to speak.<br>However, I trained another model to 180k steps as well (both still ongoing) and I notice that the model _without_ phonemes gives a far more natural result.<br>* * *<br>**Graphemes (personally recommended)**<br>1. Import the CharactersConfig<br>```<br>from TTS.tts.models.vits import CharactersConfig<br>```<br>2. Update config<br>```<br>    ...<br>    text_cleaner="basic_cleaners", # or multilingual_cleaners<br>    use_phonemes=False,<br>    phoneme_language="nl", <br>    characters=CharactersConfig(<br>        characters_class="TTS.tts.models.vits.VitsCharacters",<br>        pad="<PAD>",<br>        characters="!¡'(),-.:;¿?abcdefghijklmnopqrstuvwxyzáçèéêëìíîïñòóôöùúûü«°±µ»$%&‘’‚“`”„",<br>        punctuations="!¡'(),-.:;¿? ",<br>    ),<br>    ...<br>```<br>Change the characters to those that may appear in your dataset. As a bonus tip, add `save_step=2500,` to save yourself some headaches in case your model crashes a couple of steps before its 10k milestone.<br>3. Train!<br>   <br>   <br>   I noticed it takes about 2.5 hours on an A100 to train 10k steps.<br>* * *<br>**Phonemes**<br>In case you want to use phonemes instead, append<br>```<br>phonemizer="espeak",<br>```<br>to your config and make sure to install espeak:<br>```<br>!sudo apt-get install espeak<br>```<br>The phoneme model is slightly quicker, it took me 2 hours per 10k steps on an A100. |

Marked as answer

1You must be logged in to vote

All reactions

0 replies


Answer selected by [r-dh](https://github.com/r-dh)

[Sign up for free](https://github.com/join?source=comment-repo) **to join this conversation on GitHub**.
Already have an account?
[Sign in to comment](https://github.com/login?return_to=https%3A%2F%2Fgithub.com%2Fcoqui-ai%2FTTS%2Fdiscussions%2F2615)

Category


[❓\\
\\
General Q&A](https://github.com/coqui-ai/TTS/discussions/categories/general-q-a)

Labels


None yet


1 participant


[![@r-dh](https://avatars.githubusercontent.com/u/14875684?s=48&v=4)](https://github.com/r-dh)

Heading

Bold

Italic

Quote

Code

Link

* * *

Numbered list

Unordered list

Task list

* * *

Attach files

Mention

Reference

# Select a reply

Loading

[Create a new saved reply](https://github.com/coqui-ai/TTS/discussions/2615)

👍1 reacted with thumbs up emoji👎1 reacted with thumbs down emoji😄1 reacted with laugh emoji🎉1 reacted with hooray emoji😕1 reacted with confused emoji❤️1 reacted with heart emoji🚀1 reacted with rocket emoji👀1 reacted with eyes emoji

You can’t perform that action at this time.