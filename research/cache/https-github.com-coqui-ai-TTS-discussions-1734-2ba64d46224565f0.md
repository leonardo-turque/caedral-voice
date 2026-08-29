---
url: https://github.com/coqui-ai/TTS/discussions/1734
title: Number of steps for VITS recipe on VCTK training · coqui-ai/TTS · Discussion #1734 · GitHub
retrieved_at: 2026-08-22T19:53:30Z
api_version: v2
---

[Skip to content](https://github.com/coqui-ai/TTS/discussions/1734#start-of-content)

You signed in with another tab or window. [Reload](https://github.com/coqui-ai/TTS/discussions/1734) to refresh your session.You signed out in another tab or window. [Reload](https://github.com/coqui-ai/TTS/discussions/1734) to refresh your session.You switched accounts on another tab or window. [Reload](https://github.com/coqui-ai/TTS/discussions/1734) to refresh your session.Dismiss alert

{{ message }}

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1734).

[coqui-ai](https://github.com/coqui-ai)/ **[TTS](https://github.com/coqui-ai/TTS)** Public

- [Notifications](https://github.com/login?return_to=%2Fcoqui-ai%2FTTS) You must be signed in to change notification settings
- [Fork\\
6.1k](https://github.com/login?return_to=%2Fcoqui-ai%2FTTS)
- [Star\\
45.9k](https://github.com/login?return_to=%2Fcoqui-ai%2FTTS)


# Number of steps for VITS recipe on VCTK training  \#1734

Unanswered

[yurtmete](https://github.com/yurtmete)

asked this question in
[General Q&A](https://github.com/coqui-ai/TTS/discussions/categories/general-q-a)

[Number of steps for VITS recipe on VCTK training](https://github.com/coqui-ai/TTS/discussions/1734#top)#1734

[![@yurtmete](https://avatars.githubusercontent.com/u/52315030?s=40&v=4)\\
yurtmete](https://github.com/yurtmete)

on Jul 15, 2022Jul 15, 2022·
1 comments
·
8 replies


[Return to top](https://github.com/coqui-ai/TTS/discussions/1734#top)

Discussion options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1734).

# {{title}}

Quote reply

## [![](https://avatars.githubusercontent.com/u/52315030?s=64&v=4)\ yurtmete](https://github.com/yurtmete) [on Jul 15, 2022Jul 15, 2022](https://github.com/coqui-ai/TTS/discussions/1734\#discussion-4218980)

|     |
| --- |
| Hi all,<br>I am currently following your VCTK recipe for VITS model and I also tried your VITS model trained with VCTK dataset. The quality is very good considering the amount of data each speaker has.<br>Based on the VITS paper, they trained the model for 800k steps with 64x4 (they use 4 GPUs) batch size.<br>I was wondering, for many iterations have you trained the model. Based on the recipe, it seems that you use batch size 32 with single GPU. Did you wait until you hit 1000th epoch?<br>Thank you in advance. |

3You must be logged in to vote

All reactions

## Replies:   1 comment  ·  8 replies

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1734).

# {{title}}

Quote reply

### [![](https://avatars.githubusercontent.com/u/52315030?s=64&v=4)\ yurtmete](https://github.com/yurtmete) [on Jul 26, 2022Jul 26, 2022](https://github.com/coqui-ai/TTS/discussions/1734\#discussioncomment-3230120)   Author

|     |
| --- |
| I downloaded the pretrained model using:<br>`$ wget https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--vctk--vits.zip`<br>Number of steps = 1000000<br>Number of epochs = 556<br>Batch size = 24 (this is an estimate based on the size of the dataset and the step, epoch sizes.)<br>Coqui guys can correct me if these are wrong. |

1You must be logged in to vote

All reactions

8 replies


Show 3 previous replies

[![@yurtmete](https://avatars.githubusercontent.com/u/52315030?s=60&v=4)](https://github.com/yurtmete)

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1734).

# {{title}}

Quote reply

#### [yurtmete](https://github.com/yurtmete) [on Aug 1, 2022Aug 1, 2022](https://github.com/coqui-ai/TTS/discussions/1734\#discussioncomment-3297086)   Author

|     |
| --- |
| So the VITS VCTK model was not trained on full VCTK dataset from scratch? |

All reactions

[![@erogol](https://avatars.githubusercontent.com/u/1402048?s=60&v=4)](https://github.com/erogol)

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1734).

# {{title}}

Quote reply

#### [erogol](https://github.com/erogol) [on Aug 7, 2022Aug 7, 2022](https://github.com/coqui-ai/TTS/discussions/1734\#discussioncomment-3343139)   Maintainer

|     |
| --- |
| AFAIR it started with LJSpeech and then fine-tuned with VCTK. |

All reactions

[![@yurtmete](https://avatars.githubusercontent.com/u/52315030?s=60&v=4)](https://github.com/yurtmete)

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1734).

# {{title}}

Quote reply

edited

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1734).

# {{editor}}'s edit

{{actor}} deleted this content
.

# {{editor}}'s edit

#### [yurtmete](https://github.com/yurtmete) [on Aug 11, 2022Aug 11, 2022](https://github.com/coqui-ai/TTS/discussions/1734\#discussioncomment-3374376)   Author

|     |
| --- |
| Understand, I am now trying to do the same, take your LJSpeech VITS model and fine tune it with VCTK dataset. This I think challenging in the sense that we go from single speaker to multi speaker.<br>I followed your suggestions [here](https://tts.readthedocs.io/en/latest/finetuning.html). For the learning rates, what I have tried is that, I set the initial learning rate for the discriminator (since I do not have it in LJSpeech checkpoint) and for the generator do the following, if I am able to load the weights, I divide the initial learning rate by 10 to not to forget what LJSpeech model learnt, for the rest (like all the condition layers (there are no condition layers in LJSpeech model), text encoder (vocabulary is different), and speaker embedding) I set the initial learning since I will learn them from scratch.<br>However, this did not work, and the model started to speak very fast.<br>Do you suggest something else on finetuning LJSpeech model with VCTK voices?<br>One final remark: **When finetuning, I am not using LJSpeech data at all** |

All reactions

[![@erogol](https://avatars.githubusercontent.com/u/1402048?s=60&v=4)](https://github.com/erogol)

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1734).

# {{title}}

Quote reply

#### [erogol](https://github.com/erogol) [on Aug 15, 2022Aug 15, 2022](https://github.com/coqui-ai/TTS/discussions/1734\#discussioncomment-3396779)   Maintainer

|     |
| --- |
| Just train with LJSpeech once it converged, and continue with VCTK. You don't need to change anything I believe. |

All reactions

[![@yurtmete](https://avatars.githubusercontent.com/u/52315030?s=60&v=4)](https://github.com/yurtmete)

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1734).

# {{title}}

Quote reply

#### [yurtmete](https://github.com/yurtmete) [on Aug 23, 2022Aug 23, 2022](https://github.com/coqui-ai/TTS/discussions/1734\#discussioncomment-3455248)   Author

|     |
| --- |
| Not even reducing the learning rate? |

👍1

All reactions

- 👍1

[Sign up for free](https://github.com/join?source=comment-repo) **to join this conversation on GitHub**.
Already have an account?
[Sign in to comment](https://github.com/login?return_to=https%3A%2F%2Fgithub.com%2Fcoqui-ai%2FTTS%2Fdiscussions%2F1734)

Category


[❓\\
\\
General Q&A](https://github.com/coqui-ai/TTS/discussions/categories/general-q-a)

Labels


None yet


2 participants


[![@yurtmete](https://avatars.githubusercontent.com/u/52315030?s=48&v=4)](https://github.com/yurtmete)[![@erogol](https://avatars.githubusercontent.com/u/1402048?s=48&v=4)](https://github.com/erogol)

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

[Create a new saved reply](https://github.com/coqui-ai/TTS/discussions/1734)

👍1 reacted with thumbs up emoji👎1 reacted with thumbs down emoji😄1 reacted with laugh emoji🎉1 reacted with hooray emoji😕1 reacted with confused emoji❤️1 reacted with heart emoji🚀1 reacted with rocket emoji👀1 reacted with eyes emoji

You can’t perform that action at this time.