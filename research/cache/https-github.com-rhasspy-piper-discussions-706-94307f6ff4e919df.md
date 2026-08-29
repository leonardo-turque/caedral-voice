---
url: https://github.com/rhasspy/piper/discussions/706
title: Time Cost Estimation for Training Piper from Scratch · rhasspy/piper · Discussion #706 · GitHub
retrieved_at: 2026-08-22T19:53:36Z
api_version: v2
---

[Skip to content](https://github.com/rhasspy/piper/discussions/706#start-of-content)

You signed in with another tab or window. [Reload](https://github.com/rhasspy/piper/discussions/706) to refresh your session.You signed out in another tab or window. [Reload](https://github.com/rhasspy/piper/discussions/706) to refresh your session.You switched accounts on another tab or window. [Reload](https://github.com/rhasspy/piper/discussions/706) to refresh your session.Dismiss alert

{{ message }}

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/rhasspy/piper/discussions/706).

This repository was archived by the owner on Oct 6, 2025. It is now read-only.


[rhasspy](https://github.com/rhasspy)/ **[piper](https://github.com/rhasspy/piper)** Public archive

- [Notifications](https://github.com/login?return_to=%2Frhasspy%2Fpiper) You must be signed in to change notification settings
- [Fork\\
1.1k](https://github.com/login?return_to=%2Frhasspy%2Fpiper)
- [Star\\
11.3k](https://github.com/login?return_to=%2Frhasspy%2Fpiper)


# Time Cost Estimation for Training Piper from Scratch  \#706

Unanswered

[Lennox-Dai](https://github.com/Lennox-Dai)

asked this question in
[Q&A](https://github.com/rhasspy/piper/discussions/categories/q-a)

[Time Cost Estimation for Training Piper from Scratch](https://github.com/rhasspy/piper/discussions/706#top)#706

[![@Lennox-Dai](https://avatars.githubusercontent.com/u/140293041?s=40&v=4)\\
Lennox-Dai](https://github.com/Lennox-Dai)

on Jan 21, 2025Jan 22, 2025·
1 comment


[Return to top](https://github.com/rhasspy/piper/discussions/706#top)

Discussion options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/rhasspy/piper/discussions/706).

# {{title}}

Quote reply

edited

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/rhasspy/piper/discussions/706).

# {{editor}}'s edit

{{actor}} deleted this content
.

# {{editor}}'s edit

## [![](https://avatars.githubusercontent.com/u/140293041?s=64&v=4)\ Lennox-Dai](https://github.com/Lennox-Dai) [on Jan 21, 2025Jan 22, 2025](https://github.com/rhasspy/piper/discussions/706\#discussion-7855298)

|     |
| --- |
| It seems that I haven't found a model for Piper in some languages (such as Japanese), and when I try finetuning from models in other languages like Chinese, the results don’t seem to be ideal. So, I would like to train a Japanese model from scratch.<br>My GPU resources are 8\*H800, and I am using the DDP strategy for multi-GPU training. The dataset consists of over 7,000 high-quality single-speaker Japanese speech samples, totaling about 10 hours of audio. Currently, I have trained for epoch=17979-step=724820.ckpt, but the results are not very satisfactory; the generated speech still sounds like a bunch of noise.<br>My training command is as follows:<br>torchrun --nproc\_per\_node=8 --nnodes=1 --master\_port=29506 -m piper\_train <br>--dataset-dir "$DATASET\_DIR" <br>--accelerator "gpu" <br>--strategy "ddp" <br>--devices 8 \<br>--batch-size 64 <br>--validation-split 0.0 <br>--num-test-examples 0 <br>--max\_epochs $EPOCH <br>--checkpoint-epochs 10 <br>--precision 32 <br>--max-phoneme-ids 600<br>My loss\_disc\_all fluctuates between 1.2 and 1.8, and loss\_gen\_all is between 40 and 55.<br>Could this poor result be due to insufficient training time, or is the dataset not enough? If you have any suggestions or improvements for my training strategy, I would greatly appreciate any guidance you can provide. Thank you so much! |

1You must be logged in to vote

👀1

All reactions

- 👀1

## Replies:   1 comment

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/rhasspy/piper/discussions/706).

# {{title}}

Quote reply

### [![](https://avatars.githubusercontent.com/u/30713588?s=64&v=4)\ adekos](https://github.com/adekos) [on Jul 21, 2025Jul 21, 2025](https://github.com/rhasspy/piper/discussions/706\#discussioncomment-13831767)

|     |
| --- |
| Hi do you have news about japanese model in piper tts? |

1You must be logged in to vote

All reactions

0 replies


[Sign up for free](https://github.com/join?source=comment-repo) **to join this conversation on GitHub**.
Already have an account?
[Sign in to comment](https://github.com/login?return_to=https%3A%2F%2Fgithub.com%2Frhasspy%2Fpiper%2Fdiscussions%2F706)

Category


[🙏\\
\\
Q&A](https://github.com/rhasspy/piper/discussions/categories/q-a)

Labels


None yet


2 participants


[![@Lennox-Dai](https://avatars.githubusercontent.com/u/140293041?s=48&v=4)](https://github.com/Lennox-Dai)[![@adekos](https://avatars.githubusercontent.com/u/30713588?s=48&v=4)](https://github.com/adekos)

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

[Create a new saved reply](https://github.com/rhasspy/piper/discussions/706)

👍1 reacted with thumbs up emoji👎1 reacted with thumbs down emoji😄1 reacted with laugh emoji🎉1 reacted with hooray emoji😕1 reacted with confused emoji❤️1 reacted with heart emoji🚀1 reacted with rocket emoji👀1 reacted with eyes emoji

You can’t perform that action at this time.