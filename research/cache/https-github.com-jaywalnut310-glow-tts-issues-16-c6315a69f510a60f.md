---
url: https://github.com/jaywalnut310/glow-tts/issues/16
title: Number of training steps · Issue #16 · jaywalnut310/glow-tts
retrieved_at: 2026-08-22T19:53:28Z
api_version: v2
---

[Skip to content](https://github.com/jaywalnut310/glow-tts/issues/16#start-of-content)

You signed in with another tab or window. [Reload](https://github.com/jaywalnut310/glow-tts/issues/16) to refresh your session.You signed out in another tab or window. [Reload](https://github.com/jaywalnut310/glow-tts/issues/16) to refresh your session.You switched accounts on another tab or window. [Reload](https://github.com/jaywalnut310/glow-tts/issues/16) to refresh your session.Dismiss alert

{{ message }}

[jaywalnut310](https://github.com/jaywalnut310)/ **[glow-tts](https://github.com/jaywalnut310/glow-tts)** Public

- [Notifications](https://github.com/login?return_to=%2Fjaywalnut310%2Fglow-tts) You must be signed in to change notification settings
- [Fork\\
153](https://github.com/login?return_to=%2Fjaywalnut310%2Fglow-tts)
- [Star\\
714](https://github.com/login?return_to=%2Fjaywalnut310%2Fglow-tts)


# Number of training steps\#16

[New issue](https://github.com/login?return_to=https://github.com/jaywalnut310/glow-tts/issues/16)

Copy link

[New issue](https://github.com/login?return_to=https://github.com/jaywalnut310/glow-tts/issues/16)

Copy link

Open

Open

[Number of training steps](https://github.com/jaywalnut310/glow-tts/issues/16#top)#16

Copy link

## Description

[![@sam1373](https://avatars.githubusercontent.com/u/24465107?v=4&size=48)](https://github.com/sam1373)

[sam1373](https://github.com/sam1373)

opened [on Jun 12, 2020on Jun 12, 2020](https://github.com/jaywalnut310/glow-tts/issues/16#issue-637895000)

Issue body actions

I have been trying to replicate the results from the paper, but I'm confused about the number of training steps. The paper mentions 240k steps, but when running this code on 8 V100 gpus 240k steps takes a lot longer that the 3 days from the paper. The base config here specifies 10000 epochs, which also doesn't seem like the correct amount. Could you clarify the correct amount of training epochs/steps?

## Activity

### jaywalnut310 commented on Jun 15, 2020on Jun 15, 2020

[![@jaywalnut310](https://avatars.githubusercontent.com/u/20279210?u=3bb6b13cbab549122d2247fbdbddbac0f2c359fd&v=4&size=48)](https://github.com/jaywalnut310)

[jaywalnut310](https://github.com/jaywalnut310)

[on Jun 15, 2020on Jun 15, 2020](https://github.com/jaywalnut310/glow-tts/issues/16#issuecomment-644492433)

Last edited by jaywalnut310

Owner

More actions

10000 epochs is meaningless. You can reduce number of epochs, or just cancel during training.

I trained my model with the base config on 2 V100 gpus, and it took ~ 3.3 days.

I think using large batch size might cause the slow training problem. The monotonic alignment search always operates on CPU cores, which means if the number of CPU cores does not increase with the number of gpus proportionally, CPU could take much time to search alignments for the 4 times larger batch than base setting.

![image](https://user-images.githubusercontent.com/20279210/84724363-f78dc100-afc2-11ea-822a-d099b1f323e8.png)

And according to the 4x increased batch size, I think it would make sense to quit training at less than 24k steps.

[![](https://avatars.githubusercontent.com/u/20279210?s=64&u=3bb6b13cbab549122d2247fbdbddbac0f2c359fd&v=4)jaywalnut310](https://github.com/jaywalnut310)

mentioned this [on Jun 15, 2020on Jun 15, 2020](https://github.com/jaywalnut310/glow-tts/issues/16#event-1037937658)

- [Whats values of final losses? #18](https://github.com/jaywalnut310/glow-tts/issues/18)


[![](https://avatars.githubusercontent.com/u/36092323?s=64&u=81714bcc02f2f251565155f0f910565e395242f3&v=4)v-nhandt21](https://github.com/v-nhandt21)

mentioned this [on May 24, 2021on May 24, 2021](https://github.com/jaywalnut310/glow-tts/issues/16#event-1116013596)

- [GPU not being utilized? #37](https://github.com/jaywalnut310/glow-tts/issues/37)


[Sign up for free](https://github.com/signup?return_to=https://github.com/jaywalnut310/glow-tts/issues/16)**to join this conversation on GitHub.** Already have an account? [Sign in to comment](https://github.com/login?return_to=https://github.com/jaywalnut310/glow-tts/issues/16)

## Metadata

## Metadata

### Assignees

No one assigned

### Labels

No labels

No labels

### Projects

No projects

### Milestone

No milestone

### Relationships

None yet

### Development

No branches or pull requests

### Participants

[![@jaywalnut310](https://avatars.githubusercontent.com/u/20279210?s=64&u=3bb6b13cbab549122d2247fbdbddbac0f2c359fd&v=4)](https://github.com/jaywalnut310)[![@sam1373](https://avatars.githubusercontent.com/u/24465107?s=64&v=4)](https://github.com/sam1373)

## Issue actions

- ![](https://github.githubassets.com/assets/github-copilot-app-light-15ad5534265eeacd.svg)Open in GitHub Copilot app

You can’t perform that action at this time.