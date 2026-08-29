---
url: https://github.com/coqui-ai/TTS/discussions/1703
title: glow-tts training · coqui-ai/TTS · Discussion #1703 · GitHub
retrieved_at: 2026-08-22T19:53:26Z
api_version: v2
---

[Skip to content](https://github.com/coqui-ai/TTS/discussions/1703#start-of-content)

You signed in with another tab or window. [Reload](https://github.com/coqui-ai/TTS/discussions/1703) to refresh your session.You signed out in another tab or window. [Reload](https://github.com/coqui-ai/TTS/discussions/1703) to refresh your session.You switched accounts on another tab or window. [Reload](https://github.com/coqui-ai/TTS/discussions/1703) to refresh your session.Dismiss alert

{{ message }}

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1703).

[coqui-ai](https://github.com/coqui-ai)/ **[TTS](https://github.com/coqui-ai/TTS)** Public

- [Notifications](https://github.com/login?return_to=%2Fcoqui-ai%2FTTS) You must be signed in to change notification settings
- [Fork\\
6.1k](https://github.com/login?return_to=%2Fcoqui-ai%2FTTS)
- [Star\\
45.9k](https://github.com/login?return_to=%2Fcoqui-ai%2FTTS)


# glow-tts training  \#1703

Unanswered

[MaksimSerg](https://github.com/MaksimSerg)

asked this question in
[General Q&A](https://github.com/coqui-ai/TTS/discussions/categories/general-q-a)

[glow-tts training](https://github.com/coqui-ai/TTS/discussions/1703#top)#1703

[![@MaksimSerg](https://avatars.githubusercontent.com/u/50648550?s=40&v=4)\\
MaksimSerg](https://github.com/MaksimSerg)

on Jun 28, 2022Jun 29, 2022·
3 comments
·
10 replies


[Return to top](https://github.com/coqui-ai/TTS/discussions/1703#top)

Discussion options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1703).

# {{title}}

Quote reply

edited

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1703).

# {{editor}}'s edit

{{actor}} deleted this content
.

# {{editor}}'s edit

## [![](https://avatars.githubusercontent.com/u/50648550?s=64&v=4)\ MaksimSerg](https://github.com/MaksimSerg) [on Jun 28, 2022Jun 29, 2022](https://github.com/coqui-ai/TTS/discussions/1703\#discussion-4180587)

|     |
| --- |
| Hello! I train glow-tts network in Russian (dataset RUSLAN). At 10k steps I already begin to understand speech, but after 40k there is no improvement in quality. I think she's learning too fast... What settings can I change?<br>The latest training is on the charts. Before that, it reached 140k, also without a visible change in the range of 40k-140k<br>![Screenshot1](https://user-images.githubusercontent.com/50648550/176344044-23735ac7-9feb-443e-9f76-765e4fb0fdc4.png)<br>![Screenshot3](https://user-images.githubusercontent.com/50648550/176350269-ecd219c5-07f3-408e-9a75-e35806f191d1.png)<br>![Screenshot2](https://user-images.githubusercontent.com/50648550/176344053-b0b27762-879c-46b7-9a1e-ef2d5b3d10f8.png)<br>Original speech<br>007619\_RUSLAN.mov<br>Synthesized speech<br>test.mov |

2You must be logged in to vote

All reactions

## Replies:   3 comments  ·  10 replies

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1703).

# {{title}}

Quote reply

### [![](https://avatars.githubusercontent.com/u/50648550?s=64&v=4)\ MaksimSerg](https://github.com/MaksimSerg) [on Jun 28, 2022Jun 29, 2022](https://github.com/coqui-ai/TTS/discussions/1703\#discussioncomment-3044926)   Author

|     |
| --- |
| My config:<br>```<br>{<br>    "output_path": "/home/smaksimov/TTS/recipes/ljspeech/glow_tts",<br>    "logger_uri": null,<br>    "run_name": "run",<br>    "project_name": null,<br>    "run_description": "\ud83d\udc38Coqui trainer run.",<br>    "print_step": 50,<br>    "plot_step": 100,<br>    "model_param_stats": false,<br>    "wandb_entity": null,<br>    "dashboard_logger": "tensorboard",<br>    "log_model_step": null,<br>    "save_step": 5000,<br>    "save_n_checkpoints": 20,<br>    "save_checkpoints": true,<br>    "save_all_best": false,<br>    "save_best_after": 10000,<br>    "target_loss": null,<br>    "print_eval": false,<br>    "test_delay_epochs": 0,<br>    "run_eval": true,<br>    "distributed_backend": "nccl",<br>    "distributed_url": "tcp://localhost:54321",<br>    "mixed_precision": false,<br>    "epochs": 3000,<br>    "batch_size": 48,<br>    "eval_batch_size": 16,<br>    "grad_clip": 5.0,<br>    "scheduler_after_epoch": true,<br>    "lr": 0.001,<br>    "optimizer": "RAdam",<br>    "optimizer_params": {<br>        "betas": [<br>            0.9,<br>            0.998<br>        ],<br>        "weight_decay": 1e-06<br>    },<br>    "lr_scheduler": "NoamLR",<br>    "lr_scheduler_params": {<br>        "warmup_steps": 4000<br>    },<br>    "use_grad_scaler": false,<br>    "cudnn_enable": true,<br>    "cudnn_deterministic": false,<br>    "cudnn_benchmark": false,<br>    "training_seed": 54321,<br>    "model": "glow_tts",<br>    "num_loader_workers": 8,<br>    "num_eval_loader_workers": 8,<br>    "use_noise_augment": true,<br>    "audio": {<br>        "fft_size": 1024,<br>        "win_length": 1024,<br>        "hop_length": 256,<br>        "frame_shift_ms": null,<br>        "frame_length_ms": null,<br>        "stft_pad_mode": "reflect",<br>        "sample_rate": 22050,<br>        "resample": false,<br>        "preemphasis": 0.5,<br>        "ref_level_db": 0,<br>        "do_sound_norm": false,<br>        "log_func": "np.log10",<br>        "do_trim_silence": true,<br>        "trim_db": 20,<br>        "do_rms_norm": false,<br>        "db_level": null,<br>        "power": 1.5,<br>        "griffin_lim_iters": 60,<br>        "num_mels": 80,<br>        "mel_fmin": 75.0,<br>        "mel_fmax": 7600.0,<br>        "spec_gain": 10,<br>        "do_amp_to_db_linear": true,<br>        "do_amp_to_db_mel": true,<br>        "pitch_fmax": 640.0,<br>        "pitch_fmin": 0.0,<br>        "signal_norm": true,<br>        "min_level_db": -100,<br>        "symmetric_norm": true,<br>        "max_norm": 4.0,<br>        "clip_norm": true,<br>        "stats_path": null<br>    },<br>    "use_phonemes": false,<br>    "phonemizer": "gruut",<br>    "phoneme_language": "ru-ru",<br>    "compute_input_seq_cache": true,<br>    "text_cleaner": "basic_cleaners",<br>    "enable_eos_bos_chars": false,<br>    "test_sentences_file": "",<br>    "phoneme_cache_path": "/home/smaksimov/TTS/recipes/ljspeech/glow_tts/phoneme_cache",<br>    "characters": {<br>        "characters_class": "TTS.tts.utils.text.characters.IPAPhonemes",<br>        "vocab_dict": null,<br>        "pad": "<PAD>",<br>        "eos": "<EOS>",<br>        "bos": "<BOS>",<br>        "blank": "<BLNK>",<br>        "characters": "\u0430\u0431\u0432\u0433\u0434\u0435\u0436\u0437\u0438\u0439\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444\u0445\u0446\u0447\u0448\u0449\u044a\u044b\u044c\u044d\u044e\u044f\u0451abcdefghijklmnopqrstuvwxyz",<br>        "punctuations": "!(),-.:;?-\u2013\u2014 \"'\u2019",<br>        "phonemes": null,<br>        "is_unique": false,<br>        "is_sorted": true<br>    },<br>    "add_blank": false,<br>    "batch_group_size": 0,<br>    "loss_masking": true,<br>    "sort_by_audio_len": false,<br>    "min_audio_len": 1,<br>    "max_audio_len": Infinity,<br>    "min_text_len": 20,<br>    "max_text_len": 100,<br>    "compute_f0": false,<br>    "compute_linear_spec": false,<br>    "precompute_num_workers": 0,<br>    "start_by_longest": false,<br>    "datasets": [<br>        {<br>            "name": "ljspeech",<br>            "path": "/home/smaksimov/TTS/recipes/ljspeech/glow_tts/../LJSpeech-1.2/",<br>            "meta_file_train": "metadata_train.csv",<br>            "ignored_speakers": null,<br>            "language": "ru",<br>            "meta_file_val": "metadata_val.csv",<br>            "meta_file_attn_mask": ""<br>        }<br>    ],<br>    "test_sentences": [<br>        "\u0428\u043b\u0430 \u0441\u0430\u0448\u0430 \u043f\u043e \u0448\u043e\u0441\u0441\u0435 \u0438 \u0441\u043e\u0441\u0430\u043b\u0430 \u0441\u0443\u0448\u043a\u0443.",<br>        "\u0422\u0435\u043a\u0441\u0442, \u043d\u0430\u043f\u0438\u0441\u0430\u043d\u043d\u044b\u0439 \u0440\u0443\u0441\u0441\u043a\u0438\u043c\u0438 \u0431\u0443\u043a\u0432\u0430\u043c\u0438.",<br>        "\u0414\u043e\u0431\u0440\u043e\u0435 \u0443\u0442\u0440\u043e. \u0414\u043e\u0431\u0440\u044b\u0439 \u0434\u0435\u043d\u044c. \u0414\u043e\u0431\u0440\u044b\u0439 \u0432\u0435\u0447\u0435\u0440.",<br>        "\u0412\u0430\u0441 \u043f\u0440\u0438\u0432\u0435\u0442\u0441\u0442\u0432\u0443\u0435\u0442 \u0441\u0438\u0441\u0442\u0435\u043c\u0430 \u041a\u041e\u041c\u041f\u0410\u041a\u0421.",<br>        "\u041e\u0431\u043d\u0430\u0440\u0443\u0436\u0435\u043d\u044b \u043d\u0435\u0438\u0441\u043f\u0440\u0430\u0432\u043d\u044b\u0435 \u043a\u0430\u043d\u0430\u043b\u044b!"<br>    ],<br>    "eval_split_max_size": null,<br>    "eval_split_size": 0.01,<br>    "use_speaker_weighted_sampler": false,<br>    "speaker_weighted_sampler_alpha": 1.0,<br>    "use_language_weighted_sampler": false,<br>    "language_weighted_sampler_alpha": 1.0,<br>    "num_chars": 79,<br>    "encoder_type": "rel_pos_transformer",<br>    "encoder_params": {<br>        "kernel_size": 3,<br>        "dropout_p": 0.1,<br>        "num_layers": 6,<br>        "num_heads": 2,<br>        "hidden_channels_ffn": 768,<br>        "input_length": null<br>    },<br>    "use_encoder_prenet": true,<br>    "hidden_channels_enc": 192,<br>    "hidden_channels_dec": 192,<br>    "hidden_channels_dp": 256,<br>    "dropout_p_dp": 0.1,<br>    "dropout_p_dec": 0.05,<br>    "mean_only": true,<br>    "out_channels": 80,<br>    "num_flow_blocks_dec": 12,<br>    "inference_noise_scale": 0.667,<br>    "kernel_size_dec": 5,<br>    "dilation_rate": 1,<br>    "num_block_layers": 4,<br>    "num_speakers": 0,<br>    "c_in_channels": 0,<br>    "num_splits": 4,<br>    "num_squeeze": 2,<br>    "sigmoid_scale": false,<br>    "d_vector_dim": 0,<br>    "data_dep_init_steps": 10,<br>    "style_wav_for_test": null,<br>    "length_scale": 1.0,<br>    "use_speaker_embedding": false,<br>    "speakers_file": null,<br>    "use_d_vector_file": false,<br>    "d_vector_file": false,<br>    "min_seq_len": 3,<br>    "max_seq_len": 500,<br>    "r": 3<br>}<br>``` |

1You must be logged in to vote

All reactions

1 reply


[![@iprovalo](https://avatars.githubusercontent.com/u/1660997?s=60&v=4)](https://github.com/iprovalo)

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1703).

# {{title}}

Quote reply

#### [iprovalo](https://github.com/iprovalo) [on Jun 30, 2022Jun 30, 2022](https://github.com/coqui-ai/TTS/discussions/1703\#discussioncomment-3059870)

|     |
| --- |
| [@MaksimSerg](https://github.com/MaksimSerg) I am training Russian ljspeech dataset on glow tts. I am facing similar issues. The speech is already recognizable but it won't improve. When I combine the model with mb melgan vocoder, it is hardly recognizable during the inference.<br>I am describing various configs I am using here: [#1701](https://github.com/coqui-ai/TTS/discussions/1701) |

All reactions

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1703).

# {{title}}

Quote reply

### [![](https://avatars.githubusercontent.com/u/50648550?s=64&v=4)\ MaksimSerg](https://github.com/MaksimSerg) [on Jun 30, 2022Jul 1, 2022](https://github.com/coqui-ai/TTS/discussions/1703\#discussioncomment-3060684)   Author

|     |
| --- |
| 200k steps<br>test-200k.mov |

1You must be logged in to vote

👍1

All reactions

- 👍1

9 replies


Show 4 previous replies

[![@iprovalo](https://avatars.githubusercontent.com/u/1660997?s=60&v=4)](https://github.com/iprovalo)

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1703).

# {{title}}

Quote reply

#### [iprovalo](https://github.com/iprovalo) [on Jun 30, 2022Jul 1, 2022](https://github.com/coqui-ai/TTS/discussions/1703\#discussioncomment-3060878)

|     |
| --- |
| Thank you, [@MaksimSerg](https://github.com/MaksimSerg)! Your latest model sounds good! |

All reactions

[![@iprovalo](https://avatars.githubusercontent.com/u/1660997?s=60&v=4)](https://github.com/iprovalo)

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1703).

# {{title}}

Quote reply

#### [iprovalo](https://github.com/iprovalo) [on Jun 30, 2022Jul 1, 2022](https://github.com/coqui-ai/TTS/discussions/1703\#discussioncomment-3060888)

|     |
| --- |
| Have you tried the [ljspeech Russian dataset](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/)? |

All reactions

[![@MaksimSerg](https://avatars.githubusercontent.com/u/50648550?s=60&v=4)](https://github.com/MaksimSerg)

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1703).

# {{title}}

Quote reply

#### [MaksimSerg](https://github.com/MaksimSerg) [on Jun 30, 2022Jul 1, 2022](https://github.com/coqui-ai/TTS/discussions/1703\#discussioncomment-3060906)   Author

|     |
| --- |
| No. For the Russian language there is poor sound quality. |

All reactions

[![@iprovalo](https://avatars.githubusercontent.com/u/1660997?s=60&v=4)](https://github.com/iprovalo)

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1703).

# {{title}}

Quote reply

#### [iprovalo](https://github.com/iprovalo) [on Jun 30, 2022Jul 1, 2022](https://github.com/coqui-ai/TTS/discussions/1703\#discussioncomment-3060912)

|     |
| --- |
| Thank you! |

All reactions

[![@iprovalo](https://avatars.githubusercontent.com/u/1660997?s=60&v=4)](https://github.com/iprovalo)

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1703).

# {{title}}

Quote reply

#### [iprovalo](https://github.com/iprovalo) [on Jul 3, 2022Jul 3, 2022](https://github.com/coqui-ai/TTS/discussions/1703\#discussioncomment-3072952)

|     |
| --- |
| [@MaksimSerg](https://github.com/MaksimSerg) I switched to RUSLAN dataset, there are significant improvements in training speed and quality. Run 12 and 13 in this [doc](https://docs.google.com/document/d/1f2IC66kuogjl9Omjkbz2Ixzaw5K1iFXMNNaBjITDQw0/edit#). Thank you! |

All reactions

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/1703).

# {{title}}

Quote reply

### [![](https://avatars.githubusercontent.com/u/1402048?s=64&v=4)\ erogol](https://github.com/erogol) [on Jul 5, 2022Jul 5, 2022](https://github.com/coqui-ai/TTS/discussions/1703\#discussioncomment-3083170)   Maintainer

|     |
| --- |
| You need to train a vocoder model. Glow-TTS is without a vocoder. |

2You must be logged in to vote

👍2

All reactions

- 👍2

0 replies


[Sign up for free](https://github.com/join?source=comment-repo) **to join this conversation on GitHub**.
Already have an account?
[Sign in to comment](https://github.com/login?return_to=https%3A%2F%2Fgithub.com%2Fcoqui-ai%2FTTS%2Fdiscussions%2F1703)

Category


[❓\\
\\
General Q&A](https://github.com/coqui-ai/TTS/discussions/categories/general-q-a)

Labels


None yet


3 participants


[![@MaksimSerg](https://avatars.githubusercontent.com/u/50648550?s=48&v=4)](https://github.com/MaksimSerg)[![@erogol](https://avatars.githubusercontent.com/u/1402048?s=48&v=4)](https://github.com/erogol)[![@iprovalo](https://avatars.githubusercontent.com/u/1660997?s=48&v=4)](https://github.com/iprovalo)

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

[Create a new saved reply](https://github.com/coqui-ai/TTS/discussions/1703)

👍1 reacted with thumbs up emoji👎1 reacted with thumbs down emoji😄1 reacted with laugh emoji🎉1 reacted with hooray emoji😕1 reacted with confused emoji❤️1 reacted with heart emoji🚀1 reacted with rocket emoji👀1 reacted with eyes emoji

You can’t perform that action at this time.