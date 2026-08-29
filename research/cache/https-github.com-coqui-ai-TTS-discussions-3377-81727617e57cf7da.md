---
url: https://github.com/coqui-ai/TTS/discussions/3377
title: VITS training - why eval loss converges quickly and continues to increase? · coqui-ai/TTS · Discussion #3377 · GitHub
retrieved_at: 2026-08-22T19:53:32Z
api_version: v2
---

[Skip to content](https://github.com/coqui-ai/TTS/discussions/3377#start-of-content)

You signed in with another tab or window. [Reload](https://github.com/coqui-ai/TTS/discussions/3377) to refresh your session.You signed out in another tab or window. [Reload](https://github.com/coqui-ai/TTS/discussions/3377) to refresh your session.You switched accounts on another tab or window. [Reload](https://github.com/coqui-ai/TTS/discussions/3377) to refresh your session.Dismiss alert

{{ message }}

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/3377).

[coqui-ai](https://github.com/coqui-ai)/ **[TTS](https://github.com/coqui-ai/TTS)** Public

- [Notifications](https://github.com/login?return_to=%2Fcoqui-ai%2FTTS) You must be signed in to change notification settings
- [Fork\\
6.1k](https://github.com/login?return_to=%2Fcoqui-ai%2FTTS)
- [Star\\
45.9k](https://github.com/login?return_to=%2Fcoqui-ai%2FTTS)


# VITS training - why eval loss converges quickly and continues to increase?  \#3377

Closed
Unanswered

[nicemanis](https://github.com/nicemanis)

asked this question in
[General Q&A](https://github.com/coqui-ai/TTS/discussions/categories/general-q-a)

[VITS training - why eval loss converges quickly and continues to increase?](https://github.com/coqui-ai/TTS/discussions/3377#top)#3377

[![@nicemanis](https://avatars.githubusercontent.com/u/9463816?s=40&v=4)\\
nicemanis](https://github.com/nicemanis)

on Dec 7, 2023Dec 7, 2023·
1 comment


[Return to top](https://github.com/coqui-ai/TTS/discussions/3377#top)

Discussion options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/3377).

# {{title}}

Quote reply

## [![](https://avatars.githubusercontent.com/u/9463816?s=64&v=4)\ nicemanis](https://github.com/nicemanis) [on Dec 7, 2023Dec 7, 2023](https://github.com/coqui-ai/TTS/discussions/3377\#discussion-5933300)

|     |
| --- |
| I've been experimenting with VITS fine-tuning on my own datasets from a VITS VCTK checkpoint. While I managed to get somewhat decent results, it still had problems with the speech flow and pauses between words, similar to the issues reported in this discussion [#2834](https://github.com/coqui-ai/TTS/discussions/2834). For all of my training runs I observed that the eval loss initially rapidly decreased, but then continued to increase after 10-20K steps, suggesting overfitting.<br>In order to isolate the problem and verify the validity of the training recipe, I tried to train VITS from scratch on the VCTK dataset; however it seems that I'm encountering the same problem.<br>## Tensorboard logs<br>![image](https://private-user-images.githubusercontent.com/9463816/288649787-098595c5-97fe-4a47-8bdc-078607a8491f.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODc0Mjg3MTAsIm5iZiI6MTc4NzQyODQxMCwicGF0aCI6Ii85NDYzODE2LzI4ODY0OTc4Ny0wOTg1OTVjNS05N2ZlLTRhNDctOGJkYy0wNzg2MDdhODQ5MWYucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDgyMiUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjA4MjJUMTk1MzMwWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9NTc5NThlOWJlNzI1YjkxNTFlNjY1MDE3M2QwOTk3NGI1ODdlYjdiZGU1YTcwZjJmZWZiMTVmODBmMDBlNTFhYiZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmcmVzcG9uc2UtY29udGVudC10eXBlPWltYWdlJTJGcG5nIn0.V70cQChT9iF0CYjwGT3r9uoMFJfUP1VlAgankwyu6ME)<br>![image](https://private-user-images.githubusercontent.com/9463816/288649958-ba341ba7-3dac-4daa-a825-651a021a32c9.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODc0Mjg3MTAsIm5iZiI6MTc4NzQyODQxMCwicGF0aCI6Ii85NDYzODE2LzI4ODY0OTk1OC1iYTM0MWJhNy0zZGFjLTRkYWEtYTgyNS02NTFhMDIxYTMyYzkucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDgyMiUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjA4MjJUMTk1MzMwWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9YTY4YzA3MTliYzJhZjIyOTJhZjM2MmZiNmRkZmRkOGQ1OTdkMWUzZWM4Mjg2NzNhZDYyNzEwOGRkZTRmY2FjNyZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmcmVzcG9uc2UtY29udGVudC10eXBlPWltYWdlJTJGcG5nIn0.5mpSeJTSNCrKFToeV7rsRwHW5zfmMyn5jEQN16p3mdw)<br>![image](https://private-user-images.githubusercontent.com/9463816/288650112-89d5dd6b-cd33-4154-9881-8afa7467e42d.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODc0Mjg3MTAsIm5iZiI6MTc4NzQyODQxMCwicGF0aCI6Ii85NDYzODE2LzI4ODY1MDExMi04OWQ1ZGQ2Yi1jZDMzLTQxNTQtOTg4MS04YWZhNzQ2N2U0MmQucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDgyMiUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjA4MjJUMTk1MzMwWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9MWUyYTQ5ODU0YWQ1ZTM3ZjhkYjYwYWJjNDEzNGQ1ZWJlYmY3ZjNhODkwNDdmNGRjNzRmZjFmZGNjNGQ4NmYzMSZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmcmVzcG9uc2UtY29udGVudC10eXBlPWltYWdlJTJGcG5nIn0.Vfbvp6aEg82Qa2yE9wLeG87JqdD4IgosbSle5BEayVo)<br>## config.json<br>```<br>{<br>    "output_path": "",<br>    "logger_uri": null,<br>    "run_name": "",<br>    "project_name": null,<br>    "run_description": "",<br>    "print_step": 50,<br>    "plot_step": 100,<br>    "model_param_stats": false,<br>    "wandb_entity": null,<br>    "dashboard_logger": "tensorboard",<br>    "log_model_step": 1000,<br>    "save_step": 5000,<br>    "save_n_checkpoints": 5,<br>    "save_checkpoints": true,<br>    "save_all_best": false,<br>    "save_best_after": 10000,<br>    "target_loss": null,<br>    "print_eval": false,<br>    "test_delay_epochs": -1,<br>    "run_eval": true,<br>    "run_eval_steps": null,<br>    "distributed_backend": "nccl",<br>    "distributed_url": "tcp://localhost:54321",<br>    "mixed_precision": false,<br>    "epochs": 1000,<br>    "batch_size": 32,<br>    "eval_batch_size": 32,<br>    "grad_clip": [<br>        1000.0,<br>        1000.0<br>    ],<br>    "scheduler_after_epoch": true,<br>    "lr": 0.001,<br>    "optimizer": "AdamW",<br>    "optimizer_params": {<br>        "betas": [<br>            0.8,<br>            0.99<br>        ],<br>        "eps": 1e-09,<br>        "weight_decay": 0.01<br>    },<br>    "lr_scheduler": null,<br>    "lr_scheduler_params": {},<br>    "use_grad_scaler": false,<br>    "cudnn_enable": true,<br>    "cudnn_deterministic": false,<br>    "cudnn_benchmark": false,<br>    "training_seed": 54321,<br>    "model": "vits",<br>    "num_loader_workers": 8,<br>    "num_eval_loader_workers": 8,<br>    "use_noise_augment": false,<br>    "audio": {<br>        "fft_size": 1024,<br>        "sample_rate": 22050,<br>        "win_length": 1024,<br>        "hop_length": 256,<br>        "num_mels": 80,<br>        "mel_fmin": 0,<br>        "mel_fmax": null<br>    },<br>    "use_phonemes": true,<br>    "phonemizer": "multi_phonemizer",<br>    "phoneme_language": null,<br>    "compute_input_seq_cache": true,<br>    "text_cleaner": "multilingual_cleaners",<br>    "enable_eos_bos_chars": false,<br>    "test_sentences_file": "",<br>    "phoneme_cache_path": "/tmp/coqui-tts/phoneme_cache",<br>    "characters": {<br>        "characters_class": "TTS.tts.utils.text.characters.IPAPhonemes",<br>        "vocab_dict": null,<br>        "pad": "<PAD>",<br>        "eos": "<EOS>",<br>        "bos": "<BOS>",<br>        "blank": "<BLNK>",<br>        "characters": "iy\u0268\u0289\u026fu\u026a\u028f\u028ae\u00f8\u0258\u0259\u0275\u0264o\u025b\u0153\u025c\u025e\u028c\u0254\u00e6\u0250a\u0276\u0251\u0252\u1d7b\u0298\u0253\u01c0\u0257\u01c3\u0284\u01c2\u0260\u01c1\u029bpbtd\u0288\u0256c\u025fk\u0261q\u0262\u0294\u0274\u014b\u0272\u0273n\u0271m\u0299r\u0280\u2c71\u027e\u027d\u0278\u03b2fv\u03b8\u00f0sz\u0283\u0292\u0282\u0290\u00e7\u029dx\u0263\u03c7\u0281\u0127\u0295h\u0266\u026c\u026e\u028b\u0279\u027bj\u0270l\u026d\u028e\u029f\u02c8\u02cc\u02d0\u02d1\u028dw\u0265\u029c\u02a2\u02a1\u0255\u0291\u027a\u0267\u02b2\u025a\u02de\u026b",<br>        "punctuations": "!'(),-.:;? ",<br>        "phonemes": null,<br>        "is_unique": false,<br>        "is_sorted": true<br>    },<br>    "add_blank": true,<br>    "batch_group_size": 0,<br>    "loss_masking": null,<br>    "min_audio_len": 22050,<br>    "max_audio_len": 220500,<br>    "min_text_len": 1,<br>    "max_text_len": Infinity,<br>    "compute_f0": false,<br>    "compute_energy": false,<br>    "compute_linear_spec": true,<br>    "precompute_num_workers": 0,<br>    "start_by_longest": false,<br>    "shuffle": false,<br>    "drop_last": false,<br>    "datasets": [<br>        {<br>            "formatter": "vctk",<br>            "dataset_name": "vctk",<br>            "path": "/tmp/coqui-tts/VCTK",<br>            "meta_file_train": "",<br>            "ignored_speakers": [<br>                "p261",<br>                "p225",<br>                "p294",<br>                "p347",<br>                "p238",<br>                "p234",<br>                "p248",<br>                "p335",<br>                "p245",<br>                "p326",<br>                "p302"<br>            ],<br>            "language": "en",<br>            "phonemizer": "",<br>            "meta_file_val": "",<br>            "meta_file_attn_mask": ""<br>        }<br>    ],<br>    "test_sentences": [<br>        [<br>            "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",<br>            "VCTK_p277",<br>            null,<br>            "en"<br>        ]<br>    ],<br>    "eval_split_max_size": null,<br>    "eval_split_size": 0.01,<br>    "use_speaker_weighted_sampler": false,<br>    "speaker_weighted_sampler_alpha": 1.0,<br>    "use_language_weighted_sampler": true,<br>    "language_weighted_sampler_alpha": 1.0,<br>    "use_length_weighted_sampler": false,<br>    "length_weighted_sampler_alpha": 1.0,<br>    "model_args": {<br>        "num_chars": 131,<br>        "out_channels": 513,<br>        "spec_segment_size": 32,<br>        "hidden_channels": 192,<br>        "hidden_channels_ffn_text_encoder": 768,<br>        "num_heads_text_encoder": 2,<br>        "num_layers_text_encoder": 6,<br>        "kernel_size_text_encoder": 3,<br>        "dropout_p_text_encoder": 0.1,<br>        "dropout_p_duration_predictor": 0.5,<br>        "kernel_size_posterior_encoder": 5,<br>        "dilation_rate_posterior_encoder": 1,<br>        "num_layers_posterior_encoder": 16,<br>        "kernel_size_flow": 5,<br>        "dilation_rate_flow": 1,<br>        "num_layers_flow": 4,<br>        "resblock_type_decoder": "1",<br>        "resblock_kernel_sizes_decoder": [<br>            3,<br>            7,<br>            11<br>        ],<br>        "resblock_dilation_sizes_decoder": [<br>            [<br>                1,<br>                3,<br>                5<br>            ],<br>            [<br>                1,<br>                3,<br>                5<br>            ],<br>            [<br>                1,<br>                3,<br>                5<br>            ]<br>        ],<br>        "upsample_rates_decoder": [<br>            8,<br>            8,<br>            2,<br>            2<br>        ],<br>        "upsample_initial_channel_decoder": 512,<br>        "upsample_kernel_sizes_decoder": [<br>            16,<br>            16,<br>            4,<br>            4<br>        ],<br>        "periods_multi_period_discriminator": [<br>            2,<br>            3,<br>            5,<br>            7,<br>            11<br>        ],<br>        "use_sdp": true,<br>        "noise_scale": 1.0,<br>        "inference_noise_scale": 0.667,<br>        "length_scale": 1.0,<br>        "noise_scale_dp": 1.0,<br>        "inference_noise_scale_dp": 1.0,<br>        "max_inference_len": null,<br>        "init_discriminator": true,<br>        "use_spectral_norm_disriminator": false,<br>        "use_speaker_embedding": true,<br>        "num_speakers": 98,<br>        "speakers_file": "",<br>        "d_vector_file": null,<br>        "speaker_embedding_channels": 256,<br>        "use_d_vector_file": false,<br>        "d_vector_dim": 0,<br>        "detach_dp_input": true,<br>        "use_language_embedding": true,<br>        "embedded_language_dim": 4,<br>        "num_languages": 1,<br>        "language_ids_file": "",<br>        "use_speaker_encoder_as_loss": false,<br>        "speaker_encoder_config_path": "",<br>        "speaker_encoder_model_path": "",<br>        "condition_dp_on_speaker": true,<br>        "freeze_encoder": false,<br>        "freeze_DP": false,<br>        "freeze_PE": false,<br>        "freeze_flow_decoder": false,<br>        "freeze_waveform_decoder": false,<br>        "encoder_sample_rate": null,<br>        "interpolate_z": true,<br>        "reinit_DP": false,<br>        "reinit_text_encoder": false<br>    },<br>    "lr_gen": 0.0002,<br>    "lr_disc": 0.0002,<br>    "lr_scheduler_gen": "ExponentialLR",<br>    "lr_scheduler_gen_params": {<br>        "gamma": 0.999875,<br>        "last_epoch": -1<br>    },<br>    "lr_scheduler_disc": "ExponentialLR",<br>    "lr_scheduler_disc_params": {<br>        "gamma": 0.999875,<br>        "last_epoch": -1<br>    },<br>    "kl_loss_alpha": 1.0,<br>    "disc_loss_alpha": 1.0,<br>    "gen_loss_alpha": 1.0,<br>    "feat_loss_alpha": 1.0,<br>    "mel_loss_alpha": 45.0,<br>    "dur_loss_alpha": 1.0,<br>    "speaker_encoder_loss_alpha": 1.0,<br>    "return_wav": true,<br>    "use_weighted_sampler": false,<br>    "weighted_sampler_attrs": {},<br>    "weighted_sampler_multipliers": {},<br>    "r": 1,<br>    "num_speakers": 98,<br>    "use_speaker_embedding": true,<br>    "speakers_file": "",<br>    "speaker_embedding_channels": 256,<br>    "language_ids_file": "",<br>    "use_language_embedding": true,<br>    "use_d_vector_file": false,<br>    "d_vector_file": null,<br>    "d_vector_dim": 0<br>}<br>```<br>I compared my config with the one from the pre-trained VITS VCTK. The only notable differences that I noticed were:<br>- My config has `grad_clip: [1000.0, 1000.0]` instead of `grad_clip: [5.0, 5.0]`<br>- The pre-trained model's config has a bunch of additional paramaters for the audio config, I'm not sure if those are default values or something that has been adjusted<br>- I use IPAPhonemes instead of VitsCharacters, but both configs use Espeak phonemizer<br>- The pre-trained model has `max_audio_len` set to Infinity<br>- My config has `compute_energy: false`, `shuffle: false`, and `drop_last: false`<br>- My config has defined `periods_multi_period_discriminator`<br>- My config has `init_discriminator: true`<br>- My config has `use_speaker_embedding: true` and `use_language_embedding: true`<br>Based on the logs, there seems to be a problem with the feats and KL loss.<br>Any ideas how to fix this issue? Should I adjust the learning rate, although the pre-trained had the same settings? Maybe there is some othere issue with the config? Currently, the training is at around 185K steps and the best model is saved at 80K steps. |

1You must be logged in to vote

All reactions

## Replies:   1 comment

Comment options

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/discussions/3377).

# {{title}}

Quote reply

### [![](https://avatars.githubusercontent.com/u/1402048?s=64&v=4)\ erogol](https://github.com/erogol) [on Dec 7, 2023Dec 7, 2023](https://github.com/coqui-ai/TTS/discussions/3377\#discussioncomment-7788257)   Maintainer

|     |
| --- |
| Check the audio quality. Eval loss is not always the best metric for the final quality. |

1You must be logged in to vote

All reactions

0 replies


[Sign up for free](https://github.com/join?source=comment-repo) **to join this conversation on GitHub**.
Already have an account?
[Sign in to comment](https://github.com/login?return_to=https%3A%2F%2Fgithub.com%2Fcoqui-ai%2FTTS%2Fdiscussions%2F3377)

Category


[❓\\
\\
General Q&A](https://github.com/coqui-ai/TTS/discussions/categories/general-q-a)

Labels


None yet


2 participants


[![@nicemanis](https://avatars.githubusercontent.com/u/9463816?s=48&v=4)](https://github.com/nicemanis)[![@erogol](https://avatars.githubusercontent.com/u/1402048?s=48&v=4)](https://github.com/erogol)

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

[Create a new saved reply](https://github.com/coqui-ai/TTS/discussions/3377)

👍1 reacted with thumbs up emoji👎1 reacted with thumbs down emoji😄1 reacted with laugh emoji🎉1 reacted with hooray emoji😕1 reacted with confused emoji❤️1 reacted with heart emoji🚀1 reacted with rocket emoji👀1 reacted with eyes emoji

You can’t perform that action at this time.