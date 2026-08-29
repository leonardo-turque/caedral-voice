---
url: https://cdn.jsdelivr.net/gh/coqui-ai/TTS@dev/TTS/tts/configs/shared_configs.py
title: 
retrieved_at: 2026-08-22T20:07:24Z
api_version: v2
---

from dataclasses import asdict, dataclass, field
from typing import Dict, List

from coqpit import Coqpit, check\_argument

from TTS.config import BaseAudioConfig, BaseDatasetConfig, BaseTrainingConfig

@dataclass
class GSTConfig(Coqpit):
 """Defines the Global Style Token Module

 Args:
 gst\_style\_input\_wav (str):
 Path to the wav file used to define the style of the output speech at inference. Defaults to None.

 gst\_style\_input\_weights (dict):
 Defines the weights for each style token used at inference. Defaults to None.

 gst\_embedding\_dim (int):
 Defines the size of the GST embedding vector dimensions. Defaults to 256.

 gst\_num\_heads (int):
 Number of attention heads used by the multi-head attention. Defaults to 4.

 gst\_num\_style\_tokens (int):
 Number of style token vectors. Defaults to 10.
 """

 gst\_style\_input\_wav: str = None
 gst\_style\_input\_weights: dict = None
 gst\_embedding\_dim: int = 256
 gst\_use\_speaker\_embedding: bool = False
 gst\_num\_heads: int = 4
 gst\_num\_style\_tokens: int = 10

 def check\_values(
 self,
 ):
 """Check config fields"""
 c = asdict(self)
 super().check\_values()
 check\_argument("gst\_style\_input\_weights", c, restricted=False)
 check\_argument("gst\_style\_input\_wav", c, restricted=False)
 check\_argument("gst\_embedding\_dim", c, restricted=True, min\_val=0, max\_val=1000)
 check\_argument("gst\_use\_speaker\_embedding", c, restricted=False)
 check\_argument("gst\_num\_heads", c, restricted=True, min\_val=2, max\_val=10)
 check\_argument("gst\_num\_style\_tokens", c, restricted=True, min\_val=1, max\_val=1000)

@dataclass
class CapacitronVAEConfig(Coqpit):
 """Defines the capacitron VAE Module
 Args:
 capacitron\_capacity (int):
 Defines the variational capacity limit of the prosody embeddings. Defaults to 150.
 capacitron\_VAE\_embedding\_dim (int):
 Defines the size of the Capacitron embedding vector dimension. Defaults to 128.
 capacitron\_use\_text\_summary\_embeddings (bool):
 If True, use a text summary embedding in Capacitron. Defaults to True.
 capacitron\_text\_summary\_embedding\_dim (int):
 Defines the size of the capacitron text embedding vector dimension. Defaults to 128.
 capacitron\_use\_speaker\_embedding (bool):
 if True use speaker embeddings in Capacitron. Defaults to False.
 capacitron\_VAE\_loss\_alpha (float):
 Weight for the VAE loss of the Tacotron model. If set less than or equal to zero, it disables the
 corresponding loss function. Defaults to 0.25
 capacitron\_grad\_clip (float):
 Gradient clipping value for all gradients except beta. Defaults to 5.0
 """

 capacitron\_loss\_alpha: int = 1
 capacitron\_capacity: int = 150
 capacitron\_VAE\_embedding\_dim: int = 128
 capacitron\_use\_text\_summary\_embeddings: bool = True
 capacitron\_text\_summary\_embedding\_dim: int = 128
 capacitron\_use\_speaker\_embedding: bool = False
 capacitron\_VAE\_loss\_alpha: float = 0.25
 capacitron\_grad\_clip: float = 5.0

 def check\_values(
 self,
 ):
 """Check config fields"""
 c = asdict(self)
 super().check\_values()
 check\_argument("capacitron\_capacity", c, restricted=True, min\_val=10, max\_val=500)
 check\_argument("capacitron\_VAE\_embedding\_dim", c, restricted=True, min\_val=16, max\_val=1024)
 check\_argument("capacitron\_use\_speaker\_embedding", c, restricted=False)
 check\_argument("capacitron\_text\_summary\_embedding\_dim", c, restricted=False, min\_val=16, max\_val=512)
 check\_argument("capacitron\_VAE\_loss\_alpha", c, restricted=False)
 check\_argument("capacitron\_grad\_clip", c, restricted=False)

@dataclass
class CharactersConfig(Coqpit):
 """Defines arguments for the \`BaseCharacters\` or \`BaseVocabulary\` and their subclasses.

 Args:
 characters\_class (str):
 Defines the class of the characters used. If None, we pick \`\`\`Phonemes\`\`\` or \`\`\`Graphemes\`\`\` based on
 the configuration. Defaults to None.

 vocab\_dict (dict):
 Defines the vocabulary dictionary used to encode the characters. Defaults to None.

 pad (str):
 characters in place of empty padding. Defaults to None.

 eos (str):
 characters showing the end of a sentence. Defaults to None.

 bos (str):
 characters showing the beginning of a sentence. Defaults to None.

 blank (str):
 Optional character used between characters by some models for better prosody. Defaults to \`\_blank\`.

 characters (str):
 character set used by the model. Characters not in this list are ignored when converting input text to
 a list of sequence IDs. Defaults to None.

 punctuations (str):
 characters considered as punctuation as parsing the input sentence. Defaults to None.

 phonemes (str):
 characters considered as parsing phonemes. This is only for backwards compat. Use \`characters\` for new
 models. Defaults to None.

 is\_unique (bool):
 remove any duplicate characters in the character lists. It is a bandaid for compatibility with the old
 models trained with character lists with duplicates. Defaults to True.

 is\_sorted (bool):
 Sort the characters in alphabetical order. Defaults to True.
 """

 characters\_class: str = None

 # using BaseVocabulary
 vocab\_dict: Dict = None

 # using on BaseCharacters
 pad: str = None
 eos: str = None
 bos: str = None
 blank: str = None
 characters: str = None
 punctuations: str = None
 phonemes: str = None
 is\_unique: bool = True # for backwards compatibility of models trained with char sets with duplicates
 is\_sorted: bool = True

@dataclass
class BaseTTSConfig(BaseTrainingConfig):
 """Shared parameters among all the tts models.

 Args:

 audio (BaseAudioConfig):
 Audio processor config object instance.

 use\_phonemes (bool):
 enable / disable phoneme use.

 phonemizer (str):
 Name of the phonemizer to use. If set None, the phonemizer will be selected by \`phoneme\_language\`.
 Defaults to None.

 phoneme\_language (str):
 Language code for the phonemizer. You can check the list of supported languages by running
 \`python TTS/tts/utils/text/phonemizers/\_\_init\_\_.py\`. Defaults to None.

 compute\_input\_seq\_cache (bool):
 enable / disable precomputation of the phoneme sequences. At the expense of some delay at the beginning of
 the training, It allows faster data loader time and precise limitation with \`max\_seq\_len\` and
 \`min\_seq\_len\`.

 text\_cleaner (str):
 Name of the text cleaner used for cleaning and formatting transcripts.

 enable\_eos\_bos\_chars (bool):
 enable / disable the use of eos and bos characters.

 test\_senteces\_file (str):
 Path to a txt file that has sentences used at test time. The file must have a sentence per line.

 phoneme\_cache\_path (str):
 Path to the output folder caching the computed phonemes for each sample.

 characters (CharactersConfig):
 Instance of a CharactersConfig class.

 batch\_group\_size (int):
 Size of the batch groups used for bucketing. By default, the dataloader orders samples by the sequence
 length for a more efficient and stable training. If \`batch\_group\_size > 1\` then it performs bucketing to
 prevent using the same batches for each epoch.

 loss\_masking (bool):
 enable / disable masking loss values against padded segments of samples in a batch.

 min\_text\_len (int):
 Minimum length of input text to be used. All shorter samples will be ignored. Defaults to 0.

 max\_text\_len (int):
 Maximum length of input text to be used. All longer samples will be ignored. Defaults to float("inf").

 min\_audio\_len (int):
 Minimum length of input audio to be used. All shorter samples will be ignored. Defaults to 0.

 max\_audio\_len (int):
 Maximum length of input audio to be used. All longer samples will be ignored. The maximum length in the
 dataset defines the VRAM used in the training. Hence, pay attention to this value if you encounter an
 OOM error in training. Defaults to float("inf").

 compute\_f0 (int):
 (Not in use yet).

 compute\_energy (int):
 (Not in use yet).

 compute\_linear\_spec (bool):
 If True data loader computes and returns linear spectrograms alongside the other data.

 precompute\_num\_workers (int):
 Number of workers to precompute features. Defaults to 0.

 use\_noise\_augment (bool):
 Augment the input audio with random noise.

 start\_by\_longest (bool):
 If True, the data loader will start loading the longest batch first. It is useful for checking OOM issues.
 Defaults to False.

 shuffle (bool):
 If True, the data loader will shuffle the dataset when there is not sampler defined. Defaults to True.

 drop\_last (bool):
 If True, the data loader will drop the last batch if it is not complete. It helps to prevent
 issues that emerge from the partial batch statistics. Defaults to True.

 add\_blank (bool):
 Add blank characters between each other two characters. It improves performance for some models at expense
 of slower run-time due to the longer input sequence.

 datasets (List\[BaseDatasetConfig\]):
 List of datasets used for training. If multiple datasets are provided, they are merged and used together
 for training.

 optimizer (str):
 Optimizer used for the training. Set one from \`torch.optim.Optimizer\` or \`TTS.utils.training\`.
 Defaults to \`\`.

 optimizer\_params (dict):
 Optimizer kwargs. Defaults to \`{"betas": \[0.8, 0.99\], "weight\_decay": 0.0}\`

 lr\_scheduler (str):
 Learning rate scheduler for the training. Use one from \`torch.optim.Scheduler\` schedulers or
 \`TTS.utils.training\`. Defaults to \`\`.

 lr\_scheduler\_params (dict):
 Parameters for the generator learning rate scheduler. Defaults to \`{"warmup": 4000}\`.

 test\_sentences (List\[str\]):
 List of sentences to be used at testing. Defaults to '\[\]'

 eval\_split\_max\_size (int):
 Number maximum of samples to be used for evaluation in proportion split. Defaults to None (Disabled).

 eval\_split\_size (float):
 If between 0.0 and 1.0 represents the proportion of the dataset to include in the evaluation set.
 If > 1, represents the absolute number of evaluation samples. Defaults to 0.01 (1%).

 use\_speaker\_weighted\_sampler (bool):
 Enable / Disable the batch balancer by speaker. Defaults to \`\`\`False\`\`\`.

 speaker\_weighted\_sampler\_alpha (float):
 Number that control the influence of the speaker sampler weights. Defaults to \`\`\`1.0\`\`\`.

 use\_language\_weighted\_sampler (bool):
 Enable / Disable the batch balancer by language. Defaults to \`\`\`False\`\`\`.

 language\_weighted\_sampler\_alpha (float):
 Number that control the influence of the language sampler weights. Defaults to \`\`\`1.0\`\`\`.

 use\_length\_weighted\_sampler (bool):
 Enable / Disable the batch balancer by audio length. If enabled the dataset will be divided
 into 10 buckets considering the min and max audio of the dataset. The sampler weights will be
 computed forcing to have the same quantity of data for each bucket in each training batch. Defaults to \`\`\`False\`\`\`.

 length\_weighted\_sampler\_alpha (float):
 Number that control the influence of the length sampler weights. Defaults to \`\`\`1.0\`\`\`.
 """

 audio: BaseAudioConfig = field(default\_factory=BaseAudioConfig)
 # phoneme settings
 use\_phonemes: bool = False
 phonemizer: str = None
 phoneme\_language: str = None
 compute\_input\_seq\_cache: bool = False
 text\_cleaner: str = None
 enable\_eos\_bos\_chars: bool = False
 test\_sentences\_file: str = ""
 phoneme\_cache\_path: str = None
 # vocabulary parameters
 characters: CharactersConfig = None
 add\_blank: bool = False
 # training params
 batch\_group\_size: int = 0
 loss\_masking: bool = None
 # dataloading
 min\_audio\_len: int = 1
 max\_audio\_len: int = float("inf")
 min\_text\_len: int = 1
 max\_text\_len: int = float("inf")
 compute\_f0: bool = False
 compute\_energy: bool = False
 compute\_linear\_spec: bool = False
 precompute\_num\_workers: int = 0
 use\_noise\_augment: bool = False
 start\_by\_longest: bool = False
 shuffle: bool = False
 drop\_last: bool = False
 # dataset
 datasets: List\[BaseDatasetConfig\] = field(default\_factory=lambda: \[BaseDatasetConfig()\])
 # optimizer
 optimizer: str = "radam"
 optimizer\_params: dict = None
 # scheduler
 lr\_scheduler: str = None
 lr\_scheduler\_params: dict = field(default\_factory=lambda: {})
 # testing
 test\_sentences: List\[str\] = field(default\_factory=lambda: \[\])
 # evaluation
 eval\_split\_max\_size: int = None
 eval\_split\_size: float = 0.01
 # weighted samplers
 use\_speaker\_weighted\_sampler: bool = False
 speaker\_weighted\_sampler\_alpha: float = 1.0
 use\_language\_weighted\_sampler: bool = False
 language\_weighted\_sampler\_alpha: float = 1.0
 use\_length\_weighted\_sampler: bool = False
 length\_weighted\_sampler\_alpha: float = 1.0