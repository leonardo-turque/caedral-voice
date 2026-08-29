---
url: https://huggingface.co/docs/transformers/en/model_doc/vits
title: VITS · Hugging Face
retrieved_at: 2026-08-22T20:03:40Z
api_version: v2
---

Transformers documentation

VITS

# Transformers

🏡 View all docsAWS Trainium & InferentiaAccelerateArgillaAutoTrainBitsandbytesCLIChat UIDataset viewerDatasetsDeploying on AWSDiffusersDistilabelEvaluateGoogle CloudGoogle TPUsGradioHubHub Python LibraryHuggingface.jsInference Endpoints (dedicated)Inference ProvidersKernelsLeRobotLeaderboardsLightevalMicrosoft AzureOpenEnvOptimumPEFTReachy MiniSafetensorsSentence TransformersTRLTasksText Embeddings InferenceText Generation InferenceTokenizersTrackioTransformersTransformers.jsXetsmolagentstimm

Search documentation
`Ctrl+K`

mainv5.15.1v5.14.0v5.13.1v5.12.0v5.11.0v5.10.4v5.9.0v5.8.1v5.7.0v5.6.2v5.5.4v5.4.0v5.3.0v5.2.0v5.1.0v5.0.0v4.57.6v4.56.2v4.55.4v4.53.3v4.52.3v4.51.3v4.50.0v4.49.0v4.48.2v4.47.1v4.46.3v4.45.2v4.44.2v4.43.4v4.42.4v4.41.2v4.40.2v4.39.3v4.38.2v4.37.2v4.36.1v4.35.2v4.34.1v4.33.3v4.32.1v4.31.0v4.30.0v4.29.1v4.28.1v4.27.2v4.26.1v4.25.1v4.24.0v4.23.1v4.22.2v4.21.3v4.20.1v4.19.4v4.18.0v4.17.0v4.16.2v4.15.0v4.14.1v4.13.0v4.12.5v4.11.3v4.10.1v4.9.2v4.8.2v4.7.0v4.6.0v4.5.1v4.4.2v4.3.3v4.2.2v4.1.1v4.0.1v3.5.1v3.4.0v3.3.1v3.2.0v3.1.0v3.0.2v2.11.0v2.10.0v2.9.1v2.8.0v2.7.0v2.6.0v2.5.1v2.4.1v2.3.0v2.2.2v2.1.1v2.0.0v1.2.0v1.1.0v1.0.0doc-builder-htmlARDEENESFRHIITJAKOPTROTRZH

![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)

Join the Hugging Face community

and get access to the augmented documentation experience

Collaborate on models, datasets and Spaces

Faster examples with accelerated inference

Switch between documentation themes

[Sign Up](https://huggingface.co/join)

to get started

_This model was published in HF papers on 2021-06-11 and contributed to Hugging Face Transformers on 2023-09-01._

Copy page

# VITS

[VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)](https://huggingface.co/papers/2106.06103) is an end-to-end speech synthesis model, simplifying the traditional two-stage text-to-speech (TTS) systems. It’s unique because it directly synthesizes speech from text using variational inference, adversarial learning, and normalizing flows to produce natural and expressive speech with diverse rhythms and intonations.

You can find all the original VITS checkpoints under the [AI at Meta](https://huggingface.co/facebook?search_models=mms-tts) organization.

> Click on the VITS models in the right sidebar for more examples of how to apply VITS.

The example below demonstrates how to generate text based on an image with [Pipeline](https://huggingface.co/docs/transformers/v5.15.1/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel

Copied

```python
from scipy.io.wavfile import write

from transformers import pipeline, set_seed

set_seed(555)

pipe = pipeline(
    task="text-to-speech",
    model="facebook/mms-tts-eng",
    device=0
)

speech = pipe("Hello, my dog is cute")

# Extract audio data and sampling rate
audio_data = speech["audio"]
sampling_rate = speech["sampling_rate"]

# Save as WAV file
write("hello.wav", sampling_rate, audio_data.squeeze())
```

## Notes

- Set a seed for reproducibility because VITS synthesizes speech non-deterministically.

- For languages with non-Roman alphabets (Korean, Arabic, etc.), install the [uroman](https://github.com/isi-nlp/uroman) package to preprocess the text inputs to the Roman alphabet. You can check if the tokenizer requires uroman as shown below.








Copied




```py
# pip install -U uroman
from transformers import VitsTokenizer

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
print(tokenizer.is_uroman)
```



If your language requires uroman, the tokenizer automatically applies it to the text inputs. Python >= 3.10 doesn’t require any additional preprocessing steps. For Python < 3.10, follow the steps below.








Copied




```bash
git clone https://github.com/isi-nlp/uroman.git
cd uroman
export UROMAN=$(pwd)
```



Create a function to preprocess the inputs. You can either use the bash variable `UROMAN` or pass the directory path directly to the function.








Copied




```py
import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import os
import subprocess

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kor")
model = VitsModel.from_pretrained("facebook/mms-tts-kor", device_map="auto")

def uromanize(input_string, uroman_path):
      """Convert non-Roman strings to Roman using the `uroman` perl package."""
      script_path = os.path.join(uroman_path, "bin", "uroman.pl")

      command = ["perl", script_path]

      process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      # Execute the perl command
      stdout, stderr = process.communicate(input=input_string.encode())

      if process.returncode != 0:
          raise ValueError(f"Error {process.returncode}: {stderr.decode()}")

      # Return the output as a string and skip the new-line character at the end
      return stdout.decode()[:-1]

text = "이봐 무슨 일이야"
uromanized_text = uromanize(text, uroman_path=os.environ["UROMAN"])

inputs = tokenizer(text=uromanized_text, return_tensors="pt").to(model.device)

set_seed(555)  # make deterministic
with torch.no_grad():
     outputs = model(inputs["input_ids"])

waveform = outputs.waveform[0]
```


## VitsConfig

### classtransformers.VitsConfig

[<source>](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/vits/configuration_vits.py#L24)

(transformers\_version: str \| None = Nonearchitectures: list\[str\] \| None = Noneoutput\_hidden\_states: bool \| None = Falsereturn\_dict: bool \| None = Truedtype: typing.Union\[str, ForwardRef('torch.dtype'), NoneType\] = Nonechunk\_size\_feed\_forward: int = 0is\_encoder\_decoder: bool = Falseid2label: dict\[int, str\] \| dict\[str, str\] \| None = Nonelabel2id: dict\[str, int\] \| dict\[str, str\] \| None = Noneproblem\_type: typing.Optional\[typing.Literal\['regression', 'single\_label\_classification', 'multi\_label\_classification'\]\] = Nonevocab\_size: int = 38hidden\_size: int = 192num\_hidden\_layers: int = 6num\_attention\_heads: int = 2window\_size: int = 4use\_bias: bool = Trueffn\_dim: int = 768layerdrop: float \| int = 0.1ffn\_kernel\_size: int = 3flow\_size: int = 192spectrogram\_bins: int = 513hidden\_act: str = 'relu'hidden\_dropout: float \| int = 0.1attention\_dropout: float \| int = 0.1activation\_dropout: float \| int = 0.1initializer\_range: float = 0.02layer\_norm\_eps: float = 1e-05use\_stochastic\_duration\_prediction: bool = Truenum\_speakers: int = 1speaker\_embedding\_size: int = 0upsample\_initial\_channel: int = 512upsample\_rates: list\[int\] \| tuple\[int, ...\] = (8, 8, 2, 2)upsample\_kernel\_sizes: list\[int\] \| tuple\[int, ...\] = (16, 16, 4, 4)resblock\_kernel\_sizes: list\[int\] \| tuple\[int, ...\] = (3, 7, 11)resblock\_dilation\_sizes: list \| tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))leaky\_relu\_slope: float = 0.1depth\_separable\_channels: int = 2depth\_separable\_num\_layers: int = 3duration\_predictor\_flow\_bins: int = 10duration\_predictor\_tail\_bound: float = 5.0duration\_predictor\_kernel\_size: int = 3duration\_predictor\_dropout: float \| int = 0.5duration\_predictor\_num\_flows: int = 4duration\_predictor\_filter\_channels: int = 256prior\_encoder\_num\_flows: int = 4prior\_encoder\_num\_wavenet\_layers: int = 4posterior\_encoder\_num\_wavenet\_layers: int = 16wavenet\_kernel\_size: int = 5wavenet\_dilation\_rate: int = 1wavenet\_dropout: float \| int = 0.0speaking\_rate: float \| int = 1.0noise\_scale: float = 0.667noise\_scale\_duration: float = 0.8sampling\_rate: int = 16000pad\_token\_id: int \| None = None)

Expand 45 parameters

Parameters

- **vocab\_size** (`int`, _optional_, defaults to `38`) —
Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.
- **hidden\_size** (`int`, _optional_, defaults to `192`) —
Dimension of the hidden representations.
- **num\_hidden\_layers** (`int`, _optional_, defaults to `6`) —
Number of hidden layers in the Transformer decoder.
- **num\_attention\_heads** (`int`, _optional_, defaults to `2`) —
Number of attention heads for each attention layer in the Transformer decoder.
- **window\_size** (`int`, _optional_, defaults to 4) —
Window size for the relative positional embeddings in the attention layers of the Transformer encoder.
- **use\_bias** (`bool`, _optional_, defaults to `True`) —
Whether to use bias in the key, query, value projection layers in the Transformer encoder.
- **ffn\_dim** (`int`, _optional_, defaults to `768`) —
Dimension of the MLP representations.
- **layerdrop** (`Union[float, int]`, _optional_, defaults to `0.1`) —
The LayerDrop probability. See the \[LayerDrop paper\](see [https://huggingface.co/papers/1909.11556](https://huggingface.co/papers/1909.11556)) for
more details.
- **ffn\_kernel\_size** (`int`, _optional_, defaults to 3) —
Kernel size of the 1D convolution layers used by the feed-forward network in the Transformer encoder.
- **flow\_size** (`int`, _optional_, defaults to 192) —
Dimensionality of the flow layers.
- **spectrogram\_bins** (`int`, _optional_, defaults to 513) —
Number of frequency bins in the target spectrogram.
- **hidden\_act** (`str`, _optional_, defaults to `relu`) —
The non-linear activation function (function or string) in the decoder. For example, `"gelu"`,
`"relu"`, `"silu"`, etc.
- **hidden\_dropout** (`Union[float, int]`, _optional_, defaults to `0.1`) —
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
- **attention\_dropout** (`Union[float, int]`, _optional_, defaults to `0.1`) —
The dropout ratio for the attention probabilities.
- **activation\_dropout** (`Union[float, int]`, _optional_, defaults to `0.1`) —
The dropout ratio for activations inside the fully connected layer.
- **initializer\_range** (`float`, _optional_, defaults to `0.02`) —
The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
- **layer\_norm\_eps** (`float`, _optional_, defaults to `1e-05`) —
The epsilon used by the layer normalization layers.
- **use\_stochastic\_duration\_prediction** (`bool`, _optional_, defaults to `True`) —
Whether to use the stochastic duration prediction module or the regular duration predictor.
- **num\_speakers** (`int`, _optional_, defaults to 1) —
Number of speakers if this is a multi-speaker model.
- **speaker\_embedding\_size** (`int`, _optional_, defaults to 0) —
Number of channels used by the speaker embeddings. Is zero for single-speaker models.
- **upsample\_initial\_channel** (`int`, _optional_, defaults to 512) —
The number of input channels into the HiFi-GAN upsampling network.
- **upsample\_rates** (`tuple[int]` or `list[int]`, _optional_, defaults to `[8, 8, 2, 2]`) —
A tuple of integers defining the stride of each 1D convolutional layer in the HiFi-GAN upsampling network.
The length of `upsample_rates` defines the number of convolutional layers and has to match the length of
`upsample_kernel_sizes`.
- **upsample\_kernel\_sizes** (`tuple[int]` or `list[int]`, _optional_, defaults to `[16, 16, 4, 4]`) —
A tuple of integers defining the kernel size of each 1D convolutional layer in the HiFi-GAN upsampling
network. The length of `upsample_kernel_sizes` defines the number of convolutional layers and has to match
the length of `upsample_rates`.
- **resblock\_kernel\_sizes** (`tuple[int]` or `list[int]`, _optional_, defaults to `[3, 7, 11]`) —
A tuple of integers defining the kernel sizes of the 1D convolutional layers in the HiFi-GAN
multi-receptive field fusion (MRF) module.
- **resblock\_dilation\_sizes** (`tuple[tuple[int]]` or `list[list[int]]`, _optional_, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`) —
A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
HiFi-GAN multi-receptive field fusion (MRF) module.
- **leaky\_relu\_slope** (`float`, _optional_, defaults to 0.1) —
The angle of the negative slope used by the leaky ReLU activation.
- **depth\_separable\_channels** (`int`, _optional_, defaults to 2) —
Number of channels to use in each depth-separable block.
- **depth\_separable\_num\_layers** (`int`, _optional_, defaults to 3) —
Number of convolutional layers to use in each depth-separable block.
- **duration\_predictor\_flow\_bins** (`int`, _optional_, defaults to 10) —
Number of channels to map using the unonstrained rational spline in the duration predictor model.
- **duration\_predictor\_tail\_bound** (`float`, _optional_, defaults to 5.0) —
Value of the tail bin boundary when computing the unconstrained rational spline in the duration predictor
model.
- **duration\_predictor\_kernel\_size** (`int`, _optional_, defaults to 3) —
Kernel size of the 1D convolution layers used in the duration predictor model.
- **duration\_predictor\_dropout** (`float`, _optional_, defaults to 0.5) —
The dropout ratio for the duration predictor model.
- **duration\_predictor\_num\_flows** (`int`, _optional_, defaults to 4) —
Number of flow stages used by the duration predictor model.
- **duration\_predictor\_filter\_channels** (`int`, _optional_, defaults to 256) —
Number of channels for the convolution layers used in the duration predictor model.
- **prior\_encoder\_num\_flows** (`int`, _optional_, defaults to 4) —
Number of flow stages used by the prior encoder flow model.
- **prior\_encoder\_num\_wavenet\_layers** (`int`, _optional_, defaults to 4) —
Number of WaveNet layers used by the prior encoder flow model.
- **posterior\_encoder\_num\_wavenet\_layers** (`int`, _optional_, defaults to 16) —
Number of WaveNet layers used by the posterior encoder model.
- **wavenet\_kernel\_size** (`int`, _optional_, defaults to 5) —
Kernel size of the 1D convolution layers used in the WaveNet model.
- **wavenet\_dilation\_rate** (`int`, _optional_, defaults to 1) —
Dilation rates of the dilated 1D convolutional layers used in the WaveNet model.
- **wavenet\_dropout** (`float`, _optional_, defaults to 0.0) —
The dropout ratio for the WaveNet layers.
- **speaking\_rate** (`float`, _optional_, defaults to 1.0) —
Speaking rate. Larger values give faster synthesised speech.
- **noise\_scale** (`float`, _optional_, defaults to 0.667) —
How random the speech prediction is. Larger values create more variation in the predicted speech.
- **noise\_scale\_duration** (`float`, _optional_, defaults to 0.8) —
How random the duration prediction is. Larger values create more variation in the predicted durations.
- **sampling\_rate** (`int`, _optional_, defaults to `16000`) —
The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
- **pad\_token\_id** (`int`, _optional_) —
Token id used for padding in the vocabulary.

This is the configuration class to store the configuration of a VitsModel. It is used to instantiate a Vits
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng)

Configuration objects inherit from [PreTrainedConfig](https://huggingface.co/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](https://huggingface.co/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

Copied

```python
>>> from transformers import VitsModel, VitsConfig

>>> # Initializing a "facebook/mms-tts-eng" style configuration
>>> configuration = VitsConfig()

>>> # Initializing a model (with random weights) from the "facebook/mms-tts-eng" style configuration
>>> model = VitsModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## VitsTokenizer

### classtransformers.VitsTokenizer

[<source>](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/vits/tokenization_vits.py#L46)

(vocab\_filepad\_token = '<pad>'unk\_token = '<unk>'language = Noneadd\_blank = Truenormalize = Truephonemize = Trueis\_uroman = False\*\*kwargs)

Parameters

- **vocab\_file** (`str`) —
Path to the vocabulary file.
- **language** (`str`, _optional_) —
Language identifier.
- **add\_blank** (`bool`, _optional_, defaults to `True`) —
Whether to insert token id 0 in between the other tokens.
- **normalize** (`bool`, _optional_, defaults to `True`) —
Whether to normalize the input text by removing all casing and punctuation.
- **phonemize** (`bool`, _optional_, defaults to `True`) —
Whether to convert the input text into phonemes.
- **is\_uroman** (`bool`, _optional_, defaults to `False`) —
Whether the `uroman` Romanizer needs to be applied to the input text prior to tokenizing.

Construct a VITS tokenizer. Also supports MMS-TTS.

This tokenizer inherits from [PreTrainedTokenizer](https://huggingface.co/docs/transformers/v5.15.1/en/main_classes/tokenizer#transformers.PythonBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### \_\_call\_\_

[<source>](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/tokenization_utils_base.py#L2417)

(text: TextInput \| PreTokenizedInput \| list\[TextInput\] \| list\[PreTokenizedInput\] \| None = Nonetext\_pair: TextInput \| PreTokenizedInput \| list\[TextInput\] \| list\[PreTokenizedInput\] \| None = Nonetext\_target: TextInput \| PreTokenizedInput \| list\[TextInput\] \| list\[PreTokenizedInput\] \| None = Nonetext\_pair\_target: TextInput \| PreTokenizedInput \| list\[TextInput\] \| list\[PreTokenizedInput\] \| None = Noneadd\_special\_tokens: bool = Truepadding: bool \| str \| PaddingStrategy = Falsetruncation: bool \| str \| TruncationStrategy \| None = Nonemax\_length: int \| None = Nonestride: int = 0is\_split\_into\_words: bool = Falsepad\_to\_multiple\_of: int \| None = Nonepadding\_side: str \| None = Nonereturn\_tensors: str \| TensorType \| None = Nonereturn\_token\_type\_ids: bool \| None = Nonereturn\_attention\_mask: bool \| None = Nonereturn\_overflowing\_tokens: bool = Falsereturn\_special\_tokens\_mask: bool = Falsereturn\_offsets\_mapping: bool = Falsereturn\_length: bool = Falseverbose: bool = Truetokenizer\_kwargs: dict\[str, Any\] \| None = None\*\*kwargs)→[BatchEncoding](https://huggingface.co/docs/transformers/v5.15.1/en/main_classes/tokenizer#transformers.BatchEncoding)

Expand 22 parameters

Parameters

- **text** (`str`, `list[str]`, `list[list[str]]`, _optional_) —
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **text\_pair** (`str`, `list[str]`, `list[list[str]]`, _optional_) —
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **text\_target** (`str`, `list[str]`, `list[list[str]]`, _optional_) —
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **text\_pair\_target** (`str`, `list[str]`, `list[list[str]]`, _optional_) —
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **tokenizer\_kwargs** (`dict[str, Any]`, _optional_) —
Additional kwargs to pass to the tokenizer. These will be merged with the explicit parameters and
other kwargs, with explicit parameters taking precedence.
- **add\_special\_tokens** (`bool`, _optional_, defaults to `True`) —
Whether or not to add special tokens when encoding the sequences. This will use the underlying
`PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens
automatically.
- **padding** (`bool`, `str` or [PaddingStrategy](https://huggingface.co/docs/transformers/v5.15.1/en/internal/file_utils#transformers.utils.PaddingStrategy), _optional_, defaults to `False`) —
Activates and controls padding. Accepts the following values:



  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence is provided).
  - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).

- **truncation** (`bool`, `str` or [TruncationStrategy](https://huggingface.co/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), _optional_, defaults to `False`) —
Activates and controls truncation. Accepts the following values:



  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).

- **max\_length** (`int`, _optional_) —
Controls the maximum length to use by one of the truncation/padding parameters.


If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.

- **stride** (`int`, _optional_, defaults to 0) —
If set to a number along with `max_length`, the overflowing tokens returned when
`return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.
- **is\_split\_into\_words** (`bool`, _optional_, defaults to `False`) —
Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.
- **pad\_to\_multiple\_of** (`int`, _optional_) —
If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
`>= 7.5` (Volta).
- **padding\_side** (`str`, _optional_) —
The side on which the model should have padding applied. Should be selected between \[‘right’, ‘left’\].
Default value is picked from the class attribute of the same name.
- **return\_tensors** (`str` or [TensorType](https://huggingface.co/docs/transformers/v5.15.1/en/internal/file_utils#transformers.TensorType), _optional_) —
If set, will return tensors instead of list of python integers. Acceptable values are:



  - `'pt'`: Return PyTorch `torch.Tensor` objects.
  - `'np'`: Return Numpy `np.ndarray` objects.

- **return\_token\_type\_ids** (`bool`, _optional_) —
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer’s default, defined by the `return_outputs` attribute.


[What are token type IDs?](https://huggingface.co/docs/transformers/en/glossary#token-type-ids)

- **return\_attention\_mask** (`bool`, _optional_) —
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer’s default, defined by the `return_outputs` attribute.


[What are attention masks?](https://huggingface.co/docs/transformers/en/glossary#attention-mask)

- **return\_overflowing\_tokens** (`bool`, _optional_, defaults to `False`) —
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
of returning overflowing tokens.
- **return\_special\_tokens\_mask** (`bool`, _optional_, defaults to `False`) —
Whether or not to return special tokens mask information.
- **return\_offsets\_mapping** (`bool`, _optional_, defaults to `False`) —
Whether or not to return `(char_start, char_end)` for each token.


This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](https://huggingface.co/docs/transformers/v5.15.1/en/main_classes/tokenizer#transformers.TokenizersBackend), if using
Python’s tokenizer, this method will raise `NotImplementedError`.

- **return\_length** (`bool`, _optional_, defaults to `False`) —
Whether or not to return the lengths of the encoded inputs.
- **verbose** (`bool`, _optional_, defaults to `True`) —
Whether or not to print more information and warnings.
- \* **\*kwargs** — passed to the `self.tokenize()` method

Returns

[BatchEncoding](https://huggingface.co/docs/transformers/v5.15.1/en/main_classes/tokenizer#transformers.BatchEncoding)

A [BatchEncoding](https://huggingface.co/docs/transformers/v5.15.1/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

- **input\_ids** — List of token ids to be fed to a model.

[What are input IDs?](https://huggingface.co/docs/transformers/en/glossary#input-ids)

- **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or
if _“token\_type\_ids”_ is in `self.model_input_names`).

[What are token type IDs?](https://huggingface.co/docs/transformers/en/glossary#token-type-ids)

- **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
`return_attention_mask=True` or if _“attention\_mask”_ is in `self.model_input_names`).

[What are attention masks?](https://huggingface.co/docs/transformers/en/glossary#attention-mask)

- **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and
`return_overflowing_tokens=True`).

- **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and
`return_overflowing_tokens=True`).

- **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).

- **length** — The length of the inputs (when `return_length=True`)


Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.

#### save\_vocabulary

[<source>](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/vits/tokenization_vits.py#L233)

(save\_directory: strfilename\_prefix: str \| None = None)

## VitsModel

### classtransformers.VitsModel

[<source>](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/vits/modeling_vits.py#L1231)

(config: VitsConfig)

Parameters

- **config** ( [VitsConfig](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/vits#transformers.VitsConfig)) —
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
[from\_pretrained()](https://huggingface.co/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The complete VITS model, for text-to-speech synthesis.

This model inherits from [PreTrainedModel](https://huggingface.co/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

[<source>](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/vits/modeling_vits.py#L1258)

(input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonespeaker\_id: int \| None = Noneoutput\_attentions: bool \| None = Noneoutput\_hidden\_states: bool \| None = Nonereturn\_dict: bool \| None = Nonelabels: typing.Optional\[torch.FloatTensor\] = Nonespeaking\_rate: float \| None = None\*\*kwargs)→`VitsModelOutput` or `tuple(torch.FloatTensor)`

Parameters

- **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) —
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.


Indices can be obtained using [AutoTokenizer](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](https://huggingface.co/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
[PreTrainedTokenizer. **call**()](https://huggingface.co/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

[What are input IDs?](https://huggingface.co/docs/transformers/en/glossary#input-ids)

- **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) —
Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:



  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

[What are attention masks?](https://huggingface.co/docs/transformers/en/glossary#attention-mask)

- **speaker\_id** (`int`, _optional_) —
Which speaker embedding to use. Only used for multispeaker models.
- **output\_attentions** (`bool`, _optional_) —
Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
tensors for more detail.
- **output\_hidden\_states** (`bool`, _optional_) —
Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
more detail.
- **return\_dict** (`bool`, _optional_) —
Whether or not to return a [ModelOutput](https://huggingface.co/docs/transformers/v5.15.1/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **labels** (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`, _optional_) —
Float values of target spectrogram. Timesteps set to `-100.0` are ignored (masked) for the loss
computation.
- **speaking\_rate** (`float`, _optional_) —
Speaking rate.

Returns

`VitsModelOutput` or `tuple(torch.FloatTensor)`

A `VitsModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ( [VitsConfig](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/vits#transformers.VitsConfig)) and inputs.

The [VitsModel](https://huggingface.co/docs/transformers/v5.15.1/en/model_doc/vits#transformers.VitsModel) forward method, overrides the `__call__` special method.

> Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while
> the latter silently ignores them.

- **waveform** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — The final audio waveform predicted by the model.

- **sequence\_lengths** (`torch.FloatTensor` of shape `(batch_size,)`) — The length in samples of each element in the `waveform` batch.

- **spectrogram** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`) — The log-mel spectrogram predicted at the output of the flow model. This spectrogram is passed to the Hi-Fi
GAN decoder model to obtain the final audio waveform.

- **hidden\_states** (`tuple[torch.FloatTensor]`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

- **attentions** (`tuple[torch.FloatTensor]`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.


Example:

Copied

```python
>>> from transformers import VitsTokenizer, VitsModel, set_seed
>>> import torch

>>> tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
>>> model = VitsModel.from_pretrained("facebook/mms-tts-eng")

>>> inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

>>> set_seed(555)  # make deterministic

>>> with torch.no_grad():
...     outputs = model(inputs["input_ids"])
>>> outputs.waveform.shape
torch.Size([1, 45824])
```

[Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/vits.md)

[←VibeVoice ASR](https://huggingface.co/docs/transformers/en/model_doc/vibevoice_asr) [Wav2Vec2→](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2)