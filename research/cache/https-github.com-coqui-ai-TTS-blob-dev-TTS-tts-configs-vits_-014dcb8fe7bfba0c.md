---
url: https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/configs/vits_config.py
title: TTS/TTS/tts/configs/vits_config.py at dev · coqui-ai/TTS · GitHub
retrieved_at: 2026-08-22T20:04:43Z
api_version: v2
---

[Skip to content](https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/configs/vits_config.py#start-of-content)

You signed in with another tab or window. [Reload](https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/configs/vits_config.py) to refresh your session.You signed out in another tab or window. [Reload](https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/configs/vits_config.py) to refresh your session.You switched accounts on another tab or window. [Reload](https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/configs/vits_config.py) to refresh your session.Dismiss alert

{{ message }}

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/configs/vits_config.py).

[coqui-ai](https://github.com/coqui-ai)/ **[TTS](https://github.com/coqui-ai/TTS)** Public

- [Notifications](https://github.com/login?return_to=%2Fcoqui-ai%2FTTS) You must be signed in to change notification settings
- [Fork\\
6.1k](https://github.com/login?return_to=%2Fcoqui-ai%2FTTS)
- [Star\\
45.9k](https://github.com/login?return_to=%2Fcoqui-ai%2FTTS)


## Collapse file tree

## Files

dev

Search this repository(forward slash)` forward slash/`

/

# vits\_config.py

Copy path

Blame

More file actions

Blame

More file actions

## Latest commit

[![v4hn](https://avatars.githubusercontent.com/u/680358?v=4&size=40)](https://github.com/v4hn)[v4hn](https://github.com/coqui-ai/TTS/commits?author=v4hn)

[use default\_factory for audio parameter (](https://github.com/coqui-ai/TTS/commit/27e237ed082eaac84ba34807ac1fbc32f0b3a676) [#2576](https://github.com/coqui-ai/TTS/pull/2576) [)](https://github.com/coqui-ai/TTS/commit/27e237ed082eaac84ba34807ac1fbc32f0b3a676)

Open commit details

3 years agoMay 8, 2023

[27e237e](https://github.com/coqui-ai/TTS/commit/27e237ed082eaac84ba34807ac1fbc32f0b3a676) · 3 years agoMay 8, 2023

## History

[History](https://github.com/coqui-ai/TTS/commits/dev/TTS/tts/configs/vits_config.py)

Open commit details

[View commit history for this file.](https://github.com/coqui-ai/TTS/commits/dev/TTS/tts/configs/vits_config.py) History

176 lines (133 loc) · 6.73 KB

/

# vits\_config.py

Copy path

Top

## File metadata and controls

- Code

- Blame


176 lines (133 loc) · 6.73 KB

[Raw](https://github.com/coqui-ai/TTS/raw/refs/heads/dev/TTS/tts/configs/vits_config.py)

Copy raw file

Download raw file

You must be signed in to make or propose changes

More edit options

Open symbols panel

Edit and raw actions

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

37

38

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

54

55

56

57

58

59

60

61

62

63

64

65

66

67

68

69

70

71

72

73

74

75

76

77

78

79

80

81

82

83

84

85

86

87

88

89

90

91

92

93

94

95

96

97

98

99

100

101

102

103

104

105

106

107

108

109

110

111

112

113

114

115

116

117

118

119

120

121

122

123

124

125

126

127

128

129

130

131

132

133

134

135

136

137

138

139

140

141

142

143

144

145

146

147

148

149

150

151

152

153

154

155

156

157

158

159

160

161

162

163

164

165

166

167

168

169

170

171

172

173

174

175

176

fromdataclassesimportdataclass, field

fromtypingimportList

fromTTS.tts.configs.shared\_configsimportBaseTTSConfig

fromTTS.tts.models.vitsimportVitsArgs, VitsAudioConfig

@dataclass

classVitsConfig(BaseTTSConfig):

"""Defines parameters for VITS End2End TTS model.

Args:

model (str):

Model name. Do not change unless you know what you are doing.

model\_args (VitsArgs):

Model architecture arguments. Defaults to \`VitsArgs()\`.

audio (VitsAudioConfig):

Audio processing configuration. Defaults to \`VitsAudioConfig()\`.

grad\_clip (List):

Gradient clipping thresholds for each optimizer. Defaults to \`\[1000.0, 1000.0\]\`.

lr\_gen (float):

Initial learning rate for the generator. Defaults to 0.0002.

lr\_disc (float):

Initial learning rate for the discriminator. Defaults to 0.0002.

lr\_scheduler\_gen (str):

Name of the learning rate scheduler for the generator. One of the \`torch.optim.lr\_scheduler.\*\`. Defaults to

\`ExponentialLR\`.

lr\_scheduler\_gen\_params (dict):

Parameters for the learning rate scheduler of the generator. Defaults to \`{'gamma': 0.999875, "last\_epoch":-1}\`.

lr\_scheduler\_disc (str):

Name of the learning rate scheduler for the discriminator. One of the \`torch.optim.lr\_scheduler.\*\`. Defaults to

\`ExponentialLR\`.

lr\_scheduler\_disc\_params (dict):

Parameters for the learning rate scheduler of the discriminator. Defaults to \`{'gamma': 0.999875, "last\_epoch":-1}\`.

scheduler\_after\_epoch (bool):

If true, step the schedulers after each epoch else after each step. Defaults to \`False\`.

optimizer (str):

Name of the optimizer to use with both the generator and the discriminator networks. One of the

\`torch.optim.\*\`. Defaults to \`AdamW\`.

kl\_loss\_alpha (float):

Loss weight for KL loss. Defaults to 1.0.

disc\_loss\_alpha (float):

Loss weight for the discriminator loss. Defaults to 1.0.

gen\_loss\_alpha (float):

Loss weight for the generator loss. Defaults to 1.0.

feat\_loss\_alpha (float):

Loss weight for the feature matching loss. Defaults to 1.0.

mel\_loss\_alpha (float):

Loss weight for the mel loss. Defaults to 45.0.

return\_wav (bool):

If true, data loader returns the waveform as well as the other outputs. Do not change. Defaults to \`True\`.

compute\_linear\_spec (bool):

If true, the linear spectrogram is computed and returned alongside the mel output. Do not change. Defaults to \`True\`.

use\_weighted\_sampler (bool):

If true, use weighted sampler with bucketing for balancing samples between datasets used in training. Defaults to \`False\`.

weighted\_sampler\_attrs (dict):

Key retuned by the formatter to be used for weighted sampler. For example \`{"root\_path": 2.0, "speaker\_name": 1.0}\` sets sample probabilities

by overweighting \`root\_path\` by 2.0. Defaults to \`{}\`.

weighted\_sampler\_multipliers (dict):

Weight each unique value of a key returned by the formatter for weighted sampling.

For example \`{"root\_path":{"/raid/datasets/libritts-clean-16khz-bwe-coqui\_44khz/LibriTTS/train-clean-100/":1.0, "/raid/datasets/libritts-clean-16khz-bwe-coqui\_44khz/LibriTTS/train-clean-360/": 0.5}\`.

It will sample instances from \`train-clean-100\` 2 times more than \`train-clean-360\`. Defaults to \`{}\`.

r (int):

Number of spectrogram frames to be generated at a time. Do not change. Defaults to \`1\`.

add\_blank (bool):

If true, a blank token is added in between every character. Defaults to \`True\`.

test\_sentences (List\[List\]):

List of sentences with speaker and language information to be used for testing.

language\_ids\_file (str):

Path to the language ids file.

use\_language\_embedding (bool):

If true, language embedding is used. Defaults to \`False\`.

Note:

Check :class:\`TTS.tts.configs.shared\_configs.BaseTTSConfig\` for the inherited parameters.

Example:

>>\> from TTS.tts.configs.vits\_config import VitsConfig

>>\> config = VitsConfig()

"""

model: str="vits"

\# model specific params

model\_args: VitsArgs=field(default\_factory=VitsArgs)

audio: VitsAudioConfig=field(default\_factory=VitsAudioConfig)

\# optimizer

grad\_clip: List\[float\] =field(default\_factory=lambda: \[1000, 1000\])

lr\_gen: float=0.0002

lr\_disc: float=0.0002

lr\_scheduler\_gen: str="ExponentialLR"

lr\_scheduler\_gen\_params: dict=field(default\_factory=lambda: {"gamma": 0.999875, "last\_epoch": -1})

lr\_scheduler\_disc: str="ExponentialLR"

lr\_scheduler\_disc\_params: dict=field(default\_factory=lambda: {"gamma": 0.999875, "last\_epoch": -1})

scheduler\_after\_epoch: bool=True

optimizer: str="AdamW"

optimizer\_params: dict=field(default\_factory=lambda: {"betas": \[0.8, 0.99\], "eps": 1e-9, "weight\_decay": 0.01})

\# loss params

kl\_loss\_alpha: float=1.0

disc\_loss\_alpha: float=1.0

gen\_loss\_alpha: float=1.0

feat\_loss\_alpha: float=1.0

mel\_loss\_alpha: float=45.0

dur\_loss\_alpha: float=1.0

speaker\_encoder\_loss\_alpha: float=1.0

\# data loader params

return\_wav: bool=True

compute\_linear\_spec: bool=True

\# sampler params

use\_weighted\_sampler: bool=False\# TODO: move it to the base config

weighted\_sampler\_attrs: dict=field(default\_factory=lambda: {})

weighted\_sampler\_multipliers: dict=field(default\_factory=lambda: {})

\# overrides

r: int=1\# DO NOT CHANGE

add\_blank: bool=True

\# testing

test\_sentences: List\[List\] =field(

default\_factory=lambda: \[\
\
\["It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."\],\
\
\["Be a voice, not an echo."\],\
\
\["I'm sorry Dave. I'm afraid I can't do that."\],\
\
\["This cake is great. It's so delicious and moist."\],\
\
\["Prior to November 22, 1963."\],\
\
\]

)

\# multi-speaker settings

\# use speaker embedding layer

num\_speakers: int=0

use\_speaker\_embedding: bool=False

speakers\_file: str=None

speaker\_embedding\_channels: int=256

language\_ids\_file: str=None

use\_language\_embedding: bool=False

\# use d-vectors

use\_d\_vector\_file: bool=False

d\_vector\_file: List\[str\] =None

d\_vector\_dim: int=None

def\_\_post\_init\_\_(self):

forkey, valinself.model\_args.items():

ifhasattr(self, key):

self\[key\] =val

You can’t perform that action at this time.