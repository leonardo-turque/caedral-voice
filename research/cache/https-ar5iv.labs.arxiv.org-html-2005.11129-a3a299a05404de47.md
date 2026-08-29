---
url: https://ar5iv.labs.arxiv.org/html/2005.11129
title: [2005.11129] Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search
retrieved_at: 2026-08-22T20:02:02Z
api_version: v2
---

# Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search

Jaehyeon Kim
Affiliation: Kakao Enterprise
Email: [jay.xyz@kakaoenterprise.com](mailto:)Sungwon Kim
Affiliation: Data Science & AI Lab.
Affiliation: Seoul National University
Email: [ksw0306@snu.ac.kr](mailto:)Jungil Kong
Affiliation: Kakao Enterprise
Email: [henry.k@kakaoenterprise.com](mailto:)Sungroh Yoon
Thanks: Corresponding author
Affiliation: Data Science & AI Lab.
Affiliation: Seoul National University
Email: [sryoon@snu.ac.kr](mailto:)

###### Abstract

Recently, text-to-speech (TTS) models such as FastSpeech and ParaNet have been proposed to generate mel-spectrograms from text in parallel. Despite the advantage, the parallel TTS models cannot be trained without guidance from autoregressive TTS models as their external aligners. In this work, we propose Glow-TTS, a flow-based generative model for parallel TTS that does not require any external aligner. By combining the properties of flows and dynamic programming, the proposed model searches for the most probable monotonic alignment between text and the latent representation of speech on its own. We demonstrate that enforcing hard monotonic alignments enables robust TTS, which generalizes to long utterances, and employing generative flows enables fast, diverse, and controllable speech synthesis. Glow-TTS obtains an order-of-magnitude speed-up over the autoregressive model, Tacotron 2, at synthesis with comparable speech quality. We further show that our model can be easily extended to a multi-speaker setting.

## 1 Introduction

Text-to-speech (TTS) is a task in which speech is generated from text, and deep-learning-based TTS models have succeeded in producing natural speech.
Among neural TTS models, autoregressive models, such as Tacotron 2 \[ [23](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib23 "")\] and Transformer TTS \[ [13](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib13 "")\], have shown state-of-the-art performance.
Despite the high synthesis quality of autoregressive TTS models, there are a few difficulties in deploying them directly in real-time services. As the inference time of the models grows linearly with the output length, undesirable delay caused by generating long utterances can be propagated to the multiple pipelines of TTS systems without designing sophisticated frameworks \[ [14](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib14 "")\].
In addition, most of the autoregressive models show a lack of robustness in some cases \[ [20](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib20 ""), [16](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib16 "")\].
For example, when an input text includes repeated words, autoregressive TTS models sometimes produce serious attention errors.

To overcome such limitations of the autoregressive TTS models, parallel TTS models, such as FastSpeech \[ [20](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib20 "")\], have been proposed. These models can synthesize mel-spectrograms significantly faster than the autoregressive models. In addition to the fast sampling, FastSpeech reduces the failure cases of synthesis, such as mispronouncing, skipping, or repeating words, by constraining its alignment to be monotonic. However, to train the parallel TTS models, well-aligned attention maps between text and speech are necessary. Recently proposed parallel models extract attention maps from their external aligners, pre-trained autoregressive TTS models \[ [16](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib16 ""), [20](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib20 "")\]. Therefore, the performance of the models critically depends on that of the external aligners.

In this work, we eliminate the necessity of any external aligner and simplify the training procedure of parallel TTS models. Here, we propose Glow-TTS, a flow-based generative model for parallel TTS that can internally learn its own alignment.

By combining the properties of flows and dynamic programming, Glow-TTS efficiently searches for the most probable monotonic alignment between text and the latent representation of speech. The proposed model is directly trained to maximize the log-likelihood of speech with the alignment.
We demonstrate that enforcing hard monotonic alignments enables robust TTS, which generalizes to long utterances, and employing flows enables fast, diverse, and controllable speech synthesis.

Glow-TTS can generate mel-spectrograms 15.7 times faster than the autoregressive TTS model, Tacotron 2, while obtaining comparable performance. As for robustness, the proposed model outperforms Tacotron 2 significantly when input utterances are long. By altering the latent representation of speech, we can synthesize speech with various intonation patterns and regulate the pitch of speech. We further show that our model can be extended to a multi-speaker setting with only a few modifications. Our source code111 [https://github.com/jaywalnut310/glow-tts](https://github.com/jaywalnut310/glow-tts ""). and synthesized audio samples222 [https://jaywalnut310.github.io/glow-tts-demo/index.html](https://jaywalnut310.github.io/glow-tts-demo/index.html ""). are publicly available.

## 2 Related Work

Alignment Estimation between Text and Speech.
Traditionally, hidden Markov models (HMMs) have been used to estimate unknown alignments between text and speech \[ [19](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib19 ""), [25](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib25 "")\].
In speech recognition, CTC has been proposed as a method of alleviating the downsides of HMMs, such as the assumption of conditional independence over observations, through a discriminative neural network model \[ [6](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib6 "")\]. Both methods above can efficiently estimate alignments through forward-backward algorithms with dynamic programming.
In this work, we introduce a similar dynamic programming method to search for the most probable alignment between text and the latent representation of speech, where our modeling differs from CTC in that it is generative, and from HMMs in that it can sample sequences in parallel without the assumption of conditional independence over observations.

Text-to-Speech Models. TTS models are a family of generative models that synthesize speech from text. TTS models, such as Tacotron 2 \[ [23](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib23 "")\], Deep Voice 3 \[ [17](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib17 "")\] and Transformer TTS \[ [13](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib13 "")\], generate a mel-spectrogram from text, which is comparable to that of the human voice. Enhancing the expressiveness of TTS models has also been studied. Auxiliary embedding methods have been proposed to generate diverse speech by controlling factors such as intonation and rhythm \[ [24](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib24 ""), [32](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib32 "")\], and some works have aimed at synthesizing speech in the voices of various speakers \[ [9](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib9 ""), [5](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib5 "")\]. Recently, several works have proposed methods to generate mel-spectrogram frames in parallel. FastSpeech \[ [20](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib20 "")\], and ParaNet \[ [16](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib16 "")\] significantly speed up mel-spectrogram generation over autoregressive TTS models, while preserving the quality of synthesized speech. However, both parallel TTS models need to extract alignments from pre-trained autoregressive TTS models to alleviate the length mismatch problem between text and speech. Our Glow-TTS is a standalone parallel TTS model that internally learns to align text and speech by leveraging the properties of flows and dynamic programming.

Flow-based Generative Models. Flow-based generative models have received a lot of attention due to their advantages \[ [7](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib7 ""), [4](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib4 ""), [21](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib21 "")\]. They can estimate the exact likelihood of the data by applying invertible transformations. Generative flows are simply trained to maximize the likelihood. In addition to efficient density estimation, the transformations proposed in \[ [2](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib2 ""), [3](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib3 ""), [12](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib12 "")\] guarantee fast and efficient sampling. [Prenger et al. 2019](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib18 "") and [Kim et al. 2019](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib10 "") introduced these transformations for speech synthesis to overcome the slow sampling speed of an autoregressive vocoder, WaveNet \[ [29](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib29 "")\]. Their proposed models both synthesized raw audio significantly faster than WaveNet. By applying these transformations, Glow-TTS can synthesize a mel-spectrogram given text in parallel.

In parallel with our work, AlignTTS \[ [34](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib34 "")\], Flowtron \[ [28](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib28 "")\], and Flow-TTS \[ [15](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib15 "")\] have been proposed. AlignTTS and Flow-TTS are parallel TTS models without the need of external aligners, and Flowtron is a flow-based model which shows the ability of style transfer and controllability of speech variation. However, AlignTTS is not a flow-based model but a feed-forward network, and Flowtron and Flow-TTS use soft attention modules. By employing both hard monotonic alignments and generative flows, our model combines the best of both worlds in terms of robustness, diversity and controllability.

## 3 Glow-TTS

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2005.11129/assets/Figure1_a.png)(a)An abstract diagram of the training procedure.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2005.11129/assets/Figure1_b.png)(b)An abstract diagram of the inference procedure.

Figure 1: Training and inference procedures of Glow-TTS.

Inspired by the fact that a human reads out text in order, without skipping any words, we design Glow-TTS to generate a mel-spectrogram conditioned on a monotonic and non-skipping alignment between text and speech representations.
In Section [3.1](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.SS1 "3.1 Training and Inference Procedures ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"), we formulate the training and inference procedures of the proposed model, which are also illustrated in Figure [1](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.F1 "Figure 1 ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"). We present our alignment search algorithm in Section [3.2](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.SS2 "3.2 Monotonic Alignment Search ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"), which removes the necessity of external aligners from training, and the architecture of all components of Glow-TTS (i.e., the text encoder, duration predictor, and flow-based decoder) is covered in Section [3.3](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.SS3 "3.3 Model Architecture ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search").

### 3.1 Training and Inference Procedures

Glow-TTS models the conditional distribution of mel-spectrograms PX​(x\|c)P\_{X}(x\|c) by transforming a conditional prior distribution PZ​(z\|c)P\_{Z}(z\|c) through the flow-based decoder fd​e​c:z→xf\_{dec}:z\\rightarrow x, where xx and cc denote the input mel spectrogram and text sequence, respectively. By using the change of variables, we can calculate the exact log-likelihood of the data as follows:

|     |     |     |     |
| --- | --- | --- | --- |
|  | log⁡PX​(x\|c)=log⁡PZ​(z\|c)+log⁡\|det∂fd​e​c−1​(x)∂x\|\\log{P\_{X}(x\|c)}=\\log{P\_{Z}(z\|c)}+\\log{\\left\|\\det{\\frac{\\partial{f\_{dec}^{-1}(x)}}{\\partial{x}}}\\right\|} |  | (1) |

We parameterize the data and prior distributions with network parameters θ\\theta and an alignment function AA. The prior distribution PZP\_{Z} is the isotropic multivariate Gaussian distribution and all the statistics of the prior distribution, μ\\mu and σ\\sigma, are obtained by the text encoder fe​n​cf\_{enc}. The text encoder maps the text condition c=c1:Tt​e​x​tc=c\_{1:T\_{text}} into the statistics, μ=μ1:Tt​e​x​t\\mu=\\mu\_{1:T\_{text}} and σ=σ1:Tt​e​x​t\\sigma=\\sigma\_{1:T\_{text}}, where Tt​e​x​tT\_{text} denotes the length of the text input. In our formulation, the alignment function AA stands for the mapping from the index of the latent representation of speech to that of statistics from fe​n​cf\_{enc}: A⁡(j)=iA(j)=i if zj∼N⁡(zj,μi,σi)z\_{j}\\sim N(z\_{j};\\mu\_{i},\\sigma\_{i}). We assume the alignment function AA to be monotonic and surjective to ensure Glow-TTS not to skip or repeat the text input. Then, the prior distribution can be expressed as follows:

|     |     |     |     |
| --- | --- | --- | --- |
|  | log⁡PZ​(z\|c;θ,A)=∑j=1Tm​e​llog⁡𝒩⁡(zj,μA⁡(j),σA⁡(j)),\\log{P\_{Z}(z\|c;\\theta,A)}=\\sum\_{j=1}^{T\_{mel}}{\\log{\\mathcal{N}(z\_{j};\\mu\_{A(j)},\\sigma\_{A(j)})}}, |  | (2) |

where Tm​e​lT\_{mel} denotes the length of the input mel-spectrogram.

Our goal is to find the parameters θ\\theta and the alignment AA that maximize the log-likelihood of the data, as in Equation [3](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.E3 "In 3.1 Training and Inference Procedures ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"). However, it is computationally intractable to find the global solution. To tackle the intractability, we reduce the search space of the parameters and alignment by decomposing the objective into two subsequent problems: (i)(i) searching for the most probable monotonic alignment A∗A^{\*} with respect to the current parameters θ\\theta, as in Equation [4](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.E4 "In 3.1 Training and Inference Procedures ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"), and (i​i)(ii) updating the parameters θ\\theta to maximize the log-likelihood log⁡pX​(x\|c;θ,A∗)\\log{p\_{X}(x\|c;\\theta,A^{\*}}). In practice, we handle these two problems using an iterative approach. At each training step, we first find A∗A^{\*}, and then update θ\\theta using the gradient descent.
The iterative procedure is actually one example of widely used Viterbi training \[ [19](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib19 "")\], which maximizes log likelihood of the most likely hidden alignment.
The modified objective does not guarantee the global solution of Equation [3](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.E3 "In 3.1 Training and Inference Procedures ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"), but it still provides a good lower bound of the global solution.

|     |     |     |     |
| --- | --- | --- | --- |
|  | maxθ,A⁡L⁡(θ,A)=maxθ,A​log​PX​(x\|c;A,θ)\\max\_{\\theta,A}{L(\\theta,A)}=\\max\_{\\theta,A}{\\log{P\_{X}(x\|c;A,\\theta})} |  | (3) |

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | A∗\\displaystyle A^{\*} | =arg​maxAlogPX(x\|c;A,θ)=arg​maxA∑j=1Tmellog𝒩(zj;μA⁡(j),σA⁡(j))\\displaystyle=\\argmax\_{A}{\\log{P\_{X}(x\|c;A,\\theta})}=\\argmax\_{A}{\\sum\_{j=1}^{T\_{mel}}{\\log\\mathcal{N}(z\_{j};\\mu\_{A(j)},\\sigma\_{A(j)}}}) |  | (4) |

To solve the alignment search problem (i)(i), we introduce an alignment search algorithm, monotonic alignment search (MAS), which we describe in Section [3.2](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.SS2 "3.2 Monotonic Alignment Search ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search").

To estimate the most probable monotonic alignment A∗A^{\*} at inference, we also train the duration predictor fd​u​rf\_{dur} to match the duration label calculated from the alignment A∗A^{\*}, as in Equation [5](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.E5 "In 3.1 Training and Inference Procedures ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search").
Following the architecture of FastSpeech \[ [20](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib20 "")\], we append the duration predictor on top of the text encoder and train it with the mean squared error loss (MSE) in the logarithmic domain. We also apply the stop gradient operator s​g​\[⋅\]sg\[\\cdot\], which removes the gradient of input in the backward pass \[ [30](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib30 "")\], to the input of the duration predictor to avoid affecting the maximum likelihood objective. The loss for the duration predictor is described in Equation [6](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.E6 "In 3.1 Training and Inference Procedures ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search").

|     |     |     |     |
| --- | --- | --- | --- |
|  | di=∑j=1Tm​e​l1A∗​(j)=i,i=1,…,Tt​e​x​td\_{i}=\\sum\_{j=1}^{T\_{mel}}{1\_{A^{\*}(j)=i}},i=1,...,T\_{text} |  | (5) |

|     |     |     |     |
| --- | --- | --- | --- |
|  | Ld​u​r=M​S​E​(fd​u​r​(s​g​\[fe​n​c​(c)\]),d)L\_{dur}=MSE(f\_{dur}(sg\[f\_{enc}(c)\]),d) |  | (6) |

During inference, as shown in Figure [1(b)](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.F1.sf2 "In Figure 1 ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"), the statistics of the prior distribution and alignment are predicted by the text encoder and duration predictor. Then, a latent variable is sampled from the prior distribution, and a mel-spectrogram is synthesized in parallel by transforming the latent variable through the flow-based decoder.

### 3.2 Monotonic Alignment Search

(a)An example of monotonic alignments

(b)Calculating the maximum log-likelihood QQ.

(c)Backtracking the most probable alignment A∗A^{\*}.

Figure 2: Illustrations of the monotonic alignment search.

As mentioned in Section [3.1](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.SS1 "3.1 Training and Inference Procedures ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"), MAS searches for the most probable monotonic alignment between the latent variable and the statistics of the prior distribution, which are came from the input speech and text, respectively. Figure [2(a)](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.F2.sf1 "In Figure 2 ‣ 3.2 Monotonic Alignment Search ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search") shows one example of possible monotonic alignments.

We present our alignment search algorithm in Algorithm [1](https://ar5iv.labs.arxiv.org/html/2005.11129#alg1 "Algorithm 1 ‣ 3.2 Monotonic Alignment Search ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"). We first derive a recursive solution over partial alignments and then find the entire alignment.

Let Qi,jQ\_{i,j} be the maximum log-likelihood where the statistics of the prior distribution and the latent variable are partially given up to the ii-th and jj-th elements, respectively. Then, Qi,jQ\_{i,j} can be recursively formulated with Qi−1,j−1Q\_{i-1,j-1} and Qi,j−1Q\_{i,j-1}, as in Equation [7](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.E7 "In 3.2 Monotonic Alignment Search ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"), because if the last elements of partial sequences, zjz\_{j} and {μi,σi}\\{\\mu\_{i},\\sigma\_{i}\\}, are aligned, the previous latent variable zj−1z\_{j-1} should have been aligned to either {μi−1,σi−1}\\{\\mu\_{i-1},\\sigma\_{i-1}\\} or {μi,σi}\\{\\mu\_{i},\\sigma\_{i}\\} to satisfy monotonicity and surjection.

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
|  | Qi,j\\displaystyle Q\_{i,j} | =max⁡∑k=1jA⁡log⁡𝒩⁡(zk,μA⁡(k),σA⁡(k))=max⁡(Qi−1,j−1,Qi,j−1)+log⁡𝒩⁡(zj,μi,σi)\\displaystyle=\\max\_{A}{\\sum\_{k=1}^{j}{\\log\\mathcal{N}(z\_{k};\\mu\_{A(k)},\\sigma\_{A(k)}}})=\\max(Q\_{i-1,j-1},Q\_{i,j-1})+{\\log\\mathcal{N}(z\_{j};\\mu\_{i},\\sigma\_{i}}) |  | (7) |

This process is illustrated in Figure [2(b)](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.F2.sf2 "In Figure 2 ‣ 3.2 Monotonic Alignment Search ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"). We iteratively calculate all the values of QQ up to QTt​e​x​t,Tm​e​lQ\_{T\_{text},T\_{mel}}.

Similarly, the most probable alignment A∗A^{\*} can be obtained by determining which QQ value is greater in the recurrence relation, Equation [7](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.E7 "In 3.2 Monotonic Alignment Search ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"). Thus, A∗A^{\*} can be found efficiently with dynamic programming by caching all QQ values; all the values of A∗A^{\*} are backtracked from the end of the alignment, A∗​(Tm​e​l)=Tt​e​x​tA^{\*}(T\_{mel})=T\_{text}, as in Figure [2(c)](https://ar5iv.labs.arxiv.org/html/2005.11129#S3.F2.sf3 "In Figure 2 ‣ 3.2 Monotonic Alignment Search ‣ 3 Glow-TTS ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search").

The time complexity of the algorithm is O⁡(Tt​e​x​t×Tm​e​l)O(T\_{text}\\times T\_{mel}). Even though the algorithm is difficult to parallelize, it runs efficiently on CPU without the need for GPU executions. In our experiments, it spends less than 20 ms on each iteration, which amounts to less than 2% of the total training time. Furthermore, we do not need MAS during inference, as the duration predictor is used to estimate the alignment.

Algorithm 1 Monotonic Alignment Search

Input: latent representation zz, the statistics of prior distribution μ\\mu , σ\\sigma, the mel-spectrogram length Tm​e​lT\_{mel}, the text length Tt​e​x​tT\_{text}

Output: monotonic alignment A∗A^{\*}



Initialize Q⋅,⋅←−∞Q\_{\\cdot,\\cdot}\\leftarrow-\\infty, a cache to store the maximum log-likelihood calculations



Compute the first row Q1,j←∑k=1jlog⁡𝒩⁡(zk,μ1,σ1)Q\_{1,j}\\leftarrow\\sum\_{k=1}^{j}{\\log\\mathcal{N}(z\_{k};\\mu\_{1},\\sigma\_{1})}, for all jj

forj=2j=2toTm​e​lT\_{mel}do

fori=2i=2tomin⁡(j,Tt​e​x​t)\\min(j,T\_{text})do

Qi,j←max⁡(Qi−1,j−1,Qi,j−1)+log⁡𝒩⁡(zj,μi,σi)Q\_{i,j}\\leftarrow\\max(Q\_{i\\scalebox{0.75}\[1.0\]{$-$}1,j\\scalebox{0.75}\[1.0\]{$-$}1},Q\_{i,j\\scalebox{0.75}\[1.0\]{$-$}1})+\\log\\mathcal{N}(z\_{j};\\mu\_{i},\\sigma\_{i})

endfor

endfor



Initialize A∗​(Tm​e​l)←Tt​e​x​tA^{\*}(T\_{mel})\\leftarrow T\_{text}

forj=Tm​e​l−1j=T\_{mel}-1to11do

A∗​(j)←arg​maxi∈{A∗​(j+1)−1,A∗​(j+1)}⁡Qi,jA^{\*}(j)\\leftarrow\\argmax\_{i\\in\\{A^{\*}(j+1)-1,A^{\*}(j+1)\\}}Q\_{i,j}

endfor

### 3.3 Model Architecture

Each component of Glow-TTS is briefly explained in this section, and the overall model architecture and model configurations are shown in Appendix [A](https://ar5iv.labs.arxiv.org/html/2005.11129#Sx3 "Appendix A ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search").

#### Decoder.

The core part of Glow-TTS is the flow-based decoder. During training, we need to efficiently transform a mel-spectrogram into the latent representation for maximum likelihood estimation and our internal alignment search. During inference, it is necessary to transform the prior distribution into the mel-spectrogram distribution efficiently for parallel decoding. Therefore, our decoder is composed of a family of flows that can perform forward and inverse transformation in parallel. Specifically, our decoder is a stack of multiple blocks, each of which consists of an activation normalization layer, invertible 1x1 convolution layer, and affine coupling layer. We follow the affine coupling layer architecture of WaveGlow \[ [18](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib18 "")\], except we do not use the local conditioning \[ [29](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib29 "")\].

For computational efficiency, we split 80-channel mel-spectrogram frames into two halves along the time dimension and group them into one 160-channel feature map before the flow operations. We also modify 1x1 convolution to reduce the time-consuming calculation of the Jacobian determinant. Before every 1x1 convolution, we split the feature map into 40 groups along the channel dimension and perform 1x1 convolution on them separately.
To allow channel mixing in each group, the same number of channels are extracted from one half of the feature map separated by coupling layers and the other half, respectively. A detailed description can be found in Appendix [A.1](https://ar5iv.labs.arxiv.org/html/2005.11129#Sx3.SSx1 "A.1. Details of the Model Architecture ‣ Appendix A ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search").

#### Encoder and Duration Predictor.

We follow the encoder structure of Transformer TTS \[ [13](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib13 "")\] with two slight modifications. We remove the positional encoding and add relative position representations \[ [22](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib22 "")\] into the self-attention modules instead. We also add a residual connection to the encoder pre-net. To estimate the statistics of the prior distribution, we append a linear projection layer at the end of the encoder. The duration predictor is composed of two convolutional layers with ReLU activation, layer normalization, and dropout followed by a projection layer. The architecture and configuration of the duration predictor are the same as those of FastSpeech \[ [20](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib20 "")\].

## 4 Experiments

To evaluate the proposed methods, we conduct experiments on two different datasets. For the single speaker setting, a single female speaker dataset, LJSpeech  \[ [8](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib8 "")\], is used, which consists of 13,100 short audio clips with a total duration of approximately 24 hours. We randomly split the dataset into the training set (12,500 samples), validation set (100 samples), and test set (500 samples). For the multi-speaker setting, the train-clean-100 subset of the LibriTTS corpus \[ [33](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib33 "")\] is used, which consists of audio recordings of 247 speakers with a total duration of about 54 hours. We first trim the beginning and ending silence of all the audio clips and filter out the data with text lengths over 190. We then split it into the training (29,181 samples), validation (88 samples), and test sets (442 samples). Additionally, out-of-distribution text data are collected for the robustness test. Similar to \[ [1](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib1 "")\], we extract 227 utterances from the first two chapters of the book Harry Potter and the Philosopher’s Stone. The maximum length of the collected data exceeds 800.

We compare Glow-TTS with the best publicly available autoregressive TTS model, Tacotron 2 \[ [26](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib26 "")\]. For all the experiments, phonemes are chosen as input text tokens. We follow the configuration for the mel-spectrogram of \[ [27](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib27 "")\], and all the generated mel-spectrograms from both models are transformed to raw waveforms through the pre-trained vocoder, WaveGlow \[ [27](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib27 "")\].

During training, we simply set the standard deviation σ\\sigma of the learnable prior to be a constant 1. Glow-TTS was trained for 240K iterations using the Adam optimizer \[ [11](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib11 "")\] with the Noam learning rate schedule \[ [31](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib31 "")\]. This required only 3 days with mixed precision training on two NVIDIA V100 GPUs.

To train muli-speaker Glow-TTS, we add the speaker embedding and increase the hidden dimension. The speaker embedding is applied in all affine coupling layers of the decoder as a global conditioning  \[ [29](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib29 "")\]. The rest of the settings are the same as for the single speaker setting. For comparison, We also trained Tacotron 2 as a baseline, which concatenates the speaker embedding with the encoder output at each time step. We use the same model configuration as the single speaker one. All multi-speaker models were trained for 960K iterations on four NVIDIA V100 GPUs.

## 5 Results

### 5.1 Audio Quality

Table 1: The Mean Opinion Score (MOS) of single speaker TTS models with 95%\\% confidence intervals.

| Method | 9-scale MOS |
| --- | --- |
| GT | 4.54 ±\\pm 0.06 |
| GT (Mel + WaveGlow) | 4.19 ±\\pm 0.07 |
| Tacotron2 (Mel + WaveGlow) | 3.88 ±\\pm 0.08 |
| Glow-TTS (T=0.333T=0.333, Mel + WaveGlow) | 4.01 ±\\pm 0.08 |
| Glow-TTS (T=0.500T=0.500, Mel + WaveGlow) | 3.96 ±\\pm 0.08 |
| Glow-TTS (T=0.667T=0.667, Mel + WaveGlow) | 3.97 ±\\pm 0.08 |

We measure the mean opinion score (MOS) via Amazon Mechanical Turk to compare the quality of all audio clips, including ground truth (GT), and the synthesized samples; 50 sentences are randomly chosen from the test set for the evaluation. The results are shown in Table [1](https://ar5iv.labs.arxiv.org/html/2005.11129#S5.T1 "Table 1 ‣ 5.1 Audio Quality ‣ 5 Results ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"). The quality of speech converted from the GT mel-spectrograms by the vocoder (4.19±\\pm0.07) is the upper limit of the TTS models. We vary the standard deviation (i.e., temperature TT) of the prior distribution at inference; Glow-TTS shows the best performance at the temperature of 0.333. For any temperature, it shows comparable performance to Tacotron 2.
We also analyze side-by-side evaluation between Glow-TTS and Tacotron 2. The result is shown in Appendix [B.2](https://ar5iv.labs.arxiv.org/html/2005.11129#Sx4.SSx2 "B.2. Side-by-side Evaluation between Glow-TTS and Tacotron 2 ‣ Appendix B ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search").

### 5.2 Sampling Speed and Robustness

Sampling Speed. We use the test set to measure the sampling speed of the models. Figure [3(a)](https://ar5iv.labs.arxiv.org/html/2005.11129#S5.F3.sf1 "In Figure 3 ‣ 5.2 Sampling Speed and Robustness ‣ 5 Results ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search") demonstrates that the inference time of our model is almost constant at 40ms, regardless of the length, whereas that of Tacotron 2 linearly increases with the length due to the sequential sampling. On average, Glow-TTS shows a 15.7 times faster synthesis speed than Tacotron 2.

We also measure the total inference time for synthesizing 1-minute speech in an end-to-end setting with Glow-TTS and WaveGlow. The total inference time to synthesize the 1-minute speech is only 1.5 seconds333



We generated a speech sample from our abstract paragraph, which we mention as the 1-minute speech. and the inference time of Glow-TTS and WaveGlow accounts for 4%\\% and 96%\\% of the total inference time, respectively; the inference time of Glow-TTS takes only 55ms to synthesize the mel-spectrogram, which is negligible compared to that of the vocoder.

Robustness. We measure the character error rate (CER) of the synthesized samples from long utterances in the book Harry Potter and the Philosopher’s Stone via the Google Cloud Speech-To-Text API.444 [https://cloud.google.com/speech-to-text](https://cloud.google.com/speech-to-text "") Figure [3(b)](https://ar5iv.labs.arxiv.org/html/2005.11129#S5.F3.sf2 "In Figure 3 ‣ 5.2 Sampling Speed and Robustness ‣ 5 Results ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search") shows that the CER of Tacotron 2 starts to grow when the length of input characters exceeds about 260. On the other hand, even though our model has not seen such long texts during training, it shows robustness to long texts. We also analyze attention errors on specific sentences. The results are shown in Appendix [B.1](https://ar5iv.labs.arxiv.org/html/2005.11129#Sx4.SSx1 "B.1. Attention Error Analysis ‣ Appendix B ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search").

(a)The inference time comparison for Tacotron 2 and Glow-TTS (yellow: Tacotron2, blue: Glow-TTS).

(b)Robustness to the length of input utterances (yellow: Tacotron2, blue: Glow-TTS).

Figure 3: Comparison of inference time and length robustness.

### 5.3 Diversity and Controllability

Because Glow-TTS is a flow-based generative model, it can synthesize diverse samples; each latent representation zz sampled from an input text is converted to a different mel-spectrogram fd​e​c​(z)f\_{dec}(z).
Specifically, the latent representation z∼𝒩⁡(μ,T)z\\sim\\mathcal{N}(\\mu,T) can be expressed as follows:

|     |     |     |     |
| --- | --- | --- | --- |
|  | z=μ+ϵ∗Tz=\\mu+\\epsilon\*T |  | (8) |

where ϵ\\epsilon denotes a sample from the standard normal distribution and μ\\mu and TT denote the mean and standard deviation (i.e., temperature) of the prior distribution, respectively.

To decompose the effect of ϵ\\epsilon and TT, we draw pitch tracks of synthesized samples in Figure [4](https://ar5iv.labs.arxiv.org/html/2005.11129#S5.F4 "Figure 4 ‣ 5.3 Diversity and Controllability ‣ 5 Results ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search") by varying ϵ\\epsilon and TT one at a time. Figure [4(a)](https://ar5iv.labs.arxiv.org/html/2005.11129#S5.F4.sf1 "In Figure 4 ‣ 5.3 Diversity and Controllability ‣ 5 Results ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search") demonstrates that diverse stress or intonation patterns of speech arise from ϵ\\epsilon, whereas Figure [4(b)](https://ar5iv.labs.arxiv.org/html/2005.11129#S5.F4.sf2 "In Figure 4 ‣ 5.3 Diversity and Controllability ‣ 5 Results ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search") demonstrates that we can control the pitch of speech while maintaining similar intonation by only varying TT.
Additionally, we can control speaking rates of speech by multiplying a positive scalar value across the predicted duration of the duration predictor. The result is visualized in Figure [5](https://ar5iv.labs.arxiv.org/html/2005.11129#S5.F5 "Figure 5 ‣ 5.3 Diversity and Controllability ‣ 5 Results ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"); the values multiplied by the predicted duration are 1.25, 1.0, 0.75, and 0.5.

(a)Pitch tracks of the generated speech samples from the same sentence with different gaussian noise ϵ\\epsilon and the same temperature T=0.667T=0.667.

(b)Pitch tracks of the generated speech samples from the same sentence with different temperatures TT and the same gaussian noise ϵ\\epsilon.

Figure 4: The fundamental frequency (F0) contours of synthesized speech samples from Glow-TTS trained on the LJ dataset.![Refer to caption](https://ar5iv.labs.arxiv.org/html/2005.11129/assets/length_control.png)Figure 5: Mel-spectrograms of the generated speech samples with different speaking rates.

### 5.4 Multi-Speaker TTS

Table 2: The Mean Opinion Score (MOS) of a multi-speaker TTS with 95%\\% confidence intervals.

| Method | 9-scale MOS |
| --- | --- |
| GT | 4.54 ±\\pm 0.07 |
| GT (Mel + WaveGlow) | 4.22 ±\\pm 0.07 |
| Tacotron2 (Mel + WaveGlow) | 3.35 ±\\pm 0.12 |
| Glow-TTS (T=0.333T=0.333, Mel + WaveGlow) | 3.20 ±\\pm 0.12 |
| Glow-TTS (T=0.500T=0.500, Mel + WaveGlow) | 3.31 ±\\pm 0.12 |
| Glow-TTS (T=0.667T=0.667, Mel + WaveGlow) | 3.45 ±\\pm 0.11 |

Audio Quality. We measure the MOS as done in Section [5.1](https://ar5iv.labs.arxiv.org/html/2005.11129#S5.SS1 "5.1 Audio Quality ‣ 5 Results ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"); we select 50 speakers, and randomly sample one utterance per a speaker from the test set for evaluation.
The results are presented in Table [2](https://ar5iv.labs.arxiv.org/html/2005.11129#S5.T2 "Table 2 ‣ 5.4 Multi-Speaker TTS ‣ 5 Results ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"). The quality of speech converted from the GT mel-spectrograms (4.22±\\pm0.07) is the upper limit of the TTS models. Our model with the best configuration achieves 3.45 MOS, which results in comparable performance to Tacotron 2.

Speaker-Dependent Duration.
Figure [6(a)](https://ar5iv.labs.arxiv.org/html/2005.11129#S5.F6.sf1 "In Figure 6 ‣ 5.4 Multi-Speaker TTS ‣ 5 Results ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search") shows the pitch tracks of generated speech from the same sentence with different speaker identities. As the only difference in input is speaker identities, the result demonstrates that our model differently predicts the duration of each input token with respect to the speaker identities.

Voice Conversion.
As we do not provide any speaker identity into the encoder, the prior distribution is forced to be independent from speaker identities. In other words, Glow-TTS learns to disentangle the latent representation zz and the speaker identities. To investigate the degree of disentanglement, we transform a GT mel-spectrogram into the latent representation with the correct speaker identity and then invert it with different speaker identities. The detailed method can be found in Appendix [B.3](https://ar5iv.labs.arxiv.org/html/2005.11129#Sx4.SSx3 "B.3. Voice Conversion Method ‣ Appendix B ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search") The results are presented in Figure [6(b)](https://ar5iv.labs.arxiv.org/html/2005.11129#S5.F6.sf2 "In Figure 6 ‣ 5.4 Multi-Speaker TTS ‣ 5 Results ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"). It shows that converted samples have different pitch levels while maintaining a similar trend.

(a)Pitch tracks of the generated speech samples from the same sentence with different speaker identities.

(b)Pitch tracks of the voice conversion samples with different speaker identities.

Figure 6: The fundamental frequency (F0) contours of synthesized speech samples from Glow-TTS trained on the LibriTTS dataset.

## 6 Conclusion

In this paper, we proposed a new type of parallel TTS model, Glow-TTS.
Glow-TTS is a flow-based generative model that is directly trained with maximum likelihood estimation. As the proposed model finds the most probable monotonic alignment between text and the latent representation of speech on its own, the entire training procedure is simplified without the necessity of external aligners.
In addition to the simple training procedure, we showed that Glow-TTS synthesizes mel-spectrograms 15.7 times faster than the autoregressive baseline, Tacotron 2, while showing comparable performance. We also demonstrated additional advantages of Glow-TTS, such as the ability to control the speaking rate or pitch of synthesized speech, robustness, and extensibility to a multi-speaker setting. Thanks to these advantages, we believe the proposed model can be applied in various TTS tasks such as prosody transfer or style modeling.

## Broader Impact

In this paper, researchers introduce Glow-TTS, a diverse, robust and fast text-to-speech (TTS) synthesis model.
Neural TTS models including Glow-TTS, could be applied in many applications which require naturally synthesized speech.
Some of the applications are AI voice assistant services, audiobook services, advertisements, automotive navigation systems and automated answering services.
Therefore, by utilizing the models for synthesizing natural sounding speech, the providers of such applications could improve user satisfaction.
In addition, the fast synthesis speed of the proposed model could be beneficial for some service providers who provide real time speech synthesis services.
However, because of the ability to synthesize natural speech, the TTS models could also be abused through cyber crimes such as fake news or phishing.
It means that TTS models could be used to impersonate voices of celebrities for manipulating behaviours of people, or to imitate voices of someone’s friends or family for fraudulent purposes.
With the development of speech synthesis technology, the growth of studies to detect real human voice from synthesized voices seems to be needed.
Neural TTS models could sometimes synthesize undesirable speech with slurry or wrong pronunciations.
Therefore, it should be used carefully in some domain where even a single pronunciation mistake is critical such as news broadcast.
Additional concern is about the training data. Many corpus for speech synthesis contain speech data uttered by a handful of speakers.
Without the detailed consideration and restriction about the range of uses the TTS models have, the voices of the speakers could be overused than they might expect.

## Acknowledgments

We would like to thank Jonghoon Mo, Hyeongseok Oh, Hyunsoo Lee, Yongjin Cho, Minbeom Cho, Younghun Oh, and Jaekyoung Bae for helpful discussions and advice.
This work was partially supported by the BK21 FOUR program of the Education and Research Program for Future ICT Pioneers, Seoul National University in 2020.

## References

- Battenberg et al. \[2020\]
Eric Battenberg, R. J. Skerry-Ryan, Soroosh Mariooryad, Daisy Stanton, David
Kao, Matt Shannon, and Tom Bagby.

Location-relative attention mechanisms for robust long-form speech
synthesis.

In _International Conference on Acoustics, Speech and Signal_
_Processing_, pages 6194–6198, 2020.

- Dinh et al. \[2014\]
Laurent Dinh, David Krueger, and Yoshua Bengio.

Nice: Non-linear independent components estimation.

_arXiv preprint arXiv:1410.8516_, 2014.

- Dinh et al. \[2017\]
Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio.

Density estimation using real NVP.

In _International Conference on Learning Representations_, 2017.

- Durkan et al. \[2019\]
Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios.

Neural spline flows.

In _Advances in Neural Information Processing Systems_, pages
7509–7520, 2019.

- Gibiansky et al. \[2017\]
Andrew Gibiansky, Sercan Ömer Arik, Gregory Frederick Diamos, John
Miller, Kainan Peng, Wei Ping, Jonathan Raiman, and Yanqi Zhou.

Deep voice 2: Multi-speaker neural text-to-speech.

In _Advances in Neural Information Processing Systems_, pages
2962–2970, 2017.

- Graves et al. \[2006\]
Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen
Schmidhuber.

Connectionist temporal classification: labelling unsegmented sequence
data with recurrent neural networks.

In _International Conference on Machine Learning_, pages
369–376, 2006.

- Hoogeboom et al. \[2019\]
Emiel Hoogeboom, Rianne van den Berg, and Max Welling.

Emerging convolutions for generative normalizing flows.

In _International Conference on Machine Learning_, pages
2771–2780, 2019.

- Ito \[2017\]
Keith Ito.

The lj speech dataset.

[https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/ ""), 2017.

- Jia et al. \[2018\]
Ye Jia, Yu Zhang, Ron J. Weiss, Quan Wang, Jonathan Shen, Fei Ren, Zhifeng
Chen, Patrick Nguyen, Ruoming Pang, Ignacio Lopez-Moreno, and Yonghui Wu.

Transfer learning from speaker verification to multispeaker
text-to-speech synthesis.

In _Advances in Neural Information Processing Systems_, pages
4485–4495, 2018.

- Kim et al. \[2019\]
Sungwon Kim, Sang-Gil Lee, Jongyoon Song, Jaehyeon Kim, and Sungroh Yoon.

Flowavenet: A generative flow for raw audio.

In _International Conference on Machine Learning_, pages
3370–3378, 2019.

- Kingma and Ba \[2015\]
Diederik P. Kingma and Jimmy Ba.

Adam: A method for stochastic optimization.

In _International Conference on Learning Representations_, 2015.

- Kingma and Dhariwal \[2018\]
Diederik P. Kingma and Prafulla Dhariwal.

Glow: Generative flow with invertible 1x1 convolutions.

In _Advances in Neural Information Processing Systems_, pages
10236–10245, 2018.

- Li et al. \[2019\]
Naihan Li, Shujie Liu, Yanqing Liu, Sheng Zhao, and Ming Liu.

Neural speech synthesis with transformer network.

In _AAAI_, pages 6706–6713, 2019.

- Ma et al. \[2019\]
Mingbo Ma, Baigong Zheng, Kaibo Liu, Renjie Zheng, Hairong Liu, Kainan Peng,
Kenneth Church, and Liang Huang.

Incremental text-to-speech synthesis with prefix-to-prefix framework.

_arXiv preprint arXiv:1911.02750_, 2019.

- Miao et al. \[2020\]
Chenfeng Miao, Shuang Liang, Minchuan Chen, Jun Ma, Shaojun Wang, and Jing
Xiao.

Flow-tts: A non-autoregressive network for text to speech based on
flow.

In _International Conference on Acoustics, Speech and Signal_
_Processing_, pages 7209–7213, 2020.

- Peng et al. \[2020\]
Kainan Peng, Wei Ping, Zhao Song, and Kexin Zhao.

Non-autoregressive neural text-to-speech.

In _International Conference on Machine Learning_, pages
10192–10204, 2020.

- Ping et al. \[2018\]
Wei Ping, Kainan Peng, Andrew Gibiansky, Sercan Ömer Arik, Ajay Kannan,
Sharan Narang, Jonathan Raiman, and John Miller.

Deep voice 3: Scaling text-to-speech with convolutional sequence
learning.

In _International Conference on Learning Representations_, 2018.

- Prenger et al. \[2019\]
Ryan Prenger, Rafael Valle, and Bryan Catanzaro.

Waveglow: A flow-based generative network for speech synthesis.

In _International Conference on Acoustics, Speech and Signal_
_Processing_, pages 3617–3621, 2019.

- Rabiner \[1989\]
Lawrence R Rabiner.

A tutorial on hidden markov models and selected applications in
speech recognition.

_Proceedings of the IEEE_, 77(2):257–286,
1989.

- Ren et al. \[2019\]
Yi Ren, Yangjun Ruan, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan
Liu.

Fastspeech: Fast, robust and controllable text to speech.

In _Advances in Neural Information Processing Systems_, pages
3165–3174, 2019.

- Serrà et al. \[2019\]
Joan Serrà, Santiago Pascual, and Carlos Segura Perales.

Blow: a single-scale hyperconditioned flow for non-parallel raw-audio
voice conversion.

In _Advances in Neural Information Processing Systems_, pages
6790–6800, 2019.

- Shaw et al. \[2018\]
Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani.

Self-attention with relative position representations.

In _Conference of the North American Chapter of the Association_
_for Computational Linguistics: Human Language Technologies_, pages 464–468,
2018.

- Shen et al. \[2018\]
Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly,
Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, R. J. Skerry-Ryan,
Rif A. Saurous, Yannis Agiomyrgiannakis, and Yonghui Wu.

Natural TTS synthesis by conditioning wavenet on mel spectrogram
predictions.

In _International Conference on Acoustics, Speech and Signal_
_Processing_, pages 4779–4783, 2018.

- Skerry-Ryan et al. \[2018\]
R. J. Skerry-Ryan, Eric Battenberg, Ying Xiao, Yuxuan Wang, Daisy Stanton,
Joel Shor, Ron J. Weiss, Rob Clark, and Rif A. Saurous.

Towards end-to-end prosody transfer for expressive speech synthesis
with tacotron.

In _International Conference on Machine Learning_, pages
4700–4709, 2018.

- Tokuda et al. \[2013\]
Keiichi Tokuda, Yoshihiko Nankaku, Tomoki Toda, Heiga Zen, Junichi Yamagishi,
and Keiichiro Oura.

Speech synthesis based on hidden markov models.

_Proceedings of the IEEE_, 101(5):1234–1252, 2013.

- Valle \[2018\]
Rafael Valle.

Tacotron 2.

[https://github.com/NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2 ""), 2018.

- Valle \[2019\]
Rafael Valle.

Waveglow.

[https://github.com/NVIDIA/waveglow](https://github.com/NVIDIA/waveglow ""), 2019.

- Valle et al. \[2020\]
Rafael Valle, Kevin Shih, Ryan Prenger, and Bryan Catanzaro.

Flowtron: an autoregressive flow-based generative network for
text-to-speech synthesis.

_arXiv preprint arXiv:2005.05957_, 2020.

- van den Oord et al. \[2016\]
Aäron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol
Vinyals, Alex Graves, Nal Kalchbrenner, Andrew W. Senior, and Koray
Kavukcuoglu.

Wavenet: A generative model for raw audio.

_arXiv preprint arXiv:1609.03499_, 2016.

- van den Oord et al. \[2017\]
Aäron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu.

Neural discrete representation learning.

In _Advances in Neural Information Processing Systems_, pages
6306–6315, 2017.

- Vaswani et al. \[2017\]
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.

Attention is all you need.

In _Advances in Neural Information Processing Systems_, pages
5998–6008, 2017.

- Wang et al. \[2018\]
Yuxuan Wang, Daisy Stanton, Yu Zhang, R. J. Skerry-Ryan, Eric Battenberg,
Joel Shor, Ying Xiao, Ye Jia, Fei Ren, and Rif A. Saurous.

Style tokens: Unsupervised style modeling, control and transfer in
end-to-end speech synthesis.

In _International Conference on Machine Learning_, pages
5167–5176, 2018.

- Zen et al. \[2019\]
Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J. Weiss, Ye Jia, Zhifeng Chen,
and Yonghui Wu.

Libritts: A corpus derived from librispeech for text-to-speech.

In _Interspeech_, pages 1526–1530, 2019.

- Zeng et al. \[2020\]
Zhen Zeng, Jianzong Wang, Ning Cheng, Tian Xia, and Jing Xiao.

Aligntts: Efficient feed-forward text-to-speech system without
explicit alignment.

In _International Conference on Acoustics, Speech and Signal_
_Processing_, pages 6714–6718, 2020.


Supplementary Material of

Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search

## Appendix A

### A.1. Details of the Model Architecture

The detailed encoder architecture is depicted in Figure [7](https://ar5iv.labs.arxiv.org/html/2005.11129#Sx3.F7 "Figure 7 ‣ A.1. Details of the Model Architecture ‣ Appendix A ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search").
Some implementation details that we use in the decoder, and the decoder architecture are depicted in Figure [8](https://ar5iv.labs.arxiv.org/html/2005.11129#Sx3.F8 "Figure 8 ‣ A.1. Details of the Model Architecture ‣ Appendix A ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search").

We design the grouped 1x1 convolutions to be able to mix channels. For each group, the same number of channels are extracted from one half of the feature map separated by coupling layers and the other half, respectively. Figure [8(c)](https://ar5iv.labs.arxiv.org/html/2005.11129#Sx3.F8.sf3 "In Figure 8 ‣ A.1. Details of the Model Architecture ‣ Appendix A ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search") shows an example. If a coupling layer divides a 8-channel feature map \[a, b, g, h, m, n, s, t\] into two halves \[a, b, g, h\] and \[m, n, s, t\], we implement to group them into \[a, b, m, n\] and \[g, h, s, t\] when the number of groups is 2, or \[a, m\], \[b, n\], \[g, s\], \[h, t\] when the number of groups is 4.

Figure 7: The encoder architecture of Glow-TTS. The encoder gets a text sequence and processes it through the encoder pre-net and Transformer encoder. Then, the last projection layer and duration predictor of the encoder use the hidden representation hh to predict the statistics of prior distribution and duration, respectively.

(a)The decoder architecture of Glow-TTS. The decoder gets a mel-spectrogram and squeezes it. The, the decoder processes it through a number of flow blocks. Each flow block contains activation normalization layer, affine coupling layer, and invertible 1x1 convolution layer. The decoder reshapes the output to make equal to the input size.

(b)An illustration of S​q​u​e​e​z​eSqueeze and U​n​S​q​u​e​e​z​eUnSqueeze operations. When squeezing, the channel size doubles up and the number of time steps becomes a half. If the number of time steps is odd, we simply ignore the last element of mel-spectrogram sequence. It corresponds to about 11 ms audio, which makes no difference in quality.

(c)An illustration of our invertible 1x1 convolution. Two partitions used for coupling layers are colored blue and white, respectively. If input channel size is 8 and the number of groups is 2, we share a small 4x4 matrix as a kernel of the invertible 1x1 convolution layer. After channel mixing, we split the input into each group, and perform 1x1 convolution separately.

Figure 8: The decoder architecture of Glow-TTS and the implementation details used in the decoder.

### A.2. Hyper-parameters

Hyper-parameters of Glow-TTS are listed in Table [3](https://ar5iv.labs.arxiv.org/html/2005.11129#Sx3.T3 "Table 3 ‣ A.2. Hyper-parameters ‣ Appendix A ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"). Contrary to the prevailing thought that flow-based generative models need the huge number of parameters, the total number of parameters of Glow-TTS (28.6M) is lower than that of FastSpeech (30.1M).

Table 3: Hyper-parameters of Glow-TTS.

| Hyper-Parameter | Glow-TTS (LJ Dataset) |
| --- | --- |
| Embedding Dimension | 192 |
| Pre-net Layers | 3 |
| Pre-net Hidden Dimension | 192 |
| Pre-net Kernel Size | 5 |
| Pre-net Dropout | 0.5 |
| Encoder Blocks | 6 |
| Encoder Multi-Head Attention Hidden Dimension | 192 |
| Encoder Multi-Head Attention Heads | 2 |
| Encoder Multi-Head Attention Maximum Relative Position | 4 |
| Encoder Conv Kernel Size | 3 |
| Encoder Conv Filter Size | 768 |
| Encoder Dropout | 0.1 |
| Duration Predictor Kernel Size | 3 |
| Duration Predictor Filter Size | 256 |
| Decoder Blocks | 12 |
| Decoder Activation Norm Data-dependent Initialization | True |
| Decoder Invertible 1x1 Conv Groups | 40 |
| Decoder Affine Coupling Dilation | 1 |
| Decoder Affine Coupling Layers | 4 |
| Decoder Affine Coupling Kernel Size | 5 |
| Decoder Affine Coupling Filter Size | 192 |
| Decoder Dropout | 0.05 |
| Total Number of Parameters | 28.6M |

## Appendix B

### B.1. Attention Error Analysis

Table 4: Attention error counts for TTS models on the 100 test sentences.

| Model | Attention Mask | Repeat | Mispronounce | Skip | Total |
| --- | --- | --- | --- | --- | --- |
| DeepVoice 3 \[ [16](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib16 "")\] | X | 12 | 10 | 15 | 37 |
| DeepVoice 3 \[ [16](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib16 "")\] | O | 1 | 4 | 3 | 8 |
| ParaNet \[ [16](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib16 "")\] | X | 1 | 4 | 7 | 12 |
| ParaNet \[ [16](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib16 "")\] | O | 2 | 4 | 0 | 6 |
| Tacotron 2 | X | 0 | 2 | 1 | 3 |
| Glow-TTS (T=0.333T=0.333) | X | 0 | 3 | 1 | 4 |

We measured attention alignment results using 100 test sentences used in ParaNet \[ [16](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib16 "")\]. The average length and maximum length of test sentences are 59.65 and 315, respectively. Results are shown in Table [4](https://ar5iv.labs.arxiv.org/html/2005.11129#Sx4.T4 "Table 4 ‣ B.1. Attention Error Analysis ‣ Appendix B ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"). The results of DeepVoice 3 and ParaNet are taken from \[ [16](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib16 "")\] and are not directly comparable due to the difference of grapheme-to-phoneme conversion tools.

Attention mask \[ [16](https://ar5iv.labs.arxiv.org/html/2005.11129#bib.bib16 "")\] is a method of computing attention only over a fixed window around target position at inference time. When constraining attention to be monotonic by applying attention mask technique, models make fewer attention errors.

Tacotron 2, which uses location sensitive attention, also makes little attention errors. Though Glow-TTS perform slightly worse than Tacotron 2 on the test sentences, Glow-TTS does not lose its robustness to extremely long sentences while Tacotron 2 does as we show in Section [5.2](https://ar5iv.labs.arxiv.org/html/2005.11129#S5.SS2 "5.2 Sampling Speed and Robustness ‣ 5 Results ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search").

### B.2. Side-by-side Evaluation between Glow-TTS and Tacotron 2

We conducted 7-point CMOS evaluation between Tacotron 2 and Glow-TTS with the sampling temperature 0.333, which are both trained on the LJSpeech dataset. Through 500 ratings on 50 items, Glow-TTS wins Tacotron 2 by a gap of 0.934 as in Table [5](https://ar5iv.labs.arxiv.org/html/2005.11129#Sx4.T5 "Table 5 ‣ B.2. Side-by-side Evaluation between Glow-TTS and Tacotron 2 ‣ Appendix B ‣ Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"), which shows preference towards our model over Tacotron 2.

Table 5: The Comparative Mean Opinion Score (CMOS) of single speaker TTS models

| Model | CMOS |
| --- | --- |
| Tacotron 2 | 0 |
| Glow-TTS (T=0.333T=0.333) | 0.934 |

### B.3. Voice Conversion Method

To transform a mel-spectrogram xx of a source speaker ss to a target mel-spectrogram x^\\hat{x} of a target speaker s^\\hat{s},
we first find the latent representation zz through the inverse pass of the flow-based decoder fd​e​cf\_{dec} with the source speaker identity ss.

|     |     |     |     |
| --- | --- | --- | --- |
|  | z=fd​e​c−1​(x\|s)\\displaystyle z=f\_{dec}^{-1}(x\|s) |  | (9) |

Then, we get the target mel-spectrogram x^\\hat{x} through the forward pass of the decoder with the target speaker identity s^\\hat{s}.

|     |     |     |     |
| --- | --- | --- | --- |
|  | x^=fd​e​c​(z\|s^)\\displaystyle\\hat{x}=f\_{dec}(z\|\\hat{s}) |  | (10) |

[◄](https://ar5iv.labs.arxiv.org/html/2005.11128) [![ar5iv homepage](https://ar5iv.labs.arxiv.org/assets/ar5iv.png)](https://ar5iv.labs.arxiv.org/) [Feeling\\
\\
lucky?](https://ar5iv.labs.arxiv.org/feeling_lucky) [Conversion\\
\\
report](https://ar5iv.labs.arxiv.org/log/2005.11129) [Report\\
\\
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2005.11129) [View original\\
\\
on arXiv](https://arxiv.org/abs/2005.11129) [►](https://ar5iv.labs.arxiv.org/html/2005.11130)