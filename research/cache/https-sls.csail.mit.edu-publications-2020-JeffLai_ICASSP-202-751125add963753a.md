---
url: https://sls.csail.mit.edu/publications/2020/JeffLai_ICASSP-2020.pdf
title: Zero-Shot Multi-Speaker Text-To-Speech with State-Of-The-Art Neural Speaker Embeddings
retrieved_at: 2026-08-22T20:03:49Z
api_version: v2
---

# ZERO-SHOT MULTI-SPEAKER TEXT-TO-SPEECH WITH STATE-OF-THE-ART NEURAL

## y , Yusuke Yasuda

Erica Cooper $ ^{\star} $ , Cheng-I Lai $ ^{\dagger} $ , Yusuke Yasuda $ ^{\star} $ , Fuming Fang $ ^{\star} $ , Xin Wang $ ^{\star} $ , Nanxin Chen $ ^{\ddagger} $ , Junichi Yamagishi $ ^{\star} $

## z , Junichi Yamagishi
? National Institute of Informatics, Tokyo, Japan
## y Massachusetts Institute of Technology, Cambridge, USA

## z Johns Hopkins University, Baltimore, USA

$ ^{\dagger} $ Massachusetts Institute of Technology, Cambridge, USA $ ^{ \ddagger} $ Johns Hopkins University, Baltimore, USA

## ABSTRACT

While speaker adaptation for end-to-end speech synthesis using speaker embeddings can produce good speaker similarity for speakers seen during training, there remains a gap for zero-shot adaptation to unseen speakers. We investigate multi-speaker modeling for end-to-end text-to-speech synthesis and study the effects of different types of state-of-the-art neural speaker embeddings on speaker similarity for unseen speakers. Learnable dictionary encoding-based speaker embeddings with angular softmax loss can improve equal error rates over x-vectors in a speaker verification task; these embeddings also improve speaker similarity and naturalness for unseen speakers when used for zero-shot adaptation to new speakers in end-to-end speech synthesis.

$$
\mathrm{U S A}^{\dagger}
$$

Index Terms Speech synthesis, speaker adaptation, speaker embeddings, transfer learning, speaker verification

## 1. INTRODUCTION

An effective approach for speaker adaptation in neural TTS is to fine-tune all or part of model with a small amount of data from the target speaker [5, 6]. This approach can also be used to adapt to new speaking styles such as Lombard speech [7]. A different but complementary approach is to use speaker embeddings to model speaker identity in TTS. Prior studies have focused on training a speaker encoder network jointly with the TTS model [6,8,9] or the neural vocoder [10]; others have explored the use of speaker embeddings in combination with fine-tuning the TTS model [10,11,12]. Approaches that use fine-tuning necessarily require transcribed adaptation data, as well as more computational time and resources to adapt to a new speaker. Furthermore, speaker encoder networks that are jointly trained with the TTS model cannot benefit from data outside of the TTS training data, which is restricted to be of relatively high quality in clean recording conditions.

are the same as the best embeddings for TTS.

TTS, and these models can obtain robust speaker representations that

The second author performed the work mostly while interning at NII.

In parallel with the above-mentioned studies, there has been substantial development in end-to-end speaker recognition. Villalba et al. summarized several state-of-the-art speaker recognition systems for the NIST SRE18 Challenge [16], where x-vector based systems [17] consistently outperformed i-vector based systems [18]. There has also been a surge of interest in new encoding methods and endto-end loss functions for speaker recognition [19, 20, 21, 22, 23, 24, 25]. One prominent advancement is the use of learnable dictionary encoding (LDE) [19] and angular softmax [20] for speaker recognition, which are reported to boost the speaker recognition performance on open-source corpora such as the VoxCelebs [26, 27].

## 2. NEURAL SPEAKER EMBEDDINGS

There are three components in a typical end-to-end speaker recognition system: an encoder network, a statistical pooling layer, and a classifier [16]. An encoder network acts as a frame-level feature extractor, the statistical pooling layer summarizes frame-level representations to a fixed-dimensional utterance-level embedding, and

One aspect of our study is therefore an attempt to find out how effective these recent developments in speaker verification are for speaker adaption in TTS. More specifically we investigate the capability of neural speaker embeddings [16, 17, 19] to capture and model characteristics of speakers that were unseen during TTS model training. For this purpose, we extend an improved Tacotron system in [28] to a multi-speaker TTS system and conduct systematic analysis to answer the above question. We also analyze how the quality and similarity of generated voices are correlated with automatic speaker verification (ASV) accuracy.

---

**Fig. 1**. Learnable dictionary encoding (LDE) pooling method in an
end-to-end speaker recognition system.

the classifier determines the speaker identity based on the embedding. In most cases, the neural speaker embedding is obtained after
pooling and before classification. Below, we describe each of these
components used in our work.

## 2.1. Encoder Network

In the original x-vector paper [17], a time delay neural network
(TDNN) was used as the encoder network and its variants were explored in [16]. The TDNN is composed of 1D convolution and
fully connected layers. Several later studies [16, 19, 21, 22, 25, 27]
suggest replacing the TDNN with variants of ResNet34, composed
of 2D convolutions, as the encoder network. We used TDNN and
ResNet34 for x-vector and LDE embeddings, respectively.

## 2.2. Pooling methods

The pooling method is an important component since it summarizes
frame-level representations into a fixed-dimensional utterance-level
embedding.
**Statistical Pooling (SP)**: A statistical pooling layer was adopted in

**Statistical Pooling (SP)**: A statistical pooling layer was adopted in
the original x-vector paper [17]. It computes the mean and standard
deviation of the frame-level representations, which are concatenated
as a fixed-dimensional vector.
**Learnable Dictionary Encoding (LDE)**: Instead of the single mean

as a fixed-dimensional vector.
**Learnable Dictionary Encoding (LDE)**: Instead of the single mean
and standard deviation as in SP, the LDE layer proposed in [19] conducts soft clustering of the frame-level representations and concatenates the clusters’ means and standard deviations.
Given the frame-level representations x*T* = *fx₁; x₂; :::; xT g*

Given the frame-level representations x*T* = *fx₁; x₂; :::; xT g*
from the encoder networks, where *T* is the sequence length, an LDE
layer learns a dictionary of *C* clusters *fe₁; e₂; :::; eC g*. The learning procedure is decomposed into three steps: 1) compute some distance *rtc* from each frame *xt* to each cluster *ec*, 2) learn a soft cluster
weight *wtc* of *xt* to *ec* based on *rtc*, and 3) aggregate *xt* based on
*wtc* over time *T* to yield an utterance-level representation. Here *rtc*
is L2 distance, *rtc* = *kxt eck²*.PThe cluster weight *wtc* can be
C
computed by, *w*<sup>tc</sup> = exp( *r*<sup>tc</sup>)*=*i=1exp( rti). The aggregation of xt is similar to the supervector notion in [29]. We first com-
P
1
pute mean mc =8twtc(xt ec) and/or standard deviation
pPZ
1 2
sc =8twtc(xt ec) for each cluster, which are concate-
Z
nated for 8c 2 f1::<sup>C</sup>g to form a mean vectorPmC, and s<sup>i</sup>milarly
T
for a standard deviation vector sC. Here Z =t<sup>=1</sup>wtc. Figu*r*e 1
illus<sup>t</sup>rates these three step*s*.

## 2.3. Classifier

$$
r_{t c}
$$

$$
{tt x}_{T}\,=\,\left\{x_{1},x_{2},...,x_{T}\right\}
$$

$$
\{e_{1},e_{2},...,e_{C}\}
$$

$$
x_{t}
$$

$$
w_{t c}
$$

As shown in Figure 1, the last step is to predict speaker IDs via the
softmax layer. The standard training criterion is therefore cross en-

$$
w_{t c}
$$

$$
x_{t}
$$

$$
\underline{{m_{c}}}\,=\,\frac{1}{Z}\sum_{\forall t}\bar{w_{t c}}(\_{t}-e\_{c}
$$

dings down to 64 dimensions using a dense layer before inputting s = 1

$$
r _ {t c} = \left\| x _ {t} - e _ {c} \right\| ^ {2}.
$$

$$
r_{t c}
$$

$$
w_{t c}
$$

$$
e_{c}
$$

$$
x_{t}
$$

$$
e_{c},2
$$

$$
Z=\sum_{t=1}^{T}w_{t c}
$$

**Fig. 2**. Proposed multi-speaker TTS system. Encoder blocks are
in *orange*, decoder blocks in *blue*, post-net block in *green*, speaker
encoder blocks in *red*, and vocoder block in *yellow*.

tropy. More discriminative criteria called angular softmax loss (Asoftmax) and their variants have recently been proposed and evaluated in [20, 24, 30]. The criteria considers angular margins between
classes and is expected to produce more separable embedding representations. We examine both normal softmax and angular softmax.

## 3. MULTI-SPEAKER TTS MODEL ARCHITECTURE

The above speaker embedding vectors are used as additional inputs
to condition speaker characteristics in our multi-speaker TTS system. Our end-to-end multi-speaker text-to-speech model architecture is based on Tacotron [1], with the extension of self-attention
described in [28] to better capture long-range dependencies illustrated in Figure 2. We use phoneme input. We carry out basic rulebased text normalization to expand abbreviations and numbers. We
then convert the text to a phoneme representation using flite [31].
A self-attention block [32] is added in the encoder, and so the encoder produces two outputs: one is the original output of the long
short-term memory (LSTM), and the other is the output from the
self-attention block. The LSTM output is passed to a forward attention block [33], which speeds up the alignment, and the self-attention
output is passed to an additive attention block, which allows attention to longer-range information. The outputs of the dual attention
mechanism are concatenated before being passed to the decoder. The
output of the decoder is an 80-dimensional mel-spectrogram.
We consider three possible locations to input speaker embed-

Speaker adaptation to new speakers is zero-shot. As in the
training phase, we extract a speaker embedding vector from each
untranscribed adaptation utterance of a target speaker using the
speaker encoder. We then input the averaged speaker embedding to
generate mel spectrograms of the target speaker. No fine tuning is
used. To convert the predicted mel spectrograms into audio, we use
a WaveNet [34] vocoder. Input is mel spectrograms and output is

---

16-bit 16kHz waveforms. The code for our multi-speaker Tacotron
implementation and audio samples will be available online

## 4. EXPERIMENTS

## 4.1. Speaker Verification

is set for all LDEs. EER denotes equal error rate

denotes minimum detection cost function value with a 0 : 01

Data: Following [13], we trained speaker verification systems on VoxCeleb1+2[26, 27]. The training data were all of VoxCeleb2 plus the training portion of VoxCeleb1 (7,325 speakers and 1,277,344 utterances). The clean speech was augmented with reverberation noise, music, and babble, as described in [17], and then a random subset of these 1,000,000 augmented utterances was combined with the original clean speech. The final training data consisted of 2,277,344 utterances. We report the speaker verification results on the original VoxCeleb1 test set. Acoustic Features and Pre-processing: We trained x-vectors on 30-dimensional MFCCs and LDEs with 30-dimensional log-Mel filter banks. Kaldi-based 3-second sliding cepstral mean normalization and energy VAD were applied to the acoustic features. This is similar to the setup described in [16]. For the LDE systems, each training sample was a 3-8 second chunk randomly sampled from its original utterance. Our chunk length selection is consistent with [19, 21].

## System Details

21], and we set the number of dictionary clusters

eter combinations: embedding dimension

$$
m\,\in\,2,3,4\}
$$

$$
\mathbf{m}_{C}
$$

$$
\mathbf{s}_{C}
$$

preprocessed by first high-pass filtering at a cutoff of 80 Hz to re-

move low-frequency line noise, then normalized using sv56 [38],

Embedding Post-processing and Backend: Our backend is PLDA [35] with score normalization. We followed Kaldi's backend recipe in post-processing the embeddings prior to the PLDA: centering and LDA reduction to 200 dimensions. We also scored the original embeddings without this post-processing step, as we were interested in the effect of such a procedure for speaker adaption in TTS. Note that we did not perform length normalization nor any adaptation/tuning, as normally done in speaker verification, such as [16]. Verification Results: Table 1 presents the results of speaker verifi

Verification Results: Table 1 presents the results of speaker verification on the VoxCeleb1 test set. We denoted our 7 LDE embeddings as LDE-1, LDE-2, etc, and used superscript a N to mark those with the post-processing step described above. The LDEs attained results on par with the x-vectors. We also observed that decreasing speaker embedding size, increasing angular margin m, and pooling both $ \mathbf{m}_{C} $ and $ \mathbf{s}_{C} $ improve the performance.

Data: We used the VCTK corups [37], which consists of read English speech from 109 different speakers in different English dialects. Each speaker read about 400 sentences. Two speakers were

Since the best training method and location for inputting speaker embeddings to the TTS was unknown, we conducted preliminary experiments to learn which settings produce the best speaker similarity for unseen speakers. We wanted to learn whether it was better to train gender-dependent or gender-independent models, and whether it is best to input speaker embeddings to the prenet, concatenate with the encoder output, input at the postnet, or some combination of these.

$$
\mathbf{s}C
$$

## 4.2. Preliminary Experiments for Speaker Similarity in TTS

implementation and audio samples will be available online
min
embed. dim. pl. obj. norm EER DCF

- Gender-independent: We used data from all VCTK training speakers (male and female) for warm-start training.

excluded due to missing or inadequate data. Four development and
four test speakers were held out, chosen to be a mix of genders and

Training: We used a "warm-start" training approach to reduce experimental iteration time. We initialized our multi-speaker models with parameters from a well-trained speaker-dependent model trained on the "Nancy" data from Blizzard 2011 [39] for about 105k steps. We experimentally found that multi-speaker models trained with warm-start for one day (about 40k steps) produced synthetic speech of similar quality to models trained from scratch on VCTK data only for four days. Furthermore, [40] observed that the VCTK corpus has a relatively small number of unique words, whereas the Nancy dataset has more than three times as many; our multi-speaker model can benefit from this increased lexical coverage.

- Gender-dependent: We ran two separate warm-start trainings, one using only male VCTK training data and the other using only the female data.

- Prenet only (pre)

<sup>1</sup>https://github.com/nii-yamagishilab/multi-speaker-tacotron
<sup>2</sup>https://nii-yamagishilab.github.io/samples-multi-speaker-tacotron

$ ^{3} $https://github.com/kaldi-asr/kaldi/tree/master/egs/voxceleb/v2

---

**Table 2**. *Average cosine similarities between original and synthe-***Table 3**. *MOS and DMOS results for seen (train) and unseen (dev*

*sized speech from different model configurations for seen (training) and eval) speakers using each type of speaker embedding. Wave-* *and unseen (dev set) speakers. Waveform generation was done using form generation was done using unseen texts for all speakers. Five-* *unseen texts for both seen and unseen speakers. point and four-point scales were used for naturalness and similarity* Input location Gender-ind Gender-dep *evaluation, respectively. Blue boxes show the best results for each* train dev train dev *condition and red boxes show second and third best.* pre 0.357 0.402 0.438 0.361 Naturalness Similarity attn 0.709 0.490 0.711 0.476 system train dev test train dev test pre+attn 0.676 0.489 0.708 **0.533** pre+attn+post 0.684 0.480 0.717 0.477 vocoded 3.51 3.41 3.55 3.02 2.79 2.82 N x-Vec 3.20 3.19 3.19 2.93 1.86 2.37 Concatenate with encoder output only and input to attention LDE-1 3.15 3.16 3.21 2.87 **2.05** 2.34 mechanism ( N attn) LDE-1 3.04 3.13 **3.46** 2.87 1.97 **2.45** LDE-2 Prenet + concatenate with encoder output ( 3.11 **3.28** 3.35 2.84 2.00 2.37 pre+attn) N LDE-2 3.13 3.19 3.33 2.90 2.00 2.35 Prenet + concatenate with encoder output + postnet LDE-3 3.09 3.24 **3.48** 2.89 1.88 **2.46** N ( pre+attn+post) LDE-3 3.14 3.16 3.33 2.91 2.00 2.37 LDE-4 3.08 3.10 3.29 2.94 2.00 2.31 We did not try postnet input alone because we found that this con- N LDE-4 3.12 3.20 3.29 2.90 1.98 2.39 figuration produced poor quality synthetic speech, but we decided to LDE-5 3.07 **3.26 3.40** 2.89 1.99 **2.45** investigate its combination with other input locations. N LDE-5 3.11 3.07 3.37 2.88 **2.02** 2.41 **Evaluation and Results:** We objectively evaluated the different LDE-6 3.12 3.25 3.33 2.92 1.95 2.43 combinations of training strategy and embedding input locations N LDE-6 3.13 **3.29** 3.23 2.88 1.94 2.39 by synthesizing some sample utterances from four “seen” speakers LDE-7 3.15 3.03 3.18 2.91 1.86 2.28 (those included in training) and four “unseen” ones (development N LDE-7 3.07 3.02 3.24 2.83 **2.02** 2.42 speakers). Since we did not hold out any data from the “seen” speakers’ utterances, we synthesized seen speakers’ sample utter- by five different listeners, and 463 subjects participated. Sets were ances from a randomly selected set of texts from the test set (unseen designed to contain at least one utterance from every system to aver- during training). We then extracted x-vectors for each speaker from age out listener differences over all systems. Results are in Table 3. the *synthesized* speech, and measured cosine similarity to the target Natural speech was rated with MOS of 3.83 and DMOS of 3.25. speaker’s x-vector extracted from his or her actual speech. Cosine We found that speaker similarity scores for speakers seen during *sim* ( *A; B*) = *A B= k A kk B k* and is similarity is defined as *cos* training are very close to those for vocoded speech. Similarity scores a standard measure of similarity of speaker embedding vectors for for unseen speakers (dev and test) are also lower than seen speakers, ASV. The values range from -1 to 1, and higher values indicate that as expected, and consistent with Table 2. We observed that advanced the vectors are more similar. Cosine similarity results for different neural speaker embeddings improve speaker similarity for unseen configurations are listed in Table 2. speakers compared to x-vectors. Unexpectedly, they also improve As expected, we see a gap between seen and unseen speakers: naturalness. While LDE helps, the impact of angular softmax and seen speakers’ synthetic speech generally has higher similarity to the postprocessing (N) seems to be small. For completely unseen test original speech. Since the gender-dependent training with x-vectors set speakers, system LDE-3 was best in terms of both naturalness input at both the prenet and attention mechanism produced the syn- and speaker similarity. This system was significantly better than the thetic speech with best similarity for unseen speakers, we chose this x-vector system according to a Mann-Whitney U test both in terms configuration for our later experiments. of naturalness (p=5.9e-11) and speaker similarity (p=0.02). This was

**4.3. Comparing Different Embeddings for Speaker Similarity** also the best type of embedding in terms of EER. We did not find any
meaningful correlations between ASV and TTS scores. After we chose the best training and model settings (gender- dependent training with embedding input at both the prenet and

**5. CONCLUSIONS**
attention mechanism), we trained 15 TTS models each using a dif- ferent type of speaker embedding: the 14 types of LDE embeddings We found that the LDE-based neural speaker embeddings can im- described in Section 4.1 and x-vectors. We then conducted a crowd- prove speaker similarity and naturalness of synthetic speech for un- sourced listening test to evaluate naturalness and speaker similarity seen speakers, and this approach can be used for zero-shot speaker for both seen and unseen speakers using each speaker embedding as adaptation. However, there is still a gap between seen and unseen well as copy-synthesized speech and natural speech for comparison. speaker similarity, indicating that the TTS model may still be over- For each TTS system, we synthesized 50 sentences from each fitting to seen speakers and there is room for improvement. For fu- of the four “seen” (training) and eight “unseen” (development and ture work, we will explore ways to mitigate this overfitting by trying test) speakers, for a total of 600 unseen test utterances per system. different methods of speaker space augmentation. We would also Listeners heard one test utterance at a time and first rated it on a like to evaluate adaptation performance on more nuanced aspects of Likert scale from 1-5 for Mean Opinion Score (MOS) for natural- speaker similarity, such as dialect and speaking style. ness, then rated for speaker similarity compared to a reference utter- ance on a Differential MOS (DMOS) scale [41] from 1 (definitely **Acknowledgments** This work was partially supported by a JST CREST a different speaker) to 4 (definitely the same speaker). Reference Grant (JPMJCR18A6, VoicePersonae project), Japan, and by MEXT KAK- utterances were randomly chosen from the target speaker’s original ENHI Grants (16H06302, 17H04687, 18H04120, 18H04112, 18KT0051, speech. Listeners rated “sets” of 25 utterances, and each listener 19K24372), Japan. The numerical calculations were carried out on the TSUBAME 3.0 supercomputer at the Tokyo Institute of Technology. could complete a maximum of ten sets. Each set was completed

Authorized licensed use limited to: MIT Libraries. Downloaded on February 12,2022 at 02:03:08 UTC from IEEE Xplore. Restrictions apply.

$$
\left(A,B\right)=A\cdot B/\left\|A\right\|\left\|B\right\|
$$

---

**6. REFERENCES** [22]W. Xie, A. Nagrani, J. S. Chung, and A. Zisserman, “Utterance-level
aggregation for speaker recognition in the wild,” *ICASSP*, 2019. [1]Y. Wang, R. Skerry-Ryan, D. Stanton, Y. Wu, R. J. Weiss, Z. Y. Jaitly, [23]M. Hajibabaei and D. Dai, “Unified hypersphere embedding for speaker

Y. Xiao, Z. Chen, S. Bengio, Q. Le *et al.*, “Tacotron: Towards end-torecognition,” *arXiv preprint arXiv:1807.08312*, 2018. end speech synthesis,” in *INTERSPEECH*, 2017. [24]X. Xiang, S. Wang, H. Huang, Y. Qian, and K. Yu, “Margin mat- [2]W. Ping, K. Peng, and J. Chen, “Clarinet: Parallel wave generation in ters: Towards more discriminative deep neural network embeddings end-to-end text-to-speech,” *ICLR*, 2018. for speaker recognition,” *arXiv preprint arXiv:1906.07317*, 2019. [3]J. Shen, R. Pang, R. J. Weiss, M. Schuster, N. Jaitly, Z. Yang, Z. Chen,

Y. Zhang, Y. Wang, R. Skerry-Ryan *et al.*, “Natural TTS synthesis by [25]Y. Jung, Y. Kim, H. Lim, Y. Choi, and H. Kim, “Spatial pyramid enconditioning WaveNet on mel spectrogram predictions,” *ICASSP*, 2018. coding with convex length normalization for text-independent speaker verification,” *arXiv preprint arXiv:1906.08333*, 2019. [4]J. Park, K. Zhao, K. Peng, and W. Ping, “Multi-speaker end-to-end speech synthesis,” *CoRR*, vol. abs/1907.04462, 2019. [26]A. Nagrani, J. S. Chung, and A. Zisserman, “VoxCeleb: a large-scale speaker identification dataset,” *arXiv preprint arXiv:1706.08612*, 2017. [5]Z. Kons, S. Shechtman, A. Sorin, C. Rabinovitz, and R. Hoory, “High quality, lightweight and adaptable TTS using LPCNet,” *INTER-*[27]J. S. Chung, A. Nagrani, and A. Zisserman, “VoxCeleb2: Deep speaker *SPEECH*, 2019. recognition,” *arXiv preprint arXiv:1806.05622*, 2018. [6]Y. Chen, Y. Assael, B. Shillingford, D. Budden, S. Reed, H. Zen, [28]Y. Yasuda, X. Wang, S. Takaki, and J. Yamagishi, “Investigation of

Q. Wang, L. C. Cobo, A. Trask, B. Laurie *et al.*, “Sample efficient enhanced Tacotron text-to-speech synthesis systems with self-attention
adaptive text-to-speech,” *ICLR*, 2019. for pitch accent language,” *ICASSP*, 2019. [7]B. Bollepalli, L. Juvela, and P. Alku, “Lombard speech synthesis using [29]D. A. Reynolds, T. F. Quatieri, and R. B. Dunn, “Speaker verifica- transfer learning in a Tacotron text-to-speech system,” *INTERSPEECH*, tion using adapted gaussian mixture models,” *Digital signal processing*,

2019. vol. 10, no. 1-3, pp. 19–41, 2000. [8]J. Park, K. Zhao, K. Peng, and W. Ping, “Multi-speaker end-to-end [30]G. Bhattacharya, J. Alam, and P. Kenny, “Deep speaker recognition:
speech synthesis,” *arXiv preprint arXiv:1907.04462*. Modular or monolithic?” *Proc. Interspeech 2019*, pp. 1143–1147, [9]E. Nachmani, A. Polyak, Y. Taigman, and L. Wolf, “Fitting new speak-2019. ers based on a short untranscribed sample,” *ICML*, 2018. [31]A. W. Black and K. A. Lenzo, “Flite: a small fast run-time synthesis en- [10]Y. Deng, L. He, and F. Soong, “Modeling multi-speaker latent space gine,” in *4th ISCA Tutorial and Research Workshop (ITRW) on Speech* to improve neural TTS: Quick enrolling new speaker and enhancing *Synthesis*, 2001. premium voice,” *arXiv preprint arXiv:1812.05253*, 2018. [32]A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, [11]Q. Hu, E. Marchi, D. Winarsky, Y. Stylianou, D. Naik, and S. Kajarekar, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you “Neural text-to-speech adaptation from low quality public recordings,” need,” in *Proc. NIPS*, 2017, pp. 6000–6010. [Online]. Available: *Speech Synthesis Workshop 10*, 2019. [http://papers.nips.cc/paper/7181-attention-is-all-you-need](http://papers.nips.cc/paper/7181-attention-is-all-you-need) [12]S. Arık, J. Chen, K. Peng, W. Ping, and Y. Zhou, “Neural voice cloning [33]J.-X. Zhang, Z.-H. Ling, and L.-R. Dai, “Forward attention in with a few samples,” in *Advances in Neural Information Processing* sequence-to-sequence acoustic modeling for speech synthesis,” in *Proc.* *Systems*, 2018, pp. 10 019–10 029. *ICASSP*. IEEE, 2018, pp. 4789–4793. [13]Y. Jia, Y. Zhang, R. Weiss, Q. Wang, J. Shen, F. Ren, P. Nguyen, [34]A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals,

R. Pang, I. L. Moreno, Y. Wu *et al.*, “Transfer learning from speaker A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu,
verification to multispeaker text-to-speech synthesis,” in *Advances in* “Wavenet: A generative model for raw audio,” in *9th ISCA Speech Syn-* *neural information processing systems*, 2018, pp. 4480–4490. *thesis Workshop*, 2016. [14]S. Pascual, M. Ravanelli, J. Serr `a, A. Bonafonte, and Y. Bengio, [35]P. Kenny, T. Stafylakis, P. Ouellet, J. Alam, and P. Dumouchel, “PLDA “Learning problem-agnostic speech representations from multiple self-for speaker verification with utterances of arbitrary duration,” in *Proc.* supervised tasks,” *INTERSPEECH*, 2019. *ICASSP*, 2013. [15]M. Chen, M. Chen, S. Liang, J. Ma, L. Chen, S. Wang, and J. Xiao, [36]N. Br ¨ummer and J. du Preez, “Application-independent evaluation of “Cross-lingual, multi-speaker text-to-speech synthesis using neural speaker detection,” *Computer Speech & Language*, vol. 20, no. 2, pp. speaker embedding,” *INTERSPEECH*, 2019. 230 – 275, 2006. [16]J. Villalba, N. Chen, D. Snyder, D. Garcia-Romero, A. McCree, G. Sell, [37]C. Veaux, J. Yamagishi, and K. MacDonald, “CSTR VCTK corpus: En-

J. Borgstrom, F. Richardson, S. Shon, F. Grondin *et al.*, “State-of-theglish multi-speaker corpus for CSTR voice cloning toolkit,” University art speaker recognition for telephone and video speech: the JHU-MIT of Edinburgh, The Centre for Speech Technology Research (CSTR), submission for NIST SRE18,” *INTERSPEECH*, 2019.

2017.
[17]D. Snyder, D. Garcia-Romero, G. Sell, D. Povey, and S. Khudan- [38]International Telecommunication Union, Recommendation G.191: pur, “X-vectors: Robust DNN embeddings for speaker recognition,” Software Tools and Audio Coding Standardization, Nov 11 2005. *ICASSP*, 2018. [39]S. King and V. Karaiskos, “The blizzard challenge 2011,” in *Blizzard* [18]N. Dehak, P. J. Kenny, R. Dehak, P. Dumouchel, and P. Ouellet, “Front- *Challenge Workshop*, 2011. end factor analysis for speaker verification,” *IEEE Transactions on Au-* *dio, Speech, and Language Processing*, vol. 19, no. 4, pp. 788–798, [40]J. Taylor and K. Richmond, “Analysis of pronunciation learning in end-

2010. to-end speech synthesis,” *INTERSPEECH*, 2019. [19]W. Cai, J. Chen, and M. Li, “Exploring the encoding layer and [41]J. Lorenzo-Trueba, J. Yamagishi, T. Toda, D. Saito, F. Villavicencio,
loss function in end-to-end speaker and language recognition system,” T. Kinnunen, and Z. Ling, “The voice conversion challenge 2018: Pro- *Odyssey 2018, The Speaker and Language Recognition Workshop*, moting development of parallel and nonparallel methods,” in *Proc.*

2018. *Odyssey 2018 The Speaker and Language Recognition Workshop*, [20]Z. Huang, S. Wang, and K. Yu, “Angular softmax for short-duration 2018, pp. 195–202.
text-independent speaker verification,” in *Proc. Interspeech 2018*, 2018, pp. 3623–3627. [21]N. Chen, J. Villalba, and N. Dehak, “Tied mixture of factor analyzers layer to combine frame level representations in neural speaker embed- dings,” *INTERSPEECH*, 2019.

Authorized licensed use limited to: MIT Libraries. Downloaded on February 12,2022 at 02:03:08 UTC from IEEE Xplore. Restrictions apply.