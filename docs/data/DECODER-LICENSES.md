# CAEDRAL VOICE — Decoder Dependency License Dossier (G1 vocoder/codec candidates)

Task LIC-DEC-01 · Verification date for ALL rows: **2026-08-22** (all retrievals via Firecrawl, cached under `research/cache/`).
Scope: code license AND pretrained-weights license for each G1 decoder candidate, verified from PRIMARY artifacts only
(repo LICENSE files, HF model-card license fields via the Hub metadata API, README license statements). No downloads, no installs.
Companion docs: `LICENSE-MATRIX.md` (datasets), `docs/research/VOCODERS-DECODERS.md` (technical survey whose claims this dossier verifies).

## Verdict legend

- **Comm**: may Caedral ship products using it? `YES` permissive / `COND` see conditions / `NO` barred or unlicensed.
- **Redistribution**: what we owe when shipping the artifact (code or weights) inside our product.
- Quotes are verbatim from the cited primary artifact; emphasis added where noted.

## Master verdict matrix

| # | Artifact | Code license | Pretrained-weights license | Comm | Redistribution duty |
|---|----------|--------------|----------------------------|------|---------------------|
| 1 | **Vocos** — gemelo-ai/vocos code + charactr/vocos-mel-24khz & charactr/vocos-encodec-24khz weights | MIT [Q1.1] | MIT [Q1.3][Q1.4][Q1.5] | **YES** | Retain MIT copyright + permission notice ("Charactr Inc." for both code and mel/encodec checkpoints) |
| 2 | **BigVGAN-v2** — NVIDIA/BigVGAN code + nvidia/bigvgan_v2_24khz_100band_256x weights | MIT [Q2.1] | MIT (card field links weights to the repo MIT LICENSE) [Q2.2] | **YES** | Retain NVIDIA copyright + permission notice |
| 3 | **SNAC** — hubertsiuzdak/snac code + hubertsiuzdak/snac_24khz weights | MIT [Q3.1] | MIT (card field) [Q3.2] | **YES** | Retain Descript + Hubert Siuzdak copyright notices |
| 4 | **WavTokenizer** — jishengpeng/WavTokenizer code + novateur/WavTokenizer weights (.ckpt files in same HF repo) | MIT [Q4.1] | MIT (card field; card BODY carries no license text — API field is operative) [Q4.2] | **YES** | Retain jishengpeng / WavTokenizer notices |
| 5 | **iSTFTNet2** — kan-bayashi/ParallelWaveGAN lineage | MIT [Q5.1] — but **repo contains NO iSTFTNet/iSTFTNet2 code on master** [F2] | None official for iSTFTNet2; repo's Drive checkpoints inherit **per-corpus terms** [Q5.2][Q5.3] | Code YES; third-party checkpoints COND (corpus-dependent) | MIT notice retention; checkpoint use gated by each training corpus's license |
| 6 | **HiFi-GAN v1 original weights** (jik876/hifi-gan Google Drive) | MIT (code only) [Q6.1] | **NO STATED LICENSE** anywhere in README [F1] | Weights **NO** (treat as all-rights-reserved); code YES | None granted — redistribution of the Drive checkpoints has no legal basis without written permission |

---

## 1. Vocos (charactr)

### Code — `gemelo-ai/vocos`
- Primary artifact: https://raw.githubusercontent.com/gemelo-ai/vocos/main/LICENSE — retrieved **2026-08-22T18:35:44Z**
- [Q1.1] Exact quote (opening lines):
  > "MIT License"
  > "Copyright (c) 2023 Charactr Inc."
- Grant language (same file): "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software".
- Condition (same file): "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software."

### Weights — `charactr/vocos-mel-24khz` (13.5M, G1 reference decoder)
- Card page: https://huggingface.co/charactr/vocos-mel-24khz — retrieved **2026-08-22T18:35:45Z**
- [Q1.2] Card body License section, verbatim: "The code in this repository is released under the MIT license."
  (Note: body sentence says "code"; the machine-readable card field below governs the whole repo incl. `pytorch_model.bin`.)
- Hub metadata API (card license field): https://huggingface.co/api/models/charactr/vocos-mel-24khz — retrieved **2026-08-22T18:42:04Z**, repo sha `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`
- [Q1.3] Exact quote: `"tags":["pytorch","arxiv:2306.00814","license:mit","region:us"]` and `"cardData":{"license":"mit"}`
- Files in repo (`siblings`): `.gitattributes`, `README.md`, `config.yaml`, `pytorch_model.bin` → the license field covers the weight file itself.

### Weights (sibling) — `charactr/vocos-encodec-24khz` (EnCodec-token head, G2 option)
- Hub metadata API: https://huggingface.co/api/models/charactr/vocos-encodec-24khz — retrieved **2026-08-22T18:45:14Z**, sha `4e61d082c08045a4c11e5b148ad93b1d0c591a14`
- [Q1.4] Exact quote: `"license:mit"` (tag) and `"cardData":{"license":"mit"}`
- Caveat (unchanged from survey): pairing this head with *Meta EnCodec tokens* imports CC-BY-NC weights upstream — head is clean, tokens are not unless self-trained.

**Verdict: SHIP-SAFE. Commercial use YES for code and both checkpoint families; only obligation is MIT notice retention.**

## 2. BigVGAN-v2 (NVIDIA)

### Code — `NVIDIA/BigVGAN`
- Primary artifact: https://raw.githubusercontent.com/NVIDIA/BigVGAN/main/LICENSE — retrieved **2026-08-22T18:35:48Z**
- [Q2.1] Exact quote (opening lines):
  > "MIT License"
  > "Copyright (c) 2024 NVIDIA CORPORATION."

### Weights — `nvidia/bigvgan_v2_24khz_100band_256x` (112M)
- Card page: https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x — retrieved **2026-08-22T18:35:50Z**
- Hub metadata API: https://huggingface.co/api/models/nvidia/bigvgan_v2_24khz_100band_256x — retrieved **2026-08-22T18:42:13Z**, sha `c329ede9e9bbc100ddf5c91e2330a61921262370`
- [Q2.2] Exact quote: `"cardData":{"license":"mit","license_link":"https://huggingface.co/nvidia/BigVGAN/blob/main/LICENSE","tags":["neural-vocoder","audio-generation"],"library_name":"PyTorch","pipeline_tag":"audio-to-audio"}` plus tag `"license:mit"`
  → the weights card explicitly points its license at the same MIT LICENSE file as the code. Relicensing NC→MIT confirmed current on live artifacts (historical NC-era mirrors remain a pin-time hazard).

**Verdict: SHIP-SAFE license-wise (MIT both). Non-license caveat stands: zero CPU RTF evidence; premium/offline tier only pending Phase G.**

## 3. SNAC

### Code — `hubertsiuzdak/snac`
- Primary artifact: https://raw.githubusercontent.com/hubertsiuzdak/snac/main/LICENSE — retrieved **2026-08-22T18:35:53Z**
- [Q3.1] Exact quote (opening lines):
  > "MIT License"
  > "Copyright (c) 2023, Descript"
  > "Copyright (c) 2024, Hubert Siuzdak"
  (Dual copyright line — retain BOTH notices.)

### Weights — `hubertsiuzdak/snac_24khz` (19.8M, speech)
- Card page: https://huggingface.co/hubertsiuzdak/snac_24khz — retrieved **2026-08-22T18:35:54Z** (body confirms speech-first training; contains no separate license section — the card field is authoritative)
- Hub metadata API: https://huggingface.co/api/models/hubertsiuzdak/snac_24khz — retrieved **2026-08-22T18:42:22Z**, sha `d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec`
- [Q3.2] Exact quote: `"license:mit"` (tag) and `"cardData":{"license":"mit","tags":["audio"]}`

**Verdict: SHIP-SAFE. Commercial YES; keep both copyright notices. Speech-only training data means music/SFX variants (32k/44k) are out-of-scope anyway.**

## 4. WavTokenizer

### Code — `jishengpeng/WavTokenizer`
- Primary artifact: https://raw.githubusercontent.com/jishengpeng/WavTokenizer/main/LICENSE — retrieved **2026-08-22T18:35:55Z**
- [Q4.1] Exact quote (opening lines):
  > "MIT License"
  > "Copyright (c) 2024 jishengpeng"

### Weights — `novateur/WavTokenizer` (hosts `WavTokenizer_small_600_24k_4096.ckpt` + `_small_320_…`)
- Card page: https://huggingface.co/novateur/WavTokenizer — retrieved **2026-08-22T18:35:57Z**. Finding [F3]: the rendered card BODY contains **no license statement at all** (no License section); medium/large checkpoints listed "Coming Soon" (GitHub-hosted, not on HF). The small-600/small-320 `.ckpt` files ARE in this repo (`siblings` confirmed via API).
- Hub metadata API: https://huggingface.co/api/models/novateur/WavTokenizer — retrieved **2026-08-22T18:42:31Z**, sha `917d513b3162144a78642448d3bb0516d1f51ed2`
- [Q4.2] Exact quote: `"tags":["audio-feature-extraction","speech-language-models","gpt4-o","tokenizer","codec-representation","text-to-speech","automatic-speech-recognition","arxiv:2408.16532","arxiv:2402.12208","license:mit","region:us"]` and `"cardData":{"license":"mit", …}`

**Verdict: SHIP-SAFE for the two small speech checkpoints under MIT — but because the card body is silent, re-confirm the `license:mit` field at pin time (single-source statement). Medium/large remain unreleased; anything later published off-HF needs fresh verification.**

## 5. iSTFTNet2 — kan-bayashi/ParallelWaveGAN lineage

### Code — `kan-bayashi/ParallelWaveGAN`
- Primary artifact: https://raw.githubusercontent.com/kan-bayashi/ParallelWaveGAN/master/LICENSE — retrieved **2026-08-22T18:35:58Z**
- [Q5.1] Exact quote (opening lines):
  > "The MIT License (MIT)"
  > "Copyright (c) 2020 Tomoki Hayashi <hayashi.tomoki@g.sp.m.is.nagoya-u.ac.jp>"
- Resolves the survey's open item "[14] LICENSE text not read directly yet": now READ and CONFIRMED MIT.
- **Material correction [F2]**: full recursive git tree https://api.github.com/repos/kan-bayashi/ParallelWaveGAN/git/trees/master?recursive=1 — retrieved **2026-08-22T18:47:02Z**, response ends `"truncated": false` — contains **zero** paths matching "istft" (case-insensitive). LJSpeech recipe configs (https://api.github.com/repos/kan-bayashi/ParallelWaveGAN/contents/egs/ljspeech/voc1/conf — retrieved **2026-08-22T18:45:23Z**) list MelGAN/HiFi-GAN/StyleMelGAN/PWG variants only, no iSTFTNet. ⇒ **iSTFTNet/iSTFTNet2 is NOT in this repo on master.** The survey's "implementation home" premise is wrong for current master; if an iSTFTNet2 implementation is adopted, locate the actual source (NTT authors' release or a named fork) and run a fresh license check on IT. (Verification limited to `master`; non-default branches unchecked.)

### Pretrained models offered by that repo (none are iSTFTNet2)
- README: https://raw.githubusercontent.com/kan-bayashi/ParallelWaveGAN/master/README.md — retrieved **2026-08-22T18:35:59Z**
- [Q5.2] Exact quote: "You can listen to the samples and download pretrained models from the link to our google drive." (checkpoint table lists PWG/MelGAN/HiFi-GAN/StyleMelGAN per corpus; no iSTFTNet entries)
- [Q5.3] Exact quotes (README terms footer):
  > "The terms of use of the pretrained model follow that of each corpus used for the training. Please carefully check by yourself."
  > "Some codes are derived from ESPnet or Kaldi, which are based on Apache-2.0 licenese." [sic]
- Consequence: e.g. `ljspeech_hifigan.v1` inherits LJSpeech (PD) terms; `libritts_hifigan.v1`/`vctk_hifigan.v1` inherit CC BY 4.0 attribution duties. Usable commercially WITH attribution per corpus — but none of them is iSTFTNet2.

**Verdict: CODE MIT = SHIP-SAFE as a recipe base ONLY IF the iSTFTNet2 implementation actually comes from somewhere else (unresolved — pending item). Repo's own checkpoints: COND per-corpus, and irrelevant to iSTFTNet2. For G1, train-from-scratch remains the plan, making weights fully owned regardless.**

## 6. HiFi-GAN v1 — original weights (jik876/hifi-gan)

### Code — `jik876/hifi-gan`
- Primary artifact: https://raw.githubusercontent.com/jik876/hifi-gan/master/LICENSE — retrieved **2026-08-22T18:37:52Z**
- [Q6.1] Exact quote (opening lines):
  > "MIT License"
  > "Copyright (c) 2020 Jungil Kong"

### Original pretrained weights — Google Drive set (LJ_V1/V2/V3, LJ_FT_T2_*, VCTK_V*, UNIVERSAL_V1)
- README: https://raw.githubusercontent.com/jik876/hifi-gan/master/README.md — retrieved **2026-08-22T18:37:53Z**
- [Q6.2] Exact quotes:
  > "We provide our implementation and pretrained models as open source in this repository."
  > "[Download pretrained models](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing)"
- Finding [F1]: the README contains **NO license section and no license statement of any kind for the checkpoints** (full-text reviewed; the only "open source" sentence above is unqualified and grants nothing legally). Checkpoints sit behind Google Drive, outside any version-controlled license surface.
- Note: "as open source" in the README is the author's description, not a license grant; without an stated license, default copyright applies (all rights reserved).

**Verdict: WEIGHTS NO — do not download, use, fine-tune, or redistribute the original HiFi-GAN checkpoints in any commercial or internal capacity without written permission from the author. Code MIT is unaffected: training our own generator from scratch under MIT code produces fully-owned weights.**

---

## Final recommendation row — G1 decoder dependency

| Rank | Dependency | License posture (this dossier) | Decision |
|------|------------------------------------------|---------|
| **ADOPT** | Vocos-class iSTFT-head decoder retrained in-house on Caedral voices, using `gemelo-ai/vocos` (MIT) as reference/training code; `charactr/vocos-mel-24khz` (MIT) usable directly for benchmarking/Phase-G baselines | Code MIT + weights MIT, both verified from primary artifacts; single obligation = MIT notice retention | **Primary G1 path. Ship-safe today.** Pin commit + re-verify LICENSE at release cut (precedent: BigVGAN/EnCodec relicenses show drift is real). |
| Alternative | iSTFTNet2-style sub-1M head, train-from-scratch | Recipe MIT-safe in principle, but its claimed home repo does not contain it — implementation source unresolved [F2] | Use only after locating a real implementation and verifying ITS license; otherwise fold the iSTFT pattern into our own MIT-lineage code. |
| Fallback (G2 codec-token path) | SNAC `snac_24khz` (MIT/MIT) or WavTokenizer `small_600_24k_4096` (MIT/MIT, single-sourced license field) | Both verified MIT from HF card fields | Clean fallback; SNAC stronger provenance (dual-copyright MIT + field-proven TTS deployments). |
| Premium tier only | BigVGAN-v2 family (MIT/MIT) | Verified MIT including weights via card→LICENSE link | License-clean but CPU-unproven; offline/premium SKU pending Phase G measurement. |
| **PROHIBITED** | jik876 HiFi-GAN original Google Drive checkpoints | NO STATED LICENSE [F1] | **Never ship, fine-tune, or redistribute.** Own-training under the MIT codebase is the permitted route. |

## Findings & process notes

- [F1] HiFi-GAN v1 original checkpoints: license absence re-confirmed from the live primary artifact (README), upgrading the survey's "[REPORTED]" to VERIFIED.
- [F2] kan-bayashi/ParallelWaveGAN master tree contains no iSTFTNet/iSTFTNet2 code or configs (complete tree, `truncated:false`) — corrects survey source [14]'s "implementation home" claim. New pending item: identify and verify the real iSTFTNet2 implementation source.
- [F3] `novateur/WavTokenizer` card body has no license text; `license:mit` exists only as card metadata — acceptable but single-source; flag for pin-time re-check.
- [F4] Firecrawl markdown extraction drops HF sidebar badges; the Hub metadata API (`/api/models/<repo>`) is the reliable primary carrier of the card license field — used for all four HF repos.
- Transient HTTP 429 rate-limits hit twice during acquisition; resolved by waiting out the window per FIRECRAWL-WORKFLOW.md. No 401/403 occurred; no item blocked.

## Sources (all retrieved 2026-08-22 via Firecrawl; local artifacts in `research/cache/`)

| # | URL | retrieved_at (UTC) |
|---|-----|--------------------|
| S1 | https://raw.githubusercontent.com/gemelo-ai/vocos/main/LICENSE | 18:35:44Z |
| S2 | https://huggingface.co/charactr/vocos-mel-24khz | 18:35:45Z |
| S3 | https://raw.githubusercontent.com/NVIDIA/BigVGAN/main/LICENSE | 18:35:48Z |
| S4 | https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x | 18:35:50Z |
| S5 | https://raw.githubusercontent.com/hubertsiuzdak/snac/main/LICENSE | 18:35:53Z |
| S6 | https://huggingface.co/hubertsiuzdak/snac_24khz | 18:35:54Z |
| S7 | https://raw.githubusercontent.com/jishengpeng/WavTokenizer/main/LICENSE | 18:35:55Z |
| S8 | https://huggingface.co/novateur/WavTokenizer | 18:35:57Z |
| S9 | https://raw.githubusercontent.com/kan-bayashi/ParallelWaveGAN/master/LICENSE | 18:35:58Z |
| S10 | https://raw.githubusercontent.com/kan-bayashi/ParallelWaveGAN/master/README.md | 18:35:59Z |
| S11 | https://raw.githubusercontent.com/jik876/hifi-gan/master/LICENSE | 18:37:52Z |
| S12 | https://raw.githubusercontent.com/jik876/hifi-gan/master/README.md | 18:37:53Z |
| S13 | https://huggingface.co/api/models/charactr/vocos-mel-24khz | 18:42:04Z |
| S14 | https://huggingface.co/api/models/nvidia/bigvgan_v2_24khz_100band_256x | 18:42:13Z |
| S15 | https://huggingface.co/api/models/hubertsiuzdak/snac_24khz | 18:42:22Z |
| S16 | https://huggingface.co/api/models/novateur/WavTokenizer | 18:42:31Z |
| S17 | https://api.github.com/repos/kan-bayashi/ParallelWaveGAN/contents/egs/ljspeech/voc1/conf | 18:45:23Z |
| S18 | https://huggingface.co/api/models/charactr/vocos-encodec-24khz | 18:45:14Z |
| S19 | https://api.github.com/repos/kan-bayashi/ParallelWaveGAN/git/trees/master?recursive=1 | 18:47:02Z |
