"""CV-G1-E0028 native-iSTFT split renderer for the fused Kokoro ONNX graph.

The production graph models/kokoro/kokoro-v1.0.onnx fuses acoustic + vocoder
and ends in an iSTFT subgraph (/decoder/decoder/generator/istft/stft/*)
implemented via ConvTranspose + an edge-normalization ScatterND chain. That
subgraph is what OpenVINO cannot execute (E0025) and costs ~11% of kernel
time.

This module proves and implements the split path:

  1. build_split_model(): prunes a COPY of the graph down to the ancestors of
     the packed complex-spectrum tensor
        CUT = /decoder/decoder/generator/istft/stft/Concat_output_0
     (shape [1, 22, L]: channels 0..10 = real bins, 11..21 = imag bins,
      n_fft=20, hop=5). The original model file is never modified.

  2. native_istft(): replicates the graph's iSTFT mathematically:
        y = overlap_add(einsum('ck,cl->lk', basis, spec))   # ConvTranspose
        audio = 4.0 * y[10:-10]                             # center=True trim
     Verified against the full graph to <=4.7e-6 abs (float32 noise; the
     graph's Greater/NonZero/ScatterND "normalization" only touches samples
     [0:10] and [-10:], which the center trim discards - so it is a no-op for
     the emitted audio).

  3. librosa_istft(): librosa.istft(..., hop=5, win=20, center=True) * 1.5 -
     equivalent because librosa divides by the hann^2 COLA sum (1.5) which
     the graph's inverse basis already folds in.

Usage:
    python scripts/native_istft_render.py                # self-test/demo
    python scripts/native_istft_render.py --text "..."   # render one text
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as rt

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "models/kokoro/kokoro-v1.0.onnx"
VOICES = REPO / "models/kokoro/voices-v1.0.bin"
DEFAULT_SPLIT_OUT = REPO / "experiments/CV-G1-E0028-istft/kokoro-v1.0-noistft.onnx"

CUT_TENSOR = "/decoder/decoder/generator/istft/stft/Concat_output_0"
ISTFT_INIT_PREFIXES = (
    "decoder.decoder.generator.stft.istft.stft.",
    "/decoder/decoder/generator/istft/stft/",
)
N_FFT = 20
HOP = 5
N_BINS = 11          # n_fft // 2 + 1
OUT_SCALE = 4.0      # graph multiplies reconstruction by Constant_19 = 4.0
TRIM = N_FFT // 2    # center=True drops n_fft//2 from both ends
LIBROSA_ALPHA = 1.5  # hann^2 COLA sum librosa divides by, basis path does not


def _ancestors(graph: onnx.GraphProto, wanted: set[str]) -> set[str]:
    """Names of nodes required to produce `wanted` tensors."""
    prod = {}
    for idx, n in enumerate(graph.node):
        for o in n.output:
            prod[o] = idx
    keep: set[int] = set()
    stack = [prod[t] for t in wanted if t in prod]
    while stack:
        idx = stack.pop()
        if idx in keep:
            continue
        keep.add(idx)
        for i in graph.node[idx].input:
            j = prod.get(i)
            if j is not None and j not in keep:
                stack.append(j)
    return {graph.node[i].name for i in keep}


def build_split_model(out_path: Path = DEFAULT_SPLIT_OUT,
                      src_path: Path = MODEL) -> Path:
    """Write a pruned COPY of the Kokoro graph whose only output is the
    pre-iSTFT complex spectrum. Returns the path written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = onnx.load(str(src_path))
    g = m.graph

    keep_names = _ancestors(g, {CUT_TENSOR})
    nodes = [n for n in g.node if n.name in keep_names]
    keep_io = set()
    for n in nodes:
        keep_io.update(n.input)
    init_names = {i.name for i in g.initializer}
    inits = [i for i in g.initializer if i.name in keep_io]

    # Drop the now-unused graph outputs; expose the cut tensor instead.
    while len(g.output):
        g.output.pop()
    out = g.output.add()
    out.name = CUT_TENSOR
    tt = out.type.tensor_type
    tt.elem_type = onnx.TensorProto.FLOAT
    d0 = tt.shape.dim.add(); d0.dim_value = 1
    d1 = tt.shape.dim.add(); d1.dim_value = 2 * 11  # real|imag bins
    d2 = tt.shape.dim.add(); d2.dim_param = "frames"

    while len(g.node):
        g.node.pop()
    g.node.extend(nodes)
    while len(g.initializer):
        g.initializer.pop()
    g.initializer.extend(inits)

    onnx.save(m, str(out_path))
    return out_path


def load_inverse_basis(src_path: Path = MODEL) -> np.ndarray:
    """The (22, 1, 20) ConvTranspose weight implementing inverse DFT x window."""
    m = onnx.load(str(src_path), load_external_data=False)
    for i in m.graph.initializer:
        if i.name.endswith("inverse_basis"):
            return onnx.numpy_helper.to_array(i).copy()
    raise KeyError("inverse_basis initializer not found")


def native_istft(spec_packed: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Exact replication of the graph's iSTFT.

    spec_packed: [1, 22, L] or [22, L] float32 (real|imag along channel axis)
    basis:       [22, 1, 20] ConvTranspose weight (see load_inverse_basis)
    returns float32 waveform of length L*HOP - TRIM*2 + (N_FFT - N_FFT) ->
    (L - 1) * HOP + N_FFT - 2 * TRIM samples, matching the fused graph's
    'audio' output length for the same frames.
    """
    s = spec_packed.reshape(-1, spec_packed.shape[-1])  # (22, L)
    L = s.shape[1]
    contrib = np.einsum("ck,cl->lk", basis[:, 0, :], s)  # (L, 20)
    y = np.zeros((L - 1) * HOP + N_FFT, dtype=np.float32)
    for k in range(N_FFT):
        y[k::HOP][:L] += contrib[:, k]
    return OUT_SCALE * y[TRIM:-TRIM]


def librosa_istft(spec_packed: np.ndarray) -> np.ndarray:
    """librosa equivalent; needs the 1.5 hann^2-COLA amplitude correction."""
    import librosa
    s = spec_packed.reshape(-1, spec_packed.shape[-1])
    comp = (s[:N_BINS] + 1j * s[N_BINS:]).astype(np.complex64)
    return LIBROSA_ALPHA * librosa.istft(
        comp, hop_length=HOP, win_length=N_FFT, n_fft=N_FFT,
        window="hann", center=True)


class SplitRenderer:
    """kokoro_onnx-compatible synthesis through the split graph + native iSTFT."""

    def __init__(self, split_path: Path = DEFAULT_SPLIT_OUT,
                 voices_path: Path = VOICES, threads: int = 4):
        import kokoro_onnx
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = threads
        sess = rt.InferenceSession(str(split_path), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
        self.k = kokoro_onnx.Kokoro.from_session(sess, str(voices_path))
        self.basis = load_inverse_basis()

    def infer_raw(self, tokens: list[int], style_row: np.ndarray,
                  speed: float = 1.0) -> np.ndarray:
        """Run the split graph; return the packed spectrum [1, 22, L]."""
        feed = {
            self.k._tokens_input: np.array([[0, *tokens, 0]],
                                           dtype=self.k._input_dtypes[
                                               self.k._tokens_input]),
            "style": np.asarray(style_row, dtype=np.float32).reshape(1, -1),
            "speed": np.array([self.k._speed_value(speed)],
                              dtype=np.float32),
        }
        return self.k.sess.run([CUT_TENSOR], feed)[0]

    def synth_tokens(self, tokens: list[int], style_row: np.ndarray,
                     speed: float = 1.0) -> np.ndarray:
        spec = self.infer_raw(tokens, style_row, speed)
        return native_istft(spec, self.basis)


def _demo(text: str, wav_out: Path | None) -> dict:
    """Render `text` through full graph and split path; report equivalence."""
    import kokoro_onnx

    opts = rt.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = rt.InferenceSession(str(MODEL), sess_options=opts,
                               providers=["CPUExecutionProvider"])
    k = kokoro_onnx.Kokoro.from_session(sess, str(VOICES))

    split_path = DEFAULT_SPLIT_OUT
    if not split_path.exists():
        print(f"[split] building {split_path} ...")
        build_split_model(split_path)
    rend = SplitRenderer(split_path)

    phonemes = k.tokenizer.phonemize(text)
    tokens = k.tokenizer.tokenize(phonemes)
    style = k.get_voice_style("af_heart")
    style_row = np.asarray(style[min(len(tokens), len(style)) - 1],
                           dtype=np.float32)

    t0 = time.perf_counter()
    audio_full, _ = k._infer(tokens, style_row, 1.0)
    t_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    spec = rend.infer_raw(tokens, style_row, 1.0)
    t_graph = time.perf_counter() - t0
    audio_nat = native_istft(spec, rend.basis)
    t_native = time.perf_counter() - t0 - t_graph
    audio_lib = librosa_istft(spec)
    t_lib = time.perf_counter() - t0 - t_graph - t_native

    def cmp(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        n = min(a.size, b.size)
        d = np.abs(a[:n] - b[:n])
        corr = float(np.dot(a[:n], b[:n]) /
                     (np.linalg.norm(a[:n]) * np.linalg.norm(b[:n])))
        return {"len_a": int(a.size), "len_b": int(b.size),
                "maxabsdiff": float(d.max()),
                "rel": float(d.max() / max(np.abs(b[:n]).max(), 1e-12)),
                "corr": corr}

    rep = {
        "text": text,
        "frames_L": int(spec.shape[-1]),
        "full_graph_wall_s": round(t_full, 4),
        "split_graph_wall_s": round(t_graph, 4),
        "native_istft_wall_s": round(t_native, 6),
        "librosa_istft_wall_s": round(t_lib, 6),
        "native_vs_full": cmp(audio_nat, audio_full),
        "librosa_vs_full": cmp(audio_lib, audio_full),
    }
    if wav_out is not None:
        import soundfile as sf
        wav_out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_out.with_suffix(".full.wav"), audio_full, 24000)
        sf.write(wav_out.with_suffix(".split.wav"), audio_nat, 24000)
        rep["wavs"] = [str(wav_out.with_suffix(".full.wav")),
                       str(wav_out.with_suffix(".split.wav"))]
    print(rep)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text",
                    default="Hello. This is the first generation of Caedral "
                            "Voice.")
    ap.add_argument("--wav-out", default=None)
    ap.add_argument("--rebuild-split", action="store_true")
    args = ap.parse_args()
    if args.rebuild_split or not DEFAULT_SPLIT_OUT.exists():
        build_split_model(DEFAULT_SPLIT_OUT)
    _demo(args.text, Path(args.wav_out) if args.wav_out else None)


if __name__ == "__main__":
    main()
