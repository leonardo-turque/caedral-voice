"""E0028 step 2: capture intermediates and reverse-engineer the exact
iSTFT normalization the graph performs, then validate native replications.
"""
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as rt
from onnx import numpy_helper as nh

REPO = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
INSTRUMENTED = EXP / "instrumented.onnx"

KEEP = [
    "/decoder/decoder/generator/Exp_output_0",
    "/decoder/decoder/generator/Sin_output_0",
    "/decoder/decoder/generator/istft/stft/Concat_output_0",
    "/decoder/decoder/generator/istft/stft/ConvTranspose_output_0",
    "/decoder/decoder/generator/istft/stft/ScatterND_output_0",
    "/decoder/decoder/generator/istft/stft/Mul_4_output_0",
    "audio",
]


def get_feed():
    import kokoro_onnx
    opts = rt.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = rt.InferenceSession(str(INSTRUMENTED), sess_options=opts,
                               providers=["CPUExecutionProvider"])
    k = kokoro_onnx.Kokoro.from_session(
        sess, str(REPO / "models/kokoro/voices-v1.0.bin"))
    text = "Hello. This is the first generation of Caedral Voice."
    phonemes = k.tokenizer.phonemize(text)
    tokens = k.tokenizer.tokenize(phonemes)
    style = k.get_voice_style("af_heart")
    style_row = np.asarray(style[min(len(tokens), len(style)) - 1],
                           dtype=np.float32).reshape(1, -1)
    feed = {
        "tokens": np.array([[0, *tokens, 0]], dtype=np.int64),
        "style": style_row,
        "speed": np.array([1.0], dtype=np.float32),
    }
    return sess, feed


def main():
    sess, feed = get_feed()
    outs = sess.run(KEEP, feed)
    data = {name.split("/")[-1]: np.asarray(a) for name, a in zip(KEEP, outs)}
    spec = data["Concat_output_0"][0]              # (22, L)
    y_raw = data["ConvTranspose_output_0"].ravel() # (94820,)
    y_scat = data["ScatterND_output_0"].ravel()
    y_scaled = data["Mul_4_output_0"].ravel()
    audio = data["audio"].ravel()

    # graph constants
    m = onnx.load(str(REPO / "models/kokoro/kokoro-v1.0.onnx"),
                  load_external_data=False)
    inits = {i.name: nh.to_array(i) for i in m.graph.initializer}
    basis = inits["decoder.decoder.generator.stft.istft.stft.inverse_basis"]  # (22,1,20)
    wsum = inits["decoder.decoder.generator.stft.istft.stft.window_sum"]     # (20,)
    fwd_win = inits["/decoder/decoder/generator/Constant_17_output_0"]       # (20,) hann
    np.savez_compressed(EXP / "intermediates.npz", **data,
                        basis=basis, wsum=wsum, fwd_win=fwd_win)

    n_fft, hop = 20, 5
    L = spec.shape[1]
    report = {"frames_L": int(L), "y_raw_len": int(y_raw.size)}

    # --- hypothesis H1: manual conv_transpose replicates ConvTranspose node ---
    contrib = np.einsum("ck,cl->lk", basis[:, 0, :], spec)   # (L, 20)
    y_man = np.zeros((L - 1) * hop + n_fft, dtype=np.float64)
    for k in range(n_fft):
        y_man[k::hop][:L] += contrib[:, k]
    d1 = np.abs(y_man - y_raw).max()
    report["H1_convtranspose_manual_maxdiff"] = float(d1)

    # --- hypothesis H2: normalization = divide by sliding-sum of wsum tile ---
    env = np.zeros_like(y_man)
    for k in range(n_fft):
        env[k::hop][:L] += wsum[k]
    pos = env > 0
    y_h2 = np.where(pos, y_man / np.maximum(env, 1e-12), 0.0)
    d2 = np.abs(y_h2 - y_scat).max()
    rel2 = d2 / max(np.abs(y_scat).max(), 1e-12)
    report["H2_windowsum_env_maxdiff"] = float(d2)
    report["H2_rel"] = float(rel2)

    # --- hypothesis H3: scale 4.0 then trim [10:-10] reproduces audio ---
    a_h3 = (4.0 * y_h2)[10:-10]
    d3 = np.abs(a_h3 - audio).max()
    report["H3_full_native_maxdiff"] = float(d3)
    report["H3_rel_to_absmax"] = float(d3 / np.abs(audio).max())

    # correlation end-to-end
    num = float(np.dot(a_h3.astype(np.float64), audio.astype(np.float64)))
    den = float(np.linalg.norm(a_h3) * np.linalg.norm(audio))
    report["H3_corr"] = num / den

    # --- hypothesis H4: librosa.istft equivalence up to scalar ---
    try:
        import librosa
        comp = (spec[:11] + 1j * spec[11:]).astype(np.complex64)  # (11, L)
        y_lib = librosa.istft(comp, hop_length=hop, win_length=n_fft,
                              n_fft=n_fft, window="hann", center=True,
                              length=None)
        # align lengths
        n = min(y_lib.size, audio.size)
        # fit global scalar
        x = y_lib[:n].astype(np.float64)
        yref = audio[:n].astype(np.float64)
        alpha = float(np.dot(x, yref) / np.dot(x, x))
        resid = np.abs(alpha * x - yref)
        report["librosa_len"] = int(y_lib.size)
        report["librosa_alpha_fit"] = alpha
        report["librosa_scaled_maxdiff"] = float(resid.max())
        report["librosa_scaled_rel"] = float(resid.max() / np.abs(yref).max())
        report["librosa_corr_raw"] = float(
            np.dot(x, yref) / (np.linalg.norm(x) * np.linalg.norm(yref)))
        # edge behaviour: interior-only residual
        interior = slice(200, n - 200)
        report["librosa_interior_maxdiff"] = float(resid[interior].max())
    except Exception as e:  # noqa: BLE001
        report["librosa_error"] = repr(e)

    # --- edge analysis: how big is the boundary mismatch zone for librosa ---
    if "librosa_scaled_maxdiff" in report:
        thresh = 1e-4
        bad = np.nonzero(resid > thresh)[0]
        report["librosa_badzone_count"] = int(bad.size)
        if bad.size:
            report["librosa_badzone_first_last"] = [int(bad[0]), int(bad[-1])]

    print(json.dumps(report, indent=2))
    (EXP / "verify_math.json").write_text(json.dumps(report, indent=2),
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
