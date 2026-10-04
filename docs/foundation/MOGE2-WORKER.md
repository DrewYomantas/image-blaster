# MoGe-2 ViT-S Normal isolated CPU worker

## Rights declaration recorded before installation

Official source is microsoft/MoGe revision `74fbce054ebed49800de42d0ad0e83495065719a`. [Pinned source license](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/LICENSE) is MIT; bundled DINOv2 source headers and [official README licensing section](https://github.com/microsoft/MoGe/tree/74fbce054ebed49800de42d0ad0e83495065719a#license) declare Apache-2.0. Only `moge.model.v2` is imported. MoGe-3, FlexGEMM, Triton, training and mesh/export/demo packages are excluded.

The exact [checkpoint card](https://huggingface.co/Ruicheng/moge-2-vits-normal/blob/26b477f41595707c5db6770294c0d1721e8ed4ed/README.md) declares MIT for this specific Small Normal checkpoint. File `model.pt` is 140,550,416 bytes with SHA-256 `79a16621928c2bf0ed04659218c55c01075e950507f40bb3332fb4c873d3e1dc`, model revision `26b477f41595707c5db6770294c0d1721e8ed4ed`. Official source model table advertises approximately 35M parameters, metric points/depth and normals. Parameter count and downloaded bytes are verified during local installation, not inferred from other checkpoint licenses.

The exact helper source declared by upstream is [utils3d_moge revision62f09d58509485564e24d5d9f6aac9ee9ebc0c37](https://github.com/EasternJournalist/utils3d-moge/tree/62f09d58509485564e24d5d9f6aac9ee9ebc0c37), MIT. It is loaded directly from a pinned source checkout. Its optional rendering dependency moderngl is excluded because the reference v2 inference uses only lazy-loaded NumPy/Torch geometry functions.

Direct runtime pins: CPU PyTorch2.7.1+cpu (BSD-style), NumPy2.2.6 (BSD), SciPy1.15.3 (BSD), opencv-python-headless4.11.0.86 (wrapper MIT/OpenCV Apache-2.0), Pillow11.2.1 (MIT-CMU/HPND), huggingface-hub0.36.0 (Apache-2.0). Exact installed transitives are recorded under ignored `.runtime/installed-packages.txt` and in worker identity. Preserve all upstream licenses/notices on redistribution. NumPy/SciPy bundled BLAS/LAPACK, compiler runtimes and native wheel notices and OpenCV bundled codecs are separate obligations. Inspect installed native wheels before any distribution claim. No blanket commercially-clean or redistribution approval is made. Local inference requires no paid service or credentials.

## Scope

Independent monocular visual inference only. Native metric units are learned prediction, never field measurements, manufacturer evidence, registered geometry, installation truth, TPS terrain or recovered inter-view poses. Truth extrinsics remain evaluation-only.

## Verified local installation and native notices

The isolated runtime is Python3.12.12 and contains only the direct pins above plus recorded transitives. Identity succeeds without importing Torch or NumPy. The downloaded exact checkpoint has 35,103,656 parameters and configured token range 1200-3600. An actual official `from_pretrained` construction followed by exact key-set comparison and `load_state_dict(strict=True)` succeeded with zero forward calls. This proves loading, not geometry quality.

`.runtime/preflight-v1.json` stores full source/checkpoint identity, exact model-card hash, native DLL/notice hashes and the zero-forward checkpoint probe. `.runtime/installed-packages.txt` stores the installed transitive versions. These local receipts and all environments/model bytes are ignored by Git.

Actual Windows wheel inspection found:

| Installed component | Actual evidence | Redistribution consideration |
| --- | --- | --- |
| opencv-python-headless4.11.0.86 | `cv2/opencv_videoio_ffmpeg4110_64.dll`; `cv2/LICENSE-3RD-PARTY.txt` explicitly includes FFmpeg LGPL2.1 | Headless does not remove FFmpeg. Preserve bundled notices and evaluate LGPL distribution requirements before packaging. |
| NumPy2.2.6 | `numpy.libs/libscipy_openblas64_-13e2df515630b4a41f92893938845698.dll` and `msvcp140` DLL; wheel LICENSE lists OpenBLAS/LAPACK and GCC runtime | BSD attribution plus GPL3 GCC runtime exception and Microsoft runtime distribution terms must be preserved/appraised. |
| SciPy1.15.3 | `scipy.libs/libscipy_openblas-f07f5a5d207a3a47104dca54d6d0c86a.dll`; wheel LICENSE lists OpenBLAS/LAPACK/GCC runtime | Retain full wheel license and runtime exception text. |
| PyTorch2.7.1+cpu | `torch/lib/libiomp5md.dll`, `torch_cpu.dll`, `fbgemm.dll`, `asmjit.dll`, `uv.dll` and LICENSE/NOTICE | Top-level BSD does not erase component notices or OpenMP/compiler-runtime terms. |
| Pillow11.2.1 | Installed LICENSE contains MIT-CMU plus bundled Brotli, FreeType, HarfBuzz, libjpeg, LCMS, libpng, libtiff, WebP, OpenJPEG, raqm, XZ/zlib notices | Retain full bundled notices; FreeType provides alternative license paths rather than a blanket Pillow-only permission. |
| certifi2026.7.22 transitive | Installed LICENSE is MPL2.0 | Retain applicable certificate-bundle distribution notices. |

No binary package is redistributed by this milestone. Local MIT-checkpoint inference eligibility is separate from approval to ship these runtime binaries.

## Reference inference semantics

The adapter directly imports the pinned `moge.model.v2.MoGeModel`, with the exact upstream-declared MIT utils3d helper source. It never imports v3 or installs FlexGEMM, Triton, xformers, moderngl, training/export/demo dependencies. Source checkouts contain the official repository's tracked files, including unused versions; that does not run or benchmark them. The exact tracked Python/license/config bytes are verified against Git blobs. Setup restores literal Git bytes in these newly created ignored checkouts when Windows automatic newline conversion occurs; identity still rejects subsequent changed source bytes.

Upstream `from_pretrained` uses `weights_only=True` CPU checkpoint deserialization and constructs the model from its embedded configuration, but internally loads with `strict=False`. This worker invokes the official loader, verifies its exact model key set against the checkpoint, then repeats strict loading. Missing, unexpected or mismatched tensors fail; no randomly initialized parameter is accepted. DINOv2 is constructed with `pretrained=False` and the complete MoGe checkpoint supplies its backbone weights. No additional backbone download occurs. `HF_HUB_OFFLINE=1` is set before imports.

Each image is an independent RGB float32 tensor at original resolution. The official encoder bilinearly resizes internally to its token grid and applies ImageNet normalization. Decoder maps are interpolated back to source H/W by the official model. `resolutionLevel=9` selects 3600 tokens from the checkpoint's1200-3600 range, preserving upstream default. There is no adapter crop, pad or resize. CPU float32 uses `use_fp16=False`; official postprocessing's float32 CPU autocast emits a harmless warning and disables autocast. No mixed-precision or alternate graph is substituted.

The network predicts affine points, normalized normals, mask probability and an exponentiated learned metric-scale scalar. Official `infer()` thresholds the mask at the fixed upstream 0.5 value, solves focal/depth shift from its predicted points on a fixed 64x64 grid, and uses a normalized principal point of (0.5,0.5). With known horizontal FOV, the network forward is unchanged; the focal is supplied to this post-forward solve and only the depth shift is solved. Thus FOV is supported camera information for postprocessing, not neural camera-token conditioning.

The adapter retains `force_projection=True`: official inference recomputes camera-space points from its predicted depth/intrinsics, multiplies points/depth by the learned metric-scale head, and applies its native mask. It preserves exactly these final model-native point maps. Invalid point/depth locations remain positive infinity and normal vectors remain zero, matching official `apply_mask=True`. No outlier trimming, mask tuning, per-object scale, anchor or oracle pose is used.

Normalized intrinsics are converted through official `utils3d.pt.denormalize_intrinsics(...,pixel_convention='integer-center')`, which applies W/H scales and minus0.5 pixel offsets. A normalized(0.5,0.5) centre maps to (319.5,239.5) at 640x480. Regression tests verify exact fixture fx/fy/cx/cy and source resolution. OpenCV points are x-right,y-down,z-forward; depth equals camera Z. Normals are signed unit vectors facing the visible camera: upstream training derives normal truth with `depth_map_to_normal_map`, whose `cross(up,left)` gives negative Z for a front-facing plane, and uses signed angular supervision. Tests exercise this exact helper.

## Subprocess and artifacts

Use `--identity` for model-free immutable preflight, stdin JSON or `--request path` for one process handling one request. Stdout contains exactly one finite JSON response; diagnostics go to stderr. Parent `expectedIdentity` must match and identity/source-image hashes are rechecked after inference. Failure exits 1 with explicit protocol failure. Output must be absent or empty.

```json
{
  "inputs": [{"path":"C:/inputs/view-1.png"}],
  "outputDir":"C:/outputs/run-1",
  "parameters":{"resolutionLevel":9,"threads":6},
  "conditioning":{"mode":"known-fov","provenance":"experiment-oracle","horizontalFovDegrees":[85.0]}
}
```

Omit conditioning for native inferred-FOV mode. Known-FOV accepts exactly one finite horizontal degree value with 1<value<179 per ordered input. No truth depth, masks, labels, dimensions, anchor, K or extrinsics are accepted or discovered through directory conventions. Normal production inputs contain no oracle FOV. One to eight unique local PNG/JPEG paths of identical dimensions are accepted. Resolution is typed integer 0-9, threads integer 1-64. All other parameters fail.

`predictions.npz` holds native `camera_points[N,H,W,3]`, `depths[N,H,W]`, boolean `masks[N,H,W]`, `normals[N,H,W,3]`, `intrinsics_normalized[N,3,3]`, integer-centre pixel `intrinsics[N,3,3]`, original RGB `processed_images[N,H,W,3]`, and scalar Unicode `conditioning_mode`. No extrinsics are fabricated. Source-camera records hold projection only. Parent SceneSpec evidence remains inferred visual-only; FOV supplied by the experiment is never measured/manufacturer evidence or independent inferred-camera accuracy.

`worker-result.json` records exact input conditioning, separate inferenceSemantics, source hashes/order, identity projection transforms, coordinate/scale conventions, artifact hash, fixed inference flags, model load/preprocess/per-image and aggregate inference/write/total times, effective token count, parameter count and peak process working-set memory. Load excludes Python/ML imports; total includes preflight, imports and compressed NPZ write but excludes final JSON serialization/process exit. Engine wall time separately includes the subprocess lifecycle and cache bookkeeping. There is no resident-worker optimization.

## Verification

```powershell
powershell -NoProfile -File workers/moge2/setup.ps1
& workers/moge2/.venv/Scripts/python.exe workers/moge2/worker.py --identity
py -3.13 -m unittest discover -s workers/moge2 -p test_worker.py
& workers/moge2/.venv/Scripts/python.exe -m unittest discover -s workers/moge2 -p test_worker.py
```

Fourteen tests pass in the isolated environment. Eleven pass under ordinary Python, with three optional installed-runtime tests skipped. They exercise request/FOV boundaries, forbidden hidden truth, source/checkpoint/dependency identity drift, strict load, mask/point/depth/normal shapes, official normalized-pixel intrinsics, normal orientation, and FOV's post-forward/metric-scale behavior with a mocked network. No unit test loads model weights or requires actual neural inference. A separate zero-forward real checkpoint probe is recorded above. Real M1/M2 inference, cache replay and geometry quality belong to the benchmark report.
