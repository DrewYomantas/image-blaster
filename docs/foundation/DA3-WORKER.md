# DA3 Small isolated CPU worker

Unconditioned mode uses official DA3-SMALL relative-depth and camera inference on local images. The explicit experiment-only pose mode described below uses supplied camera scale and provenance. It predicts visual geometry, never measured geometry. Unconditioned mode does not produce metric depth. Neither mode produces PBR materials, completed hidden surfaces, a mesh, SceneSpec, installation facts, or a client import. The Node parent owns coordinate conversion, scale evidence, fusion, SceneSpec and technical QA.

## Reviewed upstream identity and rights

Reviewed on 2026-10-03:

| Component | Immutable identity | Rights |
| --- | --- | --- |
| [Official code](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4) | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | [Apache-2.0 code license](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/LICENSE) |
| [DA3-SMALL model card](https://huggingface.co/depth-anything/DA3-SMALL/blob/e08cab65ca0ec38e7826075418411ab90cab4da3/README.md) | `e08cab65ca0ec38e7826075418411ab90cab4da3` | Card declares Apache-2.0, commercial use permitted subject to license and input rights |
| `model.safetensors` | SHA-256 `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`, 137,248,940 bytes | Same pinned Small model |
| `config.json` | SHA-256 `a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2` | Same pinned Small model |

The model card describes roughly 0.08B parameters, relative depth and camera prediction. Large/Giant/Nested variants have different noncommercial rights and are excluded. The selected model repository does not include a separate LICENSE file: the license evidence is its pinned card plus the official model table. Preserve upstream license/notices on redistribution. Do not infer that all optional dependencies share the same license.

## CPU runtime and installation

Use an isolated Python 3.12 environment. System Python 3.14 is outside upstream's `>=3.9, <=3.13` declared range; selected NumPy 1.26.4 has Python 3.12 wheels. No ROCm, DirectML, CUDA, xformers, gsplat, Blender, paid endpoint or cloud GPU is used. The Ryzen 5600X CPU is selected deliberately; the RX 6800 XT is unused.

After the fixture/evaluation harness exists and installation is authorized:

```powershell
powershell -NoProfile -File workers/da3/setup.ps1
$env:IMAGE_BLASTER_DA3_PYTHON = (Resolve-Path workers/da3/.venv/Scripts/python.exe).Path
& $env:IMAGE_BLASTER_DA3_PYTHON workers/da3/worker.py --identity
```

`setup.ps1` clones only the official source at the exact revision, installs CPU PyTorch from its CPU wheel index and pinned remaining inference dependencies from PyPI, and downloads only the pinned Small checkpoint/config/card. `.venv/` and `.runtime/` are ignored. Installed transitive versions are recorded in `.runtime/installed-packages.txt` and immutable request identity; this is an initial runtime receipt, not a fully reproducible transitive lock. No model/environment belongs in Git.

The upstream full package declares unrelated serving/export/3D/GPU dependencies. Its public API eagerly imports those exporters and enables half-precision autocast. This adapter instead constructs the exact configured official network, strictly loads the official safetensors state after removing the API wrapper's `model.` key prefix and restoring its six shared LayerNorm aliases, and calls official `InputProcessor` and `OutputProcessor` around a float32 network forward. It retains the official camera head, reference-view selection and preprocessing. This bounded CPU compatibility path avoids importing unused exporters and avoids changing network math or adding a GPU port. Actual CPU inference must be verified by a real run.

Selected dependencies have their own licenses (PyTorch/torchvision BSD, NumPy BSD, OpenCV Apache-2.0, Pillow HPND, OmegaConf BSD, addict MIT, einops MIT, safetensors Apache-2.0, imageio BSD, tqdm MIT/MPL). This is a direct-dependency summary; installed package notices and all transitive redistribution obligations remain applicable.

Actual Windows `opencv-python==4.11.0.86` inspection found `cv2/opencv_videoio_ffmpeg4110_64.dll`; its installed `LICENSE-3RD-PARTY.txt` includes FFmpeg's LGPL-2.1 terms. The wrapper itself is MIT and OpenCV Apache-2.0. Preserve bundled-library notices and evaluate worker redistribution obligations before packaging binaries; top-level Apache does not erase them. This milestone installs locally and commits neither that DLL nor the environment. [Official bundled licenses](https://github.com/opencv/opencv-python/blob/4.x/LICENSE-3RD-PARTY.txt).

## Subprocess protocol

The process handles one request and exits. There is no resident server or ready-handshake. `--identity` prints one finite JSON object without importing any ML package or running inference. It hashes actual checkpoint/config bytes, validates all tracked upstream Python/YAML/TOML bytes against the exact Git tree, rejects additional Python source files, and checks pinned installed direct dependency versions. Missing or changed files fail closed. It returns exact model/code revisions, checkpoint/config/source digests, adapter file hashes, Python/dependency versions and CPU dtype.

The parent may supply `expectedIdentity` from preflight; the worker requires equality before inference and repeats identity/input hashes after inference. Configure optional local paths through `IMAGE_BLASTER_DA3_MODEL_DIR` and `IMAGE_BLASTER_DA3_SOURCE_DIR`; paths are not immutable identity.

Supply JSON through stdin or `--request <path>`:

```json
{
  "inputs": [{"path":"C:/inputs/view-1.png"},{"path":"C:/inputs/view-2.png"}],
  "outputDir":"C:/outputs/run-1",
  "parameters":{"processResolution":384,"threads":6,"profileWarmInference":false}
}
```

The request accepts only `inputs`, `outputDir`, `parameters`, and optional `expectedIdentity`. Each input contains only `path`. It allows one to eight unique local PNG/JPEG images of identical dimensions. Scale anchors, scene/camera truth, labels, depth truth, materials and masks are forbidden. `processResolution` is integer 384 or 256 (worker default 384; parent may explicitly choose 256). `threads` is integer 1 to 64, default 6, and sets actual PyTorch CPU thread count. `profileWarmInference` is a boolean, default false. The output directory must be absent or empty. The worker does not read sibling fixture truth. Model/debug logs go to stderr; stdout holds exactly one JSON response. Success exits zero; failure exits one with `{protocolVersion:1,status:"failed",error:...}` and no completion claim.

Success returns `status`, `identity`, `parameters`, `coordinates`, ordered `inputs` hashes, `cameras`, `geometryArtifact`, `artifactSha256`, `files`, and `telemetry`. Every camera has `sourceIndex`, numeric `intrinsics` (3x3), `extrinsics` (3x4), `imageWidth`, `imageHeight`, original dimensions and explicit resize scale. `files` lists absolute `predictions.npz` and `worker-result.json` paths. Telemetry includes load/preprocess/inference/total seconds and peak process working-set bytes. Startup and output compression time are not included in the current total timer.

`loadSeconds` times network construction and checkpoint state loading after Python/ML imports. `inferenceSeconds` times the first forward and output conversion on preprocessed inputs, including first-use kernels. `profileWarmInference: true` permits exactly one additional forward on the same loaded model and same processed images, records `warmInferenceSeconds`, `inferenceForwardCalls: 2` and `warmMaxDepthDifference`, and preserves the first prediction as primary output. Default production execution records one forward. The parent includes this option in normalized request identity; cache hits do not run either forward. Warm timing is a bounded local profile, not a distribution of repeated benchmarks.

## Geometry contract

`predictions.npz` has these named arrays, with no pickle objects:

| Key | Shape | Meaning |
| --- | --- | --- |
| `depths` | N,H,W float32 | Positive camera-z in relative model units |
| `confidences` | N,H,W float32 | Raw uncalibrated confidence; higher is better, not a probability |
| `intrinsics` | N,3,3 float32 | Pixel intrinsics at processed output resolution |
| `extrinsics` | N,3,4 float32 | World-to-camera OpenCV: x-right, y-down, z-forward |
| `processed_images` | N,H,W,3 uint8 | Official processed RGB views in original input order |

Upstream `vision_transformer.py` calls `reorder_by_reference`, then `restore_original_order` on each emitted feature layer before the output heads. The adapter never sorts image inputs. Official `upper_bound_resize` first resizes the longest side to the requested bound, then rounds dimensions to nearest multiples of 14. Therefore 384 becomes 378 and 256 becomes 252 on the longest side. Both steps use OpenCV resize; pixel-centre mapping is `output = (input + 0.5) * scale - 0.5`, recorded as scale and half-pixel offset. Identical dimensions eliminate batch center-cropping. Intrinsics apply directly to the predicted pixels; do not use original image intrinsics without the recorded resize. No inferred metre conversion happens in this worker.

## Verification boundary

The model-free request-boundary tests run with ordinary Python:

```powershell
python -m unittest discover -s workers/da3 -p test_worker.py
```

These tests do not prove CPU imports, model output quality, fusion, runtime performance or physical/client acceptance. Real fixture run evidence and failures are recorded separately after authorized installation.

## Installation and checkpoint-loading evidence

The isolated environment installed Python 3.12.12, PyTorch 2.7.1+cpu, torchvision 0.22.1+cpu and the direct pins above. Actual downloaded checkpoint/config hashes match the reviewed identities; source bytes match the pinned Git tree. `--identity` succeeds without ML imports. Setup initially hit uv index priority when the CPU wheel index hid the pinned tqdm version; separating CPU torch installation from remaining PyPI requirements fixed that dependency resolution.

The first real Node invocation failed before forward inference because naive safetensors `load_file` plus strict state loading reported six absent `head.scratch.output_conv2_aux.{1,2,3}.2.{weight,bias}` names. The failure remains in `.image-blaster/benchmark/first-real-run.stderr.log` and failed cache entry `b6ce96a859f94ce42dd5d30ca09edeaf78261acad3716e5da6fd12cd53dcf73c`; it is not evidence of a geometry-quality run.

These names refer to shared LayerNorm parameters, not missing trained heads. [Pinned DualDPT lines 131-150](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/model/dualdpt.py#L131-L150) reuse the same `ln_seq` module instances across all four auxiliary projections. The checkpoint's actual safetensors metadata maps exactly those six aliases to the level-zero weight/bias tensors. The [official Hub mixin](https://github.com/huggingface/huggingface_hub/blob/v0.36.0/src/huggingface_hub/hub_mixin.py) uses `safetensors.load_model`, which handles [shared-tensor deduplication](https://huggingface.co/docs/safetensors/torch_shared_tensors).

Adapter version `da3-small-cpu-2` validates the exact six metadata mappings, requires them to be the only omitted model keys, proves each alias shares actual storage address/shape/stride with its level-zero parameter, restores those dictionary aliases and retains `load_state_dict(strict=True)`. Other missing/unexpected names, altered metadata, untied storage and tensor shape mismatch fail. No learned parameter is randomly substituted or ignored. Thirteen model-free tests cover this contract. An actual pinned-model construction and strict-checkpoint-load probe returned `<All keys matched successfully>`, restored six aliases and performed zero forward calls. Inference evidence belongs to the separately recorded subsequent Node run.

The first successful real Node CPU run is preserved at cache key `5aa9505c424bdc0f38bc89ce3f8de3ffead1c1ee66782738e986357445899f90`, adapter version 2. Its four 640x480 synthetic input images produced ordered 252x196 depth/confidence/camera evidence. The worker receipt records load 0.3818 seconds, first inference 0.5671 seconds, total 5.5428 seconds, peak working set 644,894,720 bytes and a 1,499,012-byte NPZ. These are one observed synthetic run, not a hardware/scene-wide performance guarantee or geometry-quality pass. Root recorded a repeated engine cache response in 0.4913 seconds without another forward; that is cache timing, not warm model inference. Version 3 adds the explicitly requested optional warm-forward profile. Fifteen model-free tests cover its typed request boundary; its subsequent real profile evidence is recorded by the parent benchmark.


## Explicit camera-conditioning experiment (adapter cpu-4)

See [camera-conditioning benchmark](CAMERA-CONDITIONING-BENCHMARK.md) for the fixed experiment, processed intrinsics proof, provenance and Outcome2. Existing descriptions of relative/no-metric/images-only behavior above apply to `none` and original receipts. Optional top-level `conditioning` is explicitly separate from parameters: `{mode:"intrinsics-only"|"pose", provenance:"experiment-oracle", cameraConvention:"opencv-world-to-camera-scene-y-up-meters", intrinsics:Nx3x3, extrinsics:Nx4x4 (pose only)}`. Normal requests omit it. Mode none accepts no cameras; no truth is discovered from input directories.

Pose uses exact official InputProcessor, API-equivalent source0/median-distance normalization, camera tokens and official align_poses_umeyama. Output K/E are supplied input, not independent reconstructed cameras. Depth is inferred and scaled through supplied-camera alignment, never GT depth. `conditioning` metadata records original/processed/normalized matrices, scale multiplier, ordering and provenance. Raw decoder depths/K/E remain separate NPZ arrays. New scalar conditioning_mode/depth_scale_source require explicit receipts in evaluation. Worker camera hash is Python sorted compact JSON, explicitly distinct from input file byte hash and Node generation identity.

Image pixelTransform remains actual half-pixel RGB sampling; supplied K follows upstream row scaling without that offset. Do not apply image pixelTransform to supplied K. Both K conventions and numeric differences remain visible. Outputs use metres only in pose mode and source scene frame. Scale source is supplied-camera-path; technical/installation authority is never introduced.

Pinned evo==1.33.0 supplies the official alignment helper's dependency. evo is GPL-3.0-or-later, so a distributed/commercial worker needs separate licensing review; DA3 Apache code/weights do not erase that dependency's terms. No worker binary/package is distributed in this milestone.

25 isolated worker tests cover contract validation, zero-skew/proper/noncollinear cameras, camera convention/count, official resizing256/384, API-equivalent normalization, negative control backbone switch and official scale alignment. Ordinary Python without upstream packages runs21 and skips4 optional official-runtime mechanics tests. All25 were executed here without loading weights for unit tests. Real B/C/D inference and cache evidence are separate receipts.
