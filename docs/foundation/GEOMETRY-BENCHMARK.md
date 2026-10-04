# DA3 Small observed-geometry experiment

Completed October 4, 2026, in `C:\Projects\image-blaster`, starting from `4dce4527ba39f46c130d72a7248fdfd015deadce` on `codex/provider-neutral-foundation`. Origin is [DrewYomantas/image-blaster](https://github.com/DrewYomantas/image-blaster); upstream remains [neilsonnn/image-blaster](https://github.com/neilsonnn/image-blaster), foundation baseline `4acb43ba126a12358f71838d1b1a05e856b10eaf`. Neither main nor either client was modified. All execution was local, with zero API/cloud/GPU charges.

**Decision: continue controlled local experiments, not dimensional integration.** DA3 Small runs cheaply on this Windows CPU and captures useful relative depth/rotation structure. Focal estimates, single-anchor metric geometry and multi-view object bounds are too inaccurate for dimensional workflows. The calibrated opening width is not a held-out success. No production mesh, customer visualization, installation truth or client readiness is established.

## Architecture delivered

- Separate `scene-geometry` capability and `da3-small` local/free provider. Semantic `scene-analysis`, procedural OBJ and six hosted adapters remain available.
- One-request Python subprocess in `workers/da3`, isolated from Node. Engine startup, normal tests and other providers require no Python/checkpoint.
- Automatic bounded implementation fingerprint plus actual checkpoint/config/source/runtime identity in effective request, artifacts and manifest. Local transitive helper bindings and explicit variable-import roots are retained. Changed code/checkpoint misses; irrelevant execution metadata hits. Corruption, failed-job reconciliation, concurrent locks, junction containment and evidence preservation remain enforced.
- Additive SceneSpec v1 typed camera projection/pose evidence, source-image resize mapping, relative model-world poses and visual-only external tensor artifacts. Relative values are never silently called metres.
- Deterministic synthetic renderer/evaluator, explicit one-anchor scale evidence, separate aligned NPZ, annotated inferred scene bounds and synthetic measured-role override. Benchmark labels are evaluation annotations, not model object recognition.
- Benson technical export remains fail closed. The whole-scene technical gate was not broadened for mixed authority. A future scoped technical-entity contract must explicitly bind product/geometry/datum/measurement/rule provenance; one measured opening width cannot approve an approximate room.

## Frozen fixture

The renderer was built and five oracle/distortion/resize tests established before model inference. It is a small project-authored NumPy analytic ray/box CPU renderer, not stock imagery or a renderer optimized for DA3. Inputs never changed after seeing model outputs.

| Element | Exact width / height / depth (m) |
| --- | --- |
| Room | 4.80 / 2.70 / 3.60 |
| Opening/recess | 1.20 / 0.90 / 0.35 |
| Hearth | 1.80 / 0.14 / 0.55 |
| Mantel | 1.90 / 0.12 / 0.24 |
| Tall left box | 0.48 / 1.00 / 0.42 |
| Foreground stool | 0.52 / 0.60 / 0.52 |
| Right seat box | 0.76 / 0.80 / 0.66 |

Four 640x480 views use `fx=fy=350`, `cx=319.5`, `cy=239.5`. Positions in metres: `(-1.55,1.60,3.00)`, `(-0.55,1.45,3.10)`, `(0.65,1.65,3.00)`, `(1.60,1.50,2.85)`, each looking toward `(0,1.15,0.35)`. Exact CV world-to-camera matrices, analytic camera-Z depth, labels, bounds, near/far settings, deterministic settings and renderer/source version are regenerated separately under `ground-truth`. World coordinates are right-handed X-right/Y-up/Z-toward-rear-wall. Model inputs are only four staged PNG paths; no camera, depth, semantic labels, truth manifest or scale hint reaches the worker.

All views face the fireplace; rear-room depth is incompletely observed. This is an explicitly bounded first multi-view fixture, not full room coverage. Input hashes and generator protocol are in the [fixture instructions](../../benchmarks/geometry/README.md) and [numeric receipt](geometry-results/da3-small-cpu-v1.json).

## Model and installation receipt

Exact code revision: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`. Model: `depth-anything/DA3-SMALL`, revision `e08cab65ca0ec38e7826075418411ab90cab4da3`. Weight SHA-256: `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`, 137,248,940 bytes. Config SHA-256: `a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2`. Both official source license and pinned Small weight card declare Apache-2.0. Other DA3 family weight licenses are not assumed equivalent. See [primary sources and installed dependency caveats](DA3-WORKER.md).

Isolated Python 3.12.12, PyTorch `2.7.1+cpu`, torchvision `0.22.1+cpu`, NumPy `1.26.4`, OpenCV `4.11.0.86`, Pillow `11.2.1`, OmegaConf `2.3.0`, addict `2.4.0`, einops `0.8.1`, safetensors `0.5.3`, imageio `2.37.0`, tqdm `4.67.1`. Full installed transitive versions are retained in numeric/runtime identity. Direct dependencies are pinned; setup's transitive resolution is recorded, not yet a complete installation lock.

Windows 11 Home `10.0.26300`, AMD Ryzen 5 5600X, 34,254,012,416 bytes RAM, Node 24.13.1, npm 11.8.0. Six CPU threads, float32; RX 6800 XT unused. No DirectML/ROCm/WSL/ONNX port or Linux/cloud worker. An owned CPU is sufficient for this four-view reduced-resolution experiment; larger photo sets/resolutions are unbenchmarked.

The official public API imports unrelated exporter/GPU dependencies and enables autocast. The owned worker faithfully uses the pinned configured network and official input/output processors, avoiding those unused imports and using CPU float32. Initial strict checkpoint loading failed on six shared LayerNorm aliases; the repair restores only exact safetensors metadata aliases, proves identical model storage/shape/stride, and retains strict loading. No tensors were ignored or substituted with random weights. Failed entry `b6ce96a859f94ce42dd5d30ca09edeaf78261acad3716e5da6fd12cd53dcf73c` and error log remain locally preserved.

OpenCV's actual Windows wheel includes an FFmpeg DLL and LGPL-2.1 notices; top-level Apache/MIT does not erase bundled redistribution obligations. No checkpoint, environment or native wheel is committed. Upstream MIT source notices remain intact.

## Real CPU measurements

Final profiled effective key: `dec33e79cee34d31a8667fe9ac44142db9f8318ae50b245b7000721d2e0c5ffd`. Requested resolution 256 becomes 252x196 through official resize-to-multiples-of-14 processing; pixel-centre mapping is retained. All 197,568 predicted depth samples were valid. No model-tuned confidence cutoff was used.

| Measurement | Actual observation |
| --- | --- |
| New-worker model construction/checkpoint load | 0.292 s; excludes Python/ML import time |
| First loaded-model forward | 0.425 s |
| Optional second warm forward, same model/images | 0.335 s; maximum depth difference 0 |
| Worker total before output compression | 5.593 s |
| First engine request wall time | 7.418 s |
| Repeated identical engine request | 0.498 s; cache hit, zero additional forwards |
| Peak worker working set | 655,941,632 bytes, about 625.6 MiB |
| Prediction NPZ | 1,499,012 bytes |
| Total generation artifacts | 1,538,244 bytes |

The profiled initial request intentionally performs two forwards to distinguish first/warm timing; normal provider default performs one. Cache hits still preflight immutable local identity and verify cached bytes but do not import the model or forward. These are individual observations, not p50/p95 or a latency SLA. Earlier successful single-forward and profiled requests remain locally recorded with their own implementation identities.

## Objective quality

One scale anchor uses the known 1.20 m synthetic opening width and predeclared endpoint annotations covering 90% of the opening-back width. Original inferred width is 0.2533698813 relative units; fixed scale is 4.7361588276. Original tensors remain unchanged. After fixing this scale, camera-centre rigid rotation/translation aligns gauge; no second scale fit or object deformation is allowed.

| Metric | Result |
| --- | --- |
| Scale-invariant log depth RMSE | 0.07156 |
| Anchored depth AbsRel | 0.55294, or 55.29% |
| Anchored depth RMSE | 1.47176 m |
| Rotation errors, views 1-4 | 3.03 / 2.60 / 3.49 / 6.84 degrees |
| Position errors after fixed-scale alignment | 93.72 / 31.53 / 31.06 / 96.76 cm |
| Six camera-baseline relative errors | 45.79-75.88% |
| Focal errors | 32.14-35.99%, 44.43-49.59 processed pixels |
| Global depth-boundary precision / recall / F1 | 0.496 / 0.800 / 0.613 |
| Opening-ROI depth-boundary precision / recall / F1 | 0.674 / 0.255 / 0.370 |

AbsRel is mean `abs(pred-truth)/truth`; RMSE is root mean squared metre depth error. Scale-invariant log RMSE subtracts squared mean log error from mean squared log error. Boundaries use a fixed 5 cm discontinuity threshold and one-pixel tolerance; opening ROI is truth labels 7/8 with a predeclared three-pixel margin. Original relative-unit AbsRel/RMSE versus metre truth are gauge-dependent diagnostics, not metric accuracy.

Observed world-oriented point spans use **evaluation-only truth labels**. They expose accumulated depth/camera error and cross-view scatter, not detected semantic object boxes or completed meshes. Original oriented relative spans and aligned metre spans are both retained. Incomplete truth-visible spans are lower bounds of true primitive size; inaccurate predicted spans are not guaranteed bounds.

| Entity | Absolute width / height / depth errors (cm) | Relative errors (%) |
| --- | --- | --- |
| Room | 133.68 / 100.76 / 32.87 | 27.85 / 37.32 / 9.13; depth incomplete |
| Opening-back | 0.76 / 7.39 / 32.38 | 0.63 / 8.21 / undefined for zero-thickness plane |
| Opening/recess | 29.57 / 14.81 / 3.57 | 24.64 / 16.46 / 10.20; recess depth incomplete |
| Hearth | 129.71 / 35.72 / 79.28 | 72.06 / 255.13 / 144.14 |
| Mantel | 96.00 / 13.89 / 62.11 | 50.53 / 115.74 / 258.78 |
| Tall left box | 104.49 / 33.64 / 101.22 | 217.69 / 33.64 / 241.00 |
| Foreground stool | 245.47 / 24.90 / 162.87 | 472.05 / 41.49 / 313.22 |
| Right seat box | 203.04 / 36.23 / 195.48 | 267.16 / 45.29 / 296.18 |

Opening width participates in calibration and must not be presented as independent geometric accuracy. Room depth/recess depth lack complete visibility; the report flags those limitations instead of crediting apparent accuracy. No fixture/material/camera adjustment followed observed DA3 quality.

## Authority and evidence

The real inferred opening width remains 1.2075551664 m; a separately supplied **synthetic field-measurement-role** fact is 1.20 m. Both remain present, `resolveFact(requireAuthoritative)` selects measured, visualization retains the approximate room, technical export still throws and `installationApproved` stays false. A controlled unit fixture also intentionally pairs 1.03 m inference with 1.20 m supplied authority. This exercises future Benson precedence without falsely claiming an actual field visit/manufacturer product or installation approval.

Local compact package: `.image-blaster/benchmark/da3-small-cpu-v1-profile/evidence.html`, including four synthetic RGB inputs, exact camera plan/metadata, GT/inferred/aligned depth images, fixed-camera cross-view point projections, numeric report, system/cache receipt and final SceneSpec. Raw inference files and failed/successful manifests remain under `.image-blaster/cache`. Generated bulk is ignored. Only deterministic source and a small [numeric summary](geometry-results/da3-small-cpu-v1.json) are committed.

## Verification and review

Starting baseline: `npm ci`, 51 engine/Windows tests, 13 viewer tests, typecheck, 35-module syntax check and production build passed. Final: **59 engine/Windows tests, 13 viewer tests, 15 worker tests, five fixture/evaluator tests**, typecheck, 40-module syntax check and production build passed. Fixture regeneration reproduced all four PNG hashes. Node dependencies did not change. Build still transforms 5,443 modules and emits the inherited large-chunk warning; the inherited development audit has 10 advisories. No configured stylistic lint exists; syntax checks are not represented as lint.

Independent review verified current implementation/checkpoint identity, cache integrity, inferred/authoritative boundaries, real receipts and visuals, and separately reran 59 Node/15 worker tests. It found and prompted fixes for helper-binding fingerprint collisions, worker protocol mismatches, model-world frame naming, original-pixel mapping, checkpoint shared aliases, input/prediction/truth hash binding and stale documentation. No implementation blocker remains. The reviewer did not independently repeat installation/full ML forward/app build/five evaluator tests; those have parent-run receipts.

Unverified: real photography, robust camera calibration, true semantic segmentation, hidden surfaces, fusion/mesh topology, materials, large inputs, GPU acceleration, client import/runtime and physical/customer/installer acceptance. Installed transitive versions are recorded rather than fully locked; embedding processes must restart after implementation edits. Hosted mutable checkpoints remain unknown.

**Next milestone:** a separately labeled known-intrinsics conditioning experiment on this unchanged fixture, compared with this unconditioned baseline. Keep truth conditioning explicit and report held-out dimensions. This tests whether calibration addresses the large focal/scale errors before adding real photos or client integration.
