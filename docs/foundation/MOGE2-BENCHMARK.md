# MoGe-2 ViT-S Normal frozen geometry comparison

**Outcome 3: no meaningful improvement in usable held-out metric geometry over DA3 with exact cameras.** MoGe gives sharper boundaries and useful signed normals, but the independently predicted point maps remain dimensionally poor and inconsistent even with perfect evaluation poses. Building a practical camera solver around this Small tier is not justified.

Run October 4, 2026 from `55ce11da49bf1c8e7a8f2d12e14d9657de697ed3` on `codex/provider-neutral-foundation`. Origin remote matched that HEAD and the tree was clean. Node 24.13.1, existing DA3 Python 3.12.12. Baseline passed 61 Node/Windows, 13 viewer, 25 DA3 worker and 11 geometry tests, typecheck, 43-module syntax and production build. Four images regenerated separately with exact frozen hashes; no scene, cameras, textures, light, dimensions or old evaluation rules changed. Canonical original truth and every original DA3 receipt/NPZ remain unchanged. No main/client edits, paid service, GPU, SfM or mesh.

## Exact model and output semantics

Official [MoGe source](https://github.com/microsoft/MoGe/tree/74fbce054ebed49800de42d0ad0e83495065719a), revision `74fbce054ebed49800de42d0ad0e83495065719a`, MIT. Exact [Small Normal checkpoint card](https://huggingface.co/Ruicheng/moge-2-vits-normal/blob/26b477f41595707c5db6770294c0d1721e8ed4ed/README.md), revision `26b477f41595707c5db6770294c0d1721e8ed4ed`, MIT. `model.pt`: 140,550,416 bytes, SHA-256 `79a16621928c2bf0ed04659218c55c01075e950507f40bb3332fb4c873d3e1dc`. Actual strict load: 35,103,656 parameters. DINOv2 bundled components Apache-2.0; utils3d_moge helper MIT at`62f09d58509485564e24d5d9f6aac9ee9ebc0c37`. Reviewed before installation, downloaded bytes/card and source Git blobs verified.

Isolated Python 3.12.12 CPU float32: torch 2.7.1+cpu,numpy 2.2.6,scipy 1.15.3,opencv-python-headless 4.11.0.86,pillow 11.2.1,huggingface-hub 0.36.0. Full installed transitives and runtime/source/checkpoint fingerprints appear in the compact receipt. The optional provider does not load/download Python/model at Node startup. The upstream loader's warning-only strict=False is followed by exact key equality and strict=True reload. No random missing parameter is accepted.

Each image is an independent monocular forward, resolutionLevel9 (official 3600 tokens), six threads, use_fp16=False, force_projection=True, apply_mask=True. Official inferred points are projected from predicted depth/K in upstream postprocessing; they are not the raw affine point head. Native depth is camera Z, metric scale is predicted scale_head.exp(), masks use the official fixed>0.5 threshold, normals are signed camera-facing unit vectors. Original640x480 points/depth/masks/normals/normalized and pixel K/RGB are retained losslessly in NPZ, never JSON arrays. MoGe-3 dependencies are excluded.

M1 receives RGB only. M2 receives exact horizontal FOV 84.87245957707 degrees for each image in separate `experiment-oracle` conditioning. FOV affects focal/shift recovery after neural forward, not the backbone. M2's K is supplied calibration, not independently predicted camera accuracy. Both use centered principal-point assumption. Official normalized K converts to integer-centred pixels by fx*=640,fy*=480,cx=640*.5-.5=319.5,cy=480*.5-.5=239.5. Actual M2 fx/fy349.9999695. Regression tests compare official conversion and unprojection. No crop/pad or input-size transform is invented.

## Frozen evaluation

The [pre-inference declaration](geometry-results/moge2-vits-normal-v1/evaluation-declaration.json), SHA-256`c8105b66ecdbaba56e0d9c160c6ad1ac83ff8e7b0f63d8325f7b6979891f094b`, fixes analytic truth faces, same primitive/face throughout 3x3 neighbourhood and rim exclusion. Signed perpendicular known-plane residual mean/RMS/std uses every eligible valid point, with no fitted plane or rejection. Signed normals use mean/median angular error and fixed<=15deg fraction; no absolute cosine. Original depth,5cm boundary/1px tolerance, truth-label opening ROI+3px margin, visible-span97–103% full-axis rule and untrimmed bounds/cross-view eligibility helpers are reused.

Camera points transform to world only in evaluation via `(p_camera - t) @ R` using exact original oracle E. No extrinsics reach the worker, no camera reconstruction claim, rigid fit, ICP, per-view/per-object scale, deformation, semantic input, object dimension or hidden completion. Optional ANCHOR-ALIGNED uses the original1.20m opening width once globally, scaling each camera-frame point map about its fixed oracle camera centre before world transform. Native originals remain unchanged. Opening width is conservatively excluded from all independent claims; opening-back thickness is a planarity diagnostic.

MoGe output640x480 versus DA3252x196 means different sampling populations, physically different1px boundary tolerance/ROI margin and pixel reprojection units. These are native-grid comparisons, not equal-resolution architecture claims. Metric depth/cm scatter use the same formulas; dimensions use each grid's truth-visible spans. DA3 plane supplements read preserved NPZ and original metric alignments, with no new model inference. NumPy runtime also differs 1.26.4 versus 2.2.6; canonical truth bytes remain fixed.

## Primary comparison

| Lane / scale | SILog | AbsRel | RMSE m | Opening P / R / F1 | Scatter cm | Reprojection px |
| --- | --- | --- | --- | --- | --- | --- |
| DA3 A ALIGNED | 0.07156 | 0.5529 | 1.472 | 0.674 / 0.255 / 0.370 | 20.49 | 24.88 |
| DA3 C NATIVE | 0.07023 | 0.0513 | 0.181 | 0.765 / 0.175 / 0.285 | 19.60 | 5.66 |
| M1 NATIVE | 0.09980 | 0.2832 | 0.831 | 0.981 / 0.474 / 0.639 | 63.90 | 55.55 |
| M2 NATIVE | 0.08290 | 0.1284 | 0.426 | 0.981 / 0.474 / 0.639 | 40.33 | 34.75 |

All 1,228,800 MoGe samples valid, zero invalid/masked fraction in all four views. All 901,158 eligible cross-view correspondences valid and projectable; zero prediction-quality rejection. Full per-view depths and global/opening precision/recall/F1 are in the numeric receipt. Pixel reprojection normalized by native width is also useful: DA3 C5.66/252=2.25%, MoGe M25.43%. This is a grid-size disclosure, not an alignment.

## Held-out dimensions

All errors below are absolute centimetres; W/H/D ordering. Untrimmed observed bounds include edge tails. Room/recess depth target visible span only. Calibration width shown as excluded. No ugly axis omitted.

| Entity | DA3 A W/H/D error | DA3 C W/H/D error | M1 W/H/D error | M2 W/H/D error |
| --- | --- | --- | --- | --- |
| room | 133.7 / 100.8 / 77.4 | 30.1 / 39.1 / 22.1 | 154.5 / 58.7 / 91.0 | 146.0 / 68.5 / 64.1 |
| opening-back | excluded / 7.4 / 32.4 | excluded / 11.9 / 23.0 | excluded / 9.3 / 97.0 | excluded / 11.7 / 62.3 |
| opening-recess | excluded / 14.8 / 13.6 | excluded / 5.1 / 1.0 | excluded / 18.6 / 75.2 | excluded / 21.5 / 41.5 |
| hearth | 129.7 / 35.7 / 79.3 | 42.7 / 26.0 / 24.1 | 155.9 / 86.8 / 181.3 | 127.2 / 89.6 / 147.7 |
| mantel | 96.0 / 13.9 / 62.1 | 5.9 / 2.9 / 31.3 | 149.2 / 7.7 / 99.9 | 120.5 / 9.2 / 69.0 |
| tall-left-box | 104.5 / 33.6 / 101.2 | 86.2 / 19.1 / 70.7 | 139.6 / 33.5 / 139.8 | 111.4 / 28.1 / 113.4 |
| foreground-stool | 245.5 / 24.9 / 162.9 | 201.4 / 18.9 / 106.9 | 287.6 / 24.1 / 158.6 | 255.7 / 28.3 / 123.7 |
| right-seat-box | 203.0 / 36.2 / 195.5 | 138.6 / 26.3 / 109.4 | 218.8 / 47.3 / 194.8 | 191.1 / 56.5 / 158.9 |

MoGe native extents/targets/relative errors/GT coverage and claim type:

| Entity / axis | Target m | M1 predicted m / error cm / % | M2 predicted m / error cm / % | GT coverage | Claim |
| --- | --- | --- | --- | --- | --- |
| room / width | 4.800 | 6.345 / 154.5 / 32.2 | 6.260 / 146.0 / 30.4 | 100.0% | full axis |
| room / height | 2.700 | 3.287 / 58.7 / 21.8 | 3.385 / 68.5 / 25.4 | 100.0% | full axis |
| room / depth | 2.501 | 3.411 / 91.0 / 36.4 | 3.143 / 64.1 / 25.6 | 69.5% | visible span only |
| opening-back / height | 0.900 | 0.993 / 9.3 / 10.4 | 1.017 / 11.7 / 13.0 | 99.3% | full axis |
| opening-back / depth | 0.000 | 0.970 / 97.0 / n/a | 0.623 / 62.3 / n/a | n/a | planarity diagnostic |
| opening-recess / height | 0.900 | 1.086 / 18.6 / 20.7 | 1.115 / 21.5 / 23.9 | 99.3% | full axis |
| opening-recess / depth | 0.250 | 1.002 / 75.2 / 300.8 | 0.665 / 41.5 / 166.1 | 71.4% | visible span only |
| hearth / width | 1.800 | 3.359 / 155.9 / 86.6 | 3.072 / 127.2 / 70.7 | 100.0% | full axis |
| hearth / height | 0.140 | 1.008 / 86.8 / 619.9 | 1.036 / 89.6 / 640.3 | 100.0% | full axis |
| hearth / depth | 0.550 | 2.363 / 181.3 / 329.6 | 2.027 / 147.7 / 268.5 | 100.0% | full axis |
| mantel / width | 1.900 | 3.392 / 149.2 / 78.5 | 3.105 / 120.5 / 63.4 | 100.0% | full axis |
| mantel / height | 0.120 | 0.197 / 7.7 / 64.4 | 0.212 / 9.2 / 76.4 | 100.0% | full axis |
| mantel / depth | 0.240 | 1.239 / 99.9 / 416.1 | 0.930 / 69.0 / 287.5 | 99.9% | full axis |
| tall-left-box / width | 0.480 | 1.876 / 139.6 / 290.8 | 1.594 / 111.4 / 232.0 | 100.0% | full axis |
| tall-left-box / height | 1.000 | 1.335 / 33.5 / 33.5 | 1.281 / 28.1 / 28.1 | 100.0% | full axis |
| tall-left-box / depth | 0.420 | 1.818 / 139.8 / 332.9 | 1.554 / 113.4 / 270.1 | 100.0% | full axis |
| foreground-stool / width | 0.520 | 3.396 / 287.6 / 553.1 | 3.077 / 255.7 / 491.7 | 100.0% | full axis |
| foreground-stool / height | 0.600 | 0.841 / 24.1 / 40.2 | 0.883 / 28.3 / 47.2 | 99.9% | full axis |
| foreground-stool / depth | 0.520 | 2.106 / 158.6 / 305.1 | 1.757 / 123.7 / 237.9 | 100.0% | full axis |
| right-seat-box / width | 0.760 | 2.948 / 218.8 / 287.9 | 2.671 / 191.1 / 251.4 | 100.0% | full axis |
| right-seat-box / height | 0.800 | 1.273 / 47.3 / 59.1 | 1.365 / 56.5 / 70.6 | 100.0% | full axis |
| right-seat-box / depth | 0.660 | 2.608 / 194.8 / 295.2 | 2.249 / 158.9 / 240.7 | 100.0% | full axis |

## Planes and normals

Known-plane RMS / residual scatter std, centimetres. Residual RMS includes bias against the known plane; scatter is deviation around mean residual, never a fitted plane.

| Surface | DA3 A RMS/std | DA3 C RMS/std | M1 RMS/std | M2 RMS/std | M2 anchor RMS/std |
| --- | --- | --- | --- | --- | --- |
| front-wall | 139.3/11.3 | 12.1/11.3 | 85.7/36.7 | 47.1/24.4 | 20.7/20.7 |
| floor | 29.3/6.2 | 8.1/4.6 | 25.3/12.1 | 21.7/10.3 | 9.5/9.4 |
| side-walls | 46.9/8.7 | 14.9/10.1 | 55.4/26.0 | 52.5/21.5 | 18.2/18.2 |
| opening-back | 97.8/5.3 | 50.3/4.2 | 69.6/31.3 | 30.1/18.1 | 23.8/15.3 |
| opening-recess | 94.1/26.2 | 48.8/9.0 | 65.5/32.9 | 28.7/17.8 | 22.5/15.2 |
| hearth | 61.9/49.9 | 9.3/6.2 | 48.9/33.3 | 26.8/15.6 | 12.3/12.2 |
| mantel | 140.7/12.6 | 11.0/6.7 | 97.7/40.4 | 58.9/28.3 | 27.3/23.8 |

MoGe signed normal mean / median angular degrees / fraction<=15deg. Normals do not depend on the post-forward FOV option, so M1 and M2 normals are identical. DA3 supplies no normals; no comparative normal accuracy is fabricated.

| Surface | Mean deg | Median deg | <=15deg | Eligible / valid |
| --- | --- | --- | --- | --- |
| front-wall | 2.23 | 1.60 | 99.2% | 502442 / 502442 |
| floor | 4.50 | 4.14 | 99.6% | 135932 / 135932 |
| side-walls | 3.44 | 3.55 | 99.8% | 133838 / 133838 |
| opening-back | 6.83 | 3.71 | 93.6% | 34360 / 34360 |
| opening-recess | 10.98 | 4.21 | 84.5% | 39582 / 39582 |
| hearth | 3.01 | 2.84 | 99.9% | 29330 / 29330 |
| mantel | 4.84 | 3.81 | 97.0% | 10828 / 10828 |

All declared surface groups have100% prediction coverage. Individual visible major-face plane/normal statistics are retained in the full ignored reports; group summaries are sample-weighted. Accurate normal directions do not establish accurate surface position or internally consistent depth geometry.

## FOV and scale

M1 predicted horizontal FOV by view: 76.56deg / 72.68deg / 77.40deg / 76.03deg; absolute FOV errors 8.31deg / 12.19deg / 7.47deg / 8.84deg. Pixel fx/fy and errors are in the receipt. M2 FOV was supplied and is not scored as independent camera accuracy.

| Lane | Native RMSE / AbsRel | Fixed anchor multiplier | Anchor RMSE / AbsRel | Anchor scatter cm |
| --- | --- | --- | --- | --- |
| M1 | 0.831 / 0.2832 | 0.945691 | 0.647 / 0.2149 | 54.67 |
| M2 | 0.426 / 0.1284 | 0.882174 | 0.193 / 0.0674 | 29.28 |

The anchor improves global depth errors by correcting native scale bias, but does not cure local shape. M2 anchor still has182.3cm stool-width error,100.0cm stool-depth error,76.3cm hearth-height error and121.3cm hearth-depth error. M1/M2 native metric scale comes only from the learned model head; ANCHOR-ALIGNED comes from both model scale and benchmark anchor. DA3 A gets metric scale from anchor; DA3 C native gets scale from supplied-camera alignment. No anchor touches native arrays.

## Cross-view detail

| Entity | DA3 A scatter cm | DA3 C scatter cm | M1 scatter cm | M2 scatter cm |
| --- | --- | --- | --- | --- |
| room | 20.39 | 17.96 | 64.11 | 39.87 |
| opening-back | 13.15 | 26.69 | 59.92 | 29.79 |
| opening-recess | 13.10 | 26.55 | 59.47 | 29.82 |
| hearth | 20.32 | 14.30 | 67.84 | 40.37 |
| mantel | 24.36 | 15.76 | 77.45 | 50.49 |
| tall-left-box | 17.82 | 26.72 | 51.08 | 27.04 |
| foreground-stool | 34.78 | 29.90 | 72.12 | 41.41 |
| right-seat-box | 16.41 | 20.91 | 64.41 | 52.95 |

Full fixed-pair relative view offsets and residual scatter remain in the receipt, without refitting. Oracle poses expose independent monocular scale/depth disagreements. No prediction-dependent correspondence filtering was added.

## CPU observations

| Lane | Model load s | Preprocess s | Inference s | Worker total s | Engine wall s | Peak MiB |
| --- | --- | --- | --- | --- | --- | --- |
| DA3 A | 0.292 | 0.031 | 0.425 | 5.593 | 7.418 | 625.6 |
| DA3 C | 0.338 | 0.083 | 0.401 | 7.106 | 9.590 | 646.2 |
| M1 | 0.821 | 0.308 | 29.571 | 40.294 | 43.941 | 1899.2 |
| M2 | 0.837 | 0.042 | 40.067 | 53.219 | 56.535 | 1825.2 |

MoGe per-image infer calls include official neural forward plus focal/shift recovery and geometric postprocessing: M1 7.674s / 6.700s / 7.070s / 8.127s; M2 7.980s / 7.772s / 8.551s / 15.764s. Each MoGe lane loads once and performs four independent forwards. DA3 historical lanes perform multi-view forward plus one optional warm forward. Worker totals have different boundaries: MoGe includes NPZ compression/write and second identity recheck; old DA3 totals exclude final output writes. M1 overlapped final model-free evaluator tests; M2 image4 was slower than others. These are single observations, not isolated benchmark latency, repeated percentiles or conditioning-overhead claims.

## Cache and authority

Both real first lanes uncached; immediate repeat and final metadata-changed repeat hit, zero new forwards. Distinct keys:

- M1: `d0ac6d3587994eb0c47d6d27c3e4b7fab16c9414bb9c0d8a4293e02e48d7df23`.
- M2: `63fab747683dec2ed14c94ff3dd0fc92aa308738f25c6cafcb6ca5f519ada3c2`.

Final M1/M2 cache replay 1.22/1.13s; preserved DA3 C also hit its exact old key/implementation/checkpoint. Model-free Node regressions execute checkpoint revision, worker-file, resolution and FOV mutations as distinct mock generation keys; separate dry effective-key derivatives are recorded. Evaluation-only metadata hits. No live checkpoint or worker was changed to manufacture a miss. Artifact/input hashes and runtime identity rechecked before completion.

SceneSpec v1 remains sufficient: source hashes, exact provider identity, inference settings, learned metric scale and camera-frame conventions live on visual-only artifacts. Native projections are inferred; M2 projections/calibration are user-specified oracle evidence from user-input source. No pose, field-measurement/manufacturer/registered source is invented. Synthetic measured override retains 1.02m inference and resolves separate 1.20m authority; Benson technical export stays blocked, installationApproved false. TPS authored geography and Benson field measurements stay authoritative.

## Verification, review and rights limits

Final 67 Node/Windows, 13 viewer, 25 DA3 worker, 14 MoGe worker and 22 combined evaluator tests passed; 46 JS modules syntax-checked, Python compile, typecheck and production build passed. Exact-source mechanics tests run against installed pinned helper/model code. Ordinary Node tests load no model. Four frozen image hashes, original DA3 receipt/array hashes and eight real MoGe forwards verified. Automated viewer tests do not establish interactive UX; the viewer was unchanged. Existing Vite large-chunk warning remains.

Independent review inspected official rights/source/coordinate/scale/FOV/cache/evaluator contracts, exact checkpoint card, actual results and all three visual comparisons. It found a P2 NPZ/receipt mode-provenance gap, repaired before final evaluation/verification: mode/source order/FOV/coordinate/mask contracts now fail closed. Completion findings are recorded separately in `review-v1.json`.

Actual installed native audit: OpenCV headless still contains FFmpeg DLL and LGPL-2.1 notices; NumPy/SciPy OpenBLAS/LAPACK and GCC runtime exception notices, Torch/OpenMP notices, Pillow bundled codecs/fonts, Certifi MPL-2.0 and upstream MIT/Apache notices must be retained/reviewed for distribution. This local test is not blanket redistribution clearance or an installer license audit. [Worker setup/licensing details](MOGE2-WORKER.md). Weights, virtual environments, NPZs, runtime caches and generated bulk remain ignored.

## Visual evidence and next milestone

Ignored compact package: `.image-blaster/benchmark/moge2-vits-normal-v1/visual-package/evidence.html`, about 2.4MB including unchanged four RGBs/hash receipt, GT/DA3/MoGe depth panels, normals, signed plane residuals and cross-view projections. Full raw inference/evaluation artifacts remain separately ignored. Display thumbnails use nearest sampling only; numerical scoring uses original arrays. Parent and independent reviewer inspected the actual images. No synthetic output is presented as a customer deliverable.

Sharper fireplace edges and 2–11 degree mean surface normals are useful hints. They do not offset metre-scale prop/hearth bounds, native wall bias and doubled cross-view scatter versus DA3 C. Outcome3 applies to usable metric surface reconstruction, not a claim that every MoGe channel is worse.

**Next milestone:** predeclare a structural room-plane fitting prototype on these same frozen images, using photos and neural normals/depth as hints and the existing single explicit opening-width scale constraint. Oracle planes, labels, poses and object dimensions must remain evaluation-only. Compare held-out planes, room/fireplace geometry and props against these untouched receipts. Keep manual/structural assumptions explicit. Gate any SfM/fusion investment on those held-out results. Do not blindly add another lightweight model or integrate clients.

Unverified: real-photo generalization, independently solved poses, hidden completion, semantic extraction, fusion/topology/materials, site-fabric geometry, client imports and installation acceptance. This simple synthetic box room cannot establish Benson room or TPS site acceptance. No inference becomes physical authority.
