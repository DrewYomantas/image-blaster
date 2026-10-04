# Synthetic room geometry benchmark

This is project-authored synthetic geometry, not a customer room, product specification, photographic capture or installation reference. The fixture is frozen independently of DA3. It uses NumPy analytic ray/box intersections and deterministic CPU shading, with no model or rendering-package installation. NumPy 1.26.4 is already available through `py -3.13` on the current machine. SHA-256 image equality is tested within the installed environment; cross-platform float/library equivalence is not promised.

From the repository root:

```powershell
py -3.13 benchmarks/geometry/fixture.py
py -3.13 benchmarks/geometry/test_geometry.py
py -3.13 benchmarks/geometry/evaluate.py --prediction path/to/predictions.npz --out .image-blaster/benchmark/da3-report
```

All default generated files live under ignored `.image-blaster/benchmark`. Scripts and this documentation are source; no source PNGs, depth arrays or generated reports need committing.

## Frozen scene

The room is 4.8 m wide, 3.6 m deep and 2.7 m high. World axes are right handed, metres, X right, Y up, Z toward the rear wall. The fireplace front wall is Z=0. The opening spans X=-0.6 to 0.6, Y=0.5 to 1.4 and Z=-0.35 to 0: width 1.20 m, height 0.90 m and recess depth 0.35 m. These are synthetic datums, not clearance or construction rules.

| Primitive | Width / height / depth (m) | Z interval (m) |
| --- | --- | --- |
| Hearth | 1.80 / 0.14 / 0.55 | 0.00 to 0.55 |
| Mantel | 1.90 / 0.12 / 0.24 | 0.00 to 0.24 |
| Tall left box | 0.48 / 1.00 / 0.42 | 0.55 to 0.97 |
| Foreground stool box | 0.52 / 0.60 / 0.52 | 1.79 to 2.31 |
| Right seat box | 0.76 / 0.80 / 0.66 | 1.02 to 1.68 |

Four overlapping 640x480 views use `fx=fy=350`, `cx=319.5`, `cy=239.5`, zero skew. Positions are (-1.55,1.6,3.0), (-0.55,1.45,3.1), (0.65,1.65,3.0), (1.6,1.5,2.85); all look toward (0,1.15,0.35). `fixture.py` computes and writes exact world-to-camera matrices. Camera axes are CV X right, Y down, Z forward; pixel centres are integer coordinates. Depth is camera Z, not Euclidean range.

The renderer has fixed wood-like floor texture, neutral walls, coloured props, diffuse shading and analytic hard shadows. It is a deliberately simple room-photo surrogate. The camera/focal setup was selected for visible overlap and props before model inference; no fixture change may follow looking at model errors without a new version.

## Input and truth separation

`inputs/` contains only four RGB images and an image-hash/rights manifest. The worker receives only the four ordered image paths. It does not receive that manifest or any truth. The allowed 1.20 m opening-width anchor lives with evaluation truth and is applied by the evaluator, after inference.

`ground-truth/` contains `truth.npz`, exact camera matrices, primitive bounds, labels, evaluation-only anchor pixel annotations, and labelled synthetic input/depth PNGs. Ground-truth masks are evaluation aids; object identity/segmentation is not claimed as a provider output. The anchor samples two interior opening-back points, separated by 90% of opening width, to avoid depth interpolation across opening boundaries. Those annotations never enter provider arguments.

## Prediction contract

`predictions.npz` contains `depths[N,H,W]`, `confidences[N,H,W]`, `intrinsics[N,3,3]` and `extrinsics[N,3,4]` or `[N,4,4]`, in input order. N is four. Extrinsics are proper CV world-to-camera transforms and depth is positive camera-Z in original relative units. Confidence is raw, higher-is-better, and uncalibrated. Optional `processed_images[N,H,W,3]` is used to colour reprojection evidence. The evaluator rejects pickle/object arrays, malformed shapes, nonfinite cameras, negative focal length, nonrigid/reflected rotations and invalid anchor depth.

For DA3 resize-only processing, truth K is mapped with `x'=(x+0.5)*scaleX-0.5` and similarly for Y. The evaluator re-raycasts analytic truth at prediction resolution; it does not interpolate ground-truth depth at discontinuities. Mixed aspect/crop input is outside this fixture protocol. A crop or padding transform requires an explicit future contract change; do not silently run this resize-only evaluator on cropped outputs.

## Evaluation locked before inference

1. Retain the original NPZ and report original inferred anchor width and original oriented object extents in relative units.
2. Set one scale from the allowed 1.20 m opening-width anchor. Anchor width is calibration, not a held-out accuracy result.
3. Fit a proper rotation and translation from the four camera centres after fixing scale. No scale refit, all-point registration, object-based deformation or ground-truth-intrinsics substitution is applied to predictions. Collinear/coincident centre trajectories fail because rigid alignment is underdetermined.
4. Report original scale-invariant log RMSE and gauge-dependent numeric depth errors separately from aligned metre RMSE/AbsRel. A raw relative-unit RMSE is not physical metre accuracy.
5. Report aligned camera rotation/translation error, all six baseline errors, fx/fy pixel/percent error, depth discontinuity precision/recall/F1 (5 cm threshold, 1-pixel tolerance), and dimensions in absolute centimetres and percent. Report both global boundaries and a fixed fireplace/opening ROI: GT labels 7/8 with a three-pixel margin at prediction resolution. This ROI is declared before inference and never tuned to model output.

Dimension estimates are observed point bounds grouped with fixed evaluation-only semantic labels and transformed into the known world orientation. All dimensions except the calibrated opening width are held out. The visible room-depth span is incomplete because cameras face the fireplace; rear-wall depth cannot be recovered from these photos. Some prop backs/bottoms are occluded. Every axis reports a ground-truth visible extent and coverage fraction; `complete_axis_observed` requires 97-103% coverage. Incomplete extents are lower bounds and their error versus full primitive size is not proof of bad hidden-surface reconstruction. Boundary errors are depth discontinuities, not semantic instance segmentation scores.

Finite positive depths with finite nonnegative confidence are evaluated, with no model-tuned confidence cutoff. Invalid fraction is reported, and missing predictions cannot silently count as perfect depth. Invalid anchor depth prevents alignment. No pass/fail gate is invented for actual DA3 quality; report the measured errors and limitations.

## Evidence and checks

The evaluator writes numeric `report.json`, separately saved `aligned.npz`, four GT/INFERENCE/ALIGNED depth PNG sets and four ALIGNED cross-view point projections. Raster banners and filenames label synthetic GT, inference and aligned evidence. Depth colours use the same 0-6 numerical range; original inference units are uncalibrated, aligned and truth units are metres. Cross-view projections use the next source image reprojected into the target GT camera, exposing camera/depth consistency. Holes and lack of surfaces are retained, not filled.

The tests verify deterministic image hashes and truth separation, exact proper camera geometry, full truth-ray coverage, visible semantic instances, known scale plus arbitrary gauge recovery, distortion remaining after single-anchor alignment, absent-boundary failure and invalid-rotation rejection. Oracle arrays are test-only, never reconstruction-provider evidence. Passing these checks verifies fixture/evaluator mechanics. It does not verify DA3 quality, a real room, topology, materials, imports or installation use.


## Frozen camera-conditioning follow-up

The original fixture and Lane A receipt are immutable. Regenerate only into a separate hash-check directory:

```powershell
& workers/da3/.venv/Scripts/python.exe benchmarks/geometry/fixture.py --out .image-blaster/benchmark/camera-conditioning-v1/frozen-fixture
& workers/da3/.venv/Scripts/python.exe benchmarks/geometry/conditioning.py --truth .image-blaster/benchmark/synthetic-room-v1/ground-truth/geometry.json --out docs/foundation/geometry-results/camera-conditioning-v1
node benchmarks/geometry/camera-run.mjs
```

Runner uses canonical existing truth and original local A NPZ identified by the committed receipt; retain those ignored artifacts for replay. It verifies four regenerated hashes, original A numbers/hash, predeclared conditioning hashes and all cache hits. Explicit worker receipts distinguish native supplied-camera scale from the original fixed-anchor evaluation. Supplemental v2 adds per-entity scatter and correct planarity/unit labeling; initial v1 inference receipts and NPZs remain unchanged. Do not apply original evaluator to conditioned tensors without their worker receipt.

Depth/labels/dimensions/anchors never enter the provider. B is a negative control; C is an oracle upper bound; D is the frozen perturbation set in conditioning-declaration.json. See ../../docs/foundation/CAMERA-CONDITIONING-BENCHMARK.md for numerical tables, scientific figures, runtime, cache, licensing and Outcome2. The visible-span/per-entity supplements preserve old core rules and never fill geometry.

## MoGe-2 frozen comparison

`moge2-vits-normal` is a separate optional local/free monocular provider. Read `docs/foundation/MOGE2-WORKER.md` for reviewed exact source/checkpoint pins and isolated CPU setup. Node tests need no Python or weights. Existing DA3 receipts and arrays remain immutable.

Regenerate into `.image-blaster/benchmark/moge2-vits-normal-v1/frozen-fixture` and compare all four hashes before inference. The pre-inference plane/normal declaration lives under `docs/foundation/geometry-results/moge2-vits-normal-v1/evaluation-declaration.json`; its recorded hash binds the request package. Run `node benchmarks/geometry/moge-run.mjs` only after the separate worker environment is installed and that declaration exists.

M1 receives RGB only. M2 receives RGB plus an explicit experiment-oracle horizontal-FOV structure derived from original K. Neither receives truth extrinsics, depth, labels, dimensions or the opening anchor. Oracle E transforms official metric camera point maps after inference in `evaluate_moge.py`. Native metric output is primary; the optional `ANCHOR-ALIGNED` derivative applies the original one-opening-width anchor about each fixed oracle camera origin. No camera, plane, view or object fitting occurs.

Plane membership and normal eligibility use the frozen analytic primitive face with a fixed one-pixel GT neighbourhood exclusion. Signed known-plane RMS/scatter and signed outward normal angles use every eligible valid model sample. No output-driven rejection is allowed. The original depth/boundary/bounds/correspondence helpers are reused. MoGe returns640x480 while DA3 returns252x196: one-pixel boundary tolerance and sample-weighted statistics have different physical sampling, so they are not a strict equal-grid model comparison. Visibility and missingness remain explicit.

Full local tensors and visual evidence remain ignored; the committed numeric receipt and `MOGE2-BENCHMARK.md` document the decision. No mesh, client import, camera solver, authoritative geometry or installation claim follows from this synthetic upper bound.
