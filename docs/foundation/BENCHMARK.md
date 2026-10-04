# Two-scene benchmark plan

Prepared October 3, 2026. No paid generation or client import has run. No customer/client image has been copied. Inputs below are specifications awaiting Drew's chosen synthetic or authorized photos. Fixture/unit tests do not count as a reconstruction benchmark.

## Input A: TPS founding-site reference

Supply one rights-cleared founding-site environment image for a plausible 1985 North American setting, with separable barn/shed, ticket-booth-sized structure, fence, roadside sign and vegetation. Include capture/source/era notes and independently known dimensions for at least three visible items. Optional overlapping second/third views make observed reconstruction comparable to hallucinated completion. Include a TPS-owned seed/map-recipe reference only as context, never a terrain upload or replacement.

Record scene/source SHA-256, rights, camera/EXIF availability, source type (real photograph versus generated concept), actual versus desired era, dimensions' physical datums and scale references. A generated concept is `generated`, never observed historical evidence. Fixed ground-truth boxes/segmentation annotations should identify each object instance and built-in surface before a provider runs.

## Input B: Benson room/fireplace reference

Supply one rights-cleared synthetic or explicitly authorized living-room/fireplace photo; preferably three overlapping views including front/opening, side return and floor/hearth. Supply field/source-backed dimensions separately: finished floor datum, opening width/height/depth, hearth projection/elevation and at least two room references. Include exact product/manual/configuration identity only through an approved source package; never upload restricted customer/manual imagery by default.

Create a measurement override fixture where visual opening width intentionally differs from recorded authoritative width. Retain both and assert the measured value governs the technical path. Use a registered exact appliance geometry fixture separately from inferred room mesh. Unknown fuel/manual rules, mantle clearance, floor protection/support and electrical conditions remain unknown; a photo cannot resolve them. Export internal visualization and attempt a technical export that must reject inferred geometry.

## Metrics locked before execution

| Metric | Recording method and acceptance target |
| --- | --- |
| Geometry usefulness | Blind reviewer 0-5: separable objects/surfaces, editable topology, sensible pivots, complete requested parts. Report missing/duplicated objects and invented backsides; target >=3 is exploratory, not client acceptance |
| Source-image similarity | Fixed camera render, identical crop/resolution/exposure; silhouette IoU per annotated object, image SSIM as a secondary statistic, blind 0-5 appearance score. SSIM is not dimensional truth |
| Scale accuracy | Metre bounds compared with independent references; absolute cm and relative error per object/room axis. Show inferred error and authoritative override result separately. Authoritative fixture export must retain exact supplied dimensions; installation tolerance is manufacturer/client-defined |
| Texture/material quality | 0-5 visual score, UV seams/missing textures, explicit base-color/roughness/metallic/normal channel existence and interpretation. Assess neutral relighting; distinguish baked illumination from PBR |
| Cleanup effort | Hands-on minutes with tasks logged (separation, remesh, holes, UV/material repair, pivot/scale), including failures. Exclude model download/setup time from marginal cleanup, report separately |
| Unreal import quality | Authorized isolated import: parsability, asset/material slots, scale/axes/winding, collision unapproved versus validated, triangle/texture budgets, screenshots and GPU/CPU profile. No pass until actually imported |
| Benson usefulness | Internal visual composition score; inferred vs authoritative facts retained; technical export refusal and measured override tests; exact appliance replacement never fitted to inferred bounds. No installation or customer approval inferred |
| Generation time | Submit-to-provider-complete, download, validation, cleanup and import times separately; cold model setup versus warm inference; p50/p95 only with enough runs, otherwise individual values |
| Marginal cost | Provider receipt by model/options/job; GPU instance hourly rate x billable allocation time, idle/setup/storage/network separately. Record estimated versus settled; do not substitute budget for invoice |
| Cache reuse | Repeat identical request: same key/artifact hashes, cache hit, **zero new billable submissions**. Change source/model/prompt/authority facts: cache miss; never create miss by changing irrelevant execution metadata |
| Failure rate | Attempt counts, success/failure categories, ambiguous submissions, resumable IDs and cleanup rejection. No auto retries; manual approval before a replacement billable run |

## Comparison protocol and budget

Run schema/procedural/cache tests first. Choose one legally eligible object candidate and one source/world candidate only after primary-license and hardware gates pass. Preserve current hosted Image Blaster adapters as commercial comparisons, not mandatory baselines. Freeze exact model/checkpoint, inputs, prompt, params, adapter version, axes and acceptance rubric before submitting.

For each scene, start with one object and one room/world job, then evaluate before expanding. Supply a per-endpoint cost estimate and explicit total ceiling to Drew for approval. No budget has been approved here. Hosted HQ exports, extra views, retextures, image edits and retries are separate billable jobs. GPU rental/boot is also paid and needs approval.

Keep all outputs under engine-owned ignored cache/benchmark storage. Results manifest should link original input hashes, authority fixtures, provider metadata, elapsed stages, receipt/cost, artifact hashes, QA and unverified metrics. Do not publish private source imagery or diagnostics. Actual TPS/Benson integration needs separate authorization and acceptance against the current client head.

Recommended first unpaid extension: multi-view depth/camera provider emitting inferred facts into SceneSpec, plus geometric validation against a synthetic room with independently supplied known dimensions. Use the AMD Windows desktop for Node/procedural work; do not assume CUDA models run on it.
