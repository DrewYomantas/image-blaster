# Engine and SceneSpec foundation

## Scope and architecture

The core is a normal Node >=24.2 ESM library/CLI with one runtime schema dependency, Ajv. Codex is an operator. Startup, local operations and tests do not require a commercial model or credentials. The upstream React/Three.js/Rapier/Spark viewer and indexed project format are retained.

```
operator / CLI
  -> validated effective request + provider registry
  -> SHA-256 cache + immutable request/manifest + single-request lock
  -> spend policy for metered jobs
  -> provider implementation
  -> local artifacts + checksums + retained evidence
  -> internal TPS/Benson manifest adapters
```

`engine/providers.mjs` defines seven capability boundaries: SceneAnalysis (`scene-analysis`), ImageEdit (`image-edit`), Object3D (`object-3d`), WorldReconstruction (`world-reconstruction`), Material (`material`), Audio (`audio`) and Decision (`decision`). Interfaces are runtime provider objects rather than a parallel TypeScript implementation. All must supply `id`, `capability`, `mode`, `billing`, `model`, `version`, `defaults`, `extra`, `license` and `generate(request)`. `generate` returns `{result: <finite JSON>, files: <local absolute paths[]>}`. Plugins are trusted code, not a security sandbox.

| Registered implementation | Mode/billing | Current behavior |
| --- | --- | --- |
| `local-evidence` | local/free | Hashes and stages explicit source images; validates/preserves supplied SceneSpec. No semantic image understanding or reconstruction |
| `procedural-box` | local/free | Deterministic eight-vertex/twelve-triangle OBJ at explicit metre dimensions; visual-only, no textures or collision approval |
| `fal-hunyuan` | paid-api/metered | Existing Hunyuan v3 FAL object implementation |
| `fal-meshy` | paid-api/metered | Existing Meshy v6 FAL object implementation |
| `fal-nano-banana` | paid-api/metered | Existing Nano Banana 2 FAL image edit |
| `fal-gpt-image` | paid-api/metered | Existing GPT Image 2 FAL image edit |
| `world-labs` | paid-api/metered | Existing Marble 1.1 world generation/download; output directory now explicitly routable into the cache |
| `fal-elevenlabs` | paid-api/metered | Existing ElevenLabs SFX v2 with optional ffmpeg/ffprobe postprocessing |

Material and Decision boundaries have no implementation. No open model or actual remote self-hosted worker is installed. A custom self-hosted provider may declare `billing: "free"` for owned hardware; a metered cloud worker must use the spend policy. This is a trusted registration decision, not automatic cloud discovery. `auto` selects local, then configured self-hosted. It never implicitly falls back to paid APIs; explicit provider selection still requires its billing guard.

Hosted adapter versions identify the retained upstream implementation, not an immutable vendor checkpoint. Endpoint aliases can change server-side. Record a returned checkpoint revision when available; otherwise it stays unknown and invalidate adapter version before a benchmark when behavior changes. There is no claim of hosted model reproducibility.

## SceneSpec v1

Canonical schema: `engine/scene.schema.json`. Runtime/semantic validation: `engine/scene.mjs`. Store canonical scenes as `scene-spec.json`; upstream `worlds/<slug>/scene.json` remains editor placement state and is not migrated automatically.

Root fields are `schemaVersion`, `id`, `coordinates`, `sources`, `cameras`, `surfaces`, `objects`, `relationships`, `materials`, `artifacts`, `validations`. Frame is metres, right-handed, Y-up, intrinsic XYZ rotation values in radians. Adapter manifests preserve this frame and require client-specific conversion; no coordinate conversion/import is implemented yet. Pivots/camera intrinsics/era/context/physical datums can be explicit evidence in entity `properties` rather than implicit undocumented guesses.

Each important fact has an identity, typed JSON value, provenance state, source IDs and confidence. Entity dimensions hold **arrays of evidence per width/height/depth axis**, not a single replaceable scalar:

```json
{
  "width": [
    {"id":"width-visual","value":1.1,"state":"inferred","sourceIds":["photo-1"],"confidence":0.5},
    {"id":"width-field","value":0.9,"state":"measured","sourceIds":["field-1"],"confidence":1}
  ]
}
```

These numbers illustrate the contract, not a real customer measurement. States are observed, inferred, generated, measured, manufacturer-specified and user-specified. `resolveFact` prefers measured/manufacturer evidence and preserves originals; conflicting authoritative values require review. Exact equal authoritative JSON values compare independently of property insertion order. Without authority, user intent precedes observation, then inference, then generated proposals; confidence breaks ties only within those non-authoritative ranks. This is an evidence-selection policy, not installation evaluation. Client product configuration, physical datum, source revision and applicability rules still govern whether two measurements are actually comparable. Record those bindings in evidence and do not merge incompatible facts into the same slot.

Measured facts require field-measurement sources; manufacturer facts require manufacturer-document sources. Generated facts require model sources. Model sources cannot support authoritative facts or geometry. User-specified is deliberately not authoritative. Image and registered-geometry sources require SHA-256. Authoritative artifact bytes must match registered geometry provenance; an authoritative entity must bind to that same registered source/hash. A generated mesh cannot become registered geometry by attaching a measured dimension. These checks validate declarations and identity, not the truth/authenticity of a supplied field record or manufacturer document. Authoritative intake remains a trusted client/operator function.

Sources carry license ID, commercial-use status and attribution; unknown remains `UNKNOWN`/`unknown`. Artifacts carry local URI, format, hash, visual-only/authoritative role, sources, provider/model/version, parameters, license and optional manifest URI. Facts/transforms/material descriptions retain confidence/source links. Relationships and validation results reference existing scene identities. Materials can record channel descriptions and texture artifact references. Technical QA results are distinct from contract validation.

Providers receive cloned inputs/scene/parameters, separate from the immutable effective request. Scene-analysis results must preserve all original source/artifact/relationship/validation records and original facts in the same entity/axis/property slot. They may append inferred/generated evidence; they cannot mint measured/manufacturer facts, authoritative sources/geometry or registered geometry approvals. `local-evidence` preserves previously trusted operator evidence rather than producing it.

## Commands and examples

```powershell
npm run engine -- providers
npm run engine -- analyze --image input/room.png --scene-id room --out scene-spec.json
npm run engine -- analyze --image input/front.png --image input/side.png --scene-id room --evidence scene-spec.json --out scene-spec-next.json
npm run engine -- generate --request examples/box-request.json
npm run engine -- validate --scene scene-spec.json
npm run engine -- export --scene scene-spec.json --target tps --out tps-import.json
npm run engine -- export --scene scene-spec.json --target benson --out benson-import.json
```

Supply the local images first. `--cache-dir` can choose engine-owned storage. Do not place generated files in client repos. Inferred photo analysis can be entered by a model/operator in a validated SceneSpec with honest sources; it is not bundled into the local provider. Registered asset attachment to an existing SceneSpec is still explicit operator/library work, not a CLI replacement tool. There is no actual room mesh inference, optimization, Unreal importer or Benson importer yet.

Request documents use `{ "request": { capability, providerId, mode, inputs, parameters, prompt, scene? }, "spend": { ... } }`. Inputs are local `{path, mediaType?, role?, license?}`. Remote URLs/data URIs must be staged first so the engine hashes actual bytes. Effective requests include hashes, media/extension/role/license, selected provider/model/adapter version, normalized typed defaults, prompt and complete relevant SceneSpec. Unknown/incorrect parameter types fail before submission. Explicit defaults share cache identity; Hunyuan's unused polygon option is removed unless LowPoly is selected. World requests require an explicit nonempty caption, avoiding upstream implicit workspace caption fallback. No prompt or provider is guessed from an AI conversation.

## Content-addressed manifests and retry policy

`.image-blaster/cache/<sha256>/` contains `inputs/`, `artifacts/` and `manifest.json`; sibling `<sha256>.lock/` prevents overlapping identical jobs. The manifest records effective request, input bytes/hash, provider/license, spend approval/estimate, lifecycle timestamps, local artifact checksums/lengths, result and result hash. Persisted manifest writes use temporary file replacement. Identical complete requests reuse artifacts without a new submission or fresh spend approval; cache hits recheck source/artifact bytes and result/request hashes.

Generation snapshots source bytes before dispatch and rejects source mutation. Paths are checked by resolved real paths; nonempty unmanifested entries fail before provider writes. Cache entry/output junction escape and corrupted results fail closed. This prevents ordinary accidental path escape; it is not protection against malicious plugins changing files concurrently or privileged adversaries. Locks are local-filesystem locks, not a distributed queue.

Failed/running generations and stale locks are not silently retried. Existing upstream sidecars preserve request/operation IDs and provide their legacy resume mechanism. **Generic engine resume/reconciliation is not yet implemented.** Inspect the manifest and hidden legacy metadata, reconcile any provider job remotely when separately authorized, and repair/download an existing result before planning another submission. Never delete a failed entry just to provoke a replacement paid job. A crash between vendor acceptance and persisted job ID remains an ambiguous submission requiring account reconciliation; this foundation cannot guarantee vendor-side exactly-once semantics.

Cached source/result absolute paths make an entry local to its current cache root. Moving a cache tree requires an explicit relocation/manifest migration, not assumed portability. Cache/source storage may contain private images and metadata; it is ignored by Git, not encrypted or automatically privacy-scrubbed. Do not commit generated customer material.

## Spend guard

Default is deny. No paid job was authorized or executed in foundation development. A future separately approved paid request needs:

```json
{
  "allowPaid": true,
  "endpoints": ["fal-ai/hunyuan3d-v3/image-to-3d"],
  "estimatedCostUSD": 0.5,
  "maxCostUSD": 0.5,
  "approvalReference": "replace-with-actual-user-approval-record"
}
```

The example estimate is illustrative, not current provider pricing or authorization. The selected model must be on the explicitly approved endpoint list. Each billable submission reserves the quoted amount from the in-process ceiling. Multi-job operations need honest per-submission estimates and a sufficient total ceiling; the engine does not infer tier/options-based pricing. Existing source preserves FAL queue, FAL direct-run and World Labs submission guards before credentials/network. Paid credentials alone cannot trigger a job.

Legacy raw CLI jobs require deliberate process environment variables: `IMAGE_BLASTER_ALLOW_PAID=1`, `IMAGE_BLASTER_PAID_ENDPOINT`, `IMAGE_BLASTER_ESTIMATED_COST_USD`, `IMAGE_BLASTER_MAX_COST_USD`, `IMAGE_BLASTER_APPROVAL`. These flags record separately granted authorization; agents must never fabricate it. For composite legacy image-edit + object generation, prefer distinct explicit engine requests and approvals. Ordinary provider credentials remain `FAL_KEY` and `WORLD_LABS_API_KEY`; they are never included in request manifests. Native job uploads/output quality remain unverified in this pass.

This guard limits quoted submissions in one process. It does **not** cap a provider account, settle invoices, reserve across separate processes, prevent underestimated tariffs, or pay for cloud GPU startup. Account-side quotas, billing reconciliation and a durable global budget ledger are future work. Existing polling/downloads can use credentials when separately authorized and do not submit new generation jobs.

## Adapter boundaries

`engine/adapters.mjs` produces internal-only JSON manifests, without modifying either client. TPS export preserves asset hashes/license/material references, source/fact evidence, transforms, cameras/relationships and explicit unvalidated collision metadata. TPS owns terrain/simulation; placement requires deterministic client approval. Intended next validation uses Codex + VibeUE in an authorized staging level, checks GLB/OBJ parsing, axes/units/winding/pivots, materials, collisions, screenshots, triangles and profiling. A world collider/splat is not gameplay-approved terrain.

Benson visualization export retains inferred and resolved dimensions, transform evidence, materials, cameras/relationships and visual authority/confidence. Technical export requires nonempty entities, registered authoritative geometry, authoritative width/height/depth and transform evidence plus explicit existing registered-geometry dimension-match validation, with no failed/needs-review entity/asset check. Even that export has `installationApproved: false`: it is input to source-backed Benson technical workflows, never an installation approval. Manufacturer/manual configuration, datum and physical applicability remain client-owned. No customer-facing projection is implemented.

## Claude/Cursor migration

The upstream audit inventories all eight skills, six agents and two hooks. Retained code keeps atomic isolation, explicit object intent, one-pass clean plates, indexed artifacts, disk-first loading, request IDs and repair/resume metadata. Original `.claude` scaffolding remains for source compatibility and historical workflows, with root guidance taking priority for Codex. Codex uses root `AGENTS.md` and `.agents/skills/scene-engine/SKILL.md`, not mechanically copied Claude agents/settings/hooks.

Codex retires Bash-only session hooks, key-paste onboarding, provider-required startup, mandatory background-agent fan-out, automatic full world/object/audio blasting and the old ban on visual inspection. The original viewer is retained without redesign. Visual thesis: calm existing canvas/viewer continuity. Content plan: original scene workspace and controls, new operator contracts in CLI/docs. Interaction thesis: preserve navigation/placement; generation and approval are explicit CLI operations. No new decorative UI or automatic paid button was added.

## Verification limits

Tests cover real Windows subprocess entrypoints, schema/reference/provenance failures, measured overrides, conflict resolution, cache reuse/invalidation/integrity/concurrency/junction containment, spend policy and real retained provider modules using mocked fetch. The standalone CLI is exercised without an agent and with network explicitly disabled. App tests/typecheck/build are separate evidence from browser rendering or live splat quality. No lint configuration existed upstream; `npm run check:syntax` checks all engine/test/legacy script syntax, not stylistic lint. Provider quality, real hosted output, open-model hardware, actual client import/exports and physical/user acceptance remain unverified.
