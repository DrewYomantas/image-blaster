# Foundation delivery evidence

## Current geometry milestone

The subsequent local DA3 Small experiment, starting from `4dce4527ba39f46c130d72a7248fdfd015deadce`, is recorded in [GEOMETRY-BENCHMARK.md](GEOMETRY-BENCHMARK.md), the [worker receipt](DA3-WORKER.md), and the [fixture protocol](../../benchmarks/geometry/README.md). It adds real CPU depth/camera inference, explicit scale calibration, measured-role override and objective synthetic errors. No paid service, GPU worker, client change or import was used.

## Historical initial foundation receipt

All remaining statements below describe the initial foundation pass, before DA3 was installed. Its former next milestone is now the synthetic geometry experiment above. They are retained as baseline evidence, not current claims that no model has run.

Session date: October 3, 2026 (America/Chicago); UTC timestamps extend into October 4. This is a tested foundation, not a delivered photo-to-room model or a client deployment.

## Repository and environment

| Item | Recorded value |
| --- | --- |
| Upstream | https://github.com/neilsonnn/image-blaster |
| Normal GitHub fork | https://github.com/DrewYomantas/image-blaster; GitHub reports `isFork: true` and parent `neilsonnn/image-blaster` |
| Upstream baseline | `4acb43ba126a12358f71838d1b1a05e856b10eaf` |
| Branch | `codex/provider-neutral-foundation` |
| Workspace | `C:\Projects\image-blaster` |
| Remotes | `origin` Drew's fork; `upstream` original project |
| Node / npm / Git | Node 24.13.1; npm 11.8.0; Git 2.55.0.windows.3 |
| Bun | Not installed; optional runtime not verified |
| OS | Windows 11 Home x64, version 10.0.26300 |
| Hardware | AMD Radeon RX 6800 XT; about 31.9 GiB physical RAM. WMI adapter memory field cannot establish actual VRAM. NVIDIA/CUDA stack absent from detected devices/tools |
| License | Original MIT LICENSE.md and 2026 Neilson Koerner-Safrata copyright unchanged |

No upstream history rewrite or main merge. Runtime/Windows/operator foundation commit: `5e03e4886bb51077e779312a17ff5d656f6e3601`. The second logical commit covers client/research/benchmark delivery documentation. Final commit identities are recorded in the session report; the documentation cannot embed its own final Git object ID without changing it.

## Implemented

- Standalone Node engine, seven capability interfaces, eight registered implementations: two local providers and six retained hosted wrappers.
- SceneSpec v1 JSON Schema/Ajv and semantic provenance validation. Original evidence/source/fact ownership is retained; measured/manufacturer authority overrides inference; conflicts and generated authority fail closed.
- Streamed content hashes, canonical effective request cache keys, immutable provider snapshots, staged local inputs, lifecycle/artifact manifests, integrity verification, duplicate-job lock and fail-closed retry policy.
- Default-deny paid submission guards at FAL queue, FAL direct-run and World Labs. Explicit approved endpoint/estimate/ceiling/reference required; quoted per-process limits are not vendor billing caps or a global ledger.
- Codex root AGENTS and one operator skill, normal CLI commands, npm install/test/build path, legacy Claude workflow marked historical, key-paste onboarding removed.
- All fifteen defective Windows entrypoints use native `import.meta.main`, requiring Node >=24.2. Original provider APIs/indexed artifacts/viewer preserved; world output path configurable; existing object/image orchestration dispatches via adapters.
- Internal TPS/Benson manifest exporters. Benson technical path rejects inferred/unregistered/unvalidated geometry and still never approves installation.
- Current read-only client integration audit, source-linked provider/license/hardware/cost matrix and a two-scene benchmark specification. No fixture photo is presented as a customer scene.

## Current-run verification

| Command/check | Result |
| --- | --- |
| `npm install` | Baseline install succeeded; baseline defects recorded in UPSTREAM-AUDIT.md |
| `npm ci` | Reproducible lockfile install succeeded: 346 packages added, 348 audited |
| `npm test` | 51 Node engine/native CLI tests plus 13 viewer tests passed; zero failures/skips |
| `npm run typecheck` | Passed |
| `npm run check:syntax` | Passed: 35 JavaScript modules |
| `npm run build` | Passed: Vite 5,443 modules; existing large-chunk warning |
| `npm run engine -- providers` | Eight concrete registrations listed without keys/model installations |
| `npm run engine -- generate --request examples/box-request.json` | Real local OBJ generated, then identical request returned `cached: true` |
| Engine CLI subprocess test | Ingest/cache/validate/export/local OBJ exercised without an agent or keys; network explicitly disabled |
| Retained provider integration tests | Actual Hunyuan/Meshy/Nano Banana/GPT image/World Labs implementations exercised through registry with mocked fetch; all six wrappers dispatched through injected runtime; not real provider output quality |
| Native Windows regression | 17 tests cover all15 missing-argument CLIs, import-without-run and actual Windows drive/backslash/space/#/Unicode paths |
| Independent adversarial review | Source and disposable offline fixture review during implementation and at completion; reported spend/provenance/mutation/path/cache issues repaired and regression-covered; no remaining reviewed core blocker |
| `npm audit --omit=dev --json` | Zero production dependency advisories |
| Full dependency audit | Ten inherited development-tool findings: 3 moderate, 6 high, 1 critical. Vite/Vitest/Tailwind chain major migrations deferred; do not claim a clean full audit |
| License/git hygiene | Original LICENSE.md unchanged; .env variants/cache ignored; staged files reviewed before commits; no credentials or generated private data staged |

Local test receipt: `.image-blaster/verification/test.log` (ignored). Deterministic OBJ cache entry `69399007f2d32e1710cf5a52c9073c9e539c0bea2878e54a4a987fb9a1b85e23`, artifact SHA-256 `1145c24a42c7b586404fc24cda9e291cf88369960d06ed5c4fc18740bb5aba34`, 12 triangles, 1.2 x 0.8 x 0.5 metres, no materials and no collision/installation approval. This is engine smoke evidence, not either planned scene benchmark.

## Paid and unverified boundaries

Actual spend in this session: **$0 in generation/cloud GPU charges**. No paid submit, provider upload, GPU provisioning, model install, actual source reconstruction or client mutation was performed. Hosted image edit/object/world/audio remain billable after separate approval. Self-hosting could avoid API margins but needs validated hardware, software/dependency rights and, if rented, an approved cloud budget.

Free now: Node engine startup, source hashing/evidence-envelope analysis, schema/provenance QA, internal manifest exports, cache reuse, local deterministic OBJ generation, viewer tests/build. A free owned-worker provider can be registered, but no real worker is installed. On the current AMD Windows machine, common NVIDIA CUDA model stacks are unverified; TRELLIS.2's official Linux/NVIDIA 24GB baseline would require a separate compatible worker. Do not confuse WSL availability with CUDA hardware.

Unverified: semantic image understanding, multi-view room reconstruction, real hosted generation/API schema compatibility and tariffs, visual/mesh/PBR quality, ffmpeg audio behavior, Bun/Linux runtime, browser interactions/color/DOF/splats, actual Unreal/VibeUE import/profiling, actual Benson import/technical packets, customer visualization approval and physical acceptance. Contract checks are not geometric/source/installation truth. Generic cache resume, portable cache relocation, global spend ledger, settled billing and supplier account terms are unfinished.

## Decision and continuation

Keep commercial/open models replaceable. Do not install a model merely because its top-level license appears permissive. TRELLIS.2's default NVIDIA dependencies restrict commercial use; Hunyuan has territory/output terms; small/base depth checkpoints have different licenses from their larger siblings; supplier ownership/retention varies. Jev remains an optional typed decision interface because ordinary deterministic rules already handle this milestone; it is not a mesh generator or installation evaluator.

Next milestone: a rights-cleared synthetic multi-view room fixture and one commercially eligible depth/camera provider, emitting only inferred SceneSpec facts. Verify camera/depth/scale against independent known dimensions and measured override behavior before adding another paid generator. Use the prepared TPS founding-site and Benson fireplace-room metrics, then separately authorize isolated client import. No full-park generation or renderer replacement is recommended.

Documents: ENGINE.md (runtime/schema/spend/export), UPSTREAM-AUDIT.md (baseline/Claude/Windows), CLIENT-INTEGRATION.md (latest client snapshots/boundaries), PROVIDER-RESEARCH.md (primary-source matrix), BENCHMARK.md (inputs/metrics/approval).
