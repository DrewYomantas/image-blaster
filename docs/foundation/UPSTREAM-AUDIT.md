# Upstream audit and Windows baseline

Snapshot: 2026-10-03, Windows, Node 24.13.1, npm 11.8.0. Scope is source review and local unpaid verification. Provider execution, browser rendering, generated asset quality, and game-engine import are not certified by this audit.

## Source and license

- Upstream: [neilsonnn/image-blaster](https://github.com/neilsonnn/image-blaster), default branch `main`.
- Imported upstream head: `4acb43ba126a12358f71838d1b1a05e856b10eaf` (`license`). GitHub's live `commits/main` endpoint returned the same head during this run.
- Preceding commits inspected: `d13c7c11bbb331df24a2365cf450d76bdd8323d6` (README), `68955e57181ed8d67ec56eaf5a7adf6d6d9254f7` (viewer/loading), `aa8c66797704af3f14131ba42ca276d2682d157f` (cleanup/object generation), and `8b12bc0c6cbb01f0f2f1b86cb12f27f71b5d9a42` (mesh generation).
- `LICENSE.md` is preserved verbatim: MIT, copyright 2026 Neilson Koerner-Safrata. Preserve this copyright and permission notice when redistributing copies or substantial portions. No claim is made that the code license determines ownership or licensing of provider-generated assets.
- Local remotes were verified as `origin=https://github.com/DrewYomantas/image-blaster.git` and `upstream=https://github.com/neilsonnn/image-blaster.git`. No upstream PR was merged for this foundation.

Live GitHub API queries found six open PRs and three non-PR open issues:

| Item | Subject | Foundation decision |
| --- | --- | --- |
| [PR 12](https://github.com/neilsonnn/image-blaster/pull/12) | All 15 Windows CLI entrypoints exit silently | Reproduced locally; fix the defect using native `import.meta.main`. PR head `27b7a0fde127e1e968988ed291c451e904477c06` was reviewed, not merged. |
| [PR 7](https://github.com/neilsonnn/image-blaster/pull/7) | Equivalent Windows entrypoint repair | Confirms the same defect; no separate merge. Head `893c25d88d0e91cd3dfdbd756d383423eee4be88`. |
| [PR 5](https://github.com/neilsonnn/image-blaster/pull/5) | Impact-driven object SFX | Leave out of the foundation scope. |
| [PR 4](https://github.com/neilsonnn/image-blaster/pull/4) | Humanoid character generation | Leave out of scope. |
| [PR 3](https://github.com/neilsonnn/image-blaster/pull/3) | USDZ/STL/FBX export | Leave out of scope; the milestone exporter has its own contract. |
| [PR 2](https://github.com/neilsonnn/image-blaster/pull/2) | Tripo3D provider | Leave out of scope. |
| [Issue 11](https://github.com/neilsonnn/image-blaster/issues/11) | API key error | Relevant onboarding symptom; no live credentials or provider diagnosis attempted. |
| [Issue 9](https://github.com/neilsonnn/image-blaster/issues/9) | Other agents | Relevant to portability, but not evidence that other hosts execute the Claude workflow. |
| [Issue 8](https://github.com/neilsonnn/image-blaster/issues/8) | Single-generation cost | No pricing inferred or paid requests submitted. |

These are a dated status snapshot, not a promise that upstream remains unchanged.

## Architecture and behavior

The upstream product has two parts: Claude-specific orchestration files and a React/TypeScript viewer. There is no standalone backend package. Provider clients are JavaScript ESM scripts using Node built-ins and `fetch`; local viewer middleware lives in `app/vite.config.ts`.

`worlds/<slug>/project.json` stores the project envelope; `source/` stores staged images and sibling analysis JSON; root `image.json` stores merged observational analysis. `output/<object>/object.json` stores durable object intent. Generated assets are indexed `N-slug.ext` files, with colocated hidden `.N-slug-request.json` sidecars containing submission/polling state and provider provenance. Root `scene.json` is the upstream viewer's version-1 placement format, with instance transforms, physics mode, and optional lighting/ground settings. It is not the new engine SceneSpec.

`project-state.mjs` derives progress from disk, creates project directories, and can move files from `input/` when staging is explicitly requested. Downloads and local repairs write into `worlds/` or `input/`; delete is a dry run until `--yes`. Provider scripts block until completion. FAL queue and World Labs helpers persist request metadata for polling/resume, then download referenced assets. Non-loop SFX postprocessing requires external `ffmpeg`/`ffprobe`; loop audio remains unprocessed.

The viewer uses React 19, Three 0.180, React Three Fiber, Spark Gaussian splats, Rapier physics, Wouter routing, Zustand, Leva, Radix themes, and Tailwind. Its world catalog is assembled by Vite's virtual module/plugin. Development endpoints refresh the catalog, serve local world assets, and read/write placement `scene.json`. Folder-opening middleware can launch the OS file manager; terminal launch is macOS-only. This is local development middleware, not a hardened production service. A production build embeds catalog data but does not demonstrate that `/worlds` assets are packaged or that development write endpoints exist under preview/hosting.

## Eight skills, six agents, two hooks

The following upstream files were inspected as workflow source. They were not invoked as generation authorization during the foundation build.

| Skill | Actual contract |
| --- | --- |
| `image-blast-project` | Create/inspect one project envelope; stage input and run no-cost analysis before downstream generation. |
| `image-blast-uncover` | Agent image analysis, per-image/merged flat JSON; present object candidates and wait for confirmation before writing object intent. |
| `image-blast-plate` | One removal-only clean plate from confirmed objects, through the generic image-edit helper. |
| `image-blast-world` | One static World Labs environment; synthesize an empty-environment prompt and resume/download local world assets. |
| `image-blast-3d` | Exactly one atomic object, optional reference extraction, Hunyuan default or explicit Meshy; preserve provenance. |
| `image-blast-sfx` | One ambience, object-impact set, or literal custom sound request; external audio postprocessing for non-loops. |
| `image-blast-image-edit` | One explicit image/prompt/output-target edit with Nano Banana or GPT Image 2. |
| `image-blast-wildcard` | Discover a FAL endpoint; require exact endpoint confirmation before one execution using `CONFIRMED_FAL_ENDPOINT:`. |

Six `.claude/agents/` definitions cover world, 3D, plate, SFX, image-edit, and wildcard. They use inherited models, preload matching skills, and run in the background. Each rejects missing/ambiguous scope and handles one request; the wildcard agent requires the confirmation marker. These definitions describe Claude-specific orchestration, not portable engine behavior.

`.claude/settings.json` enables the eight skills and selected Bash commands. Its SessionStart hook runs `setup-check.sh`; its UserPromptSubmit hook runs `input-check.sh`. Both require Bash and Unix utilities. The setup hook reads project `.env` to identify missing/placeholder keys and prints project/input status. Its upstream wording asks users to paste keys into conversation; that onboarding is unsuitable for the fork's local-only credential policy. The input hook prints staged filenames. Neither was executed here, and native PowerShell support is not established.

Upstream `.claude/rules/project.md` assumes `lsof`, Bun, Claude background agents, and automatic viewer launching. `.claudeignore` excludes `/app` from Claude end-user scope; root `CLAUDE.md` is empty upstream. Cursor rules reinforce the Claude workflow. The development rule also contains a stale `scene/project.json`/`THREE.ObjectLoader` claim that disagrees with the current version-1 root `scene.json` implementation. These files are retained upstream material; they should not be treated as foundation authority without adaptation.

## Baseline defects and repairs

| Check | Upstream baseline | Repaired local result |
| --- | --- | --- |
| App tests | 11 passed, 2 failed because tests required live `WORLD_LABS_API_KEY` and `FAL_KEY`. | 13/13 pass; environment tests parse `.env.example` and assert the documented placeholder contract. No actual `.env` is read by these tests. |
| Typecheck/build | `SparkRenderer.encodeLinear` no longer exists in resolved Spark 2.3.1, causing compilation failure. | Root `npm run typecheck` and `npm run build` pass after removing obsolete property/constructor plumbing. |
| CLI entrypoints | 15 scripts compare a file URL against an unencoded string built from `process.argv[1]`; on Windows the main function never runs and the process exits 0 silently. | All 15 use `import.meta.main`; missing arguments now produce the expected validation error/status 1. |
| Root commands | Require Bun. | npm workspace dev/build/preview/typecheck; combined Node/app test command; `npm run engine -- ...`; Node `>=24.2.0`. |
| Dependency audit | 10 findings from frontend development tool chains. | Full audit still reports 3 moderate, 6 high, 1 critical; `npm audit --omit=dev --json` reports 0 findings. |

[Node documents `import.meta.main`](https://nodejs.org/api/esm.html#importmetamain) from Node 24.2/22.18. The fork chooses Node 24.2 or newer as its declared floor. [Bun documents the same entrypoint property](https://bun.com/docs/runtime/module-resolution#importmeta), so no Node-specific URL helper was inserted into all scripts. Bun runtime behavior remains unverified because Bun was not installed on this machine.

The [Spark 2.3.1 implementation](https://github.com/sparkjsdev/spark/blob/v2.3.1/src/SparkRenderer.ts) selects linear encoding from the active renderer/render-target color space. The application should let Spark perform that selection rather than assigning a removed property. Existing quality-dependent depth-of-field logic is preserved. The custom shader patch still depends on upstream shader text; build success does not certify that effect visually.

`tests/windows-cli.test.mjs` passed 17/17 on actual Windows Node 24.13.1. It copies scripts into a temporary path containing spaces, `#`, `café`, and `世界`, asserts native drive/backslash paths, and invokes each of the 15 CLIs as a real child process. It also imports all 15 without running their main functions. The fixture has no project `.env`, provider keys are empty, and a preloaded `fetch` blocker prevents paid network execution. This checks entrypoint and missing-argument behavior, not provider generation or generated files.

The repaired Vite build transformed 5,443 modules and emitted approximately 6.40 MB JavaScript and 729.65 KB CSS before gzip; the existing large-chunk warning remains. No frontend dependency major upgrades were introduced. Audit fixes currently suggest major Vite/Vitest/Tailwind migrations; those require a separate focused compatibility pass. Ajv is the new engine's runtime JSON Schema validator, recorded in the npm lockfile.

## Migration recommendation

Make the portable Node engine and its versioned SceneSpec the operational authority. Keep planning, provider execution, credentials, request receipts, and export validation behind explicit CLI contracts. Treat Claude/Cursor skills and the React viewer as optional clients. Retain the existing indexed asset/provenance conventions where useful, but distinguish durable source intent, provider receipts, the upstream placement document, and the new SceneSpec rather than calling them the same scene format.

Do not migrate every legacy workflow into the engine at once. First prove dry planning, offline/mock execution, interrupted-request handling, artifact validation, and package export without keys. Then connect paid adapters only through the explicit execution gate and local credentials. Add optional orchestration and viewer integration after the engine contract is stable. Leave the open character/provider/audio/export PR feature proposals out of this foundation.

Remaining acceptance: paid FAL/World Labs requests, Bash/Claude hook behavior on Windows, Bun execution, browser splat/color/DOF rendering and interactions, production asset serving, and Unreal/other target-engine import all remain unverified. Passing local tests and a static build do not establish those outcomes.
