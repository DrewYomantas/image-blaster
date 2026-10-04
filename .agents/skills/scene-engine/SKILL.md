---
name: scene-engine
description: Operate the provider-neutral scene CLI for source ingestion, visual assets, provenance validation and internal adapter manifests.
---

# Scene engine operator

## Instructions

Read root AGENTS.md and docs/foundation/ENGINE.md. Inspect the source rights, active SceneSpec, cache manifests and hidden legacy job sidecars. Use normal CLI commands from the repository root; no agent is required for their execution.

1. `npm run engine -- providers` lists actual registered capabilities. Do not represent an unimplemented interface as a working model.
2. Ingest explicit local source images using `analyze --image <path> --scene-id <slug> --out <scene-spec.json>`. Optional `--evidence <scene-spec.json>` preserves reviewed facts. Local analysis hashes and records images; it does not see or reconstruct them.
3. For source interpretation, propose observed facts and inferred geometry separately. Preserve one atomic object per extraction and distinct scene surfaces. Record source hashes, camera uncertainty and fact provenance. Do not invent measured/manufacturer evidence. Agent inference must be persisted as evidence, not hidden orchestration state.
4. `generate --request <json>` uses explicit capability/provider/mode. Paid routing needs an approved endpoint, estimate, ceiling and approval reference; input upload rights must be confirmed before actual execution. Do not generate worlds/audio automatically. This session authorized no paid work.
5. Inspect returned manifests and local artifacts. A cache hit checks hashes; a contract pass does not establish topology, scale, visual suitability, collisions, source accuracy or product correctness. Use actual image/mesh inspection when QA is requested; historical instructions to never inspect imagery do not control Codex QA.
6. `validate --scene <json>` checks schema/provenance. `export --target tps|benson` emits internal manifest contracts, not live client imports. Benson `--purpose technical` rejects missing/inferred authority and never approves installation.
7. Report what ran, costs (zero for local fixtures), cache hits, output paths and unresolved acceptance. If interrupted, leave job IDs and inputs intact and reconcile before retrying; do not remove failed entries to trigger paid resubmission.
