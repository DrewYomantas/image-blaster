# Image Blaster: provider-neutral scene foundation

This MIT fork preserves [Neilson's Image Blaster](https://github.com/neilsonnn/image-blaster) and its React/Three.js viewer, indexed artifacts and hosted providers. The new Node engine owns versioned scene evidence, routing, manifests, caching and submission guards. Codex operates ordinary commands; Claude and paid credentials are optional.

```powershell
npm ci
npm test
npm run typecheck
npm run build
npm run engine -- providers
npm run engine -- generate --request examples/box-request.json
npm run engine -- analyze --image input/room.png --scene-id room --out scene-spec.json
npm run engine -- validate --scene scene-spec.json
npm run engine -- export --scene scene-spec.json --target benson --out benson-import.json
```

Requires Node >=24.2. Windows is covered by actual CLI subprocess tests. Local analysis records source hashes and preserves supplied evidence; it does **not** perform visual recognition/reconstruction. The procedural provider generates a real visual-only OBJ box. No paid service is required for startup, tests, viewer build or these operations. `input/room.png` is supplied by the operator; no private customer imagery ships here.

`npm run dev` starts the original viewer. Keep canonical SceneSpec separate from the viewer's `worlds/<slug>/scene.json` placement format. Exports are internal manifest boundaries, not actual Unreal/Benson importers or customer packets. Generated geometry is never installation truth, and TPS retains deterministic terrain/simulation authority.

Documentation: [engine and scene contract](docs/foundation/ENGINE.md), [upstream audit](docs/foundation/UPSTREAM-AUDIT.md), [current client integration](docs/foundation/CLIENT-INTEGRATION.md), [provider research](docs/foundation/PROVIDER-RESEARCH.md), [two-scene benchmark](docs/foundation/BENCHMARK.md), [delivery evidence](docs/foundation/DELIVERY.md).

The inherited development tools have known audit advisories; consult delivery evidence before exposing a development server. Preserve [LICENSE.md](LICENSE.md) and the upstream copyright notice in copies/distributions. Hosted model licenses and generated-output rights are independent of this MIT source license.

## Historical upstream README

The following documents original behavior and claims. Claude onboarding and automatic paid blasts are superseded by root AGENTS.md, the Node CLI and explicit spend policy. Do not paste keys into agent conversations.

<img width="960" height="540" alt="image-blaster-1" src="https://github.com/user-attachments/assets/d294e420-eb48-4f00-b6a8-13005442d1a8" />

## `image-blaster`
Creates 3D environments, SFX, and meshes from a single image using Claude skills, World Labs, and FAL. 

Can take you from an image to a fully meshed 3D environment in < 5 minutes, great for jumpstarting 3D work. Go full blast.


## Quickstart

1. Open a Terminal, enter `git clone https://github.com/neilsonnn/image-blaster`
2. Enter the directory with `cd image-blaster`
3. Run `claude` (install with `curl -fsSL https://claude.ai/install.sh | bash`)
4. Say hello to Claude, and give them your API key for [World Labs](https://platform.worldlabs.ai/) and [FAL](https://fal.ai/).
5. Put an image into `input/` directory and ask Claude to `blast it and confirm each step with me`.

### Description

By default `image-blaster` will use your input image to create:

1. 3D models (`.glb`, `.obj`) of all *dynamic* objects
2. Gaussian splat (`.spz`) of the *static* environment,
3. Ambient looping sound and object specific physics SFX (`.mp3`)

### Extensions

You can embed `image-blaster` under the assets of *any game engine, DCC software, or web app*.

1. Unity, Unreal, or Godot game engine
2. Blender, 3DS Max, or Maya or other DCC software
3. Three.js web app or Electron app

## Advanced

IMAGE-BLASTER uses a few generation models:

- `marble-1.1` - World Labs Marble model creates the explorable environment.
- `nano-banana` - default image edit preference for source cleanup, clean plates, and object reference images.
- `gpt-image-2` - alternate image edit provider when the edit skill is asked to prefer it.
- `hunyuan-3d` - Hunyuan 3D model creates 3D object models through FAL.
- `elevenlabs-sfx` - ElevenLabs sound effects model creates ambient and object-specific sounds.

3D model creation supports these Hunyuan parameters:

- `--face-count <40000-1500000>`: target face count. IMAGE-BLASTER defaults to `50000`; Hunyuan's API default is `500000`.
- `--enable-pbr true|false`: enable PBR material generation. Defaults to `true`.
- `--generate-type Normal|LowPoly|Geometry`: `Normal` creates a textured model, `LowPoly` applies polygon reduction, and `Geometry` creates a white geometry-only model. Defaults to `Normal`.
- `--polygon-type triangle|quadrilateral`: polygon type for `LowPoly`. Defaults to `triangle`.

### Examples

- Video game level concepts? `IMAGE-BLAST` it.
- Your childhood bedroom? `IMAGE-BLAST` it.
- Need an environment for a robot? `IMAGE-BLAST` it.
- A film location scout? `IMAGE-BLAST` it.
- An architectural rendering? `IMAGE-BLAST` it.

### Development

- remove `/app` from the `.claudeignore` file to give Claude the ability to change the React viewer.
