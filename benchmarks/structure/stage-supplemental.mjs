import { runGeneration } from '../../engine/run.mjs';
import { mkdir, readFile, writeFile, copyFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fileDigest } from '../../engine/cache.mjs';
const root = fileURLToPath(new URL('../../', import.meta.url));
const base = path.join(root, '.image-blaster/structure-v1-supplemental');
const input = path.join(base, 'input');
await mkdir(input, { recursive: true });
const images = [];
for (let i = 1; i <= 3; i++) {
  const file = `view-${String(i).padStart(2, '0')}.png`;
  await copyFile(path.join(base, 'fixture/inputs', file), path.join(input, file));
  images.push({ file, sha256: await fileDigest(path.join(input, file)), sourceIndex: i - 1 });
}
const license = { id: 'MIT', commercialUse: 'allowed', attribution: 'Image Blaster supplemental synthetic fixture' };
const request = { capability: 'scene-geometry', providerId: 'moge2-vits-normal', mode: 'local', parameters: { sceneId: 'structural-supplemental', resolutionLevel: 9, threads: 6 }, inputs: images.map(image => ({ path: path.join(input, image.file), mediaType: 'image/png', license })) };
const result = await runGeneration(request);
const worker = JSON.parse(await readFile(result.result.artifacts.find(a => path.basename(a.uri) === 'worker-result.json').uri, 'utf8'));
const prediction = result.result.artifacts.find(a => a.format === 'npz').uri;
await copyFile(prediction, path.join(input, 'hints.npz'));
if (worker.conditioning.mode !== 'none' || worker.inputs.length !== 3 || !worker.parameters.independentMonocularViews) throw new Error('Supplemental inference eligibility failure');
for (let i = 0; i < 3; i++) if (images[i].sha256 !== worker.inputs[i].sha256) throw new Error('Supplemental input mismatch');
await writeFile(path.join(input, 'manifest.json'), JSON.stringify({ version: 1, images, neural: { file: 'hints.npz', sha256: await fileDigest(prediction), parentArtifactSha256: await fileDigest(prediction), selectedIndices: [0, 1, 2], independentMonocularViews: true, conditioning: { mode: 'none' }, identity: worker.identity }, license }, null, 2));
await writeFile(path.join(input, 'constraint.json'), JSON.stringify({ kind: 'opening-mouth-width', width_m: 1.2 }));
await writeFile(path.join(base, 'inference-receipt.json'), JSON.stringify({ key: result.key, cached: result.cached, manifestPath: result.manifestPath, worker }, null, 2));
console.log(JSON.stringify({ key: result.key, cached: result.cached, telemetry: worker.telemetry, input }, null, 2));
