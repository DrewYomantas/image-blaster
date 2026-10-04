import { spawn } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { emptyScene, assertScene } from './scene.mjs';
import { fileDigest, canonicalJSON, digest } from './cache.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const license = { id: 'MIT', commercialUse: 'allowed', attribution: 'Image Blaster experimental structural fitter; input rights retained separately' };
const json = async (file) => JSON.parse(await readFile(file, 'utf8'));
export async function structuralScene(input, output) {
  const structure = await json(path.join(output, 'structure.json'));
  const receipt = await json(path.join(output, 'receipt.json'));
  for (const [name, expected] of Object.entries(receipt.artifacts)) {
    if (path.basename(name) !== name || await fileDigest(path.join(output, name)) !== expected) throw new Error('Structural artifact integrity mismatch.');
  }
  const expectedArtifacts = ['diagnostics.json', 'geometry.obj', 'predictions.json', 'structure.json'];
  if (canonicalJSON(Object.keys(receipt.artifacts).sort()) !== canonicalJSON(expectedArtifacts)) throw new Error('Incomplete structural receipt.');
  const lane = receipt.effective.lane;
  const constraint = receipt.effective.constraint;
  const manifest = await json(path.join(input, 'manifest.json'));
  const images = manifest.images.map((image, i) => ({ id: `photo-${i + 1}`, kind: 'image', uri: path.join(input, image.file), sha256: image.sha256, license: manifest.license || { id: 'UNKNOWN', commercialUse: 'unknown', attribution: '' } }));
  for (const source of images) if (await fileDigest(source.uri) !== source.sha256 || receipt.effective.inputHashes[path.basename(source.uri)] !== source.sha256) throw new Error('Structural source image changed.');
  const sources = [...images, { id: 'structural-method', kind: 'model', uri: path.join(output, 'receipt.json'), sha256: await fileDigest(path.join(output, 'receipt.json')), license }, { id: 'explicit-assistance', kind: 'user-input', uri: path.join(output, 'receipt.json'), sha256: await fileDigest(path.join(output, 'receipt.json')), license }];
  const scene = emptyScene(`structural-${lane}`, sources);
  const sourceIds = sources.map(s => s.id);
  const fact = (id, value, state = 'inferred', ids = sourceIds) => ({ id, value, state, sourceIds: ids, confidence: 0, note: 'Experimental conditional inference; confidence is uncalibrated. No field measurement or installation authority.' });
  for (const [id, file, format] of [['structural-evidence', 'structure.json', 'json'], ['structural-obj', 'geometry.obj', 'obj'], ['structural-diagnostics', 'diagnostics.json', 'json']]) {
    const uri = path.join(output, file);
    scene.artifacts.push({ id, uri, format, sha256: await fileDigest(uri), role: 'visual-only', sourceIds, provider: { id: 'local-structural-fit', model: 'bounded-manhattan-hypothesis', version: '1' }, parameters: { lane, assumptions: structure.hypotheses || [], status: structure.status }, license });
  }
  for (const [i, surface] of (structure.surfaces || []).entries()) {
    const id = `surface-${i + 1}`;
    scene.surfaces.push({ id, label: fact(`${id}-label`, surface.id || surface.name || id), dimensions: {}, properties: { structuralEvidence: [fact(`${id}-evidence`, surface)], uncertainty: [fact(`${id}-uncertainty`, structure.status || 'underconstrained')] }, geometry: { role: 'visual-only', artifactId: 'structural-obj', sourceIds } });
  }
  for (const [i, camera] of (structure.cameras || []).entries()) {
    const id = `camera-${i + 1}`;
    scene.cameras.push({ id, label: fact(`${id}-label`, camera.id), dimensions: {}, projection: [fact(`${id}-projection`, { intrinsics: camera.intrinsics, imageWidth: camera.width, imageHeight: camera.height, convention: 'opencv' })], pose: [fact(`${id}-pose`, { worldToCamera: camera.worldToCamera, coordinateFrame: 'scene-y-up', units: 'meters', scale: 'anchored' })] });
  }
  scene.objects.push({ id: 'opening-constraint', label: fact('opening-constraint-label', 'Opening mouth width constraint', 'user-specified', ['explicit-assistance']), dimensions: { width: [fact('opening-width-input', constraint.width_m, 'user-specified', ['explicit-assistance'])] }, properties: { datum: [fact('opening-width-datum', 'horizontal opening mouth between left and right jambs in the fireplace wall plane', 'user-specified', ['explicit-assistance'])], unresolved: [fact('structural-unresolved', structure.unknowns || [])] } });
  scene.validations.push({ target: scene.id, check: 'experimental-structural-fit', status: 'needs-review', details: 'Visual-only proposal. Supplied width is not independent accuracy. See structural diagnostics and separate held-out evaluation. Installation not approved.' });
  return assertScene(scene);
}
export async function saveProjection(scene, output) {
  const implementation = {};
  for (const file of ['engine/structure.mjs', 'engine/scene.mjs', 'engine/scene.schema.json', 'engine/cache.mjs']) implementation[file] = await fileDigest(path.join(root, file));
  const identity = { implementation, canonicalSceneSha256: digest(canonicalJSON(scene)), workerReceiptSha256: await fileDigest(path.join(output, 'receipt.json')) };
  const key = digest(canonicalJSON(identity));
  const receiptPath = path.join(output, 'projection-receipt.json');
  const scenePath = path.join(output, 'scene-spec.json');
  let prior;
  try { prior = await json(receiptPath); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  if (prior) {
    if (await fileDigest(scenePath) !== prior.sceneSha256) throw new Error('Saved SceneSpec integrity mismatch.');
    if (prior.key === key) return { scenePath, projectionCached: true };
  }
  await writeFile(scenePath, `${JSON.stringify(assertScene(scene), null, 2)}\n`);
  await writeFile(receiptPath, `${JSON.stringify({ version: 1, key, identity, sceneSha256: await fileDigest(scenePath), note: 'Derived SceneSpec projection identity; fitter replay identity is separately retained in receipt.json.' }, null, 2)}\n`);
  return { scenePath, projectionCached: false };
}
export async function runStructuralFit({ input, output, lane = 'assisted', annotations, python = path.join(root, 'workers/moge2/.venv/Scripts/python.exe') }) {
  if (!input || !output) throw new Error('structure requires --input and --out.');
  if (!['automatic', 'assisted', 'ablation'].includes(lane)) throw new Error('Unknown structural lane.');
  input = path.resolve(input); output = path.resolve(output);
  annotations ||= lane === 'automatic' ? 'endpoints.json' : 'annotations.json';
  const result = await new Promise((resolve, reject) => {
    const child = spawn(python, [path.join(root, 'workers/structure/fit.py'), '--input', input, '--annotations', annotations, '--lane', lane, '--output', output], { windowsHide: true, cwd: input, stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '', stderr = '';
    child.stdout.on('data', chunk => { stdout += chunk; });
    child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-12000); });
    child.on('error', reject);
    child.on('close', code => code === 0 ? resolve(stdout) : reject(new Error(stderr || stdout || `Structural worker exited ${code}`)));
  });
  const scene = await structuralScene(input, output);
  return { worker: result.trim(), ...await saveProjection(scene, output), note: 'Saved visual-only inferred structural evidence; geometry usefulness requires independent evaluation.' };
}
