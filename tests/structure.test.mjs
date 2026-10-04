import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { structuralScene, saveProjection } from '../engine/structure.mjs';
import { fileDigest } from '../engine/cache.mjs';
import { exportBenson } from '../engine/adapters.mjs';
import { inspectionServer } from '../benchmarks/structure/inspect.mjs';

async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'structural-test-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const input = path.join(root, 'input'), output = path.join(root, 'output');
  await mkdir(input); await mkdir(output);
  await writeFile(path.join(input, 'view-01.png'), 'synthetic-test-image-bytes');
  await writeFile(path.join(input, 'manifest.json'), JSON.stringify({ images: [{ file: 'view-01.png', sha256: await fileDigest(path.join(input, 'view-01.png')) }], constraint: { width_m: 1.2, datum: 'mouth-jambs' } }));
  await writeFile(path.join(output, 'receipt.json'), JSON.stringify({ artifacts: {}, effective: { lane: 'assisted', constraint: { width_m: 1.188 }, inputHashes: { 'view-01.png': await fileDigest(path.join(input, 'view-01.png')) } } }));
  await writeFile(path.join(output, 'diagnostics.json'), '{}');
  await writeFile(path.join(output, 'geometry.obj'), 'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n');
  await writeFile(path.join(output, 'structure.json'), JSON.stringify({ lane: 'assisted', status: 'underconstrained', surfaces: [{ id: 'wall', state: 'inferred', unsupported: ['rear'] }], cameras: [], unknowns: ['rear-depth'] }));
  await writeFile(path.join(output, 'predictions.json'), '{}');
  const artifacts = {};
  for (const name of ['structure.json', 'diagnostics.json', 'predictions.json', 'geometry.obj']) artifacts[name] = await fileDigest(path.join(output, name));
  await writeFile(path.join(output, 'receipt.json'), JSON.stringify({ artifacts, effective: { lane: 'assisted', constraint: { width_m: 1.188 }, inputHashes: { 'view-01.png': await fileDigest(path.join(input, 'view-01.png')) } } }));
  return { root, input, output };
}
test('structural evidence remains inferred and supplied width is not measured authority', async t => {
  const { input, output } = await fixture(t);
  const scene = await structuralScene(input, output);
  assert.ok(scene.artifacts.every(a => a.role === 'visual-only'));
  assert.equal(scene.objects[0].dimensions.width[0].state, 'user-specified');
  assert.equal(scene.objects[0].dimensions.width[0].value, 1.188);
  assert.deepEqual(scene.objects[0].properties.unresolved[0].value, ['rear-depth']);
  assert.equal(scene.surfaces[0].label.state, 'inferred');
  assert.throws(() => exportBenson(scene, { purpose: 'technical' }), /Technical export blocked/i);
  assert.equal(exportBenson(scene).installationApproved, false);
});
test('structural envelope rejects changed RGB source bytes', async t => {
  const { input, output } = await fixture(t);
  await writeFile(path.join(input, 'view-01.png'), 'changed');
  await assert.rejects(structuralScene(input, output), /source image changed/i);
});
test('inspection server serves only local inspection artifacts and rejects traversal', async t => {
  const { root, output } = await fixture(t);
  const server = await inspectionServer(root, 0);
  t.after(() => new Promise(resolve => server.close(resolve)));
  const base = `http://127.0.0.1:${server.address().port}`;
  assert.equal((await fetch(base + '/')).status, 200);
  assert.equal((await fetch(base + '/output/geometry.obj')).status, 200);
  assert.equal((await fetch(base + '/%2e%2e%5coutside.json')).status, 404);
  assert.equal((await fetch(base + '/output/geometry.obj', { method: 'POST' })).status, 405);
  await writeFile(path.join(output, 'private.txt'), 'not an inspection artifact');
  assert.equal((await fetch(base + '/output/private.txt')).status, 404);
});



test('SceneSpec projection verifies saved bytes before reuse', async t => {
  const { input, output } = await fixture(t);
  const scene = await structuralScene(input, output);
  assert.equal((await saveProjection(scene, output)).projectionCached, false);
  assert.equal((await saveProjection(scene, output)).projectionCached, true);
  await writeFile(path.join(output, 'scene-spec.json'), '{}');
  await assert.rejects(saveProjection(scene, output), /integrity mismatch/);
});

test('projection cache includes canonical source license metadata', async t => {
  const { input, output } = await fixture(t);
  const scene = await structuralScene(input, output);
  await saveProjection(scene, output);
  const changed = structuredClone(scene);
  changed.sources[0].license = { id: 'restricted-test', commercialUse: 'restricted', attribution: 'source changed' };
  assert.equal((await saveProjection(changed, output)).projectionCached, false);
  assert.equal((await saveProjection(changed, output)).projectionCached, true);
});
