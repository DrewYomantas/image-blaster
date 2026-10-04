import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { cp, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";

const scriptsRoot = fileURLToPath(new URL("../.claude/scripts/", import.meta.url));
const engineRoot = fileURLToPath(new URL("../engine/", import.meta.url));
const clis = [
  "asset-pipeline/generate-single-asset.mjs",
  "asset-pipeline/gpt-image-2-edit.mjs",
  "asset-pipeline/hunyuan-3d.mjs",
  "asset-pipeline/image-edit.mjs",
  "asset-pipeline/meshy-3d.mjs",
  "asset-pipeline/nano-banana-edit.mjs",
  "fal/run-fal.mjs",
  "image-edit/generate-edit.mjs",
  "project/delete.mjs",
  "project/download.mjs",
  "project/ensure-local-assets.mjs",
  "project/indexed-path.mjs",
  "project/project-state.mjs",
  "sfx/fal-elevenlabs-sfx.mjs",
  "world/generate-world.mjs"
];
let fixtureRoot;
let preload;

before(async () => {
  fixtureRoot = await mkdtemp(path.join(tmpdir(), "image blaster # café 世界-"));
  await cp(scriptsRoot, path.join(fixtureRoot, ".claude", "scripts"), { recursive: true });
  await cp(engineRoot, path.join(fixtureRoot, "engine"), { recursive: true });
  preload = path.join(fixtureRoot, "offline.mjs");
  await writeFile(preload, 'globalThis.fetch = () => { throw new Error("NETWORK_FORBIDDEN_IN_CLI_TEST"); };\n');
});

after(async () => {
  await rm(fixtureRoot, { recursive: true, force: true });
});

function run(args) {
  return spawnSync(process.execPath, ["--import", pathToFileURL(preload).href, ...args], {
    cwd: fixtureRoot,
    encoding: "utf8",
    timeout: 10000,
    windowsHide: true,
    env: { ...process.env, FAL_KEY: "", WORLD_LABS_API_KEY: "" }
  });
}

test(`native ${process.platform} fixture exercises spaces, hash, Unicode, and Windows drive separators`, () => {
  assert.match(fixtureRoot, /image blaster # café 世界-/u);
  assert.match(pathToFileURL(fixtureRoot).href, /%20.*%23.*caf%C3%A9/u);
  if (process.platform === "win32") {
    assert.match(fixtureRoot, /^[A-Z]:\\/i);
    assert.ok(fixtureRoot.includes("\\"));
  }
});

for (const cli of clis) {
  test(`${cli} executes argument validation from a special-character native path`, () => {
    const result = run([path.join(fixtureRoot, ".claude", "scripts", ...cli.split("/"))]);
    assert.ifError(result.error);
    assert.equal(result.status, 1, result.stderr || "CLI exited silently");
    assert.match(result.stderr, /Usage:|A project slug or description is required/);
    assert.doesNotMatch(result.stderr, /NETWORK_FORBIDDEN_IN_CLI_TEST|ERR_MODULE_NOT_FOUND|SyntaxError/);
    assert.equal(result.stdout, "");
  });
}

test("all 15 CLI modules can be imported without running their commands", () => {
  const urls = clis.map((cli) => pathToFileURL(path.join(fixtureRoot, ".claude", "scripts", ...cli.split("/"))).href);
  const source = `for (const url of ${JSON.stringify(urls)}) await import(url);`;
  const result = run(["--input-type=module", "--eval", source]);
  assert.ifError(result.error);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout, "");
  assert.equal(result.stderr, "");
});
