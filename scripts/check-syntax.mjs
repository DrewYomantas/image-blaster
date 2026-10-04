import { readdir } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import path from "node:path";

let checked = 0;
async function inspect(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) await inspect(file);
    else if (entry.name.endsWith(".mjs")) {
      const result = spawnSync(process.execPath, ["--check", file], { encoding: "utf8" });
      if (result.status !== 0) throw new Error(result.stderr || `Syntax check failed: ${file}`);
      checked += 1;
    }
  }
}
for (const directory of ["engine", "tests", ".claude/scripts", "scripts", "benchmarks/geometry", "benchmarks/structure"]) await inspect(directory);
console.log(`Syntax checked ${checked} JavaScript modules.`);
