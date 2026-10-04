import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { digest, requestKey } from "./cache.mjs";

export async function implementationFingerprint(roots, configuration = {}) {
  const graphs = [];
  for (const root of roots) {
    const anchor = path.resolve(root instanceof URL ? fileURLToPath(root) : root);
    const files = new Map();
    async function visit(absolute) {
      if (files.has(absolute)) return;
      const bytes = await readFile(absolute);
      files.set(absolute, digest(bytes));
      if (!/\.[cm]?js$/.test(absolute)) return;
      const imports = [...bytes.toString("utf8").matchAll(/(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s*)["'](\.\.?\/[^"']+)["']/g)];
      for (const match of imports) await visit(path.resolve(path.dirname(absolute), match[1]));
    }
    await visit(anchor);
    graphs.push([...files].map(([file, sha256]) => ({ path: file === anchor ? "<root>" : path.relative(path.dirname(anchor), file).split(path.sep).join("/"), sha256 })).sort((a, b) => a.path.localeCompare(b.path, "en")));
  }
  return requestKey({ graphs, configuration });
}

export async function providerIdentity(provider) {
  const checkpoint = provider.identity ? await provider.identity() : { revision: provider.checkpointRevision || "not-applicable" };
  return {
    id: provider.id, model: provider.model, version: provider.version,
    implementation: await implementationFingerprint([new URL(import.meta.url), ...provider.implementationFiles], provider.implementationConfig || {}),
    checkpoint
  };
}
