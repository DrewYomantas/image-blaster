import http from 'node:http';
import { readFile, realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repo = fileURLToPath(new URL('../../', import.meta.url));
export async function inspectionServer(directory, port = 4318) {
  const root = await realpath(directory);
  const three = await realpath(path.join(repo, 'node_modules/three'));
  const page = path.join(repo, 'benchmarks/structure/inspect.html');
  const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.png': 'image/png', '.obj': 'text/plain' };
  const server = http.createServer(async (request, response) => {
    try {
      if (request.method !== 'GET') { response.writeHead(405).end(); return; }
      const route = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
      const base = route.startsWith('/three/') ? three : root;
      const relative = route.startsWith('/three/') ? route.slice(7) : route.slice(1);
      const file = route === '/' ? page : await realpath(path.resolve(base, relative));
      if (route !== '/' && (path.relative(base, file).startsWith('..') || path.isAbsolute(path.relative(base, file)))) throw new Error('outside inspection root');
      if (!(await stat(file)).isFile()) throw new Error('not a file');
      const ext = path.extname(file);
      if (!Object.hasOwn(types, ext)) throw new Error('not an inspection artifact');
      response.writeHead(200, { 'content-type': types[ext], 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' });
      response.end(await readFile(file));
    } catch { response.writeHead(404).end('Inspection artifact unavailable'); }
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(port, '127.0.0.1', resolve); });
  return server;
}
if (import.meta.main) {
  const root = process.argv[2] || path.join(repo, '.image-blaster/structure-v1');
  const server = await inspectionServer(root, Number(process.argv[3] || 4318));
  console.log(`Structural inspection: http://127.0.0.1:${server.address().port}/`);
}
