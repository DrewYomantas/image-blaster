export function normalizeConditioning(value, count) {
  if (value === undefined) return undefined;
  if (!value || Array.isArray(value) || typeof value !== "object") throw new Error("Conditioning must be an explicit object.");
  const { mode } = value;
  if (mode === "none") {
    if (Object.keys(value).length !== 1) throw new Error("none conditioning accepts no camera truth.");
    return undefined;
  }
  if (!["intrinsics-only", "pose"].includes(mode)) throw new Error("Unsupported conditioning mode.");
  const fields = ["mode", "provenance", "cameraConvention", "intrinsics", ...(mode === "pose" ? ["extrinsics"] : [])];
  if (Object.keys(value).length !== fields.length || fields.some((key) => !Object.hasOwn(value, key))) throw new Error("Conditioning fields must match the declared mode.");
  if (value.provenance !== "experiment-oracle" || value.cameraConvention !== "opencv-world-to-camera-scene-y-up-meters") throw new Error("Only explicit experiment oracle OpenCV cameras are supported.");
  const matrix = (items, size, name) => {
    if (!Array.isArray(items) || items.length !== count || items.some((m) => !Array.isArray(m) || m.length !== size || m.some((row) => !Array.isArray(row) || row.length !== size || row.some((n) => typeof n !== "number" || !Number.isFinite(n))))) throw new Error(`${name} camera count or finite matrix shape is invalid.`);
  };
  matrix(value.intrinsics, 3, "intrinsics");
  for (const k of value.intrinsics) if (k[0][0] <= 0 || k[1][1] <= 0 || k[0][1] !== 0 || k[1][0] !== 0 || k[2].some((n, i) => n !== [0, 0, 1][i])) throw new Error("Intrinsics must be positive zero-skew pinhole matrices.");
  if (mode === "pose") {
    if (count < 3) throw new Error("Pose conditioning requires at least three noncollinear cameras.");
    matrix(value.extrinsics, 4, "extrinsics");
    for (const e of value.extrinsics) {
      if (e[3].some((n, i) => n !== [0, 0, 0, 1][i])) throw new Error("Extrinsics must be homogeneous world-to-camera transforms.");
      const r = e.slice(0, 3).map((row) => row.slice(0, 3));
      const determinant = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
      if (Math.abs(determinant - 1) > 1e-5 || r.some((row, i) => r.some((other, j) => Math.abs(row.reduce((sum, n, a) => sum + n * other[a], 0) - Number(i === j)) > 1e-5))) throw new Error("Extrinsics require proper orthonormal rotations.");
    }
    const centres = value.extrinsics.map((e) => [0, 1, 2].map((i) => -e.slice(0, 3).reduce((sum, row) => sum + row[i] * row[3], 0)));
    const differences = centres.slice(1).map((c) => c.map((n, i) => n - centres[0][i]));
    if (!differences.some((a) => differences.some((b) => Math.hypot(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]) > 1e-8))) throw new Error("Pose camera centres must be noncollinear for supplied-camera scale alignment.");
  }
  return structuredClone(value);
}
