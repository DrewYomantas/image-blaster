import { assertScene, isAuthoritative, resolveFact } from "./scene.mjs";

function entityManifest(entity) {
  return {
    id: entity.id,
    label: entity.label,
    geometry: entity.geometry || null,
    transform: entity.transform ? resolveFact(entity.transform) : null,
    transformEvidence: entity.transform || [],
    properties: entity.properties || {},
    dimensionEvidence: entity.dimensions,
    resolvedDimensions: Object.fromEntries(Object.entries(entity.dimensions).map(([axis, evidence]) => [axis, resolveFact(evidence)])),
    materialIds: entity.materialIds || []
  };
}

export function exportTPS(scene) {
  assertScene(scene);
  return {
    schemaVersion: 1, adapter: "tps-unreal-reference", sceneId: scene.id, coordinates: scene.coordinates,
    internalOnly: true, cameras: scene.cameras, relationships: scene.relationships,
    authority: "visual-reference-only", terrainAuthority: "TPS", simulationAuthority: "TPS",
    requiresDeterministicPlacementApproval: true,
    assets: scene.artifacts,
    entities: [...scene.surfaces, ...scene.objects].map((entity) => ({ ...entityManifest(entity), collision: { mode: "unvalidated", gameplayApproved: false } })),
    sources: scene.sources, materials: scene.materials, validations: scene.validations,
    validationPlan: ["verify local artifact hashes and licenses", "convert meters/right-handed/Y-up to Unreal centimeters/left-handed/Z-up in importer", "import into isolated staging level using Codex and VibeUE", "validate scale, pivots, material slots, collision, triangles, screenshots and performance", "TPS deterministically approves placement; generated terrain remains a reference"]
  };
}

export function exportBenson(scene, { purpose = "visualization" } = {}) {
  assertScene(scene);
  if (!["visualization", "technical"].includes(purpose)) throw new Error("Invalid Benson export purpose.");
  const entities = [...scene.surfaces, ...scene.objects];
  if (purpose === "technical") {
    if (!entities.length) throw new Error("Technical export requires authoritative entities.");
    for (const entity of entities) {
      if (entity.geometry?.role !== "authoritative") throw new Error(`Technical export blocked: ${entity.id} has visual or missing geometry.`);
      for (const axis of ["width", "height", "depth"]) resolveFact(entity.dimensions[axis] || [], { requireAuthoritative: true });
      if (!entity.transform) throw new Error(`Technical export blocked: ${entity.id} has no authoritative transform.`);
      resolveFact(entity.transform, { requireAuthoritative: true });
      const evidence = scene.validations.find((result) => result.target === entity.id && result.check === "registered-geometry-dimension-match" && result.status === "pass");
      if (!evidence || scene.validations.some((result) => [entity.id, entity.geometry.artifactId].includes(result.target) && result.status !== "pass")) throw new Error(`Technical export blocked: ${entity.id} requires registered geometry validation.`);
    }
  }
  return {
    schemaVersion: 1, adapter: "benson-hearth-scene", sceneId: scene.id, purpose, coordinates: scene.coordinates,
    internalOnly: true,
    installationAuthority: "Benson measured/source-backed rules", installationApproved: false,
    entities: entities.map((entity) => ({ ...entityManifest(entity), visualOnly: entity.geometry?.role !== "authoritative", scaleAuthoritative: ["width", "height", "depth"].every((axis) => entity.dimensions[axis]?.some(isAuthoritative)) })),
    cameras: scene.cameras, relationships: scene.relationships, materials: scene.materials, artifacts: scene.artifacts, sources: scene.sources, validations: scene.validations
  };
}
