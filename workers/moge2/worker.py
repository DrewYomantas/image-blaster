import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
CODE_REVISION = "74fbce054ebed49800de42d0ad0e83495065719a"
HELPER_REVISION = "62f09d58509485564e24d5d9f6aac9ee9ebc0c37"
MODEL_REVISION = "26b477f41595707c5db6770294c0d1721e8ed4ed"
MODEL_ID = "Ruicheng/moge-2-vits-normal"
MODEL_HASH = "79a16621928c2bf0ed04659218c55c01075e950507f40bb3332fb4c873d3e1dc"
MODEL_BYTES = 140550416
ADAPTER_VERSION = "moge2-vits-normal-cpu-1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locations():
    source = Path(os.environ.get("IMAGE_BLASTER_MOGE2_SOURCE_DIR", ROOT / ".runtime/upstream"))
    helper = Path(os.environ.get("IMAGE_BLASTER_MOGE2_HELPER_DIR", ROOT / ".runtime/utils3d"))
    model = Path(os.environ.get("IMAGE_BLASTER_MOGE2_MODEL_DIR", ROOT / f".runtime/models/{MODEL_REVISION}"))
    return source.resolve(), helper.resolve(), model.resolve()


def source_identity(source, revision, package):
    actual_revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual_revision != revision:
        raise ValueError("Upstream source revision differs from reviewed revision")
    tree = subprocess.check_output(["git", "-C", str(source), "ls-tree", "-r", "HEAD"], text=True)
    files = []
    for row in tree.splitlines():
        meta, name = row.split("\t", 1)
        if not (name.startswith(package + "/") and name.endswith(".py") or name in ("pyproject.toml", "LICENSE", "README.md")):
            continue
        data = (source / name).read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != meta.split()[2]:
            raise ValueError(f"Upstream source modified: {name}")
        files.append({"path": name, "sha256": hashlib.sha256(data).hexdigest()})
    tracked = {entry["path"] for entry in files}
    actual = {path.relative_to(source).as_posix() for path in (source / package).rglob("*.py")}
    if not files or not actual or not actual.issubset(tracked):
        raise ValueError("Missing or unreviewed upstream Python source")
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def identity():
    source, helper, model = locations()
    checkpoint = model / "model.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size != MODEL_BYTES or sha256(checkpoint) != MODEL_HASH:
        raise ValueError("Missing or mismatched reviewed checkpoint bytes")
    source_hash = source_identity(source, CODE_REVISION, "moge")
    helper_hash = source_identity(helper, HELPER_REVISION, "utils3d_moge")
    dependencies = {}
    for row in (ROOT / "requirements.txt").read_text().splitlines():
        name, expected = row.split("==", 1)
        version = importlib.metadata.version(name)
        if version != expected:
            raise ValueError(f"Runtime dependency differs from pin: {name} {version}")
        dependencies[name] = version
    environment = dict(sorted((distribution.metadata["Name"], distribution.version)
                              for distribution in importlib.metadata.distributions()))
    return {"protocolVersion": 1, "providerId": "moge2-vits-normal", "model": MODEL_ID,
            "modelRevision": MODEL_REVISION, "checkpointSha256": MODEL_HASH, "checkpointBytes": MODEL_BYTES,
            "upstreamRevision": CODE_REVISION, "upstreamSourceSha256": source_hash,
            "helperRevision": HELPER_REVISION, "helperSourceSha256": helper_hash,
            "adapterVersion": ADAPTER_VERSION,
            "adapterFiles": [{"path": name, "sha256": sha256(ROOT / name)} for name in ("worker.py", "requirements.txt")],
            "pythonVersion": platform.python_version(), "runtimeDependencies": dependencies, "installedPackages": environment,
            "device": "cpu", "dtype": "float32",
            "license": {"code": "MIT", "weights": "MIT", "dinov2": "Apache-2.0", "helper": "MIT", "commercialUse": "permitted"}}


def validate_conditioning(value, count):
    if value is None:
        return {"mode": "none"}
    if not isinstance(value, dict):
        raise ValueError("Conditioning must be an explicit object")
    if value == {"mode": "none"}:
        return value
    if set(value) != {"mode", "provenance", "horizontalFovDegrees"} or value["mode"] != "known-fov" or value["provenance"] != "experiment-oracle":
        raise ValueError("Only explicit experiment-oracle known horizontal FOV is supported")
    values = value["horizontalFovDegrees"]
    if not isinstance(values, list) or len(values) != count or any(
        type(fov) not in (int, float) or not math.isfinite(fov) or not 1 < fov < 179 for fov in values
    ):
        raise ValueError("Horizontal FOV requires one finite degree value between1 and179 per input")
    return value


def validate_request(request):
    if not isinstance(request, dict) or set(request) - {"inputs", "outputDir", "parameters", "expectedIdentity", "conditioning"}:
        raise ValueError("Only inputs, outputDir, parameters, expectedIdentity and conditioning are accepted")
    inputs = request.get("inputs")
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= 8:
        raise ValueError("inputs must contain1 to8 ordered local images")
    paths = []
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {"path"} or not isinstance(item["path"], str):
            raise ValueError("Each input contains only path")
        path = Path(item["path"]).resolve()
        if not path.is_file() or path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            raise ValueError("Inputs must be local PNG/JPEG files")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate input images are not accepted")
    if not isinstance(request.get("outputDir"), str) or not request["outputDir"]:
        raise ValueError("outputDir is required")
    output = Path(request["outputDir"]).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("outputDir must be absent or empty")
    parameters = request.get("parameters", {})
    if not isinstance(parameters, dict) or set(parameters) - {"resolutionLevel", "threads"}:
        raise ValueError("Only resolutionLevel and threads are accepted")
    resolution = parameters.get("resolutionLevel", 9)
    threads = parameters.get("threads", 6)
    if type(resolution) is not int or not 0 <= resolution <= 9:
        raise ValueError("resolutionLevel must be an integer from0 to9")
    if type(threads) is not int or not 1 <= threads <= 64:
        raise ValueError("threads must be an integer from1 to64")
    conditioning = validate_conditioning(request.get("conditioning"), len(paths))
    return paths, output, resolution, threads, conditioning


def peak_memory_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage",
                )]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not query(process(), ctypes.byref(counters), counters.cb):
            raise OSError("Peak process memory query failed")
        return counters.PeakWorkingSetSize
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def load_model(model_class, checkpoint, torch):
    model = model_class.from_pretrained(str(checkpoint))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(state) != {"model", "model_config"} or set(model.state_dict()) != set(state["model"]):
        raise ValueError("Checkpoint has missing or unexpected model parameters")
    model.load_state_dict(state["model"], strict=True)
    return model


def validate_predictions(arrays, count, height, width):
    import numpy as np

    shapes = {"camera_points": (count, height, width, 3), "depths": (count, height, width),
              "masks": (count, height, width), "normals": (count, height, width, 3),
              "intrinsics_normalized": (count, 3, 3), "intrinsics": (count, 3, 3),
              "processed_images": (count, height, width, 3)}
    if any(name not in arrays or arrays[name].shape != shape for name, shape in shapes.items()):
        raise ValueError("Invalid native model output shape")
    if arrays["masks"].dtype != np.bool_:
        raise ValueError("Native validity mask must be boolean")
    mask = arrays["masks"]
    for name in ("camera_points", "depths", "normals"):
        if not np.isfinite(arrays[name][mask]).all():
            raise ValueError(f"Nonfinite valid model output: {name}")
    if not (arrays["depths"][mask] > 0).all() or not np.allclose(arrays["camera_points"][..., 2][mask], arrays["depths"][mask], atol=1e-5):
        raise ValueError("Depth must equal positive camera-point Z")
    for name in ("intrinsics", "intrinsics_normalized"):
        k = arrays[name]
        if not np.isfinite(k).all() or not (k[:, 0, 0] > 0).all() or not (k[:, 1, 1] > 0).all() or not np.allclose(k[:, 2], [0, 0, 1], atol=1e-6):
            raise ValueError("Invalid model intrinsics")
    if not np.allclose(np.linalg.norm(arrays["normals"][mask], axis=-1), 1, atol=1e-3):
        raise ValueError("Valid model normals must be signed unit vectors")


def predict(paths, resolution, threads, conditioning):
    source, helper, model_dir = locations()
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path[:0] = [str(source), str(helper)]
    import numpy as np
    import torch
    from PIL import Image
    from moge.model.v2 import MoGeModel
    import utils3d_moge as utils3d

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    started = time.perf_counter()
    model = load_model(MoGeModel, model_dir / "model.pt", torch).to(device="cpu", dtype=torch.float32).eval()
    load_seconds = time.perf_counter() - started
    shapes = []
    source_images = []
    started = time.perf_counter()
    for path in paths:
        with Image.open(path) as image:
            source_images.append(np.array(image.convert("RGB"), dtype=np.uint8))
            shapes.append(image.size)
    if len(set(shapes)) != 1:
        raise ValueError("This adapter requires identical source-image dimensions")
    tensors = [torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1) for image in source_images]
    preprocess_seconds = time.perf_counter() - started
    outputs = []
    per_image_seconds = []
    for index, image in enumerate(tensors):
        fov = conditioning["horizontalFovDegrees"][index] if conditioning["mode"] == "known-fov" else None
        started = time.perf_counter()
        prediction = model.infer(image, resolution_level=resolution, fov_x=fov,
                                 force_projection=True, apply_mask=True, use_fp16=False)
        per_image_seconds.append(time.perf_counter() - started)
        outputs.append({name: value.detach().cpu().numpy() for name, value in prediction.items()})
    height, width = source_images[0].shape[:2]
    arrays = {name: np.stack([prediction[key] for prediction in outputs]) for name, key in (
        ("camera_points", "points"), ("depths", "depth"), ("masks", "mask"),
        ("normals", "normal"), ("intrinsics_normalized", "intrinsics"))}
    arrays["intrinsics"] = utils3d.pt.denormalize_intrinsics(
        torch.from_numpy(arrays["intrinsics_normalized"]), (height, width), pixel_convention="integer-center"
    ).numpy()
    arrays["processed_images"] = np.stack(source_images)
    arrays["conditioning_mode"] = np.asarray(conditioning["mode"])
    validate_predictions(arrays, len(paths), height, width)
    telemetry = {"loadSeconds": load_seconds, "preprocessSeconds": preprocess_seconds,
                 "inferenceSeconds": sum(per_image_seconds), "perImageInferenceSeconds": per_image_seconds,
                 "inferenceForwardCalls": len(paths), "parameterCount": sum(p.numel() for p in model.parameters()),
                 "effectiveNumTokens": int(model.num_tokens_range[0] + resolution / 9 * (model.num_tokens_range[1] - model.num_tokens_range[0]))}
    return arrays, telemetry


def run(request):
    started = time.perf_counter()
    paths, output, resolution, threads, conditioning = validate_request(request)
    reviewed_identity = identity()
    if request.get("expectedIdentity", reviewed_identity) != reviewed_identity:
        raise ValueError("Runtime identity changed since parent request preparation")
    input_hashes = [sha256(path) for path in paths]
    arrays, telemetry = predict(paths, resolution, threads, conditioning)
    if input_hashes != [sha256(path) for path in paths] or identity() != reviewed_identity:
        raise ValueError("Input or runtime changed during inference")
    output.mkdir(parents=True, exist_ok=True)
    geometry = output / "predictions.npz"
    write_started = time.perf_counter()
    import numpy as np
    np.savez_compressed(geometry, **arrays)
    telemetry["artifactWriteSeconds"] = time.perf_counter() - write_started
    _, height, width = arrays["depths"].shape
    cameras = [{"sourceIndex": index, "intrinsics": arrays["intrinsics"][index].tolist(),
                "normalizedIntrinsics": arrays["intrinsics_normalized"][index].tolist(),
                "imageWidth": width, "imageHeight": height, "inputWidth": width, "inputHeight": height,
                "pixelTransform": {"scaleX": 1, "scaleY": 1, "offsetX": 0, "offsetY": 0,
                                   "pixelCenters": "integer-center", "cropLeft": 0, "cropTop": 0}}
               for index in range(len(paths))]
    summary = output / "worker-result.json"
    telemetry["totalSeconds"] = time.perf_counter() - started
    telemetry["peakMemoryBytes"] = peak_memory_bytes()
    result = {"protocolVersion": 1, "status": "complete", "identity": reviewed_identity,
              "parameters": {"resolutionLevel": resolution, "threads": threads, "useFp16": False,
                             "forceProjection": True, "applyMask": True, "independentMonocularViews": True},
              "conditioning": conditioning,
              "inferenceSemantics": {"cameraPredictionIndependent": conditioning["mode"] == "none",
                               "fovSemantics": "horizontal-degrees-post-forward-focal-and-shift-solve",
                               "depthScaleSource": "learned-metric-scale-head"},
              "coordinates": {"cameraConvention": "opencv", "axes": "x-right,y-down,z-forward",
                              "depth": "camera-z", "units": "meters", "normal": "camera-facing signed unit vectors",
                              "worldFrame": "independent-camera-frames", "scaleSource": "learned-metric-scale-head",
                              "extrinsics": "not-predicted", "intrinsics": "normalized-uv-and-integer-center-pixels"},
              "inputs": [{"sourceIndex": index, "sha256": digest} for index, digest in enumerate(input_hashes)],
              "cameras": cameras, "geometryArtifact": str(geometry), "artifactSha256": sha256(geometry),
              "files": [str(geometry), str(summary)], "telemetry": telemetry}
    summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--request")
    args = parser.parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.identity:
                response = identity()
            else:
                request = json.loads(Path(args.request).read_text(encoding="utf-8-sig")) if args.request else json.load(sys.stdin)
                response = run(request)
        print(json.dumps(response, allow_nan=False))
        return 0
    except Exception as error:
        print(f"MoGe-2 worker failed: {error}", file=sys.stderr)
        print(json.dumps({"protocolVersion": 1, "status": "failed", "error": str(error)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
