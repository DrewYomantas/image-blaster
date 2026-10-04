import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
CODE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
MODEL_REVISION = "e08cab65ca0ec38e7826075418411ab90cab4da3"
MODEL_ID = "depth-anything/DA3-SMALL"
MODEL_HASH = "364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf"
CONFIG_HASH = "a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2"
ADAPTER_VERSION = "da3-small-cpu-3"
CHECKPOINT_ALIASES = {
    f"head.scratch.output_conv2_aux.{level}.2.{parameter}": f"head.scratch.output_conv2_aux.0.2.{parameter}"
    for level in (1, 2, 3) for parameter in ("weight", "bias")
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locations():
    source = Path(os.environ.get("IMAGE_BLASTER_DA3_SOURCE_DIR", ROOT / ".runtime/upstream"))
    model = Path(os.environ.get("IMAGE_BLASTER_DA3_MODEL_DIR", ROOT / f".runtime/models/{MODEL_REVISION}"))
    return source.resolve(), model.resolve()


def identity():
    source, model = locations()
    for name, expected in (("model.safetensors", MODEL_HASH), ("config.json", CONFIG_HASH)):
        if not (model / name).is_file() or sha256(model / name) != expected:
            raise ValueError(f"Missing or mismatched reviewed checkpoint file: {name}")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != CODE_REVISION:
        raise ValueError("Upstream code revision differs from reviewed revision")
    tree = subprocess.check_output(["git", "-C", str(source), "ls-tree", "-r", "HEAD"], text=True)
    files = []
    for row in tree.splitlines():
        meta, name = row.split("\t", 1)
        if not (name.startswith("src/depth_anything_3/") or name == "pyproject.toml"):
            continue
        if Path(name).suffix not in (".py", ".yaml", ".toml"):
            continue
        data = (source / name).read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != meta.split()[2]:
            raise ValueError(f"Upstream source modified: {name}")
        files.append({"path": name, "sha256": hashlib.sha256(data).hexdigest()})
    if not files:
        raise ValueError("Upstream source tree is empty")
    tracked = {entry["path"] for entry in files}
    actual = {path.relative_to(source).as_posix() for path in (source / "src/depth_anything_3").rglob("*.py")}
    if not actual.issubset(tracked):
        raise ValueError("Unreviewed Python files in upstream source tree")
    dependencies = {}
    for row in (ROOT / "requirements.txt").read_text().splitlines():
        name, expected = row.split("==", 1)
        version = importlib.metadata.version(name)
        if version != expected:
            raise ValueError(f"Runtime dependency differs from pin: {name} {version}")
        dependencies[name] = version
    environment = dict(sorted((distribution.metadata["Name"], distribution.version)
                              for distribution in importlib.metadata.distributions()))
    return {
        "protocolVersion": 1,
        "providerId": "local-da3-small",
        "model": MODEL_ID,
        "modelRevision": MODEL_REVISION,
        "checkpointSha256": MODEL_HASH,
        "configSha256": CONFIG_HASH,
        "upstreamRevision": CODE_REVISION,
        "upstreamSourceSha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "adapterVersion": ADAPTER_VERSION,
        "adapterFiles": [{"path": name, "sha256": sha256(ROOT / name)} for name in ("worker.py", "requirements.txt")],
        "pythonVersion": platform.python_version(),
        "runtimeDependencies": dependencies,
        "installedPackages": environment,
        "device": "cpu",
        "dtype": "float32",
        "license": {"code": "Apache-2.0", "weights": "Apache-2.0", "commercialUse": "permitted"},
    }


def validate_request(request):
    if not isinstance(request, dict) or set(request) - {"inputs", "outputDir", "parameters", "expectedIdentity"}:
        raise ValueError("Only inputs, outputDir, parameters and expectedIdentity are accepted")
    inputs = request.get("inputs")
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= 8:
        raise ValueError("inputs must contain 1 to 8 ordered local images")
    paths = []
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {"path"} or not isinstance(item["path"], str):
            raise ValueError("Each input must contain only path")
        path = Path(item["path"]).resolve()
        if not path.is_file() or path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            raise ValueError("Inputs must be local PNG/JPEG files")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate image paths are not accepted")
    if not isinstance(request.get("outputDir"), str) or not request["outputDir"]:
        raise ValueError("outputDir is required")
    output = Path(request["outputDir"]).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("outputDir must be absent or empty")
    parameters = request.get("parameters", {})
    if not isinstance(parameters, dict) or set(parameters) - {"processResolution", "threads", "profileWarmInference"}:
        raise ValueError("Only processResolution, threads and profileWarmInference are accepted; cameras, truth and scale are forbidden")
    resolution = parameters.get("processResolution", 384)
    if type(resolution) is not int or resolution not in (256, 384):
        raise ValueError("processResolution must be 256 or 384")
    threads = parameters.get("threads", 6)
    if type(threads) is not int or not 1 <= threads <= 64:
        raise ValueError("threads must be an integer from 1 to 64")
    profile = parameters.get("profileWarmInference", False)
    if type(profile) is not bool:
        raise ValueError("profileWarmInference must be a boolean")
    return paths, output, resolution, threads, profile


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
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not query(process(), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage if sys.platform == "darwin" else usage * 1024


def load_checkpoint(model, state, metadata):
    expected_metadata = {f"model.{alias}": f"model.{source}" for alias, source in CHECKPOINT_ALIASES.items()}
    if metadata != expected_metadata or not all(key.startswith("model.") for key in state):
        raise ValueError("Unexpected official checkpoint sharing metadata or key layout")
    state = {key[6:]: value for key, value in state.items()}
    model_state = model.state_dict()
    if set(model_state) - set(state) != set(CHECKPOINT_ALIASES) or set(state) - set(model_state):
        raise ValueError("Checkpoint has missing or unexpected keys beyond reviewed shared LayerNorm aliases")
    for alias, source in CHECKPOINT_ALIASES.items():
        target, shared = model_state[alias], model_state[source]
        if target.data_ptr() != shared.data_ptr() or target.shape != shared.shape or target.stride() != shared.stride():
            raise ValueError(f"Reviewed checkpoint alias does not share actual model storage: {alias}")
        state[alias] = state[source]
    return model.load_state_dict(state, strict=True)


def predict(paths, resolution, threads, profile):
    source, model_dir = locations()
    sys.path.insert(0, str(source / "src"))
    import numpy as np
    import torch
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import load_file
    from depth_anything_3.cfg import create_object
    from depth_anything_3.utils.io.input_processor import InputProcessor
    from depth_anything_3.utils.io.output_processor import OutputProcessor

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    started = time.perf_counter()
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_name") != "da3-small":
        raise ValueError("Checkpoint is not DA3-SMALL")
    model = create_object(config["config"])
    state = load_file(str(model_dir / "model.safetensors"), device="cpu")
    with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as checkpoint:
        load_checkpoint(model, state, checkpoint.metadata())
    del state
    model = model.to(device="cpu", dtype=torch.float32).eval()
    load_seconds = time.perf_counter() - started
    original_sizes = []
    for path in paths:
        with Image.open(path) as image:
            original_sizes.append(image.size)
    if len(set(original_sizes)) != 1:
        raise ValueError("This adapter requires identical image dimensions to avoid implicit cropping")
    started = time.perf_counter()
    images, _, _ = InputProcessor()(
        [str(path) for path in paths], process_res=resolution,
        process_res_method="upper_bound_resize", num_workers=1, sequential=True,
        print_progress=False,
    )
    preprocess_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
        raw = model(images[None].float(), infer_gs=False, use_ray_pose=False, ref_view_strategy="saddle_balanced")
        prediction = OutputProcessor()(raw)
    inference_seconds = time.perf_counter() - started
    telemetry = {"loadSeconds": load_seconds, "preprocessSeconds": preprocess_seconds,
                 "inferenceSeconds": inference_seconds, "inferenceForwardCalls": 1}
    if profile:
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
            warm_raw = model(images[None].float(), infer_gs=False, use_ray_pose=False, ref_view_strategy="saddle_balanced")
            warm_prediction = OutputProcessor()(warm_raw)
        telemetry.update({"warmInferenceSeconds": time.perf_counter() - started,
                          "inferenceForwardCalls": 2,
                          "warmMaxDepthDifference": float(np.max(np.abs(prediction.depth - warm_prediction.depth)))})
    processed = images.permute(0, 2, 3, 1).numpy()
    processed = np.clip(processed * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
    prediction.processed_images = (processed * 255).astype(np.uint8)
    return prediction, original_sizes, telemetry


def write_prediction(prediction, original_sizes, paths, output, reviewed_identity, telemetry, resolution, threads, profile):
    import numpy as np

    arrays = {"depths": prediction.depth, "confidences": prediction.conf,
              "intrinsics": prediction.intrinsics, "extrinsics": prediction.extrinsics,
              "processed_images": prediction.processed_images}
    if any(value is None for value in arrays.values()):
        raise ValueError("Model omitted required geometry output")
    count = len(paths)
    depths = np.asarray(arrays["depths"], dtype=np.float32)
    if depths.ndim != 3 or depths.shape[0] != count:
        raise ValueError("Invalid depth output shape")
    _, height, width = depths.shape
    expected_shapes = {"confidences": depths.shape, "intrinsics": (count, 3, 3),
                       "processed_images": (count, height, width, 3)}
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Invalid {name} output shape")
    if arrays["extrinsics"].shape not in ((count, 3, 4), (count, 4, 4)):
        raise ValueError("Invalid extrinsics output shape")
    for name in ("depths", "confidences", "intrinsics", "extrinsics"):
        arrays[name] = np.asarray(arrays[name], dtype=np.float32)
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Nonfinite model output: {name}")
    if not (depths > 0).all():
        raise ValueError("Depth must be positive camera-z")
    arrays["extrinsics"] = arrays["extrinsics"][:, :3, :4]
    intrinsics = arrays["intrinsics"]
    rotations = arrays["extrinsics"][:, :3, :3]
    if not ((intrinsics[:, 0, 0] > 0).all() and (intrinsics[:, 1, 1] > 0).all()):
        raise ValueError("Predicted focal lengths must be positive")
    if not np.allclose(intrinsics[:, 2, :], [0, 0, 1], atol=1e-5):
        raise ValueError("Invalid intrinsic homogeneous row")
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3):
        raise ValueError("Predicted camera rotations are not orthogonal")
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
        raise ValueError("Predicted camera rotations are not proper rotations")
    output.mkdir(parents=True, exist_ok=True)
    geometry = output / "predictions.npz"
    np.savez_compressed(geometry, **arrays)
    cameras = []
    for index, (original_width, original_height) in enumerate(original_sizes):
        cameras.append({"sourceIndex": index, "intrinsics": arrays["intrinsics"][index].tolist(),
                        "extrinsics": arrays["extrinsics"][index].tolist(), "imageWidth": width,
                        "imageHeight": height, "inputWidth": original_width, "inputHeight": original_height,
                        "pixelTransform": {"scaleX": width / original_width, "scaleY": height / original_height,
                                           "offsetX": (width / original_width - 1) / 2,
                                           "offsetY": (height / original_height - 1) / 2,
                                           "pixelCenters": "opencv-half-pixel-resize", "cropLeft": 0, "cropTop": 0}})
    result = {"protocolVersion": 1, "status": "complete", "identity": reviewed_identity,
              "parameters": {"processResolution": resolution, "processResolutionMethod": "upper_bound_resize",
                             "referenceViewStrategy": "saddle_balanced", "useRayPose": False, "threads": threads,
                             "profileWarmInference": profile},
              "coordinates": {"cameraConvention": "opencv", "extrinsics": "world-to-camera",
                              "axes": "x-right,y-down,z-forward", "depth": "camera-z", "units": "relative",
                              "confidence": "raw-uncalibrated-higher-is-better"},
              "inputs": [{"sourceIndex": index, "sha256": sha256(path)} for index, path in enumerate(paths)],
              "cameras": cameras, "geometryArtifact": str(geometry),
              "artifactSha256": sha256(geometry), "telemetry": telemetry}
    result["telemetry"]["peakMemoryBytes"] = peak_memory_bytes()
    summary = output / "worker-result.json"
    result["files"] = [str(geometry), str(summary)]
    summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def run(request):
    started = time.perf_counter()
    paths, output, resolution, threads, profile = validate_request(request)
    reviewed_identity = identity()
    if request.get("expectedIdentity", reviewed_identity) != reviewed_identity:
        raise ValueError("Runtime identity changed since parent request preparation")
    input_hashes = [sha256(path) for path in paths]
    prediction, original_sizes, telemetry = predict(paths, resolution, threads, profile)
    if input_hashes != [sha256(path) for path in paths] or identity() != reviewed_identity:
        raise ValueError("Input or runtime changed during inference")
    telemetry["totalSeconds"] = time.perf_counter() - started
    return write_prediction(prediction, original_sizes, paths, output, reviewed_identity, telemetry, resolution, threads, profile)


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
        print(f"DA3 worker failed: {error}", file=sys.stderr)
        print(json.dumps({"protocolVersion": 1, "status": "failed", "error": str(error)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
