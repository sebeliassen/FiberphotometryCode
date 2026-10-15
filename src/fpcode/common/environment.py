"""Read environment metadata without importing optional numerical backends."""

import sys
from importlib import metadata

RECORDED_DISTRIBUTIONS = (
    "fiberphotometry-suite", "numpy", "scipy", "pandas", "h5py", "tables",
    "matplotlib", "PyYAML", "jax", "jaxlib", "jax-moseq", "keypoint-moseq",
    "numba", "opencv-python-headless", "opencv-python", "imageio", "imageio-ffmpeg",
)


def package_versions(names=RECORDED_DISTRIBUTIONS):
    """Installed distribution versions; a missing distribution is explicit null."""
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def initialized_devices():
    """Inspect existing backend clients only; never call jax.devices() to initialize one.

    JAX's cached client registry is optional implementation detail. If unavailable,
    report unknown rather than falling back to an initializing public API.
    """
    result = {"available": False, "device_name": "None", "cuda_version": "None",
              "inspection": "not_initialized", "devices": []}
    bridge = sys.modules.get("jax._src.xla_bridge")
    clients = vars(bridge).get("_backends") if bridge is not None else {}
    if not isinstance(clients, dict):
        result["inspection"] = "unknown_jax_registry"
    if isinstance(clients, dict):
        for client in tuple(clients.values()):
            try:
                devices = client.devices()
            except Exception:
                continue
            for device in devices:
                result["devices"].append({"platform": str(device.platform), "id": int(device.id),
                                          "kind": str(device.device_kind)})
    if result["devices"]:
        result["inspection"] = "initialized_jax"
        gpu = [d for d in result["devices"] if d["platform"] in {"gpu", "cuda", "rocm"}]
        result["available"] = bool(gpu)
        result["device_name"] = gpu[0]["kind"] if gpu else "None"
        return result
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            result.update(available=True, device_name=torch.cuda.get_device_name(0),
                          device_count=torch.cuda.device_count(), cuda_version=torch.version.cuda,
                          inspection="initialized_torch")
    except Exception:
        pass
    return result
