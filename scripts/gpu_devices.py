"""Resolve physical GPU selection explicitly; EGL ordinals need not match CUDA."""

import ctypes
import json
import os
import subprocess
import sys


def verify_cuda_target(physical_gpu=7, *, cuda_ordinal=0, visible_count=1):
    """Match a visible CUDA ordinal to the physical GPU's PCI address."""
    expected = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-gpu=pci.bus_id",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    cuda = ctypes.CDLL("libcuda.so.1")
    if cuda.cuInit(0):
        raise RuntimeError("CUDA device query failed")
    count = ctypes.c_int()
    if cuda.cuDeviceGetCount(ctypes.byref(count)) or count.value != visible_count:
        raise RuntimeError("visible CUDA device count differs from the launch plan")
    bus = ctypes.create_string_buffer(32)
    if cuda.cuDeviceGetPCIBusId(bus, len(bus), cuda_ordinal):
        raise RuntimeError("cannot query visible CUDA device PCI address")
    actual = bus.value.decode()

    def pci_tuple(value):
        domain, address, function = value.lower().split(":")
        return int(domain, 16), address, function

    if pci_tuple(actual) != pci_tuple(expected):
        raise RuntimeError("visible CUDA device is not the requested physical GPU")
    return {
        "gpu_physical": physical_gpu,
        "pci_bus": actual,
        "cuda_ordinal": cuda_ordinal,
    }


def configure_render_gpu(physical_gpu=7):
    """Select EGL by its CUDA device handle, then adapt robosuite's ordinal API."""
    if getattr(configure_render_gpu, "installed", False):
        return
    mapping = verify_cuda_target(physical_gpu)
    library = ctypes.CDLL("libEGL.so.1")
    library.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    library.eglGetProcAddress.restype = ctypes.c_void_p

    def function(name, prototype):
        address = library.eglGetProcAddress(name)
        if not address:
            raise RuntimeError("required EGL device query extension is unavailable")
        return prototype(address)

    query = function(
        b"eglQueryDevicesEXT",
        ctypes.CFUNCTYPE(
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
        ),
    )
    attribute = function(
        b"eglQueryDeviceAttribEXT",
        ctypes.CFUNCTYPE(
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ssize_t),
        ),
    )
    count = ctypes.c_int()
    if not query(0, None, ctypes.byref(count)) or count.value < 1:
        raise RuntimeError("cannot enumerate EGL devices")
    devices = (ctypes.c_void_p * count.value)()
    if not query(count.value, devices, ctypes.byref(count)):
        raise RuntimeError("cannot read EGL devices")
    matches = []
    for index, device in enumerate(devices):
        ordinal = ctypes.c_ssize_t(-1)
        # EGL_NV_device_cuda defines 0x323A as the CUDA handle of an EGL device.
        if attribute(device, 0x323A, ctypes.byref(ordinal)) and ordinal.value == 0:
            matches.append(index)
    if len(matches) != 1:
        raise RuntimeError("visible CUDA GPU does not map to exactly one EGL device")
    egl_index = matches[0]

    # Import while the legacy module's CUDA/EGL integer assertion still agrees.
    # Only its display creation call receives the correctly mapped EGL ordinal.
    from robosuite.renderers.context import egl_context

    original = egl_context.create_initialized_egl_device_display

    def create_display(device_id=physical_gpu):
        previous = os.environ.get("MUJOCO_EGL_DEVICE_ID")
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_index)
        try:
            return original(device_id=egl_index)
        finally:
            if previous is None:
                os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
            else:
                os.environ["MUJOCO_EGL_DEVICE_ID"] = previous

    egl_context.create_initialized_egl_device_display = create_display
    configure_render_gpu.installed = True
    print(
        json.dumps(
            {"render_device": {**mapping, "egl_index": egl_index}, "pid": os.getpid()}
        ),
        file=sys.stderr,
        flush=True,
    )
