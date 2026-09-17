#!/usr/bin/env python3
"""
Fix CUDA and cuDNN dynamic library resolution for ONNX Runtime GPU on Linux / Google Colab.
Inspects libonnxruntime_providers_cuda.so with ldd and automatically binds any missing
versioned libraries (e.g. libcudart.so.13) to available system CUDA libraries.
"""

import os
import sys
import glob
import re
import site
import subprocess

def fix_cuda_libraries():
    if sys.platform != "linux":
        print("[*] Non-Linux platform detected; skipping Linux CUDA linker setup.")
        return

    sp_paths = site.getsitepackages()
    wheel_lib_paths = []
    for sp in sp_paths:
        wheel_lib_paths.extend(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
        wheel_lib_paths.extend(glob.glob(os.path.join(sp, "torch", "lib")))

    system_cuda_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-12/targets/x86_64-linux/lib",
        "/usr/lib64-nvidia",
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    all_search_paths = [p for p in set(wheel_lib_paths + system_cuda_paths) if os.path.isdir(p)]

    # Locate onnxruntime providers cuda library
    ort_cuda_libs = glob.glob("/usr/local/lib/python3*/dist-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so")
    for sp in sp_paths:
        ort_cuda_libs.extend(glob.glob(os.path.join(sp, "onnxruntime", "capi", "libonnxruntime_providers_cuda.so")))

    if ort_cuda_libs and os.path.isfile(ort_cuda_libs[0]):
        try:
            ldd_out = subprocess.check_output(["ldd", ort_cuda_libs[0]], text=True, stderr=subprocess.STDOUT)
            missing = [line.split("=>")[0].strip() for line in ldd_out.splitlines() if "not found" in line]
            if missing:
                print(f"[+] Detected missing libraries via ldd: {missing}", flush=True)
                for lib in missing:
                    base = re.sub(r"\.so.*", "", lib)
                    candidates = []
                    for sdir in all_search_paths:
                        candidates.extend(glob.glob(os.path.join(sdir, f"{base}*.so*")))
                    candidates = [c for c in candidates if not os.path.islink(c) and os.path.isfile(c) and not c.endswith(".a")]
                    if candidates:
                        candidates.sort(reverse=True)
                        target = candidates[0]
                        link_path = os.path.join("/usr/local/lib", lib)
                        if os.path.exists(link_path) or os.path.islink(link_path):
                            os.remove(link_path)
                        os.symlink(target, link_path)
                        print(f"[+] Created symlink: {link_path} -> {target}", flush=True)
                    else:
                        print(f"[!] Warning: No candidate file found for {lib}", flush=True)
            else:
                print("[+] Zero missing libraries detected for ONNX Runtime CUDA provider.", flush=True)
        except Exception as e:
            print(f"[!] Linker check note: {e}", flush=True)

    # Register all library directories into dynamic linker cache
    try:
        with open("/etc/ld.so.conf.d/cuda_ort.conf", "w") as f:
            f.write("/usr/local/lib\n")
            for p in all_search_paths:
                f.write(p + "\n")
        subprocess.run(["ldconfig"], check=False)
        print("[+] Dynamic linker cache updated successfully via ldconfig.", flush=True)
    except Exception as e:
        print(f"[!] ldconfig note: {e}", flush=True)

if __name__ == "__main__":
    fix_cuda_libraries()
