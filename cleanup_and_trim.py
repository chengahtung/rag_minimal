# cleanup_and_trim.py
import gc
import sys
import time
import ctypes
import os

def cleanup_python_objects(names=()):
    """
    Delete globals by name (if present) and run gc.
    'names' is iterable of strings (global variable names).
    """
    g = globals()
    for n in names:
        if n in g:
            try:
                del g[n]
                print(f"deleted global {n}")
            except Exception:
                pass
    gc.collect()
    time.sleep(0.05)
    gc.collect()

def try_malloc_trim_linux():
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL("libc.so.6")
        # returns 1 on success
        res = libc.malloc_trim(0)
        return bool(res)
    except Exception as e:
        print("malloc_trim failed:", e)
        return False

def try_empty_working_set_windows():
    # Uses psapi.EmptyWorkingSet or kernel32 SetProcessWorkingSetSize to reduce working set
    if sys.platform != "win32":
        return False
    try:
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        GetCurrentProcess = kernel32.GetCurrentProcess
        GetCurrentProcess.restype = ctypes.c_void_p
        proc = GetCurrentProcess()
        # Call EmptyWorkingSet(process_handle)
        res = psapi.EmptyWorkingSet(ctypes.c_void_p(proc)) # forcefully remove as many memory pages as possible from physical RAM and page them out to disk (or discard them if unused).
        return bool(res)
    except Exception as e:
        print("EmptyWorkingSet failed:", e)
        try:
            # fallback: call SetProcessWorkingSetSize(proc, -1, -1)
            SetProcessWorkingSetSize = kernel32.SetProcessWorkingSetSize
            SetProcessWorkingSetSize.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t)
            SetProcessWorkingSetSize.restype = ctypes.c_bool
            ret = SetProcessWorkingSetSize(proc, -1, -1)
            return bool(ret)
        except Exception as e2:
            print("SetProcessWorkingSetSize fallback failed:", e2)
            return False

def best_effort_idle_release(global_names=()):
    print("[idle] starting cleanup")
    # 1) let user code close resources first (implement as needed)
    # e.g., if you have DB connections, call connection.close() here

    # 2) delete heavy globals known by name
    # cleanup_python_objects(global_names) # Commented in exchange to keep global variables intact for speed efficiency

    # 3) library-specific cleanup (PyTorch/TensorFlow)
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        # optionally: del torch
    except Exception:
        pass

    # 4) GC
    gc.collect()
    time.sleep(0.05)

    # 5) platform trim
    if sys.platform.startswith("linux"):
        ok = try_malloc_trim_linux()
        print("[idle] malloc_trim_linux:", ok)
    elif sys.platform == "win32":
        ok = try_empty_working_set_windows()
        print("[idle] empty_working_set_windows:", ok)
    else:
        print("[idle] no platform trim available")

    gc.collect()
    time.sleep(0.05)
    print("[idle] cleanup done")
