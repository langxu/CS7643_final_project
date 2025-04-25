import subprocess
import shutil
import site
import os
import sys

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("ERROR:", result.stderr)
    return result

def uninstall_pillow():
    run_cmd("pip uninstall -y Pillow")

def delete_residual_pillow():
    site_packages_dirs = site.getsitepackages() + [site.getusersitepackages()]
    for dir_path in site_packages_dirs:
        pil_path = os.path.join(dir_path, "PIL")
        if os.path.exists(pil_path):
            print(f"Deleting leftover: {pil_path}")
            shutil.rmtree(pil_path, ignore_errors=True)

def clear_pip_cache():
    run_cmd("pip cache purge")

def install_pillow(version="8.3.1"):
    run_cmd(f"pip install Pillow=={version}")

def verify_install():
    try:
        import PIL
        print(f"\n✅ Pillow version: {PIL.__version__} loaded successfully!")
    except ImportError as e:
        print("❌ Still failed to import Pillow.")
        print(e)

if __name__ == "__main__":
    uninstall_pillow()
    delete_residual_pillow()
    clear_pip_cache()
    install_pillow()
    verify_install()
