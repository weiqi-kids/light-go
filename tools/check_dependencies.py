"""Check if all required dependencies are installed and working.

Run this script to verify your Light-Go environment is set up correctly.
"""

import sys
from typing import Tuple, List


def check_module(name: str, min_version: str = None) -> Tuple[bool, str]:
    """Check if a module is installed and optionally verify version.

    Parameters
    ----------
    name : str
        Module name to import
    min_version : str, optional
        Minimum required version

    Returns
    -------
    Tuple[bool, str]
        (success, message)
    """
    try:
        module = __import__(name)
        version = getattr(module, '__version__', 'unknown')

        if min_version and version != 'unknown':
            # Simple version comparison (works for major.minor.patch)
            try:
                current = tuple(map(int, version.split('.')[:2]))
                required = tuple(map(int, min_version.split('.')[:2]))
                if current < required:
                    return False, f"❌ {name} {version} (need >={min_version})"
            except:
                pass  # Skip version check if parsing fails

        return True, f"✅ {name} {version}"

    except ImportError:
        return False, f"❌ {name} (not installed)"


def check_pytorch_cuda():
    """Check PyTorch CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "N/A"
            return True, f"✅ CUDA {cuda_version} ({gpu_count} GPU(s): {gpu_name})"
        else:
            return True, "⚠️  CUDA not available (CPU only)"
    except ImportError:
        return False, "❌ PyTorch not installed"


def check_light_go_modules():
    """Check if Light-Go modules can be imported."""
    modules_to_check = [
        ('hf_models.modeling_go_ai', 'GoAIModel'),
        ('hf_models.board_encoder', 'BoardEncoder'),
        ('core.architecture_genome', 'ArchitectureGenome'),
        ('core.game_rules', 'GoBoard'),
        ('core.trainer', 'ModelTrainer'),
        ('core.engine', 'Engine'),
    ]

    results = []
    for module_name, class_name in modules_to_check:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            results.append((True, f"✅ {module_name}.{class_name}"))
        except ImportError as e:
            results.append((False, f"❌ {module_name} (import error: {e})"))
        except AttributeError:
            results.append((False, f"❌ {module_name}.{class_name} (class not found)"))

    return results


def main():
    """Run all dependency checks."""
    print("=" * 70)
    print("  Light-Go Dependency Check")
    print("=" * 70)

    # Check Python version
    print(f"\n🐍 Python Version: {sys.version.split()[0]}")
    if sys.version_info < (3, 8):
        print("   ⚠️  Warning: Python 3.8+ recommended")

    # Check core dependencies
    print("\n📦 Core Dependencies:")
    dependencies = [
        ('numpy', '1.21'),
        ('torch', '2.0'),
        ('sgfmill', '1.1'),
    ]

    all_core_ok = True
    for name, min_ver in dependencies:
        ok, msg = check_module(name, min_ver)
        print(f"   {msg}")
        all_core_ok = all_core_ok and ok

    # Check optional dependencies
    print("\n📦 Optional Dependencies:")
    optional = [
        ('transformers', None),
        ('qdrant_client', None),
        ('flask', None),
        ('gradio', None),
        ('ray', None),
    ]

    for name, min_ver in optional:
        ok, msg = check_module(name, min_ver)
        print(f"   {msg}")

    # Check PyTorch CUDA
    print("\n🎮 GPU Support:")
    ok, msg = check_pytorch_cuda()
    print(f"   {msg}")

    # Check Light-Go modules
    print("\n🔧 Light-Go Modules:")
    module_results = check_light_go_modules()
    all_modules_ok = True
    for ok, msg in module_results:
        print(f"   {msg}")
        all_modules_ok = all_modules_ok and ok

    # Summary
    print("\n" + "=" * 70)
    if all_core_ok and all_modules_ok:
        print("  ✅ All checks passed! You're ready to use Light-Go.")
        print("\n  Try running the demo:")
        print("     python examples/minimal_self_evolution.py")
    else:
        print("  ❌ Some checks failed. Please install missing dependencies:")
        print("\n  Install all dependencies:")
        print("     pip install -r requirements.txt")
        print("\n  Or install minimal dependencies:")
        print("     pip install torch>=2.0.0 numpy>=1.21.0 sgfmill>=1.1")

    print("=" * 70)


if __name__ == "__main__":
    main()
