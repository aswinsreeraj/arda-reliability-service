"""Generate Python gRPC and protobuf bindings from reliability.proto."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def generate_protos() -> None:
    """Invoke grpc_tools.protoc to compile reliability.proto into Python packages."""
    repo_root = Path(__file__).resolve().parent.parent
    proto_root = Path("/home/aswin/second-project/arda/proto")
    out_dir = (
        repo_root
        / "src"
        / "arda_reliability_service"
        / "infrastructure"
        / "grpc"
        / "generated"
    )

    if not proto_root.exists():
        raise FileNotFoundError(f"Protobuf source directory not found: {proto_root}")

    proto_file = proto_root / "arda" / "v1" / "reliability.proto"
    if not proto_file.exists():
        raise FileNotFoundError(f"reliability.proto not found at: {proto_file}")

    target_subpkg = out_dir / "arda" / "v1"
    target_subpkg.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_root}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        f"--pyi_out={out_dir}",
        str(proto_file),
    ]

    print("Running protoc command:")
    print(" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        print("Protoc compilation failed:", file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        sys.exit(res.returncode)

    # Ensure intermediate packages have __init__.py
    (out_dir / "arda" / "__init__.py").touch()
    (out_dir / "arda" / "v1" / "__init__.py").touch()

    # Create root __init__.py in generated/ with sys.path configuration
    root_init = out_dir / "__init__.py"
    root_init.write_text(
        '"""Generated gRPC and protobuf bindings for ARDA Reliability Service."""\n\n'
        "import sys\n"
        "from pathlib import Path\n\n"
        "# Ensure generated package namespace is resolvable\n"
        "_PKG_DIR = Path(__file__).resolve().parent\n"
        "if str(_PKG_DIR) not in sys.path:\n"
        "    sys.path.insert(0, str(_PKG_DIR))\n\n"
        "from arda.v1 import reliability_pb2, reliability_pb2_grpc\n\n"
        '__all__ = ["reliability_pb2", "reliability_pb2_grpc"]\n',
        encoding="utf-8",
    )

    print(f"Protobuf bindings successfully generated in {out_dir}")


if __name__ == "__main__":
    generate_protos()
