"""Generated gRPC and protobuf bindings for ARDA Reliability Service."""

import sys
from pathlib import Path

# Ensure generated package namespace is resolvable
_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from arda.v1 import reliability_pb2, reliability_pb2_grpc

__all__ = ["reliability_pb2", "reliability_pb2_grpc"]
