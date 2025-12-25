#!/usr/bin/env bash
set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"
PROTO_DIR="$ROOT/src/chatbot/proto"
OUT_DIR="$ROOT/src/chatbot/proto_gen"

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/__init__.py"

uv run python -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR/common.proto" \
  "$PROTO_DIR/im.proto"

  # 把 proto_gen 下所有 “import xxx_pb2 as” 改成 “from . import xxx_pb2 as”
sed -i -E 's/^import ([a-zA-Z0-9_]+_pb2) as/from . import \1 as/g' src/chatbot/proto_gen/*_pb2*.py

# 特殊：你的 common_pb2 不是 *_pb2?（它是 common_pb2，符合上面规则），一般不用额外处理

