#!/usr/bin/env sh
set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"

SRC_DIR="$ROOT/src/chatbot"
PROTO_DIR="$SRC_DIR/proto"
OUT_DIR="$SRC_DIR/proto_gen"

mkdir -p "$OUT_DIR"
: > "$OUT_DIR/__init__.py"

# 1) 生成：保持 include root 为 src/chatbot，使 import "proto/common.proto" 能被解析
uv run python -m grpc_tools.protoc \
  -I "$SRC_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR/common.proto" \
  "$PROTO_DIR/im.proto"

# 2) 扁平化：把 proto_gen/proto 里的文件移到 proto_gen 根目录
if [ -d "$OUT_DIR/proto" ]; then
  # 只搬 pb2 相关文件，避免误伤其他东西
  for f in "$OUT_DIR/proto/"*_pb2.py "$OUT_DIR/proto/"*_pb2_grpc.py; do
    [ -f "$f" ] || continue
    mv "$f" "$OUT_DIR/"
  done
  # 清掉空目录（如果里面还有残留你也能一眼看到）
  rmdir "$OUT_DIR/proto" 2>/dev/null || true
fi

# 3) 修正 imports：把 "import proto.xxx_pb2" / "from proto import xxx_pb2" 改成包内相对导入
for f in "$OUT_DIR/"*_pb2.py "$OUT_DIR/"*_pb2_grpc.py; do
  [ -f "$f" ] || continue
  sed -i \
    -e 's/^import proto\.common_pb2 as /from . import common_pb2 as /' \
    -e 's/^from proto import common_pb2 as /from . import common_pb2 as /' \
    -e 's/^import proto\.im_pb2 as /from . import im_pb2 as /' \
    -e 's/^from proto import im_pb2 as /from . import im_pb2 as /' \
    "$f"
done
