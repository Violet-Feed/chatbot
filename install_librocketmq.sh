#!/bin/sh
set -eu

VERSION="${VERSION:-2.0.0}"
ARCH="${ARCH:-amd64}"
URL="https://github.com/apache/rocketmq-client-cpp/releases/download/${VERSION}/rocketmq-client-cpp-${VERSION}.${ARCH}.deb"
DEB="/tmp/rocketmq-client-cpp-${VERSION}.${ARCH}.deb"

echo "[1/4] Downloading: $URL"

if command -v wget >/dev/null 2>&1; then
  wget -O "$DEB" "$URL"
elif command -v curl >/dev/null 2>&1; then
  curl -fL -o "$DEB" "$URL"
else
  echo "Need wget or curl." >&2
  exit 1
fi

echo "[2/4] Installing deb via dpkg"

if dpkg -i "$DEB"; then
  echo "dpkg install succeeded."
else
  echo "dpkg install failed, fixing dependencies..."
  apt-get update
  apt-get -f install -y --no-install-recommends
fi

echo "[3/4] Running ldconfig"
ldconfig || true

echo "[4/4] Verify"
ldconfig -p | grep -i rocketmq || true

rm -f "$DEB"

echo "Done."