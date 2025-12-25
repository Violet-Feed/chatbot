#!/bin/sh
set -eu

VERSION="${VERSION:-2.0.0}"
ARCH="${ARCH:-amd64}"   # amd64 / arm64(若官方有对应deb)；无deb则需源码编译
URL="https://github.com/apache/rocketmq-client-cpp/releases/download/${VERSION}/rocketmq-client-cpp-${VERSION}.${ARCH}.deb"
DEB="rocketmq-client-cpp-${VERSION}.${ARCH}.deb"

echo "[1/5] Downloading: $URL"
if command -v wget >/dev/null 2>&1; then
  wget -O "$DEB" "$URL"
elif command -v curl >/dev/null 2>&1; then
  curl -L -o "$DEB" "$URL"
else
  echo "Need wget or curl." >&2
  exit 1
fi

echo "[2/5] Installing deb via dpkg"
sudo dpkg -i "$DEB" || true

echo "[3/5] Fixing dependencies (apt-get -f install)"
sudo apt-get update -y
sudo apt-get -f install -y

echo "[4/5] Running ldconfig"
sudo ldconfig

echo "[5/5] Verify"
ldconfig -p | grep -i rocketmq || true
echo "Done. If grep shows librocketmq, installation is OK."
