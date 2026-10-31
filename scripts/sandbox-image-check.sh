#!/usr/bin/env bash
# Checks that every tool the sandbox image promises actually runs in it.
#
#   scripts/sandbox-image-check.sh [IMAGE]    # default blacki-sandbox:dev
#
# Build first: docker build -t blacki-sandbox:dev deploy/sandbox-image
set -euo pipefail

image="${1:-blacki-sandbox:dev}"

# One command per tool; each must exit 0.
checks=(
  "agent-browser --version"
  "agent-browser skills get core | grep -q 'name: core'"
  "chromium --version"
  "git --version"
  "curl --version"
  "python3 --version"
  "jupyter-server --version"
  "ffmpeg -version"
  "yt-dlp --version"
  "fd --version"
  "file --version"
  "jq --version"
  "rg --version"
  "tree --version"
  "unzip -v"
  "xz --version"
  "zip -v"
  "dig -v"
  "ssh -V"
  "rsync --version"
  "wget --version"
  "python3 -c 'import requests, pandas, PIL, bs4'"
)

failed=0
for check in "${checks[@]}"; do
  if docker run --rm --entrypoint /bin/sh "$image" -c "$check" >/dev/null 2>&1; then
    echo "ok    $check"
  else
    echo "FAIL  $check"
    failed=1
  fi
done

if docker run --rm --entrypoint /bin/sh "$image" -c \
  "agent-browser open 'data:text/html,<title>Blacki sandbox</title>' >/dev/null && \
   agent-browser get title | grep -q 'Blacki sandbox' && agent-browser close >/dev/null"; then
  echo "ok    browser opens a page"
else
  echo "FAIL  browser opens a page"
  failed=1
fi

# Exercise the same startup command and Jupyter token used by Blacki.
container=$(docker run -d --entrypoint blacki-sandbox-start \
  -e JUPYTER_TOKEN=sandbox-image-test "$image")
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
if docker exec "$container" python3 -c '
import time
import urllib.request

request = urllib.request.Request(
    "http://127.0.0.1:44771/api", headers={"Authorization": "token sandbox-image-test"}
)
for attempt in range(30):
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            assert response.status == 200
        break
    except OSError:
        if attempt == 29:
            raise
        time.sleep(1)
'; then
  echo "ok    authenticated Jupyter starts"
else
  echo "FAIL  authenticated Jupyter starts"
  failed=1
fi

if docker exec "$container" python3 -c '
from jupyter_client import KernelManager

manager = KernelManager()
manager.start_kernel()
client = manager.client()
client.start_channels()
try:
    client.wait_for_ready(timeout=30)
    request_id = client.execute("print(6 * 7)")
    while True:
        message = client.get_iopub_msg(timeout=30)
        if message["parent_header"].get("msg_id") != request_id:
            continue
        if message["msg_type"] == "stream" and message["content"]["text"].strip() == "42":
            break
        if message["msg_type"] == "error":
            raise RuntimeError(message["content"])
        if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
            raise RuntimeError("Kernel did not produce the expected result")
finally:
    client.stop_channels()
    manager.shutdown_kernel(now=True)
'; then
  echo "ok    Jupyter executes Python"
else
  echo "FAIL  Jupyter executes Python"
  failed=1
fi

exit "$failed"
