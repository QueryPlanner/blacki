# The image Blacki's per-session sandboxes run.
#
# OpenSandbox injects its own execd into every sandbox, so this image only
# carries the tools: Chromium and a pinned agent-browser for the
# agent_browser tool, Python with a Jupyter server for run_code, and everyday CLI
# tools and Python libraries for shell (scripts/sandbox-image-check.sh
# lists them). Build (x86_64 or arm64, matching the sandbox host):
#
#   docker build -t blacki-sandbox:dev deploy/sandbox-image
#
# Adapted from QueryPlanner/athena commit a3ad9a5.

FROM debian:13-slim

ARG DEBIAN_FRONTEND=noninteractive

# agent-browser moves fast; bump the version and both digests together.
# Digests are the release assets' sha256 as GitHub lists them.
ARG AGENT_BROWSER_VERSION=v0.38.1
ARG AGENT_BROWSER_SHA256_AMD64=5100149a1903211c889de4e545bf36d90803740cea4f99aa22651649f9205ea1
ARG AGENT_BROWSER_SHA256_ARM64=937b315ee0761e8a62f7950ddcfef9b3d3d8e8d5eb9c9d2bf9e23e5725664511
# The usage guides `agent-browser skills get` prints ship only in the npm
# package, not in the release binaries. The digest is the tarball's sha512
# in hex; npm lists it base64-encoded as dist.integrity
# (`npm view agent-browser@<version> dist.integrity`).
ARG AGENT_BROWSER_SKILLS_SHA512=939f050b3d3250e080368364322a897be1f6cbaafab1465acea5ec585f8c62ad62442e363e3b4b701a1a83a4924ce0ff151429a6f3c3bc4e470d8121725be787
ARG TARGETARCH

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      chromium \
      curl \
      fonts-liberation \
      git \
      python3 \
      python3-pip \
      python3-venv \
      tini \
      ffmpeg \
      fd-find \
      file \
      jq \
      ripgrep \
      tree \
      unzip \
      xz-utils \
      zip \
      dnsutils \
      openssh-client \
      rsync \
      wget; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "$arch" in \
      amd64) asset=agent-browser-linux-x64; sum="$AGENT_BROWSER_SHA256_AMD64" ;; \
      arm64) asset=agent-browser-linux-arm64; sum="$AGENT_BROWSER_SHA256_ARM64" ;; \
      *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /usr/local/lib/agent-browser \
      "https://github.com/vercel-labs/agent-browser/releases/download/${AGENT_BROWSER_VERSION}/${asset}"; \
    echo "${sum}  /usr/local/lib/agent-browser" | sha256sum -c -; \
    chmod 0755 /usr/local/lib/agent-browser

# Only skill-data/ is kept from the npm tarball; the tarball is large because
# it bundles every platform's binary.
RUN set -eux; \
    version="${AGENT_BROWSER_VERSION#v}"; \
    curl -fsSL -o /tmp/agent-browser.tgz \
      "https://registry.npmjs.org/agent-browser/-/agent-browser-${version}.tgz"; \
    echo "${AGENT_BROWSER_SKILLS_SHA512}  /tmp/agent-browser.tgz" | sha512sum -c -; \
    mkdir -p /usr/local/share/agent-browser; \
    tar -xzf /tmp/agent-browser.tgz -C /usr/local/share/agent-browser \
      --strip-components=1 package/skill-data; \
    rm /tmp/agent-browser.tgz

# Jupyter in its own venv: execd's code interpreter (run_code) talks to it.
# The venv is first on PATH, so `pip install` in the shell works too (Debian
# refuses system-wide installs).
RUN set -eux; \
    python3 -m venv /opt/jupyter; \
    /opt/jupyter/bin/pip install --no-cache-dir jupyter-server==2.21.1 ipykernel==7.3.0
ENV PATH=/opt/jupyter/bin:$PATH

# Everyday Python tools and libraries, in the same venv so both the shell
# and run_code see them. Pinned: a rebuild should not pick up new versions.
# yt-dlp breaks as sites change; bump it deliberately.
RUN /opt/jupyter/bin/pip install --no-cache-dir \
      yt-dlp==2026.8.19 \
      requests==2.34.2 \
      pandas==3.0.6 \
      pillow==12.3.0 \
      beautifulsoup4==4.15.0

# Debian ships fd as fdfind; give it its usual name.
RUN ln -s /usr/bin/fdfind /usr/local/bin/fd

# agent-browser uses the system Chromium. Sandboxes run as root, which
# Chromium only allows with --no-sandbox; the OpenSandbox container is the
# isolation boundary. A wrapper, so the settings hold however execd starts
# the command.
RUN printf '%s\n' \
      '#!/bin/sh' \
      'export AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium' \
      'export AGENT_BROWSER_SKILLS_DIR=/usr/local/share/agent-browser/skill-data' \
      'export AGENT_BROWSER_ARGS="--no-sandbox,--disable-dev-shm-usage"' \
      'exec /usr/local/lib/agent-browser "$@"' \
      > /usr/local/bin/agent-browser; \
    chmod 0755 /usr/local/bin/agent-browser

COPY --chmod=0755 blacki-sandbox-start /usr/local/bin/blacki-sandbox-start

WORKDIR /workspace

# Blacki creates sandboxes with its own entrypoint, which runs
# blacki-sandbox-start when it exists; this default is for `docker run`.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/blacki-sandbox-start"]
