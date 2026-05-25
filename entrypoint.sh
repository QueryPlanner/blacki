#!/bin/sh
set -e

# Ensure directories exist and are owned by the 'app' user
# This fixes permission issues if Docker created the bind mounts as root
mkdir -p /app/logs /app/src/.adk
chown -R app:app /app/logs /app/src/.adk

# Drop root privileges and execute the main command as the 'app' user
exec runuser -u app -- "$@"
