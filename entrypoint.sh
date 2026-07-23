#!/bin/sh
set -e

# Ensure persistent bind mounts exist and are writable by the runtime user.
mkdir -p /app/data /app/logs /app/src/.adk
chown -R app:app /app/data /app/logs /app/src/.adk

# Drop root privileges and execute the main command as the 'app' user
exec runuser -u app -- "$@"
