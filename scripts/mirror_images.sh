#!/usr/bin/env bash
# Mirror the TIG dev images from GHCR to the Talos Docker Hub namespace. Maintainers run this
# once per DEV_IMAGE_TAG; Talos users never touch GHCR. Namespace and tag come from
# talos/challenges.py so there is one source of truth.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}  # needs talos importable; system python3 does not have it
TAG=$($PY -c 'from talos.challenges import DEV_IMAGE_TAG; print(DEV_IMAGE_TAG)')
NS=$($PY -c 'from talos.challenges import image_namespace; print(image_namespace())')
if [ $# -gt 0 ]; then CHALLENGES="$*"; else
  CHALLENGES=$($PY -c 'from talos.challenges import CHALLENGES; print(" ".join(CHALLENGES))')
fi
for ch in $CHALLENGES; do
  src="ghcr.io/tig-foundation/tig-monorepo/${ch}/dev:${TAG}"
  dst="docker.io/${NS}/tig-${ch}-dev:${TAG}"
  echo "== ${src} -> ${dst}"
  docker pull "${src}"
  docker tag "${src}" "${dst}"
  docker push "${dst}"
done
echo "done; verify with: curl -s https://hub.docker.com/v2/repositories/${NS}/tig-knapsack-dev/tags/${TAG} | head -c 200"
