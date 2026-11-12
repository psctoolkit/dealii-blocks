#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
dealii_build=${DEAL_II_DIR:-"$repo_dir/build"}
stokes_build=${STOKES_BUILD_DIR:-"$repo_dir/build-stokes"}

printf 'Selected DEAL_II_DIR is %s\n' "$dealii_build"

cmake \
  -S "$repo_dir" \
  -B "$stokes_build" \
  "-DDEAL_II_DIR:PATH=$dealii_build" \
  "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE:-Release}"

cmake --build "$stokes_build" \
  --target stokes \
  --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
