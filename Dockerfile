# syntax=docker/dockerfile:1

# The base image bundles MPICH, a uv-managed Python environment with mpi4py, and
# h5py compiled against parallel HDF5. It is built for linux/amd64 and
# linux/arm64. See https://github.com/astropatty/parallel-hdf5.
#
# Tag format is mpich<major>-py<python>, e.g. mpich4-py3.13 -> MPICH 4.1.2,
# CPython 3.13, HDF5 2.1.1.
ARG BASE_IMAGE=docker.io/astropatty/parallel-h5py:mpich4-py3.13
FROM ${BASE_IMAGE}

WORKDIR /app/opencosmo
COPY . .

# opencosmo ships a compiled Rust extension (maturin / pyo3), so building it from
# source needs a C toolchain plus a Rust toolchain. Both are installed, used, and
# torn down in a single layer so they do not bloat the final image. h5py and
# mpi4py are already installed (compiled against the container's parallel HDF5
# and MPICH); uv pip leaves those satisfied requirements untouched.
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential ca-certificates curl patchelf; \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | env RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo sh -s -- -y --no-modify-path --profile minimal; \
    env RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH="/opt/cargo/bin:$PATH" \
      uv pip install .; \
    rm -rf /opt/rustup /opt/cargo; \
    apt-get purge -y --auto-remove build-essential curl patchelf; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
