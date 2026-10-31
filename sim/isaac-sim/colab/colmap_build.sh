#!/bin/bash
# COLMAP 4.2.1 with CUDA (PatchMatch) on reconB while the GPU queue runs; niced so training keeps its CPU.
# Usage: nohup bash colmap_build.sh > /content/colmap_build.log 2>&1 &   (ends COLMAP_DONE <status>)
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get install -y -qq git cmake ninja-build build-essential libboost-program-options-dev libboost-graph-dev \
  libboost-system-dev libboost-filesystem-dev libeigen3-dev libfreeimage-dev libopenimageio-dev openimageio-tools \
  libmetis-dev libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev libcgal-dev libceres-dev \
  libsuitesparse-dev libcurl4-openssl-dev libssl-dev libflann-dev libopencv-dev > /content/apt_colmap.log 2>&1
cd /content && rm -rf colmap && git clone -q --depth 1 --branch 4.2.1 https://github.com/colmap/colmap.git && mkdir -p colmap/build && cd colmap/build && \
  cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DGUI_ENABLED=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed -lGLdispatch" > /content/cmake_colmap.log 2>&1 && \
  nice -n 15 ninja -j6 > /content/ninja_colmap.log 2>&1 && ninja install > /dev/null 2>&1
echo "COLMAP_DONE $?"
