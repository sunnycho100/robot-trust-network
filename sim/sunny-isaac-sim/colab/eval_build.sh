#!/bin/bash
# ETH3D multi-view evaluator (needs PCL and C++17), same recipe as sim/sunny-scene-recon/colab/setup.sh
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq cmake libpcl-dev libeigen3-dev p7zip-full > /content/apt_eval.log 2>&1
cd /content && rm -rf multi-view-evaluation && git clone -q https://github.com/ETH3D/multi-view-evaluation.git && cd multi-view-evaluation && \
  sed -i 's/-std=c++11/-std=c++17/g; s/CMAKE_CXX_STANDARD 11/CMAKE_CXX_STANDARD 17/g' CMakeLists.txt && mkdir -p build && cd build && \
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 > /content/eval_cmake.log 2>&1 && make -j8 > /content/eval_make.log 2>&1
echo "EVAL_DONE $?"
