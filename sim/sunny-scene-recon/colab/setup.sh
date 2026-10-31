#!/bin/bash
# One-time Colab VM setup (L4, Ubuntu 24.04, CUDA 13): COLMAP 4.2.1 with CUDA, the ETH3D evaluator, the Python
# packages for build_scene/walk_in_scene, and vid2scene's splat trainer. Takes about 40 min, mostly the COLMAP build.
# Usage: bash setup.sh [SCENE ...]   (ETH3D DSLR scenes to download, e.g. pipes delivery_area)
set -x
export DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=10
apt-get update -qq
apt-get install -y -qq git cmake ninja-build build-essential p7zip-full \
  libboost-program-options-dev libboost-graph-dev libboost-system-dev libboost-filesystem-dev \
  libeigen3-dev libfreeimage-dev libopenimageio-dev openimageio-tools libmetis-dev libgoogle-glog-dev \
  libgtest-dev libgmock-dev libsqlite3-dev libglew-dev libcgal-dev libceres-dev libsuitesparse-dev \
  libcurl4-openssl-dev libssl-dev libflann-dev libopencv-dev libpcl-dev ffmpeg > /content/apt.log 2>&1
pip install -q open3d mujoco coacd scikit-learn plyfile cloth-simulation-filter > /content/pip.log 2>&1

mkdir -p /content/eth3d
for SCENE in "$@"; do
  (cd /content/eth3d && wget -q https://www.eth3d.net/data/${SCENE}_dslr_undistorted.7z https://www.eth3d.net/data/${SCENE}_dslr_scan_eval.7z && \
    7z x -y ${SCENE}_dslr_undistorted.7z > /dev/null && 7z x -y ${SCENE}_dslr_scan_eval.7z > /dev/null; echo "DATA_${SCENE} $?") &
done

# ETH3D evaluator: needs PCL and C++17
cd /content && git clone -q https://github.com/ETH3D/multi-view-evaluation.git && cd multi-view-evaluation && \
  sed -i 's/-std=c++11/-std=c++17/g; s/CMAKE_CXX_STANDARD 11/CMAKE_CXX_STANDARD 17/g' CMakeLists.txt && mkdir -p build && cd build && \
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 > /content/eval_cmake.log 2>&1 && make -j4 > /content/eval_make.log 2>&1
echo "EVAL_DONE $?"

# COLMAP 4.2.1 (PatchMatch stereo needs CUDA; GLOMAP's global_mapper is built in). The linker flag fixes _glapi_tls_Current.
cd /content && git clone -q --depth 1 --branch 4.2.1 https://github.com/colmap/colmap.git && mkdir -p colmap/build && cd colmap/build && \
  cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DGUI_ENABLED=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed -lGLdispatch" > /content/cmake.log 2>&1 && \
  ninja -j10 > /content/ninja.log 2>&1 && ninja install > /dev/null 2>&1
echo "COLMAP_DONE $?"

# vid2scene's splat stack: its gsplat fork plus the CUDA extras pinned in its Worker_Dockerfile
cd /content && git clone -q --recursive https://github.com/samuelm2/gsplat v2s_gsplat
pip install -q git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5 --no-build-isolation
pip install -q git+https://github.com/harry7557558/fused-bilagrid@49f0ef06c9f81810fb9b5dd9027cf1844950cc16 --no-build-isolation
grep -v -E "^numpy|^#|^$" v2s_gsplat/examples/requirements.txt > /content/v2s_req.txt && pip install -q -r /content/v2s_req.txt
(cd v2s_gsplat && pip install -q --no-build-isolation .)
touch v2s_gsplat/examples/datasets/__init__.py  # Colab's HuggingFace `datasets` package otherwise shadows it
sed -i -E 's/np\.(uint[0-9]+)\(-1\)/np.iinfo(np.\1).max/g' /usr/local/lib/python3.13/dist-packages/pycolmap_parser/*.py  # numpy 2
python -c "import gsplat, fused_ssim, fused_bilagrid; print('SPLATS_OK', gsplat.__version__)"
wait
echo ALL_DONE
