FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ARG PRISM_LLAMA_COMMIT=7529fdaaf99ffdc5ca71ace9c7409a56b27ad92f
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git cmake build-essential python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --branch prism --depth 1 https://github.com/PrismML-Eng/llama.cpp.git prism-llama.cpp \
    && cd prism-llama.cpp \
    && test "$(git rev-parse HEAD)" = "${PRISM_LLAMA_COMMIT}"
COPY dspark-long-context-cap.patch /tmp/dspark-long-context-cap.patch
COPY yarn-long-context.patch /tmp/yarn-long-context.patch
RUN cd /opt/prism-llama.cpp \
    && git apply --check /tmp/dspark-long-context-cap.patch \
    && git apply /tmp/dspark-long-context-cap.patch \
    && git apply --check /tmp/yarn-long-context.patch \
    && git apply /tmp/yarn-long-context.patch

WORKDIR /opt/prism-llama.cpp
# BuildKit does not expose the host NVIDIA driver during image builds. Link
# against CUDA's driver stub explicitly; the NVIDIA container runtime injects
# the real libcuda.so.1 when the capsule is started with --gpus.
RUN ln -sf libcuda.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1 \
    && LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs \
       LDFLAGS="-Wl,-rpath-link,/usr/local/cuda/targets/x86_64-linux/lib/stubs" \
       cmake -B build \
         -DGGML_CUDA=ON \
         -DCMAKE_CUDA_ARCHITECTURES=86 \
         -DCMAKE_BUILD_TYPE=Release \
    && LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs \
       LDFLAGS="-Wl,-rpath-link,/usr/local/cuda/targets/x86_64-linux/lib/stubs" \
       cmake --build build --config Release -j"$(nproc)" --target llama-server

COPY entrypoint.sh /usr/local/bin/prism-bonsai
RUN chmod 755 /usr/local/bin/prism-bonsai

ENV MODEL_DIR=/models \
    MODEL=/models/Ternary-Bonsai-27B-Q2_0.gguf \
    PORT=11611 \
    CTX=65536 \
    MAIN_GPU=0 \
    NGL=99 \
    THREADS=32

EXPOSE 11611
ENTRYPOINT ["/usr/local/bin/prism-bonsai"]
