# Separate experimental image; never replaces the default production Dockerfile.
FROM ghcr.io/enthus-appdev/vllm-pcai@sha256:64436a4007db5a10d0aeba10cec0ff5ef4560eed39cbd00db57e1adff366c69f
USER root
ENV TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*
# Retain complete sources and notices for the MIT and AGPL-3.0 dependencies.
RUN mkdir -p /opt/pcai-sources && \
    git clone https://github.com/vcruz305/exllamav3.git /opt/pcai-sources/exllamav3 && \
    git -C /opt/pcai-sources/exllamav3 checkout e43b074096a206a651f18107ee5eb025eb40beb2 && \
    git clone https://github.com/vcruz305/vllm-exl3.git /opt/pcai-sources/vllm-exl3 && \
    git -C /opt/pcai-sources/vllm-exl3 checkout 053c634db54afc981becfb46c0482e7358cc00ba
RUN python3 -m pip install --no-cache-dir ninja setuptools wheel marisa-trie && \
    export CPATH="$(python3 -c 'import glob,site; print(":".join(p for s in site.getsitepackages() for p in glob.glob(s+"/nvidia/*/include")))')" && \
    export LIBRARY_PATH="$(python3 -c 'import glob,site; print(":".join(p for s in site.getsitepackages() for p in glob.glob(s+"/nvidia/*/lib")))')" && \
    python3 -c 'import glob,site; p=[p for s in site.getsitepackages() for p in glob.glob(s+"/nvidia/*/include/cusparse.h")]; print("CUDA sparse headers:",p); assert p' && \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps /opt/pcai-sources/exllamav3 && \
    EXL3_EXT_INCLUDE=/opt/pcai-sources/exllamav3/exllamav3/exllamav3_ext \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps /opt/pcai-sources/vllm-exl3
COPY experimental/eaq /opt/pcai-sources/pcai-eaq
RUN python3 -m pip install --no-cache-dir --no-build-isolation --no-deps /opt/pcai-sources/pcai-eaq && \
    python3 -c 'import torch; import exllamav3_ext; import vllm_exl3_c; import pcai_eaq; pcai_eaq.check_compat()' && \
    python3 -m unittest discover -s /opt/pcai-sources/pcai-eaq/tests -v
# Activation is opt-in through PCAI_ENGRAM_DIR. Preserve the base entrypoint, but not the
# build user: the image runs as a non-root user (PCAI enforces its own UID at runtime).
USER 10000004:1001
LABEL org.opencontainers.image.description="Experimental H100 EAQ v1.1 qualification image; not GPU-qualified"
