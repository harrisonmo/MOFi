FROM python:3.11-slim
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir numpy==2.2.6 ase==3.26.0 networkx==3.5 scipy==1.15.3
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
WORKDIR /work
USER 65534:65534
