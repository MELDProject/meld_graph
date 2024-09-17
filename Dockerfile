## Expensive calls that don't change go up top. See https://docs.docker.com/build/cache/

# freesurfer stage 
FROM mambaorg/micromamba:latest AS micromamba
USER root

ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

#Update ubuntu.
RUN apt-get -y update && apt-get install --no-install-recommends -y wget gcc g++ && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir /tmp/pkg
WORKDIR /tmp

COPY ./environment.yml ./environment.yml

# Create the meld_graph environment
RUN micromamba create -y -f environment.yml \
    && micromamba clean -afy


# meld graph stage
FROM debian:12-slim AS MELDgraph
RUN mkdir -p /opt/freesurfer-7.2.0

#Update ubuntu.
RUN apt-get -y update && apt-get install --no-install-recommends -y wget && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Download freesurfer
RUN wget -N -O freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz --no-check-certificate --progress=bar:force https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.2.0/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz && \
    tar -xzf freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz -C /opt/freesurfer-7.2.0 --owner root --group root --no-same-owner --strip-components 1 --keep-newer-files \
    --exclude='average/mult-comp-cor' \
    --exclude='lib/cuda' \
    --exclude='lib/qt' \
    --exclude='subjects/V1_average' \
    --exclude='subjects/bert' \
    --exclude='subjects/cvs_avg35' \
    --exclude='subjects/cvs_avg35_inMNI152' \
    --exclude='subjects/fsaverage3' \
    --exclude='subjects/fsaverage4' \
    --exclude='subjects/fsaverage5' \
    --exclude='subjects/fsaverage6' \
    --exclude='trctrain' && \
    rm freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz

ENV DEBIAN_FRONTEND="noninteractive"

#Install the prerequisite software
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    time \
    wget \
    git \
    tcsh \
    vim \
    csh \
    procps \
    bzip2 \
    ca-certificates \
    bc \
    python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# #Modify the environment with Freesurfer paths
ENV PATH=/opt/freesurfer-7.2.0/bin:$PATH
RUN echo "PATH=/opt/freesurfer-7.2.0/bin:$PATH" >> ~/.bashrc
ENV FREESURFER_HOME=/opt/freesurfer-7.2.0
RUN echo "FREESURFER_HOME=/opt/freesurfer-7.2.0" >> ~/.bashrc
RUN echo "FS_LICENSE=/license.txt" >> ~/.bashrc

# Install Fastsurfer
RUN  mkdir -p /fastsurfer \
&& git clone --branch v1.1.2 https://github.com/Deep-MI/FastSurfer.git /opt/fastsurfer-v1.1.2 
RUN echo "export PYTHONPATH=\"\${PYTHONPATH}:$PWD\"" >> ~/.bashrc
ENV FASTSURFER_HOME=/opt/fastsurfer-v1.1.2
RUN echo "FASTSURFER_HOME=/opt/fastsurfer-v1.1.2" >> ~/.bashrc

# Add conda to path
ENV CONDA_DIR /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Copy the micromamba bin and env
COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=micromamba /opt/conda/envs/meld_graph /opt/conda/envs/meld_graph

RUN mkdir /app

# Define working directory
WORKDIR /app

# Add meld_graph code 
COPY . .

ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"
RUN micromamba run -n meld_graph /bin/bash -c "pip install -e ." \
    && micromamba shell init -s bash \
    && echo "micromamba activate meld_graph" >> $HOME/.bashrc

ENV PATH="/opt/conda/envs/meld_graph/bin:$PATH"

# Add data folder to docker
RUN mkdir /data

# Create a cache directory for fastsurfer, otherwise permission denied
RUN mkdir /.cache
RUN chmod -R 777 /.cache

# Create a cache directory for freesurfer, otherwise permission denied
RUN mkdir /matlab
RUN chmod -R 777 /matlab

# Set permissions for the entrypoint
RUN chmod +x entrypoint.sh

ENV KEEP_DATA_PATH=1
ENV SILENT=1

ENTRYPOINT ["/bin/bash","entrypoint.sh"]
