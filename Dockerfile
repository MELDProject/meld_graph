## Expensive calls that don't change go up top. See https://docs.docker.com/build/cache/

# freesurfer stage 
FROM debian:12-slim AS freesurfer

#Update the ubuntu.
RUN apt-get -y update && apt-get install --no-install-recommends -y wget && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#Install freesurfer in /opt/freesurfer
RUN echo "Downloading FreeSurfer..."
RUN mkdir -p /opt/freesurfer-7.2.0

# RUN wget -N -O freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz --progress=bar:force https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.2.0/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz && \
COPY freesurfer.tar.gz freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz
RUN  tar -xzf freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz -C /opt/freesurfer-7.2.0 --owner root --group root --no-same-owner --strip-components 1 --keep-newer-files \
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
          --exclude='trctrain'


# micromamba stage

FROM  debian:12-slim AS meld_git

#Update the ubuntu.
RUN apt-get -y update && apt-get install --no-install-recommends -y git ca-certificates && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /meld_graph
RUN git clone --branch dev_docker https://github.com/MELDProject/meld_graph.git .


# freesurfer stage 
FROM mambaorg/micromamba:latest AS micromamba
USER root

ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

#Update the ubuntu.
RUN apt-get -y update && apt-get install --no-install-recommends -y wget gcc g++ && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*



RUN mkdir /tmp/pkg
WORKDIR /tmp

RUN wget https://github.com/MELDProject/meld_graph/raw/dev_docker/environment.yml

# RUN --mount=type=cache,target=/opt/conda/pkgs \
RUN micromamba create -y -f environment.yml \
    && micromamba clean -afy


# meld graph stage
FROM python:3.9-slim AS MELDgraph
RUN mkdir -p /opt/freesurfer-7.2.0
COPY --from=freesurfer /opt/freesurfer-7.2.0 /opt/freesurfer-7.2.0

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
    bzip2 \
    ca-certificates \
    bc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# COPY freesurfer/ /opt/freesurfer-7.2.0

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


# COPY ./environment.yml .

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=micromamba /opt/conda/envs/meld_graph /opt/conda/envs/meld_graph

# Add meld_graph code 
RUN mkdir /app

# Define working directory
WORKDIR /app

COPY --from=meld_git /meld_graph .

ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"
RUN micromamba run -n meld_graph /bin/bash -c "pip install -e ." \
    && micromamba shell init -s bash \
    && echo "micromamba activate meld_graph" >> ~/.bashrc

# COPY ./data data
# COPY ./notebooks notebooks
# COPY ./entrypoint.sh .
# COPY ./meld_config.ini .
# COPY ./MELD_logo.png .
# COPY ./pyproject.toml .
# COPY ./pytest.ini .
# COPY ./setup.py .

# Activate environment with shell because not working wih conda
# SHELL ["conda", "run", "-n", "meld_graph", "/bin/bash", "-c"]

# COPY ./meld_graph /app/meld_graph

# Install meld_graph package
# RUN conda run -n meld_graph /bin/bash -c "pip install -e ."

# COPY ./scripts /app/scripts

# Add data folder to docker
RUN mkdir /data

# Create a cache directory for fastsurfer, otherwise permission denied
RUN mkdir /.cache
RUN chmod -R 777 /.cache

# Set permissions for the entrypoint
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/bin/bash","entrypoint.sh"]
