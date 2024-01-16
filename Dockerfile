FROM ubuntu:jammy AS MELDgraph

ENV DEBIAN_FRONTEND="noninteractive"
ARG CONDA_FILE=Miniconda3-py38_4.11.0-Linux-x86_64.sh

#pdate the ubuntu.
RUN apt-get -y update && apt-get -y upgrade

#Install the prerequisite software
RUN apt-get install -y build-essential \
    apt-utils \
    vim \
    nano \
    curl \
    wget \
    pip \ 
    python3 \
    git \
    time \
	csh \
	tcsh \
    bc

#Install freesurfer in /opt/freesurfer
#TODO: need to get freesurfer from wget
RUN echo "Downloading FreeSurfer ..." \
   && mkdir -p /opt/freesurfer-7.2.0 \
   && curl -fL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.2.0/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz \
    | tar -xz -C /opt/freesurfer-7.2.0 --owner root --group root --no-same-owner --strip-components 1 \
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

RUN rm -f /opt/freesurfer-7.2.0/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz

# #Modify the environment
ENV PATH=/opt/freesurfer-7.2.0/bin:$PATH
RUN echo "PATH=/opt/freesurfer-7.2.0/bin:$PATH" >> ~/.bashrc
ENV FREESURFER_HOME=/opt/freesurfer-7.2.0
RUN echo "FREESURFER_HOME=/opt/freesurfer-7.2.0" >> ~/.bashrc

# # #TODO: get license from somewhere else
COPY license.txt ${FREESURFER_HOME}/license.txt

# Install miniconda
RUN wget --no-check-certificate -qO ~/miniconda.sh https://repo.continuum.io/miniconda/$CONDA_FILE  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh 

# Add conda to path
ENV CONDA_DIR /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Update conda
RUN conda update -n base -c defaults conda
RUN conda init bash

# Install Fastsurfer
RUN mkdir -p /fastsurfer \
&& git clone --branch v1.1.2 https://github.com/Deep-MI/FastSurfer.git /opt/fastsurfer-v1.1.2 
RUN echo "export PYTHONPATH=\"\${PYTHONPATH}:$PWD\"" >> ~/.bashrc
ENV FASTSURFER_HOME=/opt/fastsurfer-v1.1.1
RUN echo "FASTSURFER_HOME=/opt/fastsurfer-v1.1.2" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]


#add meld_graph code 
COPY . /app/
# #OR checkout and install the github repo
# RUN cd / && git clone https://github.com/MELDProject/meld_graph.git
# update current conda base environment with packages for meld_graph 
RUN cd /app/ && conda run -n base /bin/bash -c "conda env create -f environment.yml"
#activate environment with shell because not working wih conda
SHELL ["conda", "run", "-n", "meld_graph", "/bin/bash", "-c"]
# RUN cd /app/ && conda run -n base /bin/bash -c "conda activate meld_graph"
#install meld_graph package
RUN cd /app/ && conda run -n base /bin/bash -c "pip install -e ."


#add data folder to docker
RUN mkdir /data

WORKDIR /app

# Define the command to run script 
# RUN conda run --no-capture-output -n meld_classifier /bin/bash -c "python scripts/prepare_classifier.py"
# RUN cd /meld_classifier && conda run --no-capture-output -n meld_classifier /bin/bash -c "pytest"

ENTRYPOINT ["conda", "run", "-n", "meld_graph", "python", "scripts/new_patient_pipeline/new_pt_pipeline.py"]
CMD ["-h"]
