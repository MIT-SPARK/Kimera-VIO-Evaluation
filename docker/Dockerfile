FROM ros:noetic-ros-base

ARG UID
ARG USERNAME

ENV WS=/home/$USERNAME/Kimera-Evaluation
ENV DEBIAN_FRONTEND=noninteractive

# (1) Create a home directory for the user running the image
# (2) Grant sudo access (requiring no password)
RUN useradd --create-home --shell /bin/bash -u $UID $USERNAME && \
  usermod -G sudo $USERNAME && \
  echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USERNAME

# Create a directory owned by $USERNAME and set to startup directory
RUN mkdir -p /home/$USERNAME

# Copy this repo
COPY ./ $WS

# Get build tools
RUN apt-get update && apt-get install -y \
    apt-utils \ 
    git \
    python3-tk \
    python3-pip

# Install dependencies
WORKDIR /home/$USERNAME
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx
RUN pip install pandas open3d setuptools plotly --upgrade
RUN git clone https://github.com/MIT-SPARK/evo-1 evo
RUN cd evo && \
    git checkout fix/python3 && \
    pip install . --upgrade --no-binary evo

# Install Kimera-Evaluation
RUN cd Kimera-Evaluation && \
    pip install .[notebook] && \
    python3 setup.py develop

# Cleanup
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

ENV TERM xterm
ENV PYTHONIOENCODING UTF-8
