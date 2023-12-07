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
        python3-tk \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
WORKDIR /home/$USERNAME

# Install Kimera-Evaluation
RUN cd Kimera-Evaluation && \
    pip install .[notebook]

ENV TERM xterm
ENV PYTHONIOENCODING UTF-8
