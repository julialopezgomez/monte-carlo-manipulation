FROM robotlocomotion/drake:1.40.0

# Install git
RUN apt-get update && apt-get -y install git && apt-get -y install nano

# # Set PYTHONPATH
ENV PYTHONPATH=/opt/drake/lib/python3.10/site-packages:$PYTHONPATH
ENV MOSEKLM_LICENSE_FILE=/workspaces/monte-carlo-manipulation/src/mosek.lic

RUN python3 -m pip install --no-cache-dir ipywidgets==8.0.4 widgetsnbextension==4.0.5 jupyterlab-widgets
RUN python3 -m pip install --no-cache-dir scipy==1.10.1 pymcubes==0.1.4 quadprog cvxpy mosek cvxpy[MOSEK] plotly gymnasium
RUN python -m pip install gurobipy tqdm wandb
# Install PyTorch with matching CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu117 \
    torchvision==0.15.2+cu117 \
    torchaudio==2.0.2 \
    --extra-index-url https://download.pytorch.org/whl/cu117