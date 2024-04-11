conda init bash
conda create -n alma-r python=3.11 -y
conda activate alma-r

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
bash install_alma.sh