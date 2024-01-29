apt-get -y update && apt-get -y install wget

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

PATH="/miniconda/bin:${PATH}"

# Create environment
conda create -n titanic pandas PyYAML scikit-learn -c conda-forge
conda activate titanic

PATH="/miniconda/envs/titanic/bin:${PATH}"

python main.py