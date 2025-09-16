# 🛠️ Installation

## 💻 Environments

Please follow the instructions to install the conda environments and the dependencies of the codebase. We recommend using CUDA 11.x during installations to avoid compatibility issues (remember to replace `11.x` in the following commands with your own CUDA version like `11.8`).

1. Create a new conda environment and activate the environment.
    ```bash
    conda create -n dspv2 python=3.8
    conda activate dspv2
    ```

2. Install necessary dependencies.
    ```bash
    conda install cudatoolkit=11.x
    pip install -r requirements.txt
    ```

3. Install [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) manually following [the official instsallation instructions](https://github.com/NVIDIA/MinkowskiEngine?tab=readme-ov-file#cuda-11x).
    ```bash
    conda install openblas-devel -c anaconda
    export CUDA_HOME=/usr/local/cuda-11.x
    cd dependencies/MinkowskiEngine
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
    cd ../..
    ```
    For CUDA 12.1, the installation of MinkowskiEngine may be not that smooth. You may need to add some headers in MinkowskiEngine. Refer to [issue#543](https://github.com/NVIDIA/MinkowskiEngine/issues/543).

4. Install [DINOv3 base](https://github.com/facebookresearch/dinov3) or [DINOv2 base](https://github.com/facebookresearch/dinov2) weights as AutoModel pattern and save them in `weights`. You can refer [`utils/download_dino.py`](utils/download_dino.py) to download DINOv2. You can apply DINOv3 in [the huggingface repo](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/tree/main).