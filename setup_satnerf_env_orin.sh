conda create -n satnerf -c conda-forge python=3.10 \
  gdal=3.6 \
  opencv \
  jupyter \
  pillow=11.1.0 \
  chardet \
  matplotlib \
  numpy \
  affine \
  fire \
  kornia=0.5.3 \
  pyproj \
  pytorch-lightning \
  torchmetrics \
  tensorboard \
  rasterio \
  scipy \
  utm \
  scikit-image \
  numba

conda activate satnerf
pip install --ignore-installed certifi -r requirements.txt

pip install https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=907c4c1933789645ebb20dd9181d40f8647978e6bd30086ae7b01febb937d2d1
pip install https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc

conda deactivate
echo "satnerf conda env for orin jetpack 6.2 created !"
