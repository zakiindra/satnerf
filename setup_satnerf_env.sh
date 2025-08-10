### create satnerf venv

conda create -n satnerf -c conda-forge python=3.10 \
  gdal \
  opencv \
  jupyter \
  pillow \
  chardet \
  matplotlib \
  numpy \
  pytorch \
  torchvision \
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
conda deactivate
echo "satnerf conda env created !"