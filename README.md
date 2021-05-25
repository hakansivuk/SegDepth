# SedDepth

> **Segmentation-Guided Self-Supervised Monocular Depth Estimation**

In this project, semantic segmentation is used as a booster to the depth estimation task. Segmentation masks that separate foreground objects from background are created from full scale disparity maps through a U-net architecture. In addition to minimum reprojection loss and edge-aware smoothness loss, pixel-wise weighted cross entropy loss is used as an additional supervision signal.

You can find the dataset (KITTI raw) [from this link](http://www.cvlibs.net/datasets/kitti/raw_data.php)

You can find the pretrained baseline and SegDepth network on KITTI raw subset dataset [from this link](https://drive.google.com/drive/folders/1PQ_gTVE3LQwF7aQKJzNpoUXCkgcvRfhD?usp=sharing)

### Training Process

I performed all experiments on NVIDIA Geforce 1050-TI
Training baseline model on eigen_zhou_subset took 2.5 hours approximately.
Training improved model on eigen_zhou_subset took 7 hours approximately.

**Splits**

I used subset of eigen_zhou_subset split. You can find it under splits/ folder.

## Instructions for Running Scripts
In this section, necessary instructions to run scripts are described. Since downloading all the dataset is time consuming, you can skip to Toy Examples section to run with a very few samples that are already in kitti_data folder.

## 1) Download Dataset
You can download training subset data by typing the commands below:
```shell
wget -i splits/kitti_archives_to_download_subset.txt -P kitti_data/
cd kitti_data
unzip "*.zip"
cd ..
```
## 2) Creating Segmentation Masks
For creating segmentation masks, create and use the conda environment specified below:
```shell
conda create -n csm python=3.6.6
conda activate csm
pip install torch torchvision
pip install opencv-python
pip install numpy
python create_segmentation_masks.py
```

## 3) Training
Follow the steps below:
```shell
conda create -n segdepth python=3.6.6
conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
conda install scikit-image
conda install opencv=3.3.1
```

By default models are saved to `tmp/<model_name>`.
This can be changed with the `--log_dir` flag.

```shell
python train.py --model_name mono_model
```

## 4) Prediction
You can predict depth for a single image with (change model path according to your case):
```shell
python test_simple.py --image_path assets/test_image.jpg --model_name mono_model --model_path tmp/mono_model/models/weights_19/
```
## 5) Eigen split evaluation

To prepare the ground truth depth maps run:
```shell
python export_gt_depth.py --data_path kitti_data --split eigen
```
...assuming that you have placed the KITTI dataset in the default location of `./kitti_data/`.

The following example command evaluates the epoch 19 weights of a model named `mono_model`:
```shell
python evaluate_depth.py --load_weights_folder tmp/mono_model/models/weights_19/ --eval_mono
```
## Toy Examples
In this section, you can run the scripts above with a very few data without any downloading process.

## 1) Training
Follow the steps below:
```shell
conda create -n segdepth python=3.6.6
conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
conda install scikit-image
conda install opencv=3.3.1
python train.py --model_name mono_model --example_run --batch_size 1
```

## 2) Eigen split evaluation
To prepare the ground truth depth maps run:
```shell
python export_gt_depth.py --data_path kitti_data --split eigen --test_file test_files_example.txt
```
assuming that you have placed the KITTI dataset in the default location of `./kitti_data/`.

The following example command evaluates the epoch 19 weights of a model named `mono_model`:
```shell
python evaluate_depth.py --load_weights_folder tmp/mono_model/models/weights_19/ --eval_mono --test_file test_files_example.txt
```