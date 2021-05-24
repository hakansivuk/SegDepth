# SedDepth

> **Segmentation-Guided Self-Supervised Monocular Depth Estimation**

## Conda Environment

Follow the steps below:
```shell
conda create -n monodepth2 python=3.6.6
conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
pip install tensorboardX==1.4
conda install opencv=3.3.1   # just needed for evaluation
```
## Prediction for a single image

You can predict depth for a single image with:
```shell
python test_simple.py --image_path assets/test_image.jpg --model_name mono+stereo_640x192
```

On its first run this will download the `mono+stereo_640x192` pretrained model (99MB) into the `models/` folder.
We provide the following  options for `--model_name`:
## Training data

I used subset of [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php). You can download this subset by typing the commands below:
```shell
wget -i splits/kitti_archives_to_download_subset.txt -P kitti_data/
cd kitti_data
unzip "*.zip"
cd ..
```

**Splits**

I used subset of eigen_zhou split. You can find it under splits/ folder.
You can also use other split by specifying which split do you want with `--split` flag.

## Training

By default models are saved to `tmp/<model_name>`.
This can be changed with the `--log_dir` flag.

```shell
python train.py --model_name mono_model
```

### Training Process

I performed all experiments on ...
Training baseline model on eigen_zhou_subset took 2.5 hours approximately.
Training improved model on eigen_zhou_subset took 7 hours approximately.

All our experiments were performed on a single NVIDIA Titan Xp.

| Training modality | Approximate GPU memory  | Approximate training time   |
|-------------------|-------------------------|-----------------------------|
| Mono              | 9GB                     | 12 hours                    |
| Stereo            | 6GB                     | 8 hours                     |
| Mono + Stereo     | 11GB                    | 15 hours                    |

### Other training options

Run `python train.py -h` (or look at `options.py`) to see the range of other training options, such as learning rates and ablation settings.


## Eigen split evaluation

To prepare the ground truth depth maps run:
```shell
python export_gt_depth.py --data_path kitti_data --split eigen
```
...assuming that you have placed the KITTI dataset in the default location of `./kitti_data/`.

The following example command evaluates the epoch 19 weights of a model named `mono_model`:
```shell
python evaluate_depth.py --load_weights_folder tmp/mono_model/models/weights_19/ --eval_mono
```