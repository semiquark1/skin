# Skin lesion classification

This repository contains code and data accompanying manuscript [Handling dataset dependence with model ensembles for skin lesion classification from dermoscopic and clinical images](https://dx.doi.org/10.1002/ima.22827).

Code is borrowed from the following MIT-licensed repositories:

* https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution.git
* https://github.com/davda54/sam.git

## Prepare environment

    cd <project_root>
    conda create -y -p .conda/envs/skin python=3.7.9
    conda activate .conda/envs/skin
    pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 \
      -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt

and

    git clone https://github.com/ildoonet/pytorch-gradual-warmup-lr contrib/pytorch-gradual-warmup-lr

## Dataset dependence

### Download datasets

Download the skin lesion datasets HAM10000 to datasets/ham10000, and Derm7pt to datasets/derm7pt, so the relative paths of the images from these dataset roots match the `path` column in the dataset descriptor files data/ros.csv, data/vm.csv and data/d7d.csv.

### Train

An example command to train the EfficientNet-b4 model (`model_number=21`) for the ROS subset of the HAM10000 dataset with no class-weighted training and no augmentation:

    python src/skin_proc.py train --model-n 21 --dsd data/dsd_ros.csv --data-root datasets/ham10000 \
      --batch-size 14 --use-aug 0 --seed 42 data-output/m21/m21_cv_seed42/train_ros

This trains all 5 folds of the model for a single random seed, taking about 1/2 hour on a GeForce 2080 Ti. For class-weighted training add option `--weighted`, to include augmentation use `--use-aug 1`.

### Predict

An example command to predict on the same dataset (out-of-fold prediction)

    python src/skin_proc.py predict --model-n 21 --model-dir data-output/m21/m21_cv_seed42/train_ros \
      --dsd data/dsd_ros.csv --data-root datasets/ham10000 --oof --batch-size 14 --version final \
      data-output/m21/m21_cv_seed42/train_ros/ros_oof

And to predict on a different dataset (derm7pt-dermoscopic):

    python src/skin_proc.py predict --model-n 21 --model-dir data-output/m21/m21_cv_seed42/train_ros \
      --dsd data/dsd_d7d.csv --data-root datasets/derm7pt --batch-size 14 --version final \
      data-output/m21/m21_cv_seed42/train_ros/d7d

### Fit and predict the calibrated probabilities

An example command to fit the ROS dataset and predict on it:

    python src/skin_proc.py calibrate-predict-proba --beta 0.3 --n-bins 30 \
      --data data-output/m21/m21_cv_seed42/train_ros/ros_oof/softmax_m21.csv \
      --out-model-path data-output/m21/m21_cv_seed42/train_ros/model_proba_m21.h5 \
      data-output/m21/m21_cv_seed42/train_ros/ros_oof/proba_m21.csv

To predict on the derm7pt-dermoscopic dataset:

    python src/skin_proc.py predict-proba --data data-output/m21/m21_cv_seed42/train_ros/d7d/softmax_m21.csv \
      --model-path data-output/m21/m21_cv_seed42/train_ros/model_proba_m21.h5 \
      data-output/m21/m21_cv_seed42/train_ros/d7d/proba_m21.csv

### Extract results from predicted data

The repo contains in `data/` the softmax and calibrated probability predictions for four model versions: the baseline model, with class-weighted training added, with heavy augmentation added, and a version trained on the combined ISIC model (kwdset in the file names), all for random seeds 42, 43, 44, 45 and 46.

From this data generate the dataset dependence figure:

    python src/skin_fig.py plot-datasetdep

Extract numerical data for dataset dependence for run with a given random seed. The uncertainty can be calculated as a standard deviation across the 5 runs.

    python src/skin_fig.py print-datasetdep --seed 42

## Ensemble model: Committee-29

### Download models

Download the model files for the 11 models and the committee machine by opening this page:
    https://mega.nz/file/voxngK5T#uKOZkXBM9BXXAcuOQybGkfbGymli9LCBy9FxUXS61bY

Extract them into `models/`:

    tar xvf PATH/TO/models-v1.tar.gz

Download the models files for the 18 models by Ha, Liu and Liu from 
    https://www.kaggle.com/datasets/boliu0/melanoma-winning-models/download
and unzip them into `models/`. Create links with executing

    (cd models; ./create-links)

### Predict and evaluate

The simplest way to evaluate the Committee-29 ensemble model is to feed images as a simple directory structure, where subdirectories provide the ground truth labels. A template is provided at `datasets/dataset_template/`, where non-melanoma images should be placed into the `0-other` subdirectory, and melanome images into `1-melanoma`. To evaluate the ensemble on this dataset:

    python src/skin_proc.py predict-eval-dir --data-root datasets/dataset_template \
      --model-dir models/ --stages abcdef data-c29

where the final argument is the output directory. In case the memory usage of Tensorflow (used in stage c) and Pytorch (used in other stages) collide, the stages can be run as separate commands. The result of the evaluation is written in `data2/result.txt`, where the balanced accuracy calculated from softmax values at 50% threshold is in column `th_BA`. To evaluate on the derm7d or ph2 datasets, copy the appropriate dsd file from `data/` into the output directory (in the above example `data-c29`) as `dsd.csv`, and leave out stage `a`, and use the data-root of the downloaded dataset.

## Committee-29 ensemble on macro (clinical) images

### Download models

Model weights are provided for the best macro model: when ASAM is used for both the fine tuning of the 11 deepnets, and the training of the committee machine. Download the model files by opening:
    https://mega.nz/file/n5gAzArD#1RQFok_ZzIUfk_MOjnd-Jfe_JERJYdPGnfm81mha1rQ

Then extract them into `models-macro/`:

    tar xvf PATH/TO/models-macro-v1.tar.gz

Now if the model files for the 18 models by Ha, Liu and Liu are already downloaded into `models/', then simply create links to them:

    (cd models-macro; ln -s ../models/weights_m* .)

(Otherwise download them to `models-macro`, and execute `create-links` there.)

### Predict and evaluate

Use the same command as above, except that model files from `models-macro/` should be used:

    python src/skin_proc.py predict-eval-dir --data-root datasets/a_macro_dataset \
      --model-dir models-macro/ --stages abcdef data-c29macro




