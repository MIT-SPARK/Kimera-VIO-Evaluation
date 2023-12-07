# Kimera VIO Evaluation

![Python package](https://github.com/MIT-SPARK/Kimera-VIO-Evaluation/actions/workflows/tox.yml/badge.svg)](https://github.com/MIT-SPARK/Kimera-VIO-Evaluation/actions/workflows/tox.yml)

Code to evaluate [Kimera-VIO](https://github.com/MIT-SPARK/Kimera-VIO) on [Euroc's dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).

# Installation

> We strongly recommend creating a new virtual environment to avoid conflicts with system-wide installations:
> ```bash
> sudo apt-get install python3-virtualenv
> python3 -m virtualenv -p $(which python3) --download ./venv
> source ./venv/bin/activate
> ```

```bash
git clone https://github.com/MIT-SPARK/Kimera-VIO-Evaluation
cd Kimera-VIO-Evaluation
# you may want to do this instead for jupyter notebooks:
# pip install .[notebook]
pip install .
```

# Jupyter Notebooks

We provide jupyter notebooks for examining the output of Kimera-VIO. Follow the steps below to run them.

1. Set up Kimera Evaluation as stated above (using the `notebook` extra) or install the required dependencies if you didn't use the notebook extra:
```
pip install jupyter jupytext
```
2. Open the `notebooks` folder in the Jupyter browser
```
cd Kimera-Evaluation/notebooks
jupyter notebook
```
3. If the contents of the folder appear empty in your web-browser, you may have to manually add the jupytext content manager as described [here](https://github.com/mwouts/jupytext/blob/master/docs/install.md#jupytexts-contents-manager)
4. Open the notebook corresponding to what you want to analyze first. `plot-frontend.py` is a good place to start.
5. Provide the path to the folder with Kimera's debug information from your dataset (typically `Kimera-VIO-ROS/output_logs/<yourdatasetname>`)
6. Run the notebooks! A useful beginner tutorial for using Jupyter notebooks can be found [here](https://www.dataquest.io/blog/jupyter-notebook-tutorial/).

# Usage

Run `kimera-eval --help` to see usage information.

You may find the following commands useful:

- `kimera-eval run`: Runs a single experiment with Kimera-VIO using the provided experiment file

- `kimera-eval evaluate`: Evaluates the results of an experiment

- `kimera-eval website`: Construct a website containing plots and results from experiments

- `kimera-eval summary`: Write a summary of ATE errors for a single sequence to a CSV file

# Experiment Configuration

Several commands (i.e., `run` and `evaluate`) expect an experiment file with information about what to run:
```yaml
executable_path: '$HOME/Code/spark_vio/build/stereoVIOEuroc'
params_dir: '$HOME/Code/spark_vio_evaluation/experiments/params'
dataset_dir: '$HOME/datasets/euroc'
pipelines:
    name: Euroc
    param_name: Euroc
sequences:
 - name: V1_01_easy
   initial_frame: 100
   final_frame: 2100
   analysis:
     segments: [1, 5]
     discard_n_start_poses: 10
     discard_n_end_poses: 10
 - name: MH_01_easy
   initial_frame: 100
   final_frame: 2500
   analysis:
     segments: [5, 10]
     discard_n_start_poses: 0
     discard_n_end_poses: 10
```

The experiment yaml file specifies the following:
- `executable_path`: where to find the built binary executable to run Kimera-VIO.
- `params_dir`: the directory where to find the parameters to be used by Kimera-VIO.
- `dataset_dir`: the path to the Euroc dataset.
- `pipelines`: pipelines to run
    - `name`: descriptive name of the pipeline
    - `param_name`: name of the parameter folder (if different from `name`)
- `sequences`: specifies which Euroc sequences to run, with the following params:
  - `name`: the name of the Euroc dataset to run. It must exactly match the name of the subfolders in your copy of the Euroc dataset.
  - `initial/final_frame`: runs the VIO starting on `initial_frame` and finishing on `final_frame`. This is useful for datasets which start/finish by bumping against the ground, which might negatively affect IMU readings.
  - `analysis`: controls trajectory error evaluations
      - `segments`: these are the distances btw poses to use when computing the Relative Pose Error (RPE) metric. If multiple are given, then RPE will be calculated for each given distance. For example, if `segments: [1, 5]`, RPE will be calculated for all 1 meter apart poses and plotted in a boxplot, same for all 5m apart poses, etc.
      - `discard_n_X_poses`: discards `n` poses when aligning ground-truth and estimated trajectories.

# References

- [1] A. Rosinol, T. Sattler, M. Pollefeys, L. Carlone. [**Incremental Visual-Inertial 3D Mesh Generation with Structural Regularities**](https://arxiv.org/abs/1903.01067). IEEE Intl. Conf. on Robotics and Automation (ICRA), 2019. [arXiv:1903.01067](https://arxiv.org/abs/1903.01067)

```bibtex
@InProceedings{Rosinol19icra-incremental,
  title = {Incremental visual-inertial 3d mesh generation with structural regularities},
  author = {Rosinol, Antoni and Sattler, Torsten and Pollefeys, Marc and Carlone, Luca},
  year = {2019},
  booktitle = {2019 International Conference on Robotics and Automation (ICRA)},
  pdf = {https://arxiv.org/pdf/1903.01067.pdf}
}
```

- [2] A. Rosinol, M. Abate, Y. Chang, L. Carlone, [**Kimera: an Open-Source Library for Real-Time Metric-Semantic Localization and Mapping**](https://arxiv.org/abs/1910.02490). IEEE Intl. Conf. on Robotics and Automation (ICRA), 2020. [arXiv:1910.02490](https://arxiv.org/abs/1910.02490).
 
 ```bibtex
 @InProceedings{Rosinol20icra-Kimera,
   title = {Kimera: an Open-Source Library for Real-Time Metric-Semantic Localization and Mapping},
   author = {Rosinol, Antoni and Abate, Marcus and Chang, Yun and Carlone, Luca},
   year = {2020},
   booktitle = {IEEE Intl. Conf. on Robotics and Automation (ICRA)},
   url = {https://github.com/MIT-SPARK/Kimera},
   pdf = {https://arxiv.org/pdf/1910.02490.pdf}
 }
```

- [3] A. Rosinol, A. Gupta, M. Abate, J. Shi, L. Carlone. [**3D Dynamic Scene Graphs: Actionable Spatial Perception with Places, Objects, and Humans**](https://arxiv.org/abs/2002.06289). Robotics: Science and Systems (RSS), 2020. [arXiv:2002.06289](https://arxiv.org/abs/2002.06289).

```bibtex
@InProceedings{Rosinol20rss-dynamicSceneGraphs,
  title = {{3D} Dynamic Scene Graphs: Actionable Spatial Perception with Places, Objects, and Humans},
  author = {A. Rosinol and A. Gupta and M. Abate and J. Shi and L. Carlone},
  year = {2020},
  booktitle = {Robotics: Science and Systems (RSS)},
  pdf = {https://arxiv.org/pdf/2002.06289.pdf}
}
```
