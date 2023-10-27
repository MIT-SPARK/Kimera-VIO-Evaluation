# Kimera VIO Evaluation

Code to evaluate and tune [Kimera-VIO](https://github.com/MIT-SPARK/Kimera-VIO) pipeline on [Euroc's dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).

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

# Example Usage

## Main Evaluation

The script `main_evaluation.py` runs and evaluates the VIO performance by aligning estimated and ground-truth trajectories and computing error metrics.
It then saves plots showing its performance.

The script expects an **experiment** yaml file with the following syntax:
```yaml
executable_path: '$HOME/Code/spark_vio/build/stereoVIOEuroc'
results_dir: '$HOME/Code/spark_vio_evaluation/results'
params_dir: '$HOME/Code/spark_vio_evaluation/experiments/params'
dataset_dir: '$HOME/datasets/euroc'

datasets_to_run:
 - name: V1_01_easy
   segments: [1, 5]
   pipelines: ['S']
   discard_n_start_poses: 10
   discard_n_end_poses: 10
   initial_frame: 100
   final_frame: 2100
 - name: MH_01_easy
   segments: [5, 10]
   pipelines: ['S', 'SP', 'SPR']
   discard_n_start_poses: 0
   discard_n_end_poses: 10
   initial_frame: 100
   final_frame: 2500

```

The experiment yaml file specifies the following:
- `executable_path`: where to find the built binary executable to run Kimera-VIO.
- `results_dir`: the directory where to store the results for each dataset. This directory is already inside this repository.
- `params_dir`: the directory where to find the parameters to be used by Kimera-VIO.
- `dataset_dir`: the path to the Euroc dataset.
- `datasets_to_run`: specifies which Euroc datasets to run, with the following params:
  - `name`: the name of the Euroc dataset to run. It must match exactly to the subfolders in your path to Euroc dataset.
  - `segments`: these are the distances btw poses to use when computing the Relative Pose Error (RPE) metric. If multiple are given, then RPE will be calculated for each given distance. For example, if `segments: [1, 5]`, RPE will be calculated for all 1 meter apart poses and plotted in a boxplot, same for all 5m apart poses, etc.
  - `pipelines`: this can only be `S`, `SP`, and/or `SPR`; the vanilla VIO corresponds to `S` (structureless factors only). If using the RegularVIO pipeline [1] then `SP` corresponds to using Structureless and Projection factors, while `SPR` makes use of Regularity factors as well.
  - `discard_n_X_poses`: discards `n` poses when aligning ground-truth and estimated trajectories.
  - `initial/final_frame`: runs the VIO starting on `initial_frame` and finishing on `final_frame`. This is useful for datasets which start/finish by bumping against the ground, which might negatively affect IMU readings.

`./evaluation/main_evaluation.py -r -a --save_plots --save_results --save_boxplots experiments/example_euroc.yaml`

where, as explained below, the `-r` and `-a` flags run the VIO pipeline given in the `executable_path` and analyze its output.

# Usage

Run `./evaluation/main_evaluation.py --help` to get usage information.

```bash
usage: main_evaluation.py [-h] [-r] [-a] [--plot]
                          [--plot_colormap_max PLOT_COLORMAP_MAX]
                          [--plot_colormap_min PLOT_COLORMAP_MIN]
                          [--plot_colormap_max_percentile PLOT_COLORMAP_MAX_PERCENTILE]
                          [--save_plots] [--save_boxplots] [--save_results]
                          experiments_path

Full evaluation of SPARK VIO pipeline (APE trans + RPE trans + RPE rot) metric
app

optional arguments:
  -h, --help            show this help message and exit

input options:
  experiments_path      Path to the yaml file with experiments settings.

algorithm options:
  -r, --run_pipeline    Run vio?
  -a, --analyse_vio     Analyse vio, compute APE and RPE

output options:
  --plot                show plot window
  --plot_colormap_max PLOT_COLORMAP_MAX
                        The upper bound used for the color map plot (default:
                        maximum error value)
  --plot_colormap_min PLOT_COLORMAP_MIN
                        The lower bound used for the color map plot (default:
                        minimum error value)
  --plot_colormap_max_percentile PLOT_COLORMAP_MAX_PERCENTILE
                        Percentile of the error distribution to be used as the
                        upper bound of the color map plot (in %, overrides
                        --plot_colormap_min)
  --save_plots          Save plots?
  --save_boxplots       Save boxplots?
  --save_results        Save results?
  -v, --verbose_sparkvio
                        Make Kimera-VIO log all verbosity to console. Useful
                        for debugging if a run failed.
```


Run `./evaluation/regression_tests.py --help` to get usage information.
```bash
usage: regression_tests.py [-h] [-r] [-a] [--plot] [--save_plots]
                           [--save_boxplots] [--save_results]
                           experiments_path

Regression tests of SPARK VIO pipeline.

optional arguments:
  -h, --help          show this help message and exit

input options:
  experiments_path    Path to the yaml file with experiments settings.

algorithm options:
  -r, --run_pipeline  Run vio?
  -a, --analyse_vio   Analyse vio, compute APE and RPE

output options:
  --plot              show plot window
  --save_plots        Save plots?
  --save_boxplots     Save boxplots?
  --save_results      Save results?
```

# Jupyter Notebooks

Provided are jupyter notebooks for extra plotting, especially of the debug output from Kimera-VIO. Follow the steps below to run them.

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
6. Run the notebooks! A useful beginner tutorial for using Jupyter notebooks can be found [here](https://www.dataquest.io/blog/jupyter-notebook-tutorial/). A guide for interpreting the output is coming soon.

# Chart of implementation details:

![Kimera-VIO evaluation diagram](docs/chart_sparkvio_evaluation.svg)

# Notes

The behaviour for the plots depends also on `evo_config`.
For example, in Jenkins we use the default `evo_config` which does not split plots.
Yet, locally, you can use `evo_config` to allow plotting plots separately for adding them in your paper.

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
