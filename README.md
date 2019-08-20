# spark_vio_evaluation
Code to evaluate and tune SPARK VIO pipeline.

# Installation

```bash
git clone https://github.com/ToniRV/spark_vio_evaluation
git checkout devel
cd spark_vio_evaluation
python setup.py develop
```

# Example Usage 

The script `main_evaluation.py` runs and evaluates the VIO performance by aligning estimated and ground-truth trajectories and computing the accumulated errors. It then saves plots showing its performance. 

`./evaluation/main_evaluation.py experiments/example_euroc.yaml -r -a --save_plots --save_results --save_boxplots`

where, as explained below, the r and a flags run and analyze the pipeline.

You will have to specify the experiment yaml file which points to the SparkVIO executable and specifies which datasets to run.

# Usage

Run `./evaluation/main_evaluation.py` to get usage information.

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
```

