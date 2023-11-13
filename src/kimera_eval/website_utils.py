"""Helpers for creating jenkins website."""
import kimera_eval.plotting

from dataclasses import dataclass
import pathlib
import jinja2
import pandas as pd
import plotly
import plotly.subplots
import logging
import yaml


def _fig_to_html(fig, include_plotlyjs=False, output_type="div"):
    return plotly.offline.plot(
        fig, include_plotlyjs=include_plotlyjs, output_type=output_type
    )


def _get_frontend_results_as_html(results_path):
    df_stats = pd.read_csv(results_path, sep=",", index_col=False)
    html = _fig_to_html(kimera_eval.plotting.draw_feature_tracking_stats(df_stats))
    html += _fig_to_html(
        kimera_eval.plotting.draw_mono_stereo_inliers_outliers(df_stats)
    )
    return html


def _get_dataset_results_as_html(dataset, csv_results_path, x_id="#timestamp"):
    df = pd.read_csv(csv_results_path)
    fig = plotly.subplots.make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "scene"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "Position",
            "3D Trajectory",
            "Orientation",
            "Gyro Bias",
            "Velocity",
            "Accel Bias",
        ),
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    fig.update_layout(
        title_text=f"Raw VIO Output for dataset: {dataset}",
        template="plotly_white",
    )

    kimera_eval.plotting.plot_multi_line(df, x_id, ["x", "y", "z"], fig, row=1, col=1)
    kimera_eval.plotting.plot_multi_line(
        df, x_id, ["qw", "qx", "qy", "qz"], fig, row=2, col=1
    )
    kimera_eval.plotting.plot_multi_line(
        df, x_id, ["vx", "vy", "vz"], fig, row=3, col=1
    )

    kimera_eval.plotting.plot_3d_trajectory(df, x_id, fig, row=1, col=2)
    kimera_eval.plotting.plot_multi_line(
        df, x_id, ["bgx", "bgy", "bgz"], fig, row=2, col=2
    )
    kimera_eval.plotting.plot_multi_line(
        df, x_id, ["bax", "bay", "baz"], fig, row=3, col=2
    )

    return _fig_to_html(fig)


@dataclass
class ResultGroup:
    """Group of result info."""

    result_path: pathlib.Path
    plot_name: str = "plots.pdf"
    frontend_name: str = "output_frontend_stats.csv"
    vio_name: str = "traj_vio.csv"
    pgo_name: str = "traj_pgo.csv"

    def __post_init__(self):
        """Resolve result path."""
        self.result_path = pathlib.Path(self.result_path).expanduser().absolute()

    @property
    def frontend_results_path(self):
        """Path to frontend statistics for the results."""
        return self.result_path / self.frontend_name

    @property
    def plot_path(self):
        """Path to the plots for the results."""
        return self.result_path / self.plot_name

    @property
    def vio_trajectory_path(self):
        """Path to the unoptimized trajectory for the results."""
        return self.result_path / self.vio_name

    @property
    def pgo_trajectory_path(self):
        """Path to the optimized trajectory for the results."""
        return self.result_path / self.pgo_name


class WebsiteBuilder:
    """Website builder class."""

    def __init__(self):
        """
        Construct a builder from templates.

        Uses jinja to render tempaates stored in kimera_eval.website.templates
        """
        self._env = jinja2.Environment(
            loader=jinja2.PackageLoader("website", "templates"),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )

        self._detail_render = self._env.get_template(
            "detailed_performance_template.html"
        )
        self._dataset_render = self._env.get_template("datasets_template.html")
        self._boxplot_render = self._env.get_template("vio_performance_template.html")

    def write(self, results, stats, output_path):
        """Write website using the collected data."""
        output_path = pathlib.Path(output_path).expanduser().absolute()
        output_path.mkdir(parents=True, exit_ok=True)

        with (output_path / "vio_ape_euroc.html").open("w") as fout:
            fig_html = _fig_to_html(
                kimera_eval.plotting.draw_ape_boxplots_plotly(stats)
            )
            fout.write(self._boxplot_render.render(boxplot=fig_html))

        with (output_path / "detailed_performance.html").open("w") as fout:
            fout.write(
                self._detail_render.render(
                    datasets_pdf_path={x.dataset: x.plot_path for x in results}
                )
            )

        with (output_path / "frontend.html").open("w") as fout:
            frontend_html = {
                x.dataset: _get_frontend_results_as_html(x.frontend_results_path)
                for x in results
            }
            fout.write(self._dataset_render.render(datasets_html=frontend_html))

        with (output_path / "datasets.html").open("w") as fout:
            traj_html = {
                x.dataset: _get_dataset_results_as_html(x.vio_trajectory_path)
                for x in results
            }
            fout.write(self._dataset_render.render(datasets_html=traj_html))


def check_stats(stats):
    """Check stat contents."""
    if "relative_errors" not in stats:
        logging.error(f"Stats are missing required metrics: {stats}")

    if len(stats["relative_errors"]) == 0:
        logging.error(f"Stats are missing required metrics: {stats}")

    if "rpe_rot" not in list(stats["relative_errors"].values())[0]:
        logging.error(f"Stats are missing required metrics: {stats}")

    if "rpe_trans" not in list(stats["relative_errors"].values())[0]:
        logging.error(f"Stats are missing required metrics: {stats}")

    if "absolute_errors" not in stats:
        logging.error(f"Stats are missing required metrics: {stats}")
        return False

    return True


def aggregate_ape_results(results_dir, use_pgo=False):
    """
    Aggregate APE results and draw APE boxplot as well as write latex table.

    Args:
      - result_dir: path to directory containing yaml result files
      - use_pgo: whether to aggregate all results for VIO or for PGO trajectory.

    Returns:
        Dict[str, Dict[str, Any]]: results keyed by dataset then pipeline
    """
    logging.debug(f"Aggregating dataset results @ '{results_dir}'")

    yaml_filename = "results_vio.yaml"
    if use_pgo:
        yaml_filename = "results_pgo.yaml"

    stats = {}
    results_path = pathlib.Path(results_dir)
    filepaths = sorted(list(results_path.glob(f"**/{yaml_filename}")))
    for filepath in filepaths:
        pipeline_name = filepath.parent.stem
        dataset_name = filepath.parent.parent.stem
        if dataset_name not in stats:
            stats[dataset_name] = {}

        with filepath.open("r") as fin:
            stats[dataset_name][pipeline_name] = yaml.safe_load(fin.read())

        logging.debug(f"Checking stats from `{filepath}`")
        if not check_stats(stats[dataset_name][pipeline_name]):
            logging.warning(f"invalid stats for {dataset_name}:{pipeline_name}")

    return stats


def write_website():
    """Output website based on saved analysis."""
    logging.info("Writing full website...")
    stats = aggregate_ape_results(self.results_dir)

    website_builder.write_boxplot_website(stats)
    builder.write()
    logging.info("Finished writing website.")
