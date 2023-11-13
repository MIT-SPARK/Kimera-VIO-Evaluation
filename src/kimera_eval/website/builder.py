"""Helpers for creating jenkins website."""
from kimera_eval.trajectory_metrics import TrajectoryResults
import kimera_eval.plotting

from dataclasses import dataclass
import pathlib
import jinja2
import pandas as pd
import plotly
import plotly.subplots
import logging


def _fig_to_html(fig, include_plotlyjs=False, output_type="div"):
    return plotly.offline.plot(
        fig, include_plotlyjs=include_plotlyjs, output_type=output_type
    )


def _make_frontend_fig(results_path):
    if not results_path.exists():
        logging.warning("missing frontend information at '{results_path}'")
        return None

    df_stats = pd.read_csv(results_path, sep=",", index_col=False)
    html = _fig_to_html(kimera_eval.plotting.draw_feature_tracking_stats(df_stats))
    html += _fig_to_html(
        kimera_eval.plotting.draw_mono_stereo_inliers_outliers(df_stats)
    )
    return html


def _make_trajectory_fig(dataset, csv_results_path, x_id="#timestamp"):
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

    kimera_eval.plotting.plot_3d_trajectory(df, fig, row=1, col=2)
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
            loader=jinja2.PackageLoader("kimera_eval.website", "templates"),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )

        self._detail_render = self._env.get_template(
            "detailed_performance_template.html"
        )
        self._dataset_render = self._env.get_template("datasets_template.html")
        self._boxplot_render = self._env.get_template("vio_performance_template.html")

    def write(
        self, results_path: pathlib.Path, output_path: pathlib.Path, use_pgo=False
    ):
        """Collect results and write corresponding website."""
        results_path = pathlib.Path(results_path)
        logging.debug(f"Aggregating dataset results @ '{results_path}'")
        filename = "results_vio.pickle" if not use_pgo else "results_pgo.pickle"

        stats = {}
        results = {}

        filepaths = sorted(list(results_path.glob(f"**/{filename}")))
        for filepath in filepaths:
            pipeline = filepath.parent.stem
            if pipeline not in results:
                results[pipeline] = {}

            dataset = filepath.parent.parent.stem
            if dataset not in stats:
                stats[dataset] = {}

            stats[dataset][pipeline] = TrajectoryResults.load(filepath)
            results[pipeline][dataset] = ResultGroup(filepath.parent)

        output_path = pathlib.Path(output_path).expanduser().absolute()
        output_path.mkdir(parents=True, exist_ok=True)

        logging.debug("writing website to {output_path}...")
        with (output_path / "vio_ape_euroc.html").open("w") as fout:
            fig_html = _fig_to_html(kimera_eval.plotting.draw_ape_boxplots(stats))
            fout.write(self._boxplot_render.render(boxplot=fig_html))

        for pipeline, pipeline_results in results.items():
            pipeline_output = output_path / pipeline
            pipeline_output.mkdir(parents=True, exist_ok=True)
            pdfs = {}
            frontend_figs = {}
            trajs = {}

            for dataset, info in pipeline_results.items():
                pdfs[dataset] = info.plot_path
                trajs[dataset] = _make_trajectory_fig(dataset, info.vio_trajectory_path)
                frontend_fig = _make_frontend_fig(info.frontend_results_path)
                if frontend_fig is not None:
                    frontend_figs[dataset] = frontend_fig

            with (pipeline_output / "detailed_performance.html").open("w") as fout:
                fout.write(self._detail_render.render(datasets_pdf_path=pdfs))

            with (pipeline_output / "frontend.html").open("w") as fout:
                fout.write(self._dataset_render.render(datasets_html=frontend_figs))

            with (pipeline_output / "datasets.html").open("w") as fout:
                fout.write(self._dataset_render.render(datasets_html=trajs))

            logging.debug("finished writing website")
