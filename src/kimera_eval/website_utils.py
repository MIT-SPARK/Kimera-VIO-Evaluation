"""Helpers for creating jenkins website."""
import kimera_eval.plotting

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


class WebsiteBuilder:
    """Website builder class."""

    def __init__(self, traj_vio_csv_name):
        """
        Construct a builder from templates.

        Reads a template html website inside the `templates` directory of
        a `website` python package (that's why we call `import website`, which
        is a package of this project), and writes down html code with plotly figures.
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

        # We will store html snippets of each dataset indexed by dataset name in these
        # dictionaries. Each dictionary indexes the data per pipeline_type in turn:
        self.detailed_performance_html = {}
        self.frontend_html = {}
        self.datasets_html = {}

        self.traj_vio_csv_name = traj_vio_csv_name

    def write_boxplot_website(self, stats, output_path):
        """Write boxplots to website file."""
        output_path = pathlib.Path(output_path).expanduser().absolute()
        output_path.mkdir(parents=True, exit_ok=True)
        with (output_path / "vio_ape_euroc.html").open("w") as fout:
            fig_html = _fig_to_html(
                kimera_eval.plotting.draw_ape_boxplots_plotly(stats)
            )
            fout.write(self._boxplot_render.render(boxplot=fig_html))

    def add_dataset_to_website(self, dataset, pipeline, csv_results_path):
        """Add dataset results specified in csv_results_path."""
        pipeline_path = pathlib.Path(dataset) / pipeline
        csv_results_path = pathlib.Path(csv_results_path)

        self.detailed_performance_html[dataset] = pipeline_path / "plots.pdf"
        self.frontend_html[dataset] = _get_frontend_results_as_html(
            csv_results_path / "output_frontend_stats.csv"
        )

        self.datasets_html[dataset] = _get_dataset_results_as_html(
            dataset, csv_results_path / self.traj_vio_csv_name
        )

    def write_datasets_website(self, output_path):
        """Write website using the collected data."""
        output_path = pathlib.Path(output_path).expanduser().absolute()
        output_path.mkdir(parents=True, exit_ok=True)

        with (output_path / "detailed_performance.html").open("w") as fout:
            fout.write(
                self._detail_render.render(
                    datasets_pdf_path=self.detailed_performance_html
                )
            )

        with (output_path / "frontend.html").open("w") as fout:
            fout.write(self._dataset_render.render(datasets_html=self.frontend_html))

        with (output_path / "datasets.html").open("w") as fout:
            fout.write(self._dataset_render.render(datasets_html=self.datasets_html))


def write_website(self):
    """Output website based on saved analysis."""
    logging.info("Writing full website...")
    stats = aggregate_ape_results(self.results_dir)

    for dataset, pipelines in stats.items():
        for pipeline in pipelines:
            logging.info(
                f"Writing performance website for dataset: {dataset}:{pipeline}"
            )
            self.website_builder.add_dataset_to_website(
                dataset_name, pipeline_type, curr_results_path
            )

    self.website_builder.write_boxplot_website(stats)
    self.website_builder.write_datasets_website()
    logging.info("Finished writing website.")
