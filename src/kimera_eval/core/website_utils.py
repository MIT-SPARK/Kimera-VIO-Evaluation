"""Helpers for creating jenkins website."""
import os

import pandas as pd

import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from jinja2 import Environment, PackageLoader, select_autoescape

from kimera_vio_evaluation.tools import (
    draw_ape_boxplots_plotly,
    draw_feature_tracking_stats,
    draw_mono_stereo_inliers_outliers,
)


def get_fig_as_html(fig):
    """Get a plotly figure and returns html string to embed in a website."""
    return plotly.offline.plot(fig, include_plotlyjs=False, output_type="div")


# This is really making three different websites:
# a) dataset_template: which plots all raw data for a run (no need for gt).
#    It'll be useful to be able to get
#    this website up even for real-life data as well where we might not get ground-truth
# b) frontend_template: which plots the frontend stats, I don't think this needs gt
#    either.
# c) boxplot_template which is really the one needing gt as it plots APE in boxplots.
# d) detailed_performance_template which needs the plots.pdf which implicitly needs gt,
#    to just present the pdf plots
class WebsiteBuilder:
    """Website builder class."""

    def __init__(self, website_output_path, traj_vio_csv_name):
        """
        Construct a builder from templates.

        Reads a template html website inside the `templates` directory of
        a `website` python package (that's why we call `import website`, which
        is a package of this project), and writes down html code with plotly figures.
        """
        # Initialize Jinja2
        jinja_env = Environment(
            loader=PackageLoader("website", "templates"),
            autoescape=select_autoescape(["html", "xml"]),
        )

        self.boxplot_website = BoxplotWebsiteBuilder(
            jinja_env, website_output_path, traj_vio_csv_name
        )
        self.raw_output_website = RawOutputWebsiteBuilder(
            jinja_env, website_output_path, traj_vio_csv_name
        )
        self.frontend_website = FrontendWebsiteBuilder(jinja_env, website_output_path)
        self.detailed_performance_website = PdfLoaderWebsiteBuilder(
            jinja_env, website_output_path
        )

    def write_boxplot_website(self, stats):
        """
        Write website using overall stats.

        Optionally the original data of a dataset run specified in csv_results_path.

        Args:
        - stats: a nested dictionary with the results of all pipelines:
            * First level ordered with dataset_name as keys:
            * Second level ordered with pipeline_type as keys:
            * Each stats[dataset_name][pipeline_type] value has:
                * absolute_errors: an evo Result type with trajectory and APE stats.
                * relative_errors: RPE stats.
        """
        self.boxplot_website.write_boxplot_website(stats)

    def add_dataset_to_website(self, dataset_name, pipeline_type, csv_results_path):
        """
        Write dataset results specified in csv_results_path.

        Call write_datasets_website to actually write this data in the website template.
            Args:
            - dataset_name: name of the dataset that the VIO stats come from.
            - csv_results_path: path to the directory where the csv results of the VIO
                pipeline are. This is typically the directory where there is
                a `traj_vio.csv` file together with an `output` directory where all
                the stats of the VIO are.
        """
        self.raw_output_website.add_dataset_to_website(dataset_name, csv_results_path)
        self.frontend_website.add_dataset_to_website(dataset_name, csv_results_path)
        self.detailed_performance_website.add_dataset_to_website(
            dataset_name, pipeline_type
        )

    def write_datasets_website(self):
        """Write website using the collected data."""
        # Write modified template inside the website package.
        self.raw_output_website.write_datasets_website()
        self.frontend_website.write_datasets_website()
        self.detailed_performance_website.write_datasets_website()


class PdfLoaderWebsiteBuilder:
    """PDF report page."""

    def __init__(self, jinja_env, website_output_path):
        """Construct a pdf report page."""
        # Website template
        self.detailed_performance_template = jinja_env.get_template(
            "detailed_performance_template.html"
        )

        # Generate Website output path
        self.website_output_path = website_output_path

        # We will store html snippets of each dataset indexed by dataset name in these
        # dictionaries. Each dictionary indexes the data per pipeline_type in turn:
        self.detailed_performance_html = dict()

    def add_dataset_to_website(self, dataset_name, pipeline_type, pdf_name="plots.pdf"):
        """Add dataset information from results."""
        self.detailed_performance_html[dataset_name] = os.path.join(
            dataset_name, pipeline_type, pdf_name
        )

    def write_datasets_website(self, website_name="detailed_performance.html"):
        """Write website to file."""
        # Write modified template inside the website package.
        with open(os.path.join(self.website_output_path, website_name), "w") as output:
            output.write(
                self.detailed_performance_template.render(
                    datasets_pdf_path=self.detailed_performance_html
                )
            )


class BoxplotWebsiteBuilder:
    """Page builder for boxplots."""

    def __init__(self, jinja_env, website_output_path, traj_vio_csv_name):
        """Construct the page builder from jina templates."""
        # Get Website template html
        self.boxplot_template = jinja_env.get_template("vio_performance_template.html")
        self.website_output_path = website_output_path

    def write_boxplot_website(self, stats):
        """Write page to file."""
        # Save html_div inside website template using Jinja2
        with open(
            os.path.join(self.website_output_path, "vio_ape_euroc.html"), "w"
        ) as output:
            # Write modified template inside the website package.
            output.write(
                self.boxplot_template.render(boxplot=self.__get_boxplot_as_html(stats))
            )

    def __get_boxplot_as_html(self, stats):
        """Return a plotly boxplot in html."""
        # Generate plotly figure
        fig = draw_ape_boxplots_plotly(stats)
        # Get HTML code for the plotly figure
        return get_fig_as_html(fig)


class FrontendWebsiteBuilder:
    """Builds frontend webiste page."""

    def __init__(self, jinja_env, website_output_path):
        """Make a frontend page."""
        # Get Website template html
        self.frontend_template = jinja_env.get_template("datasets_template.html")
        # Generate Website output path
        self.website_output_path = website_output_path
        # We will store html snippets of each dataset indexed by dataset name in these
        # dictionaries. Each dictionary indexes the data per pipeline_type in turn:
        self.frontend_html = dict()

    def add_dataset_to_website(self, dataset_name, csv_results_path):
        """
        Add dataset to page.

        Writes dataset results specified in csv_results_path.

        Call write_datasets_website to actually write this data in the website template.

        Args:
            dataset_name: name of the dataset that the VIO stats come from
            csv_results_path: path to the directory where the csv results are
        """
        self.frontend_html[dataset_name] = self.__get_frontend_results_as_html(
            os.path.join(csv_results_path, "output_frontend_stats.csv")
        )

    def write_datasets_website(self):
        """Write website to file."""
        # Write modified template inside the website package.
        with open(
            os.path.join(self.website_output_path, "frontend.html"), "w"
        ) as output:
            output.write(
                self.frontend_template.render(datasets_html=self.frontend_html)
            )

    def __get_frontend_results_as_html(self, csv_frontend_path, show_figures=False):
        """
        Read and plot output_frontend_stats.csv.

        Args:
        - csv_frontend_path: path to the output_frontend_stats.csv file

        Returns:
        - HTML data for all plots
        """
        df_stats = pd.read_csv(csv_frontend_path, sep=",", index_col=False)
        fig_html = get_fig_as_html(draw_feature_tracking_stats(df_stats, show_figures))
        fig_html += get_fig_as_html(
            draw_mono_stereo_inliers_outliers(df_stats, show_figures)
        )
        return fig_html


class RawOutputWebsiteBuilder:
    """Make a raw website."""

    def __init__(self, jinja_env, website_output_path, traj_vio_csv_name):
        """Make raw dataset output page."""
        # Get Website template html
        self.datasets_template = jinja_env.get_template("datasets_template.html")

        # Generate Website output path
        self.website_output_path = website_output_path

        # We will store html snippets of each dataset indexed by dataset name in these
        # dictionaries. Each dictionary indexes the data per pipeline_type in turn:
        self.datasets_html = dict()

        self.traj_vio_csv_name = traj_vio_csv_name

    def add_dataset_to_website(self, dataset_name, csv_results_path):
        """
        Write dataset results specified in csv_results_path.

        Call write_datasets_website to actually write this data in the website template.

        Args:
        - dataset_name: name of the dataset that the VIO stats come from.
        - csv_results_path: path to the directory where the csv results are
        """
        self.datasets_html[dataset_name] = self.__get_dataset_results_as_html(
            dataset_name, os.path.join(csv_results_path, self.traj_vio_csv_name)
        )

    def write_datasets_website(self):
        """Write website using the collected data."""
        # Write modified template inside the website package.
        with open(
            os.path.join(self.website_output_path, "datasets.html"), "w"
        ) as output:
            output.write(
                self.datasets_template.render(datasets_html=self.datasets_html)
            )

    def __get_dataset_results_as_html(
        self, dataset_name, csv_results_path, show_figures=False
    ):
        """Read dataset results."""
        df = pd.read_csv(csv_results_path)

        x_id = "#timestamp"

        fig = make_subplots(
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
            title_text="Raw VIO Output for dataset: %s" % dataset_name,
            template="plotly_white",
        )

        # Col 1
        # Get VIO position
        y_ids = ["x", "y", "z"]
        row = 1
        col = 1
        self.__plot_multi_line(df, x_id, y_ids, fig, row, col)

        # Get VIO orientation
        y_ids = ["qw", "qx", "qy", "qz"]
        row = 2
        col = 1
        self.__plot_multi_line(df, x_id, y_ids, fig, row, col)

        # Get VIO velocity
        y_ids = ["vx", "vy", "vz"]
        row = 3
        col = 1
        self.__plot_multi_line(df, x_id, y_ids, fig, row, col)

        # Col 2
        # Plot 3D Trajectory
        row = 1
        col = 2
        fig.add_trace(
            go.Scatter3d(
                x=df["x"],
                y=df["y"],
                z=df["z"],
                mode="lines+markers",
                marker=dict(
                    size=5,
                    # set color to an array/list of desired values
                    color=df["#timestamp"],
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.8,
                ),
            ),
            row=row,
            col=col,
        )
        fig.update_layout(
            scene=dict(
                annotations=[
                    dict(
                        showarrow=False,
                        x=df["x"][0],
                        y=df["y"][0],
                        z=df["z"][0],
                        text="Start",
                        xanchor="left",
                        xshift=10,
                        opacity=0.9,
                    ),
                    dict(
                        showarrow=False,
                        x=df["x"].iloc[-1],
                        y=df["y"].iloc[-1],
                        z=df["z"].iloc[-1],
                        text="End",
                        xanchor="left",
                        xshift=10,
                        opacity=0.9,
                    ),
                ],
                xaxis_showspikes=False,
                yaxis_showspikes=False,
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1),
            )
        )

        # Get VIO gyro bias
        y_ids = ["bgx", "bgy", "bgz"]
        row = 2
        col = 2
        self.__plot_multi_line(df, x_id, y_ids, fig, row, col)

        # Get VIO accel bias
        y_ids = ["bax", "bay", "baz"]
        row = 3
        col = 2
        self.__plot_multi_line(df, x_id, y_ids, fig, row, col)

        if show_figures:
            fig.show()

        return get_fig_as_html(fig)

    def __plot_multi_line(self, df, x_id, y_ids, fig=None, row=None, col=None):
        """
        Plot a multi line plotly plot from a pandas dataframe df.

        Uses the x axis the dataframe column with id `x_id` and using on the
        the y axis the dataframe columnS specified in the array of ids `y_ids`

        Args:
        - df: PandasDataframe with the data
        - x_id: column id in the pandas dataframe containing the data for the x axis.
        - y_ids: column id in the pandas dataframe containing the data for the y axis.
        This can have multiple entries.

        Optional Args:
        - fig: plotly figure where to add the line plots, this allows updating figures.
        - row: for multiplot figures, coordinates where to put the figure
        - col: for multiplot figures, coordinates where to put the figure

        Returns:
        - Plotly figure handle
        """
        # How to draw the lines
        mode = "lines+markers"

        if fig is None:
            fig = go.Figure()
        assert x_id in df
        for y_id in y_ids:
            assert y_id in df
            fig.add_trace(
                go.Scatter(x=df[x_id], y=df[y_id], mode=mode, name=y_id),
                row=row,
                col=col,
            )
        return fig
