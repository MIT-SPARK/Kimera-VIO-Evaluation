import os

import pandas as pd 

import plotly
import plotly.express as px
import plotly.graph_objects as go

from jinja2 import Environment, PackageLoader, select_autoescape

from evaluation.tools import draw_ape_boxplots_plotly
import website

class WebsiteBuilder:
    def __init__(self):
        """ Reads a template html website inside the `templates` directory of
        a `website` python package (that's why we call `import website`, which
        is a package of this project), and writes down html code with plotly figures.
        """
        # Initialize Jinja2
        self.env = Environment(
            loader=PackageLoader('website', 'templates'),
            autoescape=select_autoescape(['html', 'xml'])
        )
        # Get Website template html
        self.template = self.env.get_template('vio_performance_template.html')
        # Generate Website output path
        self.website_output_path = os.path.join(os.path.dirname(website.__file__), "vio_ape_euroc.html")

    def write_website(self, stats, csv_results_path):
        """ Writes website using overall stats, and optionally the original data of a dataset run
        specified in csv_results_path.
            Args:
            - stats: a nested dictionary with the statistics and results of all pipelines:
                * First level ordered with dataset_name as keys:
                * Second level ordered with pipeline_type as keys:
                * Each stats[dataset_name][pipeline_type] value has:
                    * absolute_errors: an evo Result type with trajectory and APE stats.
                    * relative_errors: RPE stats.
            - csv_results_path: path to the csv raw results of the VIO pipeline with header:
                    #timestamp	x	y	z	qw	qx	qy	qz	vx	vy	vz	bgx	bgy	bgz	bax	bay	baz
        """
        # Save html_div inside website template using Jinja2
        with open(self.website_output_path, "w") as output:
            # Write modified template inside the website package.
            output.write(self.template.render(boxplot=self.__get_boxplot_as_html(stats),
                                              dataset_plots=self.__get_dataset_results_as_html(csv_results_path)))

    def __get_boxplot_as_html(self, stats):
        """ Returns a plotly boxplot in html 
            Args:
            - stats: a nested dictionary with the statistics and results of all pipelines:
                * First level ordered with dataset_name as keys:
                * Second level ordered with pipeline_type as keys:
                * Each stats[dataset_name][pipeline_type] value has:
                    * absolute_errors: an evo Result type with trajectory and APE stats.
                    * relative_errors: RPE stats.
            Returns:
            - html string to embed the boxplot in a website
        """
        # Generate plotly figure
        fig = draw_ape_boxplots_plotly(stats)
        # Get HTML code for the plotly figure
        return self.__get_fig_as_html(fig)

    def __get_dataset_results_as_html(self, csv_results_path, show_figures=False):
        """  Reads traj_vio.csv file with the following header:
                    #timestamp	x	y	z	qw	qx	qy	qz	vx	vy	vz	bgx	bgy	bgz	bax	bay	baz
            And plots lines for each group of data: position, orientation, velocity...
            Returns:
            - HTML data for all plots
        """
        df = pd.read_csv(csv_results_path) 

        x_id = '#timestamp'

        # Get VIO position
        y_ids = ['x', 'y', 'z']
        fig = self.__plot_multi_line(df, x_id, y_ids)
        position_html = self.__get_fig_as_html(fig)
        if show_figures:
            fig.show()

        # Get VIO orientation
        y_ids = ['qw', 'qx', 'qy', 'qz']
        fig = self.__plot_multi_line(df, x_id, y_ids)
        orientation_html = self.__get_fig_as_html(fig)
        if show_figures:
            fig.show()

        # Get VIO velocity
        y_ids = ['vx', 'vy', 'vz']
        fig = self.__plot_multi_line(df, x_id, y_ids)
        vel_html = self.__get_fig_as_html(fig)
        if show_figures:
            fig.show()

        # Get VIO gyro bias
        y_ids = ['bgx', 'bgy', 'bgz']
        fig = self.__plot_multi_line(df, x_id, y_ids)
        gyro_bias_html = self.__get_fig_as_html(fig)
        if show_figures:
            fig.show()

        # Get VIO accel bias
        y_ids = ['bax', 'bay', 'baz']
        fig = self.__plot_multi_line(df, x_id, y_ids)
        accel_bias_html = self.__get_fig_as_html(fig)
        if show_figures:
            fig.show()

        return position_html + orientation_html + vel_html + gyro_bias_html + accel_bias_html

    def __get_fig_as_html(self, fig):
        """ Gets a plotly figure and returns html string to embed in a website
        """
        return plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')


    def __plot_multi_line(self, df, x_id, y_ids):
        """
            Plots a multi line plotly plot from a pandas dataframe df, using 
            on the x axis the dataframe column with id `x_id` and using on the
            the y axis the dataframe columnS specified in the array of ids `y_ids`

            Args:
            - df: PandasDataframe with the data
            - x_id: column id in the pandas dataframe containing the data for the x axis.
            - y_ids: column id in the pandas dataframe containing the data for the y axis.
            This can have multiple entries.

            Returns:
            - Plotly figure handle
        """
        # How to draw the lines
        mode = 'lines+markers'

        fig = go.Figure()
        assert(x_id in df)
        for y_id in y_ids:
            assert(y_id in df)
            fig.add_trace(go.Scatter(x=df[x_id], y=df[y_id],
                                    mode=mode, name=y_id))
        return fig

