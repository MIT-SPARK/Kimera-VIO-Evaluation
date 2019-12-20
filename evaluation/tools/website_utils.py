import os

import plotly
import plotly.express as px
import plotly.graph_objects as go

from jinja2 import Environment, PackageLoader, select_autoescape

from evaluation.tools import draw_ape_boxplots_plotly
import website

def write_boxplot_to_website(stats, results_dir):
    """ Reads a template html website inside the `templates` directory of
    a `website` python package (that's why we call `import website`, which
    is a package of this project),
    and writes a plotly boxplot in it using Jinja2.
    """
    # Generate plotly figure
    figure = draw_ape_boxplots_plotly(stats)
    # Get HTML code for the plotly figure
    html_div = plotly.offline.plot(figure, include_plotlyjs=False, output_type='div')

    # Initialize Jinja2
    env = Environment(
        loader=PackageLoader('website', 'templates'),
        autoescape=select_autoescape(['html', 'xml'])
    )

    # Save html_div inside webiste template using Jinja2
    website_path = os.path.dirname(website.__file__)
    template = env.get_template('vio_performance_template.html')
    with open(os.path.join(website_path, "vio_ape_euroc.html"), "w") as output:
        # Write modified template inside the website package.
        output.write(template.render(plotly=html_div))
