from setuptools import setup
import fastentrypoints

setup(
    name="kimera_vio_evaluation",
    version="0.2",
    description="Code for evaluating the performance of the Kimera VIO pipeline",
    url="https://github.com/ToniRV/Kimera-VIO-Evaluation",
    author="Antoni Rosinol",
    author_email="arosinol@mit.edu",
    license="MIT",
    packages=["evaluation", "evaluation.tools", "website"],
    install_requires=[
        "numpy",
        "glog",
        "tqdm",
        "ruamel.yaml",
        "evo-1",
        "open3d",
        "plotly",
        "chart_studio",
        "pandas",
        "tk",
    ],
    extras_require={"notebook": ["jupyter", "jupytext"]},
    zip_safe=False,
)
