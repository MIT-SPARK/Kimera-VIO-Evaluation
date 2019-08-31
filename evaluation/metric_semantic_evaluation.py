#!/usr/bin/env python3

import os
import open3d
import glog as log

def visualize_3d_mesh(path_to_ply):
    triangles = open3d.io.read_triangle_mesh(path_to_ply)
    open3d.visualization.draw_geometries([triangles])

def parser():
    import argparse
    basic_desc = "Full evaluation of SPARK VIO pipeline (APE trans + RPE trans + RPE rot) metric app"

    shared_parser = argparse.ArgumentParser(add_help=True, description="{}".format(basic_desc))

    input_opts = shared_parser.add_argument_group("input options")

    input_opts.add_argument("path_to_ply", help="Path to the ply file with the mesh.",
                            default="./mesh.ply")

    main_parser = argparse.ArgumentParser(description="{}".format(basic_desc))
    sub_parsers = main_parser.add_subparsers(dest="subcommand")
    sub_parsers.required = True
    return shared_parser

import argcomplete
import sys
if __name__ == '__main__':
    log.setLevel("INFO")
    parser = parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    if visualize_3d_mesh(args.path_to_ply):
        sys.exit(os.EX_OK)
    else:
        raise Exception("Metric-Semantic Evaluation failed.")
