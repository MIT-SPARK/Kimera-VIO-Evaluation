#!/usr/bin/env python

import numpy as np
import os
import os.path
from os import path
import glog as log
import copy

from tqdm.notebook import tqdm

import open3d as o3d
import pandas as pd

from evaluation.tools.mesh import Mesh

# Rotation matrices:
# East North Up (ENU) frame to Unity's world frame of reference
enu_R_unity = np.array([[1, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0]])
unity_R_enu = np.transpose(enu_R_unity)

# Right Handed frame to Unity's Left Handed frame of reference
righthand_R_lefthand = np.array([[1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, 1]])
lefthand_R_righthand = np.transpose(righthand_R_lefthand)

class ICP:
    """
        Performs point-to-point ICP between two pointclouds

        Methods:
        --------
            align:
                Aligns given pointclouds
    """

    def __init__(self, visualize=False):
        self.visualize = visualize
        # ICP params
        self.icp_threshold = 1.5
        self.trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0]])
    def align(self, est_pcl, gt_pcl):
        """
            Args:
                est_pcl: open3d.pcl
                    Estimated pointcloud
                gt_pcl: open3d.pcl
                    Ground-truth pointclous
            Returns:
                ICP registration result: correspondences and transformation
        """
        # Visualize initial registration problem
        if self.visualize:
            self.draw_registration_result(est_pcl, gt_pcl, self.trans_init)

        # Evaluate current fit between pointclouds
        print("Initial registration")
        evaluation = o3d.registration.evaluate_registration(est_pcl, gt_pcl, self.icp_threshold, self.trans_init)
        print(evaluation)

        # Actual p2p ICP
        print("Apply point-to-point ICP")
        reg_p2p = o3d.registration.registration_icp(
            est_pcl, gt_pcl, self.icp_threshold, self.trans_init,
            o3d.registration.TransformationEstimationPointToPoint(),
            o3d.registration.ICPConvergenceCriteria(max_iteration = 2000))
        print("Done with point-to-point ICP")
        correspondences = reg_p2p.correspondence_set

        #print(reg_p2p)
        #print("")

        #print("Transformation is:")
        #print(reg_p2p.transformation)
        #print("")

        #print("Correspondence Set:")
        #print(reg_p2p.correspondence_set)
        #print("")

        # Draw Registration Result
        if self.visualize:
            self.draw_registration_result(est_pcl, gt_pcl, reg_p2p.transformation)

        # Draw Only Correspondences
        if self.visualize:
            c2c_lines = o3d.geometry.create_line_set_from_point_cloud_correspondences(est_pcl, gt_pcl, correspondences)
            o3d.visualization.draw_geometries([c2c_lines])

        # Draw PointClouds and Correspondences
        if self.visualize:
            self.draw_correspondences(est_pcl, gt_pcl, c2c_lines)

        return reg_p2p


    # Visualization functions
    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def draw_correspondences(self, source, target, correspondences):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        #source_temp.paint_uniform_color([1, 0.706, 0])
        #target_temp.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([source_temp, #target_temp,
                                           correspondences])

class SemanticLabelToColorCSV:
    """
        Wrapper around the semantic label to color csv file to encapsulate the
        mapping between colors and semantic labels...
    """

    def __init__(self, semantic_labels_csv_path):
        # Import Semantic Labels
        df = pd.read_csv(semantic_labels_csv_path)
        #df

        # Normalize the csv colors
        self.normalized_df = copy.deepcopy(df)
        self.normalized_df['normalized_red'] = df['red'] / 255
        self.normalized_df['normalized_green'] = df['green'] / 255
        self.normalized_df['normalized_blue'] = df['blue'] / 255

        # this is to avoid errors when comparing floats
        self.normalized_df['normalized_red'] = self.normalized_df['normalized_red'].apply(lambda x: round(x, 5))
        self.normalized_df['normalized_green'] = self.normalized_df['normalized_green'].apply(lambda x: round(x, 5))
        self.normalized_df['normalized_blue'] = self.normalized_df['normalized_blue'].apply(lambda x: round(x, 5))

    # Generate table from color to id.
    def label_from_color(self, color):
        # TODO(Toni): do you need to round again? isn't color already rounded?
        norm_r = round(color[0], 5)
        norm_g = round(color[1], 5)
        norm_b = round(color[2], 5)
        # TODO(Toni): can be greatly optimized... TOO SLOW NOW
        # TODO(Toni): you are comparing floats with == ......
        label_list = self.normalized_df.loc[(self.normalized_df['normalized_red'] == norm_r) &
                                            (self.normalized_df['normalized_green'] == norm_g) &
                                            (self.normalized_df['normalized_blue'] == norm_b)]['id'].unique().tolist()
        if len(label_list) <= 0:
            print("Missing Semantic Label from Color: %f, %f, %f "%(color[0], color[1], color[2]))
            return []
        return label_list[0]


class MeshEvaluator:
    """
        A class used to compare an estimated vs a ground-truth mesh.
        Requires a csv file specifying what is the mapping between colors and object ids.
        (there are many colors to lead to the same object).

        Attributes
        ----------
        est_mesh_path : str
            Global path to the PLY file for the estimated mesh.
        gt_mesh_path : str
            Global path to the PLY file for the ground-truth mesh.
        semantic_labels_csv_path : str
            Global path to the csv file with the colors to object id mapping.

    """
    def __init__(self, est_mesh_path, gt_mesh_path, semantic_labels_csv_path, visualize=False):
        """
        Args:
            est_mesh_path : str
                Global path to the PLY file for the estimated mesh.
            gt_mesh_path : str
                Global path to the PLY file for the ground-truth mesh.
            semantic_labels_csv_path : str
                Global path to the csv file with the colors to object id mapping.
            visualize: bool
                Whether to visualize intermediate results (ICP registration, correspondences)
        """
        #print("Init MeshEvaluator")
        self.est_mesh_path = est_mesh_path
        self.gt_mesh_path = gt_mesh_path
        self.semantic_labels_csv_path = semantic_labels_csv_path
        self.visualize = visualize

        assert(path.exists(self.est_mesh_path))
        assert(path.exists(self.gt_mesh_path))
        assert(path.exists(self.semantic_labels_csv_path))

        # Init ICP class
        self.icp = ICP(self.visualize)

        # Import Semantic Labels
        self.semantic_mapping = SemanticLabelToColorCSV(self.semantic_labels_csv_path)

        #print("Loading Ground-truth mesh...")
        self.gt_mesh_original = Mesh(gt_mesh_path)
        #print("Loading Estimated mesh...")
        self.est_mesh_original = Mesh(est_mesh_path)

    def compare_meshes(self, number_of_mesh_samples=1000000, only_geometric=False):
        """
        Args:
            number_of_mesh_samples: int
                Since the gt_mesh has typically large triangles, in order to compare with the est_mesh,
                which has a large density of triangles, we sample the gt_mesh to obtain a pointcloud which
                we then compare with the est_mesh's vertices.
                This parameter controls how many samples do we extract from the gt_mesh.

        Returns:
            inlier_rmse: float
                Inlier RMSE of the estimated mesh wrt ground-truth mesh after ICP registration
            semantic_accuracy: float
                Percentage of correct semantic matches amongst all ICP correspondences
        """

        # First, transform Meshes to same frame of reference
        gt_mesh = copy.deepcopy(self.gt_mesh_original)
        est_mesh = copy.deepcopy(self.est_mesh_original)

        #######################################
        # Align Pointclouds Manually:
        gt_mesh.transform_left(enu_R_unity)
        #######################################

        # Visualize manual alignment
        if self.visualize:
            self.visualize_meshes(gt_mesh, est_mesh)

        # Get meshes' pointclouds
        gt_pcl = o3d.geometry.sample_points_uniformly(gt_mesh.mesh_o3d, number_of_mesh_samples)
        # Don't sample estimated mesh, just pick vertices, otw you'll be mixing colors...
        # est_pcl = o3d.geometry.sample_points_uniformly(est_mesh.mesh_o3d, number_of_mesh_samples)
        est_pcl = o3d.io.read_point_cloud(self.est_mesh_path)

        if self.visualize:
            self.visualize_pcls(gt_pcl, est_pcl)

        # Align pointclouds using ICP
        reg_p2p = self.icp.align(est_pcl, gt_pcl)

        if len(reg_p2p.correspondence_set) == 0:
            print("ICP registration failed! No inlier correspondences.")
            return 0, 0

        # Calculate geometric metrics using the ICP transformation
        inlier_rmse = reg_p2p.inlier_rmse
        print("Geometric inlier RMSE [m]: ")
        print(inlier_rmse)
        print(" ")

        # Calculate semantic metrics using the ICP correspondences
        semantic_accuracy = -1
        if not only_geometric:
            print("Calculating Semantic Accuracy...")
            print("Semantic Accuracy [%]: ")
            semantic_accuracy = self.calc_corresp(est_pcl, gt_pcl, reg_p2p.correspondence_set)
            print(semantic_accuracy)
            print(" ")

        return inlier_rmse, semantic_accuracy

    def calc_corresp(self, est_pcl, gt_pcl, correspondences):
        total_negative_matches = 0
        total_positive_matches = 0
        total_correspondences = len(correspondences)

        # Compare labels between correspondences:
        # Initialize dictionaries to 0:
        total_label_correspondences = {i:0 for i in self.semantic_mapping.normalized_df['id'].unique()}
        total_label_positive_matches = copy.deepcopy(total_label_correspondences)
        total_label_negative_matches = copy.deepcopy(total_label_correspondences)

        #print("Total number of correspondences: ", total_correspondences)
        for correspondence in tqdm(correspondences):
            assert(len(correspondence) == 2)
            assert(correspondence[0] < len(est_pcl.colors))
            assert(correspondence[1] < len(gt_pcl.colors))
            est_label_id = self.semantic_mapping.label_from_color(est_pcl.colors[correspondence[0]])
            gt_label_id = self.semantic_mapping.label_from_color(gt_pcl.colors[correspondence[1]])
            if not est_label_id or not gt_label_id:
                continue

            if est_label_id == gt_label_id:
                total_positive_matches += 1
                total_label_positive_matches[est_label_id] += 1
            else:
                total_negative_matches += 1
                total_label_negative_matches[est_label_id] += 1

        #print("Positive matches: ", total_positive_matches)
        #print("Negative matches: ", total_negative_matches)
        #print("Total correspondences: ", total_correspondences)
        if total_correspondences != total_negative_matches + total_positive_matches:
            print("Some colors' label couldn't be found...")
        assert(total_correspondences > 0)
        #print ("Positive [%]: ", (total_positive_matches / total_correspondences * 100))
        #print ("Negative [%]: ", (total_negative_matches / total_correspondences * 100))
        accuracy = float(total_positive_matches) / float(total_correspondences) * 100.0
        return accuracy

    ##### Visualization methods
    def visualize_meshes(self, gt_mesh, est_mesh):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().mesh_show_back_face = True
        gt_mesh.add_to_vis(vis)
        est_mesh.add_to_vis(vis)
        mesh_frame = o3d.geometry.create_mesh_coordinate_frame(size=4,
                                                               origin=[0, 0, 0])
        vis.add_geometry(mesh_frame)
        vis.run()
        vis.destroy_window()

    def visualize_pcls(self, gt_pcl, est_pcl):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().mesh_show_back_face = True
        vis.add_geometry(gt_pcl)
        vis.add_geometry(est_pcl)
        vis.run()
        vis.destroy_window()


def parser():
    import argparse
    basic_desc = "Evaluation of metric-semantic 3D mesh."

    shared_parser = argparse.ArgumentParser(
        add_help=True, description="{}".format(basic_desc))

    input_opts = shared_parser.add_argument_group("input options")

    input_opts.add_argument("gt_mesh_path", help="Path to the ground-truth ply file with the mesh.",
                            default="./gt_mesh.ply")
    input_opts.add_argument("est_mesh_path", help="Path to the estimated ply file with the mesh.",
                            default="./est_mesh.ply")
    input_opts.add_argument("semantic_labels_to_color_csv_path",
                            help="Path to the estimated csv file with the semantic label to color mapping.",
                            default="./semantic_label_to_color.csv")
    input_opts.add_argument("--visualize", action="store_true",
                            help="Visualize meshes, ICP, and correspondences.")

    main_parser = argparse.ArgumentParser(
        description="{}".format(basic_desc))
    sub_parsers = main_parser.add_subparsers(dest="subcommand")
    sub_parsers.required = True
    return shared_parser

if __name__ == '__main__':
    import argcomplete
    import sys

    # Parse args
    log.setLevel("INFO")
    parser = parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Run evaluation
    mesh_eval = MeshEvaluator(args.est_mesh_path, args.gt_mesh_path, args.semantic_labels_to_color_csv_path, args.visualize)
    number_of_mesh_samples=1000
    mesh_eval.compare_meshes(number_of_mesh_samples)

    # TODO(Toni): write the results of compare_meshes to a file with the name of the dataset/meshes.
