"""Mesh wrapper."""
import os
import numpy as np


class Mesh:
    """Wrapper around an open3d mesh."""

    def __init__(self, filepath):
        """Read a mesh from file."""
        import open3d as o3d

        os.path.isfile(filepath)
        self.mesh_o3d = o3d.io.read_triangle_mesh(filepath)

    def visualize(self):
        """Draw a mesh."""
        import open3d as o3d

        self.mesh_o3d.compute_vertex_normals()
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().mesh_show_back_face = True
        self.add_to_vis(vis)
        mesh_frame = o3d.geometry.create_mesh_coordinate_frame(size=4, origin=[0, 0, 0])
        vis.add_geometry(mesh_frame)
        vis.run()
        vis.destroy_window()

    def add_to_vis(self, vis):
        """Add the mesh to the visualization `vis` object."""
        self.mesh_o3d.compute_vertex_normals()
        vis.add_geometry(self.mesh_o3d)

    def print_mesh(self):
        """Show mesh in console."""
        print("Testing mesh in open3d ...")
        print(self.mesh_o3d)
        print(np.asarray(self.mesh_o3d.vertices))
        print(np.asarray(self.mesh_o3d.triangles))
        print("")

    def transform_left(self, rotation_matrix):
        """Left multiply matrix."""
        import open3d as o3d

        # TODO(Toni): there is a transform method!!!
        assert isinstance(rotation_matrix, np.ndarray)
        assert np.size(rotation_matrix, 0) == 3
        assert np.size(rotation_matrix, 1) == 3
        # print("Transforming mesh according to left matrix:")
        # print(rotation_matrix)
        rotated_vertices = rotation_matrix.dot(
            np.transpose(np.asarray(self.mesh_o3d.vertices))
        )
        self.mesh_o3d.vertices = o3d.utility.Vector3dVector(
            np.transpose(rotated_vertices)
        )

    def transform_right(self, rotation_matrix):
        """Right multiply matrix."""
        import open3d as o3d

        assert isinstance(rotation_matrix, np.ndarray)
        assert np.size(rotation_matrix, 0) == 3
        assert np.size(rotation_matrix, 1) == 3
        print("Transforming mesh according to right matrix:")
        print(rotation_matrix)
        rotated_vertices = np.asarray(self.mesh_o3d.vertices).dot(rotation_matrix)
        self.mesh_o3d.vertices = o3d.utility.Vector3dVector(rotated_vertices)
