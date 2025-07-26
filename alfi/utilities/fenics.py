from fenics import *
from fenics_adjoint import *
from dolfin import *

import numpy as np


def interval_mesh(spatial):
    """
    Creates a 1D interval mesh based on the given spatial points.

    Parameters:
        spatial (list of float): A list of spatial coordinates defining the vertices of the interval mesh.

    Returns:
        dolfin.Mesh: A FEniCS Mesh object representing the 1D interval mesh.

    Notes:
        - The function uses `MeshEditor` to manually construct the mesh.
        - Each point in the `spatial` list is converted into a vertex, and consecutive vertices are connected to form cells (intervals).
    """
    domain_vertices = [Point(x) for x in spatial]
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh, 'interval', 1, 1)
    editor.init_vertices(len(domain_vertices))
    editor.init_cells(len(domain_vertices) - 1)

    for i, vertex in enumerate(domain_vertices):
        editor.add_vertex(i, vertex)
        if i < len(domain_vertices) - 1:
            editor.add_cell(i, np.array([i, i + 1]))

    return mesh
