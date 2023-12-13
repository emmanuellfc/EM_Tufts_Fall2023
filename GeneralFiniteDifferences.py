import pygalmesh
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Iterable
from typing import Callable
from typing import Optional
import math

# @title General Finite Differences Method internals.
# Rperesents an individual node of the graph data structure representation of the mesh.
class MeshPoint: pass
class MeshPoint:
  def __init__(self,pos: tuple = (0,0)):
    self.pos = tuple(pos) # (x,y) position of the node.
    self.pot: Optional[float] = None # Solution of potential for this node point.
    self.elec: Optional[tuple] = None # Electric field for this node point (never used in the solution, can be computed upon request after finding a potential solution).
    self.is_boundary = False # Is this node associated with a boundary condition?
  def __getitem__(self,index):
    return self.pos[index]
  def __eq__(self,other: MeshPoint):
    return self.pos is other.pos
  def __hash__(self):
    return hash(self.pos)

  # The "gamma" function is one half the sum of inverse square distances between these points. i.e. 0.5(1/(x1-x2)^2 + 1/(y1-y2)^2).
  def gamma(self,other: MeshPoint) -> float:
    if  len(self.pos) is not len(other.pos):
      raise Exception('Incompatable dimensionality computing gamma: ' + self.pos + ' and ' + other.pos)
    tmp: float = 0
    for dim in range(len(self.pos)):
      # Don't divide by 0.
      if math.isclose(self.pos[dim],other.pos[dim]):
        #print(f'Preventing a division by zero... pos = {self.pos}, other = {other.pos}')
        continue
      tmp += pow(self.pos[dim]-other.pos[dim],-2)
    return tmp#/2 # Small optimization - the /2 actually divides off later in the math.

# Generic error function for some analytic function.
#def generic_error(point:MeshPoint,analytic_func:Callable[[MeshPoint],Optional[float]]) -> float:
#  analytic = analytic_func(point)
#  if analytic is None: return 0
#  else: return point.pot - analytic_func(point)

# Represents a mesh (contains a graph) for the system we're trying to solve.
class FiniteDifferences:
  def __init__(self):
    self.hush: bool = False # Hush text outputs.
    self.max_edge_size: float = None # What maximum edge size should we pass to the mesher?
    self.num_neighbors: int = None # How many neighbors should we try to make per non-boundary node?
    self.boundary_conditions: Callable[[MeshPoint],Optional[float]] = None # A function that takes a node and if boundary node: return pot; else: return None.
    self.edge_points: list = None # List of (x,y) points that form the boundary of the 2D space.
    self.mesh = None # Pygalmesh mesh.points list.
    self.mesh_points: list = None # List of non-boundary condition MeshPoints that form the region of 2D space.
    self.dirichlet_points: list = None # List of MeshPoints that are boundary condition points (only Dirichlet points).
    self.all_points: list = None # Simply self.mesh_points + self.dirichlet_points after they're constructed.
    self.graph = nx.Graph() # Graph data structure representing the region. Nodes of the graph are MeshPoint objects.
    self.systems_matrix = None # m x m matrix where m is the number of non-boundary points in the space. (m is how many points we need to solve for.)
    self.systems_rhs = None # m x 1 vector that serves as the right hand side of the system of linear equations.
    self.solution = None # Solution result of the systems matrix (m x 1 vector).

  def set_hush(self,val: bool):
    self.hush = val
  def set_max_edge_size(self,max: float):
    self.max_edge_size = max
  def set_num_neighbors(self,num: int):
    self.num_neighbors = num

  # Inform the method of boundary conditions that exist for the problem.
  # We expect a function that takes a MeshPoint object and returns a boundary potential for it if it should have one. Expects None for "not a boundary point."
  def inform_with_boundary_conditions(self,conditions: Callable[[MeshPoint],Optional[float]]):
    self.boundary_conditions = conditions

  # Manually inject a list of (x,y) points instead of calling the mesher.
  def inform_with_manual_mesh(self,manual_mesh: Iterable[tuple]):
    if self.boundary_conditions is None:
      raise Exception('Error: Need to inform with boundary conditions before providing a manual mesh.')
    # First store a copy of the mesh points.
    self.mesh = list(manual_mesh)
    self.__assemble_boundary_points()

  # Make a list of points from some edge conditions. This makes a surface "mesh" from a boundary definition. The boundary is a list of points.
  def make_points_from_boundary(self,boundary: Iterable[tuple]):
    if not self.hush: print('Generating a mesh from the boundary provided...')
    if self.max_edge_size is None:
      raise Exception('Error: Need to inform with a max edge size before providing boundary points.')
    if self.boundary_conditions is None:
      raise Exception('Error: Need to inform with boundary conditions before providing boundary points.')
    # First, store the edge points that we were passed.
    self.edge_points = list(boundary)
    # The constraints we pass to pygalmesh need to look like [(0,1),(1,2),...(n-2,n-1),(n-1,0)]. The last constraint closes the boundary.
    constraints = [ (i-1,i) for i in range(1,len(self.edge_points)) ] + [(len(self.edge_points)-1,0)]
    # Ask pygalmesh to make a mesh for us.
    self.mesh = pygalmesh.generate_2d(
      self.edge_points,
      constraints,
      max_edge_size=self.max_edge_size,
      num_lloyd_steps=10,
    ).points
    self.__assemble_boundary_points()

  # Not intended to be called from outside the class (weird I know). Helper function to assemble internal data structures.
  def __assemble_boundary_points(self):
    # Store the points as MeshPoint objects.
    self.mesh_points = []
    self.dirichlet_points = []
    for pt in self.mesh:
      mesh_point = MeshPoint(pt)
      boundary_result = self.boundary_conditions(mesh_point)
      mesh_point.is_boundary = not boundary_result is None
      if mesh_point.is_boundary:
        mesh_point.pot = boundary_result
        self.dirichlet_points.append(mesh_point)
      else:
        self.mesh_points.append(mesh_point)
    # Sanity checks: These lists shouldn't ever be empty.
    if not self.mesh_points: raise Exception('Error: Mesh point list is empty.')
    if not self.dirichlet_points: raise Exception('Error: Boundary condition point list is empty.')
    # Build the convenience list that's the concactenation of the non-boundary and boundary lists.
    self.all_points = self.mesh_points + self.dirichlet_points

  # Only callable once a mesh exists. Build a graph data structure from a list of points. Don't make neighbors out of points that are too far away.
  def make_graph_from_points(self):
    if not self.hush: print(f'Building a connectivity graph for {len(self.all_points)} points...')
    if self.num_neighbors is None:
      raise Exception('Error: Need to inform with target number of node neighbors before building connectivity graph.')
    # First, add all mesh points to the graph as nodes.
    for point in self.all_points:
      self.graph.add_node(point)
    # Now, for each point...
    for point in self.all_points:
      # Optimization: If this is a boundary point, don't try to make connections to others. Let other points make connections to this one, though.
      if point.is_boundary: continue
      # Get gamma values for all of the other points...
      # This makes a copy of our point list with the closest points first.
      distance_fml = lambda other_point:np.linalg.norm([ point.pos[0]-other_point.pos[0],point.pos[1]-other_point.pos[1] ])
      #points_gammas = sorted(self.all_points,key=distance_fml)
      #points_gammas.remove(point)

      # New algorithm that should be faster: First find all points with Euclidian distance < double the max edge size?
      nearby_points = []
      for other_point in self.all_points:
        if point is other_point: continue # Don't link a point to itself.
        if distance_fml(other_point) < 1.5*self.max_edge_size:
          nearby_points.append(other_point)
      nearby_points = sorted(nearby_points,key=distance_fml)
      if not nearby_points:
        raise Exception(f'Error: Mesh point at {point.pos} is not being connected to any others.')

      # Keep the nearest <neighbor hyperparameter> points...
      # Add edges between this point and "nearby" points.
      for other_point in nearby_points[:self.num_neighbors]:
        # Make the edge between these points.
        self.graph.add_edge(point,other_point,gamma=point.gamma(other_point))

  # Only callable once a graph exists. Make the C matrix element by element.
  def make_systems_matrix(self):
    # Initialize the systems matrix to the correct size (m x m where m is our number of non-boundary points, that is, the matrix is only necessary for points we're solving for).
    m = len(self.all_points)
    if not self.hush: print(f'Building a matrix {m} by {m} to solve the system...')
    self.systems_matrix = np.array([np.zeros(m) for j in range(0,m)])
    self.systems_rhs = np.zeros(m)
    # c_{i,i} = -1, so fill that diagonal now.
    for i in range(0,m): self.systems_matrix[i][i] = -1
    # Iterate over nodes of the graph...
    for i_point,point in enumerate(self.all_points):
      if point.is_boundary: # For boundary nodes, the row should only have the diagonal element set, and this forms the equation x_i = pot_i. (This is a Dirichlet condition.)
        self.systems_matrix[ i_point ][ i_point ] = 1
        # The RHS for this index needs to be set to the pot to finish this boundary condition.
        self.systems_rhs[ i_point ] = point.pot
        continue
      # Condition for if this is NOT a boundary point (one we need to solve for).
      # ... and get the edges for this point (its neighbors).
      # For each of these, c_{i,j} = gamma_{i,j} / (k el neighbors(i) sum gamma_{i,k}).
      # self.graph[point] is a dict of { neighbor_point : attributes } elements. We expect attributes to contain a gamma attribute.
      neighbors = self.graph[point]
      # First get the gamma sum for this point (depends on all neighbors).
      gamma_sum: float = 0
      for neighbor in neighbors:
        gamma_sum += self.graph[point][neighbor]['gamma']
      # Iterate over neighbors now and fill in c_{i,j} elements.
      for neighbor in neighbors:
        # What is the index of the neighbor?
        i_neighbor = self.all_points.index(neighbor)
        self.systems_matrix[ i_point ][ i_neighbor ] = self.graph[point][neighbor]['gamma'] / gamma_sum

  # Only callable once a C matrix exists. Try to solve the C matrix.
  def solve_systems_matrix(self):
    if not self.hush: print(f'Solving the system...')
    self.solution = np.linalg.solve(self.systems_matrix,self.systems_rhs)
    # Store pot values in our mesh points themselves.
    for i_point,point in enumerate(self.all_points):
      point.pot = self.solution[i_point]

  # Debug: Draw a plot that shows the connectivity of the mesh.
  def draw_connectivity(self):
    # Build a matplotlib.collections.LineCollection which will be a big "list" of line segments.
    list_of_lines = []
    for point in self.all_points:
      for neighbor in self.graph[point]:
        list_of_lines.append([ (point.pos[0],point.pos[1]),(neighbor.pos[0],neighbor.pos[1]) ])
    line_collection = LineCollection(list_of_lines)
    fig,ax = plt.subplots()
    ax.set_aspect('equal')
    def color_func(point:MeshPoint) -> float:
      if point.is_boundary: return 0
      else: return 1
    plt.scatter( [ pt[0] for pt in self.all_points ],
                 [ pt[1] for pt in self.all_points ],
               c=[ color_func(pt) for pt in self.all_points ],
                 cmap='magma',
                 marker='.')
    ax.add_collection(line_collection)

  # Return the RMS error for this solution compared to some analytic solution (analytic is a callable function that takes a point and returns the analytic pot.)
  def get_rms_error(self,analytic:Callable[[MeshPoint],Optional[float]]) -> float:
    # Iterate over points in this solution and get the error for each.
    errors = np.array([ ( 0 if analytic(point) is None else point.pot - analytic(point) ) for point in self.mesh_points ])
    return np.sqrt( (errors**2).mean() ) # Return RMS error, that is, each error element squared, the mean of that list, then the square root of that mean.

  # Compute an approximate electric field given a potential solution.
  def compute_electric_field(self):
    if not self.hush: print('Approximating the electric field...')
    if self.solution is None:
      raise Exception('Error: Must have a completed potential solution to compute an electric field approximation.')
    # We approximate the electric field (gradient) like dV_i/dx \approx average_over_neighbors_of_i( x_hat_j deltaV_i,j/deltaX_i,j )...
    for pt in self.all_points:
      # Compute x and y components separately... This means we don't need to handle unit vectors!
      # Make two lists of values that we'll eventually average over.
      x_considerations = [ (neighbor.pot - pt.pot)/(pt.pos[0] - neighbor.pos[0]) for neighbor in self.graph[pt] ]
      y_considerations = [ (neighbor.pot - pt.pot)/(pt.pos[1] - neighbor.pos[1]) for neighbor in self.graph[pt] ]
      pt.elec = [0,0]
      # Safeguard against empty consideration lists.
      if x_considerations: pt.elec[0] = np.mean(x_considerations)
      if y_considerations: pt.elec[1] = np.mean(y_considerations)
      pt.elec = tuple(pt.elec)

  # Profit???
