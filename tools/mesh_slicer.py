import numpy as np
import trimesh
import math
import kdtree
from CGAL.CGAL_Kernel import Point_3, Vector_3, Segment_3, Plane_3
from CGAL.CGAL_AABB_tree import AABB_tree_Polyhedron_3_Halfedge_handle as ABTH
# from CGAL.CGAL_AABB_tree import
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3, Polyhedron_modifier, Polyhedron_3_Halfedge_handle
import CGAL.CGAL_Polygon_mesh_processing as PMP
import CGAL.CGAL_Surface_mesher as CGMesh
from CGAL.CGAL_Surface_mesher import Surface_mesh_default_criteria_3 as Mesh_3
from plyfile import PlyData, PlyElement
# mesh = trimesh.load_mesh(mesh_dir)

in_plane_normal = True

def get_bbox(vertices):
    min_p = np.min(vertices, axis=0)
    max_p = np.max(vertices, axis=0)
    # print(min_p)
    # print(max_p)
    return min_p, max_p

def generate_parallel_slice_planes(min_p, max_p, s_num, axis):
    min_val = min_p[axis]
    max_val = max_p[axis]
    step = (max_val - min_val) / (s_num - 1)
    # point = np.array([0, 0, 0])
    planes = []
    for i in range(s_num):
        p_val = min_val + step * i 
        if i == 0:
            p_val += 0.2 * step
        if i == s_num -1:
            p_val -= 0.2 * step 
        p_point = [0, 0, 0]
        p_point[axis] = p_val
        p_normal = [0, 0, 0]
        p_normal[axis] = 1.0
        p_point = Point_3(p_point[0], p_point[1], p_point[2])
        p_normal = Vector_3(p_normal[0], p_normal[1], p_normal[2])
        cut_plane = Plane_3(p_point, p_normal)
        planes.append(cut_plane)
        
    
    return planes
    
def construct_mesh_edges_abTree(mesh):
    # poly_mesh = Mesh_3()
    # mesh = trimesh.load_mesh(mesh_dir)
    m=Polyhedron_modifier()
    vert_num = mesh.vertices.shape[0]
    face_num = mesh.faces.shape[0]
    for i in range(vert_num):
        cur_p = Point_3(mesh.vertices[i,0], mesh.vertices[i,1], mesh.vertices[i,2])
        m.add_vertex(cur_p)    
    for i in range(face_num):
        m.begin_facet()
        m.add_vertex_to_facet(int(mesh.faces[i,0]))
        m.add_vertex_to_facet(int(mesh.faces[i,1]))
        m.add_vertex_to_facet(int(mesh.faces[i,2]))
        m.end_facet()
    
    get_bbox(mesh.vertices)
    poly_mesh = Polyhedron_3()
    poly_mesh.delegate(m)
    abTree = ABTH(poly_mesh.edges())
    return poly_mesh, abTree

def query_plane_intersection(abtree, plane):
    intersections = []
    abtree.all_intersections(plane, intersections)
    return intersections

def extract_intersect_points_and_normals(intersections, cut_plane):
    pts = []
    ptns =[]
    for intersect in intersections:
        # if not intersect.empty():
        # op = intersect
        # print(intersect.point())
        inter_p = intersect[0]
        edge_half = intersect[1]
        # print(edge_half.vertex().point())
        pa = edge_half.vertex().point()
        pb = edge_half.next().vertex().point()
        pc = edge_half.next().next().vertex().point()
        cur_p = Plane_3(pa, pb, pc)
        pn = cur_p.orthogonal_vector()
        pn = pn / math.sqrt(pn.squared_length())
        if in_plane_normal:
            cpn = cut_plane.orthogonal_vector()
            cpn = cpn / math.sqrt(cpn.squared_length())
            pn =  pn - pn * cpn * cpn
            pn = pn / math.sqrt(pn.squared_length())
            ptns.append(pn)
        else:
            ptns.append(pn)
        inter_p = inter_p.get_Point_3()
        pts.append(inter_p)
            
    return pts, ptns
        # print(inter_p.x(), inter_p.y(), inter_p.z())
        # if inter_p.is_Point_3():
            # print("intersection object is a point", inter_p.x())

def save_ptns(pts, ptns, out_dir):
    p_num = len(pts)
    new_vertices = []
    for i in range(p_num):
        pt = (pts[i].x(),pts[i].y(), pts[i].z(), ptns[i].x(),ptns[i].y(), ptns[i].z())
        new_vertices.append(pt)
    new_vertices = np.array(new_vertices, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),
                                                 ('nx', 'f4'), ('ny', 'f4'),('nz', 'f4')])
    
    pt_data = PlyElement.describe(new_vertices, 'vertex')
    PlyData([pt_data], text=True).write(out_dir)



def sample_points_with_kdtree(pts, ptns, radius):
    p_num = len(pts)
    new_pts = []
    new_ptns = []
    my_kdTree = kdtree.create(dimensions=3)
    for i in range(p_num):
        new_p = (pts[i].x(), pts[i].y(), pts[i].z())
        nearest_pts = my_kdTree.search_nn(new_p)
        if nearest_pts == None :
            my_kdTree.add(new_p)
            new_pts.append(pts[i])
            new_ptns.append(ptns[i])
            continue
        
        if nearest_pts[0].dist(new_p) < radius:  
            continue
        else :
            my_kdTree.add(new_p)
            new_pts.append(pts[i])
            new_ptns.append(ptns[i])
    return new_pts, new_ptns
        
def mesh_slicing():
    mesh_dir = '/home/jjxia/Documents/projects/VIPSS_M/data/torus_knot/torus_knot_thin_mesh.ply'
    # mesh_dir = '/home/jjxia/Documents/projects/VIPSS_M/data/chickHeart/chickHeart_mesh.ply'
    # mesh_dir = '/home/jjxia/Documents/projects/VIPSS_M/data/rect_torus/rect_torus_ori_mesh4.ply'
    ori_mesh = trimesh.load_mesh(mesh_dir)
    
    min_p, max_p = get_bbox(ori_mesh.vertices)
    data_range = max_p - min_p
    bbox_center = data_range / 2.0 + min_p
    print('center', bbox_center)
    ori_mesh.vertices = ori_mesh.vertices - bbox_center
    min_p, max_p = get_bbox(ori_mesh.vertices)
    max_scale = np.max(max_p)
    
    mesh_dir = '/home/jjxia/Documents/projects/VIPSS_M/data/torus_knot/torus_knot_thin_mesh.ply'
    
    trimesh.save(filename=None, ignore_discard=False, ignore_expires=False)
    # if max_scale > 1.0:
    #     ori_mesh.vertices = ori_mesh.vertices/max_scale
    # if max_scale < 1.0:
    ori_mesh.vertices = ori_mesh.vertices/max_scale
    min_p, max_p = get_bbox(ori_mesh.vertices)
    mesh, abTree = construct_mesh_edges_abTree(ori_mesh)
    axis_list = [0]
    s_nums = [10]
    radius = 0.002
    slice_planes = []
    # for axis in axis_list:
    for i in range(len(axis_list)):
        axis = axis_list[i]
        s_num = s_nums[i]
        slice_planes_ax = generate_parallel_slice_planes(min_p, max_p, s_num, axis)
        slice_planes = slice_planes + slice_planes_ax
    pts_all = []
    ptns_all = []
    # for i in range(slice_planes):
    #     q_plane = slice_planes[i]
    for q_plane in slice_planes:
        intersections = query_plane_intersection(abTree, q_plane)
        pts, ptns = extract_intersect_points_and_normals(intersections, q_plane)
        new_pts, new_ptns = sample_points_with_kdtree(pts, ptns, radius)
        pts_all = pts_all + new_pts 
        ptns_all = ptns_all + new_ptns
        # break
    
    # print(pts_all)
    # plane_p = Point_3(0, 0, 0)
    # plane_n = Vector_3(0,0, 1)
    # q_plane = Plane_3(plane_p, plane_n)
    # intersections = query_plane_intersection(abTree, q_plane)
    # pts, ptns = extract_intersect_points_and_normals(intersections, q_plane)
    # out_dir = '../data/cases/twist_star/twist_star_sample_ptns.ply'
    # radius = 0.005
    out_dir = './torus_knot_thin_sample_ptns.ply'
    # new_pts, new_ptns = sample_points_with_kdtree(pts, ptns, radius)
    save_ptns(pts_all, ptns_all, out_dir)

if __name__ == "__main__":
    mesh_slicing()