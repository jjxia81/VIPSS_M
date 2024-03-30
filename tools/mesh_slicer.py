import numpy as np
import trimesh
import os
import math
import kdtree
from CGAL.CGAL_Kernel import Point_3, Vector_3, Segment_3, Plane_3, squared_distance
# from CGAL import 
from CGAL.CGAL_AABB_tree import AABB_tree_Polyhedron_3_Halfedge_handle as ABTH
# from CGAL.CGAL_AABB_tree import
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3, Polyhedron_modifier, Polyhedron_3_Halfedge_handle
# import CGAL.CGAL_Polygon_mesh_processing as PMP
# import CGAL.CGAL_Surface_mesher as CGMesh
# from CGAL.CGAL_Surface_mesher import Surface_mesh_default_criteria_3 as Mesh_3
from plyfile import PlyData, PlyElement
# mesh = trimesh.load_mesh(mesh_dir)

in_plane_normal = True

def extract_planes_from_parse_contours(contour_path):
    # contour_path = 'E:\projects\VIPSS_M\data\C_Data\ChickHeart\CTR\chick_sparse.contour'
    file = open(contour_path, "r")
    planes = []
    current_plane = None
    while True:
        content=file.readline()
        if not content:
            break
        content = content.replace(' \n', '')
        content = content.replace('\n', '')
        words = content.split(' ')
        if len(words) == 4:
            cur_plane = []
            for val in words:
                cur_plane.append(float(val))
            current_plane = Plane_3(cur_plane[0], cur_plane[1], cur_plane[2], cur_plane[3])
            # planes.append(new_plane)
        if len(words) == 2:
            planes.append(current_plane)
    file.close()
    # print(planes)
    return planes

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

def cal_edge_normal(edge_half, cut_plane= None):
    pa = edge_half.vertex().point()
    pb = edge_half.next().vertex().point()
    pc = edge_half.next().next().vertex().point()
    cur_p = Plane_3(pa, pb, pc)
    pn = cur_p.orthogonal_vector()
    pn = pn / math.sqrt(pn.squared_length())
    if cut_plane:
        cpn = cut_plane.orthogonal_vector()
        cpn = cpn / math.sqrt(cpn.squared_length())
        pn =  pn - pn * cpn * cpn
        pn = pn / math.sqrt(pn.squared_length())
    return pn



def extract_intersect_points_and_normals2(intersections, cut_plane, radius):
    pts = []
    ptns =[]
    
    inter_dict = {}
    for intersect in intersections:
        inter_dict[intersect[1]] = intersect[0]
    int_edges = set()
    for intersect in intersections:
        inter_p = intersect[0]
        edge_half = intersect[1]
        # print(edge_half.id)
        if edge_half in int_edges or edge_half.opposite() in int_edges:
            continue
        # if edge_half in inter_dict.keys():
        #     print( 'this edge is in inter dict')
        c_pts = []
        c_ptns = []
        int_edges.add(edge_half)
        int_edges.add(edge_half.opposite())
        pn = cal_edge_normal(edge_half, cut_plane)
        inter_p = inter_p.get_Point_3()
        c_pts.append(inter_p)
        c_ptns.append(pn)
        stacks = [ edge_half.next(),  edge_half.prev(), edge_half.opposite().prev(), edge_half.opposite().next()]
        while len(stacks) > 0:
            # print(len(stacks))
            cur_edge = stacks.pop()
            # print('stack pop', stacks)
            int_edges.add(cur_edge)  
            int_edges.add(cur_edge.opposite())          
            cur_p = None
            if cur_edge in inter_dict.keys():
                cur_p = inter_dict[cur_edge]
            if cur_edge.opposite() in inter_dict.keys():
                cur_p = inter_dict[cur_edge.opposite()]
            
            if cur_p and cur_p.is_Point_3():
                # print(dir(cur_p))
                pn = cal_edge_normal(cur_edge, cut_plane)
                cur_p = cur_p.get_Point_3()
                if len(c_pts)  > 0:
                    if squared_distance(c_pts[-1], cur_p) >= radius * radius:
                        c_pts.append(cur_p)
                        c_ptns.append(pn)
                for next_edge in [cur_edge.next(),  cur_edge.prev(), cur_edge.opposite().prev(), cur_edge.opposite().next()]:
                    if next_edge in int_edges or next_edge.opposite() in int_edges:
                        continue
                    if next_edge in inter_dict.keys() or next_edge.opposite() in inter_dict.keys():
                        # print( 'this edge is in inter dict')
                        stacks.append(next_edge)
            
        # print(c_pts)
        # c_pts,c_ptns= sample_points_with_kdtree(c_pts, c_ptns, radius)
        if len(c_pts) > 2 :
            if squared_distance(c_pts[-1], c_pts[0]) < radius * radius:
                c_ptns.pop()
                c_pts.pop()                
        pts = pts + c_pts
        ptns = ptns + c_ptns
        # break    
    return pts, ptns


def extract_intersect_points_and_normals(intersections, cut_plane):
    pts = []
    ptns =[]
    for intersect in intersections:
        # if not intersect.empty():
        # op = intersect
        # print(intersect.point())
        inter_p = intersect[0]
        edge_half = intersect[1]
        # print(dir(edge_half))
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

def write_ply(pts, ptns, out_dir):
    p_num = len(pts)
    new_vertices = []
    for i in range(p_num):
        pt = (pts[i].x(),pts[i].y(), pts[i].z(), ptns[i].x(),ptns[i].y(), ptns[i].z())
        new_vertices.append(pt)
    new_vertices = np.array(new_vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    pt_data = PlyElement.describe(new_vertices, 'vertex')
    PlyData([pt_data], text=True).write(out_dir)





def save_ptns(pts, ptns, out_dir, save_each_slices=False):
    if save_each_slices:
        slice_num = len(pts)
        print("slice num ", slice_num)
        for i in range(slice_num):
            print('out_dir ' , out_dir)
            ply_out_path = os.path.join(out_dir, str(i) + ".ply")
            print("save slice to file : ", ply_out_path)
            write_ply(pts[i], ptns[i], ply_out_path)
    else :
        ply_out_path = os.path.join(out_dir, "sample_contour_pts.ply")
        write_ply(pts, ptns, ply_out_path)


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
            # print(nearest_pts[0].__dir__)
            # print(dir(nearest_pts[0]))
            continue
        else :
            my_kdTree.add(new_p)
            new_pts.append(pts[i])
            new_ptns.append(ptns[i])
    return new_pts, new_ptns

def generate_contours(abTree, slice_planes, radius, save_each_slice = False):
    pts_all = []
    ptns_all = []
    print('cur sample radius', radius)
    for q_plane in slice_planes:
        intersections = query_plane_intersection(abTree, q_plane)
        pts, ptns = extract_intersect_points_and_normals2(intersections, q_plane, radius)
        # pts, ptns = sample_points_with_kdtree(pts, ptns, radius)
        if save_each_slice:
            pts_all.append(pts) 
            ptns_all.append(ptns)
        else :
            pts_all = pts_all + pts 
            ptns_all = ptns_all + ptns
    return pts_all, ptns_all

def generate_slicing_planes(ori_mesh, slice_plane_nums):
    min_p, max_p = get_bbox(ori_mesh.vertices)
    data_range = max_p - min_p
    axis_list = np.argsort(data_range).tolist()
    axis_list.reverse()
    slice_axis_num = len(slice_plane_nums)
    axis_list =  axis_list[:slice_axis_num]
    slice_planes = []
    for i in range(slice_axis_num):
        axis = axis_list[i]
        s_num = slice_plane_nums[i]
        slice_planes_ax = generate_parallel_slice_planes(min_p, max_p, s_num, axis)
        slice_planes = slice_planes + slice_planes_ax
    return slice_planes

def generate_blender_slicing_planes(p_mesh):
    p_num = p_mesh.vertices.shape[0] // 4
    slice_planes = []
    pts = p_mesh.vertices
    for i in range(p_num):
        p0 = Point_3(pts[4 *i, 0], pts[4 *i, 1], pts[4 *i,2])
        p1 = Point_3(pts[4 *i + 1, 0], pts[4 *i + 1, 1], pts[4 *i + 1,2])
        p2 = Point_3(pts[4 *i + 2, 0], pts[4 *i + 2, 1], pts[4 *i + 2,2])
        new_p = Plane_3(p0, p1, p2)
        slice_planes.append(new_p)
    return slice_planes
        
def mesh_slicing():
    # mesh_dir = '../data/torus_knot/torus_knot_thin_mesh.ply'
    # mesh_dir = '../data/torus_knot/Cycle_Result_afterSmooth.obj'
    # mesh_dir = '..\data\C_Data\ChickHeart\chickHeart_mesh.ply'
    # mesh_dir = r'../data/rect_torus/rect_torus_blender_model_scaled.ply'
    mesh_dir = r'../data/chickHeart/chickHeart_mesh.ply'
    # mesh_dir = r'../data/cylinder/cylinder.ply'
    # mesh_dir = r'../data/C_Data/ChickHeart/Result_afterSmooth.ply'
    use_axis_cut_planes = False
    use_contour_planes = True
    contour_dir = r'../data/C_Data/ChickHeart/CTR/chick_dense.contour'
    use_blender_planes = False
    blender_plane_dir = r'../data/C_Data/ChickHeart/add_contour_plane.ply'
    slice_plane_nums = [20]
    scale_mesh = False
    sample_radius = 0.008
    save_each_slice = False
    # if scale_mesh:
    #     sample_radius = 1.0 / (slice_plane_nums[0]-1) 

    dir_strings = mesh_dir.split('/')
    file_name = dir_strings[-1].split('.')
    sample_out_name = file_name[0] + "_sample_pts.ply"
    sample_result_path = os.path.join(*dir_strings[:-1], sample_out_name)
    scale_mesh_dir = os.path.join(*dir_strings[:-1], file_name[0] + "_scaled.ply")
    ori_mesh = trimesh.load_mesh(mesh_dir)

    sample_result_path = r"E:\projects\VIPSS_M\data\rect_torus\slices"
    
    if scale_mesh:
        min_p, max_p = get_bbox(ori_mesh.vertices)
        data_range = max_p - min_p
        bbox_center = data_range / 2.0 + min_p
        print('center', bbox_center)
        ori_mesh.vertices = ori_mesh.vertices - bbox_center
        min_p, max_p = get_bbox(ori_mesh.vertices)
        max_scale = np.max(max_p)
        print("max scale ", max_scale)
        ori_mesh.vertices = ori_mesh.vertices/max_scale
        ori_mesh.export(scale_mesh_dir)
    slice_planes = []
    if use_blender_planes:
        p_mesh = trimesh.load_mesh(blender_plane_dir)
        slice_planes = slice_planes + generate_blender_slicing_planes(p_mesh)
    if use_contour_planes:
        slice_planes = slice_planes + extract_planes_from_parse_contours(contour_dir) 
    if use_axis_cut_planes :
        slice_planes = slice_planes + generate_slicing_planes(ori_mesh, slice_plane_nums)
    
    mesh, abTree = construct_mesh_edges_abTree(ori_mesh)
    
    pts_all, ptns_all = generate_contours(abTree, slice_planes, sample_radius, save_each_slice)
    print('save sample results to ', sample_result_path)
    save_ptns(pts_all, ptns_all, sample_result_path, save_each_slice)

if __name__ == "__main__":
    mesh_slicing()
    # parse_contours(' ')