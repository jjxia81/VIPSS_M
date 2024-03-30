import numpy as np
import trimesh
import os
import math
import kdtree


from plyfile import PlyData, PlyElement
# mesh = trimesh.load_mesh(mesh_dir)

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

def write_ply_color(pts, ptns, colors, out_dir):
    p_num = len(pts)
    new_vertices = []
    for i in range(p_num):
        pt = (pts[i].x(),pts[i].y(), pts[i].z(), ptns[i].x(),ptns[i].y(), ptns[i].z(), colors[i].x(), colors[i].y(), colors[i].z())
        new_vertices.append(pt)
    new_vertices = np.array(new_vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), 
                                                 ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    
    pt_data = PlyElement.describe(new_vertices, 'vertex')
    PlyData([pt_data], text=True).write(out_dir)

def write_ply_color( ptns, colors, out_dir):
    p_num = ptns.shape[0]
    new_vertices = []
    for i in range(p_num):
        ptn = ptns[i, :]
        
        pt = (ptn[0], ptn[1], ptn[2], ptn[3], ptn[4], ptn[5], colors[i], 0, 0)
        new_vertices.append(pt)
    new_vertices = np.array(new_vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), 
                                                 ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    
    pt_data = PlyElement.describe(new_vertices, 'vertex')
    PlyData([pt_data], text=True).write(out_dir)

def read_ptns(ply_path):
    plydata = PlyData.read(ply_path)
    pt_num = plydata.elements[0].count
    pt_data = plydata.elements[0]
    ptns = []
    for i in range(pt_num):
        cur_pt = list(pt_data[i])
        ptns.append(cur_pt)

    ptns = np.array(ptns)
    
    # print("ptns size ", ptns.shape)
    # print(ptns)
    return ptns

        
def mesh_slicing():
    # mesh_dir = '../data/torus_knot/torus_knot_thin_mesh.ply'
    # mesh_dir = '../data/torus_knot/Cycle_Result_afterSmooth.obj'
    # mesh_dir = '..\data\C_Data\ChickHeart\chickHeart_mesh.ply'
    # mesh_dir = r'../data/rect_torus/rect_torus_blender_model_scaled.ply'
    #pt_dir1 = r'E:\projects\VIPSS_M\data\cases\vipss_output\doghead\doghead_opt_input_normal.ply'
    #pt_dir2 = r'E:\projects\VIPSS_M\data\cases\vipss_output\doghead\doghead_out_ptn_normalized.ply'

    pt_dir2 = r'E:\projects\All_Elbows\output\out_mesh\6007\6007_in_ptn.ply'
    pt_dir1 = r'E:\projects\All_Elbows\output\out_mesh\6007\6007_iter_normal_0.ply'
    

    # pt_dir1 = r'E:\projects\VIPSS_M\data\cases\vipss_output\bathtub\bathtub_opt_input_normal.ply'
    # pt_dir2 = r'E:\projects\VIPSS_M\data\cases\vipss_output\bathtub\bathtub_out_ptn_normalized.ply'

    ptns1 = read_ptns(pt_dir1)
    ptns2 = read_ptns(pt_dir2)

    norm1 = ptns1[:,3:] 
    norm2 = ptns2[:,3:]

    delt_n = norm1 - norm2
    

    sum = np.sum(np.square(delt_n), axis=1)
    # sum = np.where(sum > 3.0, 3.0, sum)
    max_val = np.max(sum)

    color_r = sum / max_val * 255
    color_r = color_r.astype(np.uint8)
    print('sum', sum)
    print('color_r', color_r)

    out_dir = "walrus_color.ply"
    write_ply_color(ptns1,color_r, out_dir)
    

if __name__ == "__main__":
    mesh_slicing()
    # parse_contours(' ')