import numpy as np
import trimesh
from plyfile import PlyData, PlyElement



def save_ptns(pts, ptns, out_dir, file_type='ply'):
    p_num = len(pts)
    if file_type == 'ply':
        new_vertices = []
        for i in range(p_num):
            pt = (pts[i].x(),pts[i].y(), pts[i].z(), ptns[i].x(),ptns[i].y(), ptns[i].z())
            new_vertices.append(pt)
        new_vertices = np.array(new_vertices, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),
                                                    ('nx', 'f4'), ('ny', 'f4'),('nz', 'f4')])
        
        pt_data = PlyElement.describe(new_vertices, 'vertex')
        PlyData([pt_data], text=True).write(out_dir)

    if file_type == 'xyz':
        with open(out_dir, 'w') as f: 
            for i in range(p_num):
                f.write(str(pts[i, 0] ) + " ")
                f.write(str(pts[i, 1] ) + " ")
                f.write(str(pts[i, 2] ) + " ")
                f.write(str(ptns[i, 0] ) + " ")
                f.write(str(ptns[i, 1] ) + " ")
                f.write(str(ptns[i, 2] ) + " ")
                f.write('\n')
    print("Successfully save cv contour 3D points to file: ", out_dir)




def save_pts(pts, out_dir, file_type='ply'):
    p_num = len(pts)
    if file_type == 'ply':
        new_vertices = []
        for i in range(p_num):
            pt = (pts[i, 0], pts[i, 1], pts[i, 2])
            new_vertices.append(pt)
        new_vertices = np.array(new_vertices, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
        pt_data = PlyElement.describe(new_vertices, 'vertex')
        PlyData([pt_data], text=True).write(out_dir)
    
    if file_type == 'xyz':
        with open(out_dir, 'w') as f: 
            for i in range(p_num):
                f.write(str(pts[i, 0] ) + " ")
                f.write(str(pts[i, 1] ) + " ")
                f.write(str(pts[i, 2] ) + " ")
                f.write('\n')
    print("Successfully save cv contour 3D points to file: ", out_dir)

def save_gradient_lines(pts, grads, line_len, out_dir):
    with open(out_dir, 'w') as f: 
        p_num = 0
        i = 0   
        while i < len(pts):
            cur_points = pts[i]
            cur_normals = grads[i]
            for p_id in range(len(cur_points)):
                p = cur_points[p_id]
                pn = cur_normals[p_id]
                p_num += 1
                f.write("v ")
                f.write(str(p[0]) + " ")
                f.write(str(p[1]) + " ")
                f.write(str(p[2]) + " ")
                f.write('\n')
                f.write("v ")
                f.write(str(p[0] + line_len * pn[0]) + " ")
                f.write(str(p[1] + line_len * pn[1]) + " ")
                f.write(str(p[2] + line_len * pn[2]) + " ")
                f.write('\n')
        for id in range(p_num):
            f.write("l ")
            f.write(str(2 *id + 1) + " ")
            f.write(str(2 *id + 2) )
            f.write('\n')