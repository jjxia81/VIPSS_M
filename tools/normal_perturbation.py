import numpy as np
import ply
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R


def load_ply_file(file_path):
    with open(file_path, 'rb') as f:
        plydata = PlyData.read(f)
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        nx = plydata['vertex']['nx']
        ny = plydata['vertex']['ny']
        nz = plydata['vertex']['nz']
        # print(vertices)
        points = np.array([x, y, z]).transpose()
        normals = np.array([nx, ny, nz]).transpose()
        # print(np.sum(normals* normals))
        return points, normals
 
def add_normal_noise(normals):
    n_rows = normals.shape[0]
    noise = (np.random.normal(0, 1, size = (n_rows, 3)) - 0.5) * 0.5
    # lens = np.sqrt(np.sum(noise * noise, axis=1)).reshape((-1, 1))
    # noise = noise / lens * 0.5
    # for i in range(n_rows):
    #     if normals[i,0] > 0.0:
    #         noise[i,:] = noise[i,:] * 0.0
                 
    normals = normals + noise
    lens = np.sqrt(np.sum(normals * normals, axis=1)).reshape((-1, 1))
    normals = -1.0 * normals / lens
    return normals    

def add_normal_noise_with_condition(points, normals):
    n_rows = normals.shape[0]
    # noise = (np.random.normal(0, 1, size = (n_rows, 3)) - 0.5) * 5.5
    radius = 1.0
    # noise = np.random.uniform(-radius, radius, n_rows * 3).reshape((n_rows, 3))
    
    # noise = (np.random.uniform(a, b)(0, 1, size = (n_rows, 3)) - 0.5) * 0.5
    # lens = np.sqrt(np.sum(noise * noise, axis=1)).reshape((-1, 1))
    # noise = noise / lens * 0.5
    # for i in range(n_rows):
    #     if points[i,0] > 0.0:
    #         noise[i,:] = noise[i,:] * 0.5
        # else:
        #     normals[i,:] = normals[i,:] * 0.5
                 
    # normals = -1.0 * normals + noise 
    
        # normals[i,:] = generate_normal_sample(normals[i,:], 30)
    
    normals = -1.0 * normals
    lens = np.sqrt(np.sum(normals * normals, axis=1)).reshape((-1, 1))
    normals = normals / lens
    np.random.seed(10)
    for i in range(n_rows):
        normals[i,:] = rotate_normal(normals[i,:], 30)
    
    gradients = normals.copy()
    for i in range(n_rows):
        if points[i,0] < -0.5:
            gradients[i,:] = gradients[i,:] * 0.001
            continue
        if points[i,0] < 0.0:
            gradients[i,:] = gradients[i,:] * 0.01
            continue
        if points[i,0] < 0.5:
            gradients[i,:] = gradients[i,:] * 0.1
            
    return normals, gradients     
    # print(noise)     
    
    
    
def generate_normal_sample(degree):
    r_z = np.random.uniform(-180, 180)
    r = R.from_euler('xyz', [0, degree, r_z], degrees=True).as_matrix()
    
    # r_axis = R.from_euler('xyz', [0, degree, 0], degrees=True).as_matrix()
    normal = np.array([0,0,1]).reshape((3, 1))
    normal = normal.reshape((3,1))
    # print(r)
    # print(normal)
    normal = r @ normal
    return normal

def test_normal_sample():
    sample_num = 100
    normals = np.ones((sample_num, 3)) 
    normals[:, 1] = normals[:, 1] * -1.0
    normals[:, 0] = normals[:, 0] * -1.0
    normals[:, 2] = normals[:, 2] * -1.0
    # normals = []
    for i in range(sample_num):
        normals[i, :] = rotate_normal(normals[i, :], 30)
        # normals.append((n[0,0], n[0,1], n[0, 2]))
    # pts = np.zeros_like(normals)
    ptns = np.concatenate((normals, normals), axis=1)
    ptns = convert_vertices(ptns)
    out_dir = "../data/sample_nomral.ply"
    gd = PlyElement.describe(ptns, 'vertex')
    PlyData([gd], text=True).write(out_dir)
    
def rotate_normal(normal_vec, degree):
    # nom = np.random.uniform(-2, 2, size = (3, 1))
    # nom = np.array([1, 1, 1]).reshape((3, 1))
    
    normal = normal_vec / np.linalg.norm(normal_vec)
    normal = normal.reshape((3, 1))
    # print("init normal", normal)
    # rotate_y = np.arccos(normal[2])[0]  
    # if normal[0] < 0:  rotate_y = rotate_y * -1   
    s_normal = generate_normal_sample(degree)
    
    # rot = R.align_vectors(np.array([0,0,1]).reshape((1,3)), normal.reshape(1,3))
    # print(rot[0])
    # rot = rot[0].as_matrix() 
    # s_normal = rot @ s_normal.reshape((3,1))
    
    
    
    # s_normal = s_normal.reshape((3,1))
    
    rotate_y = np.arccos(normal[2])[0] 
    # print(normal[2], rotate_y / np.pi * 180)
    # if normal[0] < 0:  rotate_y = rotate_y * -1
    r_y = R.from_euler('y', [rotate_y])
    r_y = r_y.as_matrix()
    s_normal = r_y @ s_normal
    
    # s_normal = s_normal.reshape((3,1))
    rotate_z = np.arccos(normal[0])[0]
    if normal[1] < 0:  rotate_z = rotate_z * -1
    r_z = R.from_euler('z', [rotate_z])
    r_z = r_z.as_matrix()    
    s_normal = r_z @ s_normal
    
    
    
    # rotate_y = np.arccos(s_normal[2])[0] 
    # if s_normal
    
    # trans = np.linalg.inv(r) @ s_normal
    # trans = r @ s_normal
    return s_normal.reshape((1,3))
    # print(trans)
    
def convert_vertices(vertices):
    n_rows = vertices.shape[0]
    new_vertices = []
    for i in range(n_rows):
        new_vertices.append(tuple(vertices[i, :]))
    new_vertices = np.array(new_vertices, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),
                                                 ('nx', 'f4'), ('ny', 'f4'),('nz', 'f4')])
    return new_vertices

def add_noise_to_normal(normals, degree):
    n_rows = normals.shape[0]
    for i in range(n_rows):
        normals[i,:] = rotate_normal(normals[i,:], degree)
    return normals

def sample_normal():
    ply_file = "../data/torus/torus_energy/2000.ply"

    points, normals = load_ply_file(ply_file)
    # new_normals = add_normal_noise(normals)
    # new_normals, gradients = add_normal_noise_with_condition(points, normals)
    new_normals = add_noise_to_normal(normals, 30)
    vertices = np.concatenate((points, new_normals), axis=1)
    new_vertices = convert_vertices(vertices)
    out_dir = "../data/torus/torus_energy/2000_noise_normal.ply"
    el = PlyElement.describe(new_vertices, 'vertex')
    PlyData([el], text=True).write(out_dir)

    # tangents = np.zeros_like(gradients)
    # new_gradients = np.concatenate((gradients, tangents), axis=1)
    # new_gradients = convert_vertices(new_gradients)
    # # print(points)
    # # print(vertices)
    # out_dir = "../data/torus/torus_energy/noise_gradients.ply"
    # gd = PlyElement.describe(new_gradients, 'vertex')
    # PlyData([gd], text=True).write(out_dir)
    
    
# test_normal_sample()
sample_normal()