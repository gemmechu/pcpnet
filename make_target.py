import numpy as np
import trimesh
import scipy.sparse as sparse
from scipy.sparse import coo_matrix
from scipy.spatial import distance
import glob
import os
import robust_laplacian

def knn_cotan(mesh, k):
    def get_weight(neighbour,cotan_weight, i):
        indices = set(cotan_weight.getrow(i).indices)
        values = np.array(cotan_weight.getrow(i).todense())
        result,result_abs = [],[]
        for col in neighbour:
            if col in indices:
                result.append(values[0][col])
                result_abs.append(-1* (values[0][col]))
            else:
                result.append(0.0)
                result_abs.append(0.0)
        return result,result_abs
    vertices = mesh.vertices.view(np.ndarray)
    faces = mesh.faces.view(np.ndarray)
    L, M = robust_laplacian.mesh_laplacian(vertices, faces, mollify_factor=0)
    cotan_weight = L.tocsr()
    D = distance.squareform(distance.pdist(vertices))
    closest = np.argsort(D, axis=1)
    closest = closest[:, 1:k+1]

    data = []
    data_abs = []
    for i, neighbour in enumerate(closest):
        weight, weight_abs = get_weight(neighbour, cotan_weight, i)
        # weight.extend([0.0]*(k-len(weight)))
        data.append(weight)
        data_abs.append(weight_abs)
    
    
    # col = np.concatenate(closest)
    # row = np.concatenate([[i] * len(n)
                        #   for i, n in enumerate(closest)])
    
   
    # # data = np.concatenate([i / np.array(i).sum() if np.array(i).sum()>0 else i for i in data])
    # data = np.concatenate([i for i in data])
    
    
    
    # matrix = coo_matrix((data, (row,col)),
    #                     shape=[len(vertices)] * 2)
    return data,data_abs
def save_xyz(pts, file_name):
    # print(pts)
    s = trimesh.util.array_to_string(pts)
    with open(file_name, 'w') as f:
        f.write("%s\n" % s)

def create_target():
    files = glob.glob('data/new_data/laplacian/*.laplacian')
    for file in files:
        name = file.split('/')[-1].split('.')[0]
        try:
            print(name)
            dest_name = 'data/new_data/cotan_laplacian_mollify_0/' +name +'.laplacian'
            dest_name_abs = 'data/new_data/cotan_laplacian_mollify_0_rev/' +name +'.laplacian'
            noisy_path = 'data/new_data/noisy_mesh/' +name +'.obj'
            danger = set(["gear_n3.obj", "boy01-scanned_n1.obj","gear_n2.obj", "gear_n1.obj", "boy02-scanned_n1.obj","turbine-Lp_n2.obj", "turbine-Lp_n3"])
            if os.path.isfile(dest_name) or name in danger:
                print("skipped: ", dest_name)
                continue
            mesh = trimesh.load_mesh(noisy_path)
            result, result_abs = knn_cotan(mesh, 6)

            save_xyz(result, dest_name)
            save_xyz(result_abs, dest_name_abs)
        except:
            print("ERROR: ", name)
        # break
create_target()