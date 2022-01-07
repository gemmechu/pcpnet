import glob
import trimesh
import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.spatial import distance
import os
def get_cotangent_weight(i,neighbor,neighbors,norms, KNN):
    def get_k_h(i,j):
        list1, list2 = neighbors[i], neighbors[j]
        #if there is morethan 2 intersection choose the first 2 points
        intersection = list(set(list1).intersection(list2))
        try:
            k,h = intersection
        except:
            if len(intersection)>2:
                k,h = intersection[:2]
            elif len(intersection)== 1:
                print("error here only 1 intersection")
                k,h = intersection[0], intersection[0]
            else:
                print("error here no intersection")
                k,h = 0,0
            # print(intersection)
            # print(i,j)
            # print('k,h', )

        return k,h
    def get_distance(i,j,k,h):
        j_idx = neighbors[i].index(j)
        lij = norms[i][j_idx]

        k_idx = neighbors[j].index(k)
        ljk = norms[j][k_idx]

        i_idx = neighbors[k].index(i)
        lki = norms[k][i_idx]

        h_idx = neighbors[j].index(h)
        ljh = norms[j][h_idx]

        i_idx = neighbors[h].index(i)
        lhi = norms[h][i_idx]

        return lij, ljk, lki, ljh, lhi 

    result = []
    for j in KNN:
        #if j in neighbor find Cotangent weight else assign 0
        wij = 0
        if j in neighbor:
            
            k,h  = get_k_h(i,j)
            lij, ljk, lki, ljh, lhi = get_distance(i,j,k,h)

            s_ijk = (lij + ljk + lki)/2
            A_ijk = 8 *  np.sqrt(s_ijk * ( s_ijk - lij) * ( s_ijk- ljk) * ( s_ijk - lki))

            s_ijh = (lij + ljh + lhi)/2
            A_ijh = 8 *  np.sqrt(s_ijh * ( s_ijh - lij) * ( s_ijh- ljh) * ( s_ijh - lhi))

            
            wij = ((-lij**2 + ljk ** 2 + lki**2)/  A_ijk)  + ((-lij**2 + ljh ** 2 + lhi**2)/  A_ijh)
        
       
        result.append(wij)
        
    return result

def knn_cotangent_laplacian(mesh, k):
    neighbors = mesh.vertex_neighbors
    
    vertices = mesh.vertices.view(np.ndarray)

    ones = np.ones(3)
    norms = [np.sqrt(np.dot((vertices[i] - vertices[n]) ** 2, ones))
                for i, n in enumerate(neighbors)]
    D = distance.squareform(distance.pdist(vertices))
    closest = np.argsort(D, axis=1)
    closest = closest[:, 1:k+1]



    # norms = [i / i.sum() for i in norms]
    data = []

    for i, KNN in enumerate(closest):
        neighbor = neighbors[i]
        weight = get_cotangent_weight(i,neighbor, neighbors,norms, KNN)
        data.append(weight)
        # create the sparse matrix
    


    data = [i / np.array(i).sum() if np.array(i).sum()>0 else i for i in data]
    
    return data

def save_xyz(pts, file_name):
    # print(pts)
    s = trimesh.util.array_to_string(pts)
    with open(file_name, 'w') as f:
        f.write("%s\n" % s)

def create_target():
    files = glob.glob('data/new_data/noisy_mesh/*.obj')
    for file in files:
        dest_name = 'data/new_data/laplacian/' +file.split('/')[-1].split('.')[0] +'.laplacian'
        danger = set(["gear_n3.obj", "boy01-scanned_n1.obj","gear_n2.obj", "gear_n1.obj", "boy02-scanned_n1.obj"])
        if os.path.isfile(dest_name) or file.split("/")[-1] in danger:
            print("skipped: ", file)
            continue
        
        print(file)
        mesh = trimesh.load_mesh(file)
        result = knn_cotangent_laplacian(mesh, 6)
    
        save_xyz(result, dest_name)

create_target()