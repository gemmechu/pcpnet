import numpy as np
import trimesh
import scipy
from chamferdist import ChamferDistance
import torch
from scipy.sparse import coo_matrix, eye
from scipy.spatial import distance
from scipy import sparse

def get_chamfer_distance(mesh1, mesh2):
    chamfer_distance = ChamferDistance()
    mesh1_xyz=  torch.tensor(mesh1.vertices).float().unsqueeze(0)
    mesh2_xyz=  torch.tensor(mesh2.vertices).float().unsqueeze(0)
    distance = chamfer_distance(mesh1_xyz,mesh2_xyz)
    return distance
def get_pred_matrix(mesh, k, fname):
    neighbors = mesh.vertex_neighbors
    
    vertices = mesh.vertices.view(np.ndarray)

    ones = np.ones(3)
    norms = [np.sqrt(np.dot((vertices[i] - vertices[n]) ** 2, ones))
                for i, n in enumerate(neighbors)]
    D = distance.squareform(distance.pdist(vertices))
    closest = np.argsort(D, axis=1)
    closest = closest[:, 1:k+1]



    # norms = [i / i.sum() for i in norms]
    data = np.genfromtxt(fname, delimiter=' ')
    
    col = np.concatenate(closest)
    row = np.concatenate([[i] * len(n)
                          for i, n in enumerate(closest)])
    
   

    data = np.concatenate([i / np.array(i).sum() if np.array(i).sum()>0 else i for i in data])
    
    
    matrix = coo_matrix((data, (row, col)),
                        shape=[len(vertices)] * 2)
    return matrix
def make_smooth(mesh, matrix= None):
    if matrix != None:
        matrix = matrix
    else:
        # print("Matrix = None")
        matrix = trimesh.smoothing.laplacian_calculation(mesh, equal_weight= True)
        
    # smoothed_mesh = trimesh.smoothing.filter_taubin(mesh, lamb=0.4, nu=0.5, iterations=10, laplacian_operator=matrix)
    smoothed_mesh = trimesh.smoothing.filter_humphrey(mesh, laplacian_operator=matrix)
                    
    return smoothed_mesh
def smooth_bunch():
    with open('output/denoise_50-300_all/cotan_gt/chamfer_distance.txt', 'a') as w:
        with open('data/new_data/testset_all.txt') as f:
            for line in f:
                line = line.strip('\n') + '.obj'
                noisy_mesh = trimesh.load_mesh('data/new_data/noisy_mesh/'+line)
                smooth_mesh = trimesh.load_mesh('data/new_data/smooth_mesh/'+line)
                fname = 'data/new_data/laplacian/' + line.split('.')[0] +'.laplacian'
                matrix = get_pred_matrix(noisy_mesh, 6, fname)
        
                smoothed =  make_smooth(noisy_mesh,matrix)
                distance = get_chamfer_distance(smoothed, smooth_mesh)
                
                sparse.save_npz("output/denoise_50-300_all/cotan_gt/matrix/"+line.split('.')[0] +'.npz', matrix)
                outpath = 'output/denoise_50-300_all/cotan_gt/smoothed_mesh/'+line
                output = smoothed.export(outpath)
                
                w.write(line+' '+str(float(distance))+'\n')
                print(line, distance)
                

smooth_bunch()