import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import shutil
import pymeshlab as ml
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
import multiprocessing
from functools import partial

def writeVTKPOINTDATA(f, val, resultName, nComp):
    f.write("{} {} {} double \n".format(resultName, nComp, len(val)))
    i = 0
    while i < len(val):
        string = ""
        for j in range(9//nComp):
            if i < len(val):
                for k in range(nComp):
                    string += "{} ".format(val[i][k].item())
            i+=1
        f.write(string+"\n")

def write2D(filename, var):
    f = open("C:\\Users\\trong\\Downloads\\{}.vtk".format(filename), "w")
    f.write("# vtk DataFile Version 4.2\n")
    f.write("vtk output\n")
    f.write("ASCII\n")
    f.write("DATASET UNSTRUCTURED_GRID\n")
    f.write("POINTS {} float\n".format(len(var['coord'])))
    i = 0
    while i < len(var['coord']):
        string = ""
        for j in range(3):
            if i < len(var['coord']):
                for k in range(var['coord'].shape[1]):
                    string += "{} ".format(var['coord'][i][k].item())
            i+=1
        f.write(string+"\n")

    # f.write("CELLS {} {}\n".format(len(var['elem']), len(var['elem'])*(var['elem'].shape[1] + 1)))
    # for e in var['elem']:
    #     string = "{} ".format(var['elem'].shape[1])
    #     for j in range(var['elem'].shape[1]):
    #         string += "{} ".format(e[j])
    #     f.write(string+"\n")
    
    # f.write("CELL_TYPES {} \n".format(len(var['elem'])))
    # for e in var['elem']:
    #     f.write("5\n")

    f.write("POINT_DATA {} \n".format(len(var['coord'])))
    f.write("FIELD FieldData 1\n")
    
    writeVTKPOINTDATA(f, var["stress_mises_top"][:, None], "Stress_exact%20mises%20top", 1)
    # writeVTKPOINTDATA(f, var["stress_exact_mises_bot"][:, None], "Stress_exact%20mises%20bot", 1)

    f.close()

def exportOBJMesh(path, nodes):
    f = open(path, "w")
    for id, node in enumerate(nodes):
        f.write("v {} {} {}\n".format(str(node[0]), str(node[1]), str(node[2])))

    f.close()

def worker(path, file, max_nodes, recon_nodes_num):
    if os.path.exists(path + "/.cached/" + "{}.pth".format(file)):
        return torch.load(path + "/.cached/" + "{}.pth".format(file))
    
    try:
        a = file.split("_")
        shape = int(a[-1][-1])
        param = [float(i) for i in a[1].split("x")]
        
        result = pd.read_csv(path + "/" + file + '/circ.csv')
        data_stat = np.load(path + "/" + file +"/data.npz", encoding='latin1', allow_pickle=True)
        true_disp = result[["displacement:0", "displacement:1", "displacement:2"]].to_numpy()
        disp = np.linalg.norm(true_disp, axis=1)
        max_disp = disp.max()
        # true_stress_middle = result[["shell_stress_middle:0", "shell_stress_middle:1", "shell_stress_middle:2", "shell_stress_middle:3", "shell_stress_middle:4", "shell_stress_middle:5"]].to_numpy()
        # true_stress_top = result[["shell_stress_upper:0", "shell_stress_upper:1", "shell_stress_upper:2", "shell_stress_upper:3", "shell_stress_upper:4", "shell_stress_upper:5"]].to_numpy()
        # true_stress_bottom = result[["shell_stress_lower:0", "shell_stress_lower:1", "shell_stress_lower:2", "shell_stress_lower:3", "shell_stress_lower:4", "shell_stress_lower:5"]].to_numpy()
        
        # true_mises_middle = (0.5*((true_stress_middle[:,0] - true_stress_middle[:,1])**2 + (true_stress_middle[:,1] - true_stress_middle[:,2])**2 + (true_stress_middle[:,0] - true_stress_middle[:,2])**2) + 3*(true_stress_middle[:,3]**2 + true_stress_middle[:,4]**2 + true_stress_middle[:,5]**2))**0.5
        # true_mises_top = (0.5*((true_stress_top[:,0] - true_stress_top[:,1])**2 + (true_stress_top[:,1] - true_stress_top[:,2])**2 + (true_stress_top[:,0] - true_stress_top[:,2])**2) + 3*(true_stress_top[:,3]**2 + true_stress_top[:,4]**2 + true_stress_top[:,5]**2))**0.5
        # true_mises_bottom = (0.5*((true_stress_bottom[:,0] - true_stress_bottom[:,1])**2 + (true_stress_bottom[:,1] - true_stress_bottom[:,2])**2 + (true_stress_bottom[:,0] - true_stress_bottom[:,2])**2) + 3*(true_stress_bottom[:,3]**2 + true_stress_bottom[:,4]**2 + true_stress_bottom[:,5]**2))**0.5
        
        center = np.mean(np.mean(data_stat["node"][data_stat["elem"]], axis=1), axis=0)
        pos = data_stat["node"] - center
        ms = ml.MeshSet()
        # Load the mesh
        m = ml.Mesh(pos, data_stat["elem"])
        ms.add_mesh(m, "mesh_1")
        increase = 100
        while True:
            ms.generate_sampling_poisson_disk(samplenum = (recon_nodes_num+increase),
                                        refineflag=True,
                                        refinemesh=10,
                                        exactnumflag = True)
            vertices = ms.current_mesh().vertex_matrix()
            if len(vertices) >= recon_nodes_num:
                recon_nodes = torch.tensor(vertices[:recon_nodes_num], dtype=torch.float32)
                break
            else:
                increase+=100
        exportOBJMesh("./test.obj", recon_nodes.tolist())
        b = disp.max() - disp.min()
        interpolator = RBFInterpolator(pos, (disp - disp.min())/b)
        recon_disp = interpolator(recon_nodes)
        # write2D("loss", {
        #         "coord": recon_nodes,
        #         "disp": recon_disp
        #         })
        nodes = torch.tensor(pos, dtype=torch.float32)
        elems = data_stat["elem"]
        file = file
        num_nodes = nodes.shape[0]
        if num_nodes > max_nodes:
            print("max number of nodes incorrect! Current file {} has {} nodes".format(file, num_nodes))
        # Pad nodes and stress to max_nodes
        # mask = torch.zeros(max_nodes, 1, dtype=torch.float32)
        # mask[:num_nodes] = 1.0
        # nodes_padded = torch.zeros(max_nodes, 3, dtype=torch.float32)
        # nodes_padded[:num_nodes] = nodes
        # b = true_mises_top.max() - true_mises_top.min()
        # stress_padded = torch.zeros(max_nodes, dtype=torch.float32)
        # stress_padded[:num_nodes] = torch.from_numpy((true_mises_top - true_mises_top.min()) / b)
    except Exception as e:
        print(file, e)
        shutil.move(path + "/{}".format(file), "./tmp1/{}".format(file))
        return
    
    data = {
        "nodes": nodes,
        #"padded_nodes": nodes_padded,
        "recon_nodes": recon_nodes,
        "recon_disp": recon_disp,
        #"stress": stress_padded,
        "disp_max": max_disp,
        #"mask": mask,
        "elems": elems,
        "num_nodes": num_nodes,
        "shape": shape,
        "param": torch.tensor(param, dtype=torch.float32)
    }
    torch.save(data, path + "/.cached/" + '{}.pth'.format(file))
        # dataset.append(data)
    return data

def load_mesh_dataset(path, max_nodes=1000, recon_nodes_num=1000):
    # arr = [[path, i, max_nodes, recon_nodes_num] for i in os.listdir(path) if i.find(".cached") == -1]
    # # for ar in arr:
    # #     worker(ar[0], ar[1], ar[2], ar[3])
    # with multiprocessing.Pool() as pool:
    #     # Map only the numbers, as the factor is already set
    #     results = pool.starmap(worker, arr)

    # max_num = 0
    # for file in os.listdir(path):
    #     if file.find(".cached") != -1:
    #         continue
        
    #     try:
    #         data_stat = np.load(path + "/" + file +"/data.npz", encoding='latin1', allow_pickle=True)
    #         if max_num < len(data_stat["node"]):
    #             max_num = len(data_stat["node"])
    #             print(max_num)
    #     except Exception:
    #         shutil.move(path + "/{}".format(file), "./tmp1/{}".format(file))
    #         continue

    dataset = []
    param_list = []
    for file in os.listdir(path):
        if file.find(".cached") != -1:
            continue
        
        if os.path.exists(path + "/.cached/" + "{}.pth".format(file)):
            a = torch.load(path + "/.cached/" + "{}.pth".format(file))
            # c = a["recon_nodes"]
            # b = c.max(0).values - c.min(0).values
            # a["recon_nodes"] = (c - c.min(0).values) / b
            dataset.append(a)
            tmp = a["param"].tolist()
            tmp.append(a["disp_max"])
            param_list.append(tmp)
            continue
        continue
        a = file.split("_")
        shape = int(a[-1][-1])
        param = [float(i) for i in a[1].split("x")]
        try:
            result = pd.read_csv(path + "/" + file + '/circ.csv')
            data_stat = np.load(path + "/" + file +"/data.npz", encoding='latin1', allow_pickle=True)
            true_disp = result[["displacement:0", "displacement:1", "displacement:2"]].to_numpy()
            disp = np.linalg.norm(true_disp, axis=1)
            max_disp = disp.max()
            # true_stress_middle = result[["shell_stress_middle:0", "shell_stress_middle:1", "shell_stress_middle:2", "shell_stress_middle:3", "shell_stress_middle:4", "shell_stress_middle:5"]].to_numpy()
            # true_stress_top = result[["shell_stress_upper:0", "shell_stress_upper:1", "shell_stress_upper:2", "shell_stress_upper:3", "shell_stress_upper:4", "shell_stress_upper:5"]].to_numpy()
            # true_stress_bottom = result[["shell_stress_lower:0", "shell_stress_lower:1", "shell_stress_lower:2", "shell_stress_lower:3", "shell_stress_lower:4", "shell_stress_lower:5"]].to_numpy()
            
            # true_mises_middle = (0.5*((true_stress_middle[:,0] - true_stress_middle[:,1])**2 + (true_stress_middle[:,1] - true_stress_middle[:,2])**2 + (true_stress_middle[:,0] - true_stress_middle[:,2])**2) + 3*(true_stress_middle[:,3]**2 + true_stress_middle[:,4]**2 + true_stress_middle[:,5]**2))**0.5
            # true_mises_top = (0.5*((true_stress_top[:,0] - true_stress_top[:,1])**2 + (true_stress_top[:,1] - true_stress_top[:,2])**2 + (true_stress_top[:,0] - true_stress_top[:,2])**2) + 3*(true_stress_top[:,3]**2 + true_stress_top[:,4]**2 + true_stress_top[:,5]**2))**0.5
            # true_mises_bottom = (0.5*((true_stress_bottom[:,0] - true_stress_bottom[:,1])**2 + (true_stress_bottom[:,1] - true_stress_bottom[:,2])**2 + (true_stress_bottom[:,0] - true_stress_bottom[:,2])**2) + 3*(true_stress_bottom[:,3]**2 + true_stress_bottom[:,4]**2 + true_stress_bottom[:,5]**2))**0.5
            
            center = np.mean(np.mean(data_stat["node"][data_stat["elem"]], axis=1), axis=0)
            pos = data_stat["node"] - center
            ms = ml.MeshSet()
            # Load the mesh
            m = ml.Mesh(pos, data_stat["elem"])
            ms.add_mesh(m, "mesh_1")
            increase = 100
            while True:
                ms.generate_sampling_poisson_disk(samplenum = (recon_nodes_num+increase),
                                            refineflag=True,
                                            refinemesh=10,
                                            exactnumflag = True)
                vertices = ms.current_mesh().vertex_matrix()
                if len(vertices) >= recon_nodes_num:
                    recon_nodes = torch.tensor(vertices[:recon_nodes_num], dtype=torch.float32)
                    break
                else:
                    increase+=100
            exportOBJMesh("./test.obj", recon_nodes.tolist())

            b = disp.max() - disp.min()
            interpolator = RBFInterpolator(pos, (disp - disp.min())/b)
            recon_disp = interpolator(recon_nodes)
            # write2D("loss", {
            #         "coord": recon_nodes,
            #         "disp": recon_disp
            #         })
            nodes = torch.tensor(pos, dtype=torch.float32)
            elems = data_stat["elem"]
            file = file
            num_nodes = nodes.shape[0]
            if num_nodes > max_nodes:
                print("max number of nodes incorrect! Current file {} has {} nodes".format(file, num_nodes))
            # Pad nodes and stress to max_nodes
            # mask = torch.zeros(max_nodes, 1, dtype=torch.float32)
            # mask[:num_nodes] = 1.0
            # nodes_padded = torch.zeros(max_nodes, 3, dtype=torch.float32)
            # nodes_padded[:num_nodes] = nodes
            # b = true_mises_top.max() - true_mises_top.min()
            # stress_padded = torch.zeros(max_nodes, dtype=torch.float32)
            # stress_padded[:num_nodes] = torch.from_numpy((true_mises_top - true_mises_top.min()) / b)
        except Exception:
            shutil.move(path + "/{}".format(file), "./tmp1/{}".format(file))
            continue

        
        data = {
            "nodes": nodes,
            #"padded_nodes": nodes_padded,
            "recon_nodes": recon_nodes,
            "recon_disp": recon_disp,
            #"stress": stress_padded,
            "disp_max": max_disp,
            #"mask": mask,
            "elems": elems,
            "num_nodes": num_nodes,
            "shape": shape,
            "param": torch.tensor(param, dtype=torch.float32)
        }
        torch.save(data, path + "/.cached/" + '{}.pth'.format(file))
        dataset.append(data)

    # param_list = np.array(param_list)
    # df = pd.DataFrame(param_list)
    # df.hist(bins=30, figsize=(15, 10))
    # plt.tight_layout()
    # plt.savefig("./describe.png")

    if os.path.exists(path + "/.cached/" + 'data_decribe.npz'):
        data_stat = np.load(path + "/.cached/" + 'data_decribe.npz', encoding='latin1', allow_pickle=True)
        a = data_stat["std"]
        b = data_stat["min"]
    else:
        param_list = np.array(param_list)
        a = param_list.max(0) - param_list.min(0)
        b = param_list.min(0)
        np.savez_compressed(path + "/.cached/" + 'data_decribe.npz', std=a, min=b)
        param_list = (param_list - b) / a
    return dataset, a, b

class MeshDataset(Dataset):
    def __init__(self, path, max_nodes=1000, recon_nodes_num=1000):
        self.data, self.std, self.min = load_mesh_dataset(path, max_nodes, recon_nodes_num)
    
    def __len__(self):
        return len(self.data)
    
    def ExtractFeature(self, path, recon_nodes_num):
        a = file.split("_")
        shape = int(a[-1][-1])
        param = [float(i) for i in a[1].split("x")]
        
        result = pd.read_csv(path + "/" + file + '/circ.csv')
        data_stat = np.load(path + "/" + file +"/data.npz", encoding='latin1', allow_pickle=True)
        true_disp = result[["displacement:0", "displacement:1", "displacement:2"]].to_numpy()
        disp = np.linalg.norm(true_disp, axis=1)
        max_disp = disp.max()

        center = np.mean(np.mean(data_stat["node"][data_stat["elem"]], axis=1), axis=0)
        pos = data_stat["node"] - center
        ms = ml.MeshSet()
        # Load the mesh
        m = ml.Mesh(pos, data_stat["elem"])
        ms.add_mesh(m, "mesh_1")
        increase = 100
        while True:
            ms.generate_sampling_poisson_disk(samplenum = (recon_nodes_num+increase),
                                        refineflag=True,
                                        refinemesh=10,
                                        exactnumflag = True)
            vertices = ms.current_mesh().vertex_matrix()
            if len(vertices) >= recon_nodes_num:
                recon_nodes = torch.tensor(vertices[:recon_nodes_num], dtype=torch.float32)
                break
            else:
                increase+=100
        exportOBJMesh("./test.obj", recon_nodes.tolist())
        b = disp.max() - disp.min()
        interpolator = RBFInterpolator(pos, (disp - disp.min())/b)
        recon_disp = interpolator(recon_nodes)
        # write2D("loss", {
        #         "coord": recon_nodes,
        #         "disp": recon_disp
        #         })
        nodes = torch.tensor(pos, dtype=torch.float32)
        elems = data_stat["elem"]
        file = file
        num_nodes = nodes.shape[0]
        
        return {
            #"nodes": self.data[idx]["nodes"],
            #"padded_nodes": self.data[idx]["padded_nodes"],
            "recon_nodes": recon_nodes,
            "recon_disp": recon_disp,
            #"stress": self.data[idx]["stress"].view(-1,1),
            "disp_max":(torch.tensor(max_disp, dtype=torch.float32) - self.min[-1]) / self.std[-1],
            #"mask": self.data[idx]["mask"],
            "shape": torch.tensor(shape, dtype=torch.float32),
            "param": (torch.tensor(param, dtype=torch.float32) - self.min[:-1]) / self.std[:-1],
        }

    def __getitem__(self, idx):
        return {
            #"nodes": self.data[idx]["nodes"],
            #"padded_nodes": self.data[idx]["padded_nodes"],
            "recon_nodes": self.data[idx]["recon_nodes"],
            "recon_disp": self.data[idx]["recon_disp"],
            #"stress": self.data[idx]["stress"].view(-1,1),
            "disp_max":(torch.tensor(self.data[idx]["disp_max"], dtype=torch.float32) - self.min[-1]) / self.std[-1],
            #"mask": self.data[idx]["mask"],
            "num_nodes": self.data[idx]["num_nodes"],
            "shape": torch.tensor(self.data[idx]["shape"], dtype=torch.float32),
            "param": (self.data[idx]["param"] - self.min[:-1]) / self.std[:-1],
        }
    
if __name__ == "__main__":
    dataset = MeshDataset("./dataset2/train", max_nodes=5000, recon_nodes_num=4096)
