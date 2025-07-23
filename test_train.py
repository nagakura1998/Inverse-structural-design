import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataset import MeshDataset
from torch.utils.tensorboard import SummaryWriter
from PointNetTransfrom_Utils import *
import os
import random
import tqdm
import pymeshlab as ml
from torch_geometric.transforms import BaseTransform

class Transform_center_and_scale(BaseTransform):
    def __init__(self, valueDevider = 10):
        """
        Parameters:
        - valueDevider (int): the value to devider the bounding box
        """
        self.devider = valueDevider
        
    def __call__(self, data):
        """
        Normalize the data by:
        1. Centering the data at (0,0,0)
        2. Scaling the data so that the longest side of the bounding box is equal to self.devider
        3. Shifting the data so that the minimum value of each axis is 0.01
        
        Parameters:
        - data (np.ndarray): The data to be normalized.
        
        Returns:
        - normalized_positive (np.ndarray): The normalized data.
        """
        data = data[:8000, :]
        boundingBox = data.max(axis=0)[0] - data.min(axis=0)[0]
        centerMid = (data.max(axis=0)[0] + data.min(axis=0)[0])/2
        
        scale = self.devider / boundingBox.max()
        
        normalized_zero = (data - centerMid) * scale
        # normalized_positive = normalized_zero - normalized_zero.min(axis=0)[0] + 0.01 # add 0.01 to make the mininum is 0.01
                
        return normalized_zero

class ReaderDatasetPointV2(Dataset):
    def __init__(self, root, test=False, transform=None):
        """
        Custom Dataset for loading and processing geometry data.

        Parameters:
        - root (str): Root directory containing the data.
        - transform (callable, optional): A function/transform to apply to the data.
        """
        self.root = root
        self.test = test
        self.transform = transform
        self.dataset_path = os.path.join(self.root, "geometries_process_top_face_8000")

        # Build a list of all data files
        self.data_list = self.compile_dataset()

        # Create a mapping from skin types to labels
        # self.skin_type_to_label = self.create_skin_type_to_label()
        
        self.total_data = []
        self.excludeForTest()
        self.load_all_data()


    def compile_dataset(self):
        """
        Compiles a list of all dataset files along with their skin types.

        Returns:
        - data_list (list): A list of dictionaries containing data file information.
        """
        data_list = {}
        # List all skin types
        list_of_skins = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        for skin_type in list_of_skins:
            skin_dir = os.path.join(self.dataset_path, skin_type)
            # List all datasets in the skin type directory
            datasets = [f for f in os.listdir(skin_dir) if f.lower().endswith('.ply')]
            
            data_list_one_skin = []
            for dataset_name in datasets:
                dataset_path = os.path.join(skin_dir, dataset_name)

                data_list_one_skin.append({
                    'dataset_name': dataset_name,
                    'dataset_path': dataset_path
                })
            data_list[skin_type] = data_list_one_skin
            
        return data_list
    
    def excludeForTest(self):
        skinTotal = list(self.data_list.keys())
        random.seed(20)
        self.skinSample = random.sample(skinTotal, 5)
            
    def load_all_data(self):
        """
        Loads all data into memory and stores it in self.total_data.
        NOTE : there is also option to change the csv format to .pt 
        """
        preprocessed_dir = os.path.join(os.getcwd(), "preprocessed_point_transformer_8000")
        os.makedirs(preprocessed_dir, exist_ok=True)
        
        for skinType, data_info in self.data_list.items():
            
            # put filter in the dataset to see if the dataset is small number then how it result
            if not (skinType in ["skin_1", "skin_2", "skin_3", "skin_4", "skin_5"]): #
                continue
            
            preprocessed_file = os.path.join(preprocessed_dir, f"{skinType}.pt")
            
            if os.path.exists(preprocessed_file):
                # Load preprocessed data
                dataTensor = torch.load(preprocessed_file)
            
            else:
                dataTensor = []
                for data in data_info:
                    
                    dataset_path = data['dataset_path']
                    dataset_name = data['dataset_name']

                    # # Load data from CSV
                    # data = pd.read_csv(dataset_path, header=None).values  # Reads the CSV and converts to NumPy array
                    
                    # Load data from .ply
                    mesh = ml.MeshSet()
                    mesh.load_new_mesh(dataset_path)
                    m = mesh.current_mesh()
                    vertex_matrix = m.vertex_matrix()
                    
                    # Convert data to torch.Tensor
                    data_tensor = torch.from_numpy(vertex_matrix).float()

                    dataTensor.append({
                        'skinType' : skinType,
                        'name': dataset_name,
                        'x': data_tensor
                    })
                    mesh.clear()
                    del mesh
                    
                torch.save(dataTensor, preprocessed_file)
            
            print(f"skin type : {skinType} - number of data : {len(dataTensor)}")
            
            # train version
            if (not self.test) and (not skinType in self.skinSample):
                self.total_data.extend(dataTensor)
            
            # test version
            if self.test and (skinType in self.skinSample):
                self.total_data.extend(dataTensor)

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.total_data)
        
    def __getitem__(self, idx):
        """
        Retrieves the data item at the given index.

        Parameters:
        - idx (int): Index of the data item.

        Returns:
        - data_tensor (torch.Tensor): The data tensor.
        - label (int): The numeric label corresponding to the skin type.
        """
        data_pickup = self.total_data[idx]
        data_tensor = data_pickup["x"]
        skin_type = data_pickup["skinType"]
        datasetName = data_pickup["name"]

        # Apply transform if provided
        if self.transform:
            data_tensor = self.transform(data_tensor)
            
        return skin_type, datasetName, data_tensor

class EncoderPoineTransformDecoderLinear_Gen_4(nn.Module):
    def __init__(self, point_cloud_node = 10000, point_cloud_output = 10000, features= 3, bottleneck = 256, num_class = 16, num_layer = 4) -> None:
        super(EncoderPoineTransformDecoderLinear_Gen_4, self).__init__()
        self.features = features
        self.latent_space = bottleneck
        # npoint, input_dim, num_class, bottleneck
        self.encoder = PointTransformerCls(npoint = point_cloud_node, 
                                           input_dim = self.features,
                                           num_class = num_class, 
                                           bottleneck = bottleneck)
        self.decoder = DecoderWithSaveFeatures_Gen_4(inputs=bottleneck, 
                                                     point_cloud=point_cloud_output,
                                                     num_layer=num_layer)
        
        self.map_mu = nn.Linear(bottleneck, bottleneck)
        self.map_logvar = nn.Linear(bottleneck, bottleneck)
        
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def forward(self, features):
        _, latent_f = self.encoder(features)
        
        mu = self.map_mu(latent_f)
        logvar = self.map_logvar(latent_f)
        
        latent = self.reparameterize(mu, logvar)
        pred_pts = self.decoder(latent)  # B x n_prims x 3 x n_verts
        
        return pred_pts, mu, logvar
    
    def get_latent_space(self, x):
        # x = x.view(int(x.shape[0]/256), 256)
        # Encoding
        _, latent = self.encoder(x)
        # mu = self.map_mu(latent)
        # log_var = self.map_logvar(latent)
        # # Reparameterization
        # z = self.reparameterize(mu, log_var)
        return latent
    
    def get_point_cloud(self, latent):
        # this function is to get point cloud
        mu = self.map_mu(latent)
        logvar = self.map_logvar(latent)
        
        latent = self.reparameterize(mu, logvar)
        pointcloud = self.decoder(latent)
        
        return pointcloud
    

class DecoderWithSaveFeatures_Gen_4(nn.Module):
    def __init__(self, inputs=256, point_cloud=2048, num_layer=4) -> None:
        super(DecoderWithSaveFeatures_Gen_4, self).__init__()
        self.input = inputs
        self.point_cloud = point_cloud
        self.num_layer = num_layer
        
        # Linear layers
        self.lin1 = nn.Linear(self.input, 512)
        self.lin2 = nn.Linear(512, 1024)
        
        # ConvTranspose layers
        self.conv_trans_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        
        for i in range(num_layer):
            self.conv_trans_layers.append(
                nn.ConvTranspose1d(1024, int(self.point_cloud / num_layer), kernel_size=3, stride=1)
            )
            self.linear_layers.append(
                nn.Linear(3, 3)
            )
        # improvement :
        # 1. add Dropout (22 May) -> deactiveted in (23 may) -> its more for overfitting problem
        # 2. change ReLu to LeakyReLu or PReLU (22 May)
        # 3. add batchNorm layer / LayerNorm / Instance Norm after each ConvTranspose model
        # 4. make residual model to modfiy the ConvTranspose
        
        # Other layers
        self.relu = nn.LeakyReLU()
        self.batchnorm = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        
        # Final linear layer for reconstruction
        self.lin3 = nn.Linear(int(point_cloud * 3), int(point_cloud * 3))
        
        # Dropout for regularization, default : not used dropout
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.relu(self.batchnorm(self.lin1(x)))
        x = self.relu(self.batchnorm2(self.lin2(x)))
        x = self.dropout(x) # default not use dropout
        x = x.unsqueeze(2)
        
        outputs = []
        for conv_trans, linear in zip(self.conv_trans_layers, self.linear_layers):
            y = conv_trans(x)
            y = linear(y)
            y = self.relu(y)
            outputs.append(y) 
        
        # Concatenate outputs along dimension 1
        x = torch.cat(outputs, dim=1)
        x = torch.flatten(x, start_dim=1)
        
        # Final linear layer
        x = self.lin3(x)
        x = x.view(x.shape[0], int(x.shape[1] / 3), 3)
        
        return x


class PointTransformerCls(nn.Module):
    def __init__(self, npoint, input_dim, num_class, bottleneck, just_latent=True):
        super().__init__()
        output_channels = num_class
        self.d_points = input_dim
        
        self.npoint = npoint
        self.just_latent = just_latent
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = StackedAttention()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, bottleneck)
        self.bn7 = nn.BatchNorm1d(bottleneck)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(bottleneck, output_channels)

    def exportOBJMesh(self, path, nodes):
        f = open(path, "w")
        for id, node in enumerate(nodes):
            f.write("v {} {} {}\n".format(str(node[0]), str(node[1]), str(node[2])))

        f.close()

    def forward(self, x):
        # change it to each batch
        # x = x.view(int(x.shape[0]/self.npoint), self.npoint, self.d_points)
        
        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = x.permute(0, 2, 1)
        
        ### here i change the sample_and_group npoint x2, and nsample x 1/2, to make it more simple
        # 14 January : 512 = 256 || 256 = 128
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)
        
        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x_features = self.bn7(self.linear2(x))

        x = self.relu(x_features)
        x = self.dp2(x)
        x = self.linear3(x)

        return x, x_features

class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x
    
def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 
    
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]

    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)

    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x) # b, c, n        
        x_v = self.v_conv(x) 
        # attention mechanisms to represent the raw scores 
        # that are later normalizaes into probalities ( often using softmax)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class EnhancedDecoder(nn.Module):
    def __init__(self, inputs=256, point_cloud=2048, num_layer=4):
        super(EnhancedDecoder, self).__init__()
        self.latent_dim = inputs
        self.point_cloud_size = point_cloud
        self.num_layers = num_layer
        
        # Initial linear layers to expand latent vector
        self.lin1 = nn.Linear(inputs, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.lin2 = nn.Linear(1024, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        
        # Graph or PointNet-based layers
        self.pointnet_layers = nn.ModuleList()
        
        self.pointnet_layers.append(nn.Sequential(
            nn.Conv1d(2048, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU()
        ))
        for _ in range(num_layer-1):
            self.pointnet_layers.append(nn.Sequential(
                nn.Conv1d(1024, 1024, kernel_size=1),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU()
            ))
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose1d(1024, int(self.point_cloud_size / num_layer), kernel_size=3, stride=1) 
            for _ in range(num_layer)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(p=0.2)
        
        # Final layers to generate point coordinates
        self.lin3 = nn.Linear(3, 512)  # (batch, 512)
        self.final_conv = nn.Conv1d(512, 3, kernel_size=1)
        
        # Activation
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        # Expand latent vector || F in size (batch, latent_dim)
        x = self.relu(self.bn1(self.lin1(x)))  # (batch, 1024)
        x = self.relu(self.bn2(self.lin2(x)))  # (batch, 2048)
        x = self.dropout(x)
        x = x.unsqueeze(-1)  # (batch, 2048, 1)
        
        # Pass through PointNet-like layers
        for layer in self.pointnet_layers:
            x = layer(x)  # (batch, 1024, 1)
            x = self.dropout(x)
        
        # Upsample to desired number of points
        output = []
        for up_layer in self.upsample_layers:
            x1 = up_layer(x)  # (batch, 512, ...)
            x1 = self.relu(x1)
            output.append(x1)
        
        # concat the result of the Upsample layers
        output = torch.cat(output, dim=1)
        x = self.lin3(output) # in (batch, num_points, 3) || out = (batch, num_points, 512)
        x = x.permute(0, 2, 1) # in (batch, num_points, 512) || out = (batch, 512, num_points)
        # Generate point coordinates
        x = self.final_conv(x)  # in (batch, 512, num_points)|| out (batch, 3, num_points)
        x = x.permute(0, 2, 1)  # (batch, num_points, 3)
        
        return x

def chamfer_loss(p1, p2):
    """
    Computes Chamfer Distance between two point clouds p1 and p2.
    
    Args:
        p1: [B, N, D] or [N, D] - predicted points
        p2: [B, N, D] or [N, D] - ground truth points
        
    Returns:
        Chamfer distance (scalar)
    """
    if p1.dim() == 2:
        p1 = p1.unsqueeze(0)
        p2 = p2.unsqueeze(0)

    B, N, D = p1.shape
    _, M, _ = p2.shape

    dist = torch.cdist(p1, p2, p=2)

    min_dist_p1 = dist.min(dim=2)[0]
    min_dist_p2 = dist.min(dim=1)[0]

    cd = (min_dist_p1.pow(2).mean(dim=1) + min_dist_p2.pow(2).mean(dim=1)).mean()
    return cd

def knn_density_estimation_batched(points_batch, k=10):
    """
    Compute density weights for each point cloud in a batch.
    
    Args:
        points_batch: [B, max_N, D] zero-padded
        k: k-NN count

    Returns:
        weights_batch: list of [N_i] tensors (1D density weights per cloud)
    """
    weights_list = []
    for i in range(len(points_batch)):
        pts = points_batch[i]
        with torch.no_grad():
            dist = torch.cdist(pts, pts)  # [N_i, N_i]
            knn_dists, _ = dist.topk(k=k+1, largest=False)
            knn_dists = knn_dists[:, 1:]
            avg_dists = knn_dists.mean(dim=1) + 1e-8
            weights = 1.0 / avg_dists
            weights = weights / weights.sum()
            weights_list.append(weights)
    return weights_list


def batched_density_aware_chamfer(pred_batch, target_batch, k=10):
    """
    Computes DACD for a batch of point clouds.

    Args:
        pred_batch: [B, max_N, D]
        target_batch: [B, max_M, D]
        pred_lens: list of lengths for pred_batch
        target_lens: list of lengths for target_batch

    Returns:
        Scalar total loss over batch
    """
    B = len(pred_batch)
    total_loss = 0.0

    pred_weights_list = knn_density_estimation_batched(pred_batch, k)
    target_weights_list = knn_density_estimation_batched(target_batch, k)

    for i in range(B):
        pred = pred_batch[i]
        target = target_batch[i]
        w_pred = pred_weights_list[i].to(pred.device)
        w_target = target_weights_list[i].to(pred.device)

        dist_pt = torch.cdist(pred, target)  # [N_i, M_i]
        dist_tp = dist_pt.T

        min_dists_pred, _ = dist_pt.min(dim=1)
        min_dists_target, _ = dist_tp.min(dim=1)

        loss = (w_pred * min_dists_pred**2).sum() + (w_target * min_dists_target**2).sum()
        total_loss += loss

    return total_loss / B

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class ChamferDistance:
    def __init__(self, batch_reduction="mean", point_reduction="sum", 
                 sensitivity_threshold=None, weight_x=1.0, weight_y=1.0, loss_color=None):
        """
        Initialize the Chamfer distance calculator with optional configurations.

        Args:
        batch_reduction (str): Type of reduction across batches ("mean" or "sum").
        point_reduction (str): Type of reduction across points ("mean" or "sum").
        sensitivity_threshold (float, optional): Threshold below which distances are set to zero.
        weight_x (float): Weight for the distances computed from x to y.
        weight_y (float): Weight for the distances computed from y to x.
        """
        self.batch_reduction = batch_reduction
        self.point_reduction = point_reduction
        self.sensitivity_threshold = sensitivity_threshold
        self.weight_x = weight_x
        self.weight_y = weight_y
        self.loss_color = loss_color

    def compute(self, x, y):
        """
        Compute the Chamfer distance between two sets of points using the initialized settings.

        Args:
        x (torch.Tensor): Tensor of shape (B, N, 3). # it should be the GT
        y (torch.Tensor): Tensor of shape (B, M, 3). # this is the prediction

        Returns:
        torch.Tensor: The computed Chamfer distance.
        """
        # Compute pairwise distances
        distances = torch.cdist(x, y)  # (B, N, M)

        # Compute minimum distances
        min_dist_x, _ = torch.min(distances, dim=2) # the truth calculate the nearest prediction
        min_dist_y, _ = torch.min(distances, dim=1) # the prediction calculate the nearest truth

        if self.sensitivity_threshold is not None:
            min_dist_x = torch.where(min_dist_x < self.sensitivity_threshold, torch.zeros_like(min_dist_x), min_dist_x)
            min_dist_y = torch.where(min_dist_y < self.sensitivity_threshold, torch.zeros_like(min_dist_y), min_dist_y)

        # Apply point reduction
        if self.point_reduction == "mean":
            min_dist_x_calculate = torch.mean(min_dist_x, dim=1)
            min_dist_y_calculate = torch.mean(min_dist_y, dim=1)
        elif self.point_reduction == "sum":
            min_dist_x_calculate = torch.sum(min_dist_x, dim=1)
            min_dist_y_calculate = torch.sum(min_dist_y, dim=1)
        else:
            raise ValueError("Invalid point_reduction: {}".format(self.point_reduction))

        # Compute weighted Chamfer distance
        chamfer_dist = self.weight_x * min_dist_x_calculate + self.weight_y * min_dist_y_calculate

        loss_value = chamfer_dist.detach().cpu().numpy()
        
        # Apply batch reduction
        if self.batch_reduction == "mean":
            chamfer_dist = torch.mean(chamfer_dist)
        elif self.batch_reduction == "sum":
            chamfer_dist = torch.sum(chamfer_dist)
        else:
            raise ValueError("Invalid batch_reduction: {}".format(self.batch_reduction))

        if self.loss_color is not None:
            loss_point_color = min_dist_y
            return chamfer_dist, loss_value, loss_point_color
        else:
            return chamfer_dist, loss_value

class StepLRWithMin(torch.optim.lr_scheduler.StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, min_lr=1e-4, last_epoch=-1):
        """
        Initializes the StepLRWithMin scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            step_size (int): Period of learning rate decay.
            gamma (float): Multiplicative factor of learning rate decay.
            min_lr (float): Minimum learning rate after decay.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.min_lr = min_lr
        super(StepLRWithMin, self).__init__(optimizer, step_size, gamma, last_epoch)
    
    def get_lr(self):
        """
        Override the get_lr method to enforce min_lr.
        """
        original_lrs = super().get_lr()
        # Enforce that the learning rate does not go below min_lr
        return [max(lr, self.min_lr) for lr in original_lrs]
    
if __name__ == "__main__":
    # Initialize model
    max_nodes = 4096
    recon_nodes_num = 4096
    latent_dim = 256
    writer = SummaryWriter(log_dir="./results/encoder1")
    vae = EncoderPoineTransformDecoderLinear_Gen_4(point_cloud_node=10000,
                                                     point_cloud_output=10000,
                                                     features=3,
                                                     bottleneck=256,
                                                     num_class=16,
                                                     num_layer=4)
    loss_fn = ChamferDistance(sensitivity_threshold = None,
                              weight_x= 1,
                              weight_y= 1)
    
    optimizer = torch.optim.Adam(params = vae.parameters(), 
                                 lr = 0.001,
                                 weight_decay = 0.0001)
    
    scheduler = StepLRWithMin(optimizer, step_size=1000, gamma=0.5, min_lr=0.0001)
    

    # Load dataset
    root = "../../../new_disk/WIP/Honda_Hood_10k/CarHoods10k"
    datasetTotal = ReaderDatasetPointV2(root, transform=Transform_center_and_scale())
    train_loader = DataLoader(datasetTotal, batch_size=8, shuffle=True)
    

    # Training loop
    num_epochs = 1000
    device = "cuda:1"
    vae.to(device)

    # KL annealing
    kl_weight_max = 1  # Lowered for Chamfer
    kl_weight = 0.0
    kl_anneal_epochs = 30  # Slower annealing

    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0


        for idx, (skinType, dataName, xFeature) in enumerate(train_loader):  
            if xFeature.shape[0] == 1:
                break  
                    
            # move to device
            xFeature = xFeature.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            x_recon, mu, logvar = vae(xFeature)
            
            # compute loss
            recons_loss, loss_individual = loss_fn.compute(xFeature, x_recon)
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # reconstruction loss scale
            recons_loss = recons_loss / 1000
            loss_individual = loss_individual / 1000
            
            # KL divergence loss scale
            kl_loss = kl_loss / 1000
            
            loss_combined = recons_loss + kl_loss
            
            # backpropagation and optimizer
            loss_combined.backward()
            optimizer.step()

            total_loss += loss_combined.item()
            total_recon_loss += recons_loss.item()
            total_kl_loss += kl_loss.item()

        # Update learning rate
        scheduler.step()

        if epoch % 10 == 0:
            torch.save(vae.state_dict() , "./results/encoder1/{}.pth".format(epoch+1))
        writer.add_scalar("train_loss", total_loss, global_step=epoch+1)
        writer.add_scalar("train_kl_loss", total_kl_loss, global_step=epoch+1)
        writer.add_scalar("train_recon_loss", total_recon_loss, global_step=epoch+1)
        writer.add_scalar("kl_weight", kl_weight, global_step=epoch+1)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step=epoch+1)

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    # Save encoder only
    torch.save(vae.encoder.state_dict(), "mesh_encoder.pth")