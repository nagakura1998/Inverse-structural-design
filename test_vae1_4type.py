import torch
import torch.nn as nn
from test_train import PointTransformerCls
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataset import MeshDataset
from vae1_4type import MeshVAE
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.colors import Normalize 
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    # since density = bin_count / sample_count / bin_area
    # we will change it to density = bin_count / sample_count
    data = data * ((x_e[1] - x_e[0]) * (y_e[1] - y_e[0]))
    
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "linear", bounds_error = False)
    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    # norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    # cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    # cbar.ax.set_ylabel('Density')
    return ax

def plotComponent(path, actual, predict):
    maxValue = max(np.max(actual), np.max(predict))
    minValue = min(np.min(actual), np.min(predict))

    rainbow = LinearSegmentedColormap.from_list('rainbow', [
        (0, "#000066"),
        (0.1, '#0000ff'),
        (0.2, '#0080ff'),
        (0.3, '#00ffff'),
        (0.4, '#00ff80'),
        (0.5, '#00ff00'),
        (0.6, '#80ff00'),
        (0.7, '#ffff00'),
        (0.8, '#ff8000'),
        (0.9, '#ff0000'),
        (1.0, '#660000'),
    ], N=256)
    with plt.style.context('default'):
            vmin = 0
            vmax = 0.2
            fig, ax = plt.subplots(figsize=(8, 6))
            diag_x = np.array([minValue, maxValue])
            # diag_x = np.array([-20, 20])
            diag_y = diag_x
            r2 = r2_score(actual, predict)
            rmse = mean_squared_error(actual, predict, squared=False)  # squared=False -> return rmse
            mae = abs(actual - predict).mean()
            stat_label = f"$R^2$={r2:.2f}, MAE={mae:.3f}, RMSE={rmse:.1e}"
            ax.scatter(actual, predict, color='black', s=7, alpha=1.0, label=stat_label)
            
            density_scatter(actual, predict, ax, marker='.', bins=30, s=5, alpha=0.6, cmap=rainbow, vmin=vmin, vmax=vmax)
            model = LinearRegression()
            model.fit(actual.reshape(-1, 1), predict.reshape(-1, 1))
            fit_x = np.array([minValue, maxValue]).reshape(-1, 1)
            # fit_x = np.array([-20, 20]).reshape(-1, 1)
            fit_y = model.predict(fit_x)
            fit_label = f'y = {model.coef_.item():.3f}x {"+" if model.intercept_ >= 0 else "-"} {abs(model.intercept_.item()):.3f}'
            ax.plot(fit_x, fit_y, linewidth=2, color='magenta', label=fit_label)
            ax.plot(diag_x, diag_y, linewidth=1, linestyle='dashed', color='black', label='y = x')
        
            ax.set_xlabel("actual")
            ax.set_ylabel("predicted")
            ax.set_xlim(minValue, maxValue)
            ax.set_ylim(minValue, maxValue)
            ax.grid(True)
            ax.legend()
        
            norm = Normalize(vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(cm.ScalarMappable(norm = norm, cmap=rainbow), ax=ax)
            cbar.ax.set_ylabel('Proportion')
            plt.savefig(path)

if __name__ == "__main__":
    recon_nodes_num = 4096
    latent_dim = 64
    vae = MeshVAE(point_cloud_node=recon_nodes_num,
                                                     point_cloud_output=recon_nodes_num,
                                                     features=3,
                                                     bottleneck=latent_dim,
                                                     num_class=16,
                                                     num_layer=4)
    vae.load_state_dict(torch.load("./results/vae1/271.pth", map_location="cpu"))

    dataset = MeshDataset("./dataset2/train", max_nodes=5000, recon_nodes_num=recon_nodes_num)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    device = "cuda:1"
    vae.to(device)
    vae.eval()

    pred = {"a": [], "b":[], "c":[]}
    actual = {"a": [], "b":[], "c":[]}

    pred_type = []
    actual_type = []

    for batch_id, batch in enumerate(dataloader):
        #padded_nodes = batch["padded_nodes"].to(device)
        #stress = batch["stress"].to(device)
        recon_nodes_gt = batch["recon_nodes"].to(device)
        #recon_disp_gt = batch["recon_disp"].unsqueeze(2).to(dtype=torch.float32, device=device)
        #mask = batch["mask"].to(device)
        #sigma_max = batch["sigma_max"].view(-1,1).to(device)
        shape = batch["shape"].to(dtype=torch.int64, device=device)
        #shape = torch.nn.functional.one_hot(batch["shape"].to(torch.int64), num_classes=4).to(device)
        param = batch["param"].to(dtype=torch.float32, device=device)
        # Compute node scale from ground truth
        #node_scale = recon_nodes_gt.abs().max().item() + 1e-6
        
        #recon_nodes, mu, logvar = vae(padded_nodes, stress, mask)
        #x = torch.cat([recon_nodes_gt, recon_stress_gt], dim=2)
        plate_probs, continuous_params, mu, logvar = vae(recon_nodes_gt)

        plate_type = torch.argmax(plate_probs, dim=1)

        pred_type.extend(plate_type.tolist())
        actual_type.extend(shape.tolist())

        pred["a"].extend(continuous_params[:,0].detach().tolist())
        actual["a"].extend(param[:,0].tolist())

        pred["b"].extend(continuous_params[:,1].detach().tolist())
        actual["b"].extend(param[:,1].tolist())

        pred["c"].extend(continuous_params[:,2].detach().tolist())
        actual["c"].extend(param[:,2].tolist())

    class_names = ['Circle', 'Square', 'Diamond', 'Hexa']

    # Compute confusion matrix
    cm_data = confusion_matrix(actual_type, pred_type, labels=[0, 1, 2, 3])

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_data, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')  # 'd' = integer format
    plt.title('Confusion Matrix')
    plt.savefig("type.png")

    for i in pred:
        ac = np.array(actual[i])
        pr = np.array(pred[i])
        plotComponent("{}.png".format(i), ac, pr)