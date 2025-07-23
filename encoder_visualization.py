import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataset import MeshDataset
from vae1_4type import MeshVAE
from torch.utils.data import DataLoader, Dataset

# Assume you have a DataLoader for your dataset
# and a trained encoder model
def extract_latents(vae, dataloader, device):
    vae.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            recon_nodes_gt = batch["recon_nodes"].to(device)
            z = vae.Get_z_representation(recon_nodes_gt)  # shape: [B, latent_dim]
            latents.extend(z.tolist())
            labels.extend(batch["shape"].to(torch.int64).tolist())

    #latents = torch.cat(latents, dim=0).numpy()
    return latents, labels

def plot_latents(path, latents, labels, method='PCA'):
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 't-SNE':
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca', random_state=42)
    else:
        raise ValueError("method must be 'PCA' or 't-SNE'")

    reduced = reducer.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.title(f"{method} of Latent Space")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label="Labels")
    plt.grid(True)
    # plt.show()
    plt.savefig(path)

# Initialize model
recon_nodes_num = 4096
latent_dim = 64
vae = MeshVAE(point_cloud_node=recon_nodes_num,
              point_cloud_output=recon_nodes_num,
              features=3,
              bottleneck=latent_dim,
              num_class=16,
              num_layer=4)
vae.load_state_dict(torch.load("./results/vae1/991.pth", map_location="cpu"))
# Load dataset
dataset = MeshDataset("./dataset2/train", max_nodes=5000, recon_nodes_num=recon_nodes_num)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

device = "cuda:1"
vae.to(device)

latents, labels = extract_latents(vae, dataloader, device)

latents = torch.tensor(latents, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.int64)
# PCA plot
plot_latents("./PCA.png", latents, labels, method='PCA')

# t-SNE plot
plot_latents("./TSNE.png", latents, labels, method='t-SNE')