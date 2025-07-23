import torch
import torch.nn as nn
from test_train import PointTransformerCls
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataset import MeshDataset
from torch.utils.tensorboard import SummaryWriter

class TemporaryDecoder(nn.Module):
    def __init__(self, max_nodes, latent_dim=64):
        super(TemporaryDecoder, self).__init__()
        self.max_nodes = max_nodes
        # Learnable node embeddings
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(32, 7), 
        )
        #self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == 3:  # Final layer
                    nn.init.normal_(m.weight, mean=0, std=0.005)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, z):
        params =  self.mlp(z)
        plate_logits, continuous_params = params[:, :4], params[:, 4:]
        plate_probs = torch.nn.functional.softmax(plate_logits, dim=1)
        return plate_probs, continuous_params

class MeshVAE(nn.Module):
    def __init__(self, point_cloud_node = 10000, point_cloud_output = 10000, features= 3, bottleneck = 256, num_class = 16, num_layer = 4) -> None:
        super(MeshVAE, self).__init__()
        self.features = features
        self.latent_space = bottleneck
        # npoint, input_dim, num_class, bottleneck
        self.encoder = PointTransformerCls(npoint = point_cloud_node, 
                                           input_dim = self.features,
                                           num_class = num_class, 
                                           bottleneck = bottleneck)
        self.decoder = TemporaryDecoder(8, bottleneck)
        
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
        plate_probs, continuous_params = self.decoder(latent)  # B x n_prims x 3 x n_verts
        
        return plate_probs, continuous_params, mu, logvar
    
    def get_latent_space(self, x):
        # x = x.view(int(x.shape[0]/256), 256)
        # Encoding
        _, latent = self.encoder(x)
        # mu = self.map_mu(latent)
        # log_var = self.map_logvar(latent)
        # # Reparameterization
        # z = self.reparameterize(mu, log_var)
        return latent

    def Get_z_representation(self, features):
        _, latent_f = self.encoder(features)
        
        mu = self.map_mu(latent_f)
        logvar = self.map_logvar(latent_f)
        
        latent = self.reparameterize(mu, logvar)
        return latent
    
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

if __name__ == "__main__":
    # Initialize model
    max_nodes = 4096
    recon_nodes_num = 4096
    latent_dim = 64
    writer = SummaryWriter(log_dir="./results/vae1")
    vae = MeshVAE(point_cloud_node=recon_nodes_num,
                                                     point_cloud_output=recon_nodes_num,
                                                     features=3,
                                                     bottleneck=latent_dim,
                                                     num_class=16,
                                                     num_layer=4)
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    
    # Load dataset
    dataset = MeshDataset("./dataset2/train", max_nodes=max_nodes, recon_nodes_num=recon_nodes_num)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Training loop
    num_epochs = 1000
    device = "cuda:0"
    vae.to(device)
    # KL annealing
    kl_weight_max = 1  # Lowered for Chamfer
    kl_weight = 0.0
    kl_anneal_epochs = 30  # Slower annealing

    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        total_mse = 0
        total_ce = 0
        total_kl_loss = 0

        # Update KL weight
        if epoch < kl_anneal_epochs:
            kl_weight = kl_weight_max * (epoch / kl_anneal_epochs)

        for batch_id, batch in enumerate(dataloader):
            step = batch_id + epoch* num_epochs
            #padded_nodes = batch["padded_nodes"].to(device)
            #stress = batch["stress"].to(device)
            recon_nodes_gt = batch["recon_nodes"].to(device)
            #recon_disp_gt = batch["recon_disp"].unsqueeze(2).to(dtype=torch.float32, device=device)
            #mask = batch["mask"].to(device)
            #sigma_max = batch["sigma_max"].view(-1,1).to(device)
            shape = batch["shape"].to(dtype=torch.int64, device=device)
            #shape = torch.nn.functional.one_hot(batch["shape"].to(torch.int64), num_classes=4).to(device)
            param = batch["param"][:,:3].to(dtype=torch.float32, device=device)
            # Compute node scale from ground truth
            #node_scale = recon_nodes_gt.abs().max().item() + 1e-6

            optimizer.zero_grad()
            
            #recon_nodes, mu, logvar = vae(padded_nodes, stress, mask)
            #x = torch.cat([recon_nodes_gt, recon_stress_gt], dim=2)
            plate_probs, continuous_params, mu, logvar = vae(recon_nodes_gt)
            
            ce_loss = ce_loss_fn(plate_probs, shape)
            mse_loss = mse_loss_fn(continuous_params, param)
            #recon_loss = criterion(recon_param, recon_param_gt)
            #recon_loss = chamfer_loss(recon_nodes, recon_nodes_gt)
            #recon_loss = batched_density_aware_chamfer(recon_nodes, recon_nodes_gt, k=10)
            kl_loss = kl_divergence(mu, logvar) / max_nodes
            
            loss = 10*ce_loss + 100*mse_loss + kl_weight * kl_loss
            
            loss.backward()

            # Gradient clipping
            #grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_ce += ce_loss.item()
            total_kl_loss += kl_loss.item()

        # Update learning rate
        scheduler.step(total_loss/len(dataloader))

        if epoch % 10 == 0:
            torch.save(vae.state_dict() , "./results/vae1/{}.pth".format(epoch+1))
        writer.add_scalar("train_loss", total_loss, global_step=epoch+1)
        writer.add_scalar("train_kl_loss", total_kl_loss, global_step=epoch+1)
        writer.add_scalar("train_mse_loss", total_mse, global_step=epoch+1)
        writer.add_scalar("train_ce_loss", total_ce, global_step=epoch+1)
        writer.add_scalar("kl_weight", kl_weight, global_step=epoch+1)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step=epoch+1)

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    # Save encoder only
    torch.save(vae.encoder.state_dict(), "mesh_encoder.pth")