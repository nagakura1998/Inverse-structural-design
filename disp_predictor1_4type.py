import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataset import MeshDataset
from torch.utils.tensorboard import SummaryWriter
from vae1_4type import MeshVAE
import torch.optim as optim

class DispPredictor(nn.Module):
    def __init__(self, latent_dim=64, output_dim=1, dropout=0.2):  # e.g. max displacement
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, z):
        return self.net(z)

if __name__ == "__main__": 
    max_nodes = 5000
    recon_nodes_num = 4096
    latent_dim = 64
    device="cuda:0"
    writer = SummaryWriter(log_dir="./results/disp_predictor1")
    vae = MeshVAE(point_cloud_node=recon_nodes_num,
                                                     point_cloud_output=recon_nodes_num,
                                                     features=3,
                                                     bottleneck=latent_dim,
                                                     num_class=16,
                                                     num_layer=4)
    vae.load_state_dict(torch.load("./results/vae1/271.pth", map_location="cpu"))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    disp_predictor = DispPredictor(latent_dim=latent_dim, output_dim=1)
    vae.to(device)
    disp_predictor.to(device)

    optimizer = optim.Adam(disp_predictor.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    #mse_loss_fn = nn.MSELoss()
    mse_loss_fn = nn.SmoothL1Loss()

    # Load dataset
    dataset = MeshDataset("./dataset2/train", max_nodes=max_nodes, recon_nodes_num=recon_nodes_num)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Training loop
    num_epochs = 1000

    for epoch in range(num_epochs):
        disp_predictor.train()
        total_loss = 0

        for batch_id, batch in enumerate(dataloader):
            step = batch_id + epoch* num_epochs
            #padded_nodes = batch["padded_nodes"].to(device)
            #stress = batch["stress"].to(device)
            recon_nodes_gt = batch["recon_nodes"].to(device)
            disp_max_gt = batch["disp_max"].view(-1,1).to(device)
            
            optimizer.zero_grad()

            with torch.no_grad():
                z = vae.Get_z_representation(recon_nodes_gt)

            max_disp = disp_predictor(z)
                        
            loss = mse_loss_fn(max_disp, disp_max_gt)
            
            loss.backward()

            # Gradient clipping
            #grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()

        # Update learning rate
        scheduler.step(total_loss/len(dataloader))

        if epoch % 10 == 0:
            torch.save(disp_predictor.state_dict() , "./results/disp_predictor1/{}.pth".format(epoch+1))
        writer.add_scalar("train_loss", total_loss, global_step=epoch+1)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step=epoch+1)

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    # Save encoder only
    torch.save(vae.encoder.state_dict(), "mesh_encoder.pth")