import torch
from torch import nn, optim
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
from IPython.display import HTML
import pickle
from Models import *

# Configuration
batch_size = 64
image_size = 64
image_dim = 3
channels = 3
z_dim = 100
hidden_dim = 64
epochs = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

folder_path = "data/cats"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")
else:
    print(f"Folder '{folder_path}' already exists.")

# Dataset
transform = transforms.Compose([
	transforms.Resize(image_size),
	transforms.CenterCrop(image_size),
	transforms.ToTensor()
])

dset = ImageFolder(root="./data", transform=transform)
loader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=True)

real_batch = next(iter(loader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training images")
plt.imshow(np.transpose(make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
# plt.show()

fixed_noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
main_path = 'Logs'
outputs_folder = 'Training Outputs'
outputs_path = os.path.join(main_path, outputs_folder)

def view_outputs_during_training(epoch, batch_idx, save=True, outputs_path=outputs_path, plot=True, return_grid=False):
  with torch.no_grad():
    fake = gen(fixed_noise).detach().to(device)
    plt.figure(figsize=(8, 8))
    grid = make_grid(fake[:64], padding=5, normalize=True).cpu().numpy().transpose((1, 2, 0))

    if plot:
      plt.axis("off")
      plt.title(f"[{epoch}/{epochs}][{batch_idx}/{len(loader)}]")
      plt.imshow(grid)
      plt.show()

    if save:
        img_path = os.path.join(outputs_path, f'[{epoch}-{epochs}][{batch_idx}-{len(loader)}].png')
        plt.imsave(img_path, grid)

    # Return to save to img_list
    if return_grid:
      return grid

gen = Generator(image_dim=image_dim, z_dim=z_dim, hidden_dim=hidden_dim).to(device)
disc = Discriminator(image_dim=image_dim, hidden_dim=hidden_dim).to(device)
# Download required files from here:
### https://drive.google.com/drive/folders/1rd1cZAw4sp9lXINUy7hQEgks5sO0SVlk?usp=sharing
## Then put all of them inside "Logs" folder

gen_model_file = 'gen1.pth'
gen_model_path = os.path.join(main_path, gen_model_file)

disc_model_file = 'disc1.pth'
disc_model_path = os.path.join(main_path, disc_model_file)

gen.apply(weight_init)
disc.apply(weight_init)

learning_rate=2e-4
opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Load state dict (uncomment if you want to load the weights)
# gen.load_state_dict(torch.load(gen_model_path))
# disc.load_state_dict(torch.load(disc_model_path))


# Training loop
img_list = []
D_loss = []
G_loss = []
print_loss_interval = 50
view_grid_interval = 200

criterion = nn.BCELoss()

print("Starting training...")
for epoch in range(epochs):
  for batch_idx, (real_images, _) in enumerate(loader):
    ### Discriminator ###
    real = real_images.to(device)
    batch = real.shape[0]

    noise = torch.randn((batch, z_dim, 1, 1)).to(device)
    fake = gen(noise).to(device)

    real_pred = disc(real).reshape(-1)
    fake_pred = disc(fake.detach()).reshape(-1)

    ones = torch.ones_like(real_pred)
    zeros = torch.zeros_like(fake_pred)

    d_loss_real = criterion(real_pred, ones)
    d_loss_fake = criterion(fake_pred, zeros)

    d_loss = (d_loss_real + d_loss_fake)
    D_loss.append(d_loss.item())

    ### Update discriminator ###
    opt_disc.zero_grad()
    d_loss.backward()
    opt_disc.step()

    ### Generator ###
    output = disc(fake).reshape(-1)
    g_loss = criterion(output, torch.ones_like(output))
    G_loss.append(g_loss.item())

    ### Update Generator ###
    opt_gen.zero_grad()
    g_loss.backward()
    opt_gen.step()

    # Print loss after `print_loss_interval` batches.
    if batch_idx % print_loss_interval == 0:
      print(f"[{epoch}/{epochs}][{batch_idx}/{len(loader)}]\tD_loss: {d_loss.item()}\tG_loss: {g_loss.item()}")

    # View grid images after `view_grid_interval` batches.
    if batch_idx % view_grid_interval == 0:
      grid = view_outputs_during_training(epoch=epoch, batch_idx=batch_idx, save=True, return_grid=True)
      img_list.append(grid)


"""Things to do after training (even if it is interrupted)"""

pickle_file = 'img_list.pkl'

img_list_full_path = os.path.join(main_path, pickle_file)

# Save img_list to a file
with open(img_list_full_path, 'wb') as f:
    pickle.dump(img_list, f)

# Save losses into a file
loss_file = 'loss_analysis.csv'
loss_file_path = os.path.join(main_path, loss_file)

with open(loss_file_path, 'w') as f:
    f.write("Epoch,Batch_idx,D_loss,G_loss\n")
    for index, (d_loss, g_loss) in enumerate(zip(D_loss, G_loss)):
        epoch = index // len(loader)
        batch_idx = index % len(loader)
        f.write(f"{epoch},{batch_idx},{d_loss},{g_loss}\n")

# Save models
torch.save(gen.state_dict(), gen_model_path)
torch.save(disc.state_dict(), disc_model_path)