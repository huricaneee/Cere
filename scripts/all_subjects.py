import torch
import os
from telepath.fmriTransformer.model import FMRITransformerModel, Config, FMRIEncoderOnlyModel
from telepath.fmriTransformer.train import train_one_epoch, evaluate
from telepath.fmriTransformer.data_load_align import prepare_and_save_aligned_data, build_combined_dataloader
from torch.utils.data import DataLoader
import wandb

# Configuration
root_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
subject = [1, 2, 3, 5]
modality = "all"
window_size = 7
stride = 5
batch_size = 256

# Train/val split
# train_episodes = [f"s01e{episode:02d}{half}" for episode in range(1, 25) for half in ['a', 'b']]
# val_episodes = [f"s02e{episode:02d}{half}" for episode in range(1, 13) for half in ['a', 'b']]

# Train on seasons 1–5
# 

# Train on seasons 1–5
train_episodes = [
    's01', 's02', 's03', 's04', 's05',
]

# Validate on season 6
val_episodes = [
    's06'
]

temp_train_loader = build_combined_dataloader(
    root_data_dir=root_data_dir,
    subjects=subject,
    window_size=window_size,
    stride=stride,
    batch_size=1,  # Use small batch size just to get dimensions
    cache_dir=os.path.join(os.path.dirname(__file__), '..', 'aligned_data_all_sub', 'train'),
    excluded_trs_start=0,
    excluded_trs_end=0,
    hrf_delay=3,
    selected_episodes=train_episodes,
    shuffle=False,  # No need to shuffle for dimension checking
)

# Get input/output dimensions from a sample batch
stim_batch, fmri_batch = next(iter(temp_train_loader))
example_stim = stim_batch[0]
example_fmri = fmri_batch[0]

input_dim = example_stim.shape[-1]
output_dim = example_fmri.shape[-1]

# Initialize model
config = Config(input_dim=input_dim, output_dim=output_dim, window_size= window_size, stride=stride)
model = FMRITransformerModel(config).to(config.device)
#model = FMRIEncoderOnlyModel(config).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)

# Load data
train_loader = build_combined_dataloader(
    root_data_dir=root_data_dir,
    subjects=subject,
    window_size=config.window_size,
    stride=config.stride,
    batch_size=config.batch_size,
    cache_dir=os.path.join(os.path.dirname(__file__), '..', 'aligned_data_all_sub', 'train'),
    excluded_trs_start=0,
    excluded_trs_end=0,
    hrf_delay=3,
    selected_episodes=train_episodes,
    shuffle=True,
)

val_loader = build_combined_dataloader(
    root_data_dir=root_data_dir,
    subjects=subject,
    window_size=config.window_size,
    stride=config.stride,
    batch_size=config.batch_size,
    cache_dir=os.path.join(os.path.dirname(__file__), '..', 'aligned_data_all_sub', 'val'),
    excluded_trs_start=0,
    excluded_trs_end=0,
    hrf_delay=3,
    selected_episodes=val_episodes,
    shuffle=True,
)



# Initialize WandB
wandb.init(
    project="fmri-transformer",
    config=config.__dict__
)

print("Model Configuration:")
print(config)


# Training loop
for epoch in range(config.num_epochs):
    train_one_epoch(model, train_loader, optimizer, config.device, epoch)
    evaluate(model, val_loader, config.device, epoch)

    job_id = os.environ.get("SLURM_JOB_ID", "local")  # fallback if not running on SLURM

    checkpoint_path = os.path.join(
        os.path.dirname(__file__), '..', 'all_subs_checkpoints',
        f'{job_id}_epoch_{epoch+1}.pt'
    )

    #checkpoint_path = f"/content/checkpoints/epoch_{epoch+1}.pt"
    # checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', f'epoch_{epoch+1}.pt')
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__
    }

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    # upload to WandB
    wandb.save(checkpoint_path)  