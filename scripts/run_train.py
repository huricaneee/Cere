import torch
import os
from telepath.fmriTransformer.model import FMRITransformerModel, Config, FMRIEncoderOnlyModel
from telepath.fmriTransformer.train import train_one_epoch, evaluate
from telepath.fmriTransformer.data_load_align import prepare_and_save_aligned_data
from torch.utils.data import DataLoader
import wandb

# Configuration
root_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
subject = 1
modality = "all"
window_size = 7
stride = 5
batch_size = 128

# Train/val split
# train_episodes = [f"s01e{episode:02d}{half}" for episode in range(1, 25) for half in ['a', 'b']]
# val_episodes = [f"s02e{episode:02d}{half}" for episode in range(1, 13) for half in ['a', 'b']]

# Train on seasons 1–5
train_episodes = [
    f"s0{season}e{episode:02d}{half}"
    for season in range(1, 6)         # Seasons 1 to 5
    for episode in range(1, 25)       # 24 episodes per season
    for half in ['a', 'b']            # First and second halves
]

# Validate on season 6
val_episodes = [
    f"s06e{episode:02d}{half}"
    for episode in range(1, 25)       # 24 episodes in season 6
    for half in ['a', 'b']
]

# Load data
train_loader = prepare_and_save_aligned_data(
    root_data_dir=root_data_dir,
    subject_id=subject,
    selected_episodes=train_episodes,
    window_size=window_size,
    stride=stride,
    batch_size=batch_size,
    excluded_trs_start=0,
    excluded_trs_end=0,
    hrf_delay=3,
    shuffle=True,
    save_path=os.path.join(os.path.dirname(__file__), '..', 'aligned', 'train.pkl')
)

val_loader = prepare_and_save_aligned_data(
    root_data_dir=root_data_dir,
    subject_id=subject,
    selected_episodes=val_episodes,
    window_size=window_size,
    stride=stride,
    batch_size=batch_size,
    excluded_trs_start=0,
    excluded_trs_end=0,
    hrf_delay=3,
    shuffle=False,
    save_path= os.path.join(os.path.dirname(__file__), '..', 'aligned', 'val.pkl')

)

# Get input/output dimensions from a sample batch
stim_batch, fmri_batch = next(iter(train_loader)) 
example_stim = stim_batch[0]
example_fmri = fmri_batch[0]



input_dim = example_stim.shape[-1]
output_dim = example_fmri.shape[-1]


# Initialize model
config = Config(input_dim=input_dim, output_dim=output_dim, window_size= window_size, stride=stride)
#model = FMRITransformerModel(config).to(config.device)
model = FMRIEncoderOnlyModel(config).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

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
        os.path.dirname(__file__), '..', 'checkpoints',
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