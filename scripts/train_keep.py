import os
import torch
from telepath.fmriTransformer.model import FMRITransformerModel, Config
from telepath.fmriTransformer.train import train_one_epoch, evaluate
from telepath.fmriTransformer.data_load_align import prepare_and_save_aligned_data
from torch.utils.data import DataLoader
import wandb

# === Load the checkpoint ===
job_id = os.environ.get("SLURM_JOB_ID", "local")  # Or use a fixed string if testing locally
# checkpoint_path = os.path.join(
#     os.path.dirname(__file__), '..', 'checkpoints', f'{job_id}_epoch_3.pt'  # Replace with your last checkpoint
# )

root_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
subject = 1
modality = "all"

checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', '63851961_epoch_4.pt')

checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# === Restore config ===
config = Config()
config.__dict__.update(checkpoint['config'])

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
    window_size=config.window_size,
    stride=config.stride,
    batch_size=config.batch_size,
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
    window_size=config.window_size,
    stride=config.stride,
    batch_size=config.batch_size,
    excluded_trs_start=0,
    excluded_trs_end=0,
    hrf_delay=3,
    shuffle=False,
    save_path= os.path.join(os.path.dirname(__file__), '..', 'aligned', 'val.pkl')

)


# === Recreate model and optimizer ===
model = FMRITransformerModel(config).to(config.device)
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Ensure all optimizer tensors are on the correct device
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(config.device)


wandb.init(
    project="fmri-transformer",
    config=config.__dict__
)

print("Model Configuration:")
print(config)

# === Resume training ===
start_epoch = checkpoint['epoch'] + 1
for epoch in range(start_epoch, 10):
    train_one_epoch(model, train_loader, optimizer, config.device, epoch)
    evaluate(model, val_loader, config.device, epoch)

    # Save new checkpoint
    new_checkpoint_path = os.path.join(
        os.path.dirname(__file__), '..', 'checkpoints', f'{job_id}_epoch_{epoch+1}.pt'
    )
    os.makedirs(os.path.dirname(new_checkpoint_path), exist_ok=True)
    new_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__
    }
    torch.save(new_checkpoint, new_checkpoint_path)

    # Upload to wandb
    import wandb
    wandb.save(new_checkpoint_path)
