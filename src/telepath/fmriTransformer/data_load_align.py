import os
import math
import numpy as np
import h5py
import torch
import pickle
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr


# def load_stimulus_features(root_data_dir, modality, selected_episodes=None):
#     """
#     Load stimulus features from a structured HDF5 file.

#     Args:
#         h5_path (str): Path to features_Imagebind.h5
#         modality (str): One of 'vision', 'audio', 'text', or 'all'
#         selected_episodes (list[str] or None): Optional filter on episode keys

#     Returns:
#         dict: {modality: {episode: np.ndarray of shape [T, D]}}
#     """
#     features = {}

#     h5_path = os.path.join(root_data_dir, 'features', 'features_Imagebind_6season.h5')

#     with h5py.File(h5_path, 'r') as f:
#         available_modalities = list(f.keys())

#         if modality == "all":
#             modalities = available_modalities
#         elif modality in available_modalities:
#             modalities = [modality]
#         else:
#             raise ValueError(f"Modality '{modality}' not found in file. Available: {available_modalities}")

#         for mod in modalities:
#             features[mod] = {}
#             for ep_key in f[mod].keys():
#                 if selected_episodes is None or ep_key in selected_episodes:
#                     features[mod][ep_key] = f[f"{mod}/{ep_key}"][:]

#     return features


# def load_fmri(root_data_dir, subject, friends_episodes=None):
#     fmri = {}
#     friends_filename = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
#     friends_path = os.path.join(root_data_dir, 'fmri', friends_filename)

#     with h5py.File(friends_path, 'r') as f:
#         for key, val in f.items():
#             clip = str(key[13:])
#             if friends_episodes is None or clip in friends_episodes:
#                 fmri[clip] = val[:].astype(np.float32)

#     return fmri

# # concatenate and trim features and fMRI data
# def trim_and_concatenate_features(features_dict, fmri_dict, excluded_trs_start=0, excluded_trs_end=0, hrf_delay=0):
#     aligned_features = []
#     aligned_fmri = []
#     all_episodes = set(fmri_dict.keys())
#     for mod in features_dict:
#         all_episodes &= set(features_dict[mod].keys())

#     for ep in sorted(all_episodes):
#         fmri = fmri_dict[ep]
#         if len(fmri) <= excluded_trs_start + excluded_trs_end + hrf_delay:
#             continue
#         fmri_trimmed = fmri[excluded_trs_start : -excluded_trs_end or None]
#         fmri_trimmed = fmri_trimmed[:len(fmri_trimmed) - hrf_delay]

#         modality_features = []
#         min_len = len(fmri_trimmed)
#         valid = True

#         for mod in sorted(features_dict.keys()):
#             feat = features_dict[mod][ep]
#             if len(feat) <= excluded_trs_start + excluded_trs_end + hrf_delay:
#                 valid = False
#                 break
#             feat_trimmed = feat[excluded_trs_start : -excluded_trs_end or None]
#             feat_trimmed = feat_trimmed[hrf_delay:]
#             min_len = min(min_len, len(feat_trimmed))
#             modality_features.append(feat_trimmed)

#         if not valid:
#             continue

#         feat_concat = np.concatenate([f[:min_len] for f in modality_features], axis=1)
#         aligned_features.append(feat_concat)
#         aligned_fmri.append(fmri_trimmed[:min_len])

#     return aligned_features, aligned_fmri



# -------------------------------------------------------
# Helper: match either full episode IDs or prefix strings
# -------------------------------------------------------
def _match_episode(ep_key: str, selectors):
    """Return True if ep_key matches any selector (full ID or prefix)."""
    if selectors is None:
        return True
    for sel in selectors:
        if ep_key.startswith(sel):     # prefix match
            return True
    return False


# -------------------------------------------------------
# Updated stimulus-feature loader
# -------------------------------------------------------
def load_stimulus_features(root_data_dir, modality="all", selected_episodes=None):
    """
    Load ImageBind (or other) stimulus features.

    Args
    ----
    root_data_dir : str
    modality      : "vision" | "audio" | "text" | "all"
    selected_episodes : None | list[str] | tuple[str]
        Each string can be a *full* episode key ("s04e01a") or a *prefix*
        ("s04", "s05e").  If None, keep everything.
    """
    features = {}
    h5_path = os.path.join(root_data_dir, "features",
                           "features_Imagebind_6season.h5")

    with h5py.File(h5_path, "r") as f:
        available_modalities = list(f.keys())

        if modality == "all":
            modalities = available_modalities
        elif modality in available_modalities:
            modalities = [modality]
        else:
            raise ValueError(
                f"Modality '{modality}' not found. "
                f"Available: {available_modalities}"
            )

        for mod in modalities:
            features[mod] = {}
            for ep_key in f[mod]:
                if _match_episode(ep_key, selected_episodes):
                    features[mod][ep_key] = f[f"{mod}/{ep_key}"][:]

    return features


# -------------------------------------------------------
# Updated fMRI loader
# -------------------------------------------------------
def load_fmri(root_data_dir, subject, friends_episodes=None):
    """
    Load Schaefer-1000 parcel fMRI for one subject.

    friends_episodes : None | list[str] | tuple[str]
        Same semantics as selected_episodes above (full IDs or prefixes).
    """
    fmri = {}
    fname = (f"sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_"
             "atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5")
    fpath = os.path.join(root_data_dir, "fmri", fname)

    with h5py.File(fpath, "r") as f:
        for key, val in f.items():
            ep_key = str(key[13:])          # strip '/friends_' prefix
            if _match_episode(ep_key, friends_episodes):
                fmri[ep_key] = val[:].astype(np.float32)

    return fmri


## add features and trim fMRI data
def trim_and_concatenate_features(features_dict, fmri_dict, excluded_trs_start=0, excluded_trs_end=0, hrf_delay=0):
    aligned_features = []
    aligned_fmri = []
    all_episodes = set(fmri_dict.keys())
    for mod in features_dict:
        all_episodes &= set(features_dict[mod].keys())

    for ep in sorted(all_episodes):
        fmri = fmri_dict[ep]
        if len(fmri) <= excluded_trs_start + excluded_trs_end + hrf_delay:
            continue
        fmri_trimmed = fmri[excluded_trs_start : -excluded_trs_end or None]
        fmri_trimmed = fmri_trimmed[:len(fmri_trimmed) - hrf_delay]

        modality_features = []
        min_len = len(fmri_trimmed)
        valid = True

        for mod in sorted(features_dict.keys()):
            feat = features_dict[mod][ep]
            if len(feat) <= excluded_trs_start + excluded_trs_end + hrf_delay:
                valid = False
                break
            feat_trimmed = feat[excluded_trs_start : -excluded_trs_end or None]
            feat_trimmed = feat_trimmed[hrf_delay:]
            min_len = min(min_len, len(feat_trimmed))
            modality_features.append(feat_trimmed)

        if not valid:
            continue

        # Ensure all features are of shape [min_len, D] and then sum them
        feat_sum = np.sum([f[:min_len] for f in modality_features], axis=0)
        aligned_features.append(feat_sum)
        aligned_fmri.append(fmri_trimmed[:min_len])

    return aligned_features, aligned_fmri



def segment_sequence(seq, window_size, stride):
    segments = []
    T, D = seq.shape
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        segments.append(seq[start:end])
    return segments


class SlidingWindowFMRIDataset(Dataset):
    def __init__(self, full_data, window_size=100, stride=50):
        self.samples = []
        for stim_seq, fmri_seq in full_data:
            stim_seq = torch.tensor(stim_seq, dtype=torch.float32)
            fmri_seq = torch.tensor(fmri_seq, dtype=torch.float32)

            stim_windows = segment_sequence(stim_seq, window_size, stride)
            fmri_windows = segment_sequence(fmri_seq, window_size, stride)

            for stim, fmri in zip(stim_windows, fmri_windows):
                self.samples.append((stim, fmri))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def prepare_and_save_aligned_data(
    root_data_dir,
    subject_id,
    selected_episodes,
    window_size,
    stride,
    batch_size,
    save_path,
    excluded_trs_start=3,
    excluded_trs_end=3,
    hrf_delay=3,
    shuffle=True
):
    import pickle
    from torch.utils.data import DataLoader

    features_dict = load_stimulus_features(root_data_dir, modality="all", selected_episodes=selected_episodes)
    fmri_dict = load_fmri(root_data_dir, subject_id, friends_episodes=selected_episodes)

    aligned_features, aligned_fmri = trim_and_concatenate_features(
        features_dict,
        fmri_dict,
        excluded_trs_start=excluded_trs_start,
        excluded_trs_end=excluded_trs_end,
        hrf_delay=hrf_delay
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump((aligned_features, aligned_fmri), f)

    dataset = SlidingWindowFMRIDataset(
        full_data=list(zip(aligned_features, aligned_fmri)),
        window_size=window_size,
        stride=stride
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_no_pad)
    return dataloader


def collate_fn_no_pad(batch):
    stim_batch, fmri_batch = zip(*batch)
    stim_batch = torch.stack(stim_batch)
    fmri_batch = torch.stack(fmri_batch)
    return stim_batch, fmri_batch



class MultiSubjectDataset(Dataset):
    def __init__(self, datasets):
        """
        datasets: list of SlidingWindowFMRIDataset from each subject
        """
        self.samples = []
        for dataset in datasets:
            self.samples.extend(dataset.samples)  # flat list of (stim, fmri)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_combined_dataloader(
    root_data_dir,
    subjects,
    window_size,
    stride,
    batch_size,
    cache_dir,
    excluded_trs_start=0,
    excluded_trs_end=0,
    hrf_delay=3,
    selected_episodes=None,
    shuffle=True
):
    datasets = []

    feats_all = load_stimulus_features(root_data_dir, modality="all", selected_episodes=selected_episodes)
    feature_episodes = set(next(iter(feats_all.values())).keys())

    for sid in subjects:
        fmri_dict = load_fmri(root_data_dir, sid)
        fmri_episodes = set(fmri_dict.keys())
        selected = sorted(feature_episodes & fmri_episodes)

        features_dict = load_stimulus_features(
            root_data_dir, modality="all", selected_episodes=selected)
        fmri_dict = load_fmri(
            root_data_dir, sid, friends_episodes=selected)

        aligned_features, aligned_fmri = trim_and_concatenate_features(
            features_dict, fmri_dict,
            excluded_trs_start=excluded_trs_start,
            excluded_trs_end=excluded_trs_end,
            hrf_delay=hrf_delay)

        pkl_path = os.path.join(cache_dir, f"sub{sid}_aligned.pkl")
        os.makedirs(cache_dir, exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump((aligned_features, aligned_fmri), f)

        dataset = SlidingWindowFMRIDataset(
            list(zip(aligned_features, aligned_fmri)),
            window_size=window_size,
            stride=stride
        )
        datasets.append(dataset)
        print(f"[INFO] Subject {sid}: {len(dataset)} samples.")

    # Combine everything
    merged_dataset = MultiSubjectDataset(datasets)
    dataloader = DataLoader(
        merged_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_no_pad
    )
    print(f"[INFO] Total combined samples: {len(merged_dataset)}")
    return dataloader
