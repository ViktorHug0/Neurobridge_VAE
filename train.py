import os
import argparse
import logging
from datetime import datetime
import json
import random
import time
import sys
import shutil

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import pandas as pd

from module.dataset import EEGPreImageDataset
from module.eeg_encoder.atm.atm import ATMS
from module.eeg_encoder.model import EEGNet, EEGProject, TSConv, EEGTransformer
from module.loss import ContrastiveLoss
from module.util import retrieve_all
from module.projector import *
from module.logging import (
    init_component_sums,
    accumulate_components,
    average_components,
    format_loss_breakdown,
    write_component_scalars,
)
from module.eeg_augmentation import RandomTimeShift, RandomGaussianNoise, RandomChannelDropout, RandomSmooth
from iVAE.iVAE_utils import (
    EEGSubjectCondVAE, SubjectClassifier, grad_reverse,
    scvae_loss, WarmupMultiStepLR,
)
from module.subject_signature import get_subject_signatures
from module.plotting import save_loss_component_plots, save_subject_probe_plot


def append_loss_history(history: dict[str, list[float]], comp: dict[str, float]) -> None:
    """Append scalar component values to per-key history lists."""
    for k, v in comp.items():
        history.setdefault(k, []).append(float(v))

# Set the random seed. If not provided, generate a new one based on the current time.
def seed_everything(seed: int = None):
    if seed is None:
        # If no seed is provided, generate a new one based on the current time
        seed = int(time.time()) % (2**32 - 1)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[seed_everything] All seeds set to: {seed}")
    return seed

if __name__ == '__main__':
    # Get input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str, help='training device')
    parser.add_argument('--num_workers', default=8, type=int, help='DataLoader worker processes (set 0 to disable)')
    parser.add_argument('--num_epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--output_dir', default='./result', type=str)
    parser.add_argument('--output_name', default=None, type=str)
    parser.add_argument('--train_subject_ids', default=[8], nargs='+', type=int)
    parser.add_argument('--test_subject_ids', default=[8], nargs='+', type=int)
    parser.add_argument('--data_average', action='store_true')
    parser.add_argument('--data_random', action='store_true')
    parser.add_argument('--init_temperature', default=0.07, type=float)
    parser.add_argument('--t_learnable', action='store_true')
    parser.add_argument('--softplus', action='store_true')
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--img_l2norm', action='store_true')
    parser.add_argument('--text_l2norm', action='store_true')
    parser.add_argument('--eeg_l2norm', action='store_true')
    parser.add_argument('--eeg_data_dir', default='./things_eeg/data/preprocessed_eeg', type=str, help='where your EEG data are')
    parser.add_argument("--selected_channels", default=[], nargs='*', type=str, help="selected EEG channels, empty means all channels")
    parser.add_argument('--time_window', type=int, default=[0, 250], nargs=2, help='time window for EEG data, in sample points')
    parser.add_argument('--eeg_aug', action='store_true')
    parser.add_argument('--eeg_aug_type', type=str, choices=['noise', 'time_shift', 'channel_dropout', 'smooth'], default='noise', help='eeg augmentation type')
    parser.add_argument('--eeg_encoder_type', type=str, choices=['ATM', "EEGNet", "EEGProject", "TSConv", "EEGTransformer"], default='EEGProject')
    parser.add_argument('--image_aug', action='store_true')
    parser.add_argument('--image_test_aug', action='store_true')
    parser.add_argument('--eeg_test_aug', action='store_true')
    parser.add_argument('--frozen_eeg_prior', action='store_true', help='whether to use frozen eeg prior')
    
    parser.add_argument('--projector', type=str, choices=['direct', 'linear', 'mlp'], default='direct')
    parser.add_argument('--feature_dim', type=int, default=512, help='dont work when direct')

    parser.add_argument('--image_feature_dir', default='./data/things_eeg/image_feature/RN50', type=str, help='where your image feature are')
    parser.add_argument('--aug_image_feature_dirs', default=[], nargs='+', type=str, help='where your augmentation image feature are')
    parser.add_argument('--text_feature_dir', default='./data/things_eeg/text_feature/BLIP2', type=str, help='where your text feature are')

    parser.add_argument('--save_weights', action='store_true', help='whether to save model weights')
    
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')
    parser.add_argument('--multi_positive_loss', action='store_true', help='use multi-positive contrastive loss based on (object_idx, image_idx) within batch')
    parser.add_argument(
        '--subject_probe_holdout',
        action='store_true',
        help='use fixed per-subject 90/10 split: 90% for training losses, 10% for detached subject probe validation',
    )

    # ── iVAE arguments ──
    parser.add_argument('--ivae', action='store_true', help='enable iVAE bottleneck between EEG encoder and contrastive loss')
    parser.add_argument('--z_s_dim', type=int, default=16, help='subject-only latent block dimension')
    parser.add_argument('--z_is_dim', type=int, default=16, help='image+subject latent block dimension')
    parser.add_argument('--z_i_dim', type=int, default=128, help='image-only latent block dimension')
    parser.add_argument('--z_n_dim', type=int, default=16, help='noise latent block dimension')
    parser.add_argument('--beta_s', type=float, default=1.0, help='beta weight for subject KL')
    parser.add_argument('--beta_is', type=float, default=1.0, help='beta weight for image+subject KL')
    parser.add_argument('--beta_i', type=float, default=1.0, help='beta weight for image KL')
    parser.add_argument('--beta_n', type=float, default=1.0, help='beta weight for noise KL')
    parser.add_argument('--lambda_recon', type=float, default=1.0, help='weight applied to reconstruction loss term')
    parser.add_argument('--gamma_cl', type=float, default=1.0, help='contrastive loss weight in subject-conditioned VAE total loss')
    parser.add_argument('--lambda_subj_cls', type=float, default=1.0, help='weight of subject CE head on concat(z_s, z_is)')
    parser.add_argument('--lambda_subj_adv', type=float, default=1.0, help='weight of adversarial subject CE head on concat(z_i, z_n)')
    parser.add_argument('--grl_lambda', type=float, default=1.0, help='gradient reversal strength for subject adversary')
    parser.add_argument('--C_max', type=float, default=25.0, help='maximum KL capacity for beta-VAE ramping')
    parser.add_argument('--C_stop_iter', type=int, default=10000, help='number of global steps to ramp capacity from 0 to C_max')
    parser.add_argument('--ivae_hidden_dim', type=int, default=512, help='hidden layer width for subject-conditioned VAE MLPs')
    parser.add_argument('--ivae_n_layers', type=int, default=1, help='number of extra hidden layers in iVAE MLPs')
    parser.add_argument('--n_subjects', type=int, default=11, help='embedding table size (must be > max subject ID)')
    parser.add_argument('--cl_cond_on_subject', dest='cl_cond_on_subject', action='store_true', help='condition contrastive embedding on subject signature u by concatenation')
    parser.add_argument('--no_cl_cond_on_subject', dest='cl_cond_on_subject', action='store_false', help='disable subject-signature concatenation for contrastive embedding')
    parser.set_defaults(cl_cond_on_subject=True)
    parser.add_argument('--image_prior_hidden_dim', type=int, default=128, help='hidden width for image prior MLP')
    parser.add_argument('--image_prior_n_layers', type=int, default=1, help='number of extra hidden layers in image prior MLP')

    # ── Scheduler arguments ──
    parser.add_argument('--scheduler', action='store_true', help='enable WarmupMultiStepLR scheduler')
    parser.add_argument('--milestones', type=int, nargs='+', default=[20, 35], help='epoch milestones for LR decay')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='LR decay factor at milestones')
    parser.add_argument('--warmup_steps', type=int, default=500, help='number of warmup steps (batches)')
    parser.add_argument('--warmup_factor', type=float, default=1.0/3, help='starting LR factor during warmup')
    parser.add_argument('--warmup_method', type=str, choices=['constant', 'linear'], default='linear', help='warmup strategy')

    # ── W&B arguments ──
    parser.add_argument('--wandb', action='store_true', help='enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='Neurobridge_VAE', help='W&B project name')
    parser.add_argument('--wandb_collection', type=str, default='runs', help='W&B registry collection name for artifact linking')

    args = parser.parse_args()
    
    seed = seed_everything(seed=args.seed)
    
    if args.output_name is not None:
        log_dir = os.path.join(args.output_dir, f"{datetime.now().strftime(r'%Y%m%d-%H%M%S')}-{args.output_name}")
    else:
        log_dir = os.path.join(args.output_dir, datetime.now().strftime(r'%Y%m%d-%H%M%S'))

    log_dir_suffix = '-'.join(log_dir.split('-')[2:])
    if os.path.exists(args.output_dir):
        for existing_dir in os.listdir(args.output_dir):
            if existing_dir.endswith(log_dir_suffix):
                if os.path.exists(os.path.join(args.output_dir, existing_dir, "result.csv")):
                    print(f"Experiment with the same name '{log_dir_suffix}' already exists. Exiting to avoid overwriting.")
                    sys.exit(0)
                else:
                    shutil.rmtree(os.path.join(args.output_dir, existing_dir))
                    print(f"Removed incomplete experiment directory '{existing_dir}' to avoid conflicts.")

    writer = SummaryWriter(log_dir=log_dir)

    # Save the configuration to a JSON file
    args_dict = vars(args)
    with open(os.path.join(writer.log_dir, "train_config.json"), 'w') as f:
        json.dump(args_dict, f, indent=4)

    # W&B initialisation (optional)
    wandb_run = None
    if args.wandb:
        try:
            import wandb as _wandb
            run_name = os.path.basename(writer.log_dir)
            wandb_run = _wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=args_dict,
                dir=writer.log_dir,
            )
        except Exception as _e:
            print(f"[WARN] W&B initialisation failed: {_e}")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        encoding='utf-8',
        filename=f'{writer.log_dir}/train.log',
        filemode='w'
    )

    use_terminal_color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    def log(msg: str, console_msg: str | None = None):
        logging.info(msg)
        print(console_msg if console_msg is not None else msg)

    ansi = {
        "reset": "\033[0m",
        "train": "\033[1;36m",
        "test": "\033[1;35m",
        "metric": "\033[1;33m",
        "red": "\033[1;31m",
    }

    def colorize(text: str, color: str) -> str:
        if not use_terminal_color:
            return text
        return f"{color}{text}{ansi['reset']}"

    log('Input arguments:')
    for key, val in vars(args).items():
        log(f'{key:22} {val}')
        
    with open(os.path.join(args.output_dir, "last_run.txt"), 'w') as f:
        f.write(writer.log_dir)
        
    print("")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log(f'Using device: {device}')

    print('\n>>> Loading Train Data <<<')
    if args.eeg_aug:
        if args.eeg_aug_type == 'noise':
            eeg_transform = RandomGaussianNoise(std=0.001)
        elif args.eeg_aug_type == 'time_shift':
            eeg_transform = RandomTimeShift(max_shift=5)
        elif args.eeg_aug_type == 'channel_dropout':
            eeg_transform = RandomChannelDropout(drop_prob=0.1)
        elif args.eeg_aug_type == 'smooth':
            eeg_transform = RandomSmooth(kernel_size=5, smooth_prob=0.3)
    else:
        eeg_transform = None
    
    train_dataset = EEGPreImageDataset(args.train_subject_ids, args.eeg_data_dir, args.selected_channels, args.time_window, args.image_feature_dir, args.text_feature_dir, args.image_aug, args.aug_image_feature_dirs, args.data_average, args.data_random, eeg_transform, True, args.image_test_aug, args.eeg_test_aug, args.frozen_eeg_prior)
    
    eeg_sample_points = train_dataset.num_sample_points
    log(f'EEG sample points: {eeg_sample_points}')
    feature_dim = train_dataset.image_features.shape[-1]
    log(f'feature dimension: {feature_dim}')

    log(f'data length: {len(train_dataset)}')
    channels_num = train_dataset.channels_num
    log(f'number of channels: {channels_num}')

    pin_memory = device.type == "cuda"
    subj_probe_train_dataloader = None
    subj_probe_val_dataloader = None
    train_main_dataset = train_dataset

    if args.subject_probe_holdout:
        if args.data_random:
            raise ValueError("--subject_probe_holdout requires deterministic indexing (disable --data_random).")
        if len(args.train_subject_ids) <= 1:
            raise ValueError("--subject_probe_holdout requires at least two training subjects.")

        num_subjects_train = len(args.train_subject_ids)
        if len(train_dataset) % num_subjects_train != 0:
            raise ValueError(
                "Train dataset length is not divisible by number of train subjects; cannot build a per-subject fixed split."
            )
        per_subject_len = len(train_dataset) // num_subjects_train
        if per_subject_len <= 1:
            raise ValueError("Per-subject train length too small for a 90/10 split.")

        rng = np.random.default_rng(seed + 137)
        main_train_indices: list[int] = []
        subj_cls_indices: list[int] = []
        holdout_per_subject = min(max(1, int(np.floor(per_subject_len * 0.1))), per_subject_len - 1)

        for subject_idx in range(num_subjects_train):
            start = subject_idx * per_subject_len
            end = start + per_subject_len
            perm = rng.permutation(np.arange(start, end, dtype=np.int64))
            subj_cls_indices.extend(perm[:holdout_per_subject].tolist())
            main_train_indices.extend(perm[holdout_per_subject:].tolist())

        train_main_dataset = Subset(train_dataset, main_train_indices)
        subj_cls_val_dataset = Subset(train_dataset, subj_cls_indices)
        log(
            f"Enabled subject probe holdout split: per_subject={per_subject_len}, "
            f"train(90%)={len(main_train_indices)}, probe_val(10%)={len(subj_cls_indices)}"
        )
        subj_probe_train_dataloader = DataLoader(
            train_main_dataset,
            batch_size=256,  # Keep probe batches small to reduce GPU memory pressure.
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=(args.num_workers > 0),
        )
        subj_probe_val_dataloader = DataLoader(
            subj_cls_val_dataset,
            batch_size=256,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=(args.num_workers > 0),
        )

    dataloader = DataLoader(
        train_main_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    
    print('\n>>> Loading Test Data <<<')
    test_dataset = EEGPreImageDataset(args.test_subject_ids, args.eeg_data_dir, args.selected_channels, args.time_window, args.image_feature_dir, args.text_feature_dir, args.image_aug, args.aug_image_feature_dirs, True, False, eeg_transform, False, args.image_test_aug, args.eeg_test_aug, args.frozen_eeg_prior)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    
    args_dict = vars(args)
    keys_needed = ['eeg_encoder_type', 'eeg_data_dir', 'image_feature_dir']
    inference_config = {k: args_dict[k] for k in keys_needed}
    inference_config['eeg_sample_points'] = eeg_sample_points
    inference_config['feature_dim'] = feature_dim
    with open(os.path.join(writer.log_dir, "evaluate_config.json"), 'w') as f:
        json.dump(inference_config, f, indent=4)
    
    if args.eeg_encoder_type == 'ATM':
        model = ATMS(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    elif args.eeg_encoder_type == 'EEGNet':
        model = EEGNet(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    elif args.eeg_encoder_type == 'EEGProject':
        model = EEGProject(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    elif args.eeg_encoder_type == 'TSConv':
        model = TSConv(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    elif args.eeg_encoder_type == 'EEGTransformer':
        model = EEGTransformer(feature_dim=feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    model = model.to(device)

    # Output the number of trainable parameters in the model (formatted in millions)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_m = num_params / 1e6
    log(f'EEG Encoder trainable parameters: {num_params_m:.2f}M')
    
    log(str(model))

    # ── Subject-conditioned VAE bottleneck (optional via --ivae) ──
    ivae_model = None
    subj_cls = None
    subj_adv = None
    subj_probe_heads: dict[str, nn.Module] = {}
    subj_probe_optimizers: dict[str, optim.Optimizer] = {}
    subj_probe_history: dict[str, dict[str, list[float]]] = {
        "z_s": {"train_acc": [], "train_loss": [], "val_acc": []},
        "z_is": {"train_acc": [], "train_loss": [], "val_acc": []},
        "z_i": {"train_acc": [], "train_loss": [], "val_acc": []},
        "z_n": {"train_acc": [], "train_loss": [], "val_acc": []},
    }
    if args.ivae:
        ivae_model = EEGSubjectCondVAE(
            feature_dim=feature_dim,
            z_s_dim=args.z_s_dim,
            z_is_dim=args.z_is_dim,
            z_i_dim=args.z_i_dim,
            z_n_dim=args.z_n_dim,
            u_dim=5,
            hidden_dim=args.ivae_hidden_dim,
            n_layers=args.ivae_n_layers,
            img_dim=feature_dim,
            image_prior_hidden_dim=args.image_prior_hidden_dim,
            image_prior_n_layers=args.image_prior_n_layers,
        ).to(device)
        num_subject_classes = args.n_subjects - 1  # subject IDs are 1..10 in Things-EEG runs
        subj_cls = SubjectClassifier(args.z_s_dim + args.z_is_dim, num_subject_classes).to(device)
        subj_adv = SubjectClassifier(args.z_i_dim + args.z_n_dim, num_subject_classes).to(device)

        ivae_params = sum(p.numel() for p in ivae_model.parameters() if p.requires_grad)
        zs_head_params = sum(p.numel() for p in subj_cls.parameters() if p.requires_grad)
        zi_head_params = sum(p.numel() for p in subj_adv.parameters() if p.requires_grad)
        log(f'Subject-conditioned VAE trainable parameters: {ivae_params / 1e6:.2f}M')
        log(f'Subject head(z_s,z_is) trainable parameters: {zs_head_params / 1e6:.2f}M')
        log(f'Subject adversary(z_i,z_n) trainable parameters: {zi_head_params / 1e6:.2f}M')
        log(str(ivae_model))

        if args.subject_probe_holdout:
            # Linear probes trained on detached per-block samples; never backpropagates into iVAE.
            subj_probe_heads = {
                "z_s": nn.Linear(args.z_s_dim, num_subject_classes).to(device),
                "z_is": nn.Linear(args.z_is_dim, num_subject_classes).to(device),
                "z_i": nn.Linear(args.z_i_dim, num_subject_classes).to(device),
                "z_n": nn.Linear(args.z_n_dim, num_subject_classes).to(device),
            }
            subj_probe_optimizers = {
                latent: optim.Adam(head.parameters(), lr=1e-3)
                for latent, head in subj_probe_heads.items()
            }

    # Determine eeg projector input dim (depends on iVAE mode)
    if args.ivae:
        eeg_proj_input_dim = args.z_is_dim + args.z_i_dim + (5 if args.cl_cond_on_subject else 0)
    else:
        eeg_proj_input_dim = feature_dim

    if args.projector == 'direct':
        eeg_projector = ProjectorDirect().to(device)
        img_projector = ProjectorDirect().to(device)
        text_projector = ProjectorDirect().to(device)
    elif args.projector == 'linear':
        eeg_projector = ProjectorLinear(eeg_proj_input_dim, args.feature_dim).to(device)
        img_projector = ProjectorLinear(feature_dim, args.feature_dim).to(device)
        text_projector = ProjectorLinear(feature_dim, args.feature_dim).to(device)
    elif args.projector == 'mlp':
        eeg_projector = ProjectorMLP(eeg_proj_input_dim, args.feature_dim).to(device)
        img_projector = ProjectorMLP(feature_dim, args.feature_dim).to(device)
        text_projector = ProjectorMLP(feature_dim, args.feature_dim).to(device)
        
    num_params = sum(p.numel() for p in eeg_projector.parameters() if p.requires_grad) + sum(p.numel() for p in img_projector.parameters() if p.requires_grad) + sum(p.numel() for p in text_projector.parameters() if p.requires_grad)
    num_params_m = num_params / 1e6
    log(f'Projector trainable parameters: {num_params_m:.2f}M')
    
    criterion = ContrastiveLoss(
        args.init_temperature, args.alpha, args.beta,
        args.eeg_l2norm, args.img_l2norm, args.text_l2norm,
        args.t_learnable, args.softplus,
        multi_positive_loss=args.multi_positive_loss,
    ).to(device)
    log(str(criterion))
    
    # Collect all trainable parameters from both model and criterion
    trainable_parameters = list(model.parameters()) + list(eeg_projector.parameters()) + list(img_projector.parameters()) + list(text_projector.parameters())
    if args.ivae and ivae_model is not None:
        trainable_parameters.extend(list(ivae_model.parameters()))
        trainable_parameters.extend(list(subj_cls.parameters()))
        trainable_parameters.extend(list(subj_adv.parameters()))
    if args.t_learnable:  # Only add criterion parameters if temperature is learnable
        trainable_parameters.extend([p for p in criterion.parameters() if p.requires_grad])
    
    optimizer = optim.AdamW(trainable_parameters, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)

    # ── Learning rate scheduler (optional) ──
    scheduler = None
    if args.scheduler:
        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=args.milestones,
            gamma=args.scheduler_gamma,
            warmup_factor=args.warmup_factor,
            warmup_steps=args.warmup_steps,
            warmup_method=args.warmup_method,
        )
        log(f'Scheduler enabled: milestones={args.milestones}, gamma={args.scheduler_gamma}, '
            f'warmup_steps={args.warmup_steps}, warmup_method={args.warmup_method}')

    # Training process
    model.train()
    eeg_projector.train()
    img_projector.train()
    text_projector.train()
    if ivae_model is not None:
        ivae_model.train()
        subj_cls.train()
        subj_adv.train()
    best_top1_acc = 0.0
    best_top5_acc = 0.0
    best_test_loss = float('inf')
    best_test_cl = float('inf')
    best_test_epoch = 0
    global_step = 0
    loss_epochs: list[int] = []
    train_loss_history: dict[str, list[float]] = {}
    test_loss_history: dict[str, list[float]] = {}

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        if ivae_model is not None:
            ivae_model.train()
            subj_cls.train()
            subj_adv.train()
        total_loss = 0.0
        train_comp_sums = init_component_sums()
        train_comp_count = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{args.num_epochs} [Train]"):
            eeg_batch = batch[0].to(device)
            subject_id_batch = batch[3].to(device).long()
            image_feature_batch = batch[1].to(device)
            text_feature_batch = batch[2].to(device)
            object_idx_batch = batch[4].to(device)
            image_idx_batch = batch[5].to(device)

            optimizer.zero_grad()
            if args.eeg_encoder_type in ['ATM']:
                eeg_feature_batch = model(eeg_batch, subject_id_batch)
            else:
                eeg_feature_batch = model(eeg_batch)

            # ── iVAE path ──
            if args.ivae and ivae_model is not None:
                u_batch = get_subject_signatures(subject_id_batch, device, dtype=eeg_feature_batch.dtype)
                ivae_out = ivae_model(
                    eeg_feature_batch,
                    u_batch,
                    img_feat=image_feature_batch,
                    global_step=global_step,
                    C_max=args.C_max,
                    C_stop_iter=args.C_stop_iter,
                )

                # Contrastive latents: image-conditioned blocks (z_is, z_i), optional subject signature.
                z_for_cl = torch.cat([ivae_out['z_is'], ivae_out['z_i']], dim=-1)
                if args.cl_cond_on_subject:
                    z_for_cl = torch.cat([z_for_cl, u_batch], dim=-1)

                # Project latents and image features for contrastive loss
                eeg_proj = eeg_projector(z_for_cl)
                img_proj = img_projector(image_feature_batch)
                text_proj = text_projector(text_feature_batch)

                if args.multi_positive_loss:
                    group_ids = object_idx_batch.long() * 100 + image_idx_batch.long()
                    cl_loss = criterion(eeg_proj, img_proj, text_proj, group_ids=group_ids)
                else:
                    cl_loss = criterion(eeg_proj, img_proj, text_proj)

                # Subject heads train on the main training split only; the 10% split is reserved for probe validation.
                subj_labels = (subject_id_batch - 1).clamp(min=0, max=args.n_subjects - 2)
                subj_logits_cls = subj_cls(torch.cat([ivae_out['z_s'], ivae_out['z_is']], dim=-1))
                subj_logits_adv = subj_adv(
                    grad_reverse(torch.cat([ivae_out['z_i'], ivae_out['z_n']], dim=-1), lambda_grl=args.grl_lambda)
                )

                loss, loss_components = scvae_loss(
                    ivae_out,
                    eeg_feature_batch.detach(),
                    beta_s=args.beta_s,
                    beta_is=args.beta_is,
                    beta_i=args.beta_i,
                    beta_n=args.beta_n,
                    lambda_recon=args.lambda_recon,
                    lambda_cl=args.gamma_cl,
                    contrastive_loss_val=cl_loss,
                    subj_logits_cls=subj_logits_cls,
                    subj_logits_adv=subj_logits_adv,
                    subj_labels=subj_labels,
                    lambda_subj_cls=args.lambda_subj_cls,
                    lambda_subj_adv=args.lambda_subj_adv,
                )

                global_step += 1

            # ── Standard (non-iVAE) path ──
            else:
                eeg_feature_batch = eeg_projector(eeg_feature_batch)
                image_feature_batch = img_projector(image_feature_batch)
                text_feature_batch = text_projector(text_feature_batch)

                if args.multi_positive_loss:
                    group_ids = object_idx_batch.long() * 100 + image_idx_batch.long()
                    loss = criterion(eeg_feature_batch, image_feature_batch, text_feature_batch, group_ids=group_ids)
                else:
                    loss = criterion(eeg_feature_batch, image_feature_batch, text_feature_batch)
                loss_components = {'total': loss.detach()}

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            accumulate_components(train_comp_sums, loss_components)
            train_comp_count += 1

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        avg_train_comp = average_components(train_comp_sums, train_comp_count)
        write_component_scalars(writer, 'train', avg_train_comp, epoch)
        append_loss_history(train_loss_history, avg_train_comp)
        if wandb_run is not None:
            wandb_run.log(
                {"epoch": epoch, **{f"train/{k}": v for k, v in avg_train_comp.items()}},
                step=epoch,
            )
        train_epoch_line = f"Epoch [{epoch}/{args.num_epochs}] Train Loss: {avg_loss:.4f}"
        log(train_epoch_line, colorize(train_epoch_line, ansi["train"]))
        train_breakdown_plain = format_loss_breakdown("[Train]", avg_train_comp, args.ivae and ivae_model is not None)
        train_breakdown_color = format_loss_breakdown(
            "[Train]",
            avg_train_comp,
            args.ivae and ivae_model is not None,
            use_color=use_terminal_color,
        )
        log(train_breakdown_plain, train_breakdown_color)

        # Probe mode: train a detached linear classifier on sampled z_s and report holdout accuracy.
        if (
            args.subject_probe_holdout
            and args.ivae
            and ivae_model is not None
            and len(subj_probe_heads) > 0
            and len(subj_probe_optimizers) > 0
            and subj_probe_train_dataloader is not None
            and subj_probe_val_dataloader is not None
        ):
            model.eval()
            ivae_model.eval()
            for probe_head in subj_probe_heads.values():
                probe_head.train()
            probe_train_acc_sum = {k: 0.0 for k in subj_probe_heads}
            probe_train_loss_sum = {k: 0.0 for k in subj_probe_heads}
            probe_train_batches = 0
            for probe_batch in subj_probe_train_dataloader:
                eeg_probe = probe_batch[0].to(device)
                subject_id_probe = probe_batch[3].to(device).long()
                image_probe = probe_batch[1].to(device)
                with torch.no_grad():
                    if args.eeg_encoder_type in ['ATM']:
                        eeg_probe_feature = model(eeg_probe, subject_id_probe)
                    else:
                        eeg_probe_feature = model(eeg_probe)
                    u_probe = get_subject_signatures(subject_id_probe, device, dtype=eeg_probe_feature.dtype)
                    ivae_probe_out = ivae_model(
                        eeg_probe_feature,
                        u_probe,
                        img_feat=image_probe,
                        global_step=global_step,
                        C_max=args.C_max,
                        C_stop_iter=args.C_stop_iter,
                    )
                    q_s_mu_probe, q_s_lv_probe = ivae_probe_out['posterior_params']['s']
                    q_is_mu_probe, q_is_lv_probe = ivae_probe_out['posterior_params']['is']
                    q_i_mu_probe, q_i_lv_probe = ivae_probe_out['posterior_params']['i']
                    q_n_mu_probe, q_n_lv_probe = ivae_probe_out['posterior_params']['n']
                    probe_latents = {
                        "z_s": q_s_mu_probe + torch.randn_like(q_s_mu_probe) * torch.exp(0.5 * q_s_lv_probe),
                        "z_is": q_is_mu_probe + torch.randn_like(q_is_mu_probe) * torch.exp(0.5 * q_is_lv_probe),
                        "z_i": q_i_mu_probe + torch.randn_like(q_i_mu_probe) * torch.exp(0.5 * q_i_lv_probe),
                        "z_n": q_n_mu_probe + torch.randn_like(q_n_mu_probe) * torch.exp(0.5 * q_n_lv_probe),
                    }
                probe_labels = (subject_id_probe - 1).clamp(min=0, max=args.n_subjects - 2)
                for latent_name, probe_head in subj_probe_heads.items():
                    probe_optimizer = subj_probe_optimizers[latent_name]
                    probe_optimizer.zero_grad()
                    probe_logits = probe_head(probe_latents[latent_name].detach())
                    probe_loss = F.cross_entropy(probe_logits, probe_labels)
                    probe_loss.backward()
                    probe_optimizer.step()
                    probe_pred = torch.argmax(probe_logits, dim=1)
                    probe_train_acc_sum[latent_name] += float((probe_pred == probe_labels).float().mean().item())
                    probe_train_loss_sum[latent_name] += float(probe_loss.item())
                probe_train_batches += 1

            for probe_head in subj_probe_heads.values():
                probe_head.eval()
            probe_val_acc_sum = {k: 0.0 for k in subj_probe_heads}
            probe_val_batches = 0
            with torch.no_grad():
                for probe_val_batch in subj_probe_val_dataloader:
                    eeg_probe_val = probe_val_batch[0].to(device)
                    subject_id_probe_val = probe_val_batch[3].to(device).long()
                    image_probe_val = probe_val_batch[1].to(device)

                    if args.eeg_encoder_type in ['ATM']:
                        eeg_probe_val_feature = model(eeg_probe_val, subject_id_probe_val)
                    else:
                        eeg_probe_val_feature = model(eeg_probe_val)
                    u_probe_val = get_subject_signatures(subject_id_probe_val, device, dtype=eeg_probe_val_feature.dtype)
                    ivae_probe_val_out = ivae_model(
                        eeg_probe_val_feature,
                        u_probe_val,
                        img_feat=image_probe_val,
                        global_step=global_step,
                        C_max=args.C_max,
                        C_stop_iter=args.C_stop_iter,
                    )
                    q_s_mu_probe_val, q_s_lv_probe_val = ivae_probe_val_out['posterior_params']['s']
                    q_is_mu_probe_val, q_is_lv_probe_val = ivae_probe_val_out['posterior_params']['is']
                    q_i_mu_probe_val, q_i_lv_probe_val = ivae_probe_val_out['posterior_params']['i']
                    q_n_mu_probe_val, q_n_lv_probe_val = ivae_probe_val_out['posterior_params']['n']
                    probe_latents_val = {
                        "z_s": q_s_mu_probe_val + torch.randn_like(q_s_mu_probe_val) * torch.exp(0.5 * q_s_lv_probe_val),
                        "z_is": q_is_mu_probe_val + torch.randn_like(q_is_mu_probe_val) * torch.exp(0.5 * q_is_lv_probe_val),
                        "z_i": q_i_mu_probe_val + torch.randn_like(q_i_mu_probe_val) * torch.exp(0.5 * q_i_lv_probe_val),
                        "z_n": q_n_mu_probe_val + torch.randn_like(q_n_mu_probe_val) * torch.exp(0.5 * q_n_lv_probe_val),
                    }
                    probe_val_labels = (subject_id_probe_val - 1).clamp(min=0, max=args.n_subjects - 2)
                    for latent_name, probe_head in subj_probe_heads.items():
                        probe_val_logits = probe_head(probe_latents_val[latent_name])
                        probe_val_pred = torch.argmax(probe_val_logits, dim=1)
                        probe_val_acc_sum[latent_name] += float((probe_val_pred == probe_val_labels).float().mean().item())
                    probe_val_batches += 1

            if probe_train_batches > 0 and probe_val_batches > 0:
                probe_line_parts = []
                probe_wandb_dict = {"epoch": epoch}
                for latent_name in ("z_s", "z_is", "z_i", "z_n"):
                    probe_train_acc = probe_train_acc_sum[latent_name] / probe_train_batches
                    probe_train_loss = probe_train_loss_sum[latent_name] / probe_train_batches
                    probe_val_acc = probe_val_acc_sum[latent_name] / probe_val_batches
                    subj_probe_history[latent_name]["train_acc"].append(float(probe_train_acc))
                    subj_probe_history[latent_name]["train_loss"].append(float(probe_train_loss))
                    subj_probe_history[latent_name]["val_acc"].append(float(probe_val_acc))
                    writer.add_scalar(f'SubjectProbe/{latent_name}_train_acc', probe_train_acc, epoch)
                    writer.add_scalar(f'SubjectProbe/{latent_name}_train_loss', probe_train_loss, epoch)
                    writer.add_scalar(f'SubjectProbe/{latent_name}_val_acc', probe_val_acc, epoch)
                    probe_wandb_dict[f"subject_probe/{latent_name}_train_acc"] = probe_train_acc
                    probe_wandb_dict[f"subject_probe/{latent_name}_val_acc"] = probe_val_acc
                    probe_wandb_dict[f"subject_probe/{latent_name}_train_loss"] = probe_train_loss
                    probe_line_parts.append(
                        f"{latent_name}:TrainAcc={probe_train_acc:.3f} ValAcc={probe_val_acc:.3f} CE={probe_train_loss:.3f}"
                    )
                probe_line_plain = "SubjProbe " + " | ".join(probe_line_parts)
                if use_terminal_color:
                    probe_line_color = colorize(probe_line_plain, ansi["test"])
                else:
                    probe_line_color = probe_line_plain
                log(probe_line_plain, probe_line_color)
                if wandb_run is not None:
                    wandb_run.log(probe_wandb_dict, step=epoch)

        if args.save_weights:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'eeg_projector_state_dict': eeg_projector.state_dict(),
                'img_projector_state_dict': img_projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            if ivae_model is not None:
                ckpt['ivae_model_state_dict'] = ivae_model.state_dict()
                ckpt['subj_cls_state_dict'] = subj_cls.state_dict()
                ckpt['subj_adv_state_dict'] = subj_adv.state_dict()
            torch.save(ckpt, f"{writer.log_dir}/checkpoint_last.pth")

        model.eval()
        eeg_projector.eval()
        img_projector.eval()
        text_projector.eval()
        if ivae_model is not None:
            ivae_model.eval()
            subj_cls.eval()
            subj_adv.eval()
        total_test_loss = 0.0
        eeg_feature_list = []
        image_feature_list = []
        test_comp_sums = init_component_sums()
        test_comp_count = 0

        with torch.no_grad():
            for batch in test_dataloader:
                eeg_batch = batch[0].to(device)
                subject_id_batch = batch[3].to(device).long()
                image_feature_batch = batch[1].to(device)
                text_feature_batch = batch[2].to(device)
                object_idx_batch = batch[4].to(device)
                image_idx_batch = batch[5].to(device)

                if args.eeg_encoder_type in ['ATM']:
                    eeg_feature_batch = model(eeg_batch, subject_id_batch)
                else:
                    eeg_feature_batch = model(eeg_batch)

                # ── iVAE eval path ──
                if args.ivae and ivae_model is not None:
                    u_batch = get_subject_signatures(subject_id_batch, device, dtype=eeg_feature_batch.dtype)
                    ivae_out = ivae_model(
                        eeg_feature_batch,
                        u_batch,
                        img_feat=image_feature_batch,
                        global_step=global_step,
                        C_max=args.C_max,
                        C_stop_iter=args.C_stop_iter,
                    )

                    z_for_cl = torch.cat([ivae_out['z_is'], ivae_out['z_i']], dim=-1)
                    if args.cl_cond_on_subject:
                        z_for_cl = torch.cat([z_for_cl, u_batch], dim=-1)

                    eeg_proj = eeg_projector(z_for_cl)
                    img_proj = img_projector(image_feature_batch)
                    text_proj = text_projector(text_feature_batch)

                    if args.multi_positive_loss:
                        group_ids = object_idx_batch.long() * 100 + image_idx_batch.long()
                        cl_loss = criterion(eeg_proj, img_proj, text_proj, group_ids=group_ids)
                    else:
                        cl_loss = criterion(eeg_proj, img_proj, text_proj)

                    # Evaluate subject heads only when this batch contains training subjects.
                    # In inter-subject evaluation, test batches are unseen held-out subjects.
                    if all(int(sid) in set(args.train_subject_ids) for sid in subject_id_batch.detach().cpu().tolist()):
                        subj_labels = (subject_id_batch - 1).clamp(min=0, max=args.n_subjects - 2)
                        subj_logits_cls = subj_cls(torch.cat([ivae_out['z_s'], ivae_out['z_is']], dim=-1))
                        subj_logits_adv = subj_adv(
                            grad_reverse(torch.cat([ivae_out['z_i'], ivae_out['z_n']], dim=-1), lambda_grl=args.grl_lambda)
                        )
                    else:
                        subj_labels = None
                        subj_logits_cls = None
                        subj_logits_adv = None

                    loss, test_loss_components = scvae_loss(
                        ivae_out, eeg_feature_batch,
                        beta_s=args.beta_s,
                        beta_is=args.beta_is,
                        beta_i=args.beta_i,
                        beta_n=args.beta_n,
                        lambda_recon=args.lambda_recon,
                        lambda_cl=args.gamma_cl,
                        contrastive_loss_val=cl_loss,
                        subj_logits_cls=subj_logits_cls,
                        subj_logits_adv=subj_logits_adv,
                        subj_labels=subj_labels,
                        lambda_subj_cls=args.lambda_subj_cls,
                        lambda_subj_adv=args.lambda_subj_adv,
                    )

                    eeg_feature_list.append(eeg_proj.cpu().numpy())
                    image_feature_list.append(img_proj.cpu().numpy())

                # ── Standard eval path ──
                else:
                    eeg_feature_batch = eeg_projector(eeg_feature_batch)
                    image_feature_batch = img_projector(image_feature_batch)
                    text_feature_batch = text_projector(text_feature_batch)

                    if args.multi_positive_loss:
                        group_ids = object_idx_batch.long() * 100 + image_idx_batch.long()
                        loss = criterion(eeg_feature_batch, image_feature_batch, text_feature_batch, group_ids=group_ids)
                    else:
                        loss = criterion(eeg_feature_batch, image_feature_batch, text_feature_batch)
                    test_loss_components = {'total': loss.detach()}

                    eeg_feature_list.append(eeg_feature_batch.cpu().numpy())
                    image_feature_list.append(image_feature_batch.cpu().numpy())

                total_test_loss += loss.item()
                accumulate_components(test_comp_sums, test_loss_components)
                test_comp_count += 1

        avg_test_loss = total_test_loss / len(test_dataloader)
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        avg_test_comp = average_components(test_comp_sums, test_comp_count)
        write_component_scalars(writer, 'test', avg_test_comp, epoch)
        append_loss_history(test_loss_history, avg_test_comp)
        loss_epochs.append(epoch)

        # Concatenate all EEG and image features for retrieval
        eeg_feature_all = np.concatenate(eeg_feature_list, axis=0)
        image_feature_all = np.concatenate(image_feature_list, axis=0)
        top5_count, top1_count, total = retrieve_all(eeg_feature_all, image_feature_all, args.data_average)
        top5_acc = top5_count / total * 100
        top1_acc = top1_count / total * 100
        # Track retrieval quality alongside loss components for plotting.
        test_loss_history.setdefault("top1_acc", []).append(float(top1_acc))
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    **{f"test/{k}": v for k, v in avg_test_comp.items()},
                    "test/top1_acc": top1_acc,
                    "test/top5_acc": top5_acc,
                },
                step=epoch,
            )
        test_line_plain = f"top5 acc {top5_acc:.2f}%\ttop1 acc {top1_acc:.2f}%\tTest Loss: {avg_test_loss:.4f}"
        if use_terminal_color:
            test_line_color = (
                f"{colorize('top5 acc', ansi['test'])} {colorize(f'{top5_acc:.2f}%', ansi['red'])}\t"
                f"{colorize('top1 acc', ansi['test'])} {colorize(f'{top1_acc:.2f}%', ansi['red'])}\t"
                f"{colorize('Test Loss:', ansi['test'])} {colorize(f'{avg_test_loss:.4f}', ansi['metric'])}"
            )
        else:
            test_line_color = test_line_plain
        log(test_line_plain, test_line_color)
        test_breakdown_plain = format_loss_breakdown("[Test]", avg_test_comp, args.ivae and ivae_model is not None)
        test_breakdown_color = format_loss_breakdown(
            "[Test]",
            avg_test_comp,
            args.ivae and ivae_model is not None,
            use_color=use_terminal_color,
        )
        log(test_breakdown_plain, test_breakdown_color)

        # Save the best model
        curr_test_cl = avg_test_comp.get('contrastive', avg_test_loss)
        is_better = False
        if curr_test_cl < best_test_cl:
            is_better = True
        elif curr_test_cl == best_test_cl and top1_acc > best_top1_acc:
            # Use top1 accuracy as a secondary criterion when contrastive loss is the same.
            is_better = True

        if is_better:
            best_test_cl = curr_test_cl
            best_test_loss = avg_test_loss
            best_top5_acc = top5_acc
            best_top1_acc = top1_acc
            best_test_epoch = epoch
            if args.save_weights:
                best_ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'eeg_projector_state_dict': eeg_projector.state_dict(),
                    'img_projector_state_dict': img_projector.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_test_loss,
                }
                if ivae_model is not None:
                    best_ckpt['ivae_model_state_dict'] = ivae_model.state_dict()
                    best_ckpt['subj_cls_state_dict'] = subj_cls.state_dict()
                    best_ckpt['subj_adv_state_dict'] = subj_adv.state_dict()
                torch.save(best_ckpt, f"{writer.log_dir}/checkpoint_test_best.pth")

    loss_plot_paths: dict[str, str] = {}
    subject_probe_plot_path = ""
    try:
        loss_plot_paths = save_loss_component_plots(
            train_history=train_loss_history,
            test_history=test_loss_history,
            epochs=loss_epochs,
            out_dir=writer.log_dir,
        )
        log(f"Saved loss plot (raw): {loss_plot_paths.get('raw', '')}")
        log(f"Saved loss plot (scaled): {loss_plot_paths.get('scaled', '')}")
        log(f"Saved loss plot (raw, log-y): {loss_plot_paths.get('raw_log', '')}")
        log(f"Saved loss plot (scaled, log-y): {loss_plot_paths.get('scaled_log', '')}")
        for key in ("raw_html", "scaled_html", "raw_log_html", "scaled_log_html"):
            if loss_plot_paths.get(key):
                log(f"Saved interactive loss plot ({key}): {loss_plot_paths[key]}")
    except Exception as e:
        log(f"[WARN] Failed to save loss component plots: {e}")

    if args.subject_probe_holdout:
        try:
            subject_probe_plot_path = save_subject_probe_plot(
                probe_history=subj_probe_history,
                epochs=loss_epochs,
                out_dir=writer.log_dir,
            )
            log(f"Saved subject probe plot: {subject_probe_plot_path}")
        except Exception as e:
            log(f"[WARN] Failed to save subject probe plot: {e}")

    result_dict = {}
    result_dict['top1 acc'] = f'{top1_acc:.2f}'
    result_dict['top5 acc'] = f'{top5_acc:.2f}'
    result_dict['best top1 acc'] = f'{best_top1_acc:.2f}'
    result_dict['best top5 acc'] = f'{best_top5_acc:.2f}'
    result_dict['best test contrastive loss'] = f'{best_test_cl:.4f}'
    result_dict['best test loss'] = f'{best_test_loss:.4f}'
    result_dict['best epoch'] = best_test_epoch
    df = pd.DataFrame(result_dict, index=[0])
    df.to_csv(os.path.join(log_dir, 'result.csv'), index=False)

    log(
        f'best test contrastive loss: {best_test_cl:.4f} '
        f'(total test loss at selected epoch: {best_test_loss:.4f}) '
        f'top5 acc: {best_top5_acc:.2f} top1 acc: {best_top1_acc:.2f} '
        f'at epoch {best_test_epoch}'
    )

    # W&B artifact: upload plots + config, then link to registry
    if wandb_run is not None:
        try:
            import wandb as _wandb
            run_name = os.path.basename(writer.log_dir)
            artifact = _wandb.Artifact(
                name=run_name,
                type="run-artifacts",
                description=f"Loss plots and config for run {run_name}",
                metadata={
                    "best_top1_acc": best_top1_acc,
                    "best_top5_acc": best_top5_acc,
                    "best_test_loss": best_test_loss,
                    "best_epoch": best_test_epoch,
                },
            )
            config_path = os.path.join(writer.log_dir, "train_config.json")
            if os.path.exists(config_path):
                artifact.add_file(config_path, name="train_config.json")
            result_csv_path = os.path.join(log_dir, "result.csv")
            if os.path.exists(result_csv_path):
                artifact.add_file(result_csv_path, name="result.csv")
            for key, path in loss_plot_paths.items():
                if path and os.path.exists(path):
                    artifact.add_file(path, name=os.path.basename(path))
            if subject_probe_plot_path and os.path.exists(subject_probe_plot_path):
                artifact.add_file(subject_probe_plot_path, name=os.path.basename(subject_probe_plot_path))
            ckpt_path = os.path.join(writer.log_dir, "checkpoint_test_best.pth")
            if os.path.exists(ckpt_path):
                artifact.add_file(ckpt_path, name="checkpoint_test_best.pth")
            wandb_run.log_artifact(artifact)
            artifact.wait()
            wandb_run.link_artifact(artifact, f"wandb-registry-Neurobridge_VAE/{args.wandb_collection}")
            log(f"W&B artifact '{run_name}' linked to registry collection '{args.wandb_collection}'")
            wandb_run.finish()
        except Exception as _e:
            log(f"[WARN] W&B artifact upload/linking failed: {_e}")