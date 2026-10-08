# -*- coding: utf-8 -*-
"""Train the latent encoder, LPD, and DiT with a frozen detector (LR-V2X stage 2)."""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset


def train_parser():
    parser = argparse.ArgumentParser(description="DiffV2X Stage 2: Diffusion Training")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='YAML configuration file')
    parser.add_argument('--stage1_model', type=str, required=True,
                        help='Path to trained stage 1 model')
    parser.add_argument('--model_dir', default='',
                        help='Directory to save models')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    return parser.parse_args()


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    # Initialize distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        gpu = 0

    torch.cuda.set_device(gpu)
    distributed = world_size > 1

    if distributed:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

    # Setup logging early (only rank 0)
    if rank == 0:
        if not opt.model_dir:
            model_dir = train_utils.setup_train(hypes)
        else:
            model_dir = opt.model_dir
            os.makedirs(model_dir, exist_ok=True)

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(model_dir, f'train_{timestamp}.log')
        log_file = open(log_filename, 'w', buffering=1)  # Line buffered

        # Redirect stdout to both console and file
        class TeeOutput:
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()

        sys.stdout = TeeOutput(sys.stdout, log_file)
        print(f"[Logging] Training log will be saved to: {log_filename}")
        print()

        print("="*80)
        print("DiffV2X Stage 2: Diffusion-Only Training")
        print("="*80)
        print(f"Stage 1 model: {opt.stage1_model}")
        print(f"Distributed: {distributed}, World size: {world_size}")
        print("="*80)

    # Build datasets
    if rank == 0:
        print('\nDataset Building')
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    val_dataset = build_dataset(hypes, visualize=False, train=False)

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None
        

    train_loader = DataLoader(
        train_dataset,
        batch_size=hypes['train_params']['batch_size'],
        num_workers=8,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=train_dataset.collate_batch,  # Use collate_batch (auto-switch train/test)
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=val_dataset.collate_batch,  # Use collate_batch (auto-switch train/test)
        pin_memory=True
    )

    # Create model with diffusion
    if rank == 0:
        print('\nCreating Model with Diffusion...')

    # Add stage1_model_dir to hypes for prior path detection
    stage1_model_dir = os.path.dirname(opt.stage1_model)
    if 'model' in hypes and 'args' in hypes['model']:
        hypes['model']['args']['stage1_model_dir'] = stage1_model_dir

    model = train_utils.create_model(hypes)

    # Load pretrained stage 1 model
    if rank == 0:
        print(f'Loading stage 1 model from {opt.stage1_model}')
    checkpoint = torch.load(opt.stage1_model, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    if rank == 0:
        print('Stage 1 model loaded successfully!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model.to(device)

    # Freeze all parameters except diffusion
    if rank == 0:
        print('\n' + '='*80)
        print('Freezing parameters...')

    # Set entire model to eval and freeze all parameters
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Unfreeze diffusion module and latent encoder
    if hasattr(model, 'diffusion_model'):
        model.diffusion_model.train()
        for p in model.diffusion_model.parameters():
            p.requires_grad_(True)

    if hasattr(model, 'latent_encoder'):
        model.latent_encoder.train()
        for p in model.latent_encoder.parameters():
            p.requires_grad_(True)

    # Unfreeze prior decoder (if using latent prior)
    if hasattr(model, 'prior_decoder') and model.prior_decoder is not None:
        model.prior_decoder.train()
        for p in model.prior_decoder.parameters():
            p.requires_grad_(True)
        if rank == 0:
            print('[Stage2] Prior decoder unfrozen for training')

    # Print trainable parameters
    if rank == 0:
        print('\n----------- Trainable Parameters -----------')
        total_trainable = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.data.shape}")
                total_trainable += param.numel()
        print(f'Total trainable parameters: {total_trainable:,}')
        print('-------------------------------------------\n')

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=True
        )

    # Optimizer (only for diffusion parameters)
    diffusion_params = [p for p in model.parameters() if p.requires_grad]

    # Get optimizer class from YAML config
    optimizer_method = getattr(torch.optim, hypes['optimizer']['core_method'], None)
    if not optimizer_method:
        raise ValueError(f"{hypes['optimizer']['core_method']} is not supported")

    optimizer = optimizer_method(
        diffusion_params,
        lr=hypes['optimizer']['lr'],
        **hypes['optimizer']['args']
    )
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # NOTE: Unlike the previous version, we now ALWAYS use reconstructed features
    # This ensures diffusion learns to reconstruct features suitable for detection
    # (Similar to how codebook always uses quantized features)
    if rank == 0:
        print('\n' + '='*80)
        print('Stage 2: Diffusion training with full reconstruction pipeline')
        print('(Detection heads frozen, but reconstruction still happens)')
        print('This ensures diffusion learns detection-suitable features')
        print('='*80 + '\n')

    # Save config and setup tensorboard writer
    if rank == 0:
        import yaml
        with open(os.path.join(model_dir, 'config.yaml'), 'w') as f:
            yaml.dump(hypes, f)

        print(f"Model directory: {model_dir}")
        writer = SummaryWriter(model_dir)
    else:
        model_dir = None
        writer = None

    # Training loop
    lowest_val_loss = 1e5
    lowest_val_epoch = -1
    num_epochs = hypes['train_params'].get('epoches', 30)

    if rank == 0:
        print(f'\nStarting Stage 2 Diffusion Training for {num_epochs} epochs...')
        print("="*80 + "\n")

    for epoch in range(num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        # Important: model.train() affects BatchNorm/Dropout behavior
        # Even though most parameters are frozen, we need train mode for the trainable modules
        model.train()
        train_diffusion_loss = 0

        for i, batch_data in enumerate(train_loader):
            batch_data = train_utils.to_device(batch_data, device)
            output_dict = model(batch_data['ego'])

            # Only diffusion loss
            diffusion_loss = output_dict.get('diffusion_loss', torch.tensor(0.0, device=device))

            optimizer.zero_grad()
            # Some batches may contain only the ego agent (no reconstruction targets),
            # making `diffusion_loss` a constant 0 tensor without grad_fn.
            # Ensure backward is always valid (and DDP-safe) by attaching a zero term
            # that depends on all trainable parameters.
            if isinstance(diffusion_loss, torch.Tensor) and (not diffusion_loss.requires_grad):
                zero = torch.zeros((), device=device)
                for p in diffusion_params:
                    zero = zero + p.sum() * 0.0
                diffusion_loss = diffusion_loss + zero
            diffusion_loss.backward()

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(diffusion_params, max_norm=1.0)

            optimizer.step()

            train_diffusion_loss += diffusion_loss.item()

            if rank == 0 and i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Step [{i}/{len(train_loader)}], "
                      f"Diffusion Loss: {diffusion_loss.item():.4f}")
                global_step = epoch * len(train_loader) + i
                if writer:
                    writer.add_scalar('train/diffusion_loss', diffusion_loss.item(), global_step)

        # Validation
        if epoch % hypes['train_params']['eval_freq'] == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    batch_data = train_utils.to_device(batch_data, device)
                    output_dict = model(batch_data['ego'])
                    diffusion_loss = output_dict.get('diffusion_loss', torch.tensor(0.0, device=device))
                    val_loss += diffusion_loss.item()

            avg_val_loss = val_loss / len(val_loader)

            if rank == 0:
                print(f"\nEpoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
                if writer:
                    writer.add_scalar('val/total_loss', avg_val_loss, epoch)

                if avg_val_loss < lowest_val_loss:
                    lowest_val_loss = avg_val_loss
                    torch.save(
                        model.module.state_dict() if distributed else model.state_dict(),
                        os.path.join(model_dir, f'net_epoch_bestval_at{epoch+1}.pth')
                    )
                    if lowest_val_epoch != -1:
                        old_path = os.path.join(model_dir, f'net_epoch_bestval_at{lowest_val_epoch}.pth')
                        if os.path.exists(old_path):
                            os.remove(old_path)
                    lowest_val_epoch = epoch + 1

        if rank == 0 and (epoch + 1) % hypes['train_params']['save_freq'] == 0:
            checkpoint_path = os.path.join(model_dir, f'net_epoch{epoch+1}.pth')
            torch.save(
                model.module.state_dict() if distributed else model.state_dict(),
                checkpoint_path
            )
            print(f"Checkpoint saved: {checkpoint_path}")

        scheduler.step()
        if rank == 0:
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}\n")

    if rank == 0:
        print("="*80)
        print("Stage 2 Training Completed!")
        print(f"Best model: net_epoch_bestval_at{lowest_val_epoch}.pth")
        print(f"Models saved to: {model_dir}")
        print("="*80)
        if writer:
            writer.close()
        if log_file:
            log_file.close()

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
