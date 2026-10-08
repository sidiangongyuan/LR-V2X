# -*- coding: utf-8 -*-
"""Fine-tune reconstruction, fusion, and heads with a frozen sensor backbone (stage 3)."""

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
    parser = argparse.ArgumentParser(description="DiffV2X Stage 3: End-to-End Fine-tuning")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='YAML configuration file')
    parser.add_argument('--stage2_model', type=str, required=True,
                        help='Path to trained stage 2 model')
    parser.add_argument('--model_dir', default='',
                        help='Directory to save or resume fine-tuned models')
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
        print("DiffV2X Stage 3: End-to-End Fine-tuning")
        print("="*80)
        print(f"Stage 2 model: {opt.stage2_model}")
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

    # Create model
    if rank == 0:
        print('\nCreating Model')
    model = train_utils.create_model(hypes)

    # Load pretrained stage 2 model
    if rank == 0:
        print(f'Loading stage 2 model from {opt.stage2_model}')
    checkpoint = torch.load(opt.stage2_model, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    if rank == 0:
        print('Stage 2 model loaded successfully!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model.to(device)

    # Unfreeze all parameters for end-to-end training
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    # EXPERIMENT: Selective freezing based on config
    freeze_diffusion = hypes['train_params'].get('freeze_diffusion', False)
    freeze_detection_backbone = hypes['train_params'].get('freeze_detection_backbone', False)

    model_to_check = model.module if hasattr(model, 'module') else model

    if freeze_diffusion:
        # Freeze diffusion to prevent it from over-optimizing and losing semantic information
        modules_to_freeze = ['diffusion_model', 'latent_encoder']
        for name, module in model_to_check.named_children():
            if name in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
                if rank == 0:
                    print(f'[Stage3] Frozen module: {name}')

    if freeze_detection_backbone:
        # Freeze encoder/backbone to prevent moving target problem
        # But keep pyramid_backbone trainable for better fusion
        modules_to_freeze = ['encoder_m1', 'backbone_m1', 'shrink_conv']
        for name, module in model_to_check.named_children():
            if name in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
                if rank == 0:
                    print(f'[Stage3] Frozen module: {name}')

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nTotal trainable parameters: {total_params:,}')

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=True
        )

    # Loss and optimizer
    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # NOTE: Reconstruction is always enabled (no stage2_mode flag needed)
    # Both Stage 2 and Stage 3 use reconstructed features
    if rank == 0:
        print('\n' + '='*80)
        print('Stage 3: Joint training with full reconstruction pipeline')
        print('(Both detection and diffusion train on reconstructed features)')
        print('='*80 + '\n')

    # Diffusion weight for stage 3
    diffusion_weight = hypes['train_params'].get('stage3_diffusion_weight', 0.1)
    if rank == 0:
        print(f'Stage 3 diffusion loss weight: {diffusion_weight}')

    # Save config and setup tensorboard writer
    if rank == 0:
        import yaml
        with open(os.path.join(model_dir, 'config.yaml'), 'w') as f:
            yaml.dump(hypes, f)

        print(f"\nModel directory: {model_dir}")
        writer = SummaryWriter(model_dir)
    else:
        model_dir = None
        writer = None

    # Training loop
    lowest_val_loss = 1e5
    lowest_val_epoch = -1
    num_epochs = hypes['train_params'].get('epoches', 15)

    if rank == 0:
        print(f'\nStarting Stage 3 Fine-tuning for {num_epochs} epochs...')
        print(f"Diffusion loss weight: {diffusion_weight}")
        print("="*80 + "\n")

    for epoch in range(num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss_epoch = 0

        for i, batch_data in enumerate(train_loader):
            batch_data = train_utils.to_device(batch_data, device)
            output_dict = model(batch_data['ego'])

            # Detection loss (from criterion)
            detection_loss = criterion(output_dict, batch_data['ego']['label_dict'])

            # Diffusion loss (from model output)
            diffusion_loss = output_dict.get('diffusion_loss', torch.tensor(0.0, device=device))

            # DEBUG: Print loss components at first step
            if rank == 0 and i == 0 and epoch == 0:
                print(f"\n[DEBUG] Loss components:")
                print(f"  criterion output (detection_loss): {detection_loss.item():.6f}")
                print(f"  model output diffusion_loss: {diffusion_loss.item():.6f}")
                print(f"  diffusion_weight: {diffusion_weight}")
                if hasattr(criterion, 'loss_dict'):
                    print(f"  criterion.loss_dict: {criterion.loss_dict}")
                print()

            # Total loss: detection + weighted diffusion
            total_loss = detection_loss + diffusion_weight * diffusion_loss

            optimizer.zero_grad()
            total_loss.backward()

            # DEBUG: Check gradient flow at first step
            if rank == 0 and i == 0 and epoch == 0:
                model_to_check = model.module if hasattr(model, 'module') else model
                print(f"[DEBUG] Gradient flow check:")
                for name in ['cls_head', 'reg_head', 'dir_head', 'pyramid_backbone', 'diffusion_model']:
                    if hasattr(model_to_check, name):
                        module = getattr(model_to_check, name)
                        grad_norm = sum(p.grad.norm().item() if p.grad is not None else 0.0
                                       for p in module.parameters() if p.requires_grad)
                        print(f"  {name}: grad_norm = {grad_norm:.6f}")
                print()

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss_epoch += total_loss.item()

            if rank == 0 and i % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                      f"Detection Loss: {detection_loss.item():.4f}, "
                      f"Diffusion Loss: {diffusion_loss.item():.4f}")

                global_step = epoch * len(train_loader) + i
                if writer:
                    writer.add_scalar('train/detection_loss', detection_loss.item(), global_step)
                    writer.add_scalar('train/diffusion_loss', diffusion_loss.item(), global_step)
                    writer.add_scalar('train/total_loss', total_loss.item(), global_step)

        # Validation
        if epoch % hypes['train_params']['eval_freq'] == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    batch_data = train_utils.to_device(batch_data, device)
                    output_dict = model(batch_data['ego'])

                    # Only detection loss for validation (following codebook's approach)
                    # Note: codebook's stage3 only uses det_loss for model selection
                    det_loss = criterion(output_dict, batch_data['ego']['label_dict'])
                    val_loss += det_loss.item()

            avg_val_loss = val_loss / len(val_loader)

            if rank == 0:
                print(f"\nEpoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")
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

        # Save checkpoint
        if rank == 0 and (epoch + 1) % hypes['train_params']['save_freq'] == 0:
            ckpt_path = os.path.join(model_dir, f'net_epoch{epoch+1}.pth')
            torch.save(
                model.module.state_dict() if distributed else model.state_dict(),
                ckpt_path
            )
            print(f"Checkpoint saved: {ckpt_path}")

        scheduler.step()
        if rank == 0:
            print(f"LR after epoch {epoch+1}: {scheduler.get_last_lr()[0]:.6f}\n")

    if rank == 0:
        print("="*80)
        print("Stage 3 Training Completed!")
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
