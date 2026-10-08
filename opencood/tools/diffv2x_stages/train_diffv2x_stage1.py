# -*- coding: utf-8 -*-
"""Train the PointPillars detector and pyramid fusion (LR-V2X stage 1)."""

import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset


def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_parser():
    parser = argparse.ArgumentParser(description="DiffV2X Stage 1: Baseline Training")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='YAML configuration file')
    parser.add_argument('--model_dir', default='',
                        help='Directory to save models')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    return parser.parse_args()


def main():
    seed_all()
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
    dist_backend = 'nccl'
    distributed = world_size > 1

    if distributed:
        torch.distributed.init_process_group(
            backend=dist_backend,
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

    # Setup logging and model directory early (only rank 0)
    if rank == 0:
        if opt.model_dir:
            saved_path = opt.model_dir
            os.makedirs(saved_path, exist_ok=True)
        else:
            saved_path = train_utils.setup_train(hypes)

        import yaml
        save_name = os.path.join(saved_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

        # Setup logging to file
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(saved_path, f'train_{timestamp}.log')
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
        print("DiffV2X Stage 1: Baseline Pyramid Fusion Training")
        print("="*80)
        print(f"Training configuration: {opt.hypes_yaml}")
        print(f"Distributed: {distributed}, World size: {world_size}, Rank: {rank}")
        print("="*80)

    # Build datasets
    if rank == 0:
        print('\nDataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if distributed:
        train_sampler = DistributedSampler(opencood_train_dataset)
        val_sampler = DistributedSampler(opencood_validate_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        opencood_train_dataset,
        batch_size=hypes['train_params']['batch_size'],
        num_workers=4,
        collate_fn=opencood_train_dataset.collate_batch_train,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        opencood_validate_dataset,
        batch_size=hypes['train_params']['batch_size'],
        num_workers=4,
        collate_fn=opencood_validate_dataset.collate_batch_train,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2
    )

    # Create model
    if rank == 0:
        print('\nCreating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move to device
    if torch.cuda.is_available():
        model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=True
        )

    # Loss and optimizer
    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)

    # Setup tensorboard writer
    if rank == 0:
        print(f"\nModel directory: {saved_path}")
        writer = SummaryWriter(saved_path)
    else:
        saved_path = None
        writer = None

    # LR scheduler
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # Training loop
    lowest_val_loss = 1e5
    lowest_val_epoch = -1
    num_epochs = hypes['train_params']['epoches']

    # Check if we need single agent supervision (for pyramid loss)
    supervise_single_flag = False
    if hasattr(opencood_train_dataset, "supervise_single"):
        supervise_single_flag = opencood_train_dataset.supervise_single
    single_weight = hypes['train_params'].get("single_weight", 1.0)

    if rank == 0:
        print(f"\nStarting Stage 1 training for {num_epochs} epochs...")
        print(f"Single agent supervision: {supervise_single_flag}")
        if supervise_single_flag:
            print(f"Single supervision weight: {single_weight}")
        print("="*80 + "\n")

    for epoch in range(num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0

        for i, batch_data in enumerate(train_loader):
            batch_data = train_utils.to_device(batch_data, device)

            optimizer.zero_grad()
            output_dict = model(batch_data['ego'])

            # Main collaborative loss
            final_loss = criterion(output_dict, batch_data['ego']['label_dict'])

            # Log collaborative losses immediately after calculation
            if rank == 0 and i % 10 == 0:
                criterion.logging(epoch, i, len(train_loader), writer)

            # Single agent supervision (for pyramid loss)
            if supervise_single_flag:
                single_loss = criterion(output_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                final_loss = final_loss + single_loss * single_weight

                # Log single agent supervision losses immediately after calculation
                if rank == 0 and i % 10 == 0:
                    criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            final_loss.backward()
            optimizer.step()

            train_loss += final_loss.item()

            if rank == 0 and i % 10 == 0:
                global_step = epoch * len(train_loader) + i
                if writer:
                    writer.add_scalar('train/total_loss', final_loss.item(), global_step)

        # Validation
        if epoch % hypes['train_params']['eval_freq'] == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data in val_loader:
                    batch_data = train_utils.to_device(batch_data, device)
                    output_dict = model(batch_data['ego'])
                    loss = criterion(output_dict, batch_data['ego']['label_dict'])

                    # Add single agent supervision for validation
                    if supervise_single_flag:
                        single_loss = criterion(output_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                        loss = loss + single_loss * single_weight

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            if rank == 0:
                print(f"\nEpoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
                if writer:
                    writer.add_scalar('val/total_loss', avg_val_loss, epoch)

                # Save best model
                if avg_val_loss < lowest_val_loss:
                    lowest_val_loss = avg_val_loss
                    torch.save(
                        model.module.state_dict() if distributed else model.state_dict(),
                        os.path.join(saved_path, f'net_epoch_bestval_at{epoch+1}.pth')
                    )
                    if lowest_val_epoch != -1:
                        old_path = os.path.join(saved_path, f'net_epoch_bestval_at{lowest_val_epoch}.pth')
                        if os.path.exists(old_path):
                            os.remove(old_path)
                    lowest_val_epoch = epoch + 1

        # Save checkpoints
        if rank == 0 and (epoch + 1) % hypes['train_params']['save_freq'] == 0:
            checkpoint_path = os.path.join(saved_path, f'net_epoch{epoch+1}.pth')
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
        print("Stage 1 Training Completed!")
        print(f"Best validation model: net_epoch_bestval_at{lowest_val_epoch}.pth")
        print(f"All models saved to: {saved_path}")
        print("="*80)
        if writer:
            writer.close()
        if log_file:
            log_file.close()

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
