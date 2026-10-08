import argparse
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils
from icecream import ic
import tqdm

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}
def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    # Two-stage training support
    parser.add_argument('--training_stage', type=int, default=None,
                        help='Training stage: 1 for diffusion pretraining, 2 for detection training')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., Stage 1 checkpoint for Stage 2)')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)
    if 'seed' in hypes.get('train_params', {}):
        seed = int(hypes['train_params']['seed'])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        # train_loader = DataLoader(opencood_train_dataset,
        #                           batch_sampler=batch_sampler_train,
        #                           num_workers=12,
        #                           collate_fn=opencood_train_dataset.collate_batch_train)
        # val_loader = DataLoader(opencood_validate_dataset,
        #                         sampler=sampler_val,
        #                         num_workers=12,
        #                         collate_fn=opencood_train_dataset.collate_batch_train,
        #                         drop_last=False)

        train_loader = DataLoader(
            opencood_train_dataset,
            batch_sampler=batch_sampler_train,
            num_workers=8,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=opencood_train_dataset.collate_batch_train
        )

        val_loader = DataLoader(
            opencood_validate_dataset,
            sampler=sampler_val,
            num_workers=8,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=opencood_train_dataset.collate_batch_train,
            drop_last=False
        )
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params'][
                                      'batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        # Check if this is resuming training or starting new training in a specified directory
        checkpoint_exists = os.path.exists(os.path.join(saved_path, 'config.yaml'))

        if checkpoint_exists:
            # Resume from existing checkpoint
            if opt.rank == 0:
                print(f"Resuming training from {saved_path}")
            init_epoch, model = train_utils.load_saved_model(saved_path, model)
            lowest_val_epoch = init_epoch
        else:
            # Start new training but save to specified directory
            if opt.rank == 0:
                print(f"Starting new training, saving to {saved_path}")
            init_epoch = 0
            os.makedirs(saved_path, exist_ok=True)
            # Save config to the directory (only rank 0)
            if opt.rank == 0:
                import yaml
                save_name = os.path.join(saved_path, 'config.yaml')
                with open(save_name, 'w') as outfile:
                    yaml.dump(hypes, outfile)
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # Two-stage training: Load checkpoint from --resume_from (e.g., Stage 1 → Stage 2)
    if opt.resume_from:
        if opt.rank == 0:
            print(f"[Two-Stage Training] Loading pretrained checkpoint from: {opt.resume_from}")
        checkpoint = torch.load(opt.resume_from, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        if opt.rank == 0:
            print(f"[Two-Stage Training] Successfully loaded checkpoint from {opt.resume_from}")

    # ddp setting
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True) # True
        model_without_ddp = model.module

    # ============================================
    # Two-Stage Training: Set training stage
    # ============================================
    if hasattr(model_without_ddp, 'set_training_stage'):
        training_stage = None

        # Priority 1: Explicit --training_stage argument
        if opt.training_stage is not None:
            training_stage = opt.training_stage
        # Priority 2: Auto-detect from model_dir name
        elif 'stage1' in saved_path.lower():
            training_stage = 1
        elif 'stage2' in saved_path.lower():
            training_stage = 2

        # Set the training stage
        if training_stage == 1:
            if opt.rank == 0:
                print("\n" + "="*60)
                print("[Two-Stage Training] Setting to Stage 1")
                print("="*60)
            model_without_ddp.set_training_stage(1)
            # 模型会自动打印详细的训练状态（见 _print_training_status）
        elif training_stage == 2:
            if opt.rank == 0:
                print("\n" + "="*60)
                print("[Two-Stage Training] Setting to Stage 2")
                print("="*60)
            model_without_ddp.set_training_stage(2)
            # 模型会自动打印详细的训练状态（见 _print_training_status）
            if opt.rank == 0 and opt.resume_from:
                print(f"Loading pretrained checkpoint from: {opt.resume_from}")
                print("="*60 + "\n")
    else:
        if opt.rank == 0 and (opt.training_stage is not None or 'stage' in saved_path.lower()):
            print("[Info] Model does not support two-stage training (no set_training_stage method)")

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)

    # record training
    writer = SummaryWriter(saved_path)

    # Setup logging to file (only rank 0)
    import sys
    import datetime
    log_file = None
    original_stdout = sys.stdout
    if opt.rank == 0:
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

        sys.stdout = TeeOutput(original_stdout, log_file)
        print(f"[Logging] Training log will be saved to: {log_filename}")

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        if opt.distributed:
            sampler_train.set_epoch(epoch)
        # the model will be evaluation mode during validation
        model.train()
        try: # heter_model stage2
            model_without_ddp.model_train_init()
        except:
            print("No model_train_init function")
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])

            criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag:
                if not opt.half:
                    final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                else:
                    with torch.cuda.amp.autocast():
                        final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()


        # torch.cuda.empty_cache() # it will destroy memory buffer
        if epoch % hypes['train_params']['save_freq'] == 0 and opt.rank == 0:
            torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
            
        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_loss_sum = 0.0
            valid_sample_count = 0

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    # For Stage 1, use diffusion loss as validation metric
                    # For Stage 2, use detection loss
                    if hasattr(model_without_ddp, 'training_stage') and model_without_ddp.training_stage == 1:
                        # Stage 1: validation metric = diffusion reconstruction loss
                        if 'diffusion_loss' in ouput_dict:
                            val_loss = ouput_dict['diffusion_loss'].item()
                        else:
                            val_loss = 0.0
                    else:
                        # Stage 2 or normal training: use detection loss
                        final_loss = criterion(ouput_dict,
                                               batch_data['ego']['label_dict'])
                        val_loss = final_loss.item()

                    batch_size = int(batch_data['ego']['record_len'].shape[0])
                    valid_loss_sum += val_loss * batch_size
                    valid_sample_count += batch_size

            if valid_sample_count == 0:
                raise RuntimeError('Validation produced no finite batches.')

            loss_stats = torch.tensor(
                [valid_loss_sum, valid_sample_count],
                dtype=torch.float64,
                device=device,
            )
            if opt.distributed:
                torch.distributed.all_reduce(
                    loss_stats,
                    op=torch.distributed.ReduceOp.SUM,
                )
            valid_ave_loss = (loss_stats[0] / loss_stats[1]).item()
            if opt.rank == 0:
                print('At epoch %d, the validation loss is %f' % (
                    epoch,
                    valid_ave_loss,
                ))
                writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                if opt.rank == 0:
                    torch.save(
                        model_without_ddp.state_dict(),
                        os.path.join(
                            saved_path,
                            'net_epoch_bestval_at%d.pth' % (epoch + 1),
                        ),
                    )
                    previous_best = os.path.join(
                        saved_path,
                        'net_epoch_bestval_at%d.pth' % lowest_val_epoch,
                    )
                    if lowest_val_epoch != -1 and os.path.exists(previous_best):
                        os.remove(os.path.join(saved_path,
                                        'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        scheduler.step(epoch)
        
        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    # Close log file
    if log_file is not None:
        sys.stdout = original_stdout
        log_file.close()

    if opt.rank == 0:
        run_test = True
        
        # ddp training may leave multiple bestval
        bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
        
        if len(bestval_model_list) > 1:
            bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("net_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
            ascending_idx = np.argsort(bestval_model_epoch_list)
            for idx in ascending_idx:
                if idx != (len(bestval_model_list) - 1):
                    os.remove(bestval_model_list[idx])

        run_test=False
        if run_test:
            fusion_method = opt.fusion_method
            cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
            print(f"Running command: {cmd}")
            os.system(cmd)


if __name__ == '__main__':
    main()
