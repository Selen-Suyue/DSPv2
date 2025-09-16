import os
import torch
import argparse
import numpy as np
import torch.nn as nn

import torch.distributed as dist

from tqdm import tqdm
import MinkowskiEngine as ME
from diffusers.optimization import get_cosine_schedule_with_warmup
from policy import dspv2
from utils.training import set_seed, plot_history, sync_loss
from dataset.real import FastMinkSet, collate_fn

def train(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ['RANK'])
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    os.environ['NCCL_P2P_DISABLE'] = '1'
    dist.init_process_group(backend = 'nccl', init_method = 'env://', world_size = WORLD_SIZE, rank = RANK)

    set_seed(args.seed)
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if RANK == 0: print("Loading dataset ...")
    dataset = FastMinkSet(
        processed_root_dir = args.data_path,
        action_horizon = args.num_action,
        task = args.task
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, 
        num_replicas = WORLD_SIZE, 
        rank = RANK, 
        shuffle = True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = args.batch_size // WORLD_SIZE,
        num_workers = args.num_workers,
        sampler = sampler,
        collate_fn = collate_fn,
    )

    if RANK == 0: print("Loading policy ...")
    policy = dspv2(
        Tp = args.num_action,
        Ta = args.num_action,
        input_dim = 3,
        action_dim = 33,
    ).to(device)
    if RANK == 0:
        n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))
    policy = nn.parallel.DistributedDataParallel(
        policy, 
        device_ids = [LOCAL_RANK], 
        output_device = LOCAL_RANK, 
        find_unused_parameters = True
    )


    if args.resume_ckpt is not None:
        policy.module.load_state_dict(torch.load(args.resume_ckpt, map_location = device), strict = False)
        if RANK == 0:
            print("Checkpoint {} loaded.".format(args.resume_ckpt))

    if RANK == 0 and not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    if RANK == 0: print("Loading optimizer and scheduler ...")
    optimizer = torch.optim.AdamW(policy.parameters(), lr = args.lr, betas = [0.95, 0.999], weight_decay = 1e-6)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 2000,
        num_training_steps = len(dataloader) * args.num_epochs
    )
    lr_scheduler.last_epoch = len(dataloader) * (args.resume_epoch + 1) - 1

    train_history = []

    policy.train()
    for epoch in range(args.resume_epoch + 1, args.num_epochs):
        if RANK == 0: print("Epoch {}".format(epoch)) 
        sampler.set_epoch(epoch)
        optimizer.zero_grad()
        num_steps = len(dataloader)
        pbar = tqdm(dataloader) if RANK == 0 else dataloader
        avg_loss = 0

        for data in pbar:
            cloud_coords = data["pcd_coords"]
            cloud_feats = data["pcd_feats"]
            action_data = data["action"]
            qpos_data = data["pose"]
            imgs = data['images']
            cloud_feats, cloud_coords, action_data, qpos_data,imgs = (
                cloud_feats.to(device),
                cloud_coords.to(device),
                action_data.to(device),
                qpos_data.to(device),
                imgs.to(device)
            )
            
            cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)
            loss = policy(cloud = cloud_data,
                              actions = action_data,
                              qpos = qpos_data,
                              imgs = imgs,
                              batch_size = action_data.shape[0])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            avg_loss += loss.item()
            if RANK==0: pbar.set_description("Iter loss: {:.6f}".format(loss.item()))


        avg_loss = avg_loss / num_steps
        sync_loss(avg_loss, device)
        train_history.append(avg_loss)


        if RANK == 0:
            print("Train loss: {:.6f}".format(avg_loss))

            if (epoch + 1) % args.save_epochs == 0:
                torch.save(
                    policy.module.state_dict(),
                    os.path.join(args.ckpt_dir, "policy_epoch_{}_seed_{}.ckpt".format(epoch + 1, args.seed))
                )
                plot_history(train_history, epoch, args.ckpt_dir, args.seed)
                
    if RANK == 0:
        torch.save(
            policy.module.state_dict(),
            os.path.join(args.ckpt_dir, "policy_last.ckpt")
        ) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action = 'store', type = str, help = 'data path', required = True)
    parser.add_argument('--num_action', action = 'store', type = int, help = 'number of action steps', required = False, default = 16)
    parser.add_argument('--ckpt_dir', action = 'store', type = str, help = 'checkpoint directory', required = True)
    parser.add_argument('--resume_ckpt', action = 'store', type = str, help = 'resume checkpoint file', required = False, default = None)
    parser.add_argument('--resume_epoch', action = 'store', type = int, help = 'resume from which epoch', required = False, default = -1)
    parser.add_argument('--lr', action = 'store', type = float, help = 'learning rate', required = False, default = 3e-4)
    parser.add_argument('--batch_size', action = 'store', type = int, help = 'batch size', required = False, default = 120)
    parser.add_argument('--num_epochs', action = 'store', type = int, help = 'training epochs', required = False, default = 1000)
    parser.add_argument('--save_epochs', action = 'store', type = int, help = 'saving epochs', required = False, default = 200)
    parser.add_argument('--num_workers', action = 'store', type = int, help = 'number of workers', required = False, default = 24)
    parser.add_argument('--seed', action = 'store', type = int, help = 'seed', required = False, default = 233)
    parser.add_argument('--task', type=str, default='default', help='Task name to load specific pose limits.')
    args = parser.parse_args()
    train(parser.parse_args())
