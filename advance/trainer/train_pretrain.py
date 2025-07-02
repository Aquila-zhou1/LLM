import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from advance.model.model_main import MainConfig, MainForCausalLM
from dataset.lm_dataset import PretrainDataset
warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def evaluate_model(model, val_loader, loss_fct, ctx, args):
    """
    评估模型在验证集上的性能
    返回验证集的平均loss和困惑度(PPL)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for step, (X, Y, loss_mask) in enumerate(val_loader):
            X = X.to(args.device)
            Y = Y.to(args.device)
            loss_mask = loss_mask.to(args.device)

            with ctx:
                res = model(X)
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                loss = (loss * loss_mask).sum() / loss_mask.sum()
                loss += res.aux_loss

            total_loss += loss.item() * loss_mask.sum().item()
            total_tokens += loss_mask.sum().item()

            # 限制验证步数，避免验证时间过长
            if step >= 50:  # 只评估前50个batch
                break

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(avg_loss) if avg_loss < 10 else float('inf')  # 防止数值溢出

    model.train()
    return avg_loss, ppl


def train_epoch(epoch, wandb, val_loader):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            train_loss = loss.item() * args.accumulation_steps
            train_ppl = math.exp(train_loss) if train_loss < 10 else float('inf')  # 计算训练困惑度

            Logger(
                'Epoch:[{}/{}]({}/{}) train_loss:{:.3f} train_ppl:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    train_loss,
                    train_ppl,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 记录训练指标到wandb
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "train_loss": train_loss,
                    "train_ppl": train_ppl,
                    "learning_rate": optimizer.param_groups[-1]['lr'],
                    "epoch": epoch + 1,
                    "step": epoch * iter_per_epoch + step
                })

        # 验证集评估（每隔一定步数进行）
        if step % args.eval_interval == 0 and step > 0 and val_loader is not None:
            val_loss, val_ppl = evaluate_model(model, val_loader, loss_fct, ctx, args)
            Logger(f'Validation - loss:{val_loss:.3f} ppl:{val_ppl:.3f}')

            # 记录验证指标到wandb
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "step": epoch * iter_per_epoch + step
                })

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MainForCausalLM(lm_config).to(args.device)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Improved llm model")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=500, help="验证集评估间隔步数")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=768, type=int)  # 设置为512
    parser.add_argument('--num_hidden_layers', default=16, type=int)  # 设置为8
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    args = parser.parse_args()

    lm_config = MainConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"
    
    args.wandb_run_name = f"Main-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        # 登录wandb（如果需要的话）
        # wandb.login()  # 如果没有登录，请取消注释这行

        wandb.init(
            settings=wandb.Settings(init_timeout=120),
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden_size": args.hidden_size,
                "num_hidden_layers": args.num_hidden_layers,
                "max_seq_len": args.max_seq_len,
                "use_moe": args.use_moe,
                "accumulation_steps": args.accumulation_steps,
                "grad_clip": args.grad_clip,
                "val_ratio": args.val_ratio,
                "eval_interval": args.eval_interval
            }
        )
    else:
        wandb = None
        

    model, tokenizer = init_model(lm_config)

    # 加载完整数据集
    full_dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # 分割训练集和验证集
    if args.val_ratio > 0:
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * args.val_ratio)
        train_size = dataset_size - val_size

        # 使用torch.utils.data.random_split进行数据分割
        train_ds, val_ds = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # 固定随机种子确保可复现
        )
        Logger(f'数据集分割: 训练集 {train_size} 样本, 验证集 {val_size} 样本')
    else:
        train_ds = full_dataset
        val_ds = None
        Logger(f'使用全部数据进行训练: {len(train_ds)} 样本')

    # 创建训练数据加载器
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False if ddp else True,  # DDP时使用sampler，否则使用shuffle
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 创建验证数据加载器
    val_loader = None
    if val_ds is not None:
        val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp else None
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=args.num_workers,
            sampler=val_sampler
        )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    Logger(f'开始训练，每个epoch有 {iter_per_epoch} 个batch')

    for epoch in range(args.epochs):
        Logger(f'开始第 {epoch + 1}/{args.epochs} 个epoch的训练')
        train_epoch(epoch, wandb, val_loader)
