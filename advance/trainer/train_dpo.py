import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from advance.model.model_main import MainConfig, MainForCausalLM
from dataset.lm_dataset import DPODataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def calculate_dpo_metrics(chosen_probs, rejected_probs, ref_chosen_probs, ref_rejected_probs, beta=0.1):
    """
    计算DPO任务的评估指标
    返回: 偏好准确率, 奖励差异, 策略KL散度等
    """
    with torch.no_grad():
        # 计算log ratios
        pi_logratios = chosen_probs - rejected_probs
        ref_logratios = ref_chosen_probs - ref_rejected_probs

        # 计算偏好准确率 (preference accuracy)
        # 当chosen的概率大于rejected时，认为预测正确
        preference_accuracy = (pi_logratios > 0).float().mean()

        # 计算奖励差异 (reward margin)
        reward_margin = pi_logratios.mean()

        # 计算隐式奖励 (implicit reward)
        chosen_rewards = beta * chosen_probs
        rejected_rewards = beta * rejected_probs

        # 计算策略KL散度的近似值
        kl_divergence = (pi_logratios - ref_logratios).abs().mean()

        return {
            'preference_accuracy': preference_accuracy.item(),
            'reward_margin': reward_margin.item(),
            'chosen_rewards_mean': chosen_rewards.mean().item(),
            'rejected_rewards_mean': rejected_rewards.mean().item(),
            'kl_divergence': kl_divergence.item()
        }


def evaluate_dpo_model(model, ref_model, val_loader, ctx, args):
    """
    评估DPO模型在验证集上的性能
    返回验证集的各项指标
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_preference_accuracy = 0.0
    total_reward_margin = 0.0

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            x_chosen = batch['x_chosen'].to(args.device)
            x_rejected = batch['x_rejected'].to(args.device)
            y_chosen = batch['y_chosen'].to(args.device)
            y_rejected = batch['y_rejected'].to(args.device)
            mask_chosen = batch['mask_chosen'].to(args.device)
            mask_rejected = batch['mask_rejected'].to(args.device)
            x = torch.cat([x_chosen, x_rejected], dim=0)
            y = torch.cat([y_chosen, y_rejected], dim=0)
            mask = torch.cat([mask_chosen, mask_rejected], dim=0)

            with ctx:
                # 计算参考模型的概率
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
                ref_probs = logits_to_probs(ref_logits, y)
                ref_probs = ref_probs * mask

                # 计算当前模型的概率
                outputs = model(x)
                logits = outputs.logits
                probs = logits_to_probs(logits, y)
                probs = probs * mask

                # 计算DPO损失
                loss = dpo_loss(ref_probs, probs, mask, beta=0.1)

                # 分离chosen和rejected的概率用于指标计算
                batch_size = x_chosen.size(0)
                seq_lengths = mask.sum(dim=1, keepdim=True)
                ref_probs_avg = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()
                probs_avg = (probs * mask).sum(dim=1) / seq_lengths.squeeze()

                ref_chosen_probs = ref_probs_avg[:batch_size]
                ref_rejected_probs = ref_probs_avg[batch_size:]
                chosen_probs = probs_avg[:batch_size]
                rejected_probs = probs_avg[batch_size:]

            # 计算指标
            metrics = calculate_dpo_metrics(chosen_probs, rejected_probs, ref_chosen_probs, ref_rejected_probs, beta=0.1)

            total_loss += loss.item() * batch_size
            total_samples += batch_size
            total_preference_accuracy += metrics['preference_accuracy'] * batch_size
            total_reward_margin += metrics['reward_margin'] * batch_size

            # 限制验证步数，避免验证时间过长
            if step >= 50:
                break

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_preference_accuracy = total_preference_accuracy / total_samples if total_samples > 0 else 0.0
    avg_reward_margin = total_reward_margin / total_samples if total_samples > 0 else 0.0

    model.train()
    return {
        'val_loss': avg_loss,
        'val_preference_accuracy': avg_preference_accuracy,
        'val_reward_margin': avg_reward_margin
    }


def logits_to_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs


def dpo_loss(ref_probs, probs, mask, beta):
    # ref_probs 和 probs 都是 shape: (batch_size, seq_len)
    seq_lengths = mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    probs = (probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 将 chosen 和 rejected 数据分开
    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[:batch_size // 2]
    reject_ref_probs = ref_probs[batch_size // 2:]
    chosen_probs = probs[:batch_size // 2]
    reject_probs = probs[batch_size // 2:]

    pi_logratios = chosen_probs - reject_probs
    ref_logratios = chosen_ref_probs - reject_ref_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, wandb, val_loader):
    start_time = time.time()

    for step, batch in enumerate(train_loader):
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_probs = logits_to_probs(ref_logits, y)
            ref_probs = ref_probs * mask
            outputs = model(x)
            logits = outputs.logits
            probs = logits_to_probs(logits, y)
            probs = probs * mask
            loss = dpo_loss(ref_probs, probs, mask, beta=0.1)
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

            # 计算DPO训练指标
            batch_size = x_chosen.size(0)
            seq_lengths = mask.sum(dim=1, keepdim=True)
            ref_probs_avg = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()
            probs_avg = (probs * mask).sum(dim=1) / seq_lengths.squeeze()

            ref_chosen_probs = ref_probs_avg[:batch_size]
            ref_rejected_probs = ref_probs_avg[batch_size:]
            chosen_probs = probs_avg[:batch_size]
            rejected_probs = probs_avg[batch_size:]

            train_metrics = calculate_dpo_metrics(chosen_probs, rejected_probs, ref_chosen_probs, ref_rejected_probs, beta=0.1)

            Logger(
                'Epoch:[{}/{}]({}/{}) train_loss:{:.3f} pref_acc:{:.3f} reward_margin:{:.3f} kl_div:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    train_loss,
                    train_metrics['preference_accuracy'],
                    train_metrics['reward_margin'],
                    train_metrics['kl_divergence'],
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 记录DPO训练指标到wandb
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "train_loss": train_loss,
                    "train_preference_accuracy": train_metrics['preference_accuracy'],
                    "train_reward_margin": train_metrics['reward_margin'],
                    "train_chosen_rewards_mean": train_metrics['chosen_rewards_mean'],
                    "train_rejected_rewards_mean": train_metrics['rejected_rewards_mean'],
                    "train_kl_divergence": train_metrics['kl_divergence'],
                    "learning_rate": optimizer.param_groups[-1]['lr'],
                    "epoch": epoch + 1,
                    "step": epoch * iter_per_epoch + step
                })

        # 验证集评估（每隔一定步数进行）
        if step % args.eval_interval == 0 and step > 0 and val_loader is not None:
            val_metrics = evaluate_dpo_model(model, ref_model, val_loader, ctx, args)
            Logger(f'Validation - loss:{val_metrics["val_loss"]:.3f} pref_acc:{val_metrics["val_preference_accuracy"]:.3f} reward_margin:{val_metrics["val_reward_margin"]:.3f}')

            # 记录验证指标到wandb
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "val_loss": val_metrics["val_loss"],
                    "val_preference_accuracy": val_metrics["val_preference_accuracy"],
                    "val_reward_margin": val_metrics["val_reward_margin"],
                    "step": epoch * iter_per_epoch + step
                })

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/rlhf_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MainForCausalLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    # 初始化参考模型
    ref_model = MainForCausalLM(lm_config)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    ref_model.requires_grad_(False)

    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    ref_model = ref_model.to(args.device)

    return model, ref_model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main RLHF")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    # sft阶段学习率为 「5e-6」->「5e-7」长度512，建议离线正负样本「概率」偏好对齐阶段lr <=「1e-8」长度3000，否则很容易遗忘训坏
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Improved llm model dpo")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=500, help="验证集评估间隔步数")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=768, type=int)  # 设置为512
    parser.add_argument('--num_hidden_layers', default=16, type=int)  # 设置为8
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl")

    args = parser.parse_args()

    lm_config = MainConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"LLM-Full-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

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

        wandb.init(
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
                "eval_interval": args.eval_interval,
                "task_type": "DPO"
            }
        )
    else:
        wandb = None

    model, ref_model, tokenizer = init_model(lm_config)

    # 加载完整数据集
    full_dataset = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)

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
        Logger(f'DPO数据集分割: 训练集 {train_size} 样本, 验证集 {val_size} 样本')
    else:
        train_ds = full_dataset
        val_ds = None
        Logger(f'使用全部数据进行DPO训练: {len(train_ds)} 样本')

    # 创建训练数据加载器
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False if ddp else True,
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
    Logger(f'开始DPO训练，每个epoch有 {iter_per_epoch} 个batch')

    for epoch in range(args.epochs):
        Logger(f'开始第 {epoch + 1}/{args.epochs} 个epoch的DPO训练')
        train_epoch(epoch, wandb, val_loader)
