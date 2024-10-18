import json
import logging
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import pynvml
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from tqdm import tqdm

from .utils import get_autocast, is_master
from inf_cl import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, CLIP, CustomTextCLIP
from inf_cl.models.loss import ClipLoss

try:
    import wandb
except ImportError:
    wandb = None


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)


def get_memory():
    pynvml.nvmlInit()
    # NOTE: 0 denotes GPU index.
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    return meminfo.used / 1024**3


def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    hours = int(hours); minutes = int(minutes); seconds = int(seconds)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def cal_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def const_lr_cooldown(optimizer, base_lr, warmup_length, steps, cooldown_steps, cooldown_power=1.0, cooldown_end_lr=0.):
    def _lr_adjuster(step):
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                # linear decay if power == 1; polynomial decay otherwise;
                decay = (1 - (e/es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GradientAccum:

    def __init__(self, model, loss, scaler, autocast, input_dtype, device):
        self.model = model
        self.loss = loss
        self.scaler = scaler
        self.autocast = autocast
        self.input_dtype = input_dtype
        self.device = device

        self.logit_scale = unwrap_model(model).logit_scale
        self.arch_type = unwrap_model(model).arch_type

        self.accum_freq = 0
        self.accum_cpu_states = []
        self.accum_gpu_devices_states = []
        self.accum_images = []
        self.accum_texts = []
        self.accum_image_features = []
        self.accum_text_features = []

        self.rank = dist.get_rank()

    def clear(self):
        self.accum_image_features.clear()
        self.accum_text_features.clear()
        torch.cuda.empty_cache()

    def clear_state(self):
        self.accum_images.clear()
        self.accum_texts.clear()
        self.accum_cpu_states.clear()
        self.accum_gpu_devices_states.clear()
        self.accum_freq = 0

    @torch.no_grad()
    def accum_inference(self, images, texts):
        images = images.to(device=self.device, dtype=self.input_dtype, non_blocking=True)
        texts = texts.to(device=self.device, non_blocking=True)
        # First, cache the features without any gradient tracking.
        with self.autocast():
            # collect rand states
            self.accum_cpu_states.append(torch.get_rng_state())
            self.accum_gpu_devices_states.append(torch_checkpoint.get_device_states(*[images, texts]))

            model_out = self.model(images, texts)

            self.accum_image_features.append(model_out["image_features"].detach().clone())
            self.accum_text_features.append(model_out["text_features"].detach().clone())

        if self.arch_type == "lit":
            # lit
            accum_image = model_out["image_trunk_features"].detach().clone()
        else:
            accum_image = images.detach().clone()
        accum_text = texts.detach().clone()
        
        # offloading
        accum_image = accum_image.cpu()
        accum_text  = accum_text.cpu()

        self.accum_images.append(accum_image)
        self.accum_texts.append(accum_texts)

        self.accum_freq += 1

    def accum_forward_backward(self):
        accum_losses = {"loss": 0.0}
        for j in range(self.accum_freq):
            images = self.accum_images[j]
            texts  = self.accum_texts[j]
            
            # refer to the implementation of Gradient Cache: https://github.com/luyug/GradCache/blob/906f03835fbc183132a9db32612a9e8f180ca3b4/src/grad_cache/grad_cache.py#L235
            # DDP will sync gradients across GPUs, which is no need except the last batch.
            sync_context = self.model.no_sync if j != self.accum_freq - 1 else nullcontext

            with torch.random.fork_rng(devices=(device,)), sync_context():
                # setting random states
                torch.set_rng_state(self.accum_cpu_states[j])
                torch_checkpoint.set_device_states(*self.accum_gpu_devices_states[j])

                with autocast():
                    model_out = self.model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    inputs["image_features"] = torch.cat(self.accum_image_features[:j] + [model_out["image_features"]] + self.accum_image_features[j + 1:])
                    inputs["text_features"] = torch.cat(self.accum_text_features[:j] + [model_out["text_features"]] + self.accum_text_features[j + 1:])

                    losses = self.loss(**inputs, **inputs_no_accum)
                    show_loss = losses.pop("show_loss")
                    total_loss = sum(losses.values())
                    losses["loss"] = show_loss

                    del inputs
                    del inputs_no_accum

                backward(total_loss, scaler)
                accum_losses["loss"] += losses["loss"]

        accum_losses["loss"] /= accum_freq

        self.clear()
        self.clear_state()

        return accum_losses


class GradientCache:

    def __init__(self, model, loss, scaler, autocast, input_dtype, device):
        self.model = model
        self.loss = loss
        self.scaler = scaler
        self.autocast = autocast
        self.input_dtype = input_dtype
        self.device = device

        self.logit_scale = unwrap_model(model).logit_scale
        self.arch_type = unwrap_model(model).arch_type

        self.accum_freq = 0
        self.accum_cpu_states = []
        self.accum_gpu_devices_states = []
        self.accum_images = []
        self.accum_texts = []
        self.accum_image_features = []
        self.accum_text_features = []

        self.rank = dist.get_rank()

    def clear(self):
        self.accum_image_features.clear()
        self.accum_text_features.clear()
        torch.cuda.empty_cache()

    def clear_state(self):
        self.accum_images.clear()
        self.accum_texts.clear()
        self.accum_cpu_states.clear()
        self.accum_gpu_devices_states.clear()
        self.accum_freq = 0

    def forward_backward(self, images, texts):
        images = images.to(device=self.device, dtype=self.input_dtype, non_blocking=True)
        texts  = texts.to(device=self.device, non_blocking=True)
        with self.autocast():
            model_out = self.model(image=images, text=texts)

        model_out.pop("image_trunk_features", None)

        losses = self.loss(**model_out)
        show_loss = losses.pop("show_loss")
        total_loss = sum(losses.values())
        losses["loss"] = show_loss

        backward(total_loss, self.scaler)

        return losses

    @torch.no_grad()
    def accum_inference(self, images, texts):
        images = images.to(device=self.device, dtype=self.input_dtype, non_blocking=True)
        texts = texts.to(device=self.device, non_blocking=True)
        # First, cache the features without any gradient tracking.
        with self.autocast():
            # collect rand states
            self.accum_cpu_states.append(torch.get_rng_state())
            self.accum_gpu_devices_states.append(torch_checkpoint.get_device_states(*[images, texts]))

            model_out = self.model(image=images, text=texts)

            self.accum_image_features.append(model_out["image_features"])
            self.accum_text_features.append(model_out["text_features"])

        # Speed analysis of detach().clone(): https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
        if self.arch_type == "lit":
            # lit
            accum_image = model_out["image_trunk_features"].detach().clone()
        else:
            accum_image = images.detach().clone()
        accum_text = texts.detach().clone()
        
        # offloading
        # accum_image = accum_image.cpu()
        # accum_text  = accum_text.cpu()

        self.accum_images.append(accum_image)
        self.accum_texts.append(accum_text)

        self.accum_freq += 1

    def accum_forward_backward(self):
        accum_qs = [x.requires_grad_() for x in self.accum_image_features]; qs = torch.cat(accum_qs, dim=0)
        accum_ks = [x.requires_grad_() for x in self.accum_text_features]; ks = torch.cat(accum_ks, dim=0)
        ls = self.logit_scale.exp().detach().clone().requires_grad_()

        losses = self.loss(image_features=qs, text_features=ks, logit_scale=ls)
        show_loss = losses.pop("show_loss")
        total_loss = sum(losses.values())
        losses["loss"] = show_loss

        backward(total_loss, self.scaler)

        accum_q_grads = [q.grad for q in accum_qs]
        accum_k_grads = [k.grad for k in accum_ks]
        l_grad = ls.grad

        del accum_qs, accum_ks
        del qs, ks, ls

        # Clean trash memory from loss calculation or inference
        self.clear()

        for j in range(self.accum_freq):
            images = self.accum_images[j]
            texts = self.accum_texts[j]

            # refer to the implementation of Gradient Cache: https://github.com/luyug/GradCache/blob/906f03835fbc183132a9db32612a9e8f180ca3b4/src/grad_cache/grad_cache.py#L235
            # DDP will sync gradients across GPUs, which is no need except the last batch.
            sync_context = self.model.no_sync if j != self.accum_freq - 1 else nullcontext

            with torch.random.fork_rng(devices=(self.device, )), sync_context():
                # setting random states
                torch.set_rng_state(self.accum_cpu_states[j])
                torch_checkpoint.set_device_states(*self.accum_gpu_devices_states[j])

                with self.autocast():
                    if self.arch_type == "lit":
                        model_out = self.model(images, texts, project_only=True)
                    else:
                        model_out = self.model(images, texts)

                q = model_out["image_features"]
                k = model_out["text_features"]
                l = model_out["logit_scale"]

                _loss = torch.dot(q.flatten(), accum_q_grads[j].flatten()) + \
                        torch.dot(k.flatten(), accum_k_grads[j].flatten()) + \
                        l * l_grad / self.accum_freq

                _loss.backward()

        self.clear_state()

        return losses


def train_one_epoch(start_timestamp, model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    runner = GradientCache(model, loss, scaler, autocast, input_dtype, device)

    rest_iters = num_batches_per_epoch * (args.epochs - epoch)

    losses_m = {}
    global_batch_time_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m  = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            losses = runner.forward_backward(images, texts)
        else:
            runner.accum_inference(images, texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            losses = runner.accum_forward_backward()

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        global_batch_time_m.update(time.time() - end)
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = unwrap_model(model).logit_scale.exp().item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            grad_norm = cal_grad_norm(model.module)

            running_time = seconds_to_hms(time.time() - start_timestamp)
            rest_iters = rest_iters - 1
            whole_time   = seconds_to_hms(time.time() - start_timestamp + rest_iters * global_batch_time_m.avg)
            logging.info(
                f"{running_time}<{whole_time} "
                f"Epoch: {epoch + percent_complete:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Grad Norm: {grad_norm:.3f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log + " "
                f"Memory: {get_memory():.2f}GB "
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "grad_norm": grad_norm,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def zero_shot_run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.inference_mode():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = zero_shot_run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = zero_shot_run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results
