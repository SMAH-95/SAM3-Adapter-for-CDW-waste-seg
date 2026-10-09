import argparse
import os

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist
import numpy as np
from prettytable import PrettyTable

torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

class SegmentationMetric:
    
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusionMatrix = np.zeros((num_classes, num_classes), dtype=np.float64)

    def _generate_matrix(self, pred, gt):
        pred = np.asarray(pred).astype(np.int64)
        gt = np.asarray(gt).astype(np.int64)

        mask = (gt != self.ignore_index)
        mask &= (gt >= 0) & (gt < self.num_classes)
        mask &= (pred >= 0) & (pred < self.num_classes)

        label = self.num_classes * gt[mask] + pred[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def addBatch(self, pred, gt):
        self.confusionMatrix += self._generate_matrix(pred, gt)

    def overallAccuracy(self):
        total = self.confusionMatrix.sum()
        if total == 0:
            return 0.0
        return np.diag(self.confusionMatrix).sum() / total

    def meanIntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = (
            self.confusionMatrix.sum(axis=1)
            + self.confusionMatrix.sum(axis=0)
            - intersection
        )
        iou = intersection / np.maximum(union, 1e-10)
        return np.nanmean(iou), iou

    def precision(self):
        tp = np.diag(self.confusionMatrix)
        pred_total = self.confusionMatrix.sum(axis=0)
        return tp / np.maximum(pred_total, 1e-10)

    def recall(self):
        tp = np.diag(self.confusionMatrix)
        gt_total = self.confusionMatrix.sum(axis=1)
        return tp / np.maximum(gt_total, 1e-10)

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = self.confusionMatrix.sum(axis=1) / np.maximum(self.confusionMatrix.sum(), 1e-10)
        iu = np.diag(self.confusionMatrix) / np.maximum(
            self.confusionMatrix.sum(axis=1)
            + self.confusionMatrix.sum(axis=0)
            - np.diag(self.confusionMatrix),
            1e-10
        )
        return (freq[freq > 0] * iu[freq > 0]).sum()



def onehot_to_index_label(mask):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1,2,0).numpy()
    x = np.argmax(mask, axis=-1)
    #colour_codes = np.array(palette)
    #x = np.uint8(colour_codes[x.astype(np.uint8)])*255
    #x=x.permute(2,0,1)
    #x=x.numpy()
    #x = np.around
    return x

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log(f'  {k}: shape={tuple(v.shape)}')

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler,drop_last=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


@torch.no_grad()
def eval_psnr(loader, model, eval_type=None):
    model.eval()
    torch.cuda.empty_cache()

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    elif eval_type == 'kvasir':
        metric_fn = utils.calc_kvasir
        metric1, metric2, metric3, metric4 = 'dice', 'iou', 'none', 'none'
    elif eval_type in ['seg', 'multiclass']:
        metric_fn = None
        metric1, metric2, metric3, metric4 = 'mIoU', 'OA', 'mF1', 'FWIoU'
    else:
        raise ValueError(f"Unsupported eval_type: {eval_type}")

    # Multi-class segmentation metric accumulator
    classes_list = config['train_dataset']['dataset']['args']['classes']
    num_classes = config['model']['args'].get('num_classes', len(classes_list))
    ignore_index = config.get('ignore_index', 255)
    ignore_background = config.get('ignore_background', False)
    metric_seg = SegmentationMetric(num_classes=num_classes, ignore_index=ignore_index)
        

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    
    val_metric1 = 0
    val_metric2 = 0
    val_metric3 = 0
    val_metric4 = 0
    cnt = 0
    
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        
        if hasattr(model, "module"):
            output_masks = model.module.infer(inp)
        else:
            output_masks = model.infer(inp)
        # output_masks: [B, num_classes, H, W]
        pred = torch.argmax(output_masks, dim=1).long()  # [B, H, W]

        print("output_masks shape:", output_masks.shape)
        print("output min/max:", output_masks.min().item(), output_masks.max().item())
        print("pred unique:", torch.unique(pred, return_counts=True))

        gt = batch['gt']
        print("raw gt shape:", gt.shape)
        print("raw gt min/max:", gt.min().item(), gt.max().item())
        
        if gt.dim() == 4:
            gt = torch.argmax(gt, dim=1)
        elif gt.dim() == 3:
            gt = gt
        else:
            raise ValueError(f"Unexpected GT shape: {gt.shape}")
            
        gt = gt.long()

        print("gt shape after argmax:", gt.shape)
        print("gt unique:", torch.unique(gt, return_counts=True))


        batch_pred = [
            torch.zeros_like(pred, dtype=pred.dtype, device=pred.device)
            for _ in range(dist.get_world_size())
        ]
        batch_gt = [
            torch.zeros_like(gt, dtype=gt.dtype, device=gt.device)
            for _ in range(dist.get_world_size())
        ]

        dist.all_gather(batch_pred, pred)
        dist.all_gather(batch_gt, gt)

        for i in range(len(batch_pred)):
            batch_pred[i] = batch_pred[i].cpu()
            batch_gt[i] = batch_gt[i].cpu()

        # pred_list.extend(batch_pred)
        # gt_list.extend(batch_gt)

        if pbar is not None:
            pbar.update(1)

        for i in range(len(batch_gt)):
            output_mask = batch_pred[i][0]  # [H, W], already class-index mask
            gt_mask = batch_gt[i][0]        # [H, W], already class-index mask

            mask_index_label = output_mask.numpy().flatten()
            gt_index_label = gt_mask.numpy().flatten()

            if eval_type in ['seg', 'multiclass']:
                metric_seg.addBatch(mask_index_label, gt_index_label)
        
        # result1, result2, result3, result4 = metric_fn(pred, gt)
        if eval_type not in ['seg', 'multiclass']:
            result1, result2, result3, result4 = metric_fn(pred, gt)
            val_metric1 += (result1 * pred.shape[0])
            val_metric2 += (result2 * pred.shape[0])
            val_metric3 += (result3 * pred.shape[0])
            val_metric4 += (result4 * pred.shape[0])
        cnt += pred.shape[0]
        
        if pbar is not None:
            pbar.update(1)
    val_metric1 = torch.tensor(val_metric1).cuda()
    val_metric2 = torch.tensor(val_metric2).cuda()
    val_metric3 = torch.tensor(val_metric3).cuda()
    val_metric4 = torch.tensor(val_metric4).cuda()
    cnt = torch.tensor(cnt).cuda()
    dist.all_reduce(val_metric1)
    dist.all_reduce(val_metric2)
    dist.all_reduce(val_metric3)
    dist.all_reduce(val_metric4)
    dist.all_reduce(cnt)
          
    # if pbar is not None:
    #     pbar.close()
    
    # return val_metric1.item()/cnt, val_metric2.item()/cnt, val_metric3.item()/cnt, val_metric4.item()/cnt, metric1, metric2, metric3, metric4
    oa = metric_seg.overallAccuracy()
    oa = np.around(oa,decimals=4)
    mIoU ,IoU= metric_seg.meanIntersectionOverUnion()
    mIoU = np.around(mIoU,decimals=4)
    IoU = np.around(IoU,decimals=4)
    p = metric_seg.precision()
    p = np.around(p,decimals=4)
    mp = np.nanmean(p)
    mp = np.around(mp,decimals=4)
    r = metric_seg.recall()
    r=np.around(r,decimals=4)
    mr = np.nanmean(r)
    mr = np.around(mr,decimals=4)
    f1 = (2*p*r) / (p + r)
    f1 = np.around(f1,decimals=4)
    mf1 = np.nanmean(f1)
    mf1 = np.around(mf1,decimals=4)
    normed_confusionMatrix = metric_seg.confusionMatrix / np.maximum(metric_seg.confusionMatrix.sum(axis=0), 1e-10)
    normed_confusionMatrix = np.around(normed_confusionMatrix, decimals=3)
    fwIOU = metric_seg.Frequency_Weighted_Intersection_over_Union()
    fwIOU= np.around(fwIOU,decimals=4)

    classes_list = config['train_dataset']['dataset']['args']['classes']
    if ignore_background:
        axis_labels=classes_list[:-1]
    else:
        axis_labels=classes_list


    IOU_row = ['IOU',mIoU]
    IOU_row.extend(IoU.tolist())
    Precision_row = ['Precision',mp]
    Precision_row.extend(p.tolist())
    Recall_row = ['Recall',mr]
    Recall_row.extend(r.tolist())
    F1_row = ['F1',mf1]
    F1_row.extend(f1.tolist())
    title_row = ['metrics','average']
    title_row.extend(axis_labels)
    OA_row = ['OA',oa]#,' ',' ',' ',' ']

    fwIOU_row = ['FWIOU', fwIOU]#,' ',' ',' ',' ']
    for i in range(len(axis_labels)):
        OA_row.append(' ')
        fwIOU_row.append(' ')

    table = PrettyTable(title_row)
    table.add_row(IOU_row)
    table.add_row(Precision_row)
    table.add_row(Recall_row)
    table.add_row(F1_row)
    table.add_row(OA_row)
    table.add_row(fwIOU_row)

    if eval_type in ['seg', 'multiclass']:
        return (
            float(mIoU),
            float(oa),
            float(mf1),
            float(fwIOU),
            metric1,
            metric2,
            metric3,
            metric4,
            table,
            normed_confusionMatrix
        )

    return (
        val_metric1.item() / cnt.item(),
        val_metric2.item() / cnt.item(),
        val_metric3.item() / cnt.item(),
        val_metric4.item() / cnt.item(),
        metric1,
        metric2,
        metric3,
        metric4,
        table,
        normed_confusionMatrix
    )


def prepare_training():
    
    model = models.make(config['model']).cuda()

    optimizer = utils.make_optimizer(
        model.parameters(), config['optimizer']
    )

    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        max_epoch,
        eta_min=config.get('lr_min')
    )

    epoch_start = 1
    # max_val_v is initialized/restored before training.
    max_val_v = -1e18 if config.get('eval_type') != 'ber' else 1e8

    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))

    return model, optimizer, epoch_start, lr_scheduler, max_val_v


def train(train_loader, model):
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    for batch in train_loader:
        inp = batch['inp']
        gt = batch['gt']

        model.module.optimizer.zero_grad()

        loss = model(inp, gt)

        loss.backward()

        model.module.optimizer.step()

        batch_loss = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, loss)
        loss_list.extend(batch_loss)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler, max_val_v = prepare_training()
    
    model.optimizer = optimizer
    
    # Scheduler is already created in prepare_training().

    model = model.cuda()

    ckpt = torch.load(config['sam_checkpoint'], map_location="cpu")
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
        
    new_state_dict = {}

    ref_state_dict = model.state_dict()

    if args.local_rank == 0:
        print(f"Loading custom checkpoint with 'detector.backbone' prefix...")

    for k, v in ckpt.items():

        if k.startswith("detector.backbone."):
            new_k = k.replace("detector.backbone.", "image_encoder.")

        elif "mask_decoder" in k:
            suffix = k.split("mask_decoder.")[-1]
            new_k = f"mask_decoder.{suffix}"
            
        elif "pe_layer" in k:
            suffix = k.split("pe_layer.")[-1]
            new_k = f"pe_layer.{suffix}"

        elif "no_mask_embed" in k:
            new_k = "no_mask_embed.weight"

        else:
            new_k = k

        if new_k in ref_state_dict:
            ref_shape = ref_state_dict[new_k].shape
            if v.shape != ref_shape:
                if args.local_rank == 0:
                    print(f"Warning: Skipping {new_k} due to shape mismatch. "
                          f"Ckpt: {v.shape} vs Model: {ref_shape}")
                continue

        if new_k:
            new_state_dict[new_k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)

    if args.local_rank == 0:
        print(f"\nLoad result: {len(msg.missing_keys)} missing keys.")
        if len(msg.missing_keys) > 0:
             print("Sample missing keys:", msg.missing_keys[:3])

    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    if args.local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    
    # resume
    
    resume_path = config.get('resume_path', None)

    if resume_path is not None and os.path.isfile(resume_path):
        if args.local_rank == 0:
            print(f"\nResuming full checkpoint from: {resume_path}")

        resume_ckpt = torch.load(resume_path, map_location="cpu")


        if isinstance(resume_ckpt, dict) and "model" in resume_ckpt:
            msg = model.load_state_dict(resume_ckpt["model"], strict=False)
        else:
            msg = model.load_state_dict(resume_ckpt, strict=False)

        if args.local_rank == 0:
            print(f"Resume missing keys: {len(msg.missing_keys)}")
            print(f"Resume unexpected keys: {len(msg.unexpected_keys)}")

        if isinstance(resume_ckpt, dict) and "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            if args.local_rank == 0:
                print("Optimizer state loaded.")

        if isinstance(resume_ckpt, dict) and "lr_scheduler" in resume_ckpt:
            lr_scheduler.load_state_dict(resume_ckpt["lr_scheduler"])
            if args.local_rank == 0:
                print("LR scheduler state loaded.")

        if isinstance(resume_ckpt, dict) and "epoch" in resume_ckpt:
            epoch_start = int(resume_ckpt["epoch"]) + 1
            if args.local_rank == 0:
                print(f"Resuming from epoch: {epoch_start}")

        if isinstance(resume_ckpt, dict) and "max_val_v" in resume_ckpt:
            max_val_v = resume_ckpt["max_val_v"]
            if args.local_rank == 0:
                print(f"Previous best validation value: {max_val_v}")

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
        
    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        
        train_loss_G = train(train_loader, model)
        lr_scheduler.step()

        if args.local_rank == 0:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)
            
            model_spec = config['model']
            model_spec['sd'] = model.module.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()
            save_checkpoint(model.module, optimizer, lr_scheduler, epoch, max_val_v, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            torch.cuda.empty_cache()
            result1, result2, result3, result4, metric1, metric2, metric3, metric4, table, normed_confusionMatrix = eval_psnr(
                val_loader, model, eval_type=config.get('eval_type')
            )

            if args.local_rank == 0:
                log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
                log_info.append('train G: loss={:.4f}'.format(train_loss_G))
                
                log_info.append('val: {}={:.4f}'.format(metric1, result1))
                writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                writer.add_scalars(metric2, {'val': result2}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric3, result3))
                writer.add_scalars(metric3, {'val': result3}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric4, result4))
                writer.add_scalars(metric4, {'val': result4}, epoch)

                if config['eval_type'] != 'ber':
                    if result1 > max_val_v:
                        max_val_v = result1
                        save_checkpoint(model.module, optimizer, lr_scheduler, epoch, max_val_v, save_path, 'best')
                else:
                    if result2 < max_val_v:
                        max_val_v = result2
                        save_checkpoint(model.module, optimizer, lr_scheduler, epoch, max_val_v, save_path, 'best')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                log(str(table))
                writer.flush()
            dist.barrier()  

def save_checkpoint(model, optimizer, lr_scheduler, epoch, max_val_v, save_path, name):
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "max_val_v": max_val_v,
    }

    torch.save(
        checkpoint,
        os.path.join(save_path, f"checkpoint_{name}.pth")
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    # parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--local-rank", type=int, default=0, help="")

    args = parser.parse_args()

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)
