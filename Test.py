import argparse
import csv
import os
import time

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms
from mmcv.runner import load_checkpoint

import matplotlib.pyplot as plt
import numpy as np
import cv2

from PIL import Image

from eval_iou import SegmentationMetric

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

def color_to_list(mask, palette=[ [1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,0,0]] ):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    #mask = mask.permute(1,2,0)
    mask = mask*255
    mask.int()
    semantic_map = np.zeros([1024,1024],dtype=np.int8)
    for i,colour in enumerate( palette):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map += class_map*int(i)


def onehot_to_mask(mask, palette=[ [1,0,0], [0,1,0], [0,0,1], [1,1,0],[0,0,0]]):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1,2,0).numpy()
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
  
    return x

def onehot_to_index_label(mask):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1,2,0).numpy()
    x = np.argmax(mask, axis=-1)
    
    return x


def de_normalize(image,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]):
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    image=image*std+mean 
    image=image.numpy().transpose(1,2,0)  
    image=np.around(image * 255)  
    image=np.array(image, dtype=np.uint8)  
    return image


def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, config=None,config_name = None,
              verbose=False):
    model.eval()
    dataset_args = config['test_dataset']['dataset']['args']
    class_num = config['model']['args']['num_classes']
    color_palette = config['test_dataset']['dataset']['args']['palette']
    # ignore_background =  config['test_dataset']['dataset']['args']['ignore_bg']
    ignore_background = config['test_dataset']['dataset']['args'].get('ignore_bg', False)
    print("Number of classes:", class_num)
    print("Ignore background:", ignore_background)

    #work_dir = config['work_dir'].split('/')[-1]
    work_dir = config_name
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

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
    elif eval_type == 'seg':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
        
        metric_seg = SegmentationMetric(class_num,  ignore_background)


    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')

    id = 0

    # Inference-time tracking. CUDA operations are asynchronous, so
    # torch.cuda.synchronize() is required before and after timing.
    inference_times_ms = []
    warmup_done = False
    warmup_iterations = int(config.get('inference_warmup_iterations', 5))

    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']
        batch_size = inp.shape[0]

        
        if not warmup_done and warmup_iterations > 0:
            with torch.no_grad():
                for _ in range(warmup_iterations):
                    _ = model.infer(inp)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            warmup_done = True

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_start = time.perf_counter()

        output_masks = model.infer(inp)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_elapsed_ms = (time.perf_counter() - inference_start) * 1000.0
        per_image_time_ms = inference_elapsed_ms / batch_size
        inference_times_ms.extend([per_image_time_ms] * batch_size)

        # pred is used by the existing metric functions.
        pred = torch.sigmoid(output_masks)

        for i in range(len(output_masks)):
            #print(len(batch_pred))
            output_masks[i]=output_masks[i].to('cpu') 
            pred[i]=pred[i].to('cpu')

        
        output_mask = output_masks[0].cpu().detach()
        binary_mask = onehot_to_mask(output_mask,palette = color_palette)
        mask_index_label = onehot_to_index_label(output_mask).flatten()


        mask_dir = os.path.join('./save', work_dir, 'mask')
        gt_dir = os.path.join('./save', work_dir, 'gt')
        gt_img_dir = os.path.join('./save', work_dir, 'gt_img')
        overlay_dir = os.path.join('./save', work_dir, 'overlay_mask') 
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(gt_img_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        output_path = os.path.join(mask_dir, f'{id}.png')
        Image.fromarray(
            np.uint8(binary_mask)
        ).convert('RGB').save(output_path)
        gt_mask = batch['gt'][0].cpu().detach()
        
        gt_mask_rgb = onehot_to_mask(
            gt_mask,
            palette=color_palette
        )
        
        gt_index_label = onehot_to_index_label(
            gt_mask
        ).flatten()
        
        gt_save_path = os.path.join(gt_dir, f'{id}.png')
        
        Image.fromarray(
            np.uint8(gt_mask_rgb)
        ).convert('RGB').save(gt_save_path)
        
        gt_img = batch['inp'][0].cpu().detach()
        ori_gt_img = de_normalize(gt_img)
        img_save_path = os.path.join(gt_img_dir, f'{id}.jpg')
        Image.fromarray(
            np.uint8(ori_gt_img)
        ).convert('RGB').save(img_save_path)
        
        overlay_mask_path = os.path.join(
            overlay_dir,
            f'{id}.jpg'
        )
        
        original_pil = Image.fromarray(
            np.uint8(ori_gt_img)
        ).convert('RGB')
        
        prediction_pil = Image.fromarray(
            np.uint8(binary_mask)
        ).convert('RGB')
        
        overlay = Image.blend(
            original_pil,
            prediction_pil,
            alpha=0.5
        )
        overlay.save(overlay_mask_path)
        id += 1

        
        if eval_type == 'seg':
            metric_seg.addBatch(mask_index_label,gt_index_label)



        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])

        val_metric1.add(result1.item(), inp.shape[0])
        val_metric2.add(result2.item(), inp.shape[0])
        val_metric3.add(result3.item(), inp.shape[0])
        val_metric4.add(result4.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
            pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
            pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
            pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))

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
    normed_confusionMatrix = metric_seg.confusionMatrix / metric_seg.confusionMatrix.sum(axis=0)
    normed_confusionMatrix = np.around(normed_confusionMatrix, decimals=2)
    
    confusion_matrix = metric_seg.confusionMatrix.astype(np.float64)

    gt_pixel_count = confusion_matrix.sum(axis=1)
    pred_pixel_count = confusion_matrix.sum(axis=0)
    true_positive = np.diag(confusion_matrix)
    total_evaluated_pixels = confusion_matrix.sum()

    union = gt_pixel_count + pred_pixel_count - true_positive

    class_iou_precise = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan, dtype=np.float64),
        where=union > 0
    )

    class_frequency = np.divide(
        gt_pixel_count,
        total_evaluated_pixels,
        out=np.zeros_like(gt_pixel_count, dtype=np.float64),
        where=total_evaluated_pixels > 0
    )

    # class-specific contribution to the overall fwIoU
    class_fwiou = class_frequency * np.nan_to_num(
        class_iou_precise,
        nan=0.0
    )

    valid_classes = gt_pixel_count > 0
    fwIOU = float(class_fwiou[valid_classes].sum())

    class_frequency = np.around(class_frequency, decimals=6)
    class_fwiou = np.around(class_fwiou, decimals=6)
    fwIOU = np.around(fwIOU, decimals=4)

    # Summarise inference time after all batches have been processed.
    inference_times_array = np.asarray(inference_times_ms, dtype=np.float64)
    if inference_times_array.size > 0:
        timing_summary = {
            'num_images': int(inference_times_array.size),
            'total_inference_time_s': float(inference_times_array.sum() / 1000.0),
            'mean_inference_time_ms_per_image': float(inference_times_array.mean()),
            'std_inference_time_ms_per_image': float(inference_times_array.std()),
            'median_inference_time_ms_per_image': float(np.median(inference_times_array)),
            'min_inference_time_ms_per_image': float(inference_times_array.min()),
            'max_inference_time_ms_per_image': float(inference_times_array.max()),
            'throughput_images_per_second': float(1000.0 / inference_times_array.mean()),
        }
    else:
        timing_summary = {
            'num_images': 0,
            'total_inference_time_s': float('nan'),
            'mean_inference_time_ms_per_image': float('nan'),
            'std_inference_time_ms_per_image': float('nan'),
            'median_inference_time_ms_per_image': float('nan'),
            'min_inference_time_ms_per_image': float('nan'),
            'max_inference_time_ms_per_image': float('nan'),
            'throughput_images_per_second': float('nan'),
        }
   
    classes_list = config['test_dataset']['dataset']['args']['classes']

    if ignore_background:
        axis_labels=classes_list[:-1] 
    else: 
        axis_labels=classes_list
    axis_labels = ['..','..','..','..'] #,'background']
    plt.figure()#figsize=(8, 8))
    sns.heatmap(normed_confusionMatrix, annot=True, cmap='Greens',yticklabels=axis_labels,xticklabels=axis_labels)
    plt.tight_layout()
    #plt.ylim(0, 4)
    
    plt.ylabel('Predictions')
    plt.xlabel('Ground truths')
    plt.yticks(np.array(range(0,5)), axis_labels)
    plt.tight_layout()

    output_root = os.path.join('./save', work_dir)
    os.makedirs(output_root, exist_ok=True)
    plt.savefig(os.path.join(output_root, 'confusionmatrix.jpg'))
    plt.close()
      
    print('self.confusionMatrix:')
    print(normed_confusionMatrix)
    print('OA:',oa)

    IOU_row = ['IOU',mIoU]
    IOU_row.extend(IoU.tolist())
    Precision_row = ['Precision',mp]
    Precision_row.extend(p.tolist())
    Recall_row = ['Recall',mr]
    Recall_row.extend(r.tolist())
    F1_row = ['F1',mf1]
    F1_row.extend(f1.tolist())

    Frequency_row = ['Class frequency', '']
    Frequency_row.extend(class_frequency.tolist())

    Class_fwIOU_row = ['Class-wise fwIoU contribution', fwIOU]
    Class_fwIOU_row.extend(class_fwiou.tolist())

    title_row = ['metrics','average']
    title_row.extend(axis_labels)
    OA_row = ['OA',oa]#,' ',' ',' ',' ']
    #OA_row.extend(' '*5)
    fwIOU_row = ['FWIOU', fwIOU]#,' ',' ',' ',' ']

    for i in range(len(axis_labels)):
        OA_row.append(' ')
        fwIOU_row.append(' ')

    table = PrettyTable(title_row)
    table.add_row(IOU_row)
    table.add_row(Precision_row)
    table.add_row(Recall_row)
    table.add_row(F1_row)
    table.add_row(Frequency_row)
    table.add_row(Class_fwIOU_row)
    table.add_row(OA_row)
    table.add_row(fwIOU_row)

    # ------------------------------------------------------------------
    # Save metrics and inference times to CSV files.
    # ------------------------------------------------------------------
    overall_metrics_path = os.path.join(output_root, 'overall_metrics.csv')
    overall_metrics = {
        'config_name': work_dir,
        'OA': float(oa),
        'mIoU': float(mIoU),
        'mean_precision': float(mp),
        'mean_recall': float(mr),
        'mean_F1': float(mf1),
        'fwIoU': float(fwIOU),
        metric1: float(val_metric1.item()),
        metric2: float(val_metric2.item()),
        metric3: float(val_metric3.item()),
        metric4: float(val_metric4.item()),
        **timing_summary,
    }
    with open(overall_metrics_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(overall_metrics.keys()))
        writer.writeheader()
        writer.writerow(overall_metrics)

    class_metrics_path = os.path.join(output_root, 'per_class_metrics.csv')
    with open(class_metrics_path, 'w', newline='') as csv_file:
        fieldnames = [
            'class_index',
            'class_name',
            'ground_truth_pixel_count',
            'class_frequency',
            'IoU',
            'class_wise_fwIoU_contribution',
            'precision',
            'recall',
            'F1'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for class_index, class_name in enumerate(axis_labels):
            writer.writerow({
                'class_index': class_index,
                'class_name': class_name,
                'ground_truth_pixel_count': int(gt_pixel_count[class_index]),
                'class_frequency': float(class_frequency[class_index]),
                'IoU': float(IoU[class_index]),
                'class_wise_fwIoU_contribution': float(class_fwiou[class_index]),
                'precision': float(p[class_index]),
                'recall': float(r[class_index]),
                'F1': float(f1[class_index]),
            })

    inference_times_path = os.path.join(output_root, 'inference_times.csv')
    with open(inference_times_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=['image_index', 'inference_time_ms', 'inference_time_s']
        )
        writer.writeheader()
        for image_index, elapsed_ms in enumerate(inference_times_ms):
            writer.writerow({
                'image_index': image_index,
                'inference_time_ms': float(elapsed_ms),
                'inference_time_s': float(elapsed_ms / 1000.0),
            })

    print(f'Overall metrics saved to: {overall_metrics_path}')
    print(f'Per-class metrics saved to: {class_metrics_path}')

    print('\nClass-wise fwIoU contributions:')
    for class_index, class_name in enumerate(axis_labels):
        print(
            f'  {class_name}: '
            f'frequency={class_frequency[class_index]:.6f}, '
            f'IoU={IoU[class_index]:.4f}, '
            f'fwIoU contribution={class_fwiou[class_index]:.6f}'
        )
    print(f'Overall fwIoU (sum of contributions): {fwIOU:.4f}')

    print(f'Per-image inference times saved to: {inference_times_path}')
    print(
        'Mean inference time: '
        f"{timing_summary['mean_inference_time_ms_per_image']:.4f} ms/image "
        f"({timing_summary['throughput_images_per_second']:.4f} images/s)"
    )

    return (
        val_metric1.item(), val_metric2.item(), val_metric3.item(),
        val_metric4.item(), table, timing_summary
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    config_name = args.config.split('/')[-1].split('.yaml')[0]
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8,shuffle= False)

    # ------------------------------------------------------------
    # Build the model on CPU first
    # ------------------------------------------------------------
    model = models.make(config['model'])

    # ------------------------------------------------------------
    # Load checkpoint safely
    # ------------------------------------------------------------
    checkpoint = torch.load(args.model, map_location='cpu')

    print("\nCheckpoint type:", type(checkpoint))

    if isinstance(checkpoint, dict):
        print("Top-level checkpoint keys:", list(checkpoint.keys())[:30])

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("Using checkpoint['model']")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Using checkpoint['state_dict']")
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("Using checkpoint['model_state_dict']")
    else:
        state_dict = checkpoint
        print("Using checkpoint as a plain state_dict")

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected state_dict to be a dictionary, received {type(state_dict)}"
        )

    # Remove prefixes introduced by DDP or wrappers.
    clean_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith('module.'):
            new_key = new_key[len('module.'):]

        if new_key.startswith('model.'):
            candidate_key = new_key[len('model.'):]
            if candidate_key in model.state_dict():
                new_key = candidate_key

        clean_state_dict[new_key] = value

    load_result = model.load_state_dict(clean_state_dict, strict=False)

    print("\nCheckpoint loading summary")
    print("--------------------------")
    print("Missing keys:", len(load_result.missing_keys))
    print("Unexpected keys:", len(load_result.unexpected_keys))

    if load_result.missing_keys:
        print("\nFirst missing keys:")
        for key in load_result.missing_keys[:30]:
            print("  ", key)

    if load_result.unexpected_keys:
        print("\nFirst unexpected keys:")
        for key in load_result.unexpected_keys[:30]:
            print("  ", key)

    important_terms = (
        'adapter',
        'multiclass_head',
        'mask_decoder',
        'prompt_generator',
    )

    important_missing = [
        key for key in load_result.missing_keys
        if any(term in key for term in important_terms)
    ]

    if important_missing:
        print("\nWARNING: Important model keys are missing:")
        for key in important_missing[:50]:
            print("  ", key)

        print(
            "\nTesting can continue, but the results may be invalid if "
            "these missing layers were trained."
        )

    model = model.cuda()
    model.eval()

    with torch.no_grad():
        metric1, metric2, metric3, metric4, seg_eval_table, timing_summary = eval_psnr(loader, model,
                                                   data_norm=config.get('data_norm'),
                                                   eval_type=config.get('eval_type'),
                                                   eval_bsize=config.get('eval_bsize'),
                                                   config=config,
                                                   config_name=config_name,
                                                   verbose=True)
    print('metric1: {:.4f}'.format(metric1))
    print('metric2: {:.4f}'.format(metric2))
    print('metric3: {:.4f}'.format(metric3))
    print('metric4: {:.4f}'.format(metric4))
    print(seg_eval_table)