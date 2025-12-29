# import os
# import time
# import numpy as np
# import nibabel as nib
# import torch
# import config_2d
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from dataclasses import dataclass
# from typing import Callable, Optional, List
# import argparse
# import seg_metrics.seg_metrics as sg
# from models.Decompose import Decompose
# from models.Decompose_wdcp import Decompose_wdcp
# from joblib import Parallel, delayed
# import multiprocessing
# # Add other model imports as needed
# # from networks_mr.net_factory import net_factory
# # from core.networks import MTNet
# # from NetModel.Unet_2d import UNet2D

# # Configuration class for centralized settings
# @dataclass
# class Config:
#     patch_size_w: int = config_2d.PATCH_SIZE_W
#     patch_size_h: int = config_2d.PATCH_SIZE_H
#     batch_size: int = 1#config_2d.BATCH_SIZE
#     num_epochs: int = config_2d.NUM_EPOCHS
#     num_classes: int = config_2d.NUM_CLASSES
#     volume_rows: int = config_2d.VOLUME_ROWS
#     volume_cols: int = config_2d.VOLUME_COLS
#     volume_depth: int = config_2d.VOLUME_DEPS
#     test_imgs_path: str = config_2d.test_imgs_path
#     test_extraction_step: int = config_2d.TEST_EXTRACTION_STEP
#     model_path: str = "/mnt/disk1/tmi_review2025/exp/hcp/hcp20_4096_ours_new_aug_3_tresh_0.05_w_dcp_1_w_o_cse/fold_5_best_model.pth"
#     pred_dir: str = "/mnt/disk1/tmi_review2025/hcp_20_aug_3_tresh_0.05_w_dcp_1_w_o_cse/fold5"
#     test_data_root: str = "/mnt/disk1/new_code_ssl/ON_mydata3/test_data/"
#     model_name: str = "Decompose"
#     in_channels: int = 1
#     labels = [1]  # Exclude background (0) for metrics
#     spacing: tuple = (1.25, 1.25, 1.25)  # Voxel spacing for HCP
#     metrics = ["dice", "jaccard", "hd95", "msd"]  # Metrics to compute
#     evaluate: bool = True  # Flag to enable/disable evaluation
#     num_workers: int = 4  # Use multi-threading for data loading
#     pin_memory: bool = True  # Faster GPU data transfer

# # Model factory to dynamically select model
# def get_model(model_name: str, in_channels: int, num_classes: int, device: torch.device) -> torch.nn.Module:
#     model_map = {
#         "Decompose": lambda: Decompose(channel=in_channels).to(device),
#         "Decompose_wdcp": lambda: Decompose_wdcp(channel=in_channels).to(device),
#         # "MTNet": lambda: MTNet("resnet50", num_classes=num_classes, use_group_norm=True).to(device),
#         # "UNet2D": lambda: UNet2D(in_channels=in_channels, n_classes=num_classes).to(device),
#     }
#     if model_name not in model_map:
#         raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_map.keys())}")
#     return model_map[model_name]()

# # Data transforms
# def get_transforms() -> tuple[Callable, Callable]:
#     x_transforms = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     y_transforms = transforms.ToTensor()
#     return x_transforms, y_transforms

# # Dataset and dataloader creation
# def create_dataloader(config: Config, test_input_path: str) -> DataLoader:
#     from dataloader import CN_MyTestDataset  # Import here to avoid circular imports
#     x_transforms, y_transforms = get_transforms()
#     dataset = CN_MyTestDataset(
#         os.path.join(test_input_path, "x_t1_data/"),
#         os.path.join(test_input_path, "x_fa_data/"),
#         os.path.join(test_input_path, "y_data/"),
#         x_transform=x_transforms,
#         y_transform=y_transforms
#     )
#     return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)

# # Prediction function (store predictions in memory)
# def predict(config: Config, test_input_path: str, predict_dir: str, test_num: str) -> List[np.ndarray]:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = get_model(config.model_name, config.in_channels, config.num_classes, device)
#     checkpoint = torch.load(config.model_path, map_location="cpu")
#     model.load_state_dict(checkpoint)
#     model.eval()

#     dataloader = create_dataloader(config, test_input_path)
#     predictions = []
#     #affine = None

#     with torch.no_grad():
#         for x1, x2, y, _ in dataloader:
#             inputs1, inputs2 = x1.to(device), x2.to(device)
#             if config.model_name == "Decompose":
#                 outputs = model(inputs1, inputs2)[0]
#             else:
#                 outputs = model(inputs1, inputs2)[0]
#             outputs = F.softmax(outputs, dim=1)
#             pred = torch.max(outputs, 1)[1].cpu().numpy().astype(float)

#             # Handle batch dimension
#             for i in range(pred.shape[0]):
#                 p = pred[i]
#                 if p.ndim == 3:
#                     if p.shape[0] == 1:
#                         p = p[0]
#                     else:
#                         raise ValueError(f"Expected 2D prediction, got shape {p.shape}")
#                 if p.shape != (config.volume_rows, config.volume_cols):
#                     raise ValueError(f"Prediction shape {p.shape} does not match expected ({config.volume_rows}, {config.volume_cols})")
#                 predictions.append(p)

#             #if not affine:
#             test_t1_path = os.path.join(test_input_path, "x_t1_data", "x_t1-data_0.nii.gz")
#             affine = nib.load(test_t1_path).affine

#     # Save predictions to disk for compatibility
#     for i, pred in enumerate(predictions):
#         pred_nii = nib.Nifti1Image(pred, affine)
#         pred_path = os.path.join(predict_dir, f"pre_{i}.nii.gz")
#         nib.save(pred_nii, pred_path)

#     return predictions, affine

# # Combine predictions into final segmentation
# def combine_predictions(config: Config, predict_dir: str, test_num: str, predictions: List[np.ndarray], affine: np.ndarray) -> None:
#     final_seg = np.zeros((config.volume_rows, config.volume_cols, config.volume_depth))

#     img_path = os.path.join(config.test_imgs_path, test_num, f"{test_num}_ON-T1.nii.gz")
#     mask_path = os.path.join(config.test_imgs_path, test_num, f"{test_num}_ON-mask.nii.gz")
    
#     try:
#         img = nib.load(img_path)
#         mask = nib.load(mask_path)
#     except FileNotFoundError as e:
#         print(f"Error: Could not load image or mask for {test_num}: {e}")
#         return

#     img_data = img.get_fdata()
#     mask_data = np.squeeze(mask.get_fdata())

#     # Precompute valid slices
#     valid_slices = [i for i in range(config.volume_depth) if np.count_nonzero(mask_data[:, :, i]) and np.count_nonzero(img_data[:, :, i])]
    
#     # Assign predictions to valid slices
#     for i, slice_idx in enumerate(valid_slices[:len(predictions)]):
#         pred_data = predictions[i]
#         if pred_data.shape != (config.volume_rows, config.volume_cols):
#             print(f"Warning: Skipping invalid prediction shape {pred_data.shape} for slice {slice_idx}")
#             continue
#         final_seg[:, :, slice_idx] = pred_data

#     final_nii = nib.Nifti1Image(final_seg, affine)
#     final_path = os.path.join(predict_dir, "pre_final-label.nii.gz")
#     nib.save(final_nii, final_path)

# # Metric evaluation function
# def evaluate_metrics(config: Config, pred_dir: str) -> None:
#     test_dirs = os.listdir(config.test_imgs_path)
#     csv_file = os.path.join(pred_dir, "metrics.csv")
#     metric_results = {metric: [] for metric in config.metrics}

#     # Initialize output files
#     for metric in config.metrics:
#         with open(os.path.join(pred_dir, f"{metric.upper()}.txt"), "a+") as f:
#             f.write("epoch\n")

#     def process_test_case(test_num):
#         label_path = os.path.join(config.test_imgs_path, test_num, f"{test_num}_ON-label.nii.gz")
#         pred_path = os.path.join(pred_dir, f"test_result_{test_num}", "pre_final-label.nii.gz")
#         results = {}
#         try:
#             metrics = sg.write_metrics(
#                 labels=config.labels,
#                 gdth_path=label_path,
#                 pred_path=pred_path,
#                 csv_file=csv_file,
#                 spacing=config.spacing,
#                 metrics=config.metrics
#             )
#             for metric in config.metrics:
#                 results[metric] = metrics[0][metric][0]
#         except Exception as e:
#             print(f"Error evaluating {test_num}: {e}")
#             for metric in config.metrics:
#                 results[metric] = np.nan
#         return results

#     # Parallelize evaluation
#     results = Parallel(n_jobs=config.num_workers)(
#         delayed(process_test_case)(test_num) for test_num in test_dirs
#     )

#     # Collect and save results
#     for test_result in results:
#         for metric in config.metrics:
#             metric_results[metric].append(test_result[metric])
#             with open(os.path.join(pred_dir, f"{metric.upper()}.txt"), "a+") as f:
#                 f.write(f"{test_result[metric]:.5f}\t" if not np.isnan(test_result[metric]) else "NaN\t")

#     # Compute and save mean and standard deviation
#     for metric in config.metrics:
#         values = np.array(metric_results[metric])
#         mean_val = np.nanmean(values)
#         std_val = np.nanstd(values)
#         print(f"{metric.upper()} Mean: {mean_val:.5f}, Std: {std_val:.5f}")
#         with open(os.path.join(pred_dir, f"{metric.upper()}.txt"), "a+") as f:
#             f.write(f"{mean_val:.5f}\t{std_val:.5f}\n")

# # Main execution
# def main():
#     parser = argparse.ArgumentParser(description="Prediction and evaluation for segmentation model", epilog="by Alou")
#     parser.add_argument("--pred_path", type=str, default=None, help="Prediction directory")
#     parser.add_argument("--evaluate", action="store_true", help="Run evaluation after prediction")
#     args = parser.parse_args()

#     config = Config()
#     config.evaluate = args.evaluate or config.evaluate
#     pred_dir = args.pred_path or config.pred_dir
#     os.makedirs(pred_dir, exist_ok=True)

#     start_time = time.time()
#     test_dirs = os.listdir(config.test_imgs_path)

#     # Prediction phase
#     for test_num in test_dirs:
#         test_name = f"test_{test_num}"
#         test_pre_name = f"test_result_{test_num}"
#         test_input_path = os.path.join(config.test_data_root, test_name)
#         predict_dir = os.path.join(pred_dir, test_pre_name)
#         os.makedirs(predict_dir, exist_ok=True)
#         predictions, affine = predict(config, test_input_path, predict_dir, test_num)
#         combine_predictions(config, predict_dir, test_num, predictions, affine)

#     print(f"2D inference time: {time.time() - start_time:.3f} seconds")

#     # Evaluation phase
#     if config.evaluate:
#         start_time = time.time()
#         evaluate_metrics(config, pred_dir)
#         print(f"Evaluation time: {time.time() - start_time:.3f} seconds")

# if __name__ == "__main__":
#     main()

import os
import time
import numpy as np
import nibabel as nib
import torch
import config_2d
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple
import argparse
import seg_metrics.seg_metrics as sg
from models.Decompose import Decompose
from models.Decompose_wdcp import Decompose_wdcp
from models.SAnet1 import t1safaFuseUNet1
from joblib import Parallel, delayed
import multiprocessing
# Add other model imports as needed
# from networks_mr.net_factory import net_factory
# from core.networks import MTNet
# from NetModel.Unet_2d import UNet2D

# Configuration class for centralized settings
@dataclass
class Config:
    patch_size_w: int = config_2d.PATCH_SIZE_W
    patch_size_h: int = config_2d.PATCH_SIZE_H
    batch_size: int = 1#config_2d.BATCH_SIZE
    num_epochs: int = config_2d.NUM_EPOCHS
    num_classes: int = config_2d.NUM_CLASSES
    volume_rows: int = config_2d.VOLUME_ROWS
    volume_cols: int = config_2d.VOLUME_COLS
    volume_depth: int = config_2d.VOLUME_DEPS
    test_imgs_path: str = config_2d.test_imgs_path
    test_extraction_step: int = config_2d.TEST_EXTRACTION_STEP
    model_path: str = "/mnt/disk1/tmi_review2025/exp/hcp/mdm25_1_4096_ours_new_aug_3_tresh_0.05_w_dcp_1_nu/fold_5_best_model.pth"
    pred_dir: str = "mdm25_1_4096_ours_new_aug_3_tresh_0.05_w_dcp_1_nu/fold5"
    test_data_root: str = "/mnt/disk1/new_code_ssl/ON_mydata3/test_data"
    # model_path: str = "/mnt/disk1/tmi_review2025/exp/hcp/mmd25_512_ours_new_aug_3_tresh_0.05_w_dcp_1/fold_5_best_model.pth"
    # pred_dir: str = "/mnt/disk1/tmi_review2025/mmd25_512_ours_new_aug_3_tresh_0.05_w_dcp_1_test/fold5"
    # test_data_root: str = "/mnt/disk1/tmi_review2025/ON_MMD/test_data"
    model_name: str = "Decompose"
    in_channels: int = 1
    labels = [1]  # Exclude background (0) for metrics
    spacing: tuple = (1.25, 1.25, 1.25)  # Voxel spacing for HCP and MMD
    #spacing: tuple = (1.2, 1., 1.)  # Voxel spacing for MDM
    metrics = ["dice", "jaccard", "hd95", "msd"]  # Metrics to compute
    evaluate: bool = True  # Flag to enable/disable evaluation
    num_workers: int = 4  # Use multi-threading for data loading
    pin_memory: bool = True  # Faster GPU data transfer

# Model factory to dynamically select model
def get_model(model_name: str, in_channels: int, num_classes: int, device: torch.device) -> torch.nn.Module:
    model_map = {
        "Decompose": lambda: Decompose(channel=in_channels).to(device),
        "Decompose_wdcp": lambda: Decompose_wdcp(channel=in_channels).to(device),
        "SAnet": lambda: t1safaFuseUNet1(1, 1).to(device),
        # "UNet2D": lambda: UNet2D(in_channels=in_channels, n_classes=num_classes).to(device),
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_map.keys())}")
    return model_map[model_name]()

# Data transforms
def get_transforms() -> tuple[Callable, Callable]:
    x_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    y_transforms = transforms.ToTensor()
    return x_transforms, y_transforms

# Dataset and dataloader creation
def create_dataloader(config: Config, test_input_path: str) -> DataLoader:
    from dataloader import CN_MyTestDataset  # Import here to avoid circular imports
    x_transforms, y_transforms = get_transforms()
    dataset = CN_MyTestDataset(
        os.path.join(test_input_path, "x_t1_data/"),
        os.path.join(test_input_path, "x_fa_data/"),
        os.path.join(test_input_path, "y_data/"),
        x_transform=x_transforms,
        y_transform=y_transforms
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)

# Prediction function (store predictions in memory)
def predict(config: Config, test_input_path: str, predict_dir: str, test_num: str) -> Tuple[List[np.ndarray], np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config.model_name, config.in_channels, config.num_classes, device)
    #model = torch.nn.DataParallel(model) # Use DataParallel for multi-GPU for supervise method
    checkpoint = torch.load(config.model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    dataloader = create_dataloader(config, test_input_path)
    predictions = []
    test_save_path = predict_dir  # Directory to save outputs
    os.makedirs(test_save_path, exist_ok=True)  # Ensure directory exists

    # Use identity affine transformation as in the example
    affine = np.eye(4)

    with torch.no_grad():
        for ith, (x1, x2, y, _) in enumerate(dataloader):
            inputs1, inputs2 = x1.to(device), x2.to(device)
            if config.model_name == "Decompose":
                outputs = model(inputs1, inputs2)[0]
            elif config.model_name == "Decompose_wdcp":
                outputs = model(inputs1, inputs2)[0]
            else:
                outputs = model(inputs1, inputs2)
            score_map = F.softmax(outputs, dim=1)  # Softmax probabilities
            pred = torch.max(score_map, 1)[1].cpu().numpy().astype(np.float32)  # Predicted labels

            # Handle batch dimension
            for i in range(pred.shape[0]):
                p = pred[i]
                s = score_map[i].cpu().numpy().astype(np.float32)  # Score map for this sample
                img = x1[i].cpu().numpy().astype(np.float32)  # Input image (x1)
                gt = y[i].cpu().numpy().astype(np.float32)  # Ground truth label

                # Handle dimensions to match example
                if p.ndim == 3:
                    if p.shape[0] == 1:
                        p = p[0]  # Take first channel for prediction
                        s = s[0]  # Take first channel for score map
                        img = img[0]  # Take first channel for input image
                        gt = gt[0]  # Take first channel for ground truth
                    else:
                        raise ValueError(f"Expected 2D prediction, got shape {p.shape}")
                if p.shape != (config.volume_rows, config.volume_cols):
                    raise ValueError(f"Prediction shape {p.shape} does not match expected ({config.volume_rows}, {config.volume_cols})")

                # Apply [:] to mimic example's slicing
                p = p[:]
                s = s[:]
                img = img[:]
                gt = gt[:]

                # Save predictions, scores, image, and ground truth
                nib.save(nib.Nifti1Image(p, affine), os.path.join(test_save_path, f"{ith:02d}_pred.nii.gz"))
                nib.save(nib.Nifti1Image(s, affine), os.path.join(test_save_path, f"{ith:02d}_scores.nii.gz"))
                nib.save(nib.Nifti1Image(img, affine), os.path.join(test_save_path, f"{ith:02d}_img.nii.gz"))
                nib.save(nib.Nifti1Image(gt, affine), os.path.join(test_save_path, f"{ith:02d}_gt.nii.gz"))

                predictions.append(p)

    return predictions, affine

def combine_predictions(config: Config, predict_dir: str, test_num: str, predictions: List[np.ndarray], affine: np.ndarray) -> None:
    final_seg = np.zeros((config.volume_rows, config.volume_cols, config.volume_depth), dtype=np.float32)

    img_path = os.path.join(config.test_imgs_path, test_num, f"{test_num}_ON-T1.nii.gz")
    mask_path = os.path.join(config.test_imgs_path, test_num, f"{test_num}_ON-mask.nii.gz")
    
    try:
        img = nib.load(img_path)
        mask = nib.load(mask_path)
    except FileNotFoundError as e:
        print(f"Error: Could not load image or mask for {test_num}: {e}")
        return

    img_data = img.get_fdata()
    mask_data = np.squeeze(mask.get_fdata())

    # Precompute valid slices
    valid_slices = [i for i in range(config.volume_depth) if np.count_nonzero(mask_data[:, :, i]) and np.count_nonzero(img_data[:, :, i])]
    
    # Assign predictions to valid slices
    for i, slice_idx in enumerate(valid_slices[:len(predictions)]):
        pred_data = predictions[i]
        if pred_data.shape != (config.volume_rows, config.volume_cols):
            print(f"Warning: Skipping invalid prediction shape {pred_data.shape} for slice {slice_idx}")
            continue
        final_seg[:, :, slice_idx] = pred_data

    # Save final segmentation with the same affine (np.eye(4))
    final_nii = nib.Nifti1Image(final_seg, affine)
    final_path = os.path.join(predict_dir, "pre_final-label.nii.gz")
    nib.save(final_nii, final_path)

# Metric evaluation function
def evaluate_metrics(config: Config, pred_dir: str) -> None:
    test_dirs = os.listdir(config.test_imgs_path)
    csv_file = os.path.join(pred_dir, "metrics.csv")
    metric_results = {metric: [] for metric in config.metrics}

    # Initialize output files
    for metric in config.metrics:
        with open(os.path.join(pred_dir, f"{metric.upper()}.txt"), "a+") as f:
            f.write("epoch\n")

    def process_test_case(test_num):
        label_path = os.path.join(config.test_imgs_path, test_num, f"{test_num}_ON-label.nii.gz")
        # label_path = os.path.join(config.test_imgs_path, test_num, f"{test_num}_label.nii.gz")
        pred_path = os.path.join(pred_dir, f"test_result_{test_num}", "pre_final-label.nii.gz")
        results = {}
        try:
            metrics = sg.write_metrics(
                labels=config.labels,
                gdth_path=label_path,
                pred_path=pred_path,
                csv_file=csv_file,
                spacing=config.spacing,
                metrics=config.metrics
            )
            for metric in config.metrics:
                results[metric] = metrics[0][metric][0]
        except Exception as e:
            print(f"Error evaluating {test_num}: {e}")
            for metric in config.metrics:
                results[metric] = np.nan
        return results

    # Parallelize evaluation
    results = Parallel(n_jobs=config.num_workers)(
        delayed(process_test_case)(test_num) for test_num in test_dirs
    )

    # Collect and save results
    for test_result in results:
        for metric in config.metrics:
            metric_results[metric].append(test_result[metric])
            with open(os.path.join(pred_dir, f"{metric.upper()}.txt"), "a+") as f:
                f.write(f"{test_result[metric]:.5f}\t" if not np.isnan(test_result[metric]) else "NaN\t")

    # Compute and save mean and standard deviation
    for metric in config.metrics:
        values = np.array(metric_results[metric])
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        print(f"{metric.upper()} Mean: {mean_val:.5f}, Std: {std_val:.5f}")
        with open(os.path.join(pred_dir, f"{metric.upper()}.txt"), "a+") as f:
            f.write(f"{mean_val:.5f}\t{std_val:.5f}\n")

# Main execution
def main():
    parser = argparse.ArgumentParser(description="Prediction and evaluation for segmentation model", epilog="by Alou")
    parser.add_argument("--pred_path", type=str, default=None, help="Prediction directory")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after prediction")
    args = parser.parse_args()

    config = Config()
    config.evaluate = args.evaluate or config.evaluate
    pred_dir = args.pred_path or config.pred_dir
    os.makedirs(pred_dir, exist_ok=True)

    start_time = time.time()
    test_dirs = os.listdir(config.test_imgs_path)

    # Prediction phase
    for test_num in test_dirs:
        test_name = f"test_{test_num}"
        test_pre_name = f"test_result_{test_num}"
        test_input_path = os.path.join(config.test_data_root, test_name)
        predict_dir = os.path.join(pred_dir, test_pre_name)
        os.makedirs(predict_dir, exist_ok=True)
        predictions, affine = predict(config, test_input_path, predict_dir, test_num)
        combine_predictions(config, predict_dir, test_num, predictions, affine)

    print(f"2D inference time: {time.time() - start_time:.3f} seconds")

    # Evaluation phase
    if config.evaluate:
        start_time = time.time()
        evaluate_metrics(config, pred_dir)
        print(f"Evaluation time: {time.time() - start_time:.3f} seconds")

if __name__ == "__main__":
    main()

