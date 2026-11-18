# import os
# import glob
# import cv2
# import torch
# import numpy as np
# from torch.utils.data import Dataset
# # import albumentations as A


# class KidneyStoneDataset(Dataset):
#     """
#     Refined Dataset for 2D PNG Slices (MGMTNet Architecture).
#     Handles dual-mask extraction (Kidney & Stone), dynamic 512x512 resolution,
#     and synchronized spatial augmentations.
#     """
#     def __init__(self, img_dir, msk_dir, target_size=(512, 512), is_train=False):
#         """
#         Args:
#             img_dir (str): Path to directory containing 'Stone' and 'Non-Stone' subfolders with PNGs.
#             msk_dir (str): Path to directory containing 'Stone' and 'Non-Stone' subfolders with PNG masks.
#             target_size (tuple): (height, width) spatial dimensions. Defaults to (512, 512).
#             is_train (bool): Applies training augmentations if True.
#         """
#         self.target_size = target_size
#         self.is_train = is_train
        
#         self.image_paths = []
#         self.mask_paths = []
#         self.labels = []

#         # 1. Build lookup dictionary for masks: key (filename without ext) -> full path
#         mask_lookup = {}
#         for folder in ['Stone', 'Non-Stone']:
#             p_msk = os.path.join(msk_dir, folder)
#             if os.path.exists(p_msk):
#                 for m_path in glob.glob(os.path.join(p_msk, '*.png')):
#                     base_name = os.path.splitext(os.path.basename(m_path))[0]
#                     mask_lookup[base_name] = m_path

#         # 2. Discover PNG images and pair with ground-truth masks
#         for folder in ['Stone', 'Non-Stone']:
#             p_img = os.path.join(img_dir, folder)
#             if os.path.exists(p_img):
#                 for imp in sorted(glob.glob(os.path.join(p_img, '*.png'))):
#                     base_name = os.path.splitext(os.path.basename(imp))[0]
                    
#                     if base_name in mask_lookup:
#                         self.image_paths.append(imp)
#                         self.mask_paths.append(mask_lookup[base_name])
                        
#                         parent_dir = os.path.basename(os.path.dirname(imp))
#                         is_stone = 1 if parent_dir.lower() == 'stone' else 0
#                         self.labels.append(is_stone)

#         if not self.image_paths:
#             raise ValueError(f"No matching .png image-mask pairs found in: {img_dir} & {msk_dir}")

#         stone_count = sum(self.labels)
#         print(f"Dataset Initialized ({os.path.basename(img_dir)}) -> "
#               f"Total Pairs: {len(self.labels)} | Stone: {stone_count} | Non-stone: {len(self.labels) - stone_count}")

#         # # 3. Synchronized Spatial Augmentation Pipeline
#         # if self.is_train:
#         #     self.transform = A.Compose([
#         #         A.Resize(height=self.target_size[0], width=self.target_size[1]),
#         #         A.HorizontalFlip(p=0.5),
#         #         A.VerticalFlip(p=0.5),
#         #         A.RandomRotate90(p=0.5),
#         #         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
#         #     ])
#         # else:
#         #     self.transform = A.Compose([
#         #         A.Resize(height=self.target_size[0], width=self.target_size[1])
#         #     ])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         # 1. Read Grayscale .png Image
#         img_raw = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
#         if img_raw is None:
#             raise FileNotFoundError(f"Failed to read image at: {self.image_paths[idx]}")
        
#         # Scale 8-bit PNG pixel intensities to [0, 1] range
#         img = img_raw.astype(np.float32) / 255.0

#         # 2. Read Ground Truth .png Mask
#         msk_raw = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
#         if msk_raw is None:
#             raise FileNotFoundError(f"Failed to read mask at: {self.mask_paths[idx]}")

#         # 3. Apply Synchronized Albumentations Transforms
#         augmented = self.transform(image=img, mask=msk_raw)
#         img_aug = augmented['image']      # Shape: (H, W)
#         msk_aug = augmented['mask']       # Shape: (H, W)

#         # 4. Expand 1-channel grayscale slice to 3-channel input tensor
#         img_tensor = torch.from_numpy(img_aug).unsqueeze(0).repeat(3, 1, 1)

#         # 5. Extract Multi-Task Binary Target Masks
#         # Kidney parenchyma mask: Any foreground pixel (> 0)
#         kidney_mask = (msk_aug > 0).astype(np.float32)
        
#         # Stone mask: Accommodates both discrete class IDs (2) or 8-bit mask intensities (> 128)
#         stone_mask = ((msk_aug == 2) | (msk_aug > 128)).astype(np.float32)

#         return {
#             'image': img_tensor,                                        # Tensor (3, H, W)
#             'kidney_mask': torch.from_numpy(kidney_mask).unsqueeze(0),  # Tensor (1, H, W)
#             'stone_mask': torch.from_numpy(stone_mask).unsqueeze(0),   # Tensor (1, H, W)
#             'label': torch.tensor(self.labels[idx], dtype=torch.long)   # Scalar Tensor (0 or 1)
#         }

import glob
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class KidneyStoneDataset(Dataset):
    """Refined Dataset for 2D PNG Slices (MGMTNet Architecture).

    Handles dual-mask extraction (Kidney & Stone) and dynamic 512x512
    resolution without online augmentations.
    """

    def __init__(self, img_dir, msk_dir, target_size=(512, 512)):
        """Args:

        img_dir (str): Path to directory containing 'Stone' and 'Non-Stone'
        subfolders with PNGs. msk_dir (str): Path to directory containing
        'Stone' and 'Non-Stone' subfolders with PNG masks. target_size (tuple):
        (height, width) spatial dimensions. Defaults to (512, 512).
        """
        self.target_size = target_size

        self.image_paths = []
        self.mask_paths = []
        self.labels = []

        # 1. Build lookup dictionary for masks: key (filename without ext) -> full path
        mask_lookup = {}
        for folder in ["Stone", "Non-Stone"]:
            p_msk = os.path.join(msk_dir, folder)
            if os.path.exists(p_msk):
                for m_path in glob.glob(os.path.join(p_msk, "*.png")):
                    base_name = os.path.splitext(os.path.basename(m_path))[0]
                    mask_lookup[base_name] = m_path

        # 2. Discover PNG images and pair with ground-truth masks
        for folder in ["Stone", "Non-Stone"]:
            p_img = os.path.join(img_dir, folder)
            if os.path.exists(p_img):
                for imp in sorted(glob.glob(os.path.join(p_img, "*.png"))):
                    base_name = os.path.splitext(os.path.basename(imp))[0]

                    if base_name in mask_lookup:
                        self.image_paths.append(imp)
                        self.mask_paths.append(mask_lookup[base_name])

                        parent_dir = os.path.basename(os.path.dirname(imp))
                        is_stone = (
                            1 if parent_dir.lower() == "stone" else 0
                        )
                        self.labels.append(is_stone)

        if not self.image_paths:
            raise ValueError(
                f"No matching .png image-mask pairs found in: {img_dir} &"
                f" {msk_dir}"
            )

        stone_count = sum(self.labels)
        print(
            f"Dataset Initialized ({os.path.basename(img_dir)}) -> Total Pairs:"
            f" {len(self.labels)} | Stone: {stone_count} | Non-stone:"
            f" {len(self.labels) - stone_count}"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Read Grayscale .png Image
        img_raw = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            raise FileNotFoundError(
                f"Failed to read image at: {self.image_paths[idx]}"
            )

        # 2. Read Ground Truth .png Mask
        msk_raw = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if msk_raw is None:
            raise FileNotFoundError(
                f"Failed to read mask at: {self.mask_paths[idx]}"
            )

        # 3. Ensure target dimensions if needed
        h, w = self.target_size
        if img_raw.shape[:2] != (h, w):
            img_raw = cv2.resize(
                img_raw, (w, h), interpolation=cv2.INTER_LINEAR
            )
            msk_raw = cv2.resize(
                msk_raw, (w, h), interpolation=cv2.INTER_NEAREST
            )

        # Scale 8-bit PNG pixel intensities to [0, 1] range
        img = img_raw.astype(np.float32) / 255.0

        # 4. Expand 1-channel grayscale slice to 3-channel input tensor
        img_tensor = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)

        # 5. Extract Multi-Task Binary Target Masks
        # Kidney parenchyma mask: Any foreground pixel (> 0)
        kidney_mask = (msk_raw > 0).astype(np.float32)

        # Stone mask: Accommodates both discrete class IDs (2) or 8-bit mask intensities (> 128)
        stone_mask = ((msk_raw == 2) | (msk_raw > 128)).astype(np.float32)

        return {
            "image": (
                img_tensor
            ),  # Tensor (3, H, W)
            "kidney_mask": torch.from_numpy(kidney_mask).unsqueeze(
                0
            ),  # Tensor (1, H, W)
            "stone_mask": torch.from_numpy(stone_mask).unsqueeze(
                0
            ),  # Tensor (1, H, W)
            "label": torch.tensor(
                self.labels[idx], dtype=torch.long
            ),  # Scalar Tensor (0 or 1)
        }