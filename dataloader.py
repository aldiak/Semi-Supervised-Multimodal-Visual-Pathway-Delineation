import numpy as np
import config_2d
import os
import itertools
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision.transforms import transforms
from PIL import Image
from copy import deepcopy
import random
from scipy.ndimage import zoom
from sklearn.model_selection import KFold

batch_size = config_2d.BATCH_SIZE
patch_size_w = config_2d.PATCH_SIZE_W
patch_size_h = config_2d.PATCH_SIZE_H
n_classes = config_2d.NUM_CLASSES

def CN_make_dirset_train(train_x_1_dir, train_x_2_dir, train_y_dir):
    train_x_1_path = []
    train_x_2_path = []
    train_y_path = []
    n = len(os.listdir(train_x_1_dir))
    for i in range(n):
        img_x_1 = os.path.join(train_x_1_dir, 'x_t1-data_%d.nii.gz' % i)
        train_x_1_path.append(img_x_1)
        img_x_2 = os.path.join(train_x_2_dir, 'x_fa-data_%d.nii.gz' % i)
        train_x_2_path.append(img_x_2)
        img_y = os.path.join(train_y_dir, 'y-data_%d.nii.gz' % i)
        train_y_path.append(img_y)
    return train_x_1_path, train_x_2_path, [], train_y_path

def CN_make_dirset_train_with_folds(train_x_1_dir, train_x_2_dir, train_y_dir, n_splits=3, seed=66):
    train_x_1_path, train_x_2_path, _, train_y_path = CN_make_dirset_train(train_x_1_dir, train_x_2_dir, train_y_dir)
    indices = list(range(len(train_x_1_path)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in kf.split(indices):
        train_x_1_fold = [train_x_1_path[i] for i in train_idx]
        train_x_2_fold = [train_x_2_path[i] for i in train_idx]
        train_y_fold = [train_y_path[i] for i in train_idx]
        val_x_1_fold = [train_x_1_path[i] for i in val_idx]
        val_x_2_fold = [train_x_2_path[i] for i in val_idx]
        val_y_fold = [train_y_path[i] for i in val_idx]
        folds.append({
            'train': (train_x_1_fold, train_x_2_fold, train_y_fold),
            'val': (val_x_1_fold, val_x_2_fold, val_y_fold)
        })
    return folds

class CN_MyTrainDataset(Dataset):
    def __init__(self, train_x_1_paths, train_x_2_paths, train_y_paths, x_transform=None, y_transform=None):
        self.train_x_1_path = train_x_1_paths
        self.train_x_2_path = train_x_2_paths
        self.train_y_path = train_y_paths
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __getitem__(self, index):
        x_1_path = self.train_x_1_path[index]
        x_2_path = self.train_x_2_path[index]
        y_path = self.train_y_path[index]

        img_x_1 = nib.load(x_1_path)
        img_x_1_data = img_x_1.get_fdata()
        x_1_are_Nans = np.isnan(img_x_1_data)
        img_x_1_data[x_1_are_Nans] = 0
        img_x_1_data = np.array(img_x_1_data, dtype='uint8')
        try:
            spacing = img_x_1.header.get_zooms()[:2]  # Get in-plane spacing (x, y)
        except Exception as e:
            print(f"Warning: Failed to get spacing for {x_1_path}, using default [1.0, 1.0]. Error: {e}")
            spacing = (1.0, 1.0)

        img_x_2 = nib.load(x_2_path)
        img_x_2_data = img_x_2.get_fdata()
        x_2_are_Nans = np.isnan(img_x_2_data)
        img_x_2_data[x_2_are_Nans] = 0
        img_x_2_data = np.array(img_x_2_data, dtype='uint8')

        img_y = nib.load(y_path)
        label = img_y.get_fdata()

        if self.x_transform is not None:
            img_x_1_data = self.x_transform(img_x_1_data)
            img_x_2_data = self.x_transform(img_x_2_data)

        if self.y_transform is not None:
            label = self.y_transform(label)

        sample = {
            'image_t1': img_x_1_data,
            'image_fa': img_x_2_data,
            'label': label,
            'idx': index,
            'spacing': spacing  # Add spacing to sample
        }
        return sample

    def __len__(self):
        return len(self.train_x_1_path)

class CN_MyTestDataset(Dataset):
    def __init__(self, train_x_1_dir, train_x_2_dir, train_y_dir, x_transform=None, y_transform=None):
        train_x_1_path, train_x_2_path, _, train_y_path = CN_make_dirset_train(train_x_1_dir, train_x_2_dir, train_y_dir)
        self.train_x_1_path = train_x_1_path
        self.train_x_2_path = train_x_2_path
        self.train_y_path = train_y_path
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __getitem__(self, index):
        x_1_path = self.train_x_1_path[index]
        x_2_path = self.train_x_2_path[index]
        y_path = self.train_y_path[index]

        img_x_1 = nib.load(x_1_path)
        img_x_1_data = img_x_1.get_fdata()
        x_1_are_Nans = np.isnan(img_x_1_data)
        img_x_1_data[x_1_are_Nans] = 0
        img_x_1_data = np.array(img_x_1_data, dtype='uint8')
        try:
            spacing = img_x_1.header.get_zooms()[:2]  # Get in-plane spacing (x, y)
        except Exception as e:
            print(f"Warning: Failed to get spacing for {x_1_path}, using default [1.0, 1.0]. Error: {e}")
            spacing = (1.0, 1.0)

        img_x_2 = nib.load(x_2_path)
        img_x_2_data = img_x_2.get_fdata()
        x_2_are_Nans = np.isnan(img_x_2_data)
        img_x_2_data[x_2_are_Nans] = 0
        img_x_2_data = np.array(img_x_2_data, dtype='uint8')

        img_y = nib.load(y_path)
        label = img_y.get_fdata()

        if self.x_transform is not None:
            img_x_1_data = self.x_transform(img_x_1_data)
            img_x_2_data = self.x_transform(img_x_2_data)

        if self.y_transform is not None:
            label = self.y_transform(label)

        return img_x_1_data, img_x_2_data, label, spacing

    def __len__(self):
        return len(self.train_x_1_path)

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image_t1'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image_t1': image, 'label': label}
        return sample

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    from scipy.ndimage import rotate
    image = rotate(image, angle, order=0, reshape=False)
    label = rotate(label, angle, order=0, reshape=False)
    return image, label

class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)


# import random
# from itertools import cycle, islice
# from torch.utils.data import Sampler

# def iterate_once(indices):
#     """Yield indices one at a time in random order."""
#     indices = list(indices)  # Create a copy to avoid modifying the original
#     random.shuffle(indices)   # Shuffle to ensure randomness
#     return iter(indices)

# def iterate_eternally(indices):
#     """Cycle through indices indefinitely in random order."""
#     indices = list(indices)
#     while True:
#         random.shuffle(indices)
#         for idx in indices:
#             yield idx

# def grouper(iterable, n):
#     """Group iterable into chunks of size n, discarding incomplete chunks."""
#     iterator = iter(iterable)
#     while True:
#         chunk = list(islice(iterator, n))
#         if len(chunk) == n:
#             yield chunk
#         else:
#             break  # Stop if the chunk is incomplete


# class TwoStreamBatchSampler(Sampler):
#     def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
#         super().__init__(None)
#         self.primary_indices = list(primary_indices)
#         self.secondary_indices = list(secondary_indices)
#         self.batch_size = batch_size
#         self.secondary_batch_size = min(secondary_batch_size, len(self.secondary_indices)) if self.secondary_indices else 0
#         self.primary_batch_size = batch_size - self.secondary_batch_size

#         # Validate primary indices
#         if len(self.primary_indices) < self.primary_batch_size or self.primary_batch_size <= 0:
#             raise ValueError(f"Primary indices ({len(self.primary_indices)}) must be >= primary batch size ({self.primary_batch_size}).")
#         # Allow empty secondary indices
#         if self.secondary_indices and len(self.secondary_indices) < self.secondary_batch_size:
#             raise ValueError(f"Secondary indices ({len(self.secondary_indices)}) must be >= secondary batch size ({self.secondary_batch_size}).")

#     def __iter__(self):
#         primary_iter = iterate_once(self.primary_indices)
#         if self.secondary_indices:
#             secondary_iter = iterate_eternally(self.secondary_indices)
#             return (
#                 primary_batch + secondary_batch
#                 for (primary_batch, secondary_batch)
#                 in zip(
#                     grouper(primary_iter, self.primary_batch_size),
#                     grouper(secondary_iter, self.secondary_batch_size)
#                 )
#             )
#         else:
#             # If no secondary indices, only yield primary batches
#             return (primary_batch for primary_batch in grouper(primary_iter, self.primary_batch_size))

#     def __len__(self):
#         return len(self.primary_indices) // self.primary_batch_size


# import random
# from itertools import cycle, islice
# from torch.utils.data import Sampler

# def iterate_once(indices):
#     """Yield indices one at a time in random order."""
#     indices = list(indices)
#     random.shuffle(indices)
#     return iter(indices)

# def iterate_eternally(indices):
#     """Cycle through indices indefinitely in random order."""
#     indices = list(indices)
#     while True:
#         random.shuffle(indices)
#         for idx in indices:
#             yield idx

# def grouper(iterable, n):
#     """Group iterable into chunks of size n, discarding incomplete chunks."""
#     iterator = iter(iterable)
#     while True:
#         chunk = list(islice(iterator, n))
#         if len(chunk) == n:
#             yield chunk
#         else:
#             break

# class TwoStreamBatchSampler(Sampler):
#     def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
#         super().__init__(None)
#         self.primary_indices = list(primary_indices)
#         self.secondary_indices = list(secondary_indices)
#         self.batch_size = batch_size
#         # Cap secondary_batch_size to len(secondary_indices) if smaller
#         self.secondary_batch_size = min(secondary_batch_size, len(self.secondary_indices)) if self.secondary_indices else 0
#         self.primary_batch_size = batch_size - self.secondary_batch_size

#         # Validate inputs
#         if self.primary_batch_size <= 0:
#             raise ValueError(f"Primary batch size ({self.primary_batch_size}) must be positive.")
#         if len(self.primary_indices) < self.primary_batch_size:
#             raise ValueError(f"Primary indices ({len(self.primary_indices)}) must be >= primary batch size ({self.primary_batch_size}).")
#         if self.secondary_indices and len(self.secondary_indices) < self.secondary_batch_size:
#             raise ValueError(f"Secondary indices ({len(self.secondary_indices)}) must be >= secondary batch size ({self.secondary_batch_size}).")

#     def __iter__(self):
#         primary_iter = iterate_once(self.primary_indices)
#         if self.secondary_indices:
#             secondary_iter = iterate_eternally(self.secondary_indices)
#             return (
#                 primary_batch + secondary_batch
#                 for (primary_batch, secondary_batch)
#                 in zip(
#                     grouper(primary_iter, self.primary_batch_size),
#                     grouper(secondary_iter, self.secondary_batch_size)
#                 )
#             )
#         else:
#             # If no secondary indices, only yield primary batches
#             return (primary_batch for primary_batch in grouper(primary_iter, self.primary_batch_size))

#     def __len__(self):
#         return len(self.primary_indices) // self.primary_batch_size