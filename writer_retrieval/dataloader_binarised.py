"""
@author: sheng he
@email: heshengxgd@gmail.com

Modified to support binarized images (Sauvola thresholding).
- Replaced deprecated scipy.misc.imread / imresize
- Added auto-binarization for non-binary images
- Uses NEAREST interpolation on resize to preserve binary values
- Fixed non-deterministic set() and os.listdir() ordering

Retrieval modification:
  is_retrieval=True  — builds a fresh local writer-index from THIS split only.
                       No dependency on the training pickle whatsoever.
                       Used for the test set when test writers != train writers.
  is_retrieval=False — original behaviour: looks up indices from the shared
                       pickle (used for the training set).

__getitem__ always returns (image_tensor, integer_label, imgfile_name).
For retrieval the integer_label is a local index (0..N_writers_in_split-1)
whose only purpose is grouping samples by writer for metric computation.
"""

import os
import pickle
import numpy as np
from PIL import Image as PILImage
import cv2
from skimage.filters import threshold_sauvola
import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor
import random


# ---------------------------------------------------------------------------
# Binarization
# ---------------------------------------------------------------------------
def binarize_image(image: np.ndarray) -> np.ndarray:
    image = cv2.GaussianBlur(image, (3, 3), 0)
    thresh = threshold_sauvola(image, window_size=19, k=0.06)
    binary = (image < thresh).astype(np.uint8) * 255
    return 255 - binary   # ink=0, background=255

def is_binary(image: np.ndarray) -> bool:
    return set(np.unique(image)).issubset({0, 255})

def ensure_binary(image: np.ndarray) -> np.ndarray:
    return image if is_binary(image) else binarize_image(image)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DatasetFromFolder(data.Dataset):
    def __init__(self, dataset, foldername, labelfolder, imgtype='png',
                 scale_size=(64, 128), is_training=True, is_retrieval=False):
        """
        is_retrieval : if True, build a fresh writer→index mapping from
                       this split alone (no shared pickle).  Set this to
                       True for the test/gallery/query set when test writers
                       are disjoint from training writers.
        """
        super().__init__()

        self.is_training  = is_training
        self.is_retrieval = is_retrieval
        self.imgtype      = imgtype
        self.scale_size   = scale_size
        self.folder       = foldername
        self.dataset      = dataset
        self.cerug        = (dataset == 'bullinger')

        # sorted → deterministic across runs
        self.imglist = self._get_image_list(self.folder)
        self.idlist  = self._get_all_identity()   # sorted unique writer IDs

        if self.is_retrieval:
            # ----------------------------------------------------------------
            # Open-set path: build a fresh local index from THIS split.
            # Never touches the training pickle.
            # ----------------------------------------------------------------
            self.local_idx  = {wid: i for i, wid in enumerate(self.idlist)}
            self.num_writer = len(self.local_idx)
            print('[Retrieval mode] Building local writer index '
                  f'({self.num_writer} writers) — independent of training.')
        else:
            # ----------------------------------------------------------------
            # Closed-set path: use the shared pickle (training split).
            # ----------------------------------------------------------------
            self.labelidx_name = labelfolder + dataset + 'bina445writer_index_table.pickle'
            print(self.labelidx_name)
            self.idx_tab    = self._convert_identity2index(self.labelidx_name)
            self.num_writer = len(self.idx_tab)

        print('-' * 10)
        print('Loading dataset %s  |  images: %d  |  writers: %d'
              % (dataset, len(self.imglist), self.num_writer))
        print('-*' * 10)

        # Binarize once and cache in RAM
        print('Binarizing and caching images in RAM...')
        self.cache = {}
        for imgfile in self.imglist:
            gray = np.array(PILImage.open(self.folder + imgfile).convert('L'))
            self.cache[imgfile] = ensure_binary(gray)
        print(f'Done. {len(self.cache)} images cached.')

    # ------------------------------------------------------------------
    def _convert_identity2index(self, savename):
        if os.path.exists(savename):
            with open(savename, 'rb') as fp:
                identity_idx = pickle.load(fp)
        else:
            identity_idx = {ids: idx for idx, ids in enumerate(self.idlist)}
            try:
                with open(savename, 'wb') as fp:
                    pickle.dump(identity_idx, fp)
                print(f'Pickle saved to {savename}')
            except Exception as e:
                print(f'WARNING: Could not save pickle: {e}')
        return identity_idx

    def _get_all_identity(self):
        return sorted(set(self._get_identity(img) for img in self.imglist))

    def _get_identity(self, fname):
        return fname.split('_')[0] if self.cerug else fname.split('-')[0]

    def _get_image_list(self, folder):
        return sorted(f for f in os.listdir(folder) if f.endswith(self.imgtype))

    def transform(self):
        return Compose([ToTensor()])

    def resize(self, image):
        h, w    = image.shape[:2]
        ratio_h = float(self.scale_size[0]) / float(h)
        ratio_w = float(self.scale_size[1]) / float(w)
        ratio   = ratio_h if ratio_h < ratio_w else ratio_w

        nh, nw = int(ratio * h), int(ratio * w)
        imre   = np.array(PILImage.fromarray(image).resize((nw, nh), PILImage.NEAREST))
        imre   = 255 - imre   # ink=0→255 (white ink on black bg)

        ch, cw  = imre.shape[:2]
        new_img = np.zeros(self.scale_size)

        if self.is_training:
            dy = random.randint(0, self.scale_size[0] - ch)
            dx = random.randint(0, self.scale_size[1] - cw)
        else:
            dy = (self.scale_size[0] - ch) // 2
            dx = (self.scale_size[1] - cw) // 2

        new_img[dy:dy + ch, dx:dx + cw] = imre.astype('float')
        return new_img

    def __getitem__(self, index):
        imgfile = self.imglist[index]
        wid_str = self._get_identity(imgfile)

        if self.is_retrieval:
            writer = self.local_idx[wid_str]
        else:
            writer = self.idx_tab[wid_str]

        image = self.cache[imgfile]
        image = self.resize(image) / 255.0
        image = self.transform()(image)
        writer = torch.tensor(writer, dtype=torch.long)

        return image, writer, imgfile

    def __len__(self):
        return len(self.imglist)