"""
@author: sheng he
@email: heshengxgd@gmail.com

Modified for open-set writer retrieval:
- is_retrieval=False (train set): original behaviour — shared pickle,
  writer→index mapping fitted on the training split.
- is_retrieval=True  (test set) : builds a fresh local writer→index mapping
  from this split alone.  Never touches the training pickle.
  Test writers are completely disjoint from train writers; the integer labels
  are only used for equality checks in the retrieval metrics.

Other fixes vs the original:
- os.listdir() → sorted()  so file order is deterministic across runs
- set()        → sorted(set()) in _get_all_identity for same reason
- imgfile returned as third element of __getitem__ (was already there)
"""

import os
import pickle
import numpy as np
from PIL import Image
import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor
import random


class DatasetFromFolder(data.Dataset):
    def __init__(self, dataset, foldername, labelfolder, imgtype='png',
                 scale_size=(64, 128), is_training=True, is_retrieval=False):
        """
        is_retrieval : if True, build a fresh writer→index mapping from
                       this split alone (no shared pickle).  Set this to
                       True for the test set when test writers are disjoint
                       from training writers.
        """
        super(DatasetFromFolder, self).__init__()

        self.is_training  = is_training
        self.is_retrieval = is_retrieval
        self.imgtype      = imgtype
        self.scale_size   = scale_size
        self.folder       = foldername
        self.dataset      = dataset
        self.cerug        = (dataset == 'bullinger')

        # ── Image list: sorted for determinism ───────────────────────────
        self.imglist = self._get_image_list(self.folder)

        # ── Drop corrupted images (kept from original) ───────────────────
        valid = []
        for imgfile in self.imglist:
            try:
                Image.open(os.path.join(self.folder, imgfile)).verify()
                valid.append(imgfile)
            except (IOError, OSError, ValueError):
                print(f'Warning: skipping corrupted image {imgfile}')
        self.imglist = valid

        # ── Writer identity list: sorted for determinism ─────────────────
        self.idlist = self._get_all_identity()

        if self.is_retrieval:
            # ── Open-set path: fresh local index, never touches pickle ───
            self.local_idx  = {wid: i for i, wid in enumerate(self.idlist)}
            self.num_writer = len(self.local_idx)
            print(f'[Retrieval mode] Building local writer index '
                  f'({self.num_writer} writers) — independent of training.')
        else:
            # ── Closed-set path: shared pickle (training split) ──────────
            self.labelidx_name = labelfolder + dataset + 'graywriter_index_table.pickle'
            print(self.labelidx_name)
            self.idx_tab    = self._convert_identity2index(self.labelidx_name)
            self.num_writer = len(self.idx_tab)

        print('-' * 10)
        print('loading dataset %s with images: %d' % (dataset, len(self.imglist)))
        print('number of writers: %d' % self.num_writer)
        print('-*' * 10)

    # ── Pickle helpers (closed-set only) ─────────────────────────────────
    def _convert_identity2index(self, savename):
        if os.path.exists(savename):
            with open(savename, 'rb') as fp:
                identity_idx = pickle.load(fp)
        else:
            identity_idx = {ids: idx for idx, ids in enumerate(self.idlist)}
            with open(savename, 'wb') as fp:
                pickle.dump(identity_idx, fp)
        return identity_idx

    # ── Identity helpers ──────────────────────────────────────────────────
    def _get_all_identity(self):
        # sorted(set(...)) — deterministic ordering
        return sorted(set(self._get_identity(img) for img in self.imglist))

    def _get_identity(self, fname):
        if self.cerug:
            return fname.split('_')[0]
        return fname.split('-')[0]

    def _get_image_list(self, folder):
        # sorted — deterministic across runs (os.listdir order is arbitrary)
        return sorted(f for f in os.listdir(folder) if f.endswith(self.imgtype))

    # ── Transforms ───────────────────────────────────────────────────────
    def transform(self):
        return Compose([ToTensor()])

    def resize(self, image):
        h, w    = image.shape[:2]
        ratio_h = float(self.scale_size[0]) / float(h)
        ratio_w = float(self.scale_size[1]) / float(w)

        ratio = ratio_h if ratio_h < ratio_w else ratio_w
        nh, nw = int(ratio * h), int(ratio * w)

        imre = np.array(Image.fromarray(image).resize((nw, nh), Image.BILINEAR))
        imre = 255 - imre   # invert: ink→white on black bg

        ch, cw  = imre.shape[:2]
        new_img = np.zeros(self.scale_size, dtype=np.float32)

        if self.is_training:
            dy = random.randint(0, self.scale_size[0] - ch)
            dx = random.randint(0, self.scale_size[1] - cw)
        else:
            dy = (self.scale_size[0] - ch) // 2
            dx = (self.scale_size[1] - cw) // 2

        new_img[dy:dy + ch, dx:dx + cw] = imre.astype('float')
        return new_img

    # ── Item ──────────────────────────────────────────────────────────────
    def __getitem__(self, index):
        imgfile = self.imglist[index]
        wid_str = self._get_identity(imgfile)

        if self.is_retrieval:
            writer = self.local_idx[wid_str]
        else:
            writer = self.idx_tab[wid_str]

        image = np.array(Image.open(self.folder + imgfile).convert('L'))
        image = self.resize(image)
        image = image / 255.0
        image = self.transform()(image)
        writer = torch.tensor(writer, dtype=torch.long)

        return image, writer, imgfile

    def __len__(self):
        return len(self.imglist)