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


# -------------------------------------------------------------
# Binarization 
# -------------------------------------------------------------
def binarize_image(image: np.ndarray) -> np.ndarray:
    """
    Binarize a grayscale image using Sauvola thresholding.
    Returns an image with black ink (0) on white background (255),
    matching the convention of the pre-binarized dataset.
    """
    image = cv2.GaussianBlur(image, (3, 3), 0)
    window_size = 19
    k = 0.06
    thresh = threshold_sauvola(image, window_size=window_size, k=k)
    binary = (image < thresh).astype(np.uint8) * 255
    return 255 - binary   # ink=0, background=255


def is_binary(image: np.ndarray) -> bool:
    """Return True if the image contains only pixel values 0 and 255."""
    unique = np.unique(image)
    return set(unique).issubset({0, 255})


def ensure_binary(image: np.ndarray) -> np.ndarray:
    """
    If the image is already binary (only 0 and 255), return as-is.
    Otherwise apply Sauvola binarization.
    """
    if is_binary(image):
        return image
    return binarize_image(image)


# -------------------------------------------------------------
# Dataset
# -------------------------------------------------------------
class DatasetFromFolder(data.Dataset):
    def __init__(self, dataset, foldername, labelfolder, imgtype='png',
                 scale_size=(64, 128), is_training=True):
        super(DatasetFromFolder, self).__init__()

        self.is_training = is_training
        self.imgtype     = imgtype
        self.scale_size  = scale_size
        self.folder      = foldername
        self.dataset     = dataset

        if self.dataset == 'bullinger':
            self.cerug = True
        else:
            self.cerug = False

        self.labelidx_name = labelfolder + dataset + 'binarised_writer_index_table.pickle'
        print(self.labelidx_name)

        self.imglist    = self._get_image_list(self.folder)
        self.idlist     = self._get_all_identity()
        self.idx_tab    = self._convert_identity2index(self.labelidx_name)
        self.num_writer = len(self.idx_tab)

        print('-' * 10)
        print('loading dataset %s with images: %d' % (dataset, len(self.imglist)))
        print('number of writer is: %d' % len(self.idx_tab))
        print('-*' * 10)

        # Binarize all at once, cache in RAM 
        print("Binarizing and caching images in RAM (once for all epochs)...")
        self.cache = {}
        for imgfile in self.imglist:
            gray = np.array(PILImage.open(self.folder + imgfile).convert('L'))
            self.cache[imgfile] = ensure_binary(gray)
        print(f"Done. {len(self.cache)} images cached.")

    # ------------------------------------------------------------------

    def _convert_identity2index(self, savename):
        if os.path.exists(savename):
            with open(savename, 'rb') as fp:
                identity_idx = pickle.load(fp)
        else:
            identity_idx = {}
            for idx, ids in enumerate(self.idlist):
                identity_idx[ids] = idx
            if not os.path.exists(savename):
                try:
                    with open(savename, 'wb') as fp:
                        pickle.dump(identity_idx, fp)
                    print(f"Pickle saved to {savename}")
                except Exception as e:
                    print(f"WARNING: Could not save pickle: {e}")
        return identity_idx

    def _get_all_identity(self):
        writer_list = []
        for img in self.imglist:
            writerId = self._get_identity(img)
            writer_list.append(writerId)
        writer_list = sorted(list(set(writer_list)))
        return writer_list

    def _get_identity(self, fname):
        if self.cerug:
            return fname.split('_')[0]
        else:
            return fname.split('-')[0]

    def _get_image_list(self, folder):
        flist = sorted(os.listdir(folder))
        imglist = []
        for img in flist:
            if img.endswith(self.imgtype):
                imglist.append(img)
        return imglist

    def transform(self):
        return Compose([ToTensor()])

    def resize(self, image):
        h, w = image.shape[:2]
        ratio_h = float(self.scale_size[0]) / float(h)
        ratio_w = float(self.scale_size[1]) / float(w)

        if ratio_h < ratio_w:
            ratio  = ratio_h
            hfirst = False
        else:
            ratio  = ratio_w
            hfirst = True

        nh = int(ratio * h)
        nw = int(ratio * w)

        pil_img = PILImage.fromarray(image)
        imre    = np.array(pil_img.resize((nw, nh), PILImage.NEAREST))
        imre = 255 - imre

        ch, cw  = imre.shape[:2]
        new_img = np.zeros(self.scale_size)

        if self.is_training:
            dy = random.randint(0, self.scale_size[0] - ch)
            dx = random.randint(0, self.scale_size[1] - cw)
        else:
            dy = int((self.scale_size[0] - ch) / 2.0)
            dx = int((self.scale_size[1] - cw) / 2.0)

        new_img[dy:dy + ch, dx:dx + cw] = imre.astype('float')

        return new_img, hfirst

    def __getitem__(self, index):
        imgfile = self.imglist[index]
        writer  = self.idx_tab[self._get_identity(imgfile)]

        image = self.cache[imgfile]

        image, hfirst = self.resize(image)
        image = image / 255.0   # {0, 255} → {0.0, 1.0}

        image  = self.transform()(image)
        writer = torch.from_numpy(np.array(writer))

        return image, writer, imgfile

    def __len__(self):
        return len(self.imglist)