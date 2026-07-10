import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random
import dataloader_binarised as bindset
import dataloader as dset
import GRRNN as net
import numpy as np
import os
import argparse


# ---------------------------------------------------------------------------
# Label-smoothing cross-entropy
# ---------------------------------------------------------------------------
class LabelSomCE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence  = 1.0 - smoothing
        logprobs    = F.log_softmax(x, dim=-1)
        nll_loss    = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        return (confidence * nll_loss + smoothing * smooth_loss).mean()



# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


# ---------------------------------------------------------------------------
# Retrieval metrics  (open-set, leave-one-out, chunked GPU version)
# ---------------------------------------------------------------------------
def compute_retrieval_metrics(embeddings: torch.Tensor,
                               labels: torch.Tensor,
                               dataset:str ='CVL',
                               filenames: list,
                               ks=(1, 5, 10),
                               device: str = 'cuda',
                               chunk_size: int = 2000) -> dict:
 
    N         = embeddings.size(0)
    labels_np = labels.numpy()

    # ── Parse document IDs from filenames ────────────────────────────────
    # Format: writerid-docnum-linenum-wordnum-transcription.tif
    if dataset=='CVL':
        doc_ids_np = np.array([
            int(os.path.basename(f).split('-')[1]) for f in filenames
        ], dtype=np.int64)
    elif dataset=='bullinger':
        doc_ids_np = np.array([
            int(os.path.basename(f).split('_')[1]) for f in filenames
        ], dtype=np.int64)
        

    # Move full gallery to device once
    gallery  = embeddings.to(device)
    labels_t = labels.to(device)

    max_k = max(ks)

    # ── Word-level accumulators ───────────────────────────────────────────
    ap_per_query = np.zeros(N, dtype=np.float64)
    n_rel_all    = np.zeros(N, dtype=np.int64)
    topk_labels  = np.zeros((N, max_k), dtype=np.int64)

    for start in range(0, N, chunk_size):
        end     = min(start + chunk_size, N)
        q_emb   = gallery[start:end]
        q_lbl   = labels_t[start:end]

        sim_chunk = q_emb @ gallery.T

        for local_i in range(end - start):
            sim_chunk[local_i, start + local_i] = -1e9

        sorted_idx    = torch.argsort(sim_chunk, dim=1, descending=True)
        sorted_lbl    = labels_t[sorted_idx]
        sorted_lbl_np = sorted_lbl.cpu().numpy()
        q_lbl_np      = q_lbl.cpu().numpy()

        for local_i in range(end - start):
            global_i      = start + local_i
            q             = q_lbl_np[local_i]
            ranked_labels = sorted_lbl_np[local_i, :N-1]
            relevant      = (ranked_labels == q).astype(np.float32)
            n_rel         = int(relevant.sum())
            n_rel_all[global_i] = n_rel

            if n_rel == 0:
                continue

            cumsum  = np.cumsum(relevant)
            ranks   = np.arange(1, N, dtype=np.float32)
            prec_at = cumsum / ranks
            ap_per_query[global_i] = (prec_at * relevant).sum() / n_rel

        topk_labels[start:end] = sorted_lbl_np[:, :max_k]

    # ── Word-level global mAP ─────────────────────────────────────────────
    has_rel  = n_rel_all > 0
    mAP_word = float(ap_per_query[has_rel].mean()) * 100 if has_rel.any() else 0.0

    # ── Word-level top k ───────────────────────────────────────────────
    recall_word = {}
    for k in ks:
        k_cap = min(k, N - 1)
        hit   = (topk_labels[:, :k_cap] == labels_np[:, None]).any(axis=1).astype(float)
        recall_word[f'top {k}'] = float(hit.mean()) * 100

    # ── Per-writer word-level AP ──────────────────────────────────────────
    unique_writers  = np.unique(labels_np)
    per_writer_ap   = {}
    docs_per_writer = {}

    for w in unique_writers:
        mask     = (labels_np == w) & has_rel
        w_doc_ids = np.unique(doc_ids_np[labels_np == w])
        docs_per_writer[int(w)] = len(w_doc_ids)
        if mask.any():
            per_writer_ap[int(w)] = float(ap_per_query[mask].mean()) * 100

    mean_docs        = float(np.mean(list(docs_per_writer.values())))
    # Mean over writers (each writer contributes equally, regardless of word count)
    writer_mAP       = float(np.mean(list(per_writer_ap.values()))) if per_writer_ap else 0.0

    # ── Per-writer top k (each writer = mean hit rate over its queries) ─
    writer_recall = {}
    for k in ks:
        k_cap        = min(k, N - 1)
        hit_per_word = (topk_labels[:, :k_cap] == labels_np[:, None]).any(axis=1).astype(float)
        per_w_r      = []
        for w in unique_writers:
            mask_w = labels_np == w
            if mask_w.any():
                per_w_r.append(float(hit_per_word[mask_w].mean()))
        writer_recall[f'writer_top {k}'] = float(np.mean(per_w_r)) * 100 if per_w_r else 0.0

    # ── Document-level retrieval ──────────────────────────────────────────
    # Build one embedding per (writer, doc) pair = mean of its word embeddings
    emb_np         = embeddings.numpy()
    unique_doc_keys = []   # (writer_id, doc_id) tuples in order
    doc_embs        = []
    doc_writer_ids  = []

    for w in unique_writers:
        w_mask    = labels_np == w
        w_doc_ids = np.unique(doc_ids_np[w_mask])
        for d in w_doc_ids:
            d_mask = w_mask & (doc_ids_np == d)
            doc_emb = emb_np[d_mask].mean(axis=0)
            # Re-normalise after averaging
            doc_emb = doc_emb / (np.linalg.norm(doc_emb) + 1e-12)
            unique_doc_keys.append((int(w), int(d)))
            doc_embs.append(doc_emb)
            doc_writer_ids.append(int(w))

    doc_embs_t   = torch.tensor(np.stack(doc_embs), dtype=torch.float32)
    doc_labels_t = torch.tensor(doc_writer_ids, dtype=torch.long)
    M            = doc_embs_t.size(0)

    doc_gallery  = doc_embs_t.to(device)
    doc_lbl_t    = doc_labels_t.to(device)
    doc_lbl_np   = doc_labels_t.numpy()

    ap_per_doc  = np.zeros(M, dtype=np.float64)
    n_rel_doc   = np.zeros(M, dtype=np.int64)
    topk_doc    = np.zeros((M, max_k), dtype=np.int64)

    for start in range(0, M, chunk_size):
        end       = min(start + chunk_size, M)
        q_emb_d   = doc_gallery[start:end]
        q_lbl_d   = doc_lbl_t[start:end]

        sim_d = q_emb_d @ doc_gallery.T
        for local_i in range(end - start):
            sim_d[local_i, start + local_i] = -1e9

        sorted_idx_d  = torch.argsort(sim_d, dim=1, descending=True)
        sorted_lbl_d  = doc_lbl_t[sorted_idx_d]
        sorted_lbl_dnp = sorted_lbl_d.cpu().numpy()
        q_lbl_dnp      = q_lbl_d.cpu().numpy()

        for local_i in range(end - start):
            global_i      = start + local_i
            q             = q_lbl_dnp[local_i]
            ranked_labels = sorted_lbl_dnp[local_i, :M-1]
            relevant      = (ranked_labels == q).astype(np.float32)
            n_rel         = int(relevant.sum())
            n_rel_doc[global_i] = n_rel

            if n_rel == 0:
                continue

            cumsum  = np.cumsum(relevant)
            ranks   = np.arange(1, M, dtype=np.float32)
            prec_at = cumsum / ranks
            ap_per_doc[global_i] = (prec_at * relevant).sum() / n_rel

        topk_doc[start:end] = sorted_lbl_dnp[:, :max_k]

    has_rel_doc  = n_rel_doc > 0
    mAP_doc      = float(ap_per_doc[has_rel_doc].mean()) * 100 if has_rel_doc.any() else 0.0

    recall_doc = {}
    for k in ks:
        k_cap = min(k, M - 1)
        hit   = (topk_doc[:, :k_cap] == doc_lbl_np[:, None]).any(axis=1).astype(float)
        recall_doc[f'doc_top {k}'] = float(hit.mean()) * 100

    # Per-writer doc-level AP
    per_writer_doc_ap = {}
    for w in unique_writers:
        mask_w = (doc_lbl_np == w) & has_rel_doc
        if mask_w.any():
            per_writer_doc_ap[int(w)] = float(ap_per_doc[mask_w].mean()) * 100

    writer_doc_mAP = float(np.mean(list(per_writer_doc_ap.values()))) if per_writer_doc_ap else 0.0

    writer_doc_recall = {}
    for k in ks:
        k_cap           = min(k, M - 1)
        hit_per_doc     = (topk_doc[:, :k_cap] == doc_lbl_np[:, None]).any(axis=1).astype(float)
        per_w_r         = []
        for w in unique_writers:
            mask_w = doc_lbl_np == w
            if mask_w.any():
                per_w_r.append(float(hit_per_doc[mask_w].mean()))
        writer_doc_recall[f'writer_doc_top {k}'] = float(np.mean(per_w_r)) * 100 if per_w_r else 0.0

    return {
        # ── Word-level (query = word image) ──────────────────────────────
        'mAP':              mAP_word,
        **recall_word,
        # ── Per-writer averaged (each writer weighted equally) ────────────
        'writer_mAP':       writer_mAP,
        **writer_recall,
        # ── Document-level (query = mean doc embedding) ───────────────────
        'doc_mAP':          mAP_doc,
        **recall_doc,
        'writer_doc_mAP':   writer_doc_mAP,
        **writer_doc_recall,
        # ── Dataset stats ─────────────────────────────────────────────────
        'n_queries':        int(has_rel.sum()),
        'n_writers':        len(per_writer_ap),
        'n_docs':           M,
        'mean_docs':        mean_docs,
        'per_writer_ap':    per_writer_ap,
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class DeepWriter_Train:
    def __init__(self, dataset='CVL', imgtype='png', mode='vertical',
                 seed=42, subfolder=None, train_folder=None,dataset_path='',image_type='binarised',
                 print_label='', batch_size=16):

        set_seed(seed)
        self.dataset = dataset
        self.image_type=image_type
        self.dataset_path = dataset_path
        self.folder      = self.dataset_path + subfolder
        self.labelfolder = self.folder

        if not os.path.exists(self.folder):
            raise FileNotFoundError(
                f'Expected data folder does not exist: {self.folder!r}\n'
                f'  dataset    = {dataset!r}\n'
                f'  subfolder  = {subfolder!r}\n'
                'Please check that --subfolder points to the right location '
                'relative to the base path, and that the CVL data is already '
                'placed there.  The folder must exist before training starts.'
                )

        self.train_folder = self.folder + train_folder
        self.test_folder  = self.folder + '/test/'

        print(print_label or 'real data only')

        self.imgtype    = imgtype
        self.mode       = mode
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scale_size = (64, 128)

        if self.dataset == 'CVL':
            self.imgtype = 'tif'
        elif self.dataset=='bullinger':
            self.imgtype = 'tif'
        
        if self.image_type == 'binarised':
            dt = bindset
        elif self.image_type == 'grayscale':
            dt=dset
            
        os.makedirs('model', exist_ok=True)
        self.model_dir = 'model'

        tag            = (f'GRRNN_WriterIdentification_dataset_{dataset}'
                          f'_model_{mode}_aug_16')
        self.logfile   = tag + '.log'
        self.modelfile = tag
        self.batch_size = batch_size

        # ---- training set (closed-set, uses shared pickle) ----
        train_set = dt.DatasetFromFolder(
            dataset=dataset, labelfolder=self.labelfolder,
            foldername=self.train_folder, imgtype=self.imgtype,
            scale_size=self.scale_size, is_training=True,
            is_retrieval=False,
        )

        # ---- test set (open-set, builds its OWN writer index) ----
        test_set = dt.DatasetFromFolder(
            dataset=dataset, labelfolder=self.labelfolder,
            foldername=self.test_folder, imgtype=self.imgtype,
            scale_size=self.scale_size, is_training=False,
            is_retrieval=True,
        )

        g = torch.Generator(); g.manual_seed(seed)

        self.training_data_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=4, worker_init_fn=seed_worker, generator=g,
            pin_memory=True, persistent_workers=True,
        )
        self.testing_data_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            num_workers=4, worker_init_fn=seed_worker,
            pin_memory=True, persistent_workers=True,
        )

        print(f'\nTrain writers : {train_set.num_writer}')
        print(f'Test  writers : {test_set.num_writer}  '
              f'(disjoint from train — open-set retrieval)\n')

        self.model = net.GrnnNet(
            1, num_classes=train_set.num_writer, mode=mode
        ).to(self.device)

        self.criterion = LabelSomCE()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=0.0001, weight_decay=1e-4)
        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                              step_size=10, gamma=0.5)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    def train(self, epoch):
        self.model.train()
        losses = []

        for batch in self.training_data_loader:
            inputs = batch[0].to(self.device).float()
            target = batch[1].to(self.device).long()

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(inputs), target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        avg = float(np.mean(losses))
        print(f'Training epoch {epoch}  avg loss: {avg:.6f}')
        with open(self.logfile, 'a') as f:
            f.write(f'Training epoch {epoch} avg loss: {avg:.6f}\n')

    # -----------------------------------------------------------------------
    def _check_train_accuracy(self, epoch):
        self.model.train()
        top1 = top5 = total = 0

        with torch.no_grad():
            for batch in self.training_data_loader:
                inputs = batch[0].to(self.device).float()
                target = batch[1].to(self.device).long()
                logits = self.model(inputs)
                r = self._accuracy(logits, target, topk=(1, 5))
                top1  += r[0]; top5 += r[1]; total += inputs.size(0)

        top1 /= total; top5 /= total
        msg = (f'[Train-set check] epoch {epoch}  '
               f'top-1: {top1*100:.2f}  top-5: {top5*100:.2f}')
        print(msg)
        with open(self.logfile, 'a') as f:
            f.write(msg + '\n')

    # -----------------------------------------------------------------------
    # Full open-set retrieval evaluation on the TEST set
    # -----------------------------------------------------------------------
    def _test_retrieval(self, epoch):
        self.model.eval()
        all_emb, all_labels, all_filenames = [], [], []

        with torch.no_grad():
            for batch in self.testing_data_loader:
                inputs    = batch[0].to(self.device).float()
                labels    = batch[1].to(self.device).long()
                filenames = batch[2]                         # writerid-doc-line-word-trans.tif
                emb       = self.model(inputs)
                all_emb.append(emb.cpu())
                all_labels.append(labels.cpu())
                all_filenames.extend(filenames)

        all_emb    = torch.cat(all_emb,    dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = compute_retrieval_metrics(
            all_emb, all_labels, all_filenames,dataset=self.dataset,
            ks=(1, 5, 10), device=self.device, chunk_size=2000,
        )

        # ── Word-level line ───────────────────────────────────────────────
        msg_word = (
            f'[Retrieval/word]  epoch {epoch}  '
            f'mAP: {metrics["mAP"]:.2f}  '
            f'top 1: {metrics["top 1"]:.2f}  '
            f'top 5: {metrics["top 5"]:.2f}  '
            f'top 10: {metrics["top 10"]:.2f}  '
            f'({metrics["n_queries"]} queries / {metrics["n_writers"]} writers / '
            f'mean docs/writer: {metrics["mean_docs"]:.1f})'
        )

        # ── Per-writer averaged line ──────────────────────────────────────
        msg_writer = (
            f'[Retrieval/writer] epoch {epoch}  '
            f'writer_mAP: {metrics["writer_mAP"]:.2f}  '
            f'writer_top 1: {metrics["writer_top 1"]:.2f}  '
            f'writer_top 5: {metrics["writer_top 5"]:.2f}  '
            f'writer_top 10: {metrics["writer_top 10"]:.2f}'
        )

        # ── Document-level line ───────────────────────────────────────────
        msg_doc = (
            f'[Retrieval/doc]   epoch {epoch}  '
            f'doc_mAP: {metrics["doc_mAP"]:.2f}  '
            f'doc_top 1: {metrics["doc_top 1"]:.2f}  '
            f'doc_top 5: {metrics["doc_top 5"]:.2f}  '
            f'doc_top 10: {metrics["doc_top 10"]:.2f}  '
            f'writer_doc_mAP: {metrics["writer_doc_mAP"]:.2f}  '
            f'writer_doc_top 1: {metrics["writer_doc_top 1"]:.2f}  '
            f'writer_doc_top 5: {metrics["writer_doc_top 5"]:.2f}  '
            f'writer_doc_top 10: {metrics["writer_doc_top 10"]:.2f}  '
            f'({metrics["n_docs"]} docs)'
        )

        for msg in (msg_word, msg_writer, msg_doc):
            print(msg)
            with open(self.logfile, 'a') as f:
                f.write(msg + '\n')

        # ── Per-writer breakdown ──────────────────────────────────────────
        per_w = metrics['per_writer_ap']
        if per_w:
            aps   = list(per_w.values())
            worst = sorted(per_w.items(), key=lambda x: x[1])[:5]
            detail = (f'  per-writer AP  mean: {np.mean(aps):.2f}  '
                      f'min: {min(aps):.2f}  max: {max(aps):.2f}  '
                      f'worst 5: {worst}')
            print(detail)
            with open(self.logfile, 'a') as f:
                f.write(detail + '\n')

        return metrics

    # -----------------------------------------------------------------------
    def test(self, epoch, during_train=True):
        if not during_train:
            self.load_model(epoch)
        if during_train:
            self._check_train_accuracy(epoch)
            self.model.train()
        else:
            self._test_retrieval(epoch)

    # -----------------------------------------------------------------------
    def _model_path(self, epoch):
        return os.path.join(self.model_dir,
                            f'{self.modelfile}-model_epoch_{epoch}.pth')

    def checkpoint(self, epoch):
        torch.save(self.model.state_dict(), self._model_path(epoch))

    def load_model(self, epoch):
        self.model.load_state_dict(
            torch.load(self._model_path(epoch), map_location=self.device))
        print(f'Loaded checkpoint epoch {epoch}')

    # -----------------------------------------------------------------------
    def train_loops(self, start_epoch, num_epoch):
        if start_epoch > 0:
            self.load_model(start_epoch - 1)

        for epoch in range(start_epoch, num_epoch):
            self.model.train()
            self.train(epoch)
            self.checkpoint(epoch)

            if epoch % 10 == 0 or epoch == num_epoch - 1:
                self.test(epoch, during_train=False)

            self.scheduler.step()

    # -----------------------------------------------------------------------
    def evaluate_retrieval(self, epoch):
        self.load_model(epoch)
        return self._test_retrieval(epoch)

    # -----------------------------------------------------------------------
    @staticmethod
    def _accuracy(output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            _, pred = output.topk(maxk, 1, True, True)
            correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
            return [correct[:k].reshape(-1).float().sum().item() for k in topk]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path')
    parser.add_argument('--dataset', type=str, default='CVL', help='Dataset CVL or bullinger')
    parser.add_argument('--image_type', type=str, default='binarised', help='Dataloader images type grayscale or binarised')
    parser.add_argument('--subfolder', type=str, default='', help='Path to a subfolder if exists')
    parser.add_argument('--train_folder', type=str, default='/train/', help='trainset folder name')
    parser.add_argument('--print_label',  type=str, default='')
    parser.add_argument('--batch_size',   type=int, default=16)
    parser.add_argument('--eval_only',    type=int, default=-1,
                        help='Skip training; run retrieval eval on this epoch.')
    args = parser.parse_args()

    mod = DeepWriter_Train(
        dataset='CVL',
        mode='vertical',
        seed=42,
        subfolder=args.subfolder,
        train_folder=args.train_folder,
        print_label=args.print_label,
        batch_size=args.batch_size,dataset_path=args.dataset_path,image_type=args.image_type,
    )

    if args.eval_only >= 0:
        mod.evaluate_retrieval(args.eval_only)
    else:
        mod.train_loops(0, 150)