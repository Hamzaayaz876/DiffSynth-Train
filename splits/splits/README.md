# Writer Identification and Retrieval Splits (CVL and BullingerDB)

These are the train and test splits used in a University of Bern master's thesis
on word-level writer identification and retrieval, where GR-RNN is trained with
the help of DiffusionPen-generated synthetic handwriting and evaluated on a
contemporary dataset (CVL) and a historical one (BullingerDB).

The files list the word images (or writer ids) that make up each set. They do
not contain any image data. To use them, download the original datasets from the
links below and select the samples named in each file.

## Structure

```
splits/
|-- Bullinger/
|   |-- excluded_writers.txt        200 writer ids dropped to form SBullinger
|   |-- Writer Identification/      closed-set split (train.txt, test.txt)
|   |-- Writer Retrieval/           open-set writer lists (train / test)
|-- cvl/
    |-- Writer Identification/      closed-set split (train.txt, test.txt)
    |-- Writer Retrieval/           README only (see inside)
```

## Datasets

- CVL Database (Kleber, Fiel, Diem, Sablatnig; ICDAR 2013), 311 writers.
  https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/
- BullingerDB (Peer et al.), 796 writers, from the 16th-century Heinrich
  Bullinger correspondence.
  Paper: https://arxiv.org/abs/2605.30235
  Bullinger Digital project: https://www.bullinger-digital.ch

## File formats

- Identification (`Writer Identification/train.txt`, `test.txt`): one word image
  per line, prefixed with the split, for example `train/0001-1-0-0-Imagine.tif`
  (CVL) or `train/1001_11118_0001_w001_Humanissimo.png` (BullingerDB). The writer
  id is the first field of the filename (before the first `-` for CVL, `_` for
  BullingerDB).
- Retrieval (`Writer Retrieval/retrieval_train_writers.txt`,
  `retrieval_test_writers.txt`): one writer id per line. The train and test writer
  groups are disjoint, so the test writers are unseen during training.
- `excluded_writers.txt`: one writer id per line.

## Split sizes

| Dataset | Task | Train | Test |
| --- | --- | --- | --- |
| CVL | Identification (word images) | 79,790 | 20,114 |
| BullingerDB (SBullinger) | Identification (word images) | 362,114 | 90,526 |
| BullingerDB (SBullinger) | Retrieval (writers) | 50 | 546 |

CVL identification covers 310 writers in both train and test (closed set);
SBullinger identification covers 596 writers in both.


## Licence and citation

The CVL Database is distributed under a Creative Commons
Attribution-NonCommercial 3.0 licence; please cite Kleber et al. (2013). For
BullingerDB, please cite Peer et al. Only these split lists are shared here; the
image data belongs to the original datasets and must be obtained from them under
their own terms.

