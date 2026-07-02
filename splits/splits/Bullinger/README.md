# BullingerDB Splits

Splits for the BullingerDB dataset (Peer et al.), a historical collection of 796
writers from the 16th-century Heinrich Bullinger correspondence.

Dataset paper: https://arxiv.org/abs/2605.30235
Bullinger Digital project: https://www.bullinger-digital.ch

## Contents

- `excluded_writers.txt` : the 200 writers with the most images, removed from the
  full 796-writer set to form the SBullinger subset (596 writers) used in the
  thesis. One writer id per line.
- `Writer Identification/` : closed-set identification split (train.txt, test.txt).
- `Writer Retrieval/` : open-set retrieval writer lists (train / test writers).

Filenames follow `writerid_documentid_pageid_wordid_transcription.png`, so the
writer id is the part before the first underscore.

See the top-level README for full details, formats, and licensing.
