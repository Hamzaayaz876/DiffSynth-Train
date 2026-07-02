# BullingerDB Writer Retrieval (open set)

Open-set writer split for the SBullinger subset (596 writers). The two writer
groups are disjoint, so the test writers are never seen during training.

| File | Writers |
| --- | --- |
| retrieval_train_writers.txt | 50 |
| retrieval_test_writers.txt | 546 |

Each line is a single writer id. All word images belonging to a training writer
form the retrieval training pool; all images of a test writer form the (unseen)
evaluation pool. To rebuild the sets, download BullingerDB and select the images
whose filename begins with one of the listed writer ids.
