# BullingerDB Writer Identification (closed set)

Word-level writer identification split for the SBullinger subset (596 writers).
The same writers appear in both train and test; only the word images differ.

| File | Word images |
| --- | --- |
| train.txt | 362,114 |
| test.txt | 90,526 |

Each line names one word image, prefixed with the split, for example:

```
train/1001_11118_0001_w001_Humanissimo.png
```

The writer id is the first field of the filename (before the first underscore).
To rebuild the sets, download BullingerDB and take the listed images.
