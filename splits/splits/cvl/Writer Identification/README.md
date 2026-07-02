# CVL Writer Identification (closed set)

Word-level writer identification split for CVL, covering 310 writers. The same
writers appear in both train and test; only the word images differ.

| File | Word images |
| --- | --- |
| train.txt | 79,790 |
| test.txt | 20,114 |

Each line names one word image, prefixed with the split, for example:

```
train/0001-1-0-0-Imagine.tif
```

The writer id is the first field of the filename (before the first hyphen). To
rebuild the sets, download the CVL Database and take the listed images.
