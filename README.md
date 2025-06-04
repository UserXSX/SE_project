# SE_project

## Getting Started🚀

### Dataset

The `MEDAF/datasets/data` directory contains [BODMAS](https://whyisyoung.github.io/BODMAS/) datasets for three distinct experimental splits. Each split is represented by a subdirectory named in the format `X-Y`, where `X` signifies the number of seen families and `Y` signifies the number of unseen families. For example, the `5-177` directory corresponds to a split with 5 seen families and 177 unseen families.

### Train

```bash
python osr_main.py -g {GPU_ID} -d {SPLIT}
```

or

```bash
./run.sh
```

## References

[MEDAF](https://github.com/Vanixxz/MEDAF)