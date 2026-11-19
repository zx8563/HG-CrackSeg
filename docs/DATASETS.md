# Steelcrack

The experiments use the public Steelcrack dataset without changing its images,
annotations, resolution or official split. Download it from the authors'
[Civil-dataset repository](https://github.com/hzlbbfrog/Civil-dataset), which
provides the download links and dataset license information.

| Split | Images | Resolution |
| --- | ---: | --- |
| Train | 3300 | 512 × 512 |
| Validation | 525 | 512 × 512 |
| Test | 530 | 512 × 512 |

Place each split's RGB images in `images/` and binary masks in `masks/`, using
the same filenames, including extensions. Labels are background 0 and crack
255. The loader uses values greater than 127 as foreground; masks stored as
0/1 PNG files are not compatible without conversion to 0/255.

`manifests/steelcrack.csv` lists the exact split-relative filenames and SHA-256
checksums used in the experiments. Check the downloaded files for the expected
pairing, counts and image format, and optionally export their checksums:

```shell
python tools/check_data.py --root Steelcrack
python tools/check_data.py --root Steelcrack --manifest outputs/downloaded_data.csv
```

The first command checks pairing, counts, dimensions and mask format; it does
not compare checksums against the reference manifest. To verify file identity,
compare the exported manifest with `manifests/steelcrack.csv`.

File checksums concern the actual encoded files. Re-encoding an otherwise
identical image can change its checksum. Data loading preserves the original
512 × 512 resolution.

## Citation

He, Z., Chen, W., Zhang, J., and Wang, Y.-H. (2024). Crack segmentation on steel
structures using boundary guidance model. *Automation in Construction*, 162,
105354. https://doi.org/10.1016/j.autcon.2024.105354

```bibtex
@article{HE2024BGCrack,
  title = {Crack segmentation on steel structures using boundary guidance model},
  author = {He, Zhili and Chen, Wang and Zhang, Jian and Wang, Yu-Hsing},
  journal = {Automation in Construction},
  volume = {162},
  pages = {105354},
  year = {2024},
  doi = {10.1016/j.autcon.2024.105354}
}
```
