# How to get the data
 
All experiments use one public dataset of AFM images of DNA knots and catenanes, released with Holmes et al. (2025). Nothing else needs to be downloaded.
 
## 1. Download
 
1. Open the dataset page on the University of Sheffield repository (ORDA):
   https://orda.shef.ac.uk/articles/dataset/DNA_knots_catenanes_and_replication_intermediates_via_high-resolution_AFM/27143238
2. Click **Download all**, or download the file **Under or over paper data.zip**.
3. Unzip it anywhere on your computer.
## 2. Find the folder we use
 
Inside the unzipped folder, go to:
 
```
Data/knots_and_catenanes/AFM/knots_and_cats
```
 
This is the only folder the code reads. Copy (or move) it so it sits next to the `code` folder, like this:
 
```
project/
├── code/                  <- the Python scripts
└── data/
    └── knots_and_cats/    <- the folder from step 2
```
 
## 3. Check that it looks right
 
`knots_and_cats` should contain one folder per sample:
 
| Folder | Knot type | Images |
| --- | --- | --- |
| `trefoil` | 3₁ (trefoil knot) | 39 |
| `torus` | 5₁ (torus knot) | 43 |
| `twist` | 5₂ (twist knot) | 39 |
| `cat_supercoiled` | 4²₁ catenane, supercoiled | 102 |
| `cat_nicked` | 4²₁ catenane, nicked | 98 |
| `unknot_plasmid_sup` | 0₁ unknot, supercoiled | 38 |
| `unknot_plasmid_nic` | 0₁ unknot, nicked | 43 |
| **Total** | | **402** |
 
Each sample folder has a `processed/` folder with one `.topostats` file per image. These files hold everything the code needs: the height image, the pixel size, the molecule masks, the skeletons and TopoStats' crossing data. The top of `knots_and_cats` also has `all_mol_statistics.csv`, which holds TopoStats' knot-type call for every molecule. It is used only for the TopoStats baseline (Section 3.3 of the paper).
 
Then run, from inside `code/`:
 
```
python extract_crops.py --root ../data/knots_and_cats --check
```
 
If the data is found, it prints `found 402 .topostats files`, the number of files per sample (it should match the table above) and the details of the first file. If it prints `Nothing found`, the path is wrong. `--root` must point to the `knots_and_cats` folder itself, not to a folder above it.
 
## 5. Things to know about the data
 
- **Imaging sessions.** The imaging date is the first part of each file name (for example `20231011...`). There are 20 dates. The code uses them to keep whole sessions out of training.
- **Two imaging conditions.** 2022 images were taken in Ni²⁺ buffer, 2023 images in Mg²⁺/Ni²⁺ buffer.
- **Pixel size varies** from 0.29 to 0.98 nm per pixel. The code reads it from each file.
- **One oddly named file.** The `trefoil` folder contains an image named `3NTwistKnot`. It is still a trefoil.
- **No raw instrument files.** The dataset holds processed (flattened) images, not the original microscope files.
- **Unknot folders have a different name in one file.** In `all_node_stats.json` the unknot samples are called `cat_plasmid_nic` and `cat_plasmid_sup`. The code does not use this file.
## 6. How to cite the data
 
Gamill, M., Holmes, E., Catley, T., Wiggins, L., Pyne, A., Whittle, S., Shephard, N., Bryant, H., Provan, J., Colloms, S. D., Rusková, R., Main, K., Gilhooly, N., Gambus, A., & Račko, D. (2025). *DNA knots, catenanes, and replication intermediates via high-resolution AFM* [Data set]. The University of Sheffield.
 
Holmes, E. P., Gamill, M. C., Provan, J. I., Wiggins, L., Rusková, R., Whittle, S., Catley, T. E., Main, K. H. S., Shephard, N., Bryant, H. E., Gilhooly, N. S., Gambus, A., Račko, D., Colloms, S. D., & Pyne, A. L. B. (2025). Quantifying complexity in DNA structures with high resolution atomic force microscopy. *Nature Communications, 16*, Article 5482. https://doi.org/10.1038/s41467-025-60559-x