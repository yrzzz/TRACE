# Xenium annotations

These files contain cell IDs and hierarchical cell-type labels, not images,
coordinates, expression matrices, or Visium HD annotations.

- `lung_cancer_annotation.csv.gz` is the original `lung_cancer_annotation.csv`,
  compressed without changing its contents. The main config selects `level_2`.
- `skin_cancer_annotation_no_unknown.csv.gz` is the existing filtered skin
  annotation table. The main config selects `level_3`; unknown cells were already
  excluded in that source file.

Read them directly with `pandas.read_csv(path)`; gzip is detected automatically.
Headers may use `level1` rather than `level_1`; preparation accepts both. The first
index column contains the cell barcode. IDs are specimen-specific and must match
the selected 10x cells and boundary tables.

The manuscript attributes the associated raw public datasets to 10x Genomics:

- https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-skin
- https://www.10xgenomics.com/datasets/xenium-human-lung-cancer-post-xenium-technote

These are author-supplied derived annotations used by the research code, not a
claim of verbatim official 10x cell-type annotations. The label-derivation analysis
is not included. No additional license grant for third-party assets is asserted
by this file. Source and compressed checksums are in `docs/source_manifest.json`.
