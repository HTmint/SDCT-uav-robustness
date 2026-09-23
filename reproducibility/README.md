# Reproducibility Materials

This directory contains the fixed split lists, dataset YAML files, generation manifests, and paper-facing result tables used by the manuscript. Dataset images, labels, checkpoints, and raw runtime logs are not included.

## Data layout

Obtain the source datasets from their cited sources and place the prepared files under `data/dart_dut/` using the directory names referenced by the split lists. For the external target-domain evaluation, place the supplied DUT-Adv package under `data/DUT-Dve/` and the grayscale test copy under `data/dut_dve_zero_shot/gray_test/`. These datasets are not redistributed here.

The YAML files in `yamls/` use the repository root as their dataset root. Image-list entries are relative to the repository root, so run commands from the repository root. The lists record file names and ordering; they do not contain image data.

## Included files

- `splits/`: clean train/validation/test lists, fixed method-specific training lists, and clean, seen, ordinary held-out, and strong held-out test lists.
- `yamls/`: relative-path dataset configurations for the training methods and evaluation conditions.
- `manifests/`: fixed SDCT/Random-Aug, RandAugment-Gray, AugMix-Input-Gray, SDCT ablation, strong-held-out, and DUT-Adv grayscale-test manifests. CSV manifests are gzip-compressed; decompress a copy with `gzip -dk <file.csv.gz>` before reading them with software that does not support gzip streams.
- `tables/`: paper-facing result tables. The deployment table preserves the values reported in the manuscript.

Machine-specific absolute paths were replaced with repository-relative paths. This path normalization does not alter sample ordering, degradation parameters, image dimensions, label hashes, or result values.

## Regenerating data

Follow the commands in the repository `README.md` to build the grayscale benchmark and SDCT data. Use the scripts under `tools/` to generate DART, generic augmentation baselines, ablations, and held-out degradation suites. The fixed manifests document the realizations used for the reported experiments; regenerating with the same seed is not a substitute for using the released manifest when exact sample ordering is required.

Training and detector evaluation require the project-modified Ultralytics 8.3.138 code described in the repository `README.md`. The framework source, datasets, model weights, and deployment logs are not bundled.
