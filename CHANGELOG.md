# Changelog

## [Unreleased]
### Upgrade PyTorch to support recent GPUs
- upgrade torch 1.10.0 -> 2.7.1 (CUDA 12.8 wheels), torchvision 0.11.1 -> 0.22.1, torch-geometric 2.4.0 -> 2.6.1, captum 0.6.0 -> 0.8.0, adds support for Blackwell GPUs (sm_120)
- Dockerfile: new build arg `TORCH_VARIANT` (`cu128` by default, `cpu` for a smaller CPU-only image)
- run FastSurfer v1.1.2 with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` since torch>=2.6 defaults to `torch.load(weights_only=True)`

## [v2.2.7] - 2026-08-25
### Fix issues + more robust parallelisation
- merge with fix from Cornelius Kronlage https://github.com/ckronlage/meld_graph/tree/fixes_v2.2.7
- add check .nii.gz, update FLAIR in qc plot, change harmonised name in saliency plot

## [v2.2.7] - 2026-07-14
### Update report with info MELD + create HTML report
- add link to github and MELD project wedbpage on MELD report
- add a function to generate a HTML viewer to display predicted clusters on top of T1 and FLAIR scan + pial and white surfaces

## [v2.2.5] - 2026-02-20
### Change in data downloading + fix in FreeSurfer segmentation
- meld_data integrated into GitHub as Figshare platform preventing automated downloading of data
- added Global Expert options file in $SUBJECT_DIR to fix known bug in Freesurfer pial surface reconstruction when FLAIR is used
- save hemisphere information in csv during report creation
- fix issue bidslayout to enable flexible config file

## [v2.2.4_gpu] - 2025-11-20
### Added security
- added documentation with a first-page documentation in MELD Patient Report
- added security layer with a meld license needed to run the tool
- separate gpu/cpu version of MELD Graph (this is the GPU version)

## [v2.2.4] - 2025-11-20
### Added security
- added documentation with a first-page documentation in MELD Patient Report
- added security layer with a meld license needed to run the tool
- separate gpu/cpu version of MELD Graph (this is the CPU version)


## [v2.2.3] - 2025-10-20
### Fixed
- Fixed an issue with using GPU in the docker. Dockerfile and environment.yml modified to allow GPU use in docker and enable native installation of MELD Graph
- Removed the requirement to specify scanner strength in demographics_file.csv for harmonisation. This resolves errors when scanner strength wasn’t 3T
- Instructions for computer RAM requirement ; Clarification on FAQ about FLAIR scan usage
- Minor code cleanup for improved stability and maintainability


## [v2.2.2] - 2025-04-29
### Fixed
- Fixed an issue in the backward registration process (from fsaverage_sym surface to native T1 volume) that caused a slight spatial shift (~a few millimeters) in lesion predictions on the right hemisphere.
- Removed the requirement to specify scanner strength in demographics_file.csv for harmonisation. This resolves errors when scanner strength wasn’t 3T
- Fixed the standalone script for merging predictions with the T1 volume, which previously failed when handling predictions with different values for salient vs. non-salient vertices.
- Instructions for use of Docker Desktop ; Instructions to join the mailing list; Clarification on FAQ
- Minor code cleanup for improved stability and maintainability

### Notes
- Require to download the test data again
- Docker image `MELDproject/meld_graph:v2.2.2` is available.
- `latest` tag now points to this version.

## [v2.2.1] - 2024-11-28
- Initial stable release of MELD Graph package.