# Changelog

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