# Changelog

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