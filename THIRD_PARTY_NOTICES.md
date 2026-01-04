# Third party notices

This repository contains or references third party resources. Unless explicitly
stated, third party content is not covered by the repository licenses.

## UniProt data and sequences

Some FASTA files and metadata were retrieved from the UniProt REST API.
Users of this repository should comply with UniProt terms and provide appropriate
attribution when redistributing UniProt derived content.

Resources:
- UniProt license and terms of use: https://www.uniprot.org/help/license
- UniProt REST API: https://rest.uniprot.org/

## ESM2 model weights

ESM2 embeddings are generated using the open source fair esm package.
The model weights are downloaded at runtime from the public FAIR ESM endpoints
and are not redistributed in this repository.

Resources:
- fair esm repository: https://github.com/facebookresearch/esm
- Model checkpoint hosting used by fair esm: https://dl.fbaipublicfiles.com/fair-esm/

## TabPFN

TabPFN is used for selected benchmark comparisons. TabPFN is not vendored in this
repository and must be installed separately by the user. Users should review and
comply with the TabPFN license terms applicable to their intended use.

Resource:
- TabPFN repository: https://github.com/PriorLabs/TabPFN

## Published datasets

This repository includes processed datasets that were originally published by
other authors and are reused here for benchmarking and reproducibility. Please
cite the original publications when using these datasets.

Included files include:
- supplementary/doubling_time_package_v2/inputs/flamholz_dataset_S1.csv
- supplementary/doubling_time_package_v2/doubling_time_outputs/cyano_doubling_times_yu2015_srep08132_table1_long.csv

The original full text PDF for Yu et al. 2015 was intentionally not included in
this public deposition package.

Users are responsible for checking any dataset specific reuse terms specified by
the original publishers or authors.
