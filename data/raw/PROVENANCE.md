# Data provenance (retrieved 2026-09-28 via `git clone --depth 1` over HTTPS)

| Source | Repository | Commit | Commit date | Licence |
|---|---|---|---|---|
| ENIGMA Toolbox (summary statistics, HCP connectivity, fsaverage5 surfaces/parcellations) | https://github.com/MICA-MNI/ENIGMA | b08974b55243060cbc1fad12c87048037446e8f7 | 2026-01-15 | see enigma/LICENSE |
| Hansen et al. 2022 receptor atlas (PET receptor matrix, MEG power, Mesulam classes, Lausanne region info) | https://github.com/netneurolab/hansen_receptors | f8b41da92a733f99368c1f39d4959731aa1316c1 | 2022-10-27 | CC BY-NC-SA 4.0 (hansen/LICENSE.md) |

Kaggle, UCI, Zenodo, Mendeley Data, OpenNeuro, OSF, figshare, PhysioNet, HuggingFace and OpenML were not reachable
from the analysis environment (organisational egress policy, HTTP 403 on CONNECT); no attempt was made to circumvent this.
SHA-256 checksums of every file used are in SHA256SUMS.txt.

## Additions for version 3 (retrieved 2026-09-29, same ENIGMA commit b08974b)
- `enigma/conte69_32k_{lh,rh}_sphere.gii` (fs_LR 32k spheres) and `enigma/aparc_conte69.csv` (Desikan-Killiany labels on fs_LR),
  used for the fs_LR parcel spin.
- `enigma/glasser_360_fsa5.csv`, `enigma/glasser_360_conte69.csv`, used to quantify hemispheric sphere-frame misalignment for a
  second, bilaterally symmetric parcellation and for the fine-parcellation simulations (Supplementary Fig. 13).
- `enigma/enigmatoolbox_permutation_testing.py` (BSD-3-Clause), a verbatim copy of the ENIGMA Toolbox spin-test module, used to
  reproduce the toolbox's parcel-level spin procedure exactly (greedy re-assignment, reflected rotation, directional p-value).

Repositories inspected (code only, not used as data) to document how common tools implement parcel-level spins:
netneurolab/neuromaps (commit ffcc2e0f657943ce00a1b6a968396f32250e495c) and netneurolab/hansen_receptors (commit f8b41da).
- `enigma/fsa5_{lh,rh}_aparc.annot` (ENIGMA Toolbox annotation files used by `spin_test`), used only to verify that the re-implemented centroid extraction matches the toolbox.
