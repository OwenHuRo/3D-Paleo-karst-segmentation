# Data layout

Generated synthetic volumes are expected in the following structure:

```text
Data/
├── GroundTruth/
│   └── synthetic_seismic_final_<id>.dat
└── Label/
    └── synthetic_seismic_final_<id>.dat
```

Each file is a headerless, C-order `float32` volume. The manuscript experiments
use a source shape of `256 x 256 x 256`; training and evaluation load
`128 x 128 x 128` crops. A seismic/label pair with this source shape occupies
128 MiB, so all 200 pairs require approximately 25 GiB.

We provide `ConstructData/CreateProcess.py` so that the synthetic dataset used
by this project can be generated locally. The field SEG-Y volume and drilling
records used in our manuscript cannot be uploaded or redistributed because
they are protected by confidentiality agreements with the data owner.

We therefore provide the complete field-data inference workflow but not the
corresponding field dataset. To use `Test_Reality_All.py`, users must supply a
SEG-Y or float32 DAT volume for which they have the appropriate access and
usage rights.

