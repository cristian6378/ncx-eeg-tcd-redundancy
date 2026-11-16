# Data

Neither dataset is redistributed in this repository. Download both from
figshare and place them as shown below. Both are openly licensed at source;
check the licence on each record before redistributing anything yourself.

## Expected layout

```
data/
├── 21679035/                      # EEG deposit, unzipped
│   ├── task-motor-imagery_channels.tsv
│   ├── dataset_description.json
│   └── sourcedata/
│       ├── sub-01/sub-01_task-motor-imagery_eeg.mat
│       ├── sub-02/sub-02_task-motor-imagery_eeg.mat
│       └── ...                    # 50 subjects, ~1.8 GB total
└── cluster_data.csv               # TCD deposit, single table
```

## 1. EEG — motor imagery in acute stroke

**[dataset]** Liu, H., & Lv, X. (2022).
*EEG datasets of stroke patients.* figshare.
<https://doi.org/10.6084/m9.figshare.21679035.v5>

Data descriptor: Liu et al. (2024), *Scientific Data* **11**, 131.
<https://doi.org/10.1038/s41597-023-02787-8>

- 50 acute stroke patients, 40 trials each (2000 trials), 500 Hz.
- MATLAB v5 `.mat` files, readable with `scipy.io.loadmat`; no HDF5 reader needed.
- The deposit stores 33 traces per trial while the sidecar names 31 and the
  JSON declares 29 EEG channels. `ncx/io.py` resolves this from the signals —
  a zero-variance channel is the recording reference, a near-integer channel
  with few distinct values is the event marker, and the high-variance trailing
  channels are the EOG pair — and reports what it found rather than assuming
  an index layout.
- Only the 4 s imagery window (2.0–6.0 s) of each 8 s trial is analysed; the
  cue and break periods are intervals in which the participant was by design
  not performing the task.

## 2. Transcranial Doppler — waveform morphology

**[dataset]** Thorpe, S. G. (2019).
*TCD Clustering Data.* figshare.
<https://doi.org/10.6084/m9.figshare.9037388>

Publication: Thorpe et al. (2020), *PLOS ONE* **15**(2), e0228642.
<https://doi.org/10.1371/journal.pone.0228642>

- 996 middle cerebral artery insonations, 105 patients, 120 sessions,
  depths 45–60 mm, labelled LVO / in-hospital control / out-of-hospital control.
- 951 records from 104 patients enter the analysis; the remainder lack a
  bilateral counterpart, which the four asymmetry indices require.
- The deposit ships **derived per-insonation descriptors, not envelope time
  series**. This is why the TCD taxonomy has no complexity or coupling family:
  entropy, fractal dimension and within-beat timing cannot be computed from
  tabulated values, and the pipeline does not pretend otherwise.
- `curvature` is used as released. The source publication refers to a velocity
  curvature index from earlier work but gives no formula, and the deposit
  carries no specification, so that one descriptor cannot be recomputed from
  raw envelopes.

## The cohorts are disjoint

No patient appears in both deposits. No fusion model is fitted anywhere in
this repository, and concatenating features across the two would manufacture
exactly the kind of spurious structure the study is about.
