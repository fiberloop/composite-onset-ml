# Where does a composite start to fail?

Detection and clustering of nonlinear events in full-field strain data. CS-433 (Machine
Learning, EPFL) project, 2026.

## The problem

Notched carbon-fibre composite (CFR-PEEK) coupons were loaded to failure while two cameras
filmed one face. Digital image correlation (DIC) turns the images into a displacement and a
strain field at every load stage. Well before the coupon breaks, individual points stop
following a straight line of strain against load. The reasons differ: plasticity of the
matrix, cracking, delamination, load taken over from a failing neighbourhood, or measurement
artefacts. Each leaves its own signature in space and time.

Treating each point of the face as a multivariate time series (three strain components and a
displacement against the load), the goals are:

1. **Onset.** For every point, the load stage at which the response stops being linear
   (change-point detection), benchmarked against classical methods.
2. **Families.** Unsupervised learning on what happens after the onset, together with where it
   happens, to find distinct families of events.

What the data can and cannot decide: the eight tests are monotonic and stop at the peak, so a
smooth softening (plasticity, or diffuse damage, which unload differently) cannot be told apart
here, and no image shows the final fracture. A first unsupervised try on the eight tests (kickoff
deck) finds families that differ mainly in the shape of the shear departure, in coherent zones of
the face; whether they are mechanisms is part of the question.

Candidate methods: piecewise-linear fits, CUSUM and PELT with a spatial smoothing of the onset
map; feature-based clustering (UMAP + HDBSCAN) and DTW k-means on the post-onset curves; a
learned embedding (autoencoder) as a stretch goal.

There are no labels. Evaluation ideas: local onset against the nonlinearity of the global
load-displacement curve, onset maps against the fracture path, coherence across loading
configurations, robustness to added noise, and the purity of the clusters on a few points
labelled by an expert.

No background in composites is needed.

## The data

One HDF5 file per test, `<test>.h5`, eight tests (43 to 83 MB each, 513 MB in all). A test is
named `ARC-00_<angle>_s<n>`: the coupon was pulled in an Arcan fixture at `<angle>` degrees
(`m` for minus: 0, 45, -45, 90, -90), and `s<n>` tells two coupons at the same angle apart. The
list, with what is particular to each, is in [docs/data.md](docs/data.md#the-tests). The data
is **not** in this repository: you get the files separately. Put them in `data/` (ignored by
git), or point the environment variable `ONSETML_DATA` at the folder that holds them.

The field was measured by a finite-element DIC: the face is a triangular mesh, the
displacement is known at its nodes and the strain is constant on each triangle. One "point" of
the face is therefore one element (7100 to 7600 of them a test, about 0.25 to 0.6 mm wide).
What each variable holds, with its units, is in [docs/data.md](docs/data.md).

```python
from onsetml import data

ds = data.load("ARC-00_m45_s2")  # an xarray.Dataset, lazily read
load, strain = data.history(ds, 1234)  # the load (n_img,) and strain (n_img, 3) of element 1234
field = data.field(ds, image=300)  # the strain of every element at image 300 (n_el, 3)
```

`h5py` reads the same file directly if you prefer it (`h5py.File(path)["strain"]`).

The null model of the kickoff deck is in `onsetml.null`: the line of each element and component
fitted below 0.3 F_max, the departure from it, the noise covariance at rest and the distance psi,
the thresholds from their laws under H0 (Student for one component, Hotelling T^2 for three, both
because the noise is estimated from few static images), the noise under load, and a CUSUM:

```python
from onsetml import null

images = data.loading_images(ds)
f = ds["load_over_max"].sel(image=images).values
X = ds["strain"].sel(image=images).values          # (n_img, n_el, 3)
eta = null.departure(f, X)                         # strain minus its line
S, K = null.static_covariance(ds)                  # 3x3 noise per element, from K images
psi = null.psi(eta, S)                             # the three components together
thr = null.hotelling_threshold(1e-3, nu=K - 1)     # choose alpha, then the threshold
```

## Getting started

With Python 3.11 or later (tested on 3.12):

```bash
git clone https://github.com/fiberloop/composite-onset-ml.git
cd composite-onset-ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/first_look.py ARC-00_m45_s2
```

`first_look.py` writes a few figures to `outputs/`: the load curve, the strain field at the
peak, and the history of a few elements. Open `notebooks/00_explore_the_data.ipynb` with the
`.venv` interpreter as its kernel (VS Code and Jupyter both find it); it runs in about 15 s. Extras for the methods: `.[changepoint]` (ruptures),
`.[cluster]` (umap-learn, tslearn), `.[deep]` (PyTorch, for the stretch goal).

## Layout

```
src/onsetml/      the package: data.py (reading), plotting.py (fields on the mesh),
                  null.py (the null model, its laws under H0, a CUSUM)
scripts/          entry points (first_look.py)
notebooks/        your exploration notebooks
tests/            tests on small synthetic files (never on the real data)
docs/data.md      the file format
data/             the HDF5 files (ignored by git)
outputs/          figures and results (ignored by git)
```

## Rules of the repository

- Never commit data or large outputs.
- Code that a result depends on goes into `src/onsetml/` with a test, not only into a notebook.
- Fix the random seeds of every method so that a figure can be reproduced.
