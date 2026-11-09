# The data files

One HDF5 file per test (netCDF4 layout, so `xarray` and `h5py` both read it). Each file holds
one global (finite-element) stereo DIC of one face of one coupon, from the reference image
(image 0, at rest) to the image before the coupon breaks.

## The tests

All eight are the same notched CFR-PEEK coupon design, pulled to failure in an Arcan fixture at
the angle in the name (`m` for minus). Each coupon has its own mesh, so element numbers do not
carry from one test to another; positions (`element_x1`, `element_x3`) do, roughly.

| test | images | elements | peak [kN] | static images | to know |
| --- | --- | --- | --- | --- | --- |
| ARC-00_s0 | 317 | 7424 | 2.01 | 2 usable (5, 10) | only every 5th image was correlated (0, 5, 10, ... 1552); few static images, so its noise estimate is rough |
| ARC-00_45_s0 | 330 | 7299 | 2.55 | 40 | |
| ARC-00_45_s2 | 217 | 7269 | 2.75 | 38 | |
| ARC-00_90_s0 | 279 | 7171 | 1.70 | 40 | |
| ARC-00_90_s2 | 278 | 7132 | 1.52 | 38 | |
| ARC-00_m45_s0 | 393 | 7558 | 2.08 | 9 | 11 blurred frames and several light changes |
| ARC-00_m45_s2 | 326 | 7377 | 2.10 | 40 | the light dips by about 5 percent over images 127 to 135 |
| ARC-00_m90_s0 | 376 | 7303 | 2.30 | 40 | 3 blurred frames (142 to 144) |

The numbers below, when they are given for one test, are for ARC-00_m45_s2.

## How the field was measured

All eight files come from the same solver (onsetgp's `gdic`, runs of 5 and 6 November 2026)
with the same settings, and their meshes were all made the same way, from each coupon's own
outline (so the elements differ from test to test, the mesh settings do not). Only
`image_step` differs (5 on ARC-00_s0, 1 elsewhere). Each file
carries them in its attribute `gdic_settings` (JSON: `json.loads(ds.attrs["gdic_settings"])`),
with the regularisation weight actually used (`source_run_lambda`, derived from `l_reg_mm` and
the images of the test), the image scale (`source_run_px_per_mm`) and the solver's commit
(`source_run_git_sha`). The table lists what matters to you.

| setting | value | what it means |
| --- | --- | --- |
| stage | stereo | one 3D displacement per node, solved on both cameras together (not a 2D DIC on one camera) |
| mesh | `T3_graded_0p25_to_0p60mm` | linear triangles, about 0.25 mm near the notches to 0.6 mm far from them; about 7100 to 7600 elements and 3800 to 3900 nodes |
| image size | about 60 px per mm | so an element spans about 15 to 36 pixels |
| criterion | `ssd` | sum of squared grey-level differences between the reference and the deformed image |
| grey correction | on | a gain and an offset per camera and image (`grey_correction` in the file); none fitted where an element has fewer than 12 points or a contrast below 3 grey levels |
| regulariser | `second_gradient`, `l_reg_mm` = 1.0 | penalises curvature of the displacement below about 1 mm wavelength: a low-pass filter of about 1 mm, so features smaller than that are smoothed |
| epipolar shift | on | camera 1 may shift across its epipolar line per image, which absorbs a small drift of the calibration |
| convergence | largest nodal step below 0.001 px, or relative change of the residual below 1e-5; at most 50 iterations | every image of every file converged |
| start of each image | extrapolated from the two previous images | |
| reference | image 0 | every image is correlated against it, not against the previous one |
| images solved | every image up to the one before failure (every 5th on ARC-00_s0) | |
| blurred frames | a frame 5 percent less sharp than the running median of 9 frames of its camera | flagged in `mask_blurred`, still solved |

The rigid body motion is removed afterwards, in the export (see "Things to know"). The strain
is the Green-Lagrange strain of the in-plane displacement gradient, constant on each triangle.

## Dimensions

| dimension | size (ARC-00_m45_s2) | what it is |
| --- | --- | --- |
| `image` | 326 | load stages, one image pair every 0.5 s; image 0 is the reference; the coordinate is the image number (on ARC-00_s0 every 5th number only) |
| `node` | 3891 | nodes of the triangular mesh of the face |
| `element` | 7377 | triangles; one strain per triangle |
| `direction` | 3 | `u1`, `u3` (in the face), `w` (out of the face) |
| `component` | 3 | `e11`, `e33`, `g13` |
| `camera` | 2 | `camera0`, `camera1` |
| `corner` | 3 | the three nodes of a triangle |

## Frame and units

- Axes `x1` and `x3` lie in the imaged face of the coupon, which cuts through the laminate (the
  plies are seen edge-on): 1 is the laminate's in-plane axis along the coupon's straight edges,
  3 its thickness direction. `w` points out of the face. Lengths in mm.
- Strain is the Green-Lagrange strain from the reference image, dimensionless
  (0.01 = 1 percent); `g13` is the engineering shear (twice the tensor shear).
- Load in kN, time in s, residual in grey levels of the 8-bit images.

## Variables

**Fields**

| variable | dimensions | what it is |
| --- | --- | --- |
| `displacement` | image, node, direction | displacement of each node [mm], with the rigid body motion of the coupon removed (see below) |
| `rigid_motion` | image, rigid_mode | the rigid body motion removed from it: translations `t1_mm`, `t3_mm` and rotation `rotation_rad` in the face, mean out-of-plane motion `w_mm` and tilts `tilt1_rad`, `tilt3_rad` |
| `strain` | image, element, component | strain of each element |
| `residual_rms` | image, camera, element | how badly the measured motion explains the images in that element [grey levels]: about 2.5 is the noise of the images; a large value means something the smooth field cannot follow (a crack, the edge of the face, a reflection) |
| `residual_over_floor` | image, element | the same mismatch, sharper: each pixel's residual with its fixed part removed (its mean over the static images), its rms over the element, divided by the noise floor of a moving face; the mean of the two cameras. 1 is noise only, 2 is twice the noise: far from the notches it is about 0.85 at rest (a face that does not move is a little less noisy) and 1.0 to 1.1 under load |
| `residual_frac3` | image, element | the fraction of the element's pixels whose residual (fixed part removed) is beyond 3 times that floor: about 0.002 at rest and 0.01 under load where nothing happens; a thin crack crossing the element raises it more than it raises the rms |

**Geometry**

| variable | dimensions | what it is |
| --- | --- | --- |
| `node_x1`, `node_x3` | node | position of each node in the face [mm] |
| `node_X`, `node_Y`, `node_Z` | node | the same in the cameras' frame [mm] |
| `node_on_boundary` | node | the node lies on the outline of the face |
| `elements` | element, corner | the three nodes of each triangle (0-based) |
| `element_x1`, `element_x3` | element | centroid of each triangle [mm] |
| `element_area_mm2` | element | area [mm^2] |
| `element_edge_mm` | element | distance from the centroid to the outline of the face [mm] |
| `element_root_mm` | element | distance to the nearer notch root [mm]; the two roots are in the attribute `notch_roots_x1_x3_mm` |

**Per image**

| variable | what it is |
| --- | --- |
| `load_kN`, `load_over_max` | machine load, and the same over the peak load |
| `time_s`, `stroke_mm` | time and machine stroke |
| `grey_correction` | brightness correction fitted per camera (`a`: gain - 1, `b`: offset); the light dips by about 5 percent over images 127 to 135 of ARC-00_m45_s2, and this correction takes the dip out |
| `iterations` | iterations of the DIC solver |

**Masks** (nothing is removed from the file; masks say what to leave out)

| variable | dimensions | true when |
| --- | --- | --- |
| `mask_reference` | image | the reference image (all fields are zero there) |
| `mask_static` | image | the machine was standing still before loading: use these images to measure the noise |
| `mask_not_converged` | image | the DIC solver did not converge |
| `mask_after_peak` | image | after the peak load |
| `mask_blurred` | image | a camera moved during the exposure and the frame is blurred: leave it out (`onsetml.data.loading_images` and `static_noise` do) |
| `mask_no_pixel` | element | no pixel of the images falls in the element |

The attributes of the file say where it comes from, the peak image and load, the static
images, and count every NaN (`n_nan_*`).

## Things to know

- One element = one "point". The elements are smaller near the notches (about 0.25 mm) than
  far from them (about 0.6 mm): weigh by `element_area_mm2` when you average over an area.
- The strain at a notch root is less certain than elsewhere: a thin painted band along the
  notches has almost no speckle, so the DIC fills it in smoothly there.
- The noise of the strain at rest, the standard deviation of an element over the static
  images, is about 76 microstrain on e11 and e33 and 107 on g13 (medians over the elements of
  ARC-00_m45_s2; `onsetml.data.static_noise`). Much of it differs from element to element, so an
  average over a small area reduces it quickly.
- The whole coupon moves and turns in the grips as it is loaded (at the peak of ARC-00_m45_s2
  about 0.5 mm and 0.9 mm in the face and 1.8 degrees). That motion is taken out of
  `displacement`, image by image: in the face, the translation and the rotation of a least
  squares fit of the displacement on a rigid motion plus a uniform strain; out of it, the best
  plane of `w`. With a_1, a_3 the node coordinates about their mean, the measured displacement
  is `displacement` plus (t1 - rotation a_3, t3 + rotation a_1, w + tilt1 a_1 + tilt3 a_3).
  The strain is not affected.
- The residual is a second view of the same event, from the images rather than from the
  motion. `residual_rms` is the plain rms of the residual over the element: an element spans
  hundreds of pixels, so a crack one or two pixels wide barely shows in it.
  `residual_over_floor` and `residual_frac3` keep more of it. Three things raise them besides
  damage: the outline (leave out the elements within 0.5 mm of it), a light change (images 127
  to 135 of ARC-00_m45_s2) and steep strain gradients the mesh cannot follow. The thin painted
  band along the notches has no speckle, so the residual sees little there. The floors used
  are in the attributes (`residual_moving_floor_grey`, one per camera). The fixed part is
  measured on the images at which the machine stood still (`mask_static`, but the reference
  and blurred ones; listed in `residual_still_images`): 38 to 40 on most tests, 9 on
  ARC-00_m45_s0 and only 2 on ARC-00_s0 (images 5 and 10). ARC-00_s0 has the highest floor
  (2.3 to 2.4 grey levels against 1.5 to 2.0), so the same crack stands out less there, and its
  static images read low (about 0.46): they are the two the fixed part was taken from.
- Material properties are not used anywhere in this data.
- Maps: `onsetml.plotting.element_map(ds, values)` draws each element flat, as stored;
  with `nodal=True` it averages at the nodes (area-weighted mean of the triangles around each
  node), the smooth picture of FE and DIC software. The averaging is a filter: it lowers the
  noise at rest of e33 from about 74 to 33 microstrain and spreads the notch roots over a few
  elements, so compute on the element values, not on the averaged ones.
