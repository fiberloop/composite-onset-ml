Exploration notebooks.

- `00_explore_the_data.ipynb`: the starting point. It opens one test, tours what the file holds
  (load, mesh, strain fields, noise at rest, element histories, DIC residual, displacement),
  turns the strain into plain arrays, draws a naive onset map to beat, checks the noise model
  against its law under H0 and runs a first CUSUM (`onsetml.null`). It runs
  in about 15 s.

Keep each of your own explorations short and dated in its name (`2026-11-10_onset_cusum.ipynb`);
move code that a result depends on into `src/onsetml/`, with a test.
