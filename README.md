# Satellite-Based Early Warning: A Constellation Simulation Study

**Boost-phase IRBM detection over Southeast Asia using a 12-satellite LEO constellation**

**Darrel Jeremiah Rondonuwu · Personal research project · Python**

## Overview

I undertook this project to investigate the potential of a small satellite constellation to detect intermediate-range ballistic missiles (IRBMs) during their boost phase over Southeast Asia. My interest in aerospace encouraged me to explore how orbital motion, observation constraints, and uncertainty can be brought together in a computational study.

The project examines a constellation of 12 satellites in low Earth orbit (LEO), distributed across three orbital planes. It uses Monte Carlo simulations to estimate detection probability and the time to first detection under the assumptions of the model. The repository brings together the Python source files, saved plots, and numerical results.

This is an exploratory simulation study. Its findings describe the behavior of the model and have not been validated as real-world system performance.

## Research question

**Under the modeled conditions, how consistently and how quickly can a 12-satellite constellation detect a simulated launch within the study region?**

The study considers both single-launch and two-simultaneous-launch scenarios. This comparison helps examine how the definition of a successful detection affects the interpretation of the results.

## Approach

The analysis combines numerical orbit propagation, a simplified boost-phase trajectory model, geometric visibility checks, and modeled sensor constraints. Repeated trials sample different launch times relative to the constellation's motion. The resulting estimates are evaluated across a geographic grid and summarized through plots and tables.

The source files also include a comparison of alternative orbital-plane arrangements. These comparisons are part of the exploratory study; they do not establish an operationally optimal constellation.

## Saved results

The following values are taken from [`table_results_summary.csv`](table_results_summary.csv).

| Scenario | Mean modeled detection probability | Reported detection-time summary |
| :--- | ---: | ---: |
| Single launch | 77.58% | 26 s |
| Two simultaneous launches | 87.66% | 26 s |

### How to interpret these values

- **Detection probability:** At each sampled location, this is the fraction of trials that produce a detection. The reported mean averages these estimates across the valid grid points. It is not a geographically weighted estimate of real-world launch risk.
- **Two-launch success:** A trial succeeds when at least one of the simulated launches is detected. The 87.66% figure does not mean that both launches are detected or continuously tracked.
- **Detection time:** The model first calculates a median using successful detections at each location. The table then reports the median of those location-level medians, excluding locations without a finite value. The 26-second result is therefore neither a pooled median across every trial nor a measure of end-to-end warning delivery.

The saved outputs are included for inspection. They have not been independently reproduced as part of preparing this documentation.

## Example visualization

![Saved single-launch conditional median detection-time heatmap](heatmap_T_single.png)

*Spatial variation in the saved conditional detection-time estimates. The timing values summarize successful detections under the model's assumptions.*

## Exploring the repository

No software installation is needed to read this overview, inspect the PNG figures, or view the CSV tables on GitHub. Start with the results summary, then use the figures and accompanying tables to understand what is being reported.

| File or group | Contents |
| :--- | :--- |
| [`irbm_full_sim.py`](irbm_full_sim.py) | Main simulation and result-export source |
| [`best_raan_search.py`](best_raan_search.py) | Orbital-plane comparison source |
| [`table_results_summary.csv`](table_results_summary.csv) | Aggregate saved results |
| [`heatmap_data.csv`](heatmap_data.csv) | Saved grid-level results |
| `heatmap_*.png` | Saved probability, timing, and threshold-based plots |
| `table_*_parameters.csv` | Recorded model assumptions |
| Other CSV and PNG files | Supplementary comparisons and summaries |

### Software used

The scripts use **Python**, with **NumPy** for numerical operations, **SciPy** for numerical integration, **pandas** for tables, and **Matplotlib** for figures.

The archive does not include a pinned dependency environment or an automated test suite. A verified reproduction procedure remains to be documented. The archived grid-data file is named `heatmap_data.csv`, while the supplied main script exports that dataset under `stage3_irbm_heatmap_data.csv`; the filenames should not be assumed to match automatically.

## Limitations and research status

The project relies on simplified trajectory and sensor models. Numerical precision in an exported table does not establish physical accuracy. Monte Carlo estimates also contain sampling uncertainty, and the summary table does not provide confidence intervals.

The source files and saved outputs are preserved together, but their presence alone does not prove that every output was generated by the exact supplied source version. Independent reproduction and validation remain necessary before drawing stronger conclusions.

For me, the value of this study lies in connecting an aerospace question with a model whose assumptions and results can be examined. A useful result must be accompanied by a clear explanation of what it measures and where its interpretation ends.

## Author and acknowledgments

This is a personal research project by **Darrel Jeremiah Rondonuwu**. AI assistance was used during project development and documentation. This repository does not claim institutional endorsement, peer review, or operational deployment.

Questions about the study or corrections to its documentation can be raised through the repository's Issues tab when enabled.
