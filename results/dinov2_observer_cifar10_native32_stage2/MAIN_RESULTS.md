# Main results

Results from the shared fixed-1,000-image corpus. All TPRs below use the
5% clean-calibration target; actual test false-alarm rates can differ.

## Stage 1: PGD-only training

| Test attack | AUC | Detection rate | Clean false alarms | Mean noise false alarms |
| --- | ---: | ---: | ---: | ---: |

## Stage 2: held-out attack detection

Means over three seeds; compare arms within the same held-out attack.

| Held out | Training arm | N clean + N attacked | AUC | Detection rate | Clean false alarms | Mean noise false alarms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| apgd | mixture | 221 + 221 | 0.9712 | 85.37% | 5.81% | 59.17% |
| apgd | pgd_only | 221 + 221 | 0.9919 | 96.24% | 6.57% | 75.50% |
| cw | mixture | 289 + 289 | 0.6280 | 15.86% | 6.76% | 67.00% |
| cw | pgd_only | 289 + 289 | 0.6480 | 17.74% | 5.71% | 75.75% |
| patch | mixture | 235 + 235 | 0.4924 | 3.70% | 5.62% | 71.17% |
| patch | pgd_only | 235 + 235 | 0.5058 | 5.39% | 6.29% | 76.50% |

## Interpretation

- apgd: mixture minus matched PGD-only changes AUC by -0.0207, detection by -10.87 percentage points, and clean false alarms by -0.76 percentage points.
- cw: mixture minus matched PGD-only changes AUC by -0.0200, detection by -1.88 percentage points, and clean false alarms by +1.05 percentage points.
- patch: mixture minus matched PGD-only changes AUC by -0.0133, detection by -1.68 percentage points, and clean false alarms by -0.67 percentage points.

Stage 2 PGD-only is retrained on the matched pool; it is not a copy of Stage 1.
These means alone do not establish statistical significance. Review the primary-seed
paired-bootstrap intervals separately. The recorded cross-machine prediction differences
remain an audit limitation; original cohort membership is preserved.

See [setup](SETUP.md) and [combined results](combined_results.csv).
