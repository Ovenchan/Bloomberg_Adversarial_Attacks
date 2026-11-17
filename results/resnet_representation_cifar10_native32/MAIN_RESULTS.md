# Main results

Results from the shared fixed-1,000-image corpus. All TPRs below use the
5% clean-calibration target; actual test false-alarm rates can differ.

## Stage 1: PGD-only training

| Test attack | AUC | Detection rate | Clean false alarms | Mean noise false alarms |
| --- | ---: | ---: | ---: | ---: |
| pgd | 0.9997 | 99.69% | 2.57% | 29.25% |
| apgd | 0.9998 | 99.69% | 2.57% | 29.25% |
| cw | 0.6451 | 11.29% | 2.57% | 29.25% |
| patch | 0.8723 | 50.51% | 2.57% | 29.25% |
| fgsm | 0.8645 | 38.24% | 2.57% | 29.25% |

## Stage 2: held-out attack detection

Means over three seeds; compare arms within the same held-out attack.

| Held out | Training arm | N clean + N attacked | AUC | Detection rate | Clean false alarms | Mean noise false alarms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| apgd | mixture | 221 + 221 | 0.9855 | 93.42% | 4.00% | 18.00% |
| apgd | pgd_only | 221 + 221 | 0.9991 | 99.37% | 2.57% | 21.75% |
| cw | mixture | 289 + 289 | 0.6242 | 10.08% | 2.86% | 8.17% |
| cw | pgd_only | 289 + 289 | 0.6441 | 12.50% | 2.86% | 27.25% |
| patch | mixture | 235 + 235 | 0.8776 | 56.00% | 5.05% | 22.58% |
| patch | pgd_only | 235 + 235 | 0.8977 | 55.56% | 2.00% | 18.75% |

## Interpretation

- apgd: mixture minus matched PGD-only changes AUC by -0.0136, detection by -5.96 percentage points, and clean false alarms by +1.43 percentage points.
- cw: mixture minus matched PGD-only changes AUC by -0.0200, detection by -2.42 percentage points, and clean false alarms by +0.00 percentage points.
- patch: mixture minus matched PGD-only changes AUC by -0.0200, detection by +0.45 percentage points, and clean false alarms by +3.05 percentage points.

Stage 2 PGD-only is retrained on the matched pool; it is not a copy of Stage 1.
These means alone do not establish statistical significance. Review the primary-seed
paired-bootstrap intervals separately. The recorded cross-machine prediction differences
remain an audit limitation; original cohort membership is preserved.

See [setup](SETUP.md) and [combined results](combined_results.csv).
