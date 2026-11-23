# Main results

Results from the shared fixed-1,000-image corpus. All TPRs below use the
5% clean-calibration target; actual test false-alarm rates can differ.

## Stage 1: PGD-only training

| Test attack | AUC | Detection rate | Clean false alarms | Mean noise false alarms |
| --- | ---: | ---: | ---: | ---: |
| pgd | 1.0000 | 100.00% | 3.43% | 79.25% |
| apgd | 1.0000 | 100.00% | 3.43% | 79.25% |
| cw | 0.8797 | 55.70% | 3.43% | 79.25% |
| patch | 0.9997 | 99.68% | 3.43% | 79.25% |
| fgsm | 0.9997 | 99.52% | 3.43% | 79.25% |

## Stage 2: held-out attack detection

Means over three seeds; compare arms within the same held-out attack.

| Held out | Training arm | N clean + N attacked | AUC | Detection rate | Clean false alarms | Mean noise false alarms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| apgd | mixture | 218 + 218 | 1.0000 | 100.00% | 5.33% | 88.00% |
| apgd | pgd_only | 218 + 218 | 1.0000 | 100.00% | 2.57% | 76.50% |
| cw | mixture | 300 + 300 | 0.8393 | 42.48% | 4.38% | 56.42% |
| cw | pgd_only | 300 + 300 | 0.8776 | 55.27% | 3.43% | 79.25% |
| patch | mixture | 226 + 226 | 0.9953 | 97.64% | 5.62% | 86.08% |
| patch | pgd_only | 226 + 226 | 0.9998 | 99.68% | 2.86% | 77.50% |

## Interpretation

- apgd: mixture minus matched PGD-only changes AUC by +0.0000, detection by +0.00 percentage points, and clean false alarms by +2.76 percentage points.
- cw: mixture minus matched PGD-only changes AUC by -0.0384, detection by -12.80 percentage points, and clean false alarms by +0.95 percentage points.
- patch: mixture minus matched PGD-only changes AUC by -0.0045, detection by -2.04 percentage points, and clean false alarms by +2.76 percentage points.

Stage 2 PGD-only is retrained on the matched pool; it is not a copy of Stage 1.
These means alone do not establish statistical significance. Review the primary-seed
paired-bootstrap intervals separately. The recorded cross-machine prediction differences
remain an audit limitation; original cohort membership is preserved.

See [setup](SETUP.md) and [combined results](combined_results.csv).
