# Main results

Results from the shared fixed-1,000-image corpus. All TPRs below use the
5% clean-calibration target; actual test false-alarm rates can differ.

## Stage 1: PGD-only training

| Test attack | AUC | Detection rate | Clean false alarms | Mean noise false alarms |
| --- | ---: | ---: | ---: | ---: |
| pgd | 0.9987 | 100.00% | 6.57% | 9.75% |
| apgd | 0.9991 | 100.00% | 6.57% | 9.75% |
| cw | 0.7264 | 24.37% | 6.57% | 9.75% |
| patch | 0.9447 | 72.91% | 6.57% | 9.75% |
| fgsm | 0.8919 | 54.21% | 6.57% | 9.75% |

## Stage 2: held-out attack detection

Means over three seeds; compare arms within the same held-out attack.

| Held out | Training arm | N clean + N attacked | AUC | Detection rate | Clean false alarms | Mean noise false alarms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| apgd | mixture | 195 + 195 | 0.9956 | 98.84% | 5.52% | 10.08% |
| apgd | pgd_only | 195 + 195 | 0.9991 | 99.68% | 9.43% | 15.00% |
| cw | mixture | 195 + 195 | 0.6724 | 32.38% | 9.52% | 11.42% |
| cw | pgd_only | 195 + 195 | 0.7270 | 33.23% | 9.43% | 15.00% |
| patch | mixture | 320 + 320 | 0.9011 | 64.86% | 6.29% | 13.42% |
| patch | pgd_only | 320 + 320 | 0.9407 | 72.41% | 6.57% | 11.25% |

## Interpretation

- apgd: mixture minus matched PGD-only changes AUC by -0.0034, detection by -0.84 percentage points, and clean false alarms by -3.90 percentage points.
- cw: mixture minus matched PGD-only changes AUC by -0.0547, detection by -0.84 percentage points, and clean false alarms by +0.10 percentage points.
- patch: mixture minus matched PGD-only changes AUC by -0.0396, detection by -7.55 percentage points, and clean false alarms by -0.29 percentage points.

Stage 2 PGD-only is retrained on the matched pool; it is not a copy of Stage 1.
These means alone do not establish statistical significance. Review the primary-seed
paired-bootstrap intervals separately. The recorded cross-machine prediction differences
remain an audit limitation; original cohort membership is preserved.

See [setup](SETUP.md) and [combined results](combined_results.csv).
