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
| apgd | mixture | 195 + 195 | 0.8557 | 42.59% | 4.48% | 20.00% |
| apgd | pgd_only | 195 + 195 | 0.9055 | 52.68% | 4.57% | 23.50% |
| cw | mixture | 195 + 195 | 0.5581 | 4.85% | 4.48% | 21.50% |
| cw | pgd_only | 195 + 195 | 0.5667 | 5.06% | 4.57% | 23.50% |
| patch | mixture | 320 + 320 | 0.5450 | 8.21% | 6.57% | 30.42% |
| patch | pgd_only | 320 + 320 | 0.5534 | 7.88% | 6.57% | 31.00% |

## Interpretation

- apgd: mixture minus matched PGD-only changes AUC by -0.0497, detection by -10.09 percentage points, and clean false alarms by -0.10 percentage points.
- cw: mixture minus matched PGD-only changes AUC by -0.0085, detection by -0.21 percentage points, and clean false alarms by -0.10 percentage points.
- patch: mixture minus matched PGD-only changes AUC by -0.0083, detection by +0.33 percentage points, and clean false alarms by +0.00 percentage points.

Stage 2 PGD-only is retrained on the matched pool; it is not a copy of Stage 1.
These means alone do not establish statistical significance. Review the primary-seed
paired-bootstrap intervals separately. The recorded cross-machine prediction differences
remain an audit limitation; original cohort membership is preserved.

See [setup](SETUP.md) and [combined results](combined_results.csv).
