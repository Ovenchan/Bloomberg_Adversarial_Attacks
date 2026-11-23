# Main results

Results from the shared fixed-1,000-image corpus. All TPRs below use the
5% clean-calibration target; actual test false-alarm rates can differ.

## Stage 1: CW-only training

| Test attack | AUC | Detection rate | Clean false alarms | Mean noise false alarms |
| --- | ---: | ---: | ---: | ---: |
| pgd | 1.0000 | 100.00% | 5.71% | 92.50% |
| apgd | 1.0000 | 100.00% | 5.71% | 92.50% |
| cw | 0.9644 | 86.50% | 5.71% | 92.50% |
| patch | 0.9997 | 99.68% | 5.71% | 92.50% |
| fgsm | 1.0000 | 100.00% | 5.71% | 92.50% |

## Stage 2: held-out attack detection

Means over three seeds; compare arms within the same held-out attack.

| Held out | Training arm | N clean + N attacked | AUC | Detection rate | Clean false alarms | Mean noise false alarms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |

## Interpretation


Stage 2 PGD-only is retrained on the matched pool; it is not a copy of Stage 1.
These means alone do not establish statistical significance. Review the primary-seed
paired-bootstrap intervals separately. The recorded cross-machine prediction differences
remain an audit limitation; original cohort membership is preserved.

See [setup](SETUP.md) and [combined results](combined_results.csv).
