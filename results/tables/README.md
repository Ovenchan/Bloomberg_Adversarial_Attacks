# Grad-CAM Results

These results correspond to the Grad-CAM experiments used in
the research report.

The primary Grad-CAM detector uses predicted-class Grad-CAM heatmaps from
ResNet-50, downsampled to 14x14 and flattened to 196-dimensional vectors.
A logistic-regression detector was trained on clean and PGD examples and
evaluated without retraining across multiple attack families.

The results show strong performance on PGD and APGD but weak cross-attack
generalization to FGSM and C&W.

The results are from 80/20 holdout experiment, not the new five-fold CIFAR-10 pipeline.
