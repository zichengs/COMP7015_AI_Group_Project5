// COMP7015 group project report. Compile from the repository root:
//   typst compile --root . report/report.typ report/report.pdf
// Every number comes from the tables in outputs/ (collected in 06_discussion.ipynb).

#set document(title: "Tomato Leaf Disease Classification: Accurate on Clean Photos, Fragile Outside Them")
#set page(paper: "a4", margin: (x: 1.7cm, top: 1.5cm, bottom: 1.6cm), numbering: "1")
#set text(font: ("Libertinus Serif", "New Computer Modern"), size: 10pt, lang: "en")
#set par(justify: true, leading: 0.55em, spacing: 0.75em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(above: 1.0em, below: 0.55em, text(size: 11.5pt, it))
#show heading.where(level: 2): it => block(above: 0.85em, below: 0.5em, text(size: 10pt, it))
#set list(indent: 0.4em, spacing: 0.55em)
#set table(stroke: none, inset: (x: 4pt, y: 2.6pt))
#show table: set text(size: 8.3pt)
#show figure.caption: set text(size: 8.5pt)
#set figure(gap: 0.5em)
#show figure: set block(above: 0.9em, below: 0.9em)

#let rule = table.hline(stroke: 0.5pt)
#let todo(body) = text(fill: red, body)

#align(center)[
  #text(size: 15pt, weight: "bold")[Tomato Leaf Disease Classification: \ Accurate on Clean Photos, Fragile Outside Them]
  #v(0.1em)
  COMP7015 Artificial Intelligence · Group Project, Topic 1 (Plant Disease Classification) · Group 5
]
#v(0.2em)

#block(inset: (x: 1.2em))[
  *Summary.* We classify four classes of PlantVillage tomato leaves with a small CNN with channel attention trained from scratch (SE-CNN), a fine-tuned ResNet18 and a fine-tuned ViT-B/16. On clean test photos all three are almost perfect (macro-F1 0.980, 0.985 and 0.992). We then show that this score says little about the models. The background alone predicts the class with 82% accuracy, the two CNNs follow it, and every model falls to a macro-F1 of 0.19–0.37 on field photos. Training with random backgrounds removes the background shortcut at little or no cost on clean photos, but it does not make the models work in the field.
]

= Motivation

PlantVillage photos [1] show one leaf on a plain surface, and published accuracies on them are above 99% [2]. Each class was photographed under its own conditions, however, so the background, the light and the camera differ between the classes, and a model can score well without looking at the disease [3]. We therefore ask two questions. How do a CNN trained from scratch, a pre-trained CNN and a pre-trained vision transformer compare on clean photos, and at what compute cost? And how much of their score survives when the background, the colour, the light or the image quality changes, or when the photo comes from a field? This covers the first two advanced directions of the project brief (attention and a vision transformer; robustness), and uses Grad-CAM from the third.

= Data and protocol

*Classes and leaf groups.* We use the tomato classes healthy, bacterial spot, early blight and late blight (6,627 colour photos). Most leaves were photographed about four times, so photos of one leaf must not be spread over the splits. We keep only the 5,045 photos that have a leaf id in the repository's `leaf-map.json` (1,262 leaves). The other 1,582 photos are excluded: for 1,038 the leaf number could only be guessed from the file name, and 544 have no leaf information at all (541 of them late blight).

*Split.* Within each class the leaf ids are shuffled with seed 62 and split 70 / 15 / 15, which gives 3,529 / 760 / 756 photos (@counts). An assertion checks that no leaf id occurs in two splits. The split seed is separate from the training seeds, so every model sees the same split.

#figure(
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    rule,
    [Class], [Train], [Validation], [Test], [All], [Leaves],
    rule,
    [Healthy], [699], [152], [148], [999], [250],
    [Bacterial spot], [1,487], [320], [320], [2,127], [532],
    [Early blight], [699], [152], [148], [999], [250],
    [Late blight], [644], [136], [140], [920], [230],
    rule,
    [All], [3,529], [760], [756], [5,045], [1,262],
    rule,
  ),
  caption: [Photos per class and split (split seed 62). All photos of one leaf are in the same split.],
) <counts>

*Preprocessing.* Each photo is decoded and resized to 224×224 once and cached as a `uint8` tensor. All models get the same input, scaled to [0, 1] and normalised with the ImageNet mean and standard deviation. Augmentation is applied to training photos only and is chosen so that it does not change the symptoms: horizontal and vertical flips, rotations by multiples of 90° (lossless, no black corners) and a ±10% brightness and contrast jitter. Hue and saturation are not changed, because lesion colour is part of the diagnosis.

*Class imbalance.* Bacterial spot has about 2.1 times as many photos as each other class. The loss is a class-weighted cross-entropy with $w_c = N \/ (K n_c)$, which gives 0.59 for bacterial spot and 1.26–1.37 for the other classes. We report macro-F1 as the main metric, because it weights the four classes equally.

*Evaluation rules.* Tuning and model selection use the validation set only. The test set is evaluated once, after selection. Final results are the mean ± sample standard deviation over three training seeds (62, 63, 64), and two models are compared seed by seed ("higher in $k$ of 3 seeds"). With 756 test photos, one photo is 0.13 percentage points of accuracy.

= Models and training

- *SE-CNN (from scratch, 0.41 M parameters).* Four blocks of 3×3 convolution, batch normalisation, ReLU and max pooling (32, 64, 128, 256 channels), then global average pooling, dropout 0.3 and a linear layer. After every convolution a Squeeze-and-Excitation block [5] averages each channel to one number, passes these through a small bottleneck and re-weights the channels. This is a cheap form of attention.
- *ResNet18 (transfer learning, 11.2 M).* ImageNet weights [6] with a new 4-class head. Training has two stages: one epoch for the head only (lr 1e-3), so that the random head does not disturb the pre-trained features, then the whole network with lr 1e-4 and a cosine schedule.
- *ViT-B/16 (transfer learning, 85.8 M).* ImageNet weights [7] with a new head. Three head-only epochs (lr 1e-3) are followed by fine-tuning with lr 1e-5 for the backbone and 1e-4 for the head (batch size 16, weight decay 0.01). We compare three strategies for the second stage: head only, the last two encoder blocks, and the whole network.

All models use AdamW, gradient clipping, mixed precision, early stopping on the validation macro-F1 (patience 10 epochs for the SE-CNN, 5 for the others) and the checkpoint of the best validation epoch. Runs are deterministic for a given seed. Each experiment is a named recipe that changes one setting of a base recipe, so every ablation differs from its reference in exactly one way.

= Results on clean photos

== Tuning and ablations (validation set)

*SE-CNN.* With seed 62, every recipe reaches a validation macro-F1 between 0.987 and 0.992. The largest changes from the full recipe (0.992) are a constant learning rate (0.987) and batch size 64 (0.989). Removing the SE blocks, the augmentation, the dropout or the class weights changes the score by at most 0.002. The full recipe itself varies by ±0.003 between seeds, so these single-seed ablations cannot be told apart from seed noise. We say so plainly: on this validation set, we cannot show that attention, dropout or class weighting helps. We re-ran the three best recipes with all seeds and selected lr = 3e-3, which has the highest mean and the smallest spread (0.991 ± 0.001). Without class weights the mean is lowest and the spread largest (0.988 ± 0.005).

*ResNet18.* Every recipe that fine-tunes the backbone reaches 0.997–1.000. Freezing the backbone is the one clear effect (0.956). The two-stage recipe with a cosine schedule and lr 1e-4 reaches 1.000 in all three seeds and was selected.

*ViT strategies.* @vit shows that unfreezing more of the network helps in every seed, on validation and on test. Full fine-tuning was selected on validation. It costs 2.7 times the training time of the head-only strategy. Augmentation helps the partial ViT by 0.007 ± 0.005 on test (3 of 3 seeds).

#figure(
  table(
    columns: 6,
    align: (left, right, center, center, center, right),
    rule,
    [Strategy], [Trainable parameters], [Validation macro-F1], [Test macro-F1], [Wrong in every seed], [Training time (s)],
    rule,
    [Head only], [3 k], [0.974 ± 0.002], [0.977 ± 0.006], [11], [220 ± 87],
    [Last two blocks], [14.2 M], [0.987 ± 0.001], [0.984 ± 0.002], [7], [373 ± 53],
    [Last two blocks, no augmentation], [14.2 M], [0.985 ± 0.001], [0.976 ± 0.003], [12], [339 ± 35],
    [Full fine-tuning (selected)], [85.8 M], [0.996 ± 0.000], [0.992 ± 0.001], [3], [595 ± 116],
    rule,
  ),
  caption: [ViT-B/16 fine-tuning strategies, mean ± sd over three seeds. "Wrong in every seed" counts test photos that all three seeds misclassify.],
) <vit>

== Test results, errors and compute

#figure(
  table(
    columns: 9,
    align: (left, center, center, center, center, center, right, right, right),
    rule,
    [Model], [Accuracy], [Macro-F1], [F1 early blight], [F1 late blight], [Errors per seed], [Size (MB)], [Latency (ms)], [Training (s)],
    rule,
    [SE-CNN], [0.984 ± 0.004], [0.980 ± 0.005], [0.960 ± 0.011], [0.960 ± 0.009], [9, 12, 15], [1.7], [2.0], [264],
    [ResNet18], [0.989 ± 0.001], [0.985 ± 0.001], [0.972 ± 0.002], [0.969 ± 0.002], [9, 9, 8], [44.8], [2.6], [94],
    [ViT-B/16], [0.993 ± 0.001], [0.992 ± 0.001], [0.983 ± 0.003], [0.985 ± 0.002], [5, 4, 6], [343], [6.3], [595],
    rule,
  ),
  caption: [The selected models on the 756 test photos (mean ± sd over three seeds). F1 for healthy and bacterial spot is at least 0.998 for every model. Latency is for one image; training time is the mean over seeds. All times were measured on the same GPU with mixed precision.],
) <test>

*Ranking.* The order is ViT, ResNet18, SE-CNN (@test). The ViT is ahead of ResNet18 in 3 of 3 seeds, and ResNet18 is ahead of the SE-CNN in 2 of 3 (they tie on seed 62). The differences are small: on average 5.0, 8.7 and 12.0 wrong photos per seed.

*Errors.* Healthy and bacterial spot are recognised almost perfectly, so the larger class does not dominate the errors. Nearly all errors are between early blight and late blight (@cm): 33 of the 36 errors of the SE-CNN over all seeds, 25 of 26 for ResNet18 and 13 of 15 for the ViT. The CNNs mostly call late blight early blight, and the ViT more often does the opposite. Only 6, 4 and 3 test photos are wrong in every seed, against 19, 13 and 8 that are wrong at least once. Most errors are therefore borderline blight photos that fall on one side or the other depending on the seed. Wrong predictions are less confident (mean 0.77, 0.87 and 0.88) than correct ones (at least 0.996).

#figure(
  image("/outputs/confusion_matrices.png", width: 86%),
  caption: [Confusion matrices on the test set (seed 62). The errors are between early blight and late blight.],
) <cm>

*Compute.* The SE-CNN is 27 times smaller than ResNet18, but it needs all 50 epochs and so trains longer (264 s against 94 s). ResNet18 is the cheapest model to train and, in batches, as fast as the SE-CNN (0.37 against 0.35 ms per image). Compared with ResNet18, the ViT gains 0.007 macro-F1 for 7.7 times the parameters, 6.3 times the training time and 2.4 times the single-image latency. On clean photos alone, this gain would be hard to justify. The next section changes that picture.

= Robustness

== The same test leaves under changed conditions

Before testing the networks, we trained a random forest on only the mean colour of the four image corners (12 numbers, no leaf). It reaches 82.1% validation accuracy, against 42% for always answering the largest class. The background therefore reveals the class.

@rob shows the three models on the same 756 test leaves under five changes. Removing the background uses PlantVillage's own segmented photos.

#figure(
  image("/outputs/robustness.png", width: 100%),
  caption: [Test macro-F1 on the same test leaves under five changes (bars: mean ± sd; dots: seeds).],
) <rob>

- The SE-CNN fails under every change (0.09–0.69). Without the background it scores 0.090, far below guessing, and even a moderate darkening outside its ±10% jitter costs a third of its score (0.638).
- The pre-trained models are robust to darkening (0.978 and 0.985), and the ViT also to noise (0.969). Blur hurts every model. Removing the background separates the models most: 0.090, 0.423 and 0.848.
- Seeds that are equally good on clean photos differ widely here: the sd of ResNet18 grows from 0.001 to 0.11–0.14 under blur, noise and background removal. The clean score does not predict robustness.

*Is information lost, or is the input only unfamiliar?* We retrained the ViT on grayscale and on background-removed photos. Trained and tested on grayscale it reaches 0.975 ± 0.007, so colour is worth about 0.02, and the drop to 0.592 in @rob is mostly a mismatch between training and test input. Trained and tested without background it reaches 0.978 ± 0.003. The leaf alone carries almost all the information; the background is a shortcut, not a necessity.

== Leaf or background?

To separate the leaf from its background we paste each test leaf, with identical leaf pixels, onto a background tile taken from another test photo: once from the same class and once from another class. The same-class version controls for the unfamiliar look of a pasted leaf.

Going from a same-class to an other-class background lowers every model in every seed: the SE-CNN from 0.836 to 0.481, ResNet18 from 0.930 to 0.773 and the ViT from 0.913 to 0.894. The SE-CNN then predicts the class of the background photo for 25% of the leaves, ResNet18 for 12% and the ViT for 2%. On a background from a healthy photo, ResNet18 calls 39% of the diseased leaves healthy, which is the most harmful kind of error. The ViT barely follows the background; the two CNNs clearly do.

Grad-CAM [8] gives a weaker and partly conflicting picture (@cam). The leaf covers 45% of the image. The SE-CNN puts 53% of its Grad-CAM weight on the leaf and ResNet18 60%, but the ViT only 41%, with a large spread between seeds (±0.12). Grad-CAM on the 14×14 token grid of the last ViT block is coarse, so for the ViT we trust the behavioural tests above more than the heat maps.

#figure(
  image("/outputs/grad_cam.png", width: 60%),
  caption: [Grad-CAM maps of six test photos (seed 62; rows: photo, SE-CNN, ResNet18, ViT; red titles are wrong predictions). The SE-CNN often attends to the background (columns 3 and 5), while ResNet18 stays on the leaf.],
) <cam>

== Countermeasure: training with random backgrounds

We retrained the three selected models with one extra augmentation. With probability 0.5, everything outside the leaf mask is replaced by black, a solid colour, a gradient, smooth noise or a background tile from another training photo. The background then no longer tells the class. As a control, a second version always takes the tile from a photo of the same class, so the model sees pasted leaves but the background still matches the label. All other settings, the seeds and the selection rule stay the same (9 models, 3 seeds each).

#figure(
  table(
    columns: 9,
    align: (left, left, center, center, center, center, center, center, center),
    rule,
    [Model], [Training], [Original], [Stripes (unseen)], [Same-class bg.], [Other-class bg.], [Bg. removed], [Follows bg.], [Field photos],
    rule,
    [SE-CNN], [baseline], [0.980], [0.218], [0.836], [0.481], [0.090], [25.0%], [0.229],
    [], [random backgrounds], [0.974], [*0.919*], [0.930], [*0.924*], [*0.708*], [1.7%], [0.241],
    [], [class-matched (control)], [0.966], [0.238], [0.948], [0.684], [0.225†], [9.4%], [0.268],
    rule,
    [ResNet18], [baseline], [0.985], [0.479], [0.930], [0.773], [0.423†], [12.4%], [0.192],
    [], [random backgrounds], [0.986], [*0.973*], [0.980], [*0.978*], [*0.933*], [0.6%], [0.285],
    [], [class-matched (control)], [0.983], [0.737†], [0.977], [0.944], [0.699†], [2.1%], [0.264],
    rule,
    [ViT-B/16], [baseline], [0.992], [0.937], [0.913], [0.894], [0.848], [2.1%], [0.332],
    [], [random backgrounds], [0.981], [*0.961*], [0.971], [*0.969*], [*0.946*], [0.7%], [*0.370*],
    [], [class-matched (control)], [0.983], [0.941], [0.968], [0.948], [0.856], [1.3%], [0.320],
    rule,
  ),
  caption: [Macro-F1 (mean over three seeds) on the test leaves with changed backgrounds and on 372 PlantDoc field photos. "Follows bg." is the share of leaves on an other-class background that get the class of the background photo. The sd is at most 0.045, except in the cells marked † (0.10–0.14). Black is one of the random training backgrounds, so "Bg. removed" is not unseen for those models.],
) <cmtable>

- *Random backgrounds remove the shortcut in all three models* (@cmtable). On striped backgrounds that never occur in training, the SE-CNN rises from 0.218 to 0.919 and ResNet18 from 0.479 to 0.973. Predictions that follow the background fall below 2% for every model, and the Grad-CAM share on the leaf rises (0.53 to 0.75, 0.60 to 0.70 and 0.41 to 0.60).
- *The cost on clean photos is small.* ResNet18 loses nothing (+0.001, higher in 3 of 3 seeds). The SE-CNN loses 0.006 and the ViT 0.010, both in 3 of 3 seeds.
- *The control shows why it works.* Class-matched backgrounds give the same gain on same-class backgrounds (+0.05 to +0.11), which is the part that comes from getting used to pasted leaves. They gain far less on stripes (0.238 for the SE-CNN) and on other-class backgrounds. The rest of the gain therefore comes from breaking the link between background and label.

== Field photos and confidence

*Field photos.* We tested all nine models, without retraining, on the 372 PlantDoc photos [4] of the same four classes: web photos with several leaves, hands, soil and natural light. Every model fails (last column of @cmtable). The selected ResNet18 (0.192) and SE-CNN (0.229) are no better than guessing. Most field photos are called early blight: its recall is 0.69–0.95, while healthy leaves are recognised in at most 17% of the cases. The ViT with random backgrounds is the best model (0.370 ± 0.040), which is still far from usable. Removing the background shortcut is thus necessary but not sufficient: light, camera, leaf pose and the number of leaves differ as well. The PlantDoc labels come from web sources, so these scores are indicative only.

*Can the model's confidence warn us?* On clean photos, yes. If the model abstains on the 5% least confident photos, accuracy rises to 0.998–0.999, because these photos contain 79–92% of the errors (expected calibration error at most 0.009). Under the changes of @rob, no. The models are confidently wrong: the calibration error rises to 0.79 (SE-CNN, background removed) and 0.41 (ResNet18, noise), and abstaining on 5% moves the accuracy by less than 0.01 in these cases. Abstention cannot catch the failures that matter most.

= Discussion

*What we conclude.* (1) On clean PlantVillage photos, the choice of model hardly matters: 0.980 to 0.992, a difference of three to seven photos. Transfer learning mainly buys stability and training time; ResNet18 trains in 94 s. (2) Under changed conditions, the models differ widely. Pre-training on ImageNet gives robustness to light, and the ViT is also robust to noise and to the background without any countermeasure. Its extra compute buys robustness more than clean accuracy. (3) The small CNN trained from scratch relies most on the background, as expected for a model that has only 3,529 photos to learn from and a shortcut that is easier than the disease. (4) Random backgrounds are a cheap and effective fix for this one shortcut. (5) No model is ready for the field.

*Recommendation.* For photos taken under controlled conditions, ResNet18 trained with random backgrounds is the best trade-off: 0.986 on clean photos, 0.97–0.98 on changed backgrounds, 45 MB and 2.6 ms per photo. Where light and cameras vary and compute allows it, the ViT with random backgrounds is the safer choice.

*Limitations.* All results use one fixed split and three seeds, so the sd covers training randomness but not the choice of test leaves. The validation set is saturated, and the SE-CNN and ResNet18 ablations ran on one seed, so the choice between tied recipes is partly arbitrary. Keeping only photos with an official leaf id removes over half of the late-blight folder. The leaf masks come from PlantVillage's segmentation, which also cuts away some leaf regions. Each image change was tested at one strength, and the countermeasure addresses the background only.

*Next steps.* Field photos in training (for example the PlantDoc training split), photometric augmentation (blur, noise, darkening, grayscale) together with random backgrounds, a more faithful explanation method for the ViT such as attention rollout, and evaluation over several splits.

= Contributions

#todo[To be completed by the team before submission: one line per member with their specific work.]

#table(
  columns: (auto, 1fr),
  align: (left, left),
  rule,
  [Member], [Contribution],
  rule,
  [#todo[Name (student ID)]], [#todo[...]],
  [#todo[Name (student ID)]], [#todo[...]],
  [#todo[Name (student ID)]], [#todo[...]],
  rule,
)

= Declaration of AI usage

#todo[Team: check that this matches how each member used AI tools, then remove this note.]

We used Claude Code (Anthropic), an AI coding assistant, during this project. We used it to refactor and restructure the code (moving logic from the notebooks into the `plantdisease` package), to draft helper code such as the smoke-test script, to debug and review code, and to edit the wording of the README and this report. The research questions, the experimental design and evaluation protocol, the choice of models and experiments, the training runs and the interpretation of the results are our own. Every AI-suggested change was read, run and checked by a group member before it was committed, and we can explain all of the code. All numbers in this report come from our own training runs. Code from libraries is credited below and in the repository README.

= Acknowledgements and references

*Libraries.* PyTorch and torchvision (models, ImageNet weights of ResNet18 and ViT-B/16, transforms), pytorch-grad-cam (Grad-CAM and Score-CAM), scikit-learn, SciPy, pandas, NumPy, Matplotlib and pydantic.

#set text(size: 8.3pt)
#set par(justify: false, leading: 0.5em, spacing: 0.45em)
[1] D. Hughes, M. Salathé. An open access repository of images on plant health. arXiv:1511.08060, 2015.
[2] S. Mohanty, D. Hughes, M. Salathé. Using deep learning for image-based plant disease detection. Frontiers in Plant Science, 2016.
[3] M. A. Noyan. Uncovering bias in the PlantVillage dataset. arXiv:2206.04374, 2022.
[4] D. Singh et al. PlantDoc: a dataset for visual plant disease detection. CoDS-COMAD, 2020.
[5] J. Hu, L. Shen, G. Sun. Squeeze-and-excitation networks. CVPR, 2018.
[6] K. He et al. Deep residual learning for image recognition. CVPR, 2016.
[7] A. Dosovitskiy et al. An image is worth 16x16 words. ICLR, 2021.
[8] R. Selvaraju et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. ICCV, 2017.
