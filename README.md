# DivingVision

## How to Use

Can be used for training the existing model, creating a new model, etc.

### Training

To train an existing model, run pipeline.py

### Model Creation

Create a 

### Visualization


## Config Files


# Chat's Paper Outline:
Below is a detailed paper outline based on your project. I wrote it so you can later fill in actual numbers once the run finishes. I would frame the project as: semantic label-aware regularization for fine-grained diving classification, not as “we beat SOTA.”

Possible title

Semantic Label-Aware Training for Fine-Grained Diving Action Recognition

Alternative:

Using Structured Dive Similarity to Improve Fine-Grained Action Classification on Diving48

Abstract

Briefly summarize the whole paper in one paragraph.

Suggested content:

This project studies fine-grained action recognition on Diving48, a competitive diving dataset where classes differ by subtle combinations of takeoff direction, number of somersaults, number of twists, and body position. Diving48 is useful because it reduces common background/object shortcuts and forces models to reason about motion and pose rather than scene context. Prior action recognition methods often treat the 48 labels as unrelated categories, even though the class structure contains meaningful semantic relationships. We propose a semantic label-aware extension to a ResNet50 frame-based baseline. The baseline samples 16 frames per video, extracts ImageNet-pretrained ResNet50 features from each frame, mean-pools across time, and trains a linear classifier. Our method defines a 48×48 dive similarity matrix based on takeoff, flips, twists, and position, then uses this matrix to train with soft cross-entropy and/or semantic contrastive learning. We evaluate whether this improves not only top-1/top-5 classification accuracy, but also whether mistakes become more semantically reasonable. Results show [fill in once available]. The method is lightweight and interpretable, but limited by the hand-designed similarity matrix, weak temporal modeling, and the restricted 16-frame input.

1. Introduction

Start broad, then narrow quickly.

Main points:

Action recognition is a central computer vision task, but many benchmark datasets can be partially solved using shortcuts such as background, objects, or scene context rather than actual motion. Diving48 was designed as a fine-grained action recognition benchmark of competitive diving clips, with 48 dive classes and roughly 18k videos. Each dive class is defined by structured attributes: takeoff group, somersaults, twists, and body position. The dataset is challenging because classes are visually similar and static scene cues are less informative. The official dataset description says the 48 dive sequences are defined by combinations of takeoff, movements in flight, and entry/body position, and other work describes Diving48 as reducing static background biases from objects, scenes, and people.  ￼

The motivation of the project is that standard classification treats the 48 classes as mutually independent one-hot labels. This is a poor fit for Diving48. For example, a backward 2.5 somersault pike and a backward 2.5 somersault tuck are different classes, but visually and semantically closer than a forward twisting dive. A model trained only with hard cross-entropy receives no information about this structure: predicting a visually close dive is penalized exactly the same as predicting a completely different dive.

Your project asks whether incorporating semantic similarity between labels can improve fine-grained action recognition, or at least make the model’s errors more meaningful. Instead of only measuring full-class top-1 accuracy, the project also evaluates attribute-level correctness: takeoff accuracy, flip accuracy, twist accuracy, twist-status accuracy, position accuracy, top-5 accuracy, and semantic error severity.

End introduction with contributions:

This paper makes three contributions:

1. Implements a simple 16-frame ResNet50 mean-pooling baseline for Diving48.
2. Defines an interpretable 48×48 semantic similarity matrix over Diving48 labels.
3. Uses that similarity matrix for soft-label training and semantic contrastive learning, then evaluates both standard accuracy and semantic quality of errors.

2. Related Work

2.1 Fine-grained action recognition and Diving48

Diving48 is a fine-grained competitive diving dataset with about 18k trimmed clips and 48 dive classes. It was introduced to reduce representation bias in action recognition by limiting the usefulness of static background or object cues. This makes it a more motion-sensitive benchmark than datasets where the scene can strongly indicate the class.  ￼

Mention that high-performing modern video models exist on Diving48, including transformer and spatiotemporal architectures. You do not need to compare directly to all of them because your goal is not SOTA; your goal is to test a structured label-aware training idea under compute and frame constraints. A benchmark table reports recent methods reaching much higher accuracy, including LVMAE and Video-FocalNet, but those models are much heavier than your ResNet50 baseline and often use stronger temporal modeling or pretraining.  ￼

2.2 Temporal modeling for video classification

Many action recognition models explicitly model temporal relationships. 2D CNNs are efficient but weak at temporal reasoning; 3D CNNs model space-time jointly but are more computationally expensive. TSM addresses this by shifting part of the feature channels across the temporal dimension, allowing temporal information exchange while keeping the cost close to a 2D CNN.  ￼

In your project, you used MMAction2’s TSM pipeline as a stronger temporal reference and obtained about 78% accuracy. This supports the claim that temporal modeling is important for Diving48. However, the main semantic-label experiments used a simpler ResNet50 mean-pooling architecture because it was easier to modify and train under the 16-frame constraint.

2.3 Label structure in Diving48

Diving labels are not arbitrary. They encode components of the dive: takeoff direction, somersault count, twist count, and body position. Diving descriptions and scoring systems similarly decompose dives into direction/group, somersaults, twists, and position.  ￼

Some previous Diving48-specific methods exploit temporal or structural aspects of the sport. For example, Attentive Spatio-Temporal Representation Learning uses an attention-guided LSTM architecture for diving classification and reports improved performance over earlier 2D and 3D baselines on Diving48.  ￼

Your method differs because it does not just use the four categorical components as separate labels. Instead, it defines graded visual similarity between full dive classes. This is important because two dives can differ in multiple attributes but still appear visually close, while another pair may share some attributes but be visually very different.

3. Methodology

3.1 Task definition

Given a video clip sampled as 16 RGB frames, predict one of 48 Diving48 classes. Each class corresponds to a tuple:

[takeoff, somersaults, twists, position]

Examples:

["Back", "25som", "NoTwis", "TUCK"]

["Forward", "15som", "2Twis", "FREE"]

The main prediction target is the full 48-way class label, but the evaluation also decomposes predictions into attributes.

3.2 Baseline model

The baseline is intentionally simple.

For each video:

1. Load a tensor of shape [16, 3, 224, 224].
2. Pass each frame independently through an ImageNet-pretrained ResNet50.
3. Remove the final ResNet classification layer.
4. Mean-pool the 16 frame features.
5. Apply a linear classifier to predict one of 48 classes.

This gives a video feature vector:

z = \frac{1}{T}\sum_{t=1}^{T} f_\theta(x_t)

where T=16, f_\theta is ResNet50, and x_t is frame t.

The classifier predicts:

\hat{y} = \text{softmax}(Wz + b)

The baseline is trained with standard hard cross-entropy.

Important simplification to state clearly:

This model does not explicitly model frame order. Mean pooling gives it access to multiple frames, but it discards temporal sequence structure. This is a major limitation for Diving48 because twists and somersaults are inherently temporal.

3.3 Semantic similarity matrix

The core contribution is a manually designed semantic similarity matrix:

S \in \mathbb{R}^{48 \times 48}

where S_{ij} represents how visually/semantically similar class i is to class j.

Similarity is computed from four components:

1. Takeoff similarity
2. Flip/somersault similarity
3. Twist similarity
4. Position similarity

The final similarity is multiplicative:

S(i,j) =
S_\text{takeoff}(i,j)
\cdot
S_\text{flip}(i,j)
\cdot
S_\text{twist}(i,j)
\cdot
S_\text{position}(i,j)

Explain the design choices:

Takeoff receives high similarity if the takeoff group is the same, and lower similarity otherwise. This encodes that forward, back, inward, and reverse dives usually look different.

Flip and twist similarities are numeric. Exact matches receive similarity 1.0. Differences of full rotations are treated as more similar than half-step differences that change entry orientation. Large rotation differences are downweighted.

Position similarity is manually defined. Same position gets 1.0. Pike and straight are considered visually close. Pike and tuck are moderately close. Tuck and straight are less close. FREE is treated specially because it mainly occurs in twisting dives, so twist structure is allowed to dominate.

This matrix is interpretable and easy to inspect. For example, for a given class, you can print the top-k most similar classes and verify whether they make sense.

3.4 Soft semantic cross-entropy

Standard cross-entropy uses a one-hot target. Your soft-label method replaces the one-hot target with a probability distribution over semantically related classes.

For each true class y, define:

q_y = \text{softmax}(S_y / \tau)

where S_y is row y of the similarity matrix and \tau is the soft-target temperature.

Then train with soft cross-entropy:

\mathcal{L}_\text{soft}
=
-\sum_{c=1}^{48} q_y(c)\log p(c)

The total classification loss is:

\mathcal{L}
=
\mathcal{L}_\text{hard}
+
\lambda_\text{soft}\mathcal{L}_\text{soft}

Interpretation:

Hard CE says “predict exactly this class.”

Soft CE adds “but if the model spreads some probability mass to visually related classes, that is less bad than spreading it to unrelated classes.”

This should be especially helpful when 16 frames do not fully capture the dive or when two classes are visually ambiguous.

3.5 Semantic contrastive learning

The contrastive loss operates on the pooled video features. The goal is to shape the embedding space so that semantically similar dive classes are closer together.

For a batch of features z_i, normalize features and compute cosine similarities:

a_{ij} = \frac{z_i^\top z_j}{\tau}

Instead of treating only same-class examples as positives, use the similarity matrix to define soft positive weights between examples:

w_{ij} = S(y_i, y_j)

Self-pairs are removed, and each row is normalized. The contrastive objective encourages each example to place high probability on semantically similar examples in the batch.

The total loss becomes:

\mathcal{L}
=
\mathcal{L}_\text{hard}
+
\lambda_\text{soft}\mathcal{L}_\text{soft}
+
\lambda_\text{contrastive}\mathcal{L}_\text{contrastive}

This loss should improve the learned representation even if it does not always improve top-1 accuracy. The expected effect is that embeddings cluster by meaningful dive attributes such as takeoff, twist status, flip count, or body position.

3.6 Two-stage contrastive variant

You can describe this briefly as an additional experiment.

In the two-stage version:

Stage 1: Train the ResNet50 encoder using semantic contrastive learning so that the embedding space reflects dive similarity.

Stage 2: Freeze the encoder and train a linear classifier on top of the learned features.

This follows the common contrastive-learning pattern: first learn an embedding space, then train a classifier using the frozen or partially frozen representation. The key difference from generic contrastive learning is that your positives are not binary same/different labels; they are weighted by domain-specific semantic similarity.

If results are weak, frame this as exploratory. It may underperform end-to-end training because Diving48 is fine-grained and the classifier may need task-specific feature adaptation.

4. Experimental Setup

4.1 Dataset

Use Diving48 train/test annotations. Each video is represented by a precomputed tensor of 16 RGB frames at 224×224 resolution.

Mention:

The project uses RGB frames only.

No optical flow.

No full-video dense temporal sampling.

No Kinetics-pretrained video backbone in the main semantic experiments.

This keeps the experiment simple and isolates the effect of semantic label-aware training.

4.2 Training details

From your script:

Model: ResNet50 with ImageNet1K V2 weights.

Input: 16 frames per video.

Pooling: mean pooling over frame features.

Classifier: linear layer to 48 classes.

Optimizer: AdamW.

Learning rate: 3\times10^{-5}.

Batch size: 32.

Epochs: 10.

Augmentation: random horizontal flip during training.

Mixed precision: CUDA AMP.

Losses: hard CE, optional soft CE, optional semantic contrastive loss.

For the paper, do not emphasize the sweep. Say:

“We tuned the semantic loss weights on preliminary runs and report the best-performing semantic configuration.”

4.3 Baselines and comparisons

Include these comparisons:

1. ResNet50 mean-pooling baseline
    Hard cross-entropy only.
2. Semantic-label model
    Same architecture, but trained with the semantic loss terms.
3. TSM reference model
    MMAction2 TSM pipeline, RGB-only 16-frame setup, around 78% accuracy in your experiments.

The TSM result is not your main baseline because the architecture differs, but it establishes that temporal modeling substantially helps Diving48.

4.4 Evaluation metrics

Use both normal and semantic metrics.

Standard metrics:

Top-1 accuracy.

Top-5 accuracy.

Test cross-entropy loss.

Attribute metrics:

Takeoff accuracy.

Flip accuracy.

Twist-count accuracy.

Twist-status accuracy.

Position accuracy.

Semantic error metrics:

For each wrong prediction (y, \hat{y}), compute:

\text{severity} = 1 - S(y,\hat{y})

Then report:

Mean wrong-prediction similarity.

Mean error severity.

Number of wrong predictions.

Visualization:

UMAP of learned features, colored by:

Takeoff.

Position.

Twist status.

Flip count.

The purpose of UMAP is not to prove accuracy, but to qualitatively inspect whether the embedding space organizes around meaningful dive attributes.

5. Experimental Results

Since results are pending, structure this section with tables ready to fill.

5.1 Main classification results

Table:

Model	Top-1	Top-5	Test CE	Notes
ResNet50 mean pooling	TBD	TBD	TBD	Hard CE only
ResNet50 + semantic loss	TBD	TBD	TBD	Best semantic configuration
TSM reference	~78%	TBD	TBD	MMAction2 RGB temporal model

Expected interpretation:

If semantic loss improves top-1:

The similarity-aware objective improves full-class prediction, suggesting that structured label information helps regularize fine-grained recognition.

If top-1 is similar but top-5 improves:

The semantic loss may help the model rank plausible alternatives even when the top prediction is still wrong.

If top-1 drops slightly but error severity improves:

The method may trade exact classification for semantically smoother predictions. This is still meaningful because the model’s mistakes become closer to the correct dive.

If nothing improves:

The hand-designed similarity matrix may not align well enough with visual similarity, or the baseline architecture may be too temporally weak for semantic regularization to help.

5.2 Attribute-level results

Table:

Model	Takeoff Acc	Flip Acc	Twist Acc	Twist Status Acc	Position Acc
Baseline	TBD	TBD	TBD	TBD	TBD
Semantic model	TBD	TBD	TBD	TBD	TBD

Likely discussion:

Takeoff may be easier than exact twist/flip count because takeoff direction affects the overall motion pattern.

Twist status may be easier than exact twist count because detecting whether a dive twists is simpler than counting the number of twists.

Position may be difficult because body shape is small in the frame and changes rapidly.

Flip count and twist count are likely the hardest because they require precise temporal understanding across frames.

5.3 Semantic error severity

Table:

Model	Mean Similarity of Wrong Predictions	Mean Error Severity	Number Wrong
Baseline	TBD	TBD	TBD
Semantic model	TBD	TBD	TBD

This is one of the most important sections for your project.

Even if top-1 accuracy does not improve much, you can argue the method works if the model’s wrong predictions become more semantically reasonable. For example, confusing two dives that differ only by pike vs. tuck is less severe than confusing a forward non-twisting dive with a reverse twisting dive.

5.4 Feature-space visualization

Discuss the UMAPs.

Things to look for:

If colored by takeoff, do points form broad clusters by forward/back/inward/reverse?

If colored by twist status, do twisting and non-twisting dives separate?

If colored by flip count, is there a smooth gradient from fewer to more somersaults?

If colored by position, are TUCK/PIKE/STR/FREE separable?

Expected result:

The semantic loss may make UMAP clusters more attribute-organized than the baseline. Contrastive loss in particular should affect this more than soft CE because it directly operates on the embedding space.

Be careful with wording:

UMAP is qualitative and can distort distances. Use it as supporting evidence, not as the main proof.

6. Discussion

6.1 What worked

The project produced a lightweight way to inject domain knowledge into a standard classifier. The similarity matrix is interpretable, easy to modify, and does not require changing the architecture. This is useful because many strong video models are expensive to train, while the semantic loss can be added to a simple ResNet50 pipeline.

The evaluation is also stronger than only reporting top-1 accuracy. In a fine-grained dataset, not all errors are equally bad. The semantic error severity metric better reflects whether a model understands the structure of the task.

6.2 Why semantic labels may help

Hard labels make every incorrect class equally wrong. This is a bad assumption for Diving48. A semantic target distribution gives the model smoother supervision and may reduce overconfidence. It also provides useful information about relationships between classes, especially when the visual evidence from 16 frames is incomplete.

The contrastive version may help because it shapes the feature space directly. Instead of only making the classifier output correct labels, it encourages features from similar dives to occupy nearby regions. This could improve top-5 accuracy and semantic error severity even if top-1 accuracy changes only slightly.

6.3 Why the method may not improve top-1 accuracy

Several reasons:

First, the similarity matrix is hand-designed. It may encode your intuition about visual similarity, but not necessarily the similarity that the model needs for classification.

Second, the baseline model is temporally weak. Mean pooling discards frame order, so the model may not reliably distinguish actions that require counting rotations over time.

Third, semantic smoothing can hurt exact classification if the soft labels are too broad. If the target distribution gives too much weight to neighboring classes, the classifier may become less decisive.

Fourth, the contrastive loss depends on batch composition. If a batch does not contain enough semantically related examples, the contrastive signal may be noisy or weak.

Fifth, the method uses only 16 frames. Some dives may require more temporal resolution to distinguish exact twist or somersault count.

6.4 Why TSM does better

Your TSM result around 78% suggests that temporal modeling is very important. TSM explicitly exchanges information across neighboring frames, while the ResNet50 mean-pooling baseline only averages independent frame features. TSM was designed to add temporal reasoning to 2D CNNs without the full cost of 3D convolution.  ￼

This supports a key conclusion: semantic labels help with the supervision problem, but they do not replace temporal modeling. The best future version would combine your semantic loss with a stronger temporal architecture.

7. Limitations

State these directly.

The largest limitation is that the similarity matrix is manually designed. It reflects assumptions about which dives look similar, but it is not learned from data or validated by human annotations.

The model uses only 16 frames, which may miss important parts of the dive. Fine-grained differences in twists and somersaults often require precise temporal coverage.

The ResNet50 mean-pooling baseline ignores frame order. It can recognize pose and appearance cues, but it cannot explicitly model the sequence of motion.

The method may improve semantic error quality without improving top-1 accuracy. Depending on the application, this may or may not be valuable.

The evaluation is limited to RGB frame tensors. Optical flow, pose, segmentation, or diver tracking could provide stronger motion-focused signals.

The project reports one best semantic configuration rather than a full systematic hyperparameter study. This is reasonable for the course project, but it limits conclusions about robustness.

8. Future Work

Good future work ideas:

Combine semantic loss with TSM or another temporal model.

Learn the similarity matrix from data instead of manually defining it.

Use pose estimation or diver segmentation so the model focuses on body motion instead of background.

Use hierarchical classification: first predict takeoff/twist/position/flip attributes, then predict the full class.

Train with more frames per video or multi-clip evaluation.

Evaluate confusion matrices to see which dive families benefit most.

Use a learned metric loss where similarity is calibrated from actual confusion patterns or expert annotations.

Try soft labels based on dive-code structure plus visual embeddings from a pretrained video model.

Compare one-stage and two-stage contrastive learning more carefully.

9. Conclusion

Suggested conclusion:

This project explored whether structured label information can improve fine-grained action recognition on Diving48. Starting from a simple ResNet50 mean-pooling baseline, we constructed a semantic similarity matrix over the 48 dive classes and used it for soft cross-entropy and semantic contrastive learning. The method is motivated by the fact that Diving48 labels are not independent: classes share meaningful attributes such as takeoff, somersaults, twists, and body position. The proposed approach provides a lightweight and interpretable way to inject this structure into training.

The main finding is expected to be one of the following, depending on results:

If accuracy improves:
The semantic objective improves both accuracy and error quality, showing that domain-specific label structure can help fine-grained action recognition even with a simple architecture.

If semantic severity improves but accuracy does not:
The semantic objective does not substantially improve exact classification, but it makes the model’s mistakes more meaningful. This suggests that label-aware training changes the representation in useful ways, even if stronger temporal modeling is still needed for top-1 gains.

If results are weak:
The experiment shows that semantic label structure alone is not enough to overcome the temporal difficulty of Diving48. Stronger architectures such as TSM remain important, and future work should combine semantic supervision with explicit temporal modeling.

References to include

Use these in your final bibliography:

Li et al., Towards Action Recognition without Representation Bias. Introduces Diving48 and the motivation around reducing representation bias.  ￼

Diving48 dataset page. Dataset size and class structure.  ￼

Lin et al., TSM: Temporal Shift Module for Efficient Video Understanding, ICCV 2019. Efficient temporal modeling with 2D CNNs.  ￼

Kanojia et al., Attentive Spatio-Temporal Representation Learning for Diving Classification, CVPRW 2019. Diving-specific attentive LSTM approach.  ￼

USA Diving / diving code references for takeoff, twist, somersault, and position definitions.  ￼
