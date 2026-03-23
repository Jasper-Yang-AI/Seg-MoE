# Draft for Artificial Intelligence in Medicine

## Title

Seg-MoE: Uncertainty-guided two-stage multi-expert segmentation and patient-level failure triage for multi-parametric prostate MRI

## Running title

Uncertainty-guided multi-expert prostate MRI segmentation

## Authors

[[Author 1]], [[Author 2]], [[Author 3]], [[Corresponding Author]]

## Affiliations

[[Affiliation 1]]

## Corresponding author

[[Name, address, email]]

## Highlights

- We propose a leakage-free two-stage multi-expert segmentation framework for multi-parametric prostate MRI.
- Out-of-fold uncertainty and expert disagreement are reused for both segmentation refinement and dynamic fusion.
- The framework supports patient-level failure triage, enabling prioritization of cases for manual review.
- In a completed full fold0 validation, patient-level disagreement achieved an AUROC of 0.847 for identifying the lowest-Dice decile.

## Abstract

**Objective:** Reliable prostate MRI segmentation requires both accurate delineation and the ability to identify cases that are likely to fail in routine clinical use. We developed a two-stage multi-expert framework that combines heterogeneous segmentation models, uncertainty-aware refinement, and patient-level failure triage.

**Materials and methods:** We built a leakage-free two-stage ensemble pipeline for local multi-parametric prostate MRI, comprising T2-weighted imaging, ADC, and DWI. The dataset currently available in this project contains 51,223 annotated slices from 3,439 patients, including a fixed 75-patient test subset and a five-fold development split over the remaining 3,364 patients. Three complementary experts were used in the first stage: nnU-Net, Swin UNETR, and SegResNet. Strict out-of-fold predictions from stage 1 were used to train stage 2 experts with uncertainty channels derived from entropy and inter-expert disagreement. A patch-level gating network was designed to fuse expert predictions using stage 2 logits, stage 1 semantic priors, image context, and spatial cues. We further evaluated whether uncertainty and disagreement signals could identify low-quality segmentations at the patient level. Segmentation was evaluated using Dice, HD95, and NSD; failure triage was evaluated using Spearman correlation, AUROC, and top-k risk enrichment.

**Results:** In the currently completed fold0 validation, the stage 1 experts achieved mean foreground Dice scores of 0.8253, 0.7801, and 0.7816, while simple mean fusion reached 0.8038. After uncertainty-aware stage 2 refinement, the three stage 2 experts reached Dice scores of 0.8374, 0.8367, and 0.8386, with HD95 values of 2.62, 2.62, and 2.58, respectively. In the full fold0 patient-level failure analysis over 673 patients, disagreement-derived risk achieved a Spearman correlation of -0.467 with Dice and an AUROC of 0.847 for detecting the lowest-Dice decile, outperforming entropy alone (AUROC 0.807). [[Insert full five-fold and fixed-test segmentation results after gating inference and consolidated evaluation are completed.]]

**Conclusion:** A strict out-of-fold multi-expert design can use uncertainty not only for segmentation refinement but also for clinically meaningful failure triage. The proposed framework supports both improved prostate MRI segmentation and risk-aware review prioritization.

**Keywords:** prostate MRI; medical image segmentation; mixture of experts; uncertainty estimation; failure detection; clinical AI

## 1. Introduction

Prostate MRI has become a core imaging tool for lesion detection, risk stratification, treatment planning, and longitudinal follow-up. Accurate delineation of the prostate and its clinically relevant subregions, including the peripheral zone, transition zone, and lesion-related targets, can support both quantitative analysis and downstream decision-making. However, automatic segmentation in multi-parametric prostate MRI remains challenging because of anatomical variability, lesion heterogeneity, scanner-related differences, and inconsistent local image quality.

In recent years, deep learning has substantially improved medical image segmentation, and strong architectures such as nnU-Net, transformer-based variants, and residual encoder-decoder models have become common reference baselines. Nonetheless, no single segmentation architecture is consistently optimal across all patients, anatomical regions, and difficulty profiles. This is particularly relevant in prostate MRI, where some cases are dominated by clear zonal anatomy while others are affected by small lesions, ambiguous boundaries, or unfavorable diffusion characteristics. Static fusion strategies, such as simple averaging or voting, may improve robustness, but they do not explicitly model when one expert should be trusted more than another.

Another limitation of the current literature is that segmentation studies often emphasize average overlap scores without addressing operational reliability. In real clinical workflows, the practical question is not only whether a method performs well on average, but also whether it can identify cases that are likely to require manual review. This distinction is important for clinical adoption because even a strong average Dice score may hide a subset of high-risk failures.

To address these issues, we designed Seg-MoE, a strict out-of-fold two-stage multi-expert segmentation framework for multi-parametric prostate MRI. The framework combines three heterogeneous first-stage experts, uncertainty-aware second-stage refinement, and patch-level dynamic gating. Instead of treating uncertainty as a post hoc by-product, we explicitly reuse entropy and inter-expert disagreement as model inputs for stage 2 learning and expert routing. We additionally evaluate whether those same signals can identify low-quality segmentations at the patient level, thereby enabling a risk-aware review workflow.

The main contributions of this work are as follows. First, we propose a leakage-free two-stage multi-expert segmentation framework in which all second-stage learning is driven by strict out-of-fold first-stage predictions. Second, we introduce an uncertainty-guided design in which entropy and expert disagreement are reused for both segmentation refinement and adaptive fusion. Third, we extend the evaluation from segmentation accuracy to patient-level failure triage, showing that disagreement signals can identify likely low-quality cases for prioritized review.

## 2. Materials and methods

### 2.1. Study design and cohort

This study was designed as a retrospective development study on a local multi-parametric prostate MRI cohort. The current project contains 51,223 2D slices derived from 3,439 unique patients. Each sample contains three imaging channels corresponding to T2-weighted imaging, ADC, and DWI. The segmentation labels include four classes: background, peripheral zone, transition zone, and lesion-related foreground.

At the patient level, the dataset is organized into a fixed-test, five-fold development split. A total of 75 patients are reserved as a fixed test subset, while the remaining 3,364 patients are used for five-fold development. For fold0, the validation subset contains 10,084 slices from 673 patients, and the training subset contains 40,021 slices from 2,691 patients. All splits are patient-level to avoid leakage across slices from the same examination.

[[Insert institutional review board approval number, consent statement, and center description here.]]

### 2.2. Preprocessing and data partitioning

The dataset configuration specifies a 2D slice-based pipeline with an input size of 256 x 256 pixels and three imaging channels. The raw prostate MRI volumes are converted into 2D slices using a non-empty slice policy so that slices without relevant foreground are excluded from the training pool. Intensity normalization uses percentile-based clipping at the volume level followed by per-channel normalization. The slice axis is fixed during preprocessing to ensure consistent orientation across the cohort.

All experiments use patient-level group splitting, with `patient_id` as the grouping key. This ensures that slices from the same patient never appear in both training and validation subsets. Strict out-of-fold predictions are then generated for each validation fold and stored together with metadata describing the sample fold, predictor fold, and checkpoint provenance.

### 2.3. Stage 1 expert models

Stage 1 consists of three complementary segmentation experts:

1. nnU-Net, representing a strong convolutional baseline with self-configuring design principles.
2. Swin UNETR, representing a transformer-based segmentation architecture.
3. SegResNet, representing a residual encoder-decoder architecture.

These models were selected to encourage architectural diversity rather than to maximize similarity within a family of networks. The stage 1 experiments use architecture-specific training recipes and imported official-style checkpoints. In the current 2D configuration, nnU-Net uses a seven-stage encoder-decoder with a 256 x 256 patch size, Swin UNETR uses 2D transformer blocks with feature size 48, and SegResNet uses a MONAI SegResNetDS configuration with residual blocks and deconvolutional upsampling.

### 2.4. Strict out-of-fold stacking design

Because stage 2 learning consumes stage 1 predictions as input, data leakage would occur if those predictions were produced on the same samples used to train the stage 1 experts. We therefore adopt a strict out-of-fold design. For each fold `k`, the stage 1 experts trained on `train_foldk` are used only to predict `val_foldk`, and the resulting predictions are stored in an out-of-fold cache. Each cached record explicitly stores the sample fold, predictor fold, split name, expert list, and checkpoint paths.

This design is a methodological safeguard rather than the main novelty of the work. Its role is to ensure that stage 2 and downstream gating learn from realistic first-stage outputs rather than overfitted training-set predictions. The project includes an audit step that checks fold consistency, coverage, and manifest validity before stage 2 training.

### 2.5. Stage 2 uncertainty-aware refinement

Stage 2 uses the stage 1 out-of-fold outputs to refine segmentation. For each 2D slice, the stage 2 input is the concatenation of:

- the original three-channel prostate MRI slice,
- the stacked stage 1 out-of-fold probability maps from all experts, and
- uncertainty channels derived from the same out-of-fold predictions.

In the current implementation, uncertainty channels consist of one normalized entropy map and one per-class disagreement map. With three experts and four classes, the stage 2 input contains 20 channels in total: 3 image channels, 12 first-stage probability channels, and 5 uncertainty channels. This input design is implemented identically for training and inference to avoid feature mismatch.

To stabilize optimization, each stage 2 expert is initialized from its corresponding stage 1 model. Shared weights are copied directly, while the first convolutional stem is widened to accept the additional channels. The original image-channel weights are retained, and the additional channels are zero-initialized. This allows stage 2 to start from a model that already segments reasonably well and to focus on learning how to exploit the first-stage predictive context.

Stage 2 is trained with a dedicated configuration that differs from stage 1 in three respects. First, it uses a lower learning rate and shorter schedule to preserve transferred knowledge. Second, it uses architecture-specific optimizer overrides to retain diversity among experts. Third, it uses a combined cross-entropy, soft Dice, and boundary loss objective, motivated by the need to refine boundaries and small targets that may be missed by stage 1.

### 2.6. Patch-level dynamic gating

After stage 2 refinement, expert predictions are further fused by a patch-level gating network. The gating model operates on local patches rather than whole images, allowing expert weights to vary spatially across the image. This is motivated by the observation that one expert may be preferable in the central gland while another may perform better near lesions or ambiguous boundaries.

The 2D gating configuration used in this project is class-aware and patch-based. It uses a patch size of 64 pixels with a stride of 32 pixels and predicts either class-specific or shared fusion weights depending on configuration. In the current implementation, per-class gating is enabled. The gating features include:

- stage 2 expert logits as high-order semantic inputs,
- entropy-aware expert encoding,
- pairwise consensus and disagreement features across experts,
- optional confidence features derived from attention-weighted entropy,
- stage 1 semantic priors consisting of mean probabilities, entropy, and disagreement,
- image appearance context,
- spatial position channels, and
- slice-position embeddings.

The gating network also includes temperature annealing during training, a load-balancing regularizer to prevent expert collapse, and a spatial smoothness penalty to encourage local coherence of weight maps.

### 2.7. Patient-level failure triage

In addition to segmentation refinement, we evaluate whether stage 1 uncertainty signals can identify cases that are likely to fail. This analysis is motivated by clinical deployment, where a system should ideally flag difficult cases for manual review rather than returning a segmentation without qualification.

For each slice, we compute three risk signals from stage 1 out-of-fold predictions:

1. normalized predictive entropy from the mean expert probabilities,
2. expert disagreement measured as the standard deviation across experts, and
3. pairwise total variation disagreement between expert probability maps.

Each slice is assigned a segmentation quality score using mean foreground Dice against the ground truth. The slice-level risk signals are then aggregated to the patient level by grouping all slices sharing the same patient identifier. At the patient level, we compute mean Dice, minimum Dice, mean entropy, mean disagreement, and mean pairwise total variation.

Failure is defined as belonging to the lowest-Dice decile within the evaluation cohort. We then assess whether each risk score can identify those failures using Spearman correlation between risk and Dice, AUROC for failure classification, and top-k triage enrichment metrics, including precision, recall, and lift at the top 10% highest-risk patients.

### 2.8. Baselines and comparison methods

The experimental design includes several comparison levels:

- stage 1 single experts,
- stage 2 single experts,
- static fusion baselines, including mean ensemble and majority voting,
- learned combiners, including ordinary least squares ensemble, decision template, and WE-CLPSO, and
- dynamic gating-based fusion.

This layered comparison is intended to answer three separate questions: whether stage 2 improves upon stage 1, whether uncertainty-aware refinement improves upon simple fusion, and whether dynamic patch-level routing improves upon static ensemble rules.

### 2.9. Evaluation metrics and statistical analysis

Segmentation performance is assessed using Dice similarity coefficient, 95th percentile Hausdorff distance (HD95), and normalized surface Dice (NSD) as the core metrics, with IoU, ASD, sensitivity, and precision as supplementary measures. Metrics are reported both as aggregated foreground means and as per-class values for the clinically relevant foreground classes.

Failure triage is evaluated at both slice and patient levels. For patient-level analysis, the primary endpoint is AUROC for detection of the lowest-Dice decile. Secondary endpoints include top-10% triage precision, recall, and lift.

[[Insert final statistical comparison protocol here, e.g., Wilcoxon signed-rank testing across folds or patient-level paired testing on the fixed test set.]]

## 3. Results

### 3.1. Cohort characteristics

The 2D prostate MRI cohort used in this study contains 51,223 slices from 3,439 patients. Of these, 75 patients are reserved as a fixed test subset and 3,364 patients constitute the development cohort. In fold0, the validation subset contains 10,084 slices from 673 patients. The three-channel input corresponds to T2-weighted imaging, ADC, and DWI, and the segmentation labels contain one background class and three foreground targets.

**Table 1. Cohort summary**

| Item | Value |
| --- | --- |
| Total patients | 3,439 |
| Total slices | 51,223 |
| Fixed test patients | 75 |
| Development patients | 3,364 |
| Fold0 validation patients | 673 |
| Fold0 validation slices | 10,084 |
| Input channels | 3 (T2w, ADC, DWI) |
| Classes | 4 (background, PZ, TZ, lesion) |

### 3.2. Currently verified fold0 segmentation results

The results currently verifiable from the local workspace show a consistent improvement from stage 1 to stage 2. In fold0 validation, the stage 1 out-of-fold single experts achieved mean foreground Dice values of 0.8253 for nnU-Net, 0.7801 for Swin UNETR, and 0.7816 for SegResNet. Simple probability averaging across the three stage 1 experts reached a Dice of 0.8038, which was lower than the best individual expert.

After stage 2 uncertainty-aware refinement, all three experts improved substantially and converged to a narrower performance range. The stage 2 SegResNet achieved the best currently verified fold0 mean foreground Dice of 0.8386, followed by nnU-Net at 0.8374 and Swin UNETR at 0.8367. The corresponding NSD values were 0.8650, 0.8628, and 0.8632, while HD95 was approximately 2.58 to 2.62.

These results suggest that the stage 2 design primarily acts as a refinement module that lifts weaker stage 1 experts and regularizes performance across architectures. They also support the hypothesis that stage 1 uncertainty-aware contextual features provide useful guidance beyond what can be achieved by static averaging alone.

**Table 2. Currently verified fold0 segmentation results from the workspace**

| Method | Fold0 validation Dice | Fold0 validation HD95 | Fold0 validation NSD | Status |
| --- | --- | --- | --- | --- |
| Stage 1 nnU-Net | 0.8253 | [[Not yet consolidated]] | [[Not yet consolidated]] | Verified from OOF cache |
| Stage 1 Swin UNETR | 0.7801 | [[Not yet consolidated]] | [[Not yet consolidated]] | Verified from OOF cache |
| Stage 1 SegResNet | 0.7816 | [[Not yet consolidated]] | [[Not yet consolidated]] | Verified from OOF cache |
| Stage 1 mean ensemble | 0.8038 | [[Not yet consolidated]] | [[Not yet consolidated]] | Verified from OOF cache |
| Stage 2 nnU-Net | 0.8374 | 2.6226 | 0.8628 | Verified from layer2 summary |
| Stage 2 Swin UNETR | 0.8367 | 2.6225 | 0.8632 | Verified from layer2 summary |
| Stage 2 SegResNet | 0.8386 | 2.5827 | 0.8650 | Verified from layer2 summary |
| Dynamic gating | [[Insert after gating inference]] | [[Insert after gating inference]] | [[Insert after gating inference]] | Pending |

**Text for final revision:** Replace Table 2 with consolidated five-fold development results and fixed-test results once `gating_inference.py` and the complete evaluation tables have been generated.

### 3.3. Per-class findings

The currently available fold0 stage 2 summaries suggest balanced performance across foreground classes, with the strongest results observed for the more anatomically stable glandular regions and slightly lower scores in the more challenging foreground class. For the best currently verified stage 2 SegResNet model, the per-class Dice values in fold0 validation were approximately 0.7790, 0.8745, and 0.8622 for the three foreground labels.

This pattern is clinically plausible. Zonal anatomy tends to be more spatially coherent, whereas lesion-related targets may be smaller, less regular, and more vulnerable to ambiguity in diffusion-weighted imaging. The use of boundary-aware loss and local gating is intended to particularly benefit such difficult regions, but the final claim on lesion-specific gains should be made only after complete dynamic gating results are available.

### 3.4. Patient-level failure triage

A key secondary objective of this study was to determine whether uncertainty and disagreement signals can identify patients with poor segmentation quality. In the completed full fold0 validation, the patient-level analysis included 673 patients and defined failure as the lowest-Dice decile. At this level, disagreement-derived risk showed the strongest association with segmentation quality, with a Spearman correlation of -0.467 and an AUROC of 0.847. Pairwise total variation performed similarly with an AUROC of 0.845, while entropy alone reached an AUROC of 0.807.

The top-10% triage analysis further suggests practical enrichment. When patients were ranked by disagreement-derived risk, the top 10% captured 48.5% of all low-Dice failures, corresponding to a lift of 4.80 relative to the baseline failure rate. These findings indicate that expert disagreement can serve as a clinically meaningful proxy for segmentation reliability and may be useful for prioritizing manual review.

At the slice level, the same signals were weaker but still informative. In fold0, slice-level disagreement achieved an AUROC of 0.665, whereas patient-level aggregation improved the AUROC to 0.847. This gap supports the view that patient-level triage is more clinically relevant than slice-level uncertainty inspection.

**Table 3. Patient-level failure triage in completed full fold0 validation**

| Risk score | n patients | Spearman rho vs Dice | AUROC for lowest-Dice decile | Top-10% precision | Top-10% recall | Lift |
| --- | --- | --- | --- | --- | --- | --- |
| Entropy | 673 | -0.385 | 0.807 | 0.485 | 0.485 | 4.803 |
| Disagreement (std) | 673 | -0.467 | 0.847 | 0.485 | 0.485 | 4.803 |
| Pairwise TV | 673 | -0.463 | 0.845 | 0.485 | 0.485 | 4.803 |

### 3.5. Qualitative observations

Qualitative visualization tools in the current project include overlay plots and stage 2 Grad-CAM-style maps. Preliminary case inspection suggests that the experts often disagree in difficult slices containing small or irregular lesions, low-contrast boundaries, or mismatched appearance across modalities. In such cases, one expert may better preserve lesion extent while another better suppresses false positives, reinforcing the motivation for adaptive fusion rather than uniform averaging.

Representative figures for the final manuscript should include:

- a successful easy case with high consensus,
- a difficult case with marked expert disagreement but successful refinement,
- a failure case flagged as high risk by the triage module, and
- a visualization of patch-level expert weights from the dynamic gate.

[[Insert Figure 3 and Figure 4 once gating overlays and weight maps are exported.]]

### 3.6. Results pending completion before submission

The following analyses are defined in the codebase but are not yet fully materialized in the current workspace and should be completed before submission:

- consolidated five-fold stage 1, stage 2, and dynamic gating results,
- fixed-test evaluation using the final selected model,
- full comparison with majority voting, OLE, decision template, and WE-CLPSO,
- statistical significance testing across folds or test cases, and
- dynamic gating per-class and per-case visualizations.

The manuscript text above is therefore a submission-oriented first draft rather than the final version.

## 4. Discussion

This study was motivated by two practical observations. First, heterogeneous architectures do not fail on the same cases in prostate MRI. Second, the same uncertainty and disagreement signals that reveal difficult cases may also help a second-stage model refine predictions. The proposed framework addresses both observations within a single pipeline.

The currently verified fold0 results support three interpretations. First, stage 2 refinement appears to be more effective than simple mean fusion. Although the stage 1 mean ensemble reached a Dice of 0.8038 in fold0 validation, the uncertainty-aware stage 2 experts improved to approximately 0.837 to 0.839. This suggests that simply averaging heterogeneous predictions may be insufficient when expert outputs are complementary but imperfectly calibrated. Stage 2 provides a mechanism for learning how to use those predictions in the context of the original image and uncertainty signals.

Second, uncertainty and disagreement appear to be informative not only as auxiliary inputs but also as operational reliability markers. The patient-level failure triage analysis showed that disagreement outperformed entropy for identifying low-quality cases. This is intuitively reasonable: entropy reflects predictive ambiguity within the mean distribution, whereas inter-expert disagreement captures model diversity and conflict, which may better expose structural failure modes.

Third, patient-level aggregation is essential for clinical interpretation. In this project, slice-level uncertainty was only moderately predictive of failure, whereas patient-level aggregation substantially improved AUROC. This matters because clinical review and reporting are performed at the case level rather than the slice level. A triage module that flags a patient as high risk is more actionable than a heatmap on isolated slices.

The proposed framework is also aligned with the expectations of translational medical AI. Rather than presenting segmentation as a fully autonomous output, it explicitly acknowledges residual uncertainty and supports risk-aware review prioritization. This type of design is likely to be more compatible with clinical workflows than a pure leaderboard-oriented segmentation model.

Several limitations should be noted. First, the present draft is based on a local retrospective dataset and does not yet include an external validation cohort. Second, the currently consolidated workspace results are strongest for the 2D pipeline; although the codebase also supports 3D prostate and liver experiments, those branches are better suited for supplementary evidence or follow-up work unless they are fully completed before submission. Third, the current manuscript draft does not yet contain full dynamic gating results or complete five-fold statistical testing. Fourth, the failure triage module is retrospective and based on segmentation quality against reference labels rather than prospective radiologist review decisions.

Future work should extend this framework in three directions. The first is external validation across scanners and institutions. The second is reader-in-the-loop evaluation to determine whether risk-aware triage improves review efficiency or reduces missed failures. The third is extension from patient-level risk to structure-specific or lesion-specific reliability scoring, which may be useful in targeted biopsy and treatment planning workflows.

## 5. Conclusion

We present a strict out-of-fold two-stage multi-expert segmentation framework for multi-parametric prostate MRI that combines heterogeneous first-stage experts, uncertainty-aware second-stage refinement, dynamic patch-level fusion, and patient-level failure triage. The currently verified fold0 results indicate that stage 2 refinement improves upon both single-expert and simple mean-fusion baselines, and that disagreement-derived risk can identify low-quality cases at the patient level with strong discrimination. These findings support the broader idea that uncertainty should be treated not only as a confidence estimate but also as a trainable signal for refinement and a clinically useful signal for review prioritization.

## Declarations

### Ethics approval and consent to participate

[[Insert IRB approval number, waiver or consent statement, and institutional details.]]

### Consent for publication

[[Insert statement.]]

### Availability of data and materials

The codebase underlying this manuscript is organized in the Seg-MoE project. Public release of the full imaging dataset may be restricted by institutional policy and patient privacy regulations. [[Insert exact data-sharing statement and code release plan.]]

### Competing interests

The authors declare that they have no competing interests. [[Revise if needed.]]

### Funding

[[Insert funding information or “This research received no external funding.”]]

### Author contributions

[[Insert CRediT roles.]]

### Acknowledgments

[[Insert acknowledgments.]]

## Figure and table plan

**Figure 1.** Overall Seg-MoE pipeline: stage 1 experts -> strict out-of-fold cache -> stage 2 refinement -> stage 2 out-of-fold cache -> patch-level gating -> patient-level failure triage.

**Figure 2.** Patch-level gating architecture with stage 2 logits, stage 1 semantic priors, image context, and spatial cues.

**Figure 3.** Representative qualitative cases with expert outputs, fused output, overlay maps, and risk scores.

**Figure 4.** Patient-level failure triage curves and scatter plots showing the relationship between risk score and Dice.

**Table 1.** Cohort characteristics.

**Table 2.** Main segmentation performance across stage 1, stage 2, static fusion, and dynamic gating.

**Table 3.** Patient-level failure triage performance.

**Table 4.** Ablation study on uncertainty channels, stage 1 priors, and gating context modules.

## References

The references below are intentionally kept as a draft list and should be formatted according to journal requirements before submission.

1. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*. 2021.
2. Hatamizadeh A, Tang Y, Nath V, et al. Swin UNETR for semantic segmentation in medical imaging. [[Update exact citation details.]]
3. Myronenko A. 3D MRI brain tumor segmentation using autoencoder regularization. [[Update exact citation details.]]
4. Kervadec H, Bouchtiba J, Desrosiers C, Granger E, Dolz J, Ayed IB. Boundary loss for highly unbalanced segmentation. *MIDL*. 2019.
5. Maier-Hein L, Reinke A, Godau P, et al. Metrics Reloaded: recommendations for image analysis validation. *Nature Methods*. [[Update exact year and citation details.]]
6. Kittler J, Hatef M, Duin RPW, Matas J. On combining classifiers. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. 1998.
7. [[Add prostate MRI clinical background references.]]
8. [[Add uncertainty and failure detection references in medical image segmentation.]]

---

## Internal completion checklist

Remove this section before submission.

- Replace all `[[...]]` placeholders.
- Export consolidated five-fold evaluation tables.
- Run dynamic gating inference and insert final gating results.
- Add statistical significance testing.
- Insert ethics, funding, and author information.
- Update the abstract and conclusion with final cross-validation and fixed-test numbers.
