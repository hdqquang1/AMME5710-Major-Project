# Fruit detection and counting under occlusion: verified literature review (draft)

AMME5710 Major Project (has_sai) · topic: colour-free fruit detection and 3D de-duplicated counting behind foreground occlusion · compiled 15 October 2026 · status: DRAFT for team review

> **AI-use disclosure.** This review was compiled with AI assistance (Claude, running the `deep-research` skill in `lit-review` mode). Every reference was machine-checked against bibliographic indexes (method below), and every quantitative figure is tagged with where it came from. The team should spot-check key figures against the PDFs before quoting them in the report.

---

## 0. How this review was built

**Mode and form.** Narrative or integrative literature review (`deep-research` `lit-review` mode: bibliography → source verification → synthesis), with the user's overrides of IEEE citation style and Australian English. The skill's review-form prompt was not shown because the review was delegated to a subagent with a fixed brief. If the team would rather run a systematic or scoping review, that is a separate decision for the team.

**Search strategy (executed 15 October 2026).**

- *Indexes:* CrossRef REST API, Semantic Scholar Graph API, OpenAlex API, arXiv API. For discovery and grey literature, also web search, publisher pages and institutional repositories.
- *Query families (Boolean):*
  - (fruit OR berry OR cherry OR grape OR apple OR mango OR citrus OR pepper) AND (count\* OR yield OR detect\* OR locali\*) AND ("multiple view" OR multi-view OR "structure from motion" OR tracking OR "double count\*" OR occlu\* OR amodal OR "next best view").
  - Colour-free cues: (shape OR "radial symmetry" OR "Hough" OR specular OR texture OR Hessian OR "Laplacian of Gaussian" OR RGB-D OR depth OR "green fruit" OR immature).
  - Cherries: cherry AND (detection OR dataset OR harvesting OR yield).
  - Foundation models: ("Segment Anything" OR "SAM 3" OR "Grounding DINO" OR open-vocabulary) AND (agriculture OR fruit OR orchard).
  - Phone sensing: (iPhone OR iPad OR ARKit) AND (LiDAR OR pose OR tracking) AND accuracy.
  - Evaluation: (synthetic OR rendered) AND (fruit OR orchard) AND (dataset OR benchmark), plus point-localisation metrics.
- *Date range:* 1987 to October 2026. Seminal pre-2015 works are kept deliberately because colour-free methods peaked before deep learning. *Language:* English.
- *Inclusion:* peer-reviewed papers, plus major preprints for foundation models, that report a fruit detection, counting or localisation method, dataset or evaluation relevant to occlusion, multi-view de-duplication, colour-free cues, cherries, SAM 3 or open-vocabulary segmentation, or phone LiDAR and ARKit accuracy.
- *Exclusion:* papers whose content could not be retrieved beyond the title (listed in Appendix B rather than annotated from memory), vendor claims, and theses (recorded as grey literature, marked UNVERIFIED).

**Verification protocol (applied to every entry).** Tier 0: a DOI lookup in CrossRef, with a Levenshtein title cross-check of at least 0.70 to catch DOIs that resolve to the wrong paper. If that failed, a CrossRef bibliographic search. Then independent matches in Semantic Scholar and OpenAlex, including OpenAlex's `is_retracted` flag. Preprints were checked against the arXiv API. Grey literature was checked on the publisher's own page. Status tags used below:

| Tag | Meaning |
|---|---|
| **V3** | CrossRef DOI record plus Semantic Scholar and OpenAlex matches, all agreeing on title, authors, venue and year |
| **V2** | CrossRef DOI record plus one other index (Semantic Scholar was rate-limited on a few lookups) |
| **VarX** | Verified on arXiv (and the venue page where stated). arXiv DOIs are DataCite, not CrossRef |
| **Vpub** | Verified on the publisher's or issuer's own page (grey literature) |
| **UNVERIFIED** | Existence or content could not be confirmed. Never used as evidence |

*Retraction check:* OpenAlex returned `is_retracted = false` for every included DOI (15 October 2026).

*Corrections the verification forced on the brief:* see Appendix A. For example, Nuske 2011's title reads "…vineyards **by** visual grape detection", and "Wang/Walsh/Koirala MangoYOLO (2019)" is two separate papers.

**Read scope and provenance of numbers.** Full texts were read for [2], [3], [5], [6], [14], [39], [43], [64] and [81], and partially for [66] (open-access arXiv and institutional copies; table, figure and section numbers are cited). All other figures come from the authors' abstracts, retrieved through the indexes or institutional repository records, and are tagged *(abstract)*. Per the skill's rules, method-level weaknesses are only stated where the text was read or the authors acknowledge them. Otherwise the caveat is marked *reader-inferred* and concerns only applicability to our setup.

**Screening counts (PRISMA-style).** 104 candidate records were identified: 100 through scripted three-index lookup and 4 through arXiv and publisher checks. 1 duplicate was removed (a deliberate de-duplication test). 1 probe title matched nothing, as expected. 102 were screened. 13 were excluded: 6 because content could not be retrieved beyond the title, 7 as tangential (Appendix B). That leaves **89 included**: 64 fully annotated and 25 listed as supporting references with one-line notes.

**Distributional skew advisory.** Methodology: empirical system or method papers make up the large majority of known entries, which is expected for applied agricultural vision. This is a coverage signal, not a defect. Venue: the largest single families are *Computers and Electronics in Agriculture* and *Biosystems Engineering*, together below the 70% threshold. Geography spans Australia, the USA, Spain, Israel, the Netherlands, Germany, Austria, China and Chile, with no single region at 70% or more.

---

## Part A. Annotated bibliography (IEEE numbering, grouped by theme)

Entry format: **IEEE reference** · `verification tag` · read scope. Then a summary, the key quantitative result (with its source location), relevance to our 0.6 m rail + toy-cherry wall, and a caveat where one is warranted. Caveat provenance is labelled *author-acknowledged* or *reader-inferred*.

### A1. Multi-view and 3D counting that avoids double counting

**[1]** M. Stein, S. Bargoti, and J. Underwood, "Image based mango fruit detection, localisation and yield estimation using multiple view geometry," *Sensors*, vol. 16, no. 11, Art. no. 1915, Nov. 2016, doi: 10.3390/s16111915. `V3` · abstract
- *Summary:* A ground vehicle carries a camera, LiDAR and GPS/INS through a commercial mango orchard. Faster R-CNN detects fruit in every image. Pairwise correspondences between images come from the navigation trajectory, so each fruit is tracked and triangulated to a single 3D position. LiDAR-derived canopy masks assign each fruit to its tree. Single-view, dual-view and multi-view counting were compared on 522 trees (71,609 mangoes near Bundaberg, Qld), with 16 trees counted by hand on the tree and after harvest.
- *Key result:* only the multi-view method was accurate without a field calibration factor: per-tree error 1.36% (abstract).
- *Relevance:* This is the closest published analogue of our pipeline: known camera poses, per-image detection and triangulation give one 3D point per fruit. Our rail supplies the trajectory for free (50 mm steps), replacing GPS/INS. The canopy-mask step corresponds to our depth gate, which keeps only detections near the fence plane (~0.9 m) and rejects the foliage layer (~0.3 m).
- *Caveat (reader-inferred):* the 1.36% figure rests on 16 validation trees and an expensive sensor suite.

**[2]** X. Liu *et al.*, "Robust fruit counting: Combining deep learning, tracking, and structure from motion," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Madrid, Spain, Oct. 2018, pp. 1045–1052, doi: 10.1109/IROS.2018.8594239. `V3` · full text (arXiv:1804.00307)
- *Summary:* An FCN segments fruit pixels. Fruit are tracked frame to frame with Hungarian assignment, using a cost from a Kalman-filter-corrected KLT tracker. Incremental structure from motion (SfM) then recovers the relative 3D position and size of every track. That 3D estimate corrects the count in three ways: tracks that coincide in 3D are merged as double counts, implausibly large objects (e.g. signs) are rejected, and fruit on rows behind are removed with a depth threshold. Oranges were imaged by day and apples at night.
- *Key result (Fig. 7):* oranges went from 4,049 counted vs 3,456 ground truth (mean error 17.2%, SD 26.3%) to 3,449 vs 3,456 after the 3D correction (-0.2%, SD 7.8%). Apples went from 8.5% (SD 4.7%) to 3.3% (SD 4.1%).
- *Relevance:* All three 3D sanity checks transfer directly to our wall: merge detections that land on the same 3D point, reject by physical size (cherry ≈ 20–25 mm), and reject by depth (fence vs foliage vs background).
- *Caveat (author-acknowledged, Sec. V):* ground truth is human counts of *visible* fruit, so fully hidden fruit are outside the evaluation.

**[3]** X. Liu *et al.*, "Monocular camera based fruit counting and mapping with semantic data association," *IEEE Robot. Autom. Lett.*, vol. 4, no. 3, pp. 2296–2303, Jul. 2019, doi: 10.1109/LRA.2019.2901987. `V3` · full text (arXiv:1811.01417)
- *Summary:* A camera-only pipeline. Faster R-CNN detects fruit and trunks, a Kalman filter fuses the detections with optical flow, and a "fruit-as-feature" semantic SfM turns 2D tracks into 3D landmarks. It names three sources of double counting: (1) the same fruit in consecutive frames, (2) re-tracking a fruit after its track was lost, and (3) the same fruit seen from both sides of the row. These are resolved by matching landmark re-projections and by trunk-centroid depth voting. It was benchmarked against the sensor suite of [1] on 18 field-counted mango trees averaging 175 fruit per tree.
- *Key result (Figs. 9–10):* per-tree count error was 27.8 ± 29.6 fruit for the monocular pipeline vs 19.8 ± 22.9 for the LiDAR + GPS multi-view system. Regression slopes were 0.86 vs 0.97 and R² 0.78 vs 0.88. Dual-view counting undercounted severely.
- *Relevance:* This shows a phone-class monocular camera plus SfM approaching a LiDAR/RTK system, which is our situation. Its three-way taxonomy is a ready checklist for our de-duplication module. A single-sided rail removes source (3); sources (1) and (2) remain.
- *Caveat (author-acknowledged, Sec. V):* both systems undercount high-yield trees because fully occluded fruit are never seen.

**[4]** J. Gené-Mola *et al.*, "Fruit detection and 3D location using instance segmentation neural networks and structure-from-motion photogrammetry," *Comput. Electron. Agric.*, vol. 169, Art. no. 105165, Feb. 2020, doi: 10.1016/j.compag.2019.105165. `V3` · abstract
- *Summary:* Mask R-CNN segments apples in each image. SfM photogrammetry builds a point cloud of 11 Fuji trees, the 2D detections are projected onto the 3D points, and an SVM removes false positives. The result is one 3D location per apple (1,455 apples). The data are public as Fuji-SfM [15].
- *Key result:* F1 rose from 0.816 (2D detection) to 0.881 (3D detection and location) relative to all fruit on the trees (abstract).
- *Relevance:* This is direct evidence that voting across views in 3D raises both recall and precision, which is the core claim behind our de-duplicated count. Fuji-SfM, with its 3D apple ground truth, can test our 3D clustering and matching code before our own captures exist.
- *Caveat (author-acknowledged, abstract):* SfM processing time is high, so the method is not real-time.

**[5]** N. Häni, P. Roy, and V. Isler, "A comparative study of fruit detection and counting methods for yield mapping in apple orchards," *J. Field Robot.*, vol. 37, no. 2, pp. 263–282, 2020, doi: 10.1002/rob.21902. `V3` · full text (arXiv:1810.09499)
- *Summary:* A modular yield system. Per-frame detection uses one of three methods (semi-supervised colour GMM, U-Net, Faster R-CNN). Cluster counting uses either a GMM or a ResNet-50 that classifies each cluster patch as holding 0–6 apples. Fruit are then tracked across frames, each row side is reconstructed in 3D, and the two sides are merged to remove double counts. This was the first head-to-head comparison on shared datasets, with detection scored across the whole IoU range [0.01, 0.99].
- *Key result (Table 4):* summing the two sides independently gave 101.9–150% (GMM) and 103.9–147.8% (ResNet-50) of harvested counts. Merging them in 3D gave 91.98–94.81% and 95.56–97.83%, i.e. 2.17–4.44% error against harvest. Patch counting accuracy was 88.5–95.1% for ResNet-50 vs 76.1–88.0% for the GMM (Table 3).
- *Relevance:* This is the clearest quantitative case that unfused views inflate counts by up to ~50%. Counting a cluster as a class (0–6 fruit) maps neatly onto our cherry twins.
- *Caveat (reader-inferred):* the strongest detector here was colour-clustering, which cannot work for green-on-green fruit.

**[6]** N. Häni, P. Roy, and V. Isler, "MinneApple: A benchmark dataset for apple detection and segmentation," *IEEE Robot. Autom. Lett.*, vol. 5, no. 2, pp. 852–858, Apr. 2020, doi: 10.1109/LRA.2020.2965061. `V3` · full text (arXiv:1909.06441)
- *Summary:* A public benchmark with 1,000 orchard images (670 training images at 1280 × 720) and more than 41,000 polygon-annotated apples, up to 120 per image and mostly small. It adds 13,000 counting patches plus 4,500 negatives, and yield ground truth for three rows. Test labels are held on a CodaLab server, and metrics follow COCO (AP@[0.5:0.95], AP@0.5, AP-small).
- *Key result (Tables II, IV):* Faster R-CNN scored AP@[0.5:0.95] 0.438 and AP@0.5 0.775; Mask R-CNN scored 0.433 and 0.763. The patch-counting CNN reached 88.5–95.1%, and was most accurate on the *green-apple* set (95.1%).
- *Relevance:* This is the de facto reporting standard and includes green fruit. It was used to evaluate SAM 3 zero-shot [69], so a MinneApple result would put our detector on a common scale.
- *Caveat (reader-inferred):* box AP rewards localisation precision that matters little for counting.

**[7]** A. Koirala, K. B. Walsh, Z. Wang, and C. McCarthy, "Deep learning for real-time fruit detection and orchard fruit load estimation: Benchmarking of 'MangoYOLO'," *Precis. Agric.*, vol. 20, no. 6, pp. 1107–1135, 2019, doi: 10.1007/s11119-019-09642-0. `V3` · abstract
- *Summary:* Six detectors (Faster R-CNN variants, YOLOv2/v3, SSD) and a purpose-built "MangoYOLO" were compared on night images of 1,515 trees. Images were taken under 720 W LED lighting from a vehicle moving at 6 km/h. The labelled tiles are public (1,300 / 130 / 300).
- *Key result:* MangoYOLO(pt) reached F1 0.968 and AP 0.983 at 8 ms per 512 × 512 tile. Orchard fruit-load estimates fell within 4.6–15.2% of packhouse counts, but only after a per-orchard correction factor derived from hand-harvested trees (abstract).
- *Relevance:* Near-perfect *detection* still needs an occlusion correction for *counting*, so the bottleneck is visibility. That is the motivation for our synthetic-aperture step. Controlled night lighting is a recurring way to suppress illumination variation, and our indoor rig can do the same.

**[8]** Z. Wang, K. Walsh, and A. Koirala, "Mango fruit load estimation using a video based MangoYOLO—Kalman filter—Hungarian algorithm method," *Sensors*, vol. 19, no. 12, Art. no. 2742, Jun. 2019, doi: 10.3390/s19122742. `V3` · abstract
- *Summary:* MangoYOLO detects fruit in 10 fps video taken from a platform at 5 km/h. A Kalman filter predicts each fruit's next position, Hungarian assignment links fruit between frames (allowing multiple-to-one matches), and a "borrow" step takes the velocity of neighbouring fruit when a track has no motion model of its own. Together these avoid recounting fruit that are briefly hidden.
- *Key result:* on one video (192 fruit), the method made 9.9% double counts and missed 7.3%, a net ≈2.6% overcount. On a 21-tree row with 3,286 harvested fruit, video tracking recorded 62% of the harvest (bias-corrected RMSE 18.0 fruit/tree) against 40% for dual-view imaging (RMSE 21.7) (abstract).
- *Relevance:* Double counts and misses partly cancel in a net count, so we must report them separately. Many views saw about 1.5× more fruit than two views. Our 12–13 rail views sit between these regimes.

**[9]** S. Bargoti and J. Underwood, "Deep fruit detection in orchards," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, Singapore, May 2017, pp. 3626–3633, doi: 10.1109/ICRA.2017.7989417. `V3` · abstract
- *Summary:* Faster R-CNN was applied to mangoes, almonds and apples, with ablations on training-set size, data augmentation and transfer between orchards. A tiling scheme handles images holding 100–1,000 fruit.
- *Key result:* F1 above 0.9 for apples and mangoes. Augmentation halved the number of training images needed, while transfer between orchards gave negligible gain (abstract).
- *Relevance:* Supervised detectors need orchard-specific labels, a cost our colour-free and zero-shot route avoids. Tiling matters for small cherries in 12 MP iPhone frames.

**[10]** I. Sa, Z. Ge, F. Dayoub, B. Upcroft, T. Perez, and C. McCool, "DeepFruits: A fruit detection system using deep neural networks," *Sensors*, vol. 16, no. 8, Art. no. 1222, Aug. 2016, doi: 10.3390/s16081222. `V3` · abstract
- *Summary:* Faster R-CNN was adapted by transfer learning to RGB and near-infrared (NIR) images, comparing early and late fusion of the two. It was then retrained for seven fruits using bounding-box labels only.
- *Key result:* sweet-pepper F1 rose from 0.807 to 0.838 with multi-modal fusion. Annotating and training each new fruit took about 4 h (abstract).
- *Relevance:* NIR is a non-colour cue for green fruit. Box annotation being about 10× faster than pixel masks sets the labelling budget for our evaluation set.

**[11]** T. T. Santos, L. L. de Souza, A. A. dos Santos, and S. Avila, "Grape detection, segmentation, and tracking using deep neural networks and three-dimensional association," *Comput. Electron. Agric.*, vol. 170, Art. no. 105247, Mar. 2020, doi: 10.1016/j.compag.2020.105247. `V3` · abstract
- *Summary:* CNNs detect and instance-segment grape clusters, which vary widely in colour, shape and compactness. The clusters are then tracked along row videos using three-dimensional association. The paper releases the WGISD dataset (300 annotated images) and an annotation methodology for complex objects.
- *Key result:* instance segmentation F1 up to 0.91 on 408 field clusters (abstract).
- *Relevance:* WGISD spans varieties that differ widely in colour, which makes it a natural external benchmark for our green-grape claim (check the exact variety mix before relying on it). Its 3D-association counting follows the same pattern as ours.

**[12]** Y. Song, C. A. Glasbey, G. W. Horgan, G. Polder, J. A. Dieleman, and G. W. A. M. van der Heijden, "Automatic fruit recognition and counting from multiple images," *Biosyst. Eng.*, vol. 118, pp. 203–215, Feb. 2014, doi: 10.1016/j.biosystemseng.2013.12.008. `V3` · abstract (repository record)
- *Summary:* Pepper fruit are counted in dense greenhouse canopies where fruit resemble the foliage. A bag-of-words detector runs on each image. A statistical clustering method then aggregates repeated, incomplete sightings of the same fruit across views. The calibration and validation sets each hold more than 28,000 images of over 1,000 plants.
- *Key result:* image analysis correlated with manual counts at 94.6%, while the full multi-image method reached 74.2% without linear adjustment (abstract).
- *Relevance:* This was an early large-scale demonstration that merging views is where count accuracy is won or lost. Its probabilistic clustering of repeated sightings is an alternative to a hard 3D distance threshold.

**[13]** J. Villacrés, M. Viscaino, J. Delpiano, S. Vougioukas, and F. Auat Cheein, "Apple orchard production estimation using deep learning strategies: A comparison of tracking-by-detection algorithms," *Comput. Electron. Agric.*, vol. 204, Art. no. 107513, Jan. 2023, doi: 10.1016/j.compag.2022.107513. `V3` · abstract (repository record)
- *Summary:* Five trackers (Kalman filter, KCF, SORT, multiple-hypothesis tracking (MHT), DeepSORT) were compared for counting apples in orchard videos, using two new datasets in multiple-object-tracking (MOT) format.
- *Key result:* with perfect detections, MHT reached a MOTA of 97.0% and DeepSORT 93.0%. With real detectors, DeepSORT's mean counting error was 20.07% (YOLOv5) and 31.52% (Faster R-CNN), and the Kalman filter's 20.5% and 31.9% (abstract).
- *Relevance (reader-inferred):* near-perfect tracking on perfect detections, combined with 20–32% count error on real ones, suggests the detector dominates count error. So we should evaluate our de-duplication with oracle (ground-truth) detections as well as with our real detector.

**[14]** L. Meyer, A. Gilson, U. Schmid, and M. Stamminger, "FruitNeRF: A unified neural radiance field based fruit counting framework," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Abu Dhabi, UAE, Oct. 2024, pp. 1–8, doi: 10.1109/IROS58592.2024.10802065. `V3` · full text (arXiv:2408.06190)
- *Summary:* The inputs are posed images plus fruit-agnostic masks from Grounded-SAM: Grounding DINO boxes prompt SAM masks. These train a semantic neural radiance field (NeRF). Sampling its "fruit field" gives a fruit-only point cloud, which is clustered in two stages: DBSCAN first, then agglomerative clustering matched to a template fruit to split clusters of up to six fruit. Because counting happens in 3D, double counting is avoided by construction.
- *Key result (Table I, Figs. 7–8):* on synthetic trees (300 views), mean F1 was 0.95 with ground-truth masks and 0.88 with Grounded-SAM masks. The worst cases were plum (SAM recall 0.403, F1 0.575) and mango (recall 0.694), both attributed to occlusion. At least 20 images were needed, and 30–40 at 512–1,024 px. On real trees, detection rates were ≈89% (trees 1–2) and ≈82% (tree 3, heavier occlusion), and F1 on Fuji-SfM was 0.79 (vs 0.88 in [4]).
- *Relevance:* This is the state of the art in colour-agnostic 3D counting. Our 12–13 collinear views over 0.6 m fall below its working regime, which argues for a lighter plane-sweep or synthetic-aperture-plus-3D-clustering design. Open-vocabulary prompting was brittle: mangoes had to be prompted as "apple", and plums as "apple & plum" (Table I).
- *Caveat (author-acknowledged, Sec. V):* long training, clustering hyperparameters tuned per fruit, and a fixed nominal fruit size.

*Also verified (supporting, brief):*
- **[15]** J. Gené-Mola *et al.*, "Fuji-SfM dataset: A collection of annotated images and point clouds for Fuji apple detection and location using structure-from-motion photogrammetry," *Data Brief*, vol. 30, Art. no. 105591, Jun. 2020, doi: 10.1016/j.dib.2020.105591. `V3`. Contents: 288 annotated training images, a 582-image motion sequence, and a point cloud of 11 trees with 3D ground-truth positions for 1,455 apples (abstract). This is the only public benchmark with SfM-based 3D fruit positions that we found.
- **[16]** S. Bargoti and J. P. Underwood, "Image segmentation for fruit detection and yield estimation in apple orchards," *J. Field Robot.*, vol. 34, no. 6, pp. 1039–1060, 2017, doi: 10.1002/rob.21699. `V3`. Pixel segmentation (MLP/CNN) is followed by watershed and circular Hough transform steps that split it into individual fruit. Results: pixel F1 0.791, detection F1 0.861 (watershed), and r² = 0.826 against post-harvest counts (abstract).
- **[17]** M. Halstead, C. McCool, S. Denman, T. Perez, and C. Fookes, "Fruit quantity and ripeness estimation using a robotic vision system," *IEEE Robot. Autom. Lett.*, vol. 3, no. 4, pp. 2995–3002, Oct. 2018, doi: 10.1109/LRA.2018.2849514. `V3`. Vision-only tracking-via-detection counts sweet peppers to within 4.1% of the visual ground truth (abstract).
- **[18]** P. Roy, A. Kislay, P. A. Plonski, J. Luby, and V. Isler, "Vision-based preharvest yield mapping for apple orchards," *Comput. Electron. Agric.*, vol. 164, Art. no. 104897, Sept. 2019, doi: 10.1016/j.compag.2019.104897. `V3`. Counts are merged across views using camera motion, and across row sides. Detection F1 0.95–0.97, counting accuracy 89–98%, yield accuracy 91.98–94.81% (abstract).
- **[19]** S. W. Chen *et al.*, "Counting apples and oranges with deep learning: A data-driven approach," *IEEE Robot. Autom. Lett.*, vol. 2, no. 2, pp. 781–788, Apr. 2017, doi: 10.1109/LRA.2017.2651944. `V3`. An FCN blob detector is followed by a CNN that regresses the count, applied to oranges by day and green apples at night. This is the count-by-regression alternative to counting individual detections (the abstract gives no figures).

### A2. Occlusion, visibility versus number of views, active perception, and synthetic aperture

**[20]** J. Hemming, J. Ruizendaal, J. W. Hofstee, and E. J. van Henten, "Fruit detectability analysis for different camera positions in sweet-pepper," *Sensors*, vol. 14, no. 4, pp. 6032–6044, Mar. 2014, doi: 10.3390/s140406032. `V3` · abstract
- *Summary:* A recording rig placed a camera at different azimuth and zenith angles and positions along a sweet-pepper row. Fruit visibility was scored by hand for 14 camera positions, with a fruit counted as detectable if no more than 50% of it was occluded.
- *Key result:* no single position exceeded 69% fruit detectability. The best single views were frontal or looked 60° upward. Combining five favoured positions raised detectability to 90% (abstract).
- *Relevance:* This is the canonical "visibility versus number of views" curve, with strongly diminishing returns. We should reproduce the same curve for our rail by counting with k = 1…13 views.

**[21]** C. W. Bac, E. J. van Henten, J. Hemming, and Y. Edan, "Harvesting robots for high-value crops: State-of-the-art review and challenges ahead," *J. Field Robot.*, vol. 31, no. 6, pp. 888–911, 2014, doi: 10.1002/rob.21525. `V3` · abstract
- *Summary:* The review analyses 50 harvesting-robot projects across three decades. It covers the crop environment, performance indicators, hardware and algorithms.
- *Key result:* on average, localisation success was 85%, detachment 75% and harvest success 66%, with 5% fruit damage, 45% peduncle damage and a 33 s cycle time. Performance had not improved in three decades, and none of the robots was commercialised (abstract).
- *Relevance:* It identifies occlusion as the dominant limit on localisation. One of its four challenges is "defining requirements and measuring performance", which supports our emphasis on a standard evaluation protocol.

**[22]** P. Kurtser and Y. Edan, "Statistical models for fruit detectability: Spatial and temporal analyses of sweet peppers," *Biosyst. Eng.*, vol. 171, pp. 272–289, Jul. 2018, doi: 10.1016/j.biosystemseng.2018.04.017. `V3` · abstract
- *Summary:* Poisson-regression models relate fruit detectability to controllable variables (viewpoint, distance, sensor tilt) and measurable ones. The data are 1,312 images taken from 10–14 viewpoints for each of 56 greenhouse scenes, using an eye-in-hand arm with an illumination rig.
- *Key result:* detectability depends strongly on acquisition distance and sensor tilt. At least 12 training scenes were needed to identify the statistically significant spatial variables (abstract).
- *Relevance:* This is a ready statistical template for modelling our detectability as a function of rail position, aperture width and foliage density.

**[23]** C. Lehnert, D. Tsai, A. Eriksson, and C. McCool, "3D Move to See: Multi-perspective visual servoing towards the next best view within unstructured and occluded environments," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Macau, China, Nov. 2019, pp. 3890–3897, doi: 10.1109/IROS40897.2019.8967918. `V3` · abstract
- *Summary:* A 3D camera array on a manipulator captures several views at once. Segmentation plus a per-view objective function estimate, in a single shot, the gradient towards a less occluded viewpoint, and the robot servos along it.
- *Key result:* mean increase in target size was 29.3%, against 9.17% for a single RGB-D baseline, about three times larger (abstract).
- *Relevance:* This is the closest robotics precedent for using a *camera array* against foliage. It steers with the array, whereas our synthetic aperture integrates over the array.

**[24]** T. Zaenker, C. Smitt, C. McCool, and M. Bennewitz, "Viewpoint planning for fruit size and position estimation," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Prague, Czech Republic, Sept. 2021, pp. 3271–3277, doi: 10.1109/IROS51168.2021.9636701. `V3` · abstract
- *Summary:* An octree map with fruit regions of interest (ROIs) drives the sampling of RGB-D viewpoints. The planner switches between ROI-targeted and exploration sampling, then clusters ROI voxels to estimate fruit position and size. It was tested in simulation and on an arm mounted on a pipe-rail trolley in a capsicum glasshouse.
- *Key result:* it discovered more ROI cells than sampling that ignores ROIs (abstract; no figures stated).
- *Relevance:* Active viewpoint selection is the alternative to a dense fixed aperture. With a fixed rail, it reframes as "which subset of the 13 views suffices?".

**[25]** A. Koirala, K. B. Walsh, and Z. Wang, "Attempting to estimate the unseen—Correction for occluded fruit in tree fruit load estimation by machine vision with deep learning," *Agronomy*, vol. 11, no. 2, Art. no. 347, Feb. 2021, doi: 10.3390/agronomy11020347. `V3` · abstract
- *Summary:* Per-tree correction factors for hidden fruit were learned from image features, such as the share of partly occluded fruit, using random forests and multilayer perceptrons (MLPs). Separately, CNNs were trained directly on harvest counts.
- *Key result:* the random forest reached R² 0.98 (n = 98 trees) against 0.68 for the manual reference method. Whole-orchard error (880 trees) was 1.6% for the MLP vs 13.6% for the reference. However, performance on a second season was at best equivalent to the reference and generally poorer (abstract).
- *Relevance:* Learned "unseen fruit" corrections are fragile across seasons. That strengthens the case for *physically* recovering visibility, which our synthetic aperture does, and gives us a testable claim.

**[26]** J. Gené-Mola *et al.*, "Looking behind occlusions: A study on amodal segmentation for robust on-tree apple fruit size estimation," *Comput. Electron. Agric.*, vol. 209, Art. no. 107854, Jun. 2023, doi: 10.1016/j.compag.2023.107854. `V3` · abstract (repository record; the index abstract was corrupt)
- *Summary:* A single CNN predicts both modal (visible) and amodal (visible plus hidden) instance masks of apples. The amodal mask gives the pixel diameter, the modal mask plus depth gives range, and a pinhole model converts to millimetres. The visibility ratio (modal area over amodal area) serves as a confidence score. Training used 3,925 RGB-D images with 15,335 Fuji apples, and testing used Elstar apples.
- *Key result:* detection F1 0.86, and diameter MAE 4.5 mm (R² 0.80) *irrespective of fruit visibility* (abstract).
- *Relevance:* Amodal completion is the single-view counterpart of our multi-view de-occlusion. The visibility ratio is a natural per-fruit confidence for our counts, and could flag cherries seen only through gaps in the foliage.

**[27]** P. M. Blok, E. J. van Henten, F. K. van Evert, and G. Kootstra, "Image-based size estimation of broccoli heads under varying degrees of occlusion," *Biosyst. Eng.*, vol. 208, pp. 213–233, Aug. 2021, doi: 10.1016/j.biosystemseng.2021.06.001. `V3` · abstract
- *Summary:* An occlusion-aware R-CNN (ORCNN) segments both the visible and the amodal regions of broccoli heads under leaves. It was compared with visible-only Mask R-CNN sizing on 487 RGB-D images with systematically varied occlusion.
- *Key result:* mean sizing error was 6.4 mm for ORCNN vs 10.7 mm for Mask R-CNN, and significantly lower on 161 heads with 50–90% occlusion (abstract).
- *Relevance:* This is a green object under green leaves. Shape priors survive heavy occlusion, which is evidence for shape-based reasoning when colour contrast is absent.

**[28]** V. Vaish, M. Levoy, R. Szeliski, C. L. Zitnick, and S. B. Kang, "Reconstructing occluded surfaces using synthetic apertures: Stereo, focus and robust measures," in *Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. (CVPR)*, New York, NY, USA, 2006, vol. 2, pp. 2331–2338, doi: 10.1109/CVPR.2006.244. `V3` · abstract
- *Summary:* On camera-array light fields, classical SSD stereo is compared with shape from synthetic-aperture focus. SSD assumes every camera sees the surface. Two occlusion-robust multi-view costs, based on colour medians and entropy, are proposed and evaluated against the amount of occlusion.
- *Key result:* robust (median and entropy) costs held up under occlusion better than SSD-type costs (abstract; no figures stated).
- *Relevance:* This is the bridge from synthetic aperture to detection. Our per-depth focus measure should use robust statistics (median or trimmed variance across views) so the fruit surface "wins" over leaves that cover it in only some views.

**[29]** I. Kurmi, D. C. Schedl, and O. Bimber, "A statistical view on synthetic aperture imaging for occlusion removal," *IEEE Sens. J.*, vol. 19, no. 20, pp. 9374–9383, Oct. 2019, doi: 10.1109/JSEN.2019.2922731. `V3` · abstract (arXiv:1906.06600)
- *Summary:* A statistical model of occlusion removal by synthetic apertures, applied to drone-based airborne optical sectioning through vegetation. It shows that both the aperture size and the number of samples have practical limits, and derives guidance for designing sampling patterns.
- *Key result:* beyond those practical limits, a wider aperture or more samples add little visibility (abstract).
- *Relevance:* This is the theory for choosing our 0.6 m aperture and 12–13 samples. It predicts residual occlusion as a function of occluder density, a curve we can measure by varying the pipe-cleaner density.

**[30]** R. J. A. A. Nathan and O. Bimber, "Synthetic aperture anomaly imaging for through-foliage target detection," *Remote Sens.*, vol. 15, no. 18, Art. no. 4369, Sept. 2023, doi: 10.3390/rs15184369. `V3` · abstract
- *Summary:* Drone imagery is used to detect targets through foliage. Anomaly detection run on synthetic-aperture integral images is compared with running it on each single image and then integrating the results.
- *Key result:* integrating per-image detections was more effective, adding on average 20% to target visibility and precision (abstract).
- *Relevance:* This is a decisive design question for us. Should we detect in each raw view and then vote in 3D, or refocus first and then detect? Their evidence favours detect-then-integrate, or doing both, and our ablation should test exactly this.

*Also verified (supporting, brief):*
- **[31]** P. Kurtser and Y. Edan, "The use of dynamic sensing strategies to improve detection for a pepper harvesting robot," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Madrid, Spain, Oct. 2018, pp. 8286–8293, doi: 10.1109/IROS.2018.8593746. `V2`. The robot decides whether an extra viewpoint is worth taking, which improved detection by up to 19% on 96 images of 30 peppers (abstract).
- **[32]** R. Menon, T. Zaenker, N. Dengler, and M. Bennewitz, "NBV-SC: Next best view planning based on shape completion for fruit mapping and reconstruction," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Detroit, MI, USA, Oct. 2023, pp. 4197–4203, doi: 10.1109/IROS55552.2023.10341855. `V3`. Predicted fruit shapes direct new views at unobserved fruit surfaces. This gave better size estimates and reconstructions with less planning time (abstract; no figures stated).
- **[33]** O. Mirbod, D. Choi, P. H. Heinemann, R. P. Marini, and L. He, "On-tree apple fruit size estimation using stereo vision with deep learning-based occlusion handling," *Biosyst. Eng.*, vol. 226, pp. 27–42, Feb. 2023, doi: 10.1016/j.biosystemseng.2022.12.008. `V3`. Stereo imaging with an LED strobe, plus extrapolation of occluded fruit regions, gave diameter MAE of 1.1–4.2 mm, a mean error of 4.8% (abstract).
- **[34]** A. R. Jiménez, R. Ceres, and J. L. Pons, "A survey of computer vision methods for locating fruit on trees," *Trans. ASAE*, vol. 43, no. 6, pp. 1911–1920, 2000, doi: 10.13031/2013.3096. `V3`. Shape-based methods were more independent of hue than colour-based ones, and range-image methods found fruit of any colour without false alarms. Total occlusion limits the fruit that can be found to 40–100%, depending on conditions (abstract).

### A3. Colour-independent and green-fruit detection (shape, radial symmetry, specular highlights, texture, geometry)

**[35]** A. D. Whittaker, G. E. Miles, O. R. Mitchell, and L. D. Gaultney, "Fruit location in a partially occluded image," *Trans. ASAE*, vol. 30, no. 3, pp. 591–596, 1987, doi: 10.13031/2013.30444. `V3` · abstract. (CrossRef rejected the DOI ending .30443 on title mismatch; .30444 is correct.)
- *Summary:* Tomatoes are located in digital images with a modified circular Hough transform (CHT) applied to edge information. Detection uses shape, not colour. It worked in scenes with substantial background noise and partially hidden fruit.
- *Key result:* tomatoes were located by shape alone, despite noise and partial occlusion (abstract; no figures stated).
- *Relevance:* This is the seminal colour-free baseline for fruit that match the foliage colour, and CHT should be the classical baseline in our detector comparison.

**[36]** A. R. Jiménez, A. K. Jain, R. Ceres, and J. L. Pons, "Automatic fruit recognition: A survey and new results using range/attenuation images," *Pattern Recognit.*, vol. 32, no. 10, pp. 1719–1736, Oct. 1999, doi: 10.1016/S0031-3203(98)00170-8. `V3` · abstract (index record)
- *Summary:* A laser range-finder records co-registered range and attenuation (reflectance) images. A sensor model plus shape analysis recognises spherical fruit despite shadows, bright patches, occlusion and overlap, and returns each fruit's 3D position, radius and reflectance. Results are on artificial orange-tree scenes.
- *Key result:* spherical fruit detected with 3D position and radius estimated. Detection rates were not retrieved, but the companion survey [34] reports that range-based methods found fruit of any colour without false alarms.
- *Relevance:* This is the original colour-free *depth* detector. A depth map from plane-sweeping our rail images plays the role of the range image.

**[37]** G. Loy and A. Zelinsky, "Fast radial symmetry for detecting points of interest," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 25, no. 8, pp. 959–973, Aug. 2003, doi: 10.1109/TPAMI.2003.1217601. `V3` · abstract
- *Summary:* Each pixel's gradient votes, through orientation and magnitude projection images, for centres of radial symmetry at a set of radii. The result is a fast interest-point transform, demonstrated as a face-feature and generic region-of-interest detector.
- *Key result:* equal or better than contemporary detectors at low cost, running above 60 fps on a Pentium III (abstract).
- *Relevance:* This transform is the candidate stage of Nuske's green-berry detector [39]. It needs only intensity gradients. On synthetic-aperture refocused images it should peak at in-focus round fruit, while defocused foliage blurs out.

**[38]** T. Lindeberg, "Feature detection with automatic scale selection," *Int. J. Comput. Vis.*, vol. 30, no. 2, pp. 79–116, Nov. 1998, doi: 10.1023/A:1008045108935. `V3` · read scope: summary of the standard method (no abstract in the indexes)
- *Summary:* The paper introduces scale-normalised derivative operators and selects a feature's characteristic scale at the extremum over scale of the normalised response. The best-known instances are blob detection with the scale-normalised Laplacian of Gaussian (LoG) and the determinant of the Hessian (DoH).
- *Key result:* methodological, with no single figure.
- *Relevance:* This is the maths behind a Hessian/LoG blob stage. For a disc of radius r, the normalised LoG peaks near σ ≈ r/√2, so fruit size falls out of detection. The Hessian eigenvalues separate blobs (both eigenvalues large, same sign) from ridges (one eigenvalue near zero). That ridge test targets the teammate's stem and pipe-cleaner false positives directly.

**[39]** S. Nuske, S. Achar, T. Bates, S. Narasimhan, and S. Singh, "Yield estimation in vineyards by visual grape detection," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, San Francisco, CA, USA, Sept. 2011, pp. 2352–2358, doi: 10.1109/IROS.2011.6095069. `V3` · full text (CMU RI copy). Note the title reads "**by**" visual grape detection, not "via".
- *Summary:* The detector has three stages. (i) Candidate berry centres come from the Loy–Zelinsky radial symmetry transform [37] over a range of radii (Sec. III-A). (ii) Each candidate patch is classified by k-nearest neighbours on a 34-dimensional L\*a\*b colour plus Gabor texture descriptor (Sec. III-B). (iii) Isolated detections are removed by grouping neighbouring berries into clusters. It was demonstrated on green berries against green leaves across 224 vines.
- *Key result (Table I):* overall precision 98.0% and recall 63.7%, with recall ranging from 53.7% (Riesling) to 74.9% (Gewürztraminer). About 60% of missed berries were lost at the radial-symmetry stage, 30% at classification and 10% at clustering. Row yield was predicted to within 9.8% of crop weight (abstract).
- *Relevance:* This is the closest precedent for our green-fruit generalisation. Candidate generation, not classification, limits recall, so our candidate stage should over-generate and leave rejection to geometry and depth. Note that the classifier still used colour.

**[40]** S. Nuske, K. Wilshusen, S. Achar, L. Yoder, S. Narasimhan, and S. Singh, "Automated visual yield estimation in vineyards," *J. Field Robot.*, vol. 31, no. 5, pp. 837–860, 2014, doi: 10.1002/rob.21541. `V3` · abstract
- *Summary:* Vehicle-mounted cameras with their own illumination image the vines. Texture, colour and shape cues are combined into a classifier that detects berries even when they are the same colour as the leaves. The system was run over four seasons in wine and table-grape vineyards, with optimised mappings from image counts to yield.
- *Key result:* up to 75% of spatial yield variance captured, with an average error of 3–11% of total yield (abstract).
- *Relevance:* This is multi-season evidence that shape and texture detection scales. It also shows that the count-to-yield calibration dominates final error, which our exact-count task avoids.

**[41]** Z. S. Pothen and S. Nuske, "Texture-based fruit detection via images using the smooth patterns on the fruit," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, Stockholm, Sweden, May 2016, pp. 5171–5176, doi: 10.1109/ICRA.2016.7487722. `V3` · abstract. Not to be confused with Chaivivatrakul & Dailey, "Texture-based fruit detection," *Precis. Agric.* 2014, whose content we could not retrieve (Appendix B).
- *Summary:* A keypoint detector for round fruit. A candidate seed point must show monotonically decreasing intensity and a consistent gradient orientation outward from it, which is the smooth shading of a lit sphere. Candidates are then classified with a modified HOG plus a pairwise-intensity texture descriptor and a random forest. The detector is scale-invariant and robust to partial occlusion.
- *Key result:* F1 0.82 on grapes and 0.80 on apples, better than prior methods (abstract).
- *Relevance:* This is arguably the most transferable colour-free cue for us. A glossy sphere under a light gives a highlight with monotonic fall-off, while leaves, pipe-cleaner strands and stems do not. It is intensity-only, so it applies unchanged to limes and green grapes.

**[42]** D. Font *et al.*, "Counting red grapes in vineyards by detecting specular spherical reflection peaks in RGB images obtained at night with artificial illumination," *Comput. Electron. Agric.*, vol. 108, pp. 105–111, Oct. 2014, doi: 10.1016/j.compag.2014.07.006. `V3` · abstract (repository record)
- *Summary:* Grapes are imaged at night under artificial light and counted by detecting the specular reflection peak on each berry's spherical surface.
- *Key result:* mean counting error 14% with 7% false positives. About 33% of berries in clusters were extremely occluded (more than 75% hidden) (abstract).
- *Relevance:* This is specular-peak counting of small glossy spheres, our cherries exactly, and our rig can control the lighting. The cue is colour-independent even though these grapes were red. The occluded third shows the visibility ceiling for clustered fruit.

**[43]** Q. Wang, S. Nuske, M. Bergerman, and S. Singh, "Automated crop yield estimation for apple orchards," in *Experimental Robotics* (Springer Tracts in Advanced Robotics). Cham, Switzerland: Springer, 2013, pp. 745–758, doi: 10.1007/978-3-319-00065-7_50. `V3` · full text (ISER 2012 manuscript, CMU RI)
- *Summary:* A night-time stereo rig with ring flashes rides on an autonomous vehicle along both sides of each row. Red apples are found with HSV thresholds. Green apples are found with hue and saturation plus a *specular-reflection detector* (Sec. 3.2). Local intensity maxima are tested along four 21-pixel lines (0°, 45°, 90°, 135°) for monotonic fall-off, and a roundness score R ≥ 4 of 8 half-lines is accepted so that partly occluded apples still pass. Stereo triangulation plus vehicle pose give global coordinates. Detections from the two sides within **0.16 m (≈2 apple diameters)** are merged to avoid double counting.
- *Key result (Sec. 5):* the red block's error was -3.2% (thinned rows, uncalibrated). However, unthinned rows were undercounted by 41.3%, because the software splits a cluster into at most two apples. Raw green-block counts were -29.8% (SD 8.1%), and a calibration factor of 1.4 fitted on 10 hand-counted sections reduced the block error to 1.2%.
- *Relevance:* It gives (i) a ready recipe for a colour-free "roundness of highlight" test, (ii) a precedent for a 3D merge and matching radius of about two fruit diameters, and (iii) a warning that headline accuracies can hide calibration factors and cluster-splitting limits. Point (iii) matters for our cherry twins.

**[44]** F. Kurtulmus, W. S. Lee, and A. Vardar, "Green citrus detection using 'eigenfruit', color and circular Gabor texture features under natural outdoor conditions," *Comput. Electron. Agric.*, vol. 78, no. 2, pp. 140–149, Sept. 2011, doi: 10.1016/j.compag.2011.07.001. `V3` · abstract (university repository)
- *Summary:* Immature green citrus are detected in natural canopies by sliding-window classifiers. These use an "eigenfruit" (eigenface-style) model on intensity and saturation, colour, and circular Gabor texture, combined by majority vote. Blob analysis then merges multiple detections of the same fruit. There were 96 images (32 training, 64 validation).
- *Key result:* 75.3% of validation fruit were detected (abstract).
- *Relevance:* This is a single-view reference point for green fruit against green leaves (about 75% recall), which our multi-view method should beat.

**[45]** R. Linker, O. Cohen, and A. Naor, "Determination of the number of green apples in RGB images recorded in orchards," *Comput. Electron. Agric.*, vol. 81, pp. 45–57, Feb. 2012, doi: 10.1016/j.compag.2011.11.007. `V3` · abstract (university repository)
- *Summary:* Pixels likely to be apple are flagged by colour and smoothness and grown into seed areas. The seed-area contours are split into arcs, and the arcs are combined into circles that are checked against a simple apple model.
- *Key result:* under automatic exposure, more than 85% of visible apples were detected, but direct sun and saturation caused many false positives. In underexposed, diffuse light near sunset, detection was about 95% with under 5% false positives (abstract).
- *Relevance:* Arc-based circle fitting tolerates partial occlusion. The swing in performance with lighting shows that controlling the light matters more than the classifier.

**[46]** S. Sengupta and W. S. Lee, "Identification and determination of the number of immature green citrus fruit in a canopy under different ambient light conditions," *Biosyst. Eng.*, vol. 117, pp. 51–61, Jan. 2014, doi: 10.1016/j.biosystemseng.2013.07.007. `V3` · abstract (library record)
- *Summary:* A high-recall circular Hough transform, with its radius range taken from training images, proposes candidate fruit. Candidates are then filtered by SVM texture classification and by Canny edges with graph-based connected components and Hough *line* detection, which reject straight structures. There were 100 images (38 training, 62 validation).
- *Key result:* 80.4% of fruit were detected and counted correctly on the validation set (abstract).
- *Relevance:* This is the architecture we propose: a shape stage tuned for recall, then filters that remove line-like stems and leaf edges.

**[47]** A. Payne, K. Walsh, P. Subedi, and D. Jarvis, "Estimating mango crop yield using image analysis using fruit at 'stone hardening' stage and night time imaging," *Comput. Electron. Agric.*, vol. 100, pp. 160–167, Jan. 2014, doi: 10.1016/j.compag.2013.11.011. `V3` · abstract
- *Summary:* Immature, less-red mangoes were imaged at night under artificial light. The authors' earlier colour-segmentation algorithm failed on them, so it was changed to rely less on colour and more on texture filtering, especially *Hessian filtering*, to remove leaves, trunk and stems.
- *Key result:* on the calibration set, 78.3% of fruit were detected with a 10.6% error rate and R² 0.63 against manual counts. Validation results were mixed because foliage varied between image sets (abstract).
- *Relevance:* This is the Hessian-filter fruit-detection precedent the brief asked for: a published switch from colour to Hessian and texture cues when fruit lost their colour contrast.

**[48]** E. Barnea, R. Mairon, and O. Ben-Shahar, "Colour-agnostic shape-based 3D fruit detection for crop harvesting robots," *Biosyst. Eng.*, vol. 146, pp. 57–70, Jun. 2016, doi: 10.1016/j.biosystemseng.2016.01.013. `V3` · abstract (university page)
- *Summary:* Using RGB plus range data, fruit are detected by shape regardless of colour. The cues are 3D surface-normal features, 3D plane-reflective symmetry, and image-plane highlights from elliptic (convex) surface points. Testing was on a challenging, heavily occluded sweet-pepper dataset.
- *Key result:* the retrieved abstract gives no figures, so none are reported here.
- *Relevance:* This is the most explicitly colour-agnostic 3D detector we found, and it fuses exactly our candidate cues: geometry, symmetry and highlights. Rail-derived depth can stand in for its range sensor.

**[49]** T. T. Nguyen, K. Vandevoorde, N. Wouters, E. Kayacan, J. G. De Baerdemaeker, and W. Saeys, "Detection of red and bicoloured apples on tree with an RGB-D camera," *Biosyst. Eng.*, vol. 146, pp. 33–44, Jun. 2016, doi: 10.1016/j.biosystemseng.2016.01.007. `V3` · abstract (index record)
- *Summary:* An RGB-D camera, shielded from direct sun, captures apples on the tree. Colour and point-cloud shape features detect and locate the apples for robotic harvesting.
- *Key result:* 100% of fully visible and 82% of partially occluded apples were detected. Location error was under 10 mm per axis, and 20 apples took under 1 s (abstract).
- *Relevance:* Depth-based shape fitting gives millimetre localisation. However, this method uses colour, so it is a depth-*assisted* baseline rather than a colour-free one.

**[50]** G. Lin, Y. Tang, X. Zou, J. Xiong, and J. Li, "Guava detection and pose estimation using a low-cost RGB-D sensor in the field," *Sensors*, vol. 19, no. 2, Art. no. 428, Jan. 2019, doi: 10.3390/s19020428. `V3` · abstract. The brief's "Lin et al. guava" is this 2019 *Sensors* paper.
- *Summary:* An FCN segments fruit and branches in the RGB image. Euclidean clustering of the RGB-D point cloud separates individual fruit, and 3D line-segment detection rebuilds the branches. Each fruit's pose is estimated from its centre and the nearest branch.
- *Key result:* precision 0.983, recall 0.948, 3D pose error 23.43° ± 14.18°, and 0.565 s per fruit (abstract).
- *Relevance:* Guava is usually green on the tree (reader-inferred; the abstract does not say so). Point-cloud clustering is what splits touching fruit, and we need that step to separate cherry twins in 3D.

**[51]** C. Lehnert, I. Sa, C. McCool, B. Upcroft, and T. Perez, "Sweet pepper pose detection and grasping for automated crop harvesting," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, Stockholm, Sweden, May 2016, pp. 2428–2434, doi: 10.1109/ICRA.2016.7487394. `V3` · abstract
- *Summary:* Kinect Fusion fuses eye-in-hand RGB-D views, and colour segmentation plus clustering isolates each pepper. A superellipsoid is then fitted to it by nonlinear least squares to estimate a 6-DOF grasp pose.
- *Key result:* peppers were grasped successfully across a range of orientations, a large improvement on a naive approach that ignores orientation (abstract; no figures stated).
- *Relevance:* Fitting a parametric shape (a superellipsoid, which includes spheres) to fused multi-view geometry gives a principled "is this blob a fruit?" test. Their segmentation step, though, is colour-based.

**[52]** K. Kusumam, T. Krajník, S. Pearson, T. Duckett, and G. Cielniak, "3D-vision based detection, localization, and sizing of broccoli heads in the field," *J. Field Robot.*, vol. 34, no. 8, pp. 1505–1518, 2017, doi: 10.1002/rob.21726. `V3` · abstract
- *Summary:* Broccoli heads are detected from low-cost RGB-D data using purely 3D features: Viewpoint Feature Histograms with an SVM. A temporal filter tracks heads and builds a 3D map, and head size is also estimated. Datasets from the UK and Spain are released, and cross-country tests showed good generalisation.
- *Key result:* high precision and good cross-dataset generalisation (abstract; figures not stated there).
- *Relevance:* This is a green crop detected from geometry alone, with multi-frame filtering for de-duplication. It is the nearest geometric analogue of green fruit among green leaves.

*Also verified (supporting, brief):*
- **[53]** W. S. Qureshi, A. Payne, K. B. Walsh, R. Linker, O. Cohen, and M. N. Dailey, "Machine vision for counting fruit on mango tree canopies," *Precis. Agric.*, vol. 18, no. 2, pp. 224–244, 2017, doi: 10.1007/s11119-016-9458-5. `V3`. Texture-based dense segmentation and shape-based detection counted within 16% of actual fruit numbers (F1 > 0.68) under calibration conditions, but worse across canopy shapes and imaging conditions. The authors recommend night imaging under constant light (abstract).
- **[54]** R. Roscher, K. Herzog, A. Kunkel, A. Kicherer, R. Töpfer, and W. Förstner, "Automated image analysis framework for high-throughput determination of grapevine berry sizes using conditional random fields," *Comput. Electron. Agric.*, vol. 100, pp. 148–158, Jan. 2014, doi: 10.1016/j.compag.2013.11.008. `V3`. Circular-structure detection is followed by a one-class CRF with active learning that adapts to illumination and berry colour. Mean berry size correlated with manual measurement at 0.88 (abstract).
- **[55]** H. Gan, W. S. Lee, V. Alchanatis, R. Ehsani, and J. K. Schueller, "Immature green citrus fruit detection using color and thermal images," *Comput. Electron. Agric.*, vol. 152, pp. 117–125, Sept. 2018, doi: 10.1016/j.compag.2018.07.011. `V3`. Fusing colour and thermal images raised recall from 78.1% to 90.4% at similar precision, or precision from 86.6% to 95.5% at similar recall (abstract). Thermal is a non-colour modality for green fruit.
- **[56]** L. Fu, Y. Majeed, X. Zhang, M. Karkee, and Q. Zhang, "Faster R–CNN–based apple detection in dense-foliage fruiting-wall trees using RGB and depth features for robotic harvesting," *Biosyst. Eng.*, vol. 197, pp. 245–256, Sept. 2020, doi: 10.1016/j.biosystemseng.2020.07.007. `V3`. On planar fruiting-wall trees, removing background rows with a 1.2 m depth threshold lifted AP to 0.893 (+2.5%) (abstract). This is depth gating for a wall-like canopy, as in our scene.
- **[57]** L. Zabawa, A. Kicherer, L. Klingbeil, R. Töpfer, H. Kuhlmann, and R. Roscher, "Counting of grapevine berries in images via semantic segmentation using convolutional neural networks," *ISPRS J. Photogramm. Remote Sens.*, vol. 164, pp. 73–83, Jun. 2020, doi: 10.1016/j.isprsjprs.2020.04.002. `V3`. Green berries are detected by semantic segmentation and counted by connected components, with detection accuracy of 94.0% (VSP) and 85.6% (SMPH) (abstract).
- **[58]** A. Aquino, I. Barrio, M.-P. Diago, B. Millan, and J. Tardaguila, "vitisBerry: An Android-smartphone application to early evaluate the number of grapevine berries by means of image analysis," *Comput. Electron. Agric.*, vol. 148, pp. 19–28, May 2018, doi: 10.1016/j.compag.2018.02.021. `V3`. Smartphone berry counting on 144 images of 12 varieties gave recall 0.876–0.908 and precision 0.939–0.951, depending on the device (abstract, via repository record). This is a phone-based precedent.

### A4. Cherries

**[59]** L. Cossio-Montefinale, J. Ruiz-del-Solar, and R. Verschae, "Cherry CO dataset: A dataset for cherry detection, segmentation and maturity recognition," *IEEE Robot. Autom. Lett.*, vol. 9, no. 6, pp. 5552–5558, Jun. 2024, doi: 10.1109/LRA.2024.3393214. `V3` · abstract. The exact title is confirmed: "Cherry CO Dataset: A Dataset for Cherry Detection, Segmentation and Maturity Recognition".
- *Summary:* A high-resolution dataset of 3,006 labelled images from a cherry plantation, captured under varied conditions. Every cherry is annotated with its location, shape (mask) and classes for ripeness, health state and position. Several deep networks were benchmarked, and the dataset and evaluation software are released.
- *Key result:* among the benchmarked networks, YOLOv7 performed best (abstract). The numerical mAP values were not retrievable, so none are quoted.
- *Relevance:* This is the only public cherry dataset with instance masks and maturity labels. It can serve as a real-orchard test set for the detector, a cross-domain check against the toy wall, and its position labels may allow occlusion-stratified reporting.

**[60]** R. Gai, N. Chen, and H. Yuan, "A detection algorithm for cherry fruits based on the improved YOLO-v4 model," *Neural Comput. Appl.*, vol. 35, no. 19, pp. 13895–13906, 2023 (online 2021), doi: 10.1007/s00521-021-06029-z. `V3` · abstract (library record)
- *Summary:* YOLOv4's CSPDarknet53 backbone is densified with DenseNet-style connections, and the rectangular prior boxes are replaced by *circular* marker boxes fitted to small round cherries. It is compared against YOLOv3, YOLOv3-dense and YOLOv4 across ripeness levels.
- *Key result:* mAP was about 0.15 higher than YOLOv4 (abstract, via library record). The abstract does not make clear whether this is a fraction or percentage points.
- *Relevance:* Circular anchors are a round-fruit shape prior built into a deep detector. However, this is supervised and trained on red cherries, so it is the baseline our colour-free requirement has to beat on green fruit.

**[61]** J. F. Villacrés and F. Auat Cheein, "Detection and characterization of cherries: A deep learning usability case study in Chile," *Agronomy*, vol. 10, no. 6, Art. no. 835, Jun. 2020, doi: 10.3390/agronomy10060835. `V3` · abstract
- *Summary:* A portable vision system with a deep-network detector was field-tested in a Chilean cherry grove. It also sorts cherries into four export size classes. The motivation is that current crop estimates carry about 45% error.
- *Key result:* detection accuracy up to 85%, and production estimated with up to 25% error (abstract).
- *Relevance:* This gives a realistic orchard-scale error bar (about 25%) to set the toy-wall results against.

**[62]** S. Amatya, M. Karkee, A. Gongal, Q. Zhang, and M. D. Whiting, "Detection of cherry tree branches with full foliage in planar architecture for automated sweet-cherry harvesting," *Biosyst. Eng.*, vol. 146, pp. 3–15, Jun. 2016, doi: 10.1016/j.biosystemseng.2015.10.003. `V3` · abstract
- *Summary:* A Bayesian classifier labels pixels as branch, cherry, leaf or background in images of planar (trellised) cherry canopies. Intermittently visible branch segments are then linked by length and orientation, and fitted with linear or logarithmic branch models.
- *Key result:* 89.6% branch-pixel accuracy, and 89.2% branch detection on 141 full-foliage images (abstract).
- *Relevance:* A planar cherry canopy is the real-world version of our wall. The same linear-structure reasoning can flag stems and branches as non-fruit.

**[63]** S. Amatya, M. Karkee, Q. Zhang, and M. D. Whiting, "Automated detection of branch shaking locations for robotic cherry harvesting using machine vision," *Robotics*, vol. 6, no. 4, Art. no. 31, Oct. 2017, doi: 10.3390/robotics6040031. `V3` · abstract
- *Summary:* Branch and cherry regions found in RGB images are combined with depth from a stereo-calibrated 3D camera to choose shaking points. Harvest trials were run on Y-trellis and vertical-trellis canopies.
- *Key result:* RMSE to the desired shaking points was 0.064 m. Fruit removal reached 92.9% on the Y-trellis (up to five shakes) and 86.6% on the vertical trellis (up to four shakes) (abstract).
- *Relevance:* Centimetre-scale 3D localisation is attainable in planar cherry canopies, so our rail-derived depth error should be reported on this scale.

**[64]** A. Gilson *et al.*, "Cherry yield forecast: Harvest prediction for individual sweet cherry trees," in *Proc. 16th Int. Conf. Precision Agriculture (ICPA)*, Monticello, IL, USA: Int. Soc. Precision Agriculture, 2024 (unpaginated). [Online]. Available: https://arxiv.org/abs/2503.20419. `VarX` · full text (arXiv). The venue is as stated in the arXiv journal reference. The ISPA proceedings page was not checked.
- *Summary:* Three 'Satin' sweet-cherry trees were followed from dormancy to harvest in 2023, with manual counts at each development stage, to find which stages predict yield. The state-of-the-art FruitNeRF counter [14] was also tried on the cherries.
- *Key result:* linear regression on manual counts gave R² from 0.39 at early stages up to 0.99 two weeks before harvest. FruitNeRF could not produce a valid cherry count. The authors locate the cause in *visibility*, not 2D segmentation: cherries hang in clusters that are poorly visible from multiple angles and stay hidden behind leaves, so only a small percentage is visible even at close range.
- *Relevance:* This is the most directly relevant negative result we found. Occlusion, not recognition, defeats current 3D counting on cherries, which is exactly the gap our synthetic-aperture de-occlusion targets.

**[65]** L. Long, G. Lang, S. Musacchi, and M. Whiting, *Cherry Training Systems*, Pacific Northwest Extension Publ. PNW 667. Corvallis, OR, USA: Oregon State Univ. Extension Service, 2015. `Vpub` (Oregon State University Extension catalogue; reviewed 2023; no DOI) · publisher page summary
- *Summary:* An extension guide to eight commercial sweet-cherry training systems: Kym Green Bush, Spanish Bush, Steep Leader, Super Slender Axe, Tall Spindle Axe, UFO, UFO-Y and Vogel Central Leader. Its stated goals are early fruiting, well-exposed fruiting wood, renewal, and simple repeated canopy units. UFO ("upright fruiting offshoots", developed by Whiting) forms a narrow, fully trellised fruiting wall that is easy to harvest and prune.
- *Key result:* qualitative (grey literature).
- *Relevance:* This justifies the "orchard wall" geometry. Modern planar systems present the fruit near one plane, so a fronto-parallel rail with a depth gate around the wall is realistic, not just a lab convenience.

### A5. SAM 3, open-vocabulary detection, and foundation models for fruit

**[66]** N. Carion *et al.*, "SAM 3: Segment anything with concepts," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2026. [Online]. Available: https://arxiv.org/abs/2511.16719, doi: 10.48550/arXiv.2511.16719. `VarX` · partial full text (arXiv HTML, Tables 1–3). There are 38 authors at Meta, first Nicolas Carion and last Christoph Feichtenhofer. arXiv v1 is dated 20 November 2025 and v2 28 March 2026, and the ICLR 2026 proceedings entry was checked.
- *Summary:* The paper defines **Promptable Concept Segmentation** (PCS). Given a short noun phrase, image exemplars (positive and negative boxes), or both, SAM 3 detects, segments and tracks *all* matching instances in images and short videos, returning masks and identities. An image-level detector and a memory-based video tracker share one backbone, and a "presence head" separates recognising a concept from localising it. A data engine produced 4 M unique concept labels, including hard negatives, and the paper releases the SA-Co benchmark.
- *Key result:* SA-Co/Gold cgF1 54.1, against 24.6 for the strongest baseline (OWLv2\*) and 72.8 for humans. CountBench MAE 0.12 (93.8% accuracy). ODinW13 zero-shot AP 61.0. About 30 ms per image with more than 100 objects on an H200.
- *Relevance:* This is the team's named tool. Concept prompts ("cherry", "lime", "green grape") give colour-agnostic instance masks without training, and exemplar prompts can adapt it to toy cherries. But it works per image in 2D: our multi-view geometry is still needed for de-duplication, and occlusion still limits it [14], [69].
- *Caveat (reader-inferred):* we found no peer-reviewed agricultural *counting* benchmark for SAM 3 (search, October 2026).

**[67]** A. Kirillov *et al.*, "Segment anything," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, Paris, France, Oct. 2023, pp. 3992–4003, doi: 10.1109/ICCV51070.2023.00371. `V3` · abstract
- *Summary:* The paper introduces promptable segmentation from points, boxes or masks, the SAM model, and the SA-1B dataset (over 1 billion masks on 11 M images), built with a model-in-the-loop data engine.
- *Key result:* zero-shot performance often competitive with, or better than, fully supervised methods (abstract).
- *Relevance:* SAM can turn our colour-free point or blob candidates into masks via point or box prompts. That is a cleaner route than text prompts, although SAM itself has no notion of "fruit".

**[68]** S. Liu *et al.*, "Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection," in *Computer Vision – ECCV 2024* (Lecture Notes in Computer Science). Cham, Switzerland: Springer, 2024, pp. 38–55, doi: 10.1007/978-3-031-72970-6_3. `V2` (Semantic Scholar rate-limited; arXiv:2303.05499 checked) · abstract
- *Summary:* An open-set detector that fuses language and vision through a feature enhancer, language-guided query selection and a cross-modality decoder. It detects arbitrary objects named by category or by referring expression.
- *Key result:* 52.5 AP on COCO zero-shot transfer, and a mean of 26.1 AP on ODinW zero-shot (arXiv abstract).
- *Relevance:* It supplies the text-prompted box half of Grounded-SAM, which FruitNeRF [14] used. Fruit-specific prompts proved brittle there.

**[69]** R. Sapkota, K. I. Roumeliotis, M. Karkee, and N. D. Tselikas, "Generalization vs. specialization: Evaluating segment anything model (SAM3) zero-shot segmentation against fine-tuned YOLO detectors," arXiv:2512.11884, Dec. 2025, doi: 10.48550/arXiv.2512.11884. `VarX` · abstract. This is a preprint, not peer reviewed, and post-2024, which the skill flags as a contamination risk.
- *Summary:* SAM 3 in zero-shot mode is compared with YOLO11 (nano, medium, large) fine-tuned for instance segmentation on MinneApple [6] (670 images, 28,179 apples), focusing on dense, occluded fruit and on the effect of the IoU threshold.
- *Key result:* at IoU 0.15, YOLO11 scored F1 68.9–72.2% against 59.8% for SAM 3 zero-shot. The choice of IoU threshold could inflate the gap by up to 30%. YOLO lost 48–50 points across IoU thresholds, while SAM 3 lost only 4 (abstract).
- *Relevance:* This is the only agricultural SAM 3 evaluation we found. Zero-shot SAM 3 is usable but about 10 F1 points behind a fine-tuned detector on dense apples. The *matching criterion alone* changes the conclusion, so our protocol must fix it in advance and report it.

*Also verified (supporting, brief):*
- **[70]** N. Ravi *et al.*, "SAM 2: Segment anything in images and videos," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025. [Online]. Available: https://arxiv.org/abs/2408.00714. `VarX` (ICLR 2025 proceedings listing checked). Streaming-memory video segmentation that is more accurate and 6× faster than SAM on images (abstract). Its video propagation could carry masks along our 13-frame rail sequence.
- **[71]** A. Gilson, L. Hennig, and P. Pietrzyk, "Zero-shot 3D plant organ segmentation with SAM3 and semantic NeRFs," arXiv:2609.07724, Sept. 2026. `VarX` (preprint, submitted to the CVPPA workshop at ECCV 2026). Text-prompted SAM 3 masks fused by a semantic NeRF reach 92.6% mIoU (95.9% of the oracle with ground-truth masks), and 0.856 mIoU across ten plants. Multi-view fusion acts as an implicit consensus over imperfect per-frame masks (abstract).
- **[72]** N. Amini-Naieni, T. Han, and A. Zisserman, "CountGD: Multi-modal open-world counting," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 37, Vancouver, BC, Canada, 2024, pp. 48810–48837, doi: 10.52202/079017-1547. `V3`. Open-world counting prompted by text and/or visual exemplars. It is a candidate single-image counting baseline to run on each refocused image.
- **[73]** A. Carraro, M. Sozzi, and F. Marinello, "The segment anything model (SAM) for accelerating the smart farming revolution," *Smart Agric. Technol.*, vol. 6, Art. no. 100367, Dec. 2023, doi: 10.1016/j.atech.2023.100367. `V3`. An early evaluation of SAM for segmenting objects in precision-agriculture imagery (abstract; no figures stated).

### A6. iPhone LiDAR and ARKit: depth and pose accuracy

**[74]** G. Luetzenburg, A. Kroon, and A. A. Bjørk, "Evaluation of the Apple iPhone 12 Pro LiDAR for an application in geosciences," *Sci. Rep.*, vol. 11, Art. no. 22221, Nov. 2021, doi: 10.1038/s41598-021-01763-9. `V3` · abstract
- *Summary:* The LiDAR in the iPad Pro 2020 and iPhone 12 Pro was characterised, first on objects and then on a coastal cliff, against SfM multi-view-stereo point clouds.
- *Key result:* objects with sides longer than 10 cm were modelled to ±1 cm absolute accuracy, and models up to 130 × 15 × 10 m to ±10 cm (abstract).
- *Relevance:* A 2 cm toy cherry is below the validated object scale. Phone LiDAR is therefore enough to separate the foliage layer (0.3 m) from the fence plane (0.9 m), but not to resolve individual cherries.

**[75]** C. Gollob, T. Ritter, R. Kraßnitzer, A. Tockner, and A. Nothdurft, "Measurement of forest inventory parameters with Apple iPad Pro and integrated LiDAR technology," *Remote Sens.*, vol. 13, no. 16, Art. no. 3129, Aug. 2021, doi: 10.3390/rs13163129. `V3` · abstract
- *Summary:* iPad Pro LiDAR point clouds were compared with personal laser scanning (PLS) for measuring individual trees on 21 forest plots of 7 m radius.
- *Key result:* tree detection was 97.3% with the iPad vs 99.5% with PLS (diameter at breast height (dbh) ≥ 10 cm). The best dbh RMSE was 3.13 cm vs 1.59 cm, and a plot took about 7.5 min (abstract).
- *Relevance:* Few-centimetre accuracy at tree scale is suitable for canopy and wall geometry, not for sizing fruit.

**[76]** M. Vogt, A. Rips, and C. Emmelmann, "Comparison of iPad Pro®'s LiDAR and TrueDepth capabilities with an industrial 3D scanning solution," *Technologies*, vol. 9, no. 2, Art. no. 25, Apr. 2021, doi: 10.3390/technologies9020025. `V3` · abstract
- *Summary:* Lego bricks were scanned with the iPad Pro LiDAR, the TrueDepth front sensor and an industrial scanner, and the scans compared on shape and position tolerances.
- *Key result:* the industrial scanner was consistently more accurate. The authors judge the phone sensors possibly sufficient depending on the application (abstract; no figures stated).
- *Relevance:* At centimetre object scale, phone depth sensing is coarse. For 2 cm targets, multi-view geometry from the rail images is likely to beat the LiDAR depth map.

**[77]** E. Grobler and G. Celano, "Photogrammetric and LiDAR scanning with iPhone 13 Pro: Accuracy, precision and field application on hazelnut trees," *Sensors*, vol. 25, no. 18, Art. no. 5629, Sept. 2025, doi: 10.3390/s25185629. `V3` · abstract
- *Summary:* An iPhone 13 Pro scanned cylindrical targets of known size and hazelnut trees (height, stem diameter, leaf area) in both the LiDAR and the photogrammetric ("Photo") modes of the Polycam app.
- *Key result:* Photo mode was most accurate, with 0.007 m systematic error and R² 0.99, and agreed with manual leaf-area measurements at R² 0.93 (abstract).
- *Relevance:* In the only orchard-crop iPhone study we found, *photogrammetry beat the phone's LiDAR*. That supports image-based geometry from the rail (plane sweep or SfM) over LiDAR depth.

**[78]** T. Feigl, A. Porada, S. Steiner, C. Löffler, C. Mutschler, and M. Philippsen, "Localization limitations of ARCore, ARKit, and Hololens in dynamic large-scale industry environments," in *Proc. 15th Int. Joint Conf. Comput. Vis., Imag. Comput. Graph. Theory Appl. (VISIGRAPP), GRAPP*, 2020, pp. 307–318, doi: 10.5220/0008989903070318. `V3` · abstract
- *Summary:* ARKit, ARCore and HoloLens tracking were compared with sub-millimetre optical ground truth in a 1,600 m² industrial hall, with 60 m walks and changing features.
- *Key result:* out of the box, the average error was about 17 m per 120 m of path, with scale error up to 14.4 cm/m (abstract).
- *Relevance:* ARKit drift grows with path length and scene change, and scale error is the worry. On a 0.6 m mechanical rail, the known 50 mm spacing is a stronger pose prior. Use ARKit poses only after checking them against the rail.

*Also verified (supporting, brief):*
- **[79]** S. Tatsumi, K. Yamaguchi, and N. Furuya, "ForestScanner: A mobile application for measuring and mapping trees with LiDAR-equipped iPhone and iPad," *Methods Ecol. Evol.*, vol. 14, no. 7, pp. 1603–1609, 2023, doi: 10.1111/2041-210X.13900. `V3`. Real-time instance segmentation plus circle fitting on iPhone or iPad LiDAR mapped 672 trees in about 1.6 h. Stem diameter agreed with tape measurements at R² 0.963 (iPhone), and tree mapping at Mantel R² 0.999 (abstract). This is on-device primitive fitting, the larger-scale cousin of fruit sphere fitting.

*No peer-reviewed study of iPhone LiDAR for on-tree fruit sizing or counting was found* (search, October 2026). A 2022 Auckland University of Technology master's thesis reportedly found sizing errors within 6% for apples and cherries at 0.5–2 m. It is listed as **UNVERIFIED** in Appendix B and is not used as evidence.

### A7. Evaluation protocols, synthetic benchmarks and reviews

**[80]** M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman, "The PASCAL visual object classes (VOC) challenge," *Int. J. Comput. Vis.*, vol. 88, no. 2, pp. 303–338, Jun. 2010, doi: 10.1007/s11263-009-0275-4. `V2` (CrossRef and OpenAlex; no Semantic Scholar match) · read scope: summary of the standard protocol
- *Summary:* The paper defines the VOC detection benchmark and its protocol. Detections are matched to ground truth by box overlap (IoU ≥ 0.5) in order of confidence, duplicate detections of one object count as false positives, and average precision summarises the precision–recall curve.
- *Key result:* methodological (the IoU ≥ 0.5 matching rule and AP).
- *Relevance:* This is the default detection metric. For roughly 40 px fruit, IoU punishes a few pixels of box error and the threshold choice can swing conclusions by up to 30% [69]. That is the argument for centre-distance matching [81].

**[81]** J. Ribera, D. Güera, Y. Chen, and E. J. Delp, "Locating objects without bounding boxes," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Long Beach, CA, USA, Jun. 2019, pp. 6472–6482, doi: 10.1109/CVPR.2019.00664. `V3` · full text (arXiv:1806.07564), metric definitions (Sec. 5, Eqs. 11–12)
- *Summary:* Objects are located as points by a network trained with a loss based on a weighted average Hausdorff distance, so no boxes or proposals are needed. It was tested on head, pupil-centre and plant-centre datasets. **Evaluation:** an estimate is a true positive if it lies within distance *r* of a ground-truth point, and a false positive if it has none within *r*. Precision, recall and F-score follow from these. Count error is reported per image as MAE, RMSE and MAPE (MAPE excludes images with a true count of zero).
- *Key result:* plant-centre dataset precision 88.1%, recall 89.2%, F-score 88.6% (reported at the paper's chosen *r*). It outperformed generic detectors on all three datasets.
- *Relevance:* This is the cleanest published template for our point-based fruit evaluation: a matching radius *r*, P/R/F1 and count-error metrics together.

**[82]** R. Barth, J. IJsselmuiden, J. Hemming, and E. J. van Henten, "Data synthesis methods for semantic segmentation in agriculture: A Capsicum annuum dataset," *Comput. Electron. Agric.*, vol. 144, pp. 284–296, Jan. 2018, doi: 10.1016/j.compag.2017.12.001. `V3` · abstract
- *Summary:* 42 sweet-pepper plant models were procedurally generated from 21 empirically measured plant properties, and 10,500 Blender renders were produced with automatic per-pixel class and depth labels. Fruit came from 3D scans and textures from photographs. 750 empirical images of 50 plants served as the reference, and the data are public.
- *Key result:* the synthetic images were judged highly similar to the empirical ones, both qualitatively and quantitatively (abstract).
- *Relevance:* This is the template for our synthetic benchmark. Render the wall with known 3D fruit positions, sweep foliage density, and export labels plus depth. That gives exact count ground truth at controlled occlusion levels.

**[83]** M. Rahnemoonfar and C. Sheppard, "Deep count: Fruit counting based on deep simulated learning," *Sensors*, vol. 17, no. 4, Art. no. 905, Apr. 2017, doi: 10.3390/s17040905. `V3` · abstract
- *Summary:* A count-regression network (a modified Inception-ResNet) was trained entirely on synthetic images and tested on real ones. It counts through shadow, foliage occlusion and overlap.
- *Key result:* 91% average test accuracy on real images and 93% on synthetic images (abstract).
- *Relevance:* Synthetic training and testing can transfer to real counting. The synthetic-to-real gap should still be reported explicitly.

**[84]** A. Koirala, K. B. Walsh, Z. Wang, and C. McCarthy, "Deep learning – Method overview and review of use for fruit detection and yield estimation," *Comput. Electron. Agric.*, vol. 162, pp. 219–234, Jul. 2019, doi: 10.1016/j.compag.2019.04.017. `V3` · abstract
- *Summary:* A review of deep learning for fruit detection. It recommends standard metrics (e.g. F1 for binary detection) and public fruit-in-orchard image sets, and reviews how image counts are extrapolated to orchard yield, including corrections for occluded fruit.
- *Key result:* recommendations (no single figure).
- *Relevance:* It backs our protocol choices: F1 with an explicit matching rule, public benchmarks, and a clear separation between visible-fruit counts and true totals.

*Also verified (supporting, brief):*
- **[85]** Z. Fei, A. G. Olenskyj, B. N. Bailey, and M. Earles, "Enlisting 3D crop models and GANs for more data efficient and generalizable fruit detection," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops (ICCVW)*, Oct. 2021, pp. 1269–1277, doi: 10.1109/ICCVW54120.2021.00147. `V3`. A semantically constrained GAN translates synthetic 3D crop-model images into realistic grape images while preserving fruit positions, which improves data efficiency and domain adaptation (abstract).
- **[86]** A. Gongal, S. Amatya, M. Karkee, Q. Zhang, and K. Lewis, "Sensors and systems for fruit detection and localization: A review," *Comput. Electron. Agric.*, vol. 116, pp. 8–19, Aug. 2015, doi: 10.1016/j.compag.2015.05.021. `V3`. A general review of fruit-detection sensors and systems. The abstract was not retrieved, so it is cited only as background.
- **[87]** K. Kapach, E. Barnea, R. Mairon, Y. Edan, and O. Ben-Shahar, "Computer vision for fruit harvesting robots – State of the art and challenges ahead," *Int. J. Comput. Vis. Robot.*, vol. 3, no. 1/2, 2012, doi: 10.1504/IJCVR.2012.046419. `V3`. A review of the visual cues and algorithms used by harvesting robots, with their limits (abstract).
- **[88]** N. T. Anderson, K. B. Walsh, and D. Wulfsohn, "Technologies for forecasting tree fruit load and harvest timing—From ground, sky and time," *Agronomy*, vol. 11, no. 7, Art. no. 1409, Jul. 2021, doi: 10.3390/agronomy11071409. `V3`. A review concluding that machine vision dominates fruit-load research, while better sampling strategies may have the widest impact (abstract).
- **[89]** C. Neupane, M. Pereira, A. Koirala, and K. B. Walsh, "Fruit sizing in orchard: A review from caliper to machine vision with deep learning," *Sensors*, vol. 23, no. 8, Art. no. 3868, Apr. 2023, doi: 10.3390/s23083868. `V3`. A review of in-orchard fruit sizing covering depth measurement, occluded-fruit recognition and commercial capability (abstract).

---

## Part B. Comparison of counting and detection methods (reported figures)

Figures are as reported in the cited sources; the provenance of each is given in the Part A entry. **Rows are not directly comparable.** Ground truth varies between visible-fruit counts (VIS), field or harvest counts (HARV), 3D positions (3D) and pixel or box labels (LAB). Some systems also embed a calibration factor. "n/r" means not reported in the text we read.

| Ref | Crop (fruit vs leaf colour) | Views / sensing | How double counts are avoided | Colour-free detection? | GT | Reported count error / accuracy | Detection F1 or P/R | Calibrated? |
|---|---|---|---|---|---|---|---|---|
| [1] | Mango | Multi-view video, camera + LiDAR + GPS/INS | Trajectory-based correspondence, triangulation, LiDAR tree masks | No (Faster R-CNN) | HARV (16 trees) | 1.36% per-tree error | n/r | No (single/dual-view needed it) |
| [2] | Orange (day), apple (night) | Monocular video | KLT + Kalman + Hungarian tracks, then SfM 3D merge, size and depth rejection | No (FCN) | VIS | Orange -0.2% (SD 7.8%), from 17.2% before 3D. Apple 3.3% (SD 4.1%), from 8.5% | n/r | No |
| [3] | Mango | Monocular video, both row sides | Semantic SfM landmarks, re-projection matching, trunk-centroid voting | No | HARV (18 trees) | 27.8 ± 29.6 fruit/tree (≈16% of the 175 mean, our calc.). R² 0.78, slope 0.86 | n/r | No |
| [4] | Apple (red) | SfM photogrammetry sequence | 2D masks projected onto the 3D cloud, SVM false-positive filter | No (Mask R-CNN) | 3D (1,455) | n/a (localisation task) | F1 0.816 (2D) → 0.881 (3D) | No |
| [5] | Apple (red/green) | Monocular, both sides | Tracking, then 3D merge of both sides | Partly (colour GMM detector; CNN counter) | HARV | 2.17–4.44% after 3D merge, vs 101.9–150% of truth if sides are summed | Varies by set (IoU sweep) | No |
| [6] | Apple (incl. green) | Single images (benchmark) | n/a | n/a | LAB | Patch counting 88.5–95.1% (CNN) | AP@0.5 0.775 (Faster R-CNN) | n/a |
| [7] | Mango (night, LED) | Two-sided images | None, per-image counts | No | HARV (packhouse) | 4.6–15.2% error | F1 0.968 | Yes, per orchard |
| [8] | Mango | 10 fps video | Kalman + Hungarian (many-to-one, "borrow") | No | VIS / HARV | +2.6% net (9.9% double, 7.3% missed). 62% of harvest (RMSE 18.0/tree) vs 40% dual-view | n/r | Bias-corrected RMSE |
| [12] | Pepper (green on green) | Multiple images per plant | Statistical clustering of repeated sightings | Partly (bag of words) | HARV (manual) | Correlation 74.2% (multi-image) vs 94.6% (image analysis) | n/r | No linear adjustment |
| [13] | Apple | Video | DeepSORT / MHT / SORT / KF / KCF | No | VIS | 20.07% (DeepSORT + YOLOv5), 31.52% (+ Faster R-CNN) | MOTA 93–97% with perfect detections | No |
| [14] | 6 species (synthetic), apple (real) | 20–300 posed views | Semantic NeRF, then 3D two-stage clustering | **Yes** (Grounded-SAM masks) | Synthetic exact, VIS, 3D | Real detection rate ≈89% / ≈82% | Synthetic F1 0.95 (GT masks) / 0.88 (SAM). Fuji-SfM F1 0.79 | No |
| [16] | Apple | Images spaced to avoid overlap | Image spacing only | No (CNN + watershed/CHT) | HARV | r² = 0.826 | F1 0.861 | Regression to harvest |
| [17] | Sweet pepper | Robot video | Tracking via detection | No | VIS | Within 4.1% | F1 77.3 | No |
| [18] | Apple | Monocular, both sides | Camera-motion merge plus side merge | No (colour clustering) | HARV | Yield accuracy 91.98–94.81%. Counting 89–98% | F1 0.95–0.97 | No |
| [39] | Grape (green on green) | Vehicle camera + lights | None (per-image), count→yield model | Partly (radial-symmetry candidates; L\*a\*b + Gabor k-NN) | HARV (weight) | 9.8% row-yield error | P 98.0%, R 63.7% | Yes (count → yield) |
| [40] | Grape | Vehicle, 4 seasons | None | Partly (shape + texture + colour) | HARV | 3–11% total-yield error | n/r | Yes |
| [41] | Grape, apple | Single images | n/a | **Yes** (intensity and gradient profiles + texture) | LAB | n/a | F1 0.82 / 0.80 | n/a |
| [42] | Red grape (night) | Single images | n/a | **Yes** (specular peaks) | VIS (manual) | 14% counting error, 7% false positives | n/r | No |
| [43] | Apple, red and green (night, flash) | Stereo, both sides | 3D merge within 0.16 m (≈2 diameters) | Partly (HSV + specular roundness test) | HARV (sections) | Red -3.2% (unthinned rows -41.3%). Green raw -29.8% → 1.2% after ×1.4 | n/r | Green yes (×1.4) |
| [44] | Green citrus | Single images | Blob merge per image | Partly (eigenfruit + Gabor + colour) | LAB | 75.3% of fruit detected | n/r | n/a |
| [45] | Green apple | Single images | n/a | Partly (colour/smoothness + circle arcs) | LAB | >85% (auto exposure) / ≈95% detected (<5% FP, diffuse light) | n/r | n/a |
| [46] | Green citrus | Single images | n/a | Mostly (CHT + SVM texture + line rejection) | LAB / manual | 80.4% detected and counted | n/r | n/a |
| [47] | Immature mango (night) | Single images | n/a | Reduced (texture + Hessian filters) | Manual | 78.3% detected, 10.6% error, R² 0.63 | n/r | n/a |
| [57] | Green grape berries | Overlapping harvester images | n/r | No (CNN segmentation) | LAB | Berry detection accuracy 94.0% / 85.6% | n/r | n/a |
| [58] | Grape berries (smartphone) | Single images | n/a | No | LAB | n/r | P 0.939–0.951, R 0.876–0.908 | n/a |
| [61] | **Cherry** | Portable camera | n/r | No (DNN) | HARV (production) | Production error up to 25% | Detection accuracy up to 85% | n/r |
| [64] | **Cherry** | Multi-view (FruitNeRF) | NeRF 3D clustering | Yes | Manual | **No valid count** (visibility). Manual-count regression R² 0.39–0.99 | n/a | n/a |
| [69] | Apple (MinneApple) | Single images | n/a | **Yes** (SAM 3 zero-shot) | LAB | n/a | F1 59.8% (SAM 3) vs 68.9–72.2% (YOLO11) at IoU 0.15 | n/a |
| [83] | Tomato (synthetic → real) | Single images | n/a | No | LAB | 91% accuracy (real) | n/r | n/a |

**Reading the table.** (i) Every system that reports a *calibration-free* error under ~5% against field truth fuses views in 3D ([1], [2], [5]). (ii) Single-view colour-free detectors for green fruit plateau at about 64–80% recall ([39], [44], [46], [47]). (iii) Cherries are the only crop here where 3D multi-view counting has been reported to fail outright [64].

---

## Part C. Synthesis

<!-- synthesis-start -->
### C1. How existing systems avoid double counting

Four de-duplication families appear, each more geometric than the last.

1. **Image-space tracking** (Kalman or KLT prediction with Hungarian assignment [2], [8], [17]; DeepSORT-style trackers [13]). It is cheap, but tracks fragment when fruit are briefly hidden, and the resulting double counts (9.9%) partly cancel the misses (7.3%) [8].
2. **Pose-based association.** Known camera poses constrain the matches, and triangulated points closer than a merge radius are fused: 0.16 m, about two apple diameters, in [43], with LiDAR tree masks in [1].
3. **Structure-from-motion lifting.** Duplicates, oversized outliers and background-row fruit are rejected geometrically [2]–[5], [11]. This cut an orange-count error from 17.2% to -0.2% [2], and turned 102–150% overcounts from summing both row sides into 2–4% errors [5].
4. **Implicit 3D fields.** NeRF fruit clouds are clustered, so double counting is avoided by construction [14].

Two limits recur. Association is bounded by detection: trackers reached 93–97% MOTA on perfect detections, yet left 20–32% count error with real detectors [13]. And no association recovers fruit that no view sees. Video still captured only 62% of harvested mangoes [8], multi-view systems undercounted dense trees [3], and FruitNeRF failed on sweet cherries because clustered fruit were "poorly visible from multiple angles" [64].

Our rail belongs to family 2. Its poses are known mechanically, so association reduces to 1D epipolar geometry. Our back-of-envelope estimate assumes a ≈26 mm-equivalent main camera (f ≈ 2,800 px). Between adjacent rail positions, the fence plane then shifts ≈156 px against ≈467 px for foliage at 0.3 m. Triangulating across the full 0.6 m aperture gives ≈0.5 mm of depth per pixel of disparity error, ample to gate out foliage, split cherry twins and merge duplicates within one fruit diameter.

### C2. Colour-free cues that work for round fruit

Every cue that succeeds on fruit coloured like their leaves exploits 3D shape or its shading.

- **Radial symmetry** [37] proposes candidate centres from gradients. It gave 98.0% precision but only 63.7% recall for green berries, with about 60% of misses lost at the symmetry stage [39].
- **Specular highlights with monotonic fall-off** suit glossy spheres best. Wang et al. accept a maximum when at least 4 of 8 half-lines descend [43]. Intensity and gradient-orientation profiles reached F1 0.80–0.82 [41], and specular peaks counted grapes with 14% error [42].
- **Circle evidence** (circular Hough, arc fitting) is the oldest cue [35]. Followed by texture and line rejection, it found 80.4% of green citrus [46].
- **Scale-normalised LoG/Hessian blobs** [38] add size estimation. Hessian eigenvalues separate blobs from ridge-like stems, and Hessian filtering improved detection of immature mangoes that lacked red colour [47].
- **Geometry** is colour-agnostic by construction: range images [36], surface normals with symmetry and highlights [48], superellipsoid fitting [51], point-cloud clustering [50] and 3D descriptors [52].

Single-view detectors that lean little on colour plateau at about 64–80% recall [39], [44], [46], [47], and precision is the easier half. Multiple views should therefore buy recall, since a fruit need only be seen well somewhere. Geometry supplies precision: depth gate, size and sphericity.

### C3. Why colour thresholding fails for green fruit

The colour distributions overlap. Green apples and foliage shared hues of 49°–75° [43], and immature citrus and mango closely match their leaves [44], [47], [55]. Thresholds also reject the fruit's own specular centre, which has low saturation [43], and they swing with lighting: one green-apple detector went from >85% detection with many false positives to ≈95% with <5% false positives by changing exposure and lighting [45]. Because thresholds classify pixels, not objects, reddish stems and shadows pass while shaded cherries fail. That is our teammate's failure pattern. One tension remains: once fruit are isolated by non-colour means, green clusters were counted *more* accurately than red ones (95.1%) [6], because there is less colour variation to model.

### C4. A rigorous evaluation protocol for our project

- **Ground truth.** Record each cherry's 3D lattice position (a twin counts as two fruit). Re-shoot every scene without foliage to obtain per-fruit visibility and occlusion fractions.
- **2D detection** (raw and refocused views). Use one-to-one Hungarian matching of centres. A true positive lies within *r* of its match [81]: primary *r* = 0.5 *D* (fruit diameter in pixels), with a sweep over {0.25, 0.5, 1.0} *D*, because the matching rule alone shifted conclusions by up to 30% [69]. Add AP@0.5 for comparability [6], [59], [80].
- **3D count (primary outcome).** Use one-to-one assignment within *R* = 1 *D* (≈20–25 mm). Report TP, FP, FN, F1, signed error (Ĉ - C)/C, and double-count and miss rates separately [8].
- **Across scenes.** Report MAE, RMSE and MAPE [81], plus slope and R² against truth [3], with bootstrap confidence intervals.
- **Stratification.** Break results down by occlusion fraction, by views *k* = 1…13 (a visibility curve as in [20]) and by foliage density, which tests the aperture limits of [29].
- **Ablations.** Compare the colour-threshold baseline against the colour-free detector. Compare detecting then integrating with integrating then detecting [30]. Run oracle detections to isolate de-duplication error [13]. Test with and without the depth gate [56], and against SAM 3 concept prompts [66], [69].
- **Generalisation.** Repeat identical scenes with limes and green grapes. Add rendered walls with exact 3D truth [82], external 2D sets (MinneApple green apples, WGISD, Cherry CO [6], [11], [59]), and Fuji-SfM for the 3D code [15].
- **Discipline.** Freeze *r*, *R* and all detector thresholds on development scenes before the held-out captures.

### C5. Gaps we can claim (bounded by the search in Section 0)

1. No optical synthetic-aperture or light-field method has been applied to fruit detection or counting. Through-foliage synthetic aperture exists for drones [29], [30], and a camera array has been used to steer a harvesting robot [23].
2. No colour-free detector has been evaluated on cherries. Published cherry detectors are colour-trained networks [59]–[61].
3. State-of-the-art 3D counting has been reported to fail on sweet cherries through lack of visibility [64].
4. There is no agricultural counting benchmark for SAM 3 [69].
5. There is no peer-reviewed fruit-scale validation of iPhone LiDAR or ARKit poses [74]–[79].
6. Double counts and misses are rarely reported separately [8], [13].

A small-baseline, mechanically posed linear array, between video tracking and wide-baseline SfM, is itself unstudied.
<!-- synthesis-end -->

---

## Part D. The 10 essential references for the report (ranked)

| Rank | Ref | Why it must be cited |
|---|---|---|
| 1 | [5] Häni, Roy & Isler 2020 (*J. Field Robot.*) | The strongest quantitative case that views must be fused in 3D. Summing row sides gives 102–150% of the true count, while 3D merging gives 2.2–4.4% error. It is also the first head-to-head comparison of detection and counting methods. |
| 2 | [1] Stein, Bargoti & Underwood 2016 (*Sensors*) | The closest analogue of our pipeline: known poses, per-image detection and triangulation, giving a calibration-free 1.36% per-tree error. |
| 3 | [2] Liu *et al.* 2018 (IROS) | The template for 3D sanity checks after 2D tracking (merge, size and depth rejection), which cut orange-count error from 17.2% to -0.2%. |
| 4 | [14] Meyer *et al.* 2024 FruitNeRF (IROS) | State-of-the-art colour-agnostic 3D counting with Grounded-SAM masks. It documents its own limits: at least 20–40 views needed, recall collapses under occlusion, and prompts are brittle. |
| 5 | [64] Gilson *et al.* 2024 (ICPA) | Direct evidence that visibility, not recognition, defeats current 3D counting on *sweet cherries*. This is the problem statement for our synthetic-aperture step. |
| 6 | [43] Wang *et al.* 2013 (ISER / Springer) | It gives a colour-free specular "roundness" test for green apples and a 3D merge radius of about two fruit diameters. It also shows how calibration factors and cluster-splitting limits shape headline accuracies. |
| 7 | [39] Nuske *et al.* 2011 (IROS) | Radial-symmetry detection of green berries on green leaves (P 98.0%, R 63.7%), showing that candidate generation is the recall bottleneck. |
| 8 | [41] Pothen & Nuske 2016 (ICRA) | An intensity- and gradient-profile keypoint detector for round fruit (F1 0.80–0.82). It is the most directly reusable colour-free cue for glossy cherries and limes. |
| 9 | [66] Carion *et al.* 2026 SAM 3 (ICLR), read with [69] Sapkota *et al.* 2025 | The team's named tool (concept-prompt segmentation, cgF1 54.1 on SA-Co/Gold), and its only agricultural test (F1 59.8% vs 68.9–72.2% for fine-tuned YOLO11, sensitive to IoU). |
| 10 | [20] Hemming *et al.* 2014 (*Sensors*) | The canonical visibility-vs-viewpoints result: one view reaches at most 69% detectability, and five views reach 90%. It motivates our *k* = 1…13 analysis. |

*Runners-up:* [3] (monocular SfM counting and the double-counting taxonomy), [30] (detect-then-integrate beats integrate-then-detect behind foliage), [29] (aperture-size and sample-count limits), [81] (point-matching protocol with radius *r*), [48] (colour-agnostic 3D shape detector), [4] (2D → 3D F1 gain).

---

## Appendix A. Corrections and clarifications the verification forced on the brief

1. **Nuske 2011:** the title is "Yield estimation in vineyards **by** visual grape detection" (IROS 2011, pp. 2352–2358) [39], not "via".
2. **"Wang/Walsh/Koirala MangoYOLO (2019)"** is two papers. The MangoYOLO benchmark is Koirala, Walsh, Wang & McCarthy, *Precis. Agric.* 20(6) [7]. The video tracking paper is Wang, Walsh & Koirala, *Sensors* 19(12), 2742 [8].
3. **Lin et al. guava** is *Sensors* 19(2), 428 (2019) [50], not 2016. **Nguyen et al. 2016** uses colour plus depth, so it is not colour-free [49]. **Lehnert et al. 2016** fits a superellipsoid *after* colour segmentation [51].
4. **Cherry CO** has the exact title "Cherry CO Dataset: A Dataset for Cherry Detection, Segmentation and Maturity Recognition", *IEEE RA-L* 9(6), 5552–5558, June 2024 [59].
5. **Gai et al.** was published online in 2021 and in print in 2023, in vol. 35, no. 19, pp. 13895–13906 [60]. **Häni et al. (JFR)** was online in August 2019, with the issue dated 2020 [5].
6. **Whittaker et al. 1987:** the correct DOI is 10.13031/2013.30444. CrossRef rejected the DOI ending .30443 on title mismatch [35].
7. **Specular-highlight detection:** the worked colour-free specular method is Wang *et al.* 2013 on apples [43] (verified in the full text), together with Font *et al.* 2014 on grapes [42]. The Payne and Walsh mango work we read relies on texture and Hessian filtering rather than highlights [47].
8. **SAM 3** is N. Carion *et al.* (38 authors, Meta), arXiv:2511.16719 (v1 20 November 2025), ICLR 2026 [66]. **SAM 2** appeared at ICLR 2025 [70].
9. **"Long et al. 2015 / Whiting"** is the Pacific Northwest Extension guide PNW 667, *Cherry Training Systems*, by Long, Lang, Musacchi and Whiting [65]. It is grey literature with no DOI.
10. **Bargoti & Underwood's circular Hough work** is the 2017 *J. Field Robot.* paper [16], not the ICRA paper [9].

## Appendix B. Excluded records and UNVERIFIED items (not used as evidence)

**Existence verified (`V3`), but content not retrievable beyond the title, so excluded rather than annotated from memory:**
- F. Gao *et al.*, "A novel apple fruit detection and counting methodology based on deep learning and trunk tracking in modern orchard," *Comput. Electron. Agric.*, vol. 197, Art. no. 107000, 2022, doi: 10.1016/j.compag.2022.107000.
- R. Kirk, M. Mangan, and G. Cielniak, "Robust counting of soft fruit through occlusions with re-identification," in *Computer Vision Systems (ICVS 2021)*, Lecture Notes in Computer Science, Springer, 2021, pp. 211–222, doi: 10.1007/978-3-030-87156-7_17.
- S. Chaivivatrakul and M. N. Dailey, "Texture-based fruit detection," *Precis. Agric.*, vol. 15, no. 6, pp. 662–683, 2014, doi: 10.1007/s11119-014-9361-x.
- G. Lin, Y. Tang, X. Zou, J. Xiong, and Y. Fang, "Color-, depth-, and shape-based 3D fruit detection," *Precis. Agric.*, vol. 21, no. 1, pp. 1–17, 2020, doi: 10.1007/s11119-019-09654-w.
- J. Rakun, D. Stajnko, and D. Zazula, "Detecting fruits in natural scenes by using spatial-frequency based texture analysis and multiview geometry," *Comput. Electron. Agric.*, vol. 76, no. 1, pp. 80–88, 2011, doi: 10.1016/j.compag.2011.01.007. The title suggests texture plus multi-view geometry, so it is worth getting through the library.
- D. Stajnko, M. Lakota, and M. Hočevar, "Estimation of number and diameter of apple fruits in an orchard during the growing season by thermal imaging," *Comput. Electron. Agric.*, vol. 42, no. 1, pp. 31–42, 2004, doi: 10.1016/S0168-1699(03)00086-3.

**Tangential (verified, but excluded at screening):**
- Kang & Chen 2020 (DaSNet-V2, doi: 10.1016/j.compag.2020.105302).
- Tian *et al.* 2019 (YOLOv3-dense apples, doi: 10.1016/j.compag.2019.01.012).
- Gongal *et al.* 2018 (apple sizing with a time-of-flight camera, doi: 10.1016/j.inpa.2018.06.002).
- Roy, Dong & Isler 2018 (registering two row sides, doi: 10.1109/IROS.2018.8594167).
- Zhang *et al.* 2018 (branch detection in fruiting-wall apples).
- Williams *et al.* 2024 (Leaf Only SAM, doi: 10.1016/j.atech.2024.100515).
- Sapkota *et al.* 2025, "The SAM2-to-SAM3 gap…" (arXiv:2512.06032), an analysis paper with no experiments.

**UNVERIFIED (grey literature, vendor material or unlocated sources):**
- An Auckland University of Technology master's thesis (2022) reporting iPhone LiDAR fruit-sizing errors within 6% for apples and cherries at 0.5–2 m. **UNVERIFIED**: author and title not confirmed in any index.
- Croptracker HQV vendor claims of sub-millimetre fruit sizing with iPhone or iPad LiDAR. **UNVERIFIED** vendor statement.
- The "SYNTOM-SAM3" Hugging Face model card (tomato fruit IoU about 0.53 zero-shot vs 0.66 fine-tuned) and its ECCV 2026 workshop paper. **UNVERIFIED**: paper not located in any index.
- A 2026 preprint comparing ARKit with Vicon motion capture (translational RPE under 5 cm), and a 2022 benchmark reporting about 0.02 m/s ARKit drift, both seen only in search summaries. **UNVERIFIED**.

## Appendix C. Cross-paper tension inventory (scoped advisory scan)

| ID | Pair | Basis | Assessment | Resolution | Scholar confirmation |
|---|---|---|---|---|---|
| CP-001 | [6] vs [44], [46] | Shared topic: green fruit | Conditional difference. Green fruit are hard for *colour-based detection*, but counting green clusters once isolated was the most accurate case (95.1%) | Resolved in Part C, C3 | pending |
| CP-002 | [66] vs [69] | Shared measure: SAM 3 accuracy | Conditional difference. State of the art on generic concepts (SA-Co), yet behind fine-tuned YOLO on dense, occluded apples, and sensitive to the IoU threshold | Resolved in C4 (fixed matching rule) | pending |
| CP-003 | [1] vs [43], [7] | Shared outcome: need for calibration | Conditional difference. Calibration-free multi-view counting [1] vs ×1.4–1.7 factors [43] and per-orchard factors [7]. The difference tracks canopy density, clustering and ground-truth type | Resolved in C1 | pending |
| CP-004 | [30] vs [28] | Shared question: where to integrate | Insufficient overlap. Drone anomaly detection vs light-field stereo, with no fruit study | Flagged unresolved; our ablation tests it | pending |
| CP-005 | [77] vs [74] | Shared measure: phone LiDAR accuracy | No material conflict. Accuracy is scale-dependent (±1 cm for objects over 10 cm; photogrammetry better on trees) | n/a | pending |
| CP-006 | [25] vs [1] | Shared outcome: correcting for unseen fruit | Conditional difference. A learned correction was excellent within one season but did not transfer, while geometric multi-view needed no correction | Resolved in C1 and C5 | pending |

*Coverage note:* 89 papers are in the corpus and 6 candidate pairs were assessed, selected by shared subtopic, shared measure or opposite findings. This is a scoped advisory scan, not complete pairwise contradiction detection. Cross-cluster tensions not listed may exist and are not claimed absent. The team should confirm each resolution.

## Appendix D. Search limitations

- Index abstracts were missing for many Elsevier and Springer papers. Their figures come from institutional-repository abstracts, and five full texts were read. Page-level verification of every figure was not possible.
- Semantic Scholar rate-limited a few lookups ([31], [68] are tagged `V2`). No retraction flags were found.
- English-language only. A Chinese-language cherry-detection literature exists, mostly YOLO variants in *Trans. CSAE*, and was not screened.
- The SAM 3 literature is moving fast (preprints from December 2025 to September 2026). Re-check before submission.
