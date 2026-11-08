# Literature review: synthetic aperture imaging, light fields and seeing through foreground occluders

*Draft for the AMME5710 major project (has_sai). Prepared 15 October 2026. IEEE citation style, Australian English.*

**Scope.** Synthetic aperture imaging (SAI), camera arrays and light fields, and removing foreground occluders, with an eye to our setup: one iPhone main camera (about 26 mm equivalent, 4032 × 3024 px, f_x ≈ 2900–3000 px) translated along a 0.6 m rail, 12–13 shots 50 mm apart, looking at toy cherries on a timber diamond-lattice fence (about 0.91 m) behind green pipe-cleaner foliage (about 0.57 m). The aim is to remove the occluders without colour thresholding, so the method carries over to green fruit, and to count every cherry once.

**Method (deep-research skill, lit-review mode).** This is a narrative/integrative review, as requested. Search, verification and synthesis follow the skill's bibliography, source-verification and synthesis agents, with two overrides: IEEE citations instead of APA, and Australian English.
- *Indexes searched (15 Oct 2026):* OpenAlex, Semantic Scholar Graph API, CrossRef REST API and the arXiv API, plus web search to fill gaps. IEEE Xplore blocks automated access (CloudFront WAF challenge, HTTP 202), so the two tutor-supplied document numbers were resolved through CrossRef deposit metadata instead (see §A).
- *Key search strings:* "synthetic aperture imaging" AND (occlusion OR occluded OR "all-in-focus" OR "camera selection" OR "pixel labeling"); "airborne optical sectioning"; "light field" AND ("occlusion removal" OR "de-occlusion"); "see through" AND (obstruction OR foliage); NeRF/Gaussian splatting AND (occlusion removal OR fruit counting); (fruit OR orchard OR plant) AND ("light field" OR "camera array" OR "multi-view" OR plenoptic OR "synthetic aperture"); plus the named classics. Records were screened on title and abstract first, then on full text where it was open.
- *Verification (Tier 0/1):* every reference below was resolved by DOI in CrossRef, with the title cross-checked (normalised Levenshtein similarity ≥ 0.70), and also matched in OpenAlex and Semantic Scholar unless noted otherwise. The arXiv preprints, whose DOIs are registered with DataCite and so are absent from CrossRef, were matched in OpenAlex and Semantic Scholar. Volume, issue and page data come from the CrossRef deposits. Anything not confirmed this way is marked **UNVERIFIED**.
- *Read scope:* "full text" means the authors' open PDF was read (arXiv, CVF Open Access, the authors' own sites); "abstract" means the summary relies on the indexed abstract only. Numbers are quoted only from what was read.
- *AI disclosure:* the searching, verification scripts and drafting were done with an AI assistant (Claude). The team must check every claim against the sources before submission.

---

## A. The two tutor-supplied IEEE Xplore papers

### A1. IEEE Xplore document 1641039: Vaish *et al.*, CVPR 2006

**Reference.** V. Vaish, R. Szeliski, C. L. Zitnick, S. B. Kang, and M. Levoy, "Reconstructing occluded surfaces using synthetic apertures: Stereo, focus and robust measures," in *Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. (CVPR'06)*, New York, NY, USA, 2006, vol. 2, pp. 2331–2338, doi: 10.1109/CVPR.2006.244.

**How it was identified.** The CrossRef record for DOI 10.1109/CVPR.2006.244 gives the primary resource URL `https://ieeexplore.ieee.org/document/1641039/` and a full-text link ending `01641039.pdf?arnumber=1641039`, which confirms document 1641039. OpenAlex and Semantic Scholar both match (title similarity 1.0; OpenAlex counted about 225 citations on 15 Oct 2026). The authors' final version, at graphics.stanford.edu/papers/sap-recons/final.pdf, was read in full. *Author order:* the PDF byline is Vaish, Szeliski, Zitnick, Kang, Levoy, whereas the IEEE/CrossRef metadata lists Vaish, Levoy, Szeliski, Zitnick, Kang. Cite the byline order shown above.

**Problem.** Most reconstruction costs (SSD, SAD, NCC) assume every camera sees the surface, so they break down when a surface is hidden in many views. With a synthetic aperture wider than the occluders, enough rays pass around them. The paper asks which cost function best exploits that.

**Method.** All costs are built in Collins' plane-sweep framework [58]. For each candidate depth (disparity) *d*, the *N* images are warped onto the plane, giving a disparity-space image *D*(*x*, *y*, *d*), and depth is taken as the per-pixel argmin. Four costs are compared:
1. **Stereo:** the variance of the *N* warped rays.
2. **Synthetic-aperture focus:** the negative squared spatial gradient of the mean (refocused) image.
3. **Median:** the median absolute deviation of the rays from their component-wise median colour.
4. **Entropy:** the Shannon entropy of the ray-colour histogram (16 bins per channel, RGB cubes for colour). This penalises how many distinct bins the rays occupy rather than how far apart their colours are, and its precision is O(1/*N*).

Theorem 1 proves that stereo can recover surfaces whose radiance has vanishing second- and higher-order derivatives, whereas focus needs non-zero third-order derivatives. Each method's colour estimate (mean, median or modal colour) is projected into the central camera as a "winner colour image".

**Key results (full text).**
- *Synthetic two-plane scene:* 81 views on a randomly perturbed 9 × 9 grid, 40 px of disparity between the planes across the aperture, and occluding bars covering 19–75 % of the background (white-noise, pink-noise or uniform texture). Entropy did best, with near-perfect reconstruction up to about 65 % occlusion. The median failed once occlusion passed about 50 %. Focus recovered about 15 % more background pixels than stereo.
- *Real "CD case behind plants" light field:* a single camera on a computer-controlled gantry, 21 × 5 = 105 images, aperture 60 cm × 10 cm. Ground-truth occlusion came from differencing against an occluder-free capture, and accuracy meant within one disparity level. All costs degraded as occlusion rose. Entropy was best, the median failed above 50 %, and focus overtook the stereo-type costs above about 60 % occlusion. Voxel colouring had no single threshold that worked.
- *Dense ivy wall:* an 88-camera array, about 70 % occlusion and a uniformly coloured occluder. Entropy, median and voxel colouring all failed, and focus recovered slightly more (a face, a torch) than stereo.

**Stated limitations.** There is no spatial regularisation (§5). The search is restricted to depths behind the occluders, and occluders are never reconstructed (§1, §4). The authors call for a theory linking occluder size, fill factor and texture to aperture size and number of views (§5). Kurmi *et al.* [26] later supplied that theory.

**Method weaknesses** (empirical; read: full text). *Checked, found:* evaluation design, generalisation. *Checked, none found:* statistical testing. Weaknesses:
- The evaluation scenes use about 81–105 views, so the conclusions about median and entropy may not hold at *N* ≈ 13, where a 16-bin histogram holds at most 13 samples. [reader-inferred]
- The uniform-colour occluder is the case where median and entropy fail. [author-acknowledged, section: 3.2/4.3]

**Relevance to our rail and cherry setup.**
- This is the single closest classical precedent. Experiment 2 is a moving single camera over a 60 cm aperture, essentially our rail, though sampled on a 2D 21 × 5 grid rather than with 12–13 views on a line.
- The four costs map directly onto the plane-sweep stage of our pipeline.
- The paper's main failure case is our scene: a uniformly coloured (green) occluder covering more than half the views of some fruit points. That motivates our step beyond robust statistics, which is explicit per-view visibility: decide geometrically which rays hit foliage and exclude them, instead of hoping the occluder rays are a minority.

### A2. IEEE Xplore document 11628076: Ma *et al.*, IEEE TIP 2026 ("UncNeRF")

**Reference.** J. Ma, Q. Wang, Y. Zhao, and J. Li, "UncNeRF: Uncovering heavily occluded object with multi-view clues," *IEEE Trans. Image Process.*, vol. 35, pp. 8260–8274, 2026, doi: 10.1109/TIP.2026.3716400.

**How it was identified.** Searching by the number alone found nothing. CrossRef deposits for IEEE content carry the Xplore article number in their resource and full-text link fields, so CrossRef records with the 10.1109 prefix published from 2025 onwards were scanned for "11628076" across several topic queries. The match was DOI 10.1109/TIP.2026.3716400: resource `https://ieeexplore.ieee.org/document/11628076/`, link `.../11628076.pdf?arnumber=11628076`. It was confirmed in Semantic Scholar (online 29 July 2026; DBLP key journals/tip/MaWZL26; PubMed 42525947) and in OpenAlex (W7171623554). All authors are at Beihang University; the funders listed include NSFC grant 62132002.

**Read scope.** Abstract only: the paper is closed access (Unpaywall and OpenAlex both report "closed", and no arXiv version was found). The summary below also draws on the authors' public code and dataset and on the 56-item reference list deposited in CrossRef. Those additions are labelled as such, and numerical results are **UNVERIFIED** because the paper's tables could not be read.

**Method (from the abstract).**
- *Problem:* NeRF reconstructions of a target object hidden behind a heavy foreground occluder. The authors argue that prior removal methods rely on external 2D visual priors that are not constrained to be 3D-consistent for the specific scene. (The reference list suggests these are mainly 2D inpainters and diffusion priors.)
- *Approach:* UncNeRF instead mines "multi-view clues" from the captured, defective images.
  1. Object-centric forward warping supplies complementary supervision: target pixels seen in one view are warped into the views where they are hidden.
  2. Pseudo-training views are sampled.
  3. An external spatial-relation regulariser strengthens the target reconstruction.
- *Dataset:* the authors introduce the Heavy Occlusion Removal (HOR) dataset, with synthetic and real scenes whose targets are heavily occluded.
- *Results:* the abstract reports state-of-the-art heavy-occlusion removal against other methods. Specific PSNR, SSIM and LPIPS values are UNVERIFIED (full text not accessible).

**Observations from the public code and dataset (not the paper; reader-inferred).** The repository is github.com/iCVTEAM/UncNeRF, created 11 May 2026, and the dataset is huggingface.co/datasets/jiawei919/UncNeRF_dataset.
- The implementation is a NeRF (64 coarse + 64 fine samples per ray) on LLFF-style forward-facing captures with COLMAP poses. The training folder of the real "car" scene holds 18 images.
- Masks for the occluder and for the occluded object come from Segment Anything (SAM) [54], prompted with point coordinates specified by hand in each scene's configuration file for one reference view. The prompts are propagated to other views by back-projecting with a per-view depth map and reprojecting with the camera poses, then filtered by depth and RGB similarity.
- Optional depth losses push the occluder and target depths apart.

So occluder identification is semantic, prompted and partly manual, not purely geometric.

**Context from the reference list (reader-inferred).** UncNeRF positions itself against NeRF/3DGS inpainting and removal: SPIn-NeRF [52], NeRF object removal, MVIP-NeRF, NeRFiller, InFusion, IMFine, Gaussian-splatting object removal, SpotLessSplats, and the occlusion-free NeRF of Zhu *et al.* [51]. It does not cite any classical synthetic-aperture work (no Vaish, Levoy, Isaksen or AOS). The two tutor papers therefore sit in literatures that do not reference each other, and our project can bridge them. *Aside:* one deposited reference ("SAM2") is auto-linked in CrossRef to an unrelated 1988 yeast-biology DOI. That is a publisher reference-linking error and does not affect our citation.

**Method weaknesses:** not assessed (read scope: abstract only).

**Relevance to our rail and cherry setup.** This is the learned, per-scene-optimised counterpart of our pipeline.
- Its capture regime (a forward-facing sweep of roughly 18–22 images with COLMAP poses) is close to our 12–13 rail images.
- Its central idea, warping pixels that are visible in some views into the views where they are hidden, is what geometric SAI does with per-view visibility.
- The contrast we can draw: UncNeRF needs SAM masks seeded by hand-specified points plus a per-scene NeRF optimisation. Our approach finds the occluders from multi-view photo-consistency and depth, with no prompts, training or colour model.

> **Correction to the brief.** Joshi *et al.*'s "Synthetic aperture tracking" appeared at ICCV 2007, not 2006 (CrossRef and OpenAlex agree; see [11]).

---

## B. Annotated bibliography (grouped by theme)

*Entry format: IEEE reference; verification and read scope; summary; key quantitative result; relevance to our rail and cherry setup. "Our calculation" marks numbers derived from our own capture geometry (f_x ≈ 2950 px, 50 mm steps, leaves ≈ 572 mm, fruit ≈ 912 mm), not from the cited paper.*

### B1. Light fields and classical synthetic aperture imaging

**[3]** M. Levoy and P. Hanrahan, "Light field rendering," in *Proc. 23rd Annu. Conf. Comput. Graph. Interact. Techn. (SIGGRAPH '96)*, Aug. 1996, pp. 31–42, doi: 10.1145/237170.237199.
*Verified: CrossRef, OpenAlex, S2. Read: full text.*
- **Summary:** Treats the images of a static scene as 2D slices of a 4D light field, parameterised by a ray's intersections with two planes (the "light slab"). New views are rendered by resampling the stored rays, with no depth or feature matching. The paper also covers antialiasing by prefiltering with a finite aperture, and compression.
- **Key result:** Light fields compressed by more than 100:1 with little loss of fidelity.
- **Relevance:** Supplies the vocabulary for our capture. Our 13 rail images are a 3D slice of the light field (one camera axis by two image axes), and refocusing is resampling it on a chosen focal plane.

**[4]** S. J. Gortler, R. Grzeszczuk, R. Szeliski, and M. F. Cohen, "The lumigraph," in *Proc. 23rd Annu. Conf. Comput. Graph. Interact. Techn. (SIGGRAPH '96)*, Aug. 1996, pp. 43–54, doi: 10.1145/237170.237200.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** The companion to [3]. It samples and reconstructs a 4D subset of the plenoptic function (the Lumigraph) to render new views of real and synthetic objects independently of geometric complexity, and describes a full capture, construction and rendering system.
- **Key result:** Rendering speed independent of scene complexity (qualitative, from the abstract).
- **Relevance:** A foundational citation alongside [3]. Our view of each fruit is likewise a 4D-to-3D ray set.

**[5]** A. Isaksen, L. McMillan, and S. J. Gortler, "Dynamically reparameterized light fields," in *Proc. 27th Annu. Conf. Comput. Graph. Interact. Techn. (SIGGRAPH '00)*, 2000, pp. 297–306, doi: 10.1145/344779.344929.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Lets the focal surface and aperture of a light field move interactively, so variable focus and depth of field can be rendered without any scene geometry. The paper analyses this reparameterisation in the frequency domain and in ray space, and also gives a passive autostereoscopic display.
- **Key result:** Interactive variable-focus rendering of moderately sampled light fields with significant, unknown depth variation (qualitative).
- **Relevance:** The theoretical root of shift-and-add refocusing, which is what our teammate's baseline does. Our per-view visibility weights generalise Isaksen's aperture filter: each fruit pixel gets its own aperture that leaves out the views blocked by foliage.

**[6]** J.-X. Chai, S.-C. Chan, H.-Y. Shum, and X. Tong, "Plenoptic sampling," in *Proc. 27th Annu. Conf. Comput. Graph. Interact. Techn. (SIGGRAPH '00)*, 2000, pp. 307–318, doi: 10.1145/344779.344932.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Spectral analysis shows that a light field's spectral support is bounded by the scene's minimum and maximum depths alone. This sets the minimum sampling rate and an optimal constant depth for the reconstruction filter. The "minimum sampling curve" trades the number of images against the amount of geometry known.
- **Key result:** An analytical minimum sampling rate depending only on z_min and z_max.
- **Relevance:** In our capture, foliage moves about 258 px and fruit about 162 px per 50 mm step (our calculation). That is orders of magnitude above the roughly one-pixel-per-view anti-aliasing condition, so our light field is severely undersampled. Out-of-focus leaves will therefore appear as 13 discrete, faint ghost copies rather than a smooth blur. Following Chai's curve, depth knowledge (geometry) has to make up for the missing views.

**[7]** V. Vaish, B. Wilburn, N. Joshi, and M. Levoy, "Using plane + parallax for calibrating dense camera arrays," in *Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Washington, DC, USA, 2004, vol. 1, pp. 2–9, doi: 10.1109/CVPR.2004.1315006.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** For cameras lying on a plane, camera positions can be recovered up to an affine ambiguity. Images can then be reprojected onto a family of planes using only the planar parallax of a single scene point, which is all that synthetic-aperture focusing needs; no metric calibration is required.
- **Key result:** Better synthetic-aperture results than full metric calibration on real light fields (qualitative).
- **Relevance:** After a homography to a reference plane (the fence), refocusing for collinear rail cameras is a 1D shift proportional to rail position, and one cherry's parallax fixes the shift per depth. Phone rotation and rail wobble break the planar assumption, which justifies our full-pose (COLMAP) plane sweep.

**[8]** V. Vaish *et al.*, "Synthetic aperture focusing using a shear-warp factorization of the viewing transform," in *Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. (CVPR'05) Workshops*, San Diego, CA, USA, 2005, vol. 3, p. 129, doi: 10.1109/CVPR.2005.537.
*Verified: CrossRef, OpenAlex, S2. Read: full text.*
- **Summary:** Analyses the warps needed to focus on tilted planes with arbitrary camera layouts, using a rank-1 constraint that avoids metric calibration. For some configurations, a tilted focal plane factorises into an initial homography followed by shifts. This "shear-warp" makes tilted planes as cheap to synthesise as frontoparallel ones.
- **Key result:** Real-time synthetic-aperture focusing on an array of 30 video-resolution cameras, with homographies on per-camera FPGAs and the summing on 3 PCs.
- **Relevance:** Our fence and foliage layers need not be parallel to the rail. A homography-then-shift model is the cheap way to sweep tilted planes, and it is a special case of our full-pose plane sweep.

**[9]** B. Wilburn *et al.*, "High performance imaging using large camera arrays," *ACM Trans. Graph.*, vol. 24, no. 3, pp. 765–776, Jul. 2005, doi: 10.1145/1073204.1073259.
*Verified: CrossRef, OpenAlex, S2. Read: full text.*
- **Summary:** Describes the Stanford array of 100 custom video cameras and its uses: high resolution, high dynamic range, high frame rate, large-aperture (synthetic aperture) video, and spatiotemporal view interpolation. The authors note that their view interpolation does not reason about occlusions.
- **Key result:** VGA video at 30 fps from 100 cameras to four PCs. Measured effective resolution was about 400 px horizontally per camera.
- **Relevance:** The canonical see-through-foliage hardware. It sets the scale of the classical results (around 100 views) against which our 13-view rail has to be judged.

**[10]** M. Levoy, "Light fields and computational imaging," *Computer*, vol. 39, no. 8, pp. 46–55, Aug. 2006, doi: 10.1109/MC.2006.270.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** A survey of light-field theory and practice, covering capture devices (camera arrays, gantries, plenoptic cameras) and the techniques for computing new images from them.
- **Key result:** None (survey).
- **Relevance:** A concise background citation for the report's introduction.

### B2. Occlusion-aware, visibility-labelled synthetic aperture imaging

**[11]** N. Joshi, S. Avidan, W. Matusik, and D. J. Kriegman, "Synthetic aperture tracking: Tracking through occlusions," in *Proc. IEEE 11th Int. Conf. Comput. Vis. (ICCV)*, Rio de Janeiro, Brazil, 2007, pp. 1–8, doi: 10.1109/ICCV.2007.4409032. *(The brief gave 2006; the indexed year is 2007.)*
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Tracks a 3D object through heavy, prolonged occlusion with a camera array. Because occluder and target sit at different depths, parallax exposes different parts of the target in each view, and aggregating across views recovers it. The method builds on standard single-camera trackers and needs no explicit scene reconstruction.
- **Key result:** Tracking succeeds when upwards of 70 % of the object is occluded in every camera view.
- **Relevance:** A direct precedent for running detection on synthetic-aperture or focal-stack images instead of on single views. The 70 % per-view occlusion figure is comparable to our worst-occluded cherries.

**[12]** Z. Pei, Y. Zhang, X. Chen, and Y.-H. Yang, "Synthetic aperture imaging using pixel labeling via energy minimization," *Pattern Recognit.*, vol. 46, no. 1, pp. 174–187, Jan. 2013, doi: 10.1016/j.patcog.2012.06.014.
*Verified: CrossRef, OpenAlex, S2. Read: abstract (from the NWPU repository record).*
- **Summary:** Replaces SAI's average over all views with a labelling problem: graph-cut energy minimisation labels every pixel in every view as occluder or not, and refocusing averages only the unlabelled pixels. According to DeOccNet's review [44], the labels come from stereo matching, and the output is focused at one depth only.
- **Key result:** Outperforms standard SAI and its variants, which only dim and blur occluders. The authors claim the first such method needing no training set and no static-scene assumption. No numbers are given in the abstract.
- **Relevance:** The closest classical analogue of our per-view geometric visibility masks. We extend it with a full-pose plane sweep, all-in-focus fusion across depths, and counting.

**[13]** T. Yang, Y. Zhang, X. Tong, X. Zhang, and R. Yu, "A new hybrid synthetic aperture imaging model for tracking and seeing people through occlusion," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 23, no. 9, pp. 1461–1475, Sep. 2013, doi: 10.1109/TCSVT.2013.2242553.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Combines detection, imaging and tracking in one multi-camera synthetic-aperture framework. People are localised in a 3D foreground-silhouette synthetic-aperture volume using multiple constraints that remove "ghost" objects, and imaged with an occluder-removal step. Tracking then runs in colour synthetic-aperture images.
- **Key result:** Reliable localisation and imaging of people in crowded, occluded scenes (qualitative and quantitative evaluation; numbers not in the abstract).
- **Relevance:** Ghost removal in a 3D synthetic-aperture volume is exactly the problem of de-duplicating detections across focal slices when we count cherries in 3D.

**[14]** T. Yang, Y. Zhang, X. Tong, W. Ma, and R. Yu, "High performance imaging through occlusion via energy minimization-based optimal camera selection," *Int. J. Adv. Robot. Syst.*, vol. 10, no. 11, Art. no. 393, Nov. 2013, doi: 10.5772/57175.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Recasts occluded-object imaging as choosing, for each pixel, the visible ray from the best camera view, which works with a camera array or a single moving camera. Visibility is propagated greedily from front to back through the depth planes. On each plane, a multi-label energy (multi-view intensity consistency, previous visibility and camera-view smoothness) is minimised with graph cuts.
- **Key result:** Higher imaging quality than standard SAI and earlier labelling methods (qualitative in the abstract).
- **Relevance:** Close to our design: a single moving camera, explicit visibility propagated from the foliage layer back to the fruit layer, and ray selection rather than averaging.

**[15]** T. Yang *et al.*, "All-in-focus synthetic aperture imaging," in *Computer Vision – ECCV 2014* (Lecture Notes in Computer Science). Cham, Switzerland: Springer, 2014, pp. 1–15, doi: 10.1007/978-3-319-10599-4_1.
**[16]** T. Yang *et al.*, "Multiple-layer visibility propagation-based synthetic aperture imaging through occlusion," *Sensors*, vol. 15, no. 8, pp. 18965–18984, Aug. 2015, doi: 10.3390/s150818965.
*Both verified: CrossRef, OpenAlex, S2. Read: abstracts. [16] is the extended journal version of [15].*
- **Summary:** A "depth-free" all-in-focus SAI method. The scene is partitioned into visibility layers, and visibility is propagated from layer to layer. On each layer, visibility and best focus depth are solved as a multi-label energy combining earlier layers' visibility masks, multi-view intensity consistency and depth smoothness.
- **Key result:** Better than prior SAI and inpainting approaches (qualitative and quantitative; numbers not in the abstracts). A caution: a 2026 preprint [40] lists values for "All-in-Focus [59]" (this paper) that are identical to the values DeOccNet [44] reported for Pei *et al.* [20]. We therefore cite no figure for [15]/[16].
- **Relevance:** The nearest prior art to our "all-in-focus occluder-free reconstruction" stage, using layered visibility with no colour model.

**[17]** T. Yang, W. Ma, S. Wang, J. Li, J. Yu, and Y. Zhang, "Kinect based real-time synthetic aperture imaging through occlusion," *Multimedia Tools Appl.*, vol. 75, no. 12, pp. 6925–6943, Jun. 2016, doi: 10.1007/s11042-015-2618-1.
*Verified: CrossRef, OpenAlex, S2. Read: abstract (from the NWPU repository record).*
- **Summary:** A handheld colour-plus-depth (Kinect) system that builds a "virtual camera array" from a moving sensor, refining poses with colour and depth features. It generates object-probability maps and performs sequential synthetic-aperture imaging in real time.
- **Key result:** The authors claim the first real-time handheld system for seeing occluded objects with colour and depth, more robust and efficient than prior algorithms (qualitative).
- **Relevance:** Depth-aware occluder handling with an active depth sensor. It points to an alternative we could test with the iPhone's LiDAR, whereas our main method obtains depth passively from the image sweep.

**[18]** X. Zhang, Y. Zhang, T. Yang, and Y.-H. Yang, "Synthetic aperture photography using a moving camera-IMU system," *Pattern Recognit.*, vol. 62, pp. 175–188, Feb. 2017, doi: 10.1016/j.patcog.2016.07.019.
*Verified: CrossRef, OpenAlex, S2. Read: abstract summary (NWPU repository record).*
- **Summary:** Replaces the camera array with a handheld camera and IMU moving over a plane. The refocusing transform factorises into an infinite homography, estimated from the IMU rotation, followed by a shift, so no scene-specific calibration is needed.
- **Key result:** See-through results "with reasonable accuracy" across several scenes (no numbers available to us).
- **Relevance:** This is our rail geometry with a phone: a rotation-compensating homography plus a translation-proportional shift. The iPhone's IMU could supply the rotation term if COLMAP pose estimation is disrupted by the occluders.

**[19]** Z. Xiao, L. Si, and G. Zhou, "Seeing beyond foreground occlusion: A joint framework for SAP-based scene depth and appearance reconstruction," *IEEE J. Sel. Topics Signal Process.*, vol. 11, no. 7, pp. 979–991, Oct. 2017, doi: 10.1109/JSTSP.2017.2715012.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Reconstructs depth and appearance behind foreground occluders when no single reference view sees the whole target. Optimal views are selected per pixel by clustering ray colours (k-means, per [44]), then refined coarse-to-fine in a global optimisation.
- **Key result:** Accurate depth and appearance even when every view is partly occluded; outperforms prior methods (qualitative).
- **Relevance:** Matches our goal (depth plus appearance behind foliage). Its view selection uses colour clustering of rays, which is still appearance-based. Our photo-consistency-plus-depth visibility avoids assuming the occluder and target colours separate.

**[20]** Z. Pei, X. Chen, and Y.-H. Yang, "All-in-focus synthetic aperture imaging using image matting," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 28, no. 2, pp. 288–301, Feb. 2018, doi: 10.1109/TCSVT.2016.2615328.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Builds an all-in-focus "see-through" image in four steps:
  1. Estimate the out-of-focus regions by focusing on the background in each view.
  2. Label pixels to form a sharp synthetic-aperture image of the hidden object.
  3. Extract the hidden object's alpha matte.
  4. Composite it with the reconstructed background.
- **Key result:** As reported in DeOccNet's comparison [44] (secondary), mean PSNR 19.16 dB and SSIM 0.587 over the Stanford CD scene and four synthetic scenes, against 22.45 dB and 0.727 for DeOccNet.
- **Relevance:** A template for our final compositing stage. It also gives a classical-method baseline number for the report.

**[21]** J. Martínez Sotoca, P. Latorre-Carmona, F. Pla, and B. Javidi, "Depth and all-in-focus image estimation in synthetic aperture integral imaging under partial occlusions," *IEEE Access*, vol. 7, pp. 1052–1067, 2019, doi: 10.1109/ACCESS.2018.2886235.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** In synthetic-aperture integral imaging, a median-based photo-consistency measure over the elemental images detects which cameras actually see a partly occluded object at a given depth. Detected occlusion boundaries then guide regularised depth estimation and all-in-focus synthesis.
- **Key result:** Outperforms other depth-estimation methods in the integral-imaging setting (qualitative).
- **Relevance:** Robust consistency used to decide per-camera visibility. This is the integral-imaging counterpart of our approach, and a citation showing the optics community reached the same idea.

**[22]** Z. Pei *et al.*, "Occluded-object 3D reconstruction using camera array synthetic aperture imaging," *Sensors*, vol. 19, no. 3, Art. no. 607, Jan. 2019, doi: 10.3390/s19030607.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Labels occlusion pixels by ray variance and reconstructs a 3D point cloud of the occluded object from synthetic-aperture images. Accuracy is assessed as the spatial difference between reconstructions with and without the occluder.
- **Key result:** Handles occlusion well (qualitative in the abstract).
- **Relevance:** Variance-based occluder labelling, which is colour-free, leading to 3D points of hidden objects. This is the representation we need for counting each cherry once in 3D.

**[23]** Z. Pei, M. Jin, Y. Zhang, M. Ma, and Y.-H. Yang, "All-in-focus synthetic aperture imaging using generative adversarial network-based semantic inpainting," *Pattern Recognit.*, vol. 111, Art. no. 107669, Mar. 2021, doi: 10.1016/j.patcog.2020.107669.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Extends [20] with GAN inpainting for regions that no camera captured, compositing the inpainted hidden object with the reconstructed background.
- **Key result:** Handles heavy occlusion and generates occlusion labels quickly and without noise, compared with earlier labelling methods (qualitative).
- **Relevance:** Shows that points hidden in every view need a learned prior. For counting, we should flag such regions as unobserved rather than hallucinate fruit.

### B3. Airborne Optical Sectioning (AOS) and synthetic-aperture sensing through foliage (Bimber group, JKU Linz)

**[24]** I. Kurmi, D. C. Schedl, and O. Bimber, "Airborne optical sectioning," *J. Imaging*, vol. 4, no. 8, Art. no. 102, Aug. 2018, doi: 10.3390/jimaging4080102.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Introduces AOS: a low-cost camera drone records unstructured video over a wide synthetic aperture, and the registered frames are integrated (averaged) on a chosen focal surface. Occluders are suppressed by the extremely shallow depth of field instead of being reconstructed in 3D. The authors argue AOS avoids photogrammetry's correspondence errors and long processing, and is cheaper than LiDAR while keeping colour.
- **Key result:** Synthetic apertures of 30–100 m diameter sampled by a single drone.
- **Relevance:** The "unstructured light field from one moving camera" idea behind our rail, and the argument for integrating images over reconstructing a point cloud of the occluders.

**[25]** O. Bimber, I. Kurmi, and D. C. Schedl, "Synthetic aperture imaging with drones," *IEEE Comput. Graph. Appl.*, vol. 39, no. 3, pp. 8–15, May 2019, doi: 10.1109/MCG.2019.2896024.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** A magazine-length overview of AOS, demonstrated by revealing the ruins of a 19th-century fortification hidden under dense forest.
- **Key result:** Qualitative. The scan is the 504-image, 100 m aperture example analysed in [26].
- **Relevance:** An accessible citation that presents SAI as a sensing technique, not just a graphics trick.

**[26]** I. Kurmi, D. C. Schedl, and O. Bimber, "A statistical view on synthetic aperture imaging for occlusion removal," *IEEE Sensors J.*, vol. 19, no. 20, pp. 9374–9383, Oct. 2019, doi: 10.1109/JSEN.2019.2922731.
*Verified: CrossRef, OpenAlex, S2. Read: full text (arXiv:1906.06600).*
- **Summary:** Models SAI as signal averaging, with occluders treated as random binary noise: opaque cubes of size *o* at per-slice density *D* in a volume of depth *l*. A single view's integrated occlusion density is D̃ = 1 − (1 − D)^(l/o). Averaging *N* views shifted by disparity *d* gives a closed-form visibility. When the normalised disparity d̂ = d/o ≥ 1, this simplifies to V = 1 − D̃² − D̃(1 − D̃)/N.
- **Key results:**
  - Visibility stops improving once the disparity between views equals the projected occluder size; larger baselines only add "bokeh" artefacts of the sampling grid.
  - Visibility is bounded between 1 − D̃ (one view) and 1 − D̃² (infinitely many views), and the gain is largest at D̃ = 50 %. Even infinite sampling leaves a constant occluder bias.
  - Normalised visibility V̂ = 1 − 1/N, regardless of density. So V̂ = 98 % needs N ≈ 50.
  - On a LiDAR forest and real AOS data, apertures 6.5–12.75 times smaller in area and 10–20 times fewer samples gave no significant visual loss (for example, a 14 m aperture with N = 25 matched a 50 m aperture with N = 504 in SSIM).
- **Relevance:** This is the quantitative design theory for our rail (our calculations):
  - Foliage moves about 96 px per step relative to the fruit plane. A 5–10 mm pipe-cleaner strand at 0.57 m projects to about 26–52 px, so d̂ ≈ 2–4 ≥ 1 and a 50 mm step is already at or past the optimal baseline.
  - With N = 13, the model's normalised visibility gain is about 92 %.
  - Mean shift-and-add can never remove the green bias D̃ (the averaged occluders contribute D̃ everywhere). That is the core argument for excluding occluded rays rather than averaging them.
  - The model assumes 2D sampling and random, isotropic occluders. With 1D sampling, horizontally elongated occluders decorrelate poorly; vertical pipe cleaners are the favourable case.

**[27]** I. Kurmi, D. C. Schedl, and O. Bimber, "Thermal airborne optical sectioning," *Remote Sens.*, vol. 11, no. 14, Art. no. 1668, Jul. 2019, doi: 10.3390/rs11141668.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Integrates thermal images from many viewpoints to reveal heat signatures, such as people under dense shrub, that are invisible in RGB and unidentifiable in single thermal frames. Drone poses are estimated with computer-vision algorithms on the high-resolution RGB images.
- **Key result:** Qualitative recovery of strongly occluded human heat signatures.
- **Relevance:** Estimating poses on one image stream while integrating another is a pattern we can reuse. Poses come from the full-resolution RGB images, while the integrated channel can be any per-view map: visibility masks, fruit-likelihood maps or edge maps.

**[28]** D. C. Schedl, I. Kurmi, and O. Bimber, "Search and rescue with airborne optical sectioning," *Nat. Mach. Intell.*, vol. 2, no. 12, pp. 783–790, Nov. 2020, doi: 10.1038/s42256-020-00261-3.
*Verified: CrossRef, OpenAlex, S2. Read: full text (arXiv:2009.08835).*
- **Summary:** Person detection with a deep detector works much better when the multi-perspective thermal images are integrated before classification than when detections from single images are combined.
- **Key result:** Precision 96 % and recall 93 % (AP 92.2 %; 53 of 57 persons detected) on AOS integral images. Single-image detection was far weaker: the preprint reports precision/recall from 0.48/0.44 to 0.90/0.24 depending on augmentation.
- **Relevance:** Supports our "detect on the refocused or focal-stack image" stage rather than detecting in each raw view.

**[29]** D. C. Schedl, I. Kurmi, and O. Bimber, "Airborne optical sectioning for nesting observation," *Sci. Rep.*, vol. 10, Art. no. 7254, Apr. 2020, doi: 10.1038/s41598-020-63317-9.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Images registered to a common 3D frame let ornithologists identify, classify and localise birds and nests through the canopy.
- **Key result:** In a seven-minute flight over a 40 × 12 m patch, 65 herons and 27 nests were identified and localised in a sparse 3D reconstruction.
- **Relevance:** A biology and ecology counting application of AOS: localising and counting objects in 3D through foliage, which mirrors our cherry count.

**[30]** D. C. Schedl, I. Kurmi, and O. Bimber, "An autonomous drone for search and rescue in forests using airborne optical sectioning," *Sci. Robot.*, vol. 6, no. 55, Art. no. eabg1188, Jun. 2021, doi: 10.1126/scirobotics.abg1188.
*Verified: CrossRef, OpenAlex, S2. Read: full text (arXiv:2105.04328).*
- **Summary:** A fully autonomous drone integrates and classifies on board, adapting its flight path to double-check candidate findings.
- **Key result:** 38 of 42 hidden persons found across 17 field experiments, with AP 86 % on predefined paths; adaptive planning raised confidence by 15 %. Classification was unaffected by sparse, error-prone sampling within **one-dimensional** synthetic apertures, which needed about one-tenth the images of 2D apertures.
- **Relevance:** The strongest published evidence that a 1D aperture (our rail) with modest sample counts is enough for detection behind foliage.

**[31]** I. Kurmi, D. C. Schedl, and O. Bimber, "Combined person classification with airborne optical sectioning," *Sci. Rep.*, vol. 12, Art. no. 3804, Mar. 2022, doi: 10.1038/s41598-022-07733-z.
*Verified: CrossRef, OpenAlex, S2. Read: full text (arXiv:2106.10077).*
- **Summary:** Combining classifications from several strongly overlapping integral images suppresses false detections and boosts true ones, especially under occlusion. The imaging was modified for real-time on-board use at ground speeds up to 10 m/s.
- **Key result:** The authors report a baseline AP of 88.7 % without combination. The combined-classification gains were not extracted for this review.
- **Relevance:** Decision-level fusion across several refocused images, here across focal slices, is the analogue of fusing detections across our focal stack before 3D de-duplication.

**[32]** F. Seits, I. Kurmi, R. J. A. A. Nathan, R. Ortner, and O. Bimber, "On the role of field of view for occlusion removal with airborne optical sectioning," arXiv:2204.13371, Apr. 2022, doi: 10.48550/arXiv.2204.13371. *(Preprint; no peer-reviewed version found in CrossRef.)*
*Verified: OpenAlex, S2 (an arXiv DataCite DOI, so absent from CrossRef). Read: full text.*
- **Summary:** Uses a procedural forest simulator to show that a wider field of view (FOV) lowers occlusion-removal efficiency, because oblique rays cross more of the occluder volume. The optimal FOV is not simply "as narrow as possible", though: it depends on tree height and spacing. Smaller sampling distances (more samples) always helped.
- **Key result:** Optimal FOVs of about 46–70° for the simulated forests; a 50° default crop was adopted in the group's DJI app.
- **Relevance:** The iPhone main camera's horizontal FOV is about 69° (our calculation, from f_x and image width). Pixels near the frame edges view the leaf-to-fruit gap more obliquely, so restricting analysis to the central field may improve visibility.

**[33]** R. J. A. A. Nathan, I. Kurmi, D. C. Schedl, and O. Bimber, "Through-foliage tracking with airborne optical sectioning," *J. Remote Sens.*, vol. 2022, Art. no. 9812765, 2022, doi: 10.34133/2022/9812765.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** A light-weight, drone-carried **1D camera array** performs parallel synthetic-aperture imaging, so moving people can be detected and tracked through dense forest.
- **Key result:** Colour anomaly detection reached 97 % precision on integral images versus 42 % on raw frames (field-experiment average).
- **Relevance:** A 1D linear array, our rail's geometry, with a large measured precision gain from integration.

**[34]** R. J. A. A. Nathan and O. Bimber, "Synthetic aperture anomaly imaging for through-foliage target detection," *Remote Sens.*, vol. 15, no. 18, Art. no. 4369, Sep. 2023, doi: 10.3390/rs15184369.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Running anomaly detection on each single aerial image and integrating the results works better than running anomaly detection on the integral image.
- **Key result:** About 20 % higher target visibility and precision on average.
- **Relevance:** Shows that the order of "detect" and "integrate" matters and depends on the detector; see the synthesis (§C). It supports integrating per-view evidence maps, such as visibility masks, rather than only raw pixels.

**[35]** R. J. A. A. Nathan, I. Kurmi, and O. Bimber, "Drone swarm strategy for the detection and tracking of occluded targets in complex environments," *Commun. Eng.*, vol. 2, Art. no. 55, Aug. 2023, doi: 10.1038/s44172-023-00104-0.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** An adaptive particle-swarm strategy positions drones to maximise target visibility through forest, instead of sampling blindly.
- **Key result:** 72 % target visibility within 14 s, versus 51 % after 75 s (sequential brute force) and 19 % in 3 s (parallel sampling), in simulation.
- **Relevance:** Adaptive viewpoint choice beats fixed grids. A possible extension is a non-uniform rail schedule concentrated where occlusion is densest.

**[36]** R. J. A. A. Nathan, S. Strand, D. Shutin, and O. Bimber, "Reciprocal visibility for guided occlusion removal with drones," *IEEE Geosci. Remote Sens. Lett.*, vol. 21, pp. 1–5, 2024, doi: 10.1109/LGRS.2024.3451486.
*Verified: CrossRef, OpenAlex, S2. Read: full text (arXiv:2402.06991).*
- **Summary:** A pre-scanned LiDAR point cloud gives each candidate drone pose a binary visibility mask, computed by keeping only projected points on or near the ground. By Helmholtz reciprocity this is turned into a map of where each ground point is visible from, which drives a greedy sampling plan.
- **Key result:** 5–20 % higher visibility with 9–17 times fewer samples than unguided sampling.
- **Relevance:** The closest conceptual match to our per-view geometric visibility (depth tells each view which rays reach the target layer). AOS gets the depth from LiDAR; we get it from the image sweep itself.

**[37]** R. Kerschner, R. J. A. A. Nathan, R. K. Mantiuk, and O. Bimber, "Stereoscopic depth perception through foliage," *Sci. Rep.*, vol. 14, Art. no. 23056, Oct. 2024, doi: 10.1038/s41598-024-74666-0.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** People could not judge the depth of objects under foliage from monoscopic drone video (motion parallax) or from ordinary stereo video, because the foliage occludes too much. Computational stereo matching also failed. Synthetic-aperture integration followed by disparity-scaled stereo let observers succeed.
- **Key result:** Depth discrimination succeeded only with synthetic-aperture-integrated, disparity-scaled stereo (user study; qualitative in the abstract).
- **Relevance:** Independent evidence that plain stereo or MVS fails behind foliage. Depth of the occluded layer has to be estimated with occlusion-aware or integrated costs, which is our plane-sweep design.

**[38]** M. Youssef, J. Peng, and O. Bimber, "DeepForest: Sensing into self-occluding volumes of vegetation with aerial imaging," *J. Remote Sens.*, vol. 5, Art. no. 0907, 2025, doi: 10.34133/remotesensing.0907. *(Published version of arXiv:2502.02171.)*
*Verified: CrossRef, OpenAlex (journal DOI); S2 (arXiv-linked record). Read: abstract.*
- **Summary:** Synthetic-aperture focal stacks from drones are corrected with pre-trained 3D CNNs that remove out-of-focus contributions, giving volumetric reflectance stacks through the canopy.
- **Key result:** About seven-fold average improvement (minimum about ×2, maximum about ×12) over uncorrected stacks for 220–1,680 trees/ha in simulation, and an MSE of 0.05 against the top layer in a field test.
- **Relevance:** A learned way to remove the residual defocused-occluder signal from a focal stack. It is relevant if our focal-stack detector suffers from leaf ghosts.

**[39]** O. Bimber *et al.*, "How robot dogs see the unseeable: Improving visual interpretability via peering for exploratory robots," *Sci. Robot.*, vol. 11, no. 118, Art. no. eaed8577, Sep. 2026, doi: 10.1126/scirobotics.aed8577.
*Verified: CrossRef, OpenAlex; S2 (arXiv-linked record, PubMed 42748212). Read: full text (arXiv:2511.16262).*
- **Summary:** A quadruped robot "peers", moving its camera side to side like an insect, to form a synthetic aperture at ground level through vegetation. A large multimodal model then interprets the integral images. Peering motions included horizontal shifts (for example 82 images over 12 cm, or 151 over 9 cm), a diagonal shift (32 images over 19 cm) and an axial rotation (300 images over about 11 cm). Optional occlusion masks are thresholded from a vegetation index (VDVI, colour-based) or from monocular or video depth networks (Depth Anything v2/v3, Video Depth Anything, Pixel-Perfect Depth), on the assumption that occluders are nearer.
- **Key results:** SfM, NeRF, VGGT and Depth Anything 3 "fail", reconstructing mainly the foreground occluders. SfM and NeRF took 5–6 h and VGGT/DA3 7–15 min. Simple averaging leaves residual occluder blur, which masking reduces. VDVI misses non-vegetation occluders such as a metal fence.
- **Relevance:** The published system closest to ours: a ground-level, lateral 1D camera translation through foliage. The contrasts are that its masks are colour- or monocular-depth-based, whereas ours come from multi-view photo-consistency; and that it uses hundreds of closely spaced images, whereas we have 13 images over 0.6 m.

**[40]** M. Youssef and O. Bimber, "Vision-reasoning-guided occlusion removal from light fields," arXiv:2606.19985v2, Aug. 2026, doi: 10.48550/arXiv.2606.19985. *(Preprint; not peer reviewed.)*
*Verified: OpenAlex, S2 (DataCite DOI, so absent from CrossRef). Read: full text.*
- **Summary:** Light-field integration, with an optional occlusion mask, produces an initial occlusion-suppressed image. A vision-language image model (Gemini 3.1 or Qwen-Image) then restores it as a conditional semantic prior, and several generated hypotheses are averaged to curb hallucination.
- **Key result:** Highest mean SSIM (0.883) on the 4-Syn benchmark of [44], with PSNR 26.91 dB against the best PSNR of 29.75 dB (Mask4D [45]), as listed in the paper's Table I. SSIM saturates at about 20 averaged generations.
- **Note on reliability:** Table I mis-cites Mask4D as an unrelated LiDAR paper, and its "All-in-Focus" row duplicates DeOccNet's values for [20].
- **Relevance:** Shows the field drifting towards generative priors. For counting, generated content is a liability: a hallucinated cherry is a counting error.

### B4. Learned and neural de-occlusion (motion layers, light-field networks, NeRF/3DGS, segmentation)

**[41]** T. Xue, M. Rubinstein, C. Liu, and W. T. Freeman, "A computational approach for obstruction-free photography," *ACM Trans. Graph.*, vol. 34, no. 4, pp. 1–11, Jul. 2015, doi: 10.1145/2766940.
*Verified: CrossRef, OpenAlex, S2. Read: full text.*
- **Summary:** A user takes a short sequence while slightly moving a phone camera. The background and the obstruction (fence, reflection, raindrops) are separated by their different motions: dense per-layer flow fields are initialised by "edge flow" and refined alternately with the layer images, coarse to fine. The output is a clean background, the obstruction layer and its alpha.
- **Key result:** Five images along a horizontal baseline of a few centimetres are usually enough to remove the obstructing layer.
- **Relevance:** Colour-free separation of occluder and background from motion parallax alone, on a phone. Our rail gives much larger, known parallax (about 96 px per step between layers; our calculation), which makes layer assignment easier.

**[42]** Y.-L. Liu, W.-S. Lai, M.-H. Yang, Y.-Y. Chuang, and J.-B. Huang, "Learning to see through obstructions," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Seattle, WA, USA, Jun. 2020, pp. 14203–14212, doi: 10.1109/CVPR42600.2020.01422.
**[43]** Y.-L. Liu, W.-S. Lai, M.-H. Yang, Y.-Y. Chuang, and J.-B. Huang, "Learning to see through obstructions with layered decomposition," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 44, no. 11, pp. 8387–8402, Nov. 2022, doi: 10.1109/TPAMI.2021.3111847.
*Both verified: CrossRef, OpenAlex, S2. Read: [42] full text (arXiv:2004.01180); [43] abstract (journal extension).*
- **Summary:** A learned version of [41]. The method alternates between estimating dense optical flow for the background and obstruction layers and reconstructing each layer from flow-warped frames with a CNN, coarse to fine. Learning absorbs flow errors and violations of brightness constancy, and training on synthetic sequences transfers to real ones.
- **Key result:** Reflection, fence and raindrop removal on real sequences; quantitative comparisons are in the papers (not extracted here).
- **Relevance:** Shows the learned two-layer route. It depends on optical flow, which our large inter-view disparities (150–250 px) would strain, whereas a known-pose plane sweep handles them directly.

**[44]** Y. Wang, T. Wu, J. Yang, L. Wang, W. An, and Y. Guo, "DeOccNet: Learning to see through foreground occlusions in light fields," in *Proc. IEEE Winter Conf. Appl. Comput. Vis. (WACV)*, Snowmass Village, CO, USA, Mar. 2020, pp. 118–127, doi: 10.1109/WACV45572.2020.9093448.
*Verified: CrossRef, OpenAlex, S2. Read: full text (arXiv:1912.04459).*
- **Summary:** The first deep light-field de-occlusion network: an encoder–decoder with residual ASPP takes the camera-array sub-aperture images (5 × 5, or 5 × 15 for the Stanford CD scene) and outputs an occlusion-free centre view. Training data is synthesised by embedding occluder masks into existing light fields. The network learns to separate occluders by disparity discrepancy.
- **Key result:** Mean PSNR 22.45 dB and SSIM 0.727 over the Stanford CD scene and four synthetic scenes (4-Syn). On the same data, Pei *et al.* [20] scored 19.16 dB / 0.587 and single-image inpainting 19.83 dB / 0.651.
- **Relevance:** Defines the standard benchmark (4-Syn, Stanford CD) and the "sparse, camera-array" de-occlusion setting closest to ours. Note, though, that it needs a regular 2D grid and training data, and outputs only the centre view.

**[45]** Y. Li *et al.*, "Mask4D: 4D convolution network for light field occlusion removal," in *Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)*, Toronto, ON, Canada, Jun. 2021, pp. 2480–2484, doi: 10.1109/ICASSP39728.2021.9413449.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Keeps the spatial layout of the sub-aperture images as a 5D tensor, processes it with 4D convolutions, and predicts an occlusion mask jointly with the centre view, separating mask from content.
- **Key result:** Outperforms prior methods across datasets (abstract). The 2026 preprint [40] lists it at 29.75 dB / 0.804 mean PSNR/SSIM on 4-Syn (secondary report).
- **Relevance:** Joint mask-and-image prediction is the learned counterpart of our explicit visibility masks.

**[46]** S. Zhang, Z. Shen, and Y. Lin, "Removing foreground occlusions in light field using micro-lens dynamic filter," in *Proc. 30th Int. Joint Conf. Artif. Intell. (IJCAI)*, Montreal, Canada, Aug. 2021, pp. 1302–1308, doi: 10.24963/ijcai.2021/180.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Micro-lens dynamic filters "seek" background points hidden in some views by searching shifted micro-lens images (angular information). A second stage "generates" the remaining background from spatial context.
- **Key result:** As reported in ISTY's comparison [48]: 26.37 dB / 0.871 on 4-Syn and 27.97 dB / 0.901 on 9-Syn, but only 18.30 dB / 0.662 on the real Stanford CD scene.
- **Relevance:** The gap between synthetic and real results is a warning that learned de-occluders generalise poorly to real captures like ours.

**[47]** X. Wang, J. Liu, S. Chen, and G. Wei, "Effective light field de-occlusion network based on Swin Transformer," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 33, no. 6, pp. 2590–2599, Jun. 2023, doi: 10.1109/TCSVT.2022.3226227.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Argues that CNN de-occluders fail on large occlusions because their receptive fields are too small. The network uses CNN layers for local features and Swin Transformer layers for global context. The authors build a dataset with mild, moderate and severe occlusion and propose evaluating de-occlusion by a downstream detection task.
- **Key result:** 27.87 dB on the public dataset and 29.46 dB on their own dataset (average occlusion rates 19.82 %, 32.13 % and 40.56 %).
- **Relevance:** Supports evaluating by downstream detection, which matches our counting goal. *Note:* no light-field paper called "ISTNet" (named in the brief) was found in CrossRef, OpenAlex or S2, so it is **UNVERIFIED**. The closest matches are this paper and ISTY [48].

**[48]** J. Hur, J. Y. Lee, J. Choi, and J. Kim, "I see-through you: A framework for removing foreground occlusion in both sparse and dense light field images," in *Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV)*, Waikoloa, HI, USA, Jan. 2023, pp. 229–238, doi: 10.1109/WACV56688.2023.00031.
*Verified: CrossRef, OpenAlex, S2. Read: full text (arXiv:2301.06392).*
- **Summary:** Separates "sparse" light fields (camera arrays with large disparities, where most hidden background is visible in some other view) from "dense" ones (lenslet cameras with small disparities, where it must be inpainted). The framework splits into a feature extractor, an explicit occlusion-mask generator and an inpainter. The explicit mask also allows removing occluders at any chosen depth.
- **Key result:** 26.42 dB / 0.836 on 4-Syn and 27.04 dB / 0.849 on 9-Syn. On the real Stanford CD scene, 25.17 dB / 0.870, against DeOccNet's 21.22 dB / 0.740 and Zhang *et al.*'s [46] 18.30 dB / 0.662.
- **Relevance:** In ISTY's terms our rail capture is a *sparse* light field, so de-occlusion is mainly about finding the views that see each point rather than inventing content. This justifies a geometric visibility method.

**[49]** S. Zhang, S. Chang, Z. Shi, and Y. Lin, "Progressive multi-plane images construction for light field occlusion removal," *IEEE Trans. Vis. Comput. Graph.*, vol. 31, no. 10, pp. 8012–8025, Oct. 2025, doi: 10.1109/TVCG.2025.3561374.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Builds a multi-plane image (MPI) layer by layer from near to far, using the occluders found in nearer layers as a prior when estimating the next layer. An occlusion-aware attention network does the per-layer estimation. Deleting the occluder layers and re-rendering gives occlusion-free views from any viewpoint, not only the centre.
- **Key result:** Better than prior light-field occlusion-removal methods on synthetic and real scenes (abstract).
- **Relevance:** Near-to-far layered visibility is exactly our planned order (foliage layer, then fruit layer), with an MPI as the representation.

**[50]** M. F. Senussi, M. Abdalla, M. S. Kasem, M. Mahmoud, B. Yagoub, and H.-S. Kang, "A comprehensive review on light field occlusion removal: Trends, challenges, and future directions," *IEEE Access*, vol. 13, pp. 42472–42493, 2025, doi: 10.1109/ACCESS.2025.3548133.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Surveys single-view inpainting adapted to light fields and dedicated light-field occlusion-removal methods, together with datasets, metrics, loss functions and open problems such as large occlusions.
- **Key result:** None (review).
- **Relevance:** A recent survey for the report. It frames the field almost entirely around learned methods on regular grids, the gap we exploit.

**[51]** C. Zhu, R. Wan, Y. Tang, and B. Shi, "Occlusion-free scene recovery via neural radiance fields," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Vancouver, BC, Canada, Jun. 2023, pp. 20722–20731, doi: 10.1109/CVPR52729.2023.01985.
*Verified: CrossRef, OpenAlex, S2. Read: full text (CVF Open Access).*
- **Summary:** Removes occlusions without external supervision:
  1. A scene NeRF (with the occluders) is jointly optimised with the camera poses.
  2. A cost volume of feature variance across M = 4 neighbouring views guides depth.
  3. A mask network turns the bidirectional depth difference into a supervision mask.
  4. A background NeRF is trained only on the rays that see the background.
- **Key result:** PSNR 19.39 dB, SSIM 0.83 and LPIPS 0.37 on their ground-truth scene, against 18.25 / 0.79 / 0.45 for flow-based removal plus NeRF, 11.06 / 0.45 / 0.61 for NeRF-W, and 15.41 / 0.70 / 0.41 for Ha-NeRF. Scenes used 20–60 images.
- **Relevance:** The neural method most like ours in spirit: occluders are found by multi-view photo-consistency and depth ordering, not semantics. It costs a per-scene NeRF optimisation; we aim for the same visibility logic in a direct, training-free plane sweep.

**[52]** A. Mirzaei *et al.*, "SPIn-NeRF: Multiview segmentation and perceptual inpainting with neural radiance fields," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Vancouver, BC, Canada, Jun. 2023, pp. 20669–20679, doi: 10.1109/CVPR52729.2023.01980.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** From sparse annotations in one image, SPIn-NeRF builds a 3D object mask, then removes the object by distilling a 2D inpainter into the NeRF with a perceptual loss that keeps views consistent. It also releases a with/without-object benchmark.
- **Key result:** Strong 3D-inpainting quality on the new dataset (about 143 citations by Oct 2026; numbers not extracted).
- **Relevance:** A representative "segment then inpaint" baseline. It needs user annotation and invents content, which is unacceptable for counting.

**[53]** W. Liu, Z. Xiong, X. Li, and N. Jacobs, "DeclutterNeRF: Generative-free 3D scene recovery for occlusion removal," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, Nashville, TN, USA, Jun. 2025, pp. 380–390, doi: 10.1109/CVPRW67362.2025.00042.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Avoids generative priors altogether. It combines joint multi-view optimisation of learnable camera parameters, an "occlusion annealing" regulariser and a stochastic structural-similarity loss, and it introduces the DeclutterSet dataset with occluders in the foreground, midground and background.
- **Key result:** Artifact-free reconstructions that outperform generative baselines (abstract).
- **Relevance:** Corroborates our design choice that, for measurement tasks, recovering only what was observed beats hallucination.

**[54]** A. Kirillov *et al.*, "Segment anything," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, Paris, France, Oct. 2023, pp. 3992–4003, doi: 10.1109/ICCV51070.2023.00371.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** A promptable segmentation foundation model trained on SA-1B (over 1 billion masks on 11 million images), with strong zero-shot transfer.
- **Key result:** Zero-shot performance often competitive with, or better than, fully supervised models.
- **Relevance:** UncNeRF [2] uses SAM, seeded with clicks, for occluder masks. In our pipeline SAM could segment fruit on the *de-occluded* image without a colour threshold, but it should not be used to define the occluders, which would bring back a semantic prior.

**[55]** L. Yu, X. Zhang, W. Liao, W. Yang, and G.-S. Xia, "Learning to see through with events," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 45, no. 7, pp. 8660–8678, Jul. 2023, doi: 10.1109/TPAMI.2022.3227448.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Event-based SAI: events from a moving event camera are refocused (Refocus-Net) so that in-focus events align and off-focus occluder events scatter. A hybrid spiking/CNN network then reconstructs an image of the occluded target.
- **Key result:** Handles very dense occlusion and extreme lighting where frame-based SAI degrades (qualitative in the abstract).
- **Relevance:** Shows SAI generalising to new sensors. Our frame-based rail is the low-cost baseline such work is compared against.

### B5. Multi-baseline stereo, plane sweep, EPI analysis and SfM/MVS (depth for occlusion reasoning)

**[56]** R. C. Bolles, H. H. Baker, and D. H. Marimont, "Epipolar-plane image analysis: An approach to determining structure from motion," *Int. J. Comput. Vis.*, vol. 1, no. 1, pp. 7–55, 1987, doi: 10.1007/BF00128525.
*Verified: CrossRef, OpenAlex, S2. Read: no abstract is indexed; the description is corroborated by [61].*
- **Summary:** Introduced epipolar-plane image (EPI) analysis. With a camera translating along a straight line and taking closely spaced images, slicing the image volume along an epipolar plane makes each scene point trace a straight line whose slope encodes its depth, so structure comes from line fitting rather than correspondence search.
- **Key result:** The seminal formulation (about 890 citations by Oct 2026).
- **Relevance:** Our rail sweep is an EPI volume with only 13 rows. The fruit's trails slope about 162 px per row and the leaves' about 258 px per row (our calculation), far too sparse for line tracking, but the EPI's occlusion-ordering logic still applies.

**[57]** M. Okutomi and T. Kanade, "A multiple-baseline stereo," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 15, no. 4, pp. 353–363, Apr. 1993, doi: 10.1109/34.206955.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Uses several baselines produced by laterally displacing one camera. The SSD curves of all the stereo pairs are expressed in inverse distance and summed (SSSD-in-inverse-distance), which gives a single clear minimum.
- **Key result:** The analysis shows the summed cost removes ambiguity, including from repetitive patterns, and improves precision.
- **Relevance:** This is exactly our capture (a laterally displaced camera with multiple baselines). Summing in inverse depth is the plane-sweep cost we use, and it directly addresses false matches on the periodic diamond lattice.

**[58]** R. T. Collins, "A space-sweep approach to true multi-image matching," in *Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. (CVPR)*, San Francisco, CA, USA, 1996, pp. 358–363, doi: 10.1109/CVPR.1996.517097.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Defines "true multi-image" matching: it scales to any number of images at linear cost and treats all images equally. It does this by sweeping a plane through space and accumulating back-projected evidence in each cell.
- **Key result:** Demonstrated on a seven-image aerial matching example.
- **Relevance:** The framework underlying both [1] and our full-pose plane sweep.

**[59]** S. M. Seitz and C. R. Dyer, "Photorealistic scene reconstruction by voxel coloring," *Int. J. Comput. Vis.*, vol. 35, no. 2, pp. 151–173, Nov. 1999, doi: 10.1023/A:1008176507526.
*Verified: CrossRef, OpenAlex, S2. Read: abstract (authors' preprint).*
- **Summary:** Traverses a discretised scene volume in a fixed visibility order, near to far, and keeps colour-consistent voxels. Because occlusion is accounted for during the sweep, cameras can be far apart.
- **Key result:** Reconstructions from inward- and outward-facing camera sets that are spatially and photometrically consistent with the input images.
- **Relevance:** Near-to-far commitment is how our method removes foliage before matching fruit. Vaish *et al.* [1] showed its weak point: one hard colour-consistency threshold either keeps foliage or deletes background. Soft or robust visibility is preferable.

**[60]** S. B. Kang, R. Szeliski, and J. Chai, "Handling occlusions in dense multi-view stereo," in *Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Kauai, HI, USA, 2001, vol. 1, pp. I-103–I-110, doi: 10.1109/CVPR.2001.990462.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Semi-occluded pixels (visible in some views but not all) become more common as views are added. Two remedies are proposed: shiftable windows combined with matching against a dynamically selected subset of neighbouring images ("temporal selection"), and explicit occlusion labelling with visibility reasoning in a global energy.
- **Key result:** Spatial plus temporal selection gave a "dramatic" improvement over conventional multi-baseline stereo. Explicit labelling helped but not significantly once selection was applied.
- **Relevance:** On a 1D rail, a fruit point beside a leaf edge is usually visible in a contiguous run of views on one side. Choosing the left or right subset is a cheap, colour-free visibility heuristic worth testing before full labelling.

**[61]** A. Criminisi, S. B. Kang, R. Swaminathan, R. Szeliski, and P. Anandan, "Extracting layers and analyzing their specular properties using epipolar-plane-image analysis," *Comput. Vis. Image Underst.*, vol. 97, no. 1, pp. 51–85, Jan. 2005, doi: 10.1016/j.cviu.2004.06.001.
*Verified: CrossRef, OpenAlex, S2. Read: full text (authors' PDF).*
- **Summary:** Densely segments the EPI volume into "EPI-strips" and "EPI-tubes", which are coherent layers. Under linear constant-speed camera motion, trails are straight with orientation set by inverse depth, and trail intersections (Y-junctions) mark occlusions. The front-most strip is extracted first and blanked out. Each strip's cost combines variance photo-consistency with a crossing penalty. The paper also separates diffuse and specular components.
- **Key result:** Dense layered reconstruction and highlight removal on complex, heavily occluded sequences (qualitative).
- **Relevance:** Layer extraction from a linear sweep, front layer first, is our foliage-then-fruit strategy. The specularity analysis also bears on the glossy toy cherries.

**[62]** S. Wanner and B. Goldluecke, "Globally consistent depth labeling of 4D light fields," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Providence, RI, USA, Jun. 2012, pp. 41–48, doi: 10.1109/CVPR.2012.6247656.
**[63]** S. Wanner and B. Goldluecke, "Variational light field analysis for disparity estimation and super-resolution," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 36, no. 3, pp. 606–619, Mar. 2014, doi: 10.1109/TPAMI.2013.147.
*Both verified: CrossRef, OpenAlex, S2. Read: abstracts.*
- **Summary:** Depth is estimated from local EPI orientation (structure tensor) with no matching cost. [62] adds a constrained labelling with global visibility constraints, solved by convex relaxation. [63] adds a continuous variational framework for disparity and spatial/angular super-resolution, plus a benchmark dataset.
- **Key result:** [63] is reported as faster and more accurate than a multi-view stereo method on their benchmark.
- **Relevance:** EPI-orientation methods assume densely sampled light fields with disparities of a few pixels per view. Our 160–260 px steps break that assumption, which is why we use plane-sweep matching (reader-inferred).

**[64]** C. Kim, H. Zimmer, Y. Pritch, A. Sorkine-Hornung, and M. Gross, "Scene reconstruction from high spatio-angular resolution light fields," *ACM Trans. Graph.*, vol. 32, no. 4, pp. 1–12, Jul. 2013, doi: 10.1145/2461912.2461926.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Reconstructs detailed scenes from very dense 3D light fields captured with linear camera motion (around 10⁹ rays). Depth is estimated per ray, first at object boundaries, then propagated fine-to-coarse into homogeneous regions without global optimisation.
- **Key result:** "Gigaray" light fields processed on a standard GPU with precise object contours (qualitative).
- **Relevance:** The dense-sampling counterpart of our linear rail. It shows what 1D capture can do at high angular density, and why our 13-view case needs explicit matching and visibility reasoning instead.

**[65]** T.-C. Wang, A. A. Efros, and R. Ramamoorthi, "Occlusion-aware depth estimation using light-field cameras," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, Santiago, Chile, Dec. 2015, pp. 3487–3495, doi: 10.1109/ICCV.2015.398.
*Verified: CrossRef, OpenAlex, S2. Read: full text (CVF Open Access).*
- **Summary:** At an occlusion boundary, a pixel is still photo-consistent in roughly half the views. The line dividing the consistent views from the occluded ones has the same orientation as the spatial occlusion edge. Matching only the consistent half improves depth and yields occlusion-boundary predictions for regularisation.
- **Key result:** Lowest depth RMSE on both synthetic datasets (0.0355 and 0.0848), and an occlusion-boundary F-measure of 0.68, against 0.54 for a competing method in the same precision-recall plot.
- **Relevance:** On a horizontal rail the dividing line is always "left or right of some view", which reduces the visibility problem to choosing a contiguous run of views (compare [60]).

**[66]** J. L. Schönberger and J.-M. Frahm, "Structure-from-motion revisited," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Las Vegas, NV, USA, Jun. 2016, pp. 4104–4113, doi: 10.1109/CVPR.2016.445.
**[67]** J. L. Schönberger, E. Zheng, J.-M. Frahm, and M. Pollefeys, "Pixelwise view selection for unstructured multi-view stereo," in *Computer Vision – ECCV 2016* (Lecture Notes in Computer Science). Cham, Switzerland: Springer, 2016, pp. 501–518, doi: 10.1007/978-3-319-46487-9_31.
*Both verified: CrossRef, OpenAlex, S2. Read: abstracts (S2 and ML Anthology).*
- **Summary:** These are COLMAP. [66] improves incremental SfM for robustness, accuracy, completeness and scalability and is released open source. [67] estimates depth and normals jointly, with pixelwise view selection using photometric and geometric priors, and a multi-view geometric-consistency term for refinement and fusion.
- **Key result:** State-of-the-art accuracy, completeness and efficiency on benchmarks and Internet photo collections (abstracts; about 2,300 citations for [67]).
- **Relevance:** Supplies our camera poses and our teammate's dense cloud. Pixelwise view selection is a per-pixel visibility model, but a fruit behind foliage has few consistent views. The 2026 peering study [39] found SfM-type reconstructions dominated by the foreground occluders. Constraining poses to a 1D rail translation would stabilise them.

**[68]** B. Mildenhall *et al.*, "Local light field fusion: Practical view synthesis with prescriptive sampling guidelines," *ACM Trans. Graph.*, vol. 38, no. 4, pp. 1–14, Jul. 2019, doi: 10.1145/3306346.3322980.
*Verified: CrossRef, OpenAlex, S2. Read: full text (arXiv:1905.00889). The indexes store the title and subtitle separately, which tripped an automatic title-similarity flag; it was checked manually.*
- **Summary:** Promotes each captured view to a multi-plane image and blends neighbouring ones. The derived capture rule is that the nearest object should move at most *D* pixels between adjacent views (*D* = number of MPI planes), rather than about one pixel at the Nyquist rate.
- **Key result:** Nyquist-level perceptual quality with up to 64² ≈ 4000 times fewer views.
- **Relevance:** Our foliage moves about 258 px per step (our calculation), roughly four times LLFF's D = 64 guideline, so our sweep is unsuitable for view synthesis or NeRF-style novel views. That is exactly why occluder ghosts in refocused images appear as discrete copies. Wide spacing nevertheless helps occlusion removal [26], so the sampling regime depends on the goal.

**[69]** Y. Wang, L. Wang, Z. Liang, J. Yang, W. An, and Y. Guo, "Occlusion-aware cost constructor for light field depth estimation," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, New Orleans, LA, USA, Jun. 2022, pp. 19777–19786, doi: 10.1109/CVPR52688.2022.01919.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Replaces shift-based cost volumes with dilated convolutions over the sub-aperture image array. The resulting cost constructor dynamically re-weights views to handle occlusion.
- **Key result:** Ranked first in MSE on the 4D light-field benchmark, with faster run time.
- **Relevance:** Learned occlusion-aware matching for dense light fields. It is not directly usable with 13 widely spaced views, but it is a useful contrast for the report.

### B6. Light fields, multi-view occlusion handling and 3D counting in agriculture

*Search note: despite targeted queries (synthetic aperture, light field, plenoptic, camera array or integral imaging combined with fruit, orchard, harvesting or plant), we found **no peer-reviewed study applying optical synthetic-aperture refocusing to see fruit through foliage**. The nearest agricultural work uses multi-view geometry, light-field cameras for phenotyping, NeRF counting, or non-optical synthetic apertures (ultrasound) [79]. This is the main gap our project can claim (§C).*

**[70]** J. Hemming, J. Ruizendaal, J. W. Hofstee, and E. J. van Henten, "Fruit detectability analysis for different camera positions in sweet-pepper," *Sensors*, vol. 14, no. 4, pp. 6032–6044, Mar. 2014, doi: 10.3390/s140406032.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Measures how camera position affects visibility of greenhouse sweet peppers. A rig places the camera at different azimuth and zenith angles and moves it horizontally along the row, and fruit visibility at 14 positions was scored manually.
- **Key result:** From any single position, at most 69 % of fruit were detectable (criterion: at most 50 % occlusion). The best combination of five viewpoints raised this to 90 %.
- **Relevance:** Hard numbers for the motivating claim that one view misses a third of the fruit and multiple viewpoints recover most of it. Their horizontal-translation rig is a crude version of our rail.

**[71]** G. Polder and J. W. Hofstee, "Phenotyping large tomato plants in the greenhouse using a 3D light-field camera," presented at the *ASABE Annu. Int. Meeting*, Jul. 2014, pp. 1–7, doi: 10.13031/aim.20141882255.
*Verified: CrossRef (deposit lists no authors; authors taken from OpenAlex and S2), OpenAlex, S2. Read: abstract.*
- **Summary:** A plenoptic (micro-lens array) camera is carried to the plants and outputs a pixel-registered colour image plus depth map, so focus and viewpoint can be changed after capture.
- **Key result:** Feasibility for greenhouse tomato phenotyping (qualitative).
- **Relevance:** Early use of a light-field camera on crops. Its narrow lenslet baseline gives a *dense* light field that cannot see around leaves, in contrast to our wide-baseline rail (compare [48]).

**[72]** M. Stein, S. Bargoti, and J. Underwood, "Image based mango fruit detection, localisation and yield estimation using multiple view geometry," *Sensors*, vol. 16, no. 11, Art. no. 1915, Nov. 2016, doi: 10.3390/s16111915.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** From the University of Sydney's ACFR. Faster R-CNN detects mangoes. Detections are associated across images using navigation-system trajectories, assigned to trees with LiDAR canopy masks, and triangulated to 3D so each fruit is counted once.
- **Key result:** 522 trees and 71,609 mangoes scanned near Bundaberg, Queensland. The multi-view method needed no field calibration and had a 1.36 % per-tree count error against hand counts.
- **Relevance:** The benchmark for "3D de-duplicated counting" from many views, and a local, citable precedent. Our variant counts in a focal stack or 3D volume after occluder removal rather than tracking 2D detections.

**[73]** C. McCool, I. Sa, F. Dayoub, C. Lehnert, T. Perez, and B. Upcroft, "Visual detection of occluded crop: For automated harvesting," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, Stockholm, Sweden, May 2016, pp. 2506–2512, doi: 10.1109/ICRA.2016.7487405.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Detects field sweet peppers, which are heavily occluded and green on green, by per-pixel segmentation with local binary pattern texture followed by region detection.
- **Key result:** Segmentation AUC improved from 0.10 to 0.56 with LBP. 69.2 % of peppers were detected, against 66.8 % for people viewing the same images.
- **Relevance:** Shows that green-on-green crops defeat colour cues and need texture or geometry. This supports our colour-free design and the green-fruit generalisation test.

**[74]** R. Schima *et al.*, "Imagine all the plants: Evaluation of a light-field camera for on-site crop growth monitoring," *Remote Sens.*, vol. 8, no. 10, Art. no. 823, Oct. 2016, doi: 10.3390/rs8100823.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** A long-term field evaluation of a consumer Lytro light-field camera for measuring plant growth and morphology on site.
- **Key result:** Suitable for monitoring growth dynamics and plant traits, and robust to ambient conditions (qualitative).
- **Relevance:** Another agricultural light-field example, again with a dense, narrow-baseline camera used for 3D plant structure rather than for seeing through occluders.

**[75]** P. Kurtser and Y. Edan, "Statistical models for fruit detectability: Spatial and temporal analyses of sweet peppers," *Biosyst. Eng.*, vol. 171, pp. 272–289, Jul. 2018, doi: 10.1016/j.biosystemseng.2018.04.017.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Develops Poisson-regression models of fruit detectability as a function of controllable variables (viewpoint and others) and measurable ones, from an eye-in-hand RGB camera on a 6-DOF arm.
- **Key result:** 1,312 images from 10–14 viewpoints over 56 greenhouse scenes; the models identified the best viewpoints.
- **Relevance:** A statistical approach to viewpoint planning for fruit. It complements AOS visibility theory [26] and could justify where on the rail to concentrate shots.

**[76]** X. Liu *et al.*, "Robust fruit counting: Combining deep learning, tracking, and structure from motion," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Madrid, Spain, Oct. 2018, pp. 1045–1052, doi: 10.1109/IROS.2018.8594239.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** A fully convolutional network segments fruit. Tracks are formed by KLT tracking with a Kalman filter and Hungarian assignment, and SfM gives relative 3D positions and sizes that reject outliers and double-counted tracks.
- **Key result:** Counts were accurate against human-annotated visual counts, and the SfM correction "significantly" improved accuracy (numbers not in the abstract).
- **Relevance:** De-duplication through 3D consistency, which is the role our 3D cherry-centre clustering plays.

**[77]** C. Lehnert, D. Tsai, A. Eriksson, and C. McCool, "3D move to see: Multi-perspective visual servoing towards the next best view within unstructured and occluded environments," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Macau, China, Nov. 2019, pp. 3890–3897, doi: 10.1109/IROS40897.2019.8967918.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** A custom camera array on a harvesting robot's end-effector captures several perspectives at once. Segmentation plus a per-view objective gives a gradient towards the next best, less occluded view in a single shot.
- **Key result:** Mean increase in visible target size of 29.3 %, against 9.17 % for a single RGB-D baseline (about three times larger).
- **Relevance:** A camera array used to see past occluding leaves in a crop, though it moves the robot rather than computing a synthetic aperture. An Australian (QUT) precedent worth citing.

**[78]** J. Gené-Mola *et al.*, "Fruit detection and 3D location using instance segmentation neural networks and structure-from-motion photogrammetry," *Comput. Electron. Agric.*, vol. 169, Art. no. 105165, Feb. 2020, doi: 10.1016/j.compag.2019.105165.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Mask R-CNN detections are projected onto an SfM point cloud, and an SVM removes false positives, giving 3D fruit positions.
- **Key result:** 11 Fuji apple trees with 1,455 apples. F1 rose from 0.816 for 2D detection to 0.881 for 3D detection and location, but SfM processing was slow.
- **Relevance:** Quantifies the benefit of lifting detections into 3D. It is close to our teammate's COLMAP pipeline and a baseline for our counting.

**[79]** B. Parr, M. Legg, S. Bradley, and F. Alam, "Occluded grape cluster detection and vine canopy visualisation using an ultrasonic phased array," *Sensors*, vol. 21, no. 6, Art. no. 2182, Mar. 2021, doi: 10.3390/s21062182.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Uses an air-coupled ultrasonic phased array, a synthetic aperture in acoustics, to image grape bunches through leaves, with near-field focusing, chirp excitation and a fan to separate leaf echoes from grape echoes. The authors note that synthetic-aperture radar had also been explored but is expensive.
- **Key result:** Grape-volume overestimation fell from 222 % to 112 % against a photogrammetry scan (and from 56 % to 2.5 % against its convex hull).
- **Relevance:** The closest "see the fruit through the leaves" work we found, but non-optical. It shows the synthetic-aperture idea has been tried for occluded fruit in other modalities, not with cameras.

**[80]** J. Gené-Mola *et al.*, "Looking behind occlusions: A study on amodal segmentation for robust on-tree apple fruit size estimation," *Comput. Electron. Agric.*, vol. 209, Art. no. 107854, Jun. 2023, doi: 10.1016/j.compag.2023.107854.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** A CNN predicts both the visible (modal) and the complete (amodal) masks of apples from single RGB-D images, then estimates diameter with the pinhole model.
- **Key result:** 3,925 RGB-D images with 15,335 apples; detection F1 0.86; diameter MAE 4.5 mm with R² = 0.80, regardless of visibility.
- **Relevance:** The learned single-view alternative to geometric de-occlusion. It infers hidden shape rather than observing it, which is useful as a contrast and for sizing partly hidden cherries.

**[81]** L. Meyer, A. Gilson, U. Schmid, and M. Stamminger, "FruitNeRF: A unified neural radiance field based fruit counting framework," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Abu Dhabi, UAE, Oct. 2024, pp. 1–8, doi: 10.1109/IROS58592.2024.10802065.
*Verified: CrossRef, OpenAlex, S2. Read: full text (arXiv:2408.06190).*
- **Summary:** Takes posed images (from COLMAP) with fruit masks from Grounded-SAM or a U-Net, trains a semantic NeRF, samples a fruit-only point cloud and counts fruit by cascaded clustering, so fruit are counted in 3D without double counting.
- **Key results:**
  - Synthetic, six fruit types: F1 0.95 with ground-truth masks and 0.88 with SAM masks (SAM recall only 0.40 for plums and 0.69 for mangoes).
  - About 89 % detection on two real apple trees and about 82 % on a more occluded third.
  - F1 0.79 on Fuji-SfM.
  - FruitNeRF "struggles" with fewer than 20 images and needs about 30–40.
- **Relevance:** The state of the art for 3D fruit counting. With only 13 images we are below its working range, which supports our training-free geometric pipeline. Its SAM failures on some fruit types also caution against semantic masks for green fruit.

**[82]** L. Meyer, A.-T. Ardelean, T. Weyrich, and M. Stamminger, "FruitNeRF++: A generalized multi-fruit counting method utilizing contrastive learning and neural radiance fields," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Hangzhou, China, Oct. 2025, pp. 1717–1724, doi: 10.1109/IROS60139.2025.11247341.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Removes FruitNeRF's per-fruit-type clustering by learning instance embeddings from foundation-model instance masks in a neural instance field, then clusters the embedded point cloud in a fruit-agnostic way.
- **Key result:** Easier to control, and compares favourably with prior methods on a synthetic multi-fruit set and a real apple benchmark (qualitative).
- **Relevance:** Shape-agnostic 3D instance counting, the learned counterpart of our 3D de-duplication.

**[83]** H. Gan, W. S. Lee, V. Alchanatis, R. Ehsani, and J. K. Schueller, "Immature green citrus fruit detection using color and thermal images," *Comput. Electron. Agric.*, vol. 152, pp. 117–125, Sep. 2018, doi: 10.1016/j.compag.2018.07.011.
*Verified: CrossRef, OpenAlex, S2. Read: abstract.*
- **Summary:** Green citrus is hard to detect because fruit and leaf colours are very similar. The authors register thermal to colour images and fuse them in a colour-thermal combined probability classifier.
- **Key result:** Recall rose from 78.1 % (colour only) to 90.4 % at similar precision, and precision from 86.6 % to 95.5 % at similar recall.
- **Relevance:** Quantifies how colour alone falls short for green fruit and how adding a non-colour cue helps. Our extra cue is geometric (depth and visibility) rather than thermal.

---

## C. Synthesis

**C1. What fails when foliage hides a point in most views.**

Three generations of methods fail in different ways once foliage covers a fruit point in more than about half the views.
- *Mean refocusing.* Plain shift-and-add [5], [8], [9] never removes an occluder; it spreads it out. Kurmi *et al.*'s statistical model makes this precise [26]. With decorrelated views, visibility tends to 1 − D̃² rather than 1, so a haze set by the occlusion density remains however many views are added. The normalised gain is only 1 − 1/*N*, about 92 % for our 13 views.
- *Robust ray statistics.* These remove the haze only while the occluder is in the minority. Vaish *et al.*'s median cost collapses beyond about 50 % occlusion, and both median and entropy failed outright behind a uniformly coloured ivy wall at about 70 % occlusion [1]. Our uniform-green pipe cleaners are exactly that case.
- *Learned light-field networks.* These raise synthetic-benchmark scores (around 26–30 dB on 4-Syn [40], [48]) but transfer unevenly to real captures (18.3–25.2 dB on the real Stanford CD scene [46], [48]), and they are trained on regular 5 × 5 grids [44].

When a point is hidden in *every* view, only a prior can fill it (GAN, diffusion or vision-language inpainting [23], [40], [52]), and for counting a hallucinated cherry is an error. The common lesson of [12], [14], [36] and [51] is that the 50 % barrier disappears once occluded rays are identified and *excluded* rather than out-voted. A point seen in only two or three of 13 views can then still be reconstructed, much as Joshi *et al.* track targets more than 70 % occluded in every view by aggregating across views [11].

**C2. Colour-free ways to find occluders.**

The literature offers five cues that need no colour model:
1. *Photo-consistency in a plane sweep.* An occluder is whatever surface is consistent at a nearer depth [1], [22], [57], [58]. Zhu *et al.* turn a feature-variance cost volume and depth ordering into supervision masks without any labels [51].
2. *Front-to-back visibility propagation.* Commit the nearest layer, then match the next layer using only the remaining rays [14]–[16], [49], [59]. Voxel colouring shows that hard thresholds are brittle [1], which favours soft or energy-based labelling [12].
3. *Parallax layer separation* from camera motion [41]–[43].
4. *EPI structure.* Under linear camera motion, occlusions appear as crossing trails, and the occluding trail is always the more slanted one [56], [61]. At a boundary, roughly half the views stay consistent and the split follows the edge orientation [65]. On a horizontal rail this reduces to choosing a contiguous left or right run of views, which is Kang *et al.*'s temporal selection [60].
5. *External depth* from LiDAR [36] or RGB-D [17].

By contrast, the newest through-foliage systems still define occluders by a vegetation colour index or a monocular depth network [39], by click-prompted SAM masks [2], or by colour clustering [19]. These break on non-vegetation occluders such as fences [39] and on green fruit, where colour separation is weak [73], [83].

**C3. What suits a sparse 13-view rail.**

Our sweep moves the foliage about 258 px and the fruit about 162 px per 50 mm step (our calculation). For view synthesis the light field is undersampled by two orders of magnitude [6], [68]. That rules out, or makes risky, several families:
- EPI-orientation depth, which assumes a few pixels of disparity per view [62]–[64];
- optical-flow layer separation, which assumes small inter-frame motion [41]–[43];
- NeRF-based removal and counting, which used 20–60 images [51], with FruitNeRF degrading below about 20 [81];
- light-field networks, which expect dense 2D grids [44]–[49].

For occlusion removal, however, the wide baseline is an asset. Kurmi's model says visibility saturates once the inter-view disparity exceeds the projected occluder size [26]. Our relative shift is about 96 px per step, against about 26–52 px for a 5–10 mm strand, so each view sees an almost independent occluder pattern. AOS field trials show that sparsely sampled one-dimensional apertures are enough for detection [30], [33].

The methods that fit are therefore:
- known-pose multi-baseline plane sweeps [57], [58] with explicit per-view visibility labels [12], [14];
- front-to-back layering [16];
- 1D view-subset selection [60], [65].

Poses can come from COLMAP [66], constrained to rail translation, or from an IMU-based homography model [18]. Because adjacent views share few occluder pixels, we should not aim at novel-view synthesis. The right outputs are a single all-in-focus reference image and 3D fruit centres.

**C4. Gaps we can claim.**
1. We found no peer-reviewed application of *optical* synthetic-aperture imaging to seeing fruit through foliage. Agricultural work uses multi-view tracking and triangulation [72], [76], [78], NeRF counting [81], [82], dense light-field cameras for phenotyping [71], [74], or ultrasound [79].
2. Occluder masks in current SAI systems are colour-based, semantic, prompted or sensor-based [2], [17], [36], [39]. Purely multi-view, photo-consistency masks for a sparse linear sweep have barely been evaluated on real scenes since Pei *et al.* [12] and Yang *et al.* [14].
3. The visibility theory [26] assumes 2D sampling and isotropic occluders. It is untested for 1D rails and elongated occluders.
4. Evaluation is almost always PSNR/SSIM [44]–[49]; only [47] proposes downstream-task evaluation. No SAI study reports counting accuracy on occluded fruit, or tests colour independence by swapping red fruit for green.
5. Training-free, hallucination-free recovery is under-served as the field moves towards generative priors [23], [40], [52].

**Supplementary: cross-paper tensions (scoped scan, not exhaustive; scholar confirmation pending).**

| Tension | Assessment | Resolution |
|---|---|---|
| Integrating before detection wins for DNN person detection [28], but per-image anomaly detection followed by integration wins for colour anomalies [34]. | Conditional difference | The best order depends on the detector. Object classifiers need the coherent shape that integration reveals, whereas per-pixel detectors gain from integrating per-view binary evidence. For us: integrate per-view visibility masks and detect fruit on the de-occluded image. |
| Focus beats stereo under heavy occlusion [1], yet Theorem 1 shows stereo succeeds where focus cannot [1]. | Conditional difference | Focus needs third-order texture. Smooth, glossy cherries have little texture, so both costs are weak on fruit interiors; use them on textured lattice and leaf edges, and use detection for fruit. |
| "Multi-view 3D vision fails behind foliage" [37], [39], yet SfM-based fruit counting works [72], [78]. | Conditional difference | Orchard scans use hundreds of images in which most fruit is seen many times; [39] has dense, close-range occluders. Our 13-view, heavily occluded case is closer to [39], so do not rely on MVS fruit points alone. |
| A narrower FOV always helps [26] versus the optimal FOV depending on forest structure [32]. | Resolved by model refinement | [32] adds overlap and coverage effects missing from [26]. For our flat scene, testing a central crop is the practical step. |

*Coverage note:* about 82 papers; the tensions were drawn from shared-topic pairs only. This is not a complete pairwise contradiction search.

---

## D. Essential references for the report (ranked)

1. **[1] Vaish *et al.*, CVPR 2006** (tutor paper). The plane-sweep formulation of robust synthetic-aperture costs (variance, focus, median, entropy), with measured failure points: median beyond about 50 % occlusion, and median and entropy behind uniformly coloured occluders. It is our starting point and the failure our method addresses.
2. **[26] Kurmi *et al.*, IEEE Sensors J. 2019.** The only quantitative theory of SAI occlusion removal: optimal baseline (disparity at least the occluder size), visibility bounds, V̂ = 1 − 1/*N*, and the residual-haze argument. Use it to justify the rail spacing and the 13 views, and to explain why averaging alone cannot work.
3. **[2] Ma *et al.*, IEEE TIP 2026** (tutor paper, UncNeRF). The modern neural counterpart: multi-view forward warping of visible object pixels to uncover heavily occluded objects. It contrasts prompted SAM masks and per-scene NeRF optimisation with our training-free geometric visibility.
4. **[12] Pei *et al.*, Pattern Recognit. 2013.** The direct precedent for per-view occluder labelling, with refocusing that averages only unlabelled rays.
5. **[5] Isaksen *et al.*, SIGGRAPH 2000.** The theory of dynamically reparameterised refocusing and synthetic apertures, needed to explain shift-and-add and our per-pixel aperture weighting.
6. **[57] Okutomi and Kanade, TPAMI 1993.** Multi-baseline stereo from a laterally displaced camera, summing costs in inverse depth. This is the formal basis of our rail plane sweep and of its robustness to the periodic lattice.
7. **[39] Bimber *et al.*, Sci. Robot. 2026.** The closest contemporary system: ground-level lateral "peering" synthetic apertures through vegetation. It documents that SfM, NeRF and VGGT fail and that occluder masks are still colour- or monocular-depth-based, which is our gap.
8. **[30] Schedl *et al.*, Sci. Robot. 2021.** Field evidence that sparse one-dimensional synthetic apertures are enough for deep detection behind foliage, with about one-tenth the images of 2D sampling (38 of 42 people found).
9. **[15]/[16] Yang *et al.*, ECCV 2014 / Sensors 2015.** Layered visibility propagation for all-in-focus, occluder-free SAI, the prior art for our reconstruction stage.
10. **[72] Stein *et al.*, Sensors 2016.** Multi-view geometry for counting each fruit once (1.36 % per-tree error, from Sydney's ACFR), the benchmark idea for our 3D de-duplicated count.

*Runners-up:*
- [48] ISTY, for the sparse-versus-dense light-field argument;
- [51] Zhu *et al.*, for geometry-derived masks in NeRF;
- [60] Kang *et al.* and [65] Wang *et al.*, for view-subset visibility;
- [70] Hemming *et al.*, for single-view detectability of at most 69 % against 90 % with multiple views;
- [81] FruitNeRF, the state of the art for 3D counting, which needs at least 20–30 images.

---

## E. Verification log, search limitations and notes for the team

**Verification summary (15 Oct 2026).** 83 references are cited.
- 81 matched by DOI in CrossRef with a title cross-check (similarity 1.0) and also in OpenAlex and Semantic Scholar. For [38] and [39], S2 holds the arXiv-linked record rather than the journal DOI.
- 2 are arXiv preprints ([32], [40]) matched in OpenAlex and S2 only, because arXiv DOIs are registered with DataCite.
- No DOI mismatches remain. One automatic flag was a false alarm: [68]'s title and subtitle are stored in separate fields, and it was checked manually.
- Volume, issue and page data are from CrossRef. Two journal years follow the print issue rather than first online publication: [17] (online 2015, print 2016) and [21] (DOI suffix 2018, volume 7 dated 2019).

**UNVERIFIED or corrected items from the brief.**
- *"ISTNet" (2023/24) for light-field occlusion removal:* no such paper was found in CrossRef, OpenAlex or S2, so it is **UNVERIFIED**. The likely intended papers are ISTY [48] (WACV 2023) or the Swin-Transformer network of Wang *et al.* [47].
- *Joshi et al. synthetic aperture tracking:* the brief said 2006; it is ICCV **2007** [11].
- *Vaish et al. 2006 author order:* the PDF byline is Vaish, Szeliski, Zitnick, Kang, Levoy, whereas the IEEE metadata lists Levoy second. We cite the byline [1].
- *UncNeRF [2]:* full text is closed, so every quantitative result in it is **UNVERIFIED**. The method description comes from the abstract, and the implementation notes come from the authors' public code and dataset.
- *Pei et al. 2013 [12], Yang et al. 2016 [17] and Zhang et al. 2017 [18]:* no abstract is indexed in OpenAlex or S2. Summaries rely on the institutional repository record (NWPU Pure); for [18], a web-search summary of that record was used.
- *Polder and Hofstee [71]:* the CrossRef deposit has no author list; authors are from OpenAlex and S2.

**Secondary-source cautions.**
- PSNR/SSIM values quoted for [20], [45] and [46] come from comparison tables in [44], [40] and [48], not from the original papers.
- The 2026 preprint [40] mis-cites "Mask4D" as an unrelated LiDAR paper, and its "All-in-Focus" row repeats DeOccNet's figures for Pei *et al.* [20]. Prefer DeOccNet's original table [44] for classical-method baselines.
- Benchmark protocols differ between [44] and [48] (for example, DeOccNet's 4-Syn score is reported as 22.74 dB, the mean over Syn01–04, in [44] but 25.04 dB in [48]). Do not mix numbers across papers in one table.

**Search limitations.**
- IEEE Xplore, Springer and MDPI block scripted access, so some full texts could not be read. Read scope is stated in every entry.
- Google Scholar was not queried directly.
- Non-English literature (for example, Chinese-language SAI work from NWPU) was not searched.
- The agricultural-SAI gap claim (§C4) is bounded by the queries listed in the header; a short librarian-assisted check of CAB Abstracts or Web of Science would strengthen it.

**Distributional skew advisory (bibliography skill, step 4.6).**

DISTRIBUTIONAL_SKEW_ADVISORY:
- Dimension: methodological distribution
- Concentration: computational and engineering studies = 83/83 (100 %)
- Advisory: this is a coverage signal, not a defect. It is expected for a computer-vision methods review, and no change is needed.

There are also two research-group clusters: JKU Linz (AOS), 17/83 (20 %), and NWPU/Alberta (SAI labelling), 13/83 (16 %). Both are below the 70 % threshold, so no advisory is triggered, but the AOS and SAI-labelling themes each rest largely on a single group's work.

**Disclosure.** The search, verification scripts and drafting were done with an AI assistant (Claude, Anthropic). Every factual claim should be checked against the cited source before submission, and the course's AI-use disclosure requirements should be followed.


---

## F. Consolidated reference list (IEEE, copy-ready)

*Numbering matches the annotated entries above. arXiv items [32] and [40] are preprints; cite them as such or replace them with peer-reviewed sources.*

[1] V. Vaish, R. Szeliski, C. L. Zitnick, S. B. Kang, and M. Levoy, "Reconstructing occluded surfaces using synthetic apertures: Stereo, focus and robust measures," in *Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. (CVPR'06)*, New York, NY, USA, 2006, vol. 2, pp. 2331–2338, doi: 10.1109/CVPR.2006.244.

[2] J. Ma, Q. Wang, Y. Zhao, and J. Li, "UncNeRF: Uncovering heavily occluded object with multi-view clues," *IEEE Trans. Image Process.*, vol. 35, pp. 8260–8274, 2026, doi: 10.1109/TIP.2026.3716400.

[3] M. Levoy and P. Hanrahan, "Light field rendering," in *Proc. 23rd Annu. Conf. Comput. Graph. Interact. Techn. (SIGGRAPH '96)*, Aug. 1996, pp. 31–42, doi: 10.1145/237170.237199.

[4] S. J. Gortler, R. Grzeszczuk, R. Szeliski, and M. F. Cohen, "The lumigraph," in *Proc. 23rd Annu. Conf. Comput. Graph. Interact. Techn. (SIGGRAPH '96)*, Aug. 1996, pp. 43–54, doi: 10.1145/237170.237200.

[5] A. Isaksen, L. McMillan, and S. J. Gortler, "Dynamically reparameterized light fields," in *Proc. 27th Annu. Conf. Comput. Graph. Interact. Techn. (SIGGRAPH '00)*, 2000, pp. 297–306, doi: 10.1145/344779.344929.

[6] J.-X. Chai, S.-C. Chan, H.-Y. Shum, and X. Tong, "Plenoptic sampling," in *Proc. 27th Annu. Conf. Comput. Graph. Interact. Techn. (SIGGRAPH '00)*, 2000, pp. 307–318, doi: 10.1145/344779.344932.

[7] V. Vaish, B. Wilburn, N. Joshi, and M. Levoy, "Using plane + parallax for calibrating dense camera arrays," in *Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Washington, DC, USA, 2004, vol. 1, pp. 2–9, doi: 10.1109/CVPR.2004.1315006.

[8] V. Vaish *et al.*, "Synthetic aperture focusing using a shear-warp factorization of the viewing transform," in *Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. (CVPR'05) Workshops*, San Diego, CA, USA, 2005, vol. 3, p. 129, doi: 10.1109/CVPR.2005.537.

[9] B. Wilburn *et al.*, "High performance imaging using large camera arrays," *ACM Trans. Graph.*, vol. 24, no. 3, pp. 765–776, Jul. 2005, doi: 10.1145/1073204.1073259.

[10] M. Levoy, "Light fields and computational imaging," *Computer*, vol. 39, no. 8, pp. 46–55, Aug. 2006, doi: 10.1109/MC.2006.270.

[11] N. Joshi, S. Avidan, W. Matusik, and D. J. Kriegman, "Synthetic aperture tracking: Tracking through occlusions," in *Proc. IEEE 11th Int. Conf. Comput. Vis. (ICCV)*, Rio de Janeiro, Brazil, 2007, pp. 1–8, doi: 10.1109/ICCV.2007.4409032.

[12] Z. Pei, Y. Zhang, X. Chen, and Y.-H. Yang, "Synthetic aperture imaging using pixel labeling via energy minimization," *Pattern Recognit.*, vol. 46, no. 1, pp. 174–187, Jan. 2013, doi: 10.1016/j.patcog.2012.06.014.

[13] T. Yang, Y. Zhang, X. Tong, X. Zhang, and R. Yu, "A new hybrid synthetic aperture imaging model for tracking and seeing people through occlusion," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 23, no. 9, pp. 1461–1475, Sep. 2013, doi: 10.1109/TCSVT.2013.2242553.

[14] T. Yang, Y. Zhang, X. Tong, W. Ma, and R. Yu, "High performance imaging through occlusion via energy minimization-based optimal camera selection," *Int. J. Adv. Robot. Syst.*, vol. 10, no. 11, Art. no. 393, Nov. 2013, doi: 10.5772/57175.

[15] T. Yang *et al.*, "All-in-focus synthetic aperture imaging," in *Computer Vision – ECCV 2014* (Lecture Notes in Computer Science). Cham, Switzerland: Springer, 2014, pp. 1–15, doi: 10.1007/978-3-319-10599-4_1.

[16] T. Yang *et al.*, "Multiple-layer visibility propagation-based synthetic aperture imaging through occlusion," *Sensors*, vol. 15, no. 8, pp. 18965–18984, Aug. 2015, doi: 10.3390/s150818965.

[17] T. Yang, W. Ma, S. Wang, J. Li, J. Yu, and Y. Zhang, "Kinect based real-time synthetic aperture imaging through occlusion," *Multimedia Tools Appl.*, vol. 75, no. 12, pp. 6925–6943, Jun. 2016, doi: 10.1007/s11042-015-2618-1.

[18] X. Zhang, Y. Zhang, T. Yang, and Y.-H. Yang, "Synthetic aperture photography using a moving camera-IMU system," *Pattern Recognit.*, vol. 62, pp. 175–188, Feb. 2017, doi: 10.1016/j.patcog.2016.07.019.

[19] Z. Xiao, L. Si, and G. Zhou, "Seeing beyond foreground occlusion: A joint framework for SAP-based scene depth and appearance reconstruction," *IEEE J. Sel. Topics Signal Process.*, vol. 11, no. 7, pp. 979–991, Oct. 2017, doi: 10.1109/JSTSP.2017.2715012.

[20] Z. Pei, X. Chen, and Y.-H. Yang, "All-in-focus synthetic aperture imaging using image matting," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 28, no. 2, pp. 288–301, Feb. 2018, doi: 10.1109/TCSVT.2016.2615328.

[21] J. Martínez Sotoca, P. Latorre-Carmona, F. Pla, and B. Javidi, "Depth and all-in-focus image estimation in synthetic aperture integral imaging under partial occlusions," *IEEE Access*, vol. 7, pp. 1052–1067, 2019, doi: 10.1109/ACCESS.2018.2886235.

[22] Z. Pei *et al.*, "Occluded-object 3D reconstruction using camera array synthetic aperture imaging," *Sensors*, vol. 19, no. 3, Art. no. 607, Jan. 2019, doi: 10.3390/s19030607.

[23] Z. Pei, M. Jin, Y. Zhang, M. Ma, and Y.-H. Yang, "All-in-focus synthetic aperture imaging using generative adversarial network-based semantic inpainting," *Pattern Recognit.*, vol. 111, Art. no. 107669, Mar. 2021, doi: 10.1016/j.patcog.2020.107669.

[24] I. Kurmi, D. C. Schedl, and O. Bimber, "Airborne optical sectioning," *J. Imaging*, vol. 4, no. 8, Art. no. 102, Aug. 2018, doi: 10.3390/jimaging4080102.

[25] O. Bimber, I. Kurmi, and D. C. Schedl, "Synthetic aperture imaging with drones," *IEEE Comput. Graph. Appl.*, vol. 39, no. 3, pp. 8–15, May 2019, doi: 10.1109/MCG.2019.2896024.

[26] I. Kurmi, D. C. Schedl, and O. Bimber, "A statistical view on synthetic aperture imaging for occlusion removal," *IEEE Sensors J.*, vol. 19, no. 20, pp. 9374–9383, Oct. 2019, doi: 10.1109/JSEN.2019.2922731.

[27] I. Kurmi, D. C. Schedl, and O. Bimber, "Thermal airborne optical sectioning," *Remote Sens.*, vol. 11, no. 14, Art. no. 1668, Jul. 2019, doi: 10.3390/rs11141668.

[28] D. C. Schedl, I. Kurmi, and O. Bimber, "Search and rescue with airborne optical sectioning," *Nat. Mach. Intell.*, vol. 2, no. 12, pp. 783–790, Nov. 2020, doi: 10.1038/s42256-020-00261-3.

[29] D. C. Schedl, I. Kurmi, and O. Bimber, "Airborne optical sectioning for nesting observation," *Sci. Rep.*, vol. 10, Art. no. 7254, Apr. 2020, doi: 10.1038/s41598-020-63317-9.

[30] D. C. Schedl, I. Kurmi, and O. Bimber, "An autonomous drone for search and rescue in forests using airborne optical sectioning," *Sci. Robot.*, vol. 6, no. 55, Art. no. eabg1188, Jun. 2021, doi: 10.1126/scirobotics.abg1188.

[31] I. Kurmi, D. C. Schedl, and O. Bimber, "Combined person classification with airborne optical sectioning," *Sci. Rep.*, vol. 12, Art. no. 3804, Mar. 2022, doi: 10.1038/s41598-022-07733-z.

[32] F. Seits, I. Kurmi, R. J. A. A. Nathan, R. Ortner, and O. Bimber, "On the role of field of view for occlusion removal with airborne optical sectioning," arXiv:2204.13371, Apr. 2022, doi: 10.48550/arXiv.2204.13371.

[33] R. J. A. A. Nathan, I. Kurmi, D. C. Schedl, and O. Bimber, "Through-foliage tracking with airborne optical sectioning," *J. Remote Sens.*, vol. 2022, Art. no. 9812765, 2022, doi: 10.34133/2022/9812765.

[34] R. J. A. A. Nathan and O. Bimber, "Synthetic aperture anomaly imaging for through-foliage target detection," *Remote Sens.*, vol. 15, no. 18, Art. no. 4369, Sep. 2023, doi: 10.3390/rs15184369.

[35] R. J. A. A. Nathan, I. Kurmi, and O. Bimber, "Drone swarm strategy for the detection and tracking of occluded targets in complex environments," *Commun. Eng.*, vol. 2, Art. no. 55, Aug. 2023, doi: 10.1038/s44172-023-00104-0.

[36] R. J. A. A. Nathan, S. Strand, D. Shutin, and O. Bimber, "Reciprocal visibility for guided occlusion removal with drones," *IEEE Geosci. Remote Sens. Lett.*, vol. 21, pp. 1–5, 2024, doi: 10.1109/LGRS.2024.3451486.

[37] R. Kerschner, R. J. A. A. Nathan, R. K. Mantiuk, and O. Bimber, "Stereoscopic depth perception through foliage," *Sci. Rep.*, vol. 14, Art. no. 23056, Oct. 2024, doi: 10.1038/s41598-024-74666-0.

[38] M. Youssef, J. Peng, and O. Bimber, "DeepForest: Sensing into self-occluding volumes of vegetation with aerial imaging," *J. Remote Sens.*, vol. 5, Art. no. 0907, 2025, doi: 10.34133/remotesensing.0907.

[39] O. Bimber *et al.*, "How robot dogs see the unseeable: Improving visual interpretability via peering for exploratory robots," *Sci. Robot.*, vol. 11, no. 118, Art. no. eaed8577, Sep. 2026, doi: 10.1126/scirobotics.aed8577.

[40] M. Youssef and O. Bimber, "Vision-reasoning-guided occlusion removal from light fields," arXiv:2606.19985v2, Aug. 2026, doi: 10.48550/arXiv.2606.19985.

[41] T. Xue, M. Rubinstein, C. Liu, and W. T. Freeman, "A computational approach for obstruction-free photography," *ACM Trans. Graph.*, vol. 34, no. 4, pp. 1–11, Jul. 2015, doi: 10.1145/2766940.

[42] Y.-L. Liu, W.-S. Lai, M.-H. Yang, Y.-Y. Chuang, and J.-B. Huang, "Learning to see through obstructions," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Seattle, WA, USA, Jun. 2020, pp. 14203–14212, doi: 10.1109/CVPR42600.2020.01422.

[43] Y.-L. Liu, W.-S. Lai, M.-H. Yang, Y.-Y. Chuang, and J.-B. Huang, "Learning to see through obstructions with layered decomposition," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 44, no. 11, pp. 8387–8402, Nov. 2022, doi: 10.1109/TPAMI.2021.3111847.

[44] Y. Wang, T. Wu, J. Yang, L. Wang, W. An, and Y. Guo, "DeOccNet: Learning to see through foreground occlusions in light fields," in *Proc. IEEE Winter Conf. Appl. Comput. Vis. (WACV)*, Snowmass Village, CO, USA, Mar. 2020, pp. 118–127, doi: 10.1109/WACV45572.2020.9093448.

[45] Y. Li *et al.*, "Mask4D: 4D convolution network for light field occlusion removal," in *Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)*, Toronto, ON, Canada, Jun. 2021, pp. 2480–2484, doi: 10.1109/ICASSP39728.2021.9413449.

[46] S. Zhang, Z. Shen, and Y. Lin, "Removing foreground occlusions in light field using micro-lens dynamic filter," in *Proc. 30th Int. Joint Conf. Artif. Intell. (IJCAI)*, Montreal, Canada, Aug. 2021, pp. 1302–1308, doi: 10.24963/ijcai.2021/180.

[47] X. Wang, J. Liu, S. Chen, and G. Wei, "Effective light field de-occlusion network based on Swin Transformer," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 33, no. 6, pp. 2590–2599, Jun. 2023, doi: 10.1109/TCSVT.2022.3226227.

[48] J. Hur, J. Y. Lee, J. Choi, and J. Kim, "I see-through you: A framework for removing foreground occlusion in both sparse and dense light field images," in *Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV)*, Waikoloa, HI, USA, Jan. 2023, pp. 229–238, doi: 10.1109/WACV56688.2023.00031.

[49] S. Zhang, S. Chang, Z. Shi, and Y. Lin, "Progressive multi-plane images construction for light field occlusion removal," *IEEE Trans. Vis. Comput. Graph.*, vol. 31, no. 10, pp. 8012–8025, Oct. 2025, doi: 10.1109/TVCG.2025.3561374.

[50] M. F. Senussi, M. Abdalla, M. S. Kasem, M. Mahmoud, B. Yagoub, and H.-S. Kang, "A comprehensive review on light field occlusion removal: Trends, challenges, and future directions," *IEEE Access*, vol. 13, pp. 42472–42493, 2025, doi: 10.1109/ACCESS.2025.3548133.

[51] C. Zhu, R. Wan, Y. Tang, and B. Shi, "Occlusion-free scene recovery via neural radiance fields," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Vancouver, BC, Canada, Jun. 2023, pp. 20722–20731, doi: 10.1109/CVPR52729.2023.01985.

[52] A. Mirzaei *et al.*, "SPIn-NeRF: Multiview segmentation and perceptual inpainting with neural radiance fields," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Vancouver, BC, Canada, Jun. 2023, pp. 20669–20679, doi: 10.1109/CVPR52729.2023.01980.

[53] W. Liu, Z. Xiong, X. Li, and N. Jacobs, "DeclutterNeRF: Generative-free 3D scene recovery for occlusion removal," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, Nashville, TN, USA, Jun. 2025, pp. 380–390, doi: 10.1109/CVPRW67362.2025.00042.

[54] A. Kirillov *et al.*, "Segment anything," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, Paris, France, Oct. 2023, pp. 3992–4003, doi: 10.1109/ICCV51070.2023.00371.

[55] L. Yu, X. Zhang, W. Liao, W. Yang, and G.-S. Xia, "Learning to see through with events," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 45, no. 7, pp. 8660–8678, Jul. 2023, doi: 10.1109/TPAMI.2022.3227448.

[56] R. C. Bolles, H. H. Baker, and D. H. Marimont, "Epipolar-plane image analysis: An approach to determining structure from motion," *Int. J. Comput. Vis.*, vol. 1, no. 1, pp. 7–55, 1987, doi: 10.1007/BF00128525.

[57] M. Okutomi and T. Kanade, "A multiple-baseline stereo," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 15, no. 4, pp. 353–363, Apr. 1993, doi: 10.1109/34.206955.

[58] R. T. Collins, "A space-sweep approach to true multi-image matching," in *Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. (CVPR)*, San Francisco, CA, USA, 1996, pp. 358–363, doi: 10.1109/CVPR.1996.517097.

[59] S. M. Seitz and C. R. Dyer, "Photorealistic scene reconstruction by voxel coloring," *Int. J. Comput. Vis.*, vol. 35, no. 2, pp. 151–173, Nov. 1999, doi: 10.1023/A:1008176507526.

[60] S. B. Kang, R. Szeliski, and J. Chai, "Handling occlusions in dense multi-view stereo," in *Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Kauai, HI, USA, 2001, vol. 1, pp. I-103–I-110, doi: 10.1109/CVPR.2001.990462.

[61] A. Criminisi, S. B. Kang, R. Swaminathan, R. Szeliski, and P. Anandan, "Extracting layers and analyzing their specular properties using epipolar-plane-image analysis," *Comput. Vis. Image Underst.*, vol. 97, no. 1, pp. 51–85, Jan. 2005, doi: 10.1016/j.cviu.2004.06.001.

[62] S. Wanner and B. Goldluecke, "Globally consistent depth labeling of 4D light fields," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Providence, RI, USA, Jun. 2012, pp. 41–48, doi: 10.1109/CVPR.2012.6247656.

[63] S. Wanner and B. Goldluecke, "Variational light field analysis for disparity estimation and super-resolution," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 36, no. 3, pp. 606–619, Mar. 2014, doi: 10.1109/TPAMI.2013.147.

[64] C. Kim, H. Zimmer, Y. Pritch, A. Sorkine-Hornung, and M. Gross, "Scene reconstruction from high spatio-angular resolution light fields," *ACM Trans. Graph.*, vol. 32, no. 4, pp. 1–12, Jul. 2013, doi: 10.1145/2461912.2461926.

[65] T.-C. Wang, A. A. Efros, and R. Ramamoorthi, "Occlusion-aware depth estimation using light-field cameras," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, Santiago, Chile, Dec. 2015, pp. 3487–3495, doi: 10.1109/ICCV.2015.398.

[66] J. L. Schönberger and J.-M. Frahm, "Structure-from-motion revisited," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Las Vegas, NV, USA, Jun. 2016, pp. 4104–4113, doi: 10.1109/CVPR.2016.445.

[67] J. L. Schönberger, E. Zheng, J.-M. Frahm, and M. Pollefeys, "Pixelwise view selection for unstructured multi-view stereo," in *Computer Vision – ECCV 2016* (Lecture Notes in Computer Science). Cham, Switzerland: Springer, 2016, pp. 501–518, doi: 10.1007/978-3-319-46487-9_31.

[68] B. Mildenhall *et al.*, "Local light field fusion: Practical view synthesis with prescriptive sampling guidelines," *ACM Trans. Graph.*, vol. 38, no. 4, pp. 1–14, Jul. 2019, doi: 10.1145/3306346.3322980.

[69] Y. Wang, L. Wang, Z. Liang, J. Yang, W. An, and Y. Guo, "Occlusion-aware cost constructor for light field depth estimation," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, New Orleans, LA, USA, Jun. 2022, pp. 19777–19786, doi: 10.1109/CVPR52688.2022.01919.

[70] J. Hemming, J. Ruizendaal, J. W. Hofstee, and E. J. van Henten, "Fruit detectability analysis for different camera positions in sweet-pepper," *Sensors*, vol. 14, no. 4, pp. 6032–6044, Mar. 2014, doi: 10.3390/s140406032.

[71] G. Polder and J. W. Hofstee, "Phenotyping large tomato plants in the greenhouse using a 3D light-field camera," presented at the *ASABE Annu. Int. Meeting*, Jul. 2014, pp. 1–7, doi: 10.13031/aim.20141882255.

[72] M. Stein, S. Bargoti, and J. Underwood, "Image based mango fruit detection, localisation and yield estimation using multiple view geometry," *Sensors*, vol. 16, no. 11, Art. no. 1915, Nov. 2016, doi: 10.3390/s16111915.

[73] C. McCool, I. Sa, F. Dayoub, C. Lehnert, T. Perez, and B. Upcroft, "Visual detection of occluded crop: For automated harvesting," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, Stockholm, Sweden, May 2016, pp. 2506–2512, doi: 10.1109/ICRA.2016.7487405.

[74] R. Schima *et al.*, "Imagine all the plants: Evaluation of a light-field camera for on-site crop growth monitoring," *Remote Sens.*, vol. 8, no. 10, Art. no. 823, Oct. 2016, doi: 10.3390/rs8100823.

[75] P. Kurtser and Y. Edan, "Statistical models for fruit detectability: Spatial and temporal analyses of sweet peppers," *Biosyst. Eng.*, vol. 171, pp. 272–289, Jul. 2018, doi: 10.1016/j.biosystemseng.2018.04.017.

[76] X. Liu *et al.*, "Robust fruit counting: Combining deep learning, tracking, and structure from motion," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Madrid, Spain, Oct. 2018, pp. 1045–1052, doi: 10.1109/IROS.2018.8594239.

[77] C. Lehnert, D. Tsai, A. Eriksson, and C. McCool, "3D move to see: Multi-perspective visual servoing towards the next best view within unstructured and occluded environments," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Macau, China, Nov. 2019, pp. 3890–3897, doi: 10.1109/IROS40897.2019.8967918.

[78] J. Gené-Mola *et al.*, "Fruit detection and 3D location using instance segmentation neural networks and structure-from-motion photogrammetry," *Comput. Electron. Agric.*, vol. 169, Art. no. 105165, Feb. 2020, doi: 10.1016/j.compag.2019.105165.

[79] B. Parr, M. Legg, S. Bradley, and F. Alam, "Occluded grape cluster detection and vine canopy visualisation using an ultrasonic phased array," *Sensors*, vol. 21, no. 6, Art. no. 2182, Mar. 2021, doi: 10.3390/s21062182.

[80] J. Gené-Mola *et al.*, "Looking behind occlusions: A study on amodal segmentation for robust on-tree apple fruit size estimation," *Comput. Electron. Agric.*, vol. 209, Art. no. 107854, Jun. 2023, doi: 10.1016/j.compag.2023.107854.

[81] L. Meyer, A. Gilson, U. Schmid, and M. Stamminger, "FruitNeRF: A unified neural radiance field based fruit counting framework," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Abu Dhabi, UAE, Oct. 2024, pp. 1–8, doi: 10.1109/IROS58592.2024.10802065.

[82] L. Meyer, A.-T. Ardelean, T. Weyrich, and M. Stamminger, "FruitNeRF++: A generalized multi-fruit counting method utilizing contrastive learning and neural radiance fields," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Hangzhou, China, Oct. 2025, pp. 1717–1724, doi: 10.1109/IROS60139.2025.11247341.

[83] H. Gan, W. S. Lee, V. Alchanatis, R. Ehsani, and J. K. Schueller, "Immature green citrus fruit detection using color and thermal images," *Comput. Electron. Agric.*, vol. 152, pp. 117–125, Sep. 2018, doi: 10.1016/j.compag.2018.07.011.

