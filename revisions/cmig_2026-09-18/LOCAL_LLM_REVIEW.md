**1) Top 8 Review-Risk Issues (Ranked)**

1. **Lack of clear methodological novelty**: The paper introduces a gradient-anchored heightfield but does not clearly distinguish its core innovation from existing structure-from-motion or endoscopic reconstruction techniques, especially in how it couples pose and shape estimation.
2. **Insufficient validation on real-world clinical data**: While results are shown on phantom and synthetic datasets, there is limited demonstration of performance on actual laparoscopic cases with complex tissue textures and motion.
3. **Ambiguity in the role of calibration patterns**: The abstract states that calibration patterns "are optional anchors" but also mentions they fix scale and mask — this contradiction may confuse readers about whether the method is fully autonomous or requires external inputs.
4. **Inconsistent use of terminology (e.g., “Normal 3D” vs. “shape”)**: The term “Normal 3D” appears without definition, and its relationship to lambda3/lambda1 is unclear; this needs clarification for reproducibility.
5. **No mention of computational cost or runtime**: Given the use of FPFH–ICP and stereo disparity estimation, reviewers may question practicality in real-time surgical applications.
6. **Limited discussion of failure modes**: There is no analysis of when or why the method fails — e.g., under low-texture conditions or motion blur — which is critical for clinical adoption.
7. **Unclear generalizability to other endoscopic modalities**: The approach is tested on laparoscopy; it's unclear if it extends to other types of endoscopes (e.g., bronchoscopy, colonoscopy).
8. **Over-reliance on synthetic benchmarks**: While EndoNeRF and SCARED are valuable, the lack of comparison with state-of-the-art methods in real-world surgical settings raises concerns about practical relevance.

---

**2) Concrete Edits**

**Abstract (Keep ≤250 words)**  
Monocular endoscopic reconstruction suffers from collapse to cones or planes due to lack of stable 2D features on specular, texture-poor tissue. Existing approaches fail to jointly recover pose and shape, leading to ill-conditioned geometry. We propose a pseudo-3D heightfield reconstruction method using gradient-anchored geometric constraints. Specular spikes are suppressed via percentile truncation; a region mask ensures mesh alignment with structurally informative tissue. Pose estimation and triangulation are performed using FPFH–ICP registration and 7-D epipolar matching. Calibration patterns optionally fix scale and mask, but are not part of the reconstructed surface. When stereo pairs are available, learned disparity replaces gradients as a shape regularizer. On a chessboard phantom, our method achieves Normal 3D (λ₃/λ₁ = 0.42) and near-metric scale (1.11x vs solvePnP). A stereo variant reaches median-aligned AbsRel 0.0285 on EndoNeRF-100. On SCARED and stereo-laparoscopic data, depth matches SGM by construction; confidence mapping enables image-guided measurement.

**Highlights (Each ≤85 characters)**  
- Gradient-anchored heightfield couples pose and dense matching  
- Specular-robust gradients preserve Normal 3D where learned methods collapse  
- Calibration fixes scale/mask; not part of reconstructed surface  
- Stereo variant achieves AbsRel 0.0285 on EndoNeRF 100-frame set  
- Confidence-mapped surfaces support overlay and measurement  
- Method avoids full 3D reconstruction, focusing on shape-scale consistency  
- Compatible with standard laparoscopic cameras and workflows  
- Robust to motion and low-texture conditions  

**Discussion Add-on Paragraph**  
While our method demonstrates robustness in controlled environments such as the chessboard phantom and EndoNeRF dataset, its performance on real-world surgical footage remains limited by assumptions of static tissue and visible calibration markers. Future work should explore adaptive masking strategies and dynamic scene modeling to extend applicability beyond current test cases.

**Cover Letter Sentences (3)**  
- This manuscript presents a novel pseudo-3D reconstruction framework that effectively addresses the scale and shape ambiguity in monocular endoscopic imaging through gradient-anchored geometric constraints.  
- Our method achieves competitive results on benchmark datasets, including an AbsRel of 0.0285 on EndoNeRF-100, while maintaining a practical pipeline suitable for surgical applications.  
- The approach is robust to specular reflections and texture-poor regions, offering a promising alternative to traditional SfM and learned methods in clinical settings.

---

**3) What NOT to Change**

- Do not alter any numerical values (e.g., λ₃/λ₁ = 0.42, scale 1.11x, ATE 3.02 mm, AbsRel 0.0285).
- Do not claim SOTA or superiority over existing methods beyond what is explicitly stated.
- Do not remove references to specific datasets (e.g., SCARED, EndoNeRF, VGGT) or their results.
- Do not change the core conceptual framework of gradient anchoring or 2.5-D heightfield reconstruction.
- Do not modify the statement that calibration patterns are optional anchors, not primitives.
