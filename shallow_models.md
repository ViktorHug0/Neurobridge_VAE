Deep Models, Shallow Alignment: Uncovering the Granularity Mismatch in Neural Decoding

Yang Du^{1} Siyuan Dai^{1} Yonghao Song^{2} Paul M. Thompson^{3}
Haoteng Tang^{4} Liang Zhan^{1}

^{1}Dept. of Electrical & Computer Engineering, University of Pittsburgh, USA
^{2}Dept. of Biomedical Engineering, Tsinghua University, China
^{3}Dept. of Neurology, University of Southern California, USA
^{4}Dept. of Computer Science, University of Texas Rio Grande Valley, USA
liang.zhan@pitt.edu

###### Abstract

Neural visual decoding is a central problem in brain–computer interface research, aiming to reconstruct human visual perception and to elucidate the structure of neural representations. However, existing approaches overlook a fundamental granularity mismatch between human and machine vision, where deep vision models emphasize semantic invariance by suppressing local texture information, whereas neural signals preserve an intricate mixture of low-level visual attributes and high-level semantic content. To address this mismatch, we propose Shallow Alignment, a novel contrastive learning strategy that aligns neural signals with intermediate representations of visual encoders rather than their final outputs, thereby striking a better balance between low-level texture details and high-level semantic features. Extensive experiments across multiple benchmarks demonstrate that Shallow Alignment significantly outperforms standard final-layer alignment, with performance gains ranging from 22% to 58% across diverse vision backbones. Notably, our approach effectively unlocks the scaling law in neural visual decoding, enabling decoding performance to scale predictably with the capacity of pre-trained vision backbones. We further conduct systematic empirical analyses to shed light on the mechanisms underlying the observed performance gains.

## 1 Introduction

Understanding how the brain encodes visual information is a fundamental problem in both neuroscience and artificial intelligence*[18, 37, 28, 40, 25]*. Recent advances in brain-computer interfaces, particularly studies based on electroencephalography (EEG) and magnetoencephalography (MEG), have demonstrated the feasibility of decoding visual stimuli directly from neural activity *[34, 15, 17, 33]*. This task, commonly referred to as neural visual decoding, aims to retrieve perceived visual stimuli from non-invasive neural recordings. The core challenge lies in learning an alignment function that can translate high-dimensional, noisy neural dynamics into structured visual representations.

The prevailing paradigm in neural visual decoding adopts large-scale pre-trained vision models (e.g., CLIP) as feature extractors, aligning neural signals to visual representations via contrastive learning on the final-layer embeddings of these models *[32, 23, 43, 39, 44]*. However, these approaches overlook a fundamental granularity mismatch between human and machine vision. The inductive bias of contemporary vision models aims to maximize semantic invariance, systematically discarding local variations to optimize for categorization *[21, 30, 20]*. In contrast, visually evoked neural responses are not confined to a single level of abstraction. Instead, they encode information across multiple representational scales, ranging from low-level attributes such as contours, color, and spatial frequency to high-level semantic content. *[7, 3, 6, 11]*. As a result, aligning neural signals to the final, most abstract layer of vision models forces a complex, multi-scale neural representation to match a

---

![[Deep_Models_Shallow_Alignment_p2_img1.jpeg]]
Figure 1: Overview of the proposed Shallow Alignment framework. The model aligns neural signals with intermediate visual representations to mitigate information-lossy alignment at high semantic levels.

texture-insensitive semantic embedding. This granularity mismatch introduces significant ambiguity into the contrastive objective, as distinct neural patterns with differing low-level characteristics may be mapped to similar high-level visual representations. Consequently, the resulting supervision signal is insufficiently informative to guide effective neural representation learning.

To bridge this gap, we propose Shallow Alignment, a strategy that shifts the neural decoding target from the final output layer to the intermediate layers of vision encoders. We posit that intermediate representations offer a more appropriate granularity match for non-invasive neural signals, balancing high-level semantic abstraction with the retention of structural fidelity.

Our contributions are summarized as follows:

- We propose Shallow Alignment, a neural visual decoding framework that mitigates the granularity mismatch between neural signals and vision encoders, yielding consistent improvements across multiple benchmarks.
- We resolve a Depth-Capacity Paradox, where increased semantic abstraction in deeper layers hinders effective alignment. By addressing this, we unlock the scaling law for neural decoding, enabling performance to scale with model capacity.
- Through extensive empirical analysis showing superior alignment with intermediate layers, we provide evidence for a shared hierarchical granularity between human visual processing and deep vision models.

# 2 Related Work

# 2.1 Semantic Granularity in Neural Decoding

Recent advancements in neural decoding have predominantly focused on aligning neural signals with the latent spaces of large-scale pre-trained vision models. [32] introduced a contrastive learning approach that incorporates plug-and-play self-attention and graph attention modules to capture spatial correlations in EEG electrodes for effective image decoding. [23] proposed an end-to-end zero-shot framework that utilizes a tailored EEG encoder named Adaptive Thinking Mapper (ATM)

---

to align EEG signals with CLIP embeddings. Leveraging contrastive learning, *[43]* developed a method to align EEG signals not only with images but also with generated captions and depth maps, demonstrating that complementary multimodal information enhances brain signal decoding. However, emerging evidence suggests that maximizing semantic abstraction does not necessarily yield optimal decoding performance. For instance, *[39]* identified a ”System GAP” between human perception and digital stimuli, proposing an Uncertainty-Aware Blur Prior (UBP) that improves alignment by dynamically adjusting the blur radius based on sample uncertainty. Similarly, *[44]* introduced NeuroBridge leverages Cognitive Prior Augmentation to simulate perceptual variability via image transformations, including Gaussian blur, Gaussian noise, mosaic effects, and low-resolution downsampling. Notably, they also reported a counter-intuitive phenomenon where architecturally simpler backbones (e.g., ResNet-50) often exhibit competitive or even superior performance compared to large-scale foundation models (e.g., ViT) under standard settings. This observation motivates our hypothesis that the key to alignment lies not in maximizing semantic level of the image or neural representation, but in precisely calibrating the representational granularity between modalities.

### 2.2 Hierarchical Representations in Human and Artificial Vision

Neuroscience has established that the primate visual cortex operates via a cascade of processing stages, where early areas (e.g., V1, V2) encode low-level features like spatial frequencies and orientation, while high-level object semantics emerge only in later stages (e.g., IT cortex)*[7, 6]*. An interesting parallelism exists in deep artificial neural networks, which evolve from detecting low-level primitives in shallow layers to representing abstract concepts in deep layers*[41, 26]*. In addition, non-invasive neural recordings (e.g., EEG/MEG) are inherently noisy and characterized by sparse information density. They are susceptible to multiple sources of contamination, including environmental interference, muscle artifacts, and intrinsic biological variability, leading to substantially reduced signal-to-noise ratios*[13, 35]*. We posit that these create a fundamental granularity mismatch, where the intricate feature granularity of the brain signals is incompatible with the highly abstract, semantically compressed representations at the final output of the model.

### 2.3 Granularity Balance in Intermediate Representations

Prior research has demonstrated the utility of intermediate layers for diverse tasks, ranging from elucidating training dynamics *[1]* to enhancing transfer learning *[9]* and improving robustness against distribution shifts *[22, 36]*. Crucially, these works point to a fundamental trade-off in feature evolution. As networks deepen, they undergo Neural Collapse *[30]*, where class representations inevitably collapse toward their mean centers in the final layer to maximize separability. Although advantageous for classification, this abstraction process severely compresses the feature manifold and reduces its intrinsic dimensionality *[2]*, filtering out the structural details critical for fidelity reconstruction. In contrast, intermediate layers maintain a Granularity Balance: they possess sufficient semantic density to distinguish concepts while retaining the high intrinsic dimensionality and structural redundancy. We argue that this characteristic makes intermediate layers, rather than the collapsed final output, superior alignment target for neural visual decoding.

## 3 Methodology

As illustrated in Figure 1, we propose Shallow Alignment, which mitigates the granularity mismatch between neural signals and visual representations.

---

3.1 Problem Definition

We formulate neural visual decoding as a cross-modal representation alignment problem between neural signals and visual stimuli. Let $\mathcal{D}=(x_{N}^{(k)},x_{I}^{(k)})_{k=1}^{M}$ denote a paired dataset of $M$ samples, where $x_{N}^{(k)}\in\mathbb{R}^{C\times T}$ represents a non-invasive neural signal with $C$ channels and $T$ time points, and $x_{I}^{(k)}\in\mathbb{R}^{H\times W\times 3}$ denotes the corresponding visual stimulus. The objective is to learn a neural encoder $f_{\theta}:\mathcal{X}_{N}\rightarrow\mathcal{Z}$ that maps raw neural signals into a latent embedding space $\mathcal{Z}$, aligning them with visual representations extracted by a pre-trained vision encoder $E_{\phi}$. In conventional approaches, the neural embedding $z_{N}=f_{\theta}(x_{N})$ is aligned with the final-layer output of the vision encoder, denoted as $z_{\text{last}}=E_{\phi}(x_{I})$. However, the final-layer representation $z_{\text{last}}$ is typically a highly compressed and semantically abstract embedding in which local structural information is largely suppressed. Aligning such representations with noisy and multi-scale neural signals exacerbates a granularity mismatch between these two modalities.

### 3.2 Pretrained Vision Models

For the vision encoder $E_{\phi}$, we consider a diverse set of pre-trained visual backbones, ranging from conventional convolutional architectures to large-scale Vision Transformers. Specifically, we include ResNet-50 and ResNet-101 *[16]*, as well as Vision Transformer models of increasing capacity, including ViT-B/16 *[8]*, ViT-H/14 *[42]*, and ViT-bigG/14 *[5]*, all using pretrained weights provided by OpenCLIP*[19]*. To systematically study the effect of model scale and architecture on neural visual decoding, we further incorporate several state-of-the-art large-scale visual encoders, including DINOv2 *[29]*, EVA-02 *[10]*, and InternViT *[4]*.

### 3.3 Shallow Alignment via Intermediate Representations

To mitigate the granularity mismatch, we introduce Shallow Alignment, which shifts the alignment target from the final output to intermediate representations of the vision encoder. Consider a deep vision encoder $E_{\phi}$ composed of $L$ layers. Given an input image $x_{I}$, the encoder produces a sequence of hidden representations $\mathbf{H}=\{h^{(1)},h^{(2)},\dots,h^{(L)}\}$, where $h^{(l)}$ denotes the feature map at the $l$-th layer.

In contrast to the final embedding $z_{\text{last}}$, which is optimized for semantic invariance and is largely insensitive to local spatial variations, intermediate representations capture a more balanced level of abstraction. In particular, the representation at a selected depth $l^{*}$ preserves informative structural details while remaining sufficiently discriminative at the semantic level.

Formally, we define the target visual embedding $z_{I}$ as the pooled feature from the intermediate layer $l^{*}$:

$z_{I}=\text{Pool}\Big{(}h^{(l^{*})}(x_{I})\Big{)}\,,\quad 1\leq l^{*}<L$ (1)

where $\text{Pool}(\cdot)$ denotes a pooling operation that aggregates spatial features into a fixed-dimensional vector.

### 3.4 Linear Semantic Projector

To align the two modalities, we project the neural features $z_{N}$ and visual features $z_{I}$ into a shared latent space using linear mappings:

$\mathbf{v}=W_{N}z_{N}+b_{N},\quad\mathbf{w}=W_{I}z_{I}+b_{I},$ (2)

where

---

$W_{I}$ are learnable projection matrices, and $b_{N}$ and $b_{I}$ denote bias terms.

These linear transformations act as learnable projections that distill high-dimensional, redundant intermediate features into a latent subspace aligned with the neural manifold. By restricting the projection to be linear, we explicitly limit model capacity, encouraging the alignment performance to stem from the quality of the intermediate representations themselves rather than from expressive but potentially overfitting decoders.

### 3.5 Contrastive Objective

We employ a symmetric contrastive objective *[31]* that encourages matched neural–visual pairs $(\mathbf{v}^{(k)},\mathbf{w}^{(k)})$ to exhibit high similarity, while separating mismatched pairs within each mini-batch. The contrastive loss is defined as

$\mathcal{L}_{\mathrm{C}}$ $=-\frac{1}{2M}\sum_{k=1}^{M}\Bigg{[}\log\frac{\exp\bigl{(}\text{sim}(\mathbf{w}^{(k)},\mathbf{v}^{(k)})/\tau\bigr{)}}{\sum_{j=1}^{M}\exp\bigl{(}\text{sim}(\mathbf{w}^{(k)},\mathbf{v}^{(j)})/\tau\bigr{)}}$ (3)
$+\log\frac{\exp\bigl{(}\text{sim}(\mathbf{v}^{(k)},\mathbf{w}^{(k)})/\tau\bigr{)}}{\sum_{j=1}^{M}\exp\bigl{(}\text{sim}(\mathbf{v}^{(k)},\mathbf{w}^{(j)})/\tau\bigr{)}}\Bigg{]},$

where $\text{sim}(\cdot,\cdot)$ denotes cosine similarity, $\tau$ is a temperature hyperparameter, and $M$ denotes the number of paired samples in the batch. This bidirectional formulation enforces consistent alignment across modalities and is used as the primary training objective*[38]*.

## 4 Experiments

### 4.1 Datasets

THINGS-EEG *[12, 15]* provides high-density electroencephalography (EEG) recordings from 10 participants collected under a Rapid Serial Visual Presentation (RSVP) paradigm*[14]*. It comprises a training set of 1,654 unique object concepts and a disjoint test set of 200 concepts. For training, each concept is associated with 10 distinct images, each presented 4 times. In the test set, a single image is used per concept and repeated 80 times.

THINGS-MEG *[17]* contains magnetoencephalography (MEG) recordings from 4 participants and covers a total of 2,054 object concepts. The dataset is partitioned into 1,854 training concepts and 200 testing concepts. During training, each concept includes 12 distinct images, whereas the test set consists of 12 repeated presentations of a single image per concept.

For data preprocessing, we follow the methodology described in previous work*[39, 44]*. Comprehensive details are provided in Appendix A.1.

### 4.2 Implementation Details

All experiments are implemented in PyTorch and conducted on NVIDIA GeForce RTX 3090 GPUs. We utilize EEGProject as the neural encoder and select the channels corresponding to the overlying occipital and parietal cortex related to visual. Models are trained for 50 epochs with a batch size of 1,024 using the AdamW optimizer, with a learning rate of $1\times 10^{-4}$ and a weight decay of $1\times 10^{-4}$. For evaluation, we compute retrieval accuracy using cosine similarity between neural and image embeddings, reporting Top-1 and Top-5 accuracies under intra-subject and inter-subject settings. Reported results are averaged over five independent runs with different random seeds.

---

Table 1: Overall accuracy (%) of 200-way zero-shot retrieval on THINGS-EEG: Top-1 and Top-5.

|  Method | Metric | Sub1 | Sub2 | Sub3 | Sub4 | Sub5 | Sub6 | Sub7 | Sub8 | Sub9 | Sub10 | Avg.  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  Intra-Subject: train and test on one subject  |   |   |   |   |   |   |   |   |   |   |   |   |
|  NICE | Top-1 | 13.2 | 13.5 | 14.5 | 20.6 | 10.1 | 16.5 | 17.0 | 22.9 | 15.4 | 17.4 | 16.1  |
|   |  Top-5 | 39.5 | 40.3 | 42.7 | 52.7 | 31.5 | 44.0 | 42.1 | 56.1 | 41.6 | 45.8 | 43.6  |
|  ATM | Top-1 | 25.6 | 22.0 | 25.0 | 31.4 | 12.9 | 21.3 | 30.5 | 38.8 | 34.4 | 29.1 | 27.1  |
|   |  Top-5 | 60.4 | 54.5 | 62.4 | 60.9 | 43.0 | 51.1 | 61.5 | 72.0 | 51.5 | 63.5 | 58.1  |
|  Neural-MCRL | Top-1 | 27.5 | 28.5 | 37.0 | 35.0 | 22.5 | 31.5 | 31.5 | 42.0 | 30.5 | 37.5 | 32.4  |
|   |  Top-5 | 64.0 | 61.5 | 69.0 | 66.0 | 51.5 | 61.0 | 62.5 | 74.5 | 59.5 | 71.0 | 64.1  |
|  UBP | Top-1 | 41.2 | 51.2 | 51.2 | 51.1 | 42.2 | 57.5 | 49.0 | 58.6 | 45.1 | 61.5 | 50.9  |
|   |  Top-5 | 70.5 | 80.9 | 82.0 | 76.9 | 72.8 | 83.5 | 79.9 | 85.8 | 76.2 | 88.2 | 79.7  |
|  NeuroBridge | Top-1 | 50.0 | 63.2 | 61.6 | 61.4 | 54.8 | 69.7 | 62.7 | 71.2 | 64.0 | 73.6 | 63.2  |
|   |  Top-5 | 77.6 | 90.6 | 91.1 | 90.0 | 85.0 | 92.9 | 88.8 | 95.1 | 91.0 | 97.1 | 89.9  |
|  Ours | Top-1 | 75.0 | 87.5 | 83.2 | 79.5 | 74.6 | 89.9 | 78.5 | 86.9 | 81.3 | 89.3 | 82.6  |
|   |  Top-5 | 94.3 | 98.9 | 98.2 | 96.1 | 96.4 | 99.3 | 97.3 | 99.4 | 97.8 | 99.1 | 97.7  |
|  Inter-Subject: leave one subject out for test  |   |   |   |   |   |   |   |   |   |   |   |   |
|  NICE | Top-1 | 7.6 | 5.9 | 6.0 | 6.3 | 4.4 | 5.6 | 5.6 | 6.3 | 5.7 | 8.4 | 6.2  |
|   |  Top-5 | 22.8 | 20.5 | 22.3 | 20.7 | 18.3 | 22.2 | 19.7 | 22.0 | 17.6 | 28.3 | 21.4  |
|  ATM | Top-1 | 10.5 | 7.1 | 11.9 | 14.7 | 7.0 | 11.1 | 16.1 | 15.0 | 4.9 | 20.5 | 11.9  |
|   |  Top-5 | 26.8 | 24.8 | 33.8 | 39.4 | 23.9 | 35.8 | 43.5 | 40.3 | 22.7 | 46.5 | 33.8  |
|  Neural-MCRL | Top-1 | 13.0 | 12.0 | 14.5 | 12.5 | 11.5 | 13.5 | 14.0 | 18.5 | 13.5 | 17.0 | 14.0  |
|   |  Top-5 | 31.5 | 30.5 | 35.5 | 35.5 | 29.0 | 35.5 | 36.0 | 38.5 | 32.5 | 39.0 | 34.3  |
|  UBP | Top-1 | 11.5 | 15.5 | 9.8 | 13.0 | 8.8 | 11.7 | 10.2 | 12.2 | 15.5 | 16.0 | 12.4  |
|   |  Top-5 | 29.7 | 40.0 | 27.0 | 32.3 | 33.8 | 31.0 | 23.8 | 32.2 | 40.5 | 43.5 | 33.4  |
|  NeuroBridge | Top-1 | 23.2 | 21.2 | 13.2 | 17.0 | 14.5 | 25.0 | 15.3 | 20.1 | 13.7 | 27.2 | 19.0  |
|   |  Top-5 | 52.4 | 49.3 | 36.5 | 45.3 | 37.7 | 55.0 | 45.1 | 44.9 | 36.5 | 56.3 | 45.9  |
|  Ours | Top-1 | 23.5 | 30.6 | 10.0 | 19.5 | 18.1 | 22.7 | 18.6 | 17.3 | 23.0 | 34.4 | 21.8  |
|   |  Top-5 | 53.2 | 60.4 | 28.1 | 48.3 | 45.2 | 49.8 | 46.0 | 46.1 | 54.8 | 62.0 | 49.4  |

# 4.3 Bridging Cross-Modal Granularity Mismatch via Intermediate-Layer Alignment

We compared our Shallow Alignment strategy against state-of-the-art methods, including NICE[32], ATM[23], Neural-MCRL[24], NeuroBridge[44], on both THINGS-EEG and THINGS-MEG datasets.

As shown in Table 1, our method achieves substantial improvements across all evaluation metrics. Baseline approaches such as NICE, ATM, and Neural-MCRL primarily aim to enhance alignment with high-level semantic representations, while largely neglecting the granularity mismatch between neural signals and visual features, potentially limiting retrieval performance.

UBP and NeuroBridge improve performance to  $50.9\%$  and  $63.2\%$  Top-1 accuracy, respectively, by incorporating data augmentation strategies. Although these gains are commonly attributed to improved robustness against perceptual variability and low-level acquisition noise, the observed improvements can also be interpreted as arising from implicit granularity adaptation. By blurring or perturbing images, these methods attenuates fine-grained texture details. This reduction in visual complexity shifts the feature manifold to a lower granularity, thereby enabling a more robust match with the coarse and noisy neural recordings. However, this improved alignment comes at the expense of high-fidelity information for accurate neural decoding.

In contrast, our method explicitly aligns neural signals with intermediate visual representations,

---

![[Deep_Models_Shallow_Alignment_p7_img2.jpeg]]

![[Deep_Models_Shallow_Alignment_p7_img3.jpeg]]
Relative Layer Depth

![[Deep_Models_Shallow_Alignment_p7_img4.jpeg]]

![[Deep_Models_Shallow_Alignment_p7_img5.jpeg]]

![[Deep_Models_Shallow_Alignment_p7_img6.jpeg]]
(b)

![[Deep_Models_Shallow_Alignment_p7_img7.jpeg]]
(c)
Figure 2: Comparative analysis of representational performance between intermediate and final layers on THINGS-EEG. (a) Top 1 accuracies of intermediate features across vision backbones. Relative depth is computed as  $(\ell - 1) / (L - 1)$ . Dashed orange lines mark the Final Output performance. For ResNet models, the final feature is obtained by attention pooling of the last convolutional layer. For Transformer-based models, the final feature corresponds to the CLS token embedding from the last layer. (b) Performance comparison across architectures. The bar chart summarizes the Top-1 accuracy gap between the best-performing intermediate layer (orange) and the final output layer (blue). (c) Scaling analysis. Linear regression analysis reveals the relationship between model scale (number of parameters in ln scale) and Top-1 accuracy. Statistical significance is denoted by asterisks (\*\* for  $p &lt; 0.01$  ) or by "ns" for non-significant results ( $p &gt; 0.05$ ).

directly exploiting shared structural properties between the human visual system and deep vision models. This design leads to a more appropriate granularity match and results in a Top-1 accuracy of  $82.6\%$  under the intra-subject retrieval setting. These results indicate that addressing granularity mismatch plays a critical role in improving neural visual decoding performance. Under the inter-subject setting, our method also yields consistent gains. However, the improvement  $(2.8\%)$  Top-1 accuracy) remains limited, suggesting the presence of an additional between-subject granularity mismatch that is not fully addressed.

A consistent trend is observed on the THINGS-MEG dataset (Table 2). Under the intra-subject protocol, our method achieves  $48.0\%$  Top-1 accuracy and  $74.4\%$  Top-5 accuracy, substantially outperforming NeuroBridge  $(32.2\% / 60.8\%)$  and UBP  $(26.7\% / 55.2\%)$ . Nevertheless, performance remains low for all methods in the inter-subject setting, which may reflect a more pronounced between-subject granularity mismatch in MEG recordings.

---

Table 2: Overall accuracy (%) of 200-way zero-shot retrieval on THINGS-MEG: Top-1 and Top-5.

|  Method | Metric | Sub1 | Sub2 | Sub3 | Sub4 | Avg.  |
| --- | --- | --- | --- | --- | --- | --- |
|  Intra-subject: train and test on the same subject  |   |   |   |   |   |   |
|  UBP | Top-1 | 15.0 | 46.0 | 27.3 | 18.5 | 26.7  |
|   |  Top-5 | 38.0 | 80.5 | 59.0 | 43.5 | 55.2  |
|  NeuroBridge | Top-1 | 16.5 | 53.7 | 40.4 | 18.1 | 32.2  |
|   |  Top-5 | 41.6 | 85.3 | 73.2 | 43.1 | 60.8  |
|  Ours | Top-1 | 25.5 | 81.9 | 56.0 | 28.6 | 48.0  |
|   |  Top-5 | 54.5 | 97.4 | 87.5 | 58.3 | 74.4  |
|  Inter-subject: leave-one-subject-out (LOSO)  |   |   |   |   |   |   |
|  UBP | Top-1 | 2.0 | 1.5 | 2.7 | 2.5 | 2.2  |
|   |  Top-5 | 5.7 | 17.2 | 10.5 | 8.0 | 10.4  |
|  NeuroBridge | Top-1 | 4.3 | 3.6 | 3.0 | 2.5 | 3.4  |
|   |  Top-5 | 13.1 | 15.6 | 11.2 | 11.3 | 12.8  |
|  Ours | Top-1 | 1.3 | 6.6 | 5.4 | 1.5 | 3.7  |
|   |  Top-5 | 6.9 | 18.5 | 18.5 | 7.9 | 13.0  |

# 4.4 Unlocking Scaling Laws via Granularity-Matched Alignment

We extend our evaluation across a diverse set of vision backbones that vary in architecture, training objectives, and model scale. For relatively shallow architectures (e.g., ResNet), we evaluate representations from all layers. For ultra-deep Transformer-based models, we adopt a uniform sampling strategy, probing approximately ten evenly spaced intermediate layers to estimate layer-wise performance. The results are summarized in Figure 2.

Across models, the Top-1 retrieval accuracy exhibits a consistent inverted U-shaped trend as a function of layer depth, first increasing and then declining. This behavior aligns with the hierarchical nature of visual representations in deep networks, where early layers capture low-level texture information and deeper layers progressively encode more abstract semantic concepts. As a result, representations at different depths correspond to different degrees of semantic-texture entanglement. We argue that peak performance occurs when the granularity of the visual feature mixture most closely matches the inherent characteristics of neural signals, creating an optimal bridge for alignment. As illustrated in Figure 2(a), different models possess unique feature extraction capabilities, leading to distinct dynamic balances between texture and semantic information across their layers. For instance, ViT-H/14 achieves peak performance at approximately  $35.5\%$  of its relative depth, whereas InternViT peaks at around  $60\%$ . Detailed results are provided in Appendix B.1.

As shown in Figure 2(b), conventional alignment using only the final output layer reveals a counterintuitive trend: increasing model size does not necessarily improve neural decoding performance. In fact, large-scale models often underperform due to excessive semantic abstraction at their final layers. For example, DINOv2 achieves a Top-1 accuracy of only  $17.5\%$  when aligned at its final layer, substantially lower than the performance of the much smaller ResNet-50  $(40.3\%)$ . This observation indicates that highly compressed and invariant final-layer representations are poorly matched to neural signals, as they discard fine-grained structural information. In contrast, selecting an appropriate intermediate layer fundamentally alters this behavior. When alignment is performed at the optimal depth, a clear scaling trend emerges: decoding performance consistently improves with increasing model capacity (see Figure 2(c)). The most significant gain is observed in DINOv2, which improves by  $58.4\%$  when shifting the alignment target from the final output to its optimal intermediate representation.

---

Taken together, these results indicate that the primary bottleneck in neural visual decoding is not the quality of visual representations in large models, but the choice of alignment target. By calibrating representational granularity through intermediate-layer alignment, Shallow Alignment enables effective utilization of the representational capacity of large-scale vision models.

# 4.5 Revealing Performance Trade-offs via Concept Analysis

We report Concept Accuracy, defined as the proportion of Top-5 retrieved images that share the same concept category (animals, food, vehicles, tools, clothing, or others) as the query, excluding the query image itself. Formally, for each query, we count the number of concept-matched items within the Top-5 results and normalize by  $5M$ , where  $M$  denotes the number of queries.

As illustrated in Figure 3, layer-wise analysis reveals a clear divergence between concept accuracy and retrieval accuracy. Concept accuracy increases monotonically with network depth, reflecting progressively stronger semantic abstraction. In contrast, Top-1 retrieval accuracy follows a non-monotonic trend, peaking at an intermediate layer and declining sharply at the final layer. This behavior aligns with observations from the human visual system, which integrates mid-level visual cues (e.g., contours and texture) together with high-level semantic information, rather than collapsing representations purely to category identity. As a result, improvements in semantic consistency alone do not guarantee better retrieval performance. Instead, retrieval accuracy is maximized when visual representations preserve fine-grained structural details while maintaining sufficient semantic coherence.

These results indicate that the performance degradation observed at the final layer is not caused by insufficient semantic representation, but by the loss of texture-level information in highly abstract visual features. By retaining such details, intermediate representations achieve a more favorable balance between semantic correctness and discriminative precision. Figure 4 presents representative examples of the Top-5 retrieval results.

![[Deep_Models_Shallow_Alignment_p9_img8.jpeg]]
Figure 3: Concept accuracy and retrieval Top-1 accuracy on THINGS-EEG versus layer depth for InternViT. Relative depth is computed as  $(\ell - 1) / (L - 1)$ .

---

![[Deep_Models_Shallow_Alignment_p10_img9.jpeg]]
Figure 4: Top-5 retrieved samples of Subject 7 on THINGS-EEG. (a) Retrieval based on the best intermediate-layer embeddings. (b) Retrieval based on final output embeddings. The red box indicates the ground-truth target.

# 4.6 Visualizing Granularity Consistency via Manifold Distributions

We employ UMAP [27] to visualize the geometric distributions of the projected neural embeddings  $\mathbf{v}$  and visual embeddings  $\mathbf{w}$  on the test set. As shown in Figure 5, when alignment is performed at the early layer or the final output layer, the embeddings from the two modalities form clearly separated clusters with distinct boundaries. This separation reflects a pronounced modality gap, indicating that the granularity of these layer visual representations is poorly matched to that of neural signals. In contrast, at the proper intermediate layer, EEG and visual embeddings substantially overlap and intermingle in the projected space. This observation suggests that intermediate representations induce a manifold structure whose granularity is more consistent with neural signals, thereby supporting more effective alignment. More visualization results are provided in Appendix B.3.

![[Deep_Models_Shallow_Alignment_p10_img10.jpeg]]
Figure 5: UMAP visualization of cross-modal alignment on THINGS-EEG (Subject 7). We illustrate the feature alignment between EEG signals and visual representations on the test set ( $M = 200$ ). Image features are extracted from selected intermediate layers versus the final output layer, aligned with EEG embeddings from the neural encoder. Gray lines connect ground-truth image-EEG pairs.

# 4.7 Validating Semantic Collapse via Linear Projection Ablation

We conduct an ablation study on the linear projection to examine its role in alignment. As shown in Figure 6, we observe a clear contrast in the effectiveness of linear projection across different representation depths. When applied to the final-layer visual embeddings, introducing a learnable

---

linear projector yields only marginal performance improvements. In contrast, linear projection substantially improves alignment performance when applied to intermediate representations. The most pronounced gain is observed for ResNet-50, where Top-1 accuracy increases from  $28.8\%$  to  $67.7\%$ , corresponding to an absolute improvement of nearly  $40\%$ . The limited benefit at the final layer suggests that these representations have undergone severe semantic collapse, such that a simple linear transformation is insufficient to recover the structural information necessary for effective alignment. Conversely, the significant improvements observed at intermediate layers indicate that these representations retain richer structural fidelity. This richness allows the linear layer to function as a selective filter, identifying the specific visual primitives that best match the granularity of neural signals while discarding task-irrelevant noise.

![[Deep_Models_Shallow_Alignment_p11_img11.jpeg]]
Figure 6: Top 1 accuracy comparison across different projector types on THINGS-EEG. Results are shown for the best intermediate layer and the final output layer.

![[Deep_Models_Shallow_Alignment_p11_img12.jpeg]]

# 4.8 Evaluating Encoder Efficacy via Architectural Comparison

To assess the impact of encoder architecture on cross-modal alignment performance, we examine combinations of five EEG encoders and eight vision encoders (See details in Appendix A.2). Our empirical evaluation highlights the superior efficacy of EEGProject. Despite its architectural simplicity, it outperforms more complex baselines (e.g., EEGNet, ATM).

As shown in Figure 7, the lightweight EEGProject model consistently attains the highest performance across all vision backbones, achieving the highest average accuracy of  $73.1\%$ . This outcome aligns with the intrinsic nature of EEG data, which is characterized by a low signal-to-noise ratio and data scarcity. In this regime, complex architectures struggle to generalize. For instance, despite its high capacity, the widely used EEGNet averages only  $53.8\%$  accuracy, lagging behind our simpler MLP-based approach by nearly  $20\%$ .

Conversely, EEGProject imposes a tighter information bottleneck that forces the encoder to distill the most discriminative features while suppressing task-irrelevant artifacts. This results in representations that are better aligned with the visual embedding space. These findings suggest that, for neural visual decoding, a simple architecture can be more effective when paired with appropriately chosen visual representations.

