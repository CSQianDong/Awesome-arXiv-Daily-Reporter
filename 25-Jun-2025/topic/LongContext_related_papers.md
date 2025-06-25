# LLM-Driven Medical Document Analysis: Enhancing Trustworthy Pathology and Differential Diagnosis 

**Authors**: Lei Kang, Xuanshuo Fu, Oriol Ramos Terrades, Javier Vazquez-Corral, Ernest Valveny, Dimosthenis Karatzas  

**Link**: [PDF](https://arxiv.org/pdf/2506.19702)  

**Abstract**: Medical document analysis plays a crucial role in extracting essential clinical insights from unstructured healthcare records, supporting critical tasks such as differential diagnosis. Determining the most probable condition among overlapping symptoms requires precise evaluation and deep medical expertise. While recent advancements in large language models (LLMs) have significantly enhanced performance in medical document analysis, privacy concerns related to sensitive patient data limit the use of online LLMs services in clinical settings. To address these challenges, we propose a trustworthy medical document analysis platform that fine-tunes a LLaMA-v3 using low-rank adaptation, specifically optimized for differential diagnosis tasks. Our approach utilizes DDXPlus, the largest benchmark dataset for differential diagnosis, and demonstrates superior performance in pathology prediction and variable-length differential diagnosis compared to existing methods. The developed web-based platform allows users to submit their own unstructured medical documents and receive accurate, explainable diagnostic results. By incorporating advanced explainability techniques, the system ensures transparent and reliable predictions, fostering user trust and confidence. Extensive evaluations confirm that the proposed method surpasses current state-of-the-art models in predictive accuracy while offering practical utility in clinical settings. This work addresses the urgent need for reliable, explainable, and privacy-preserving artificial intelligence solutions, representing a significant advancement in intelligent medical document analysis for real-world healthcare applications. The code can be found at \href{this https URL}{this https URL}. 

---
# Radial Attention: $O(n\log n)$ Sparse Attention with Energy Decay for Long Video Generation 

**Authors**: Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, Maneesh Agrawala, Ion Stoica, Kurt Keutzer, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.19852)  

**Abstract**: Recent advances in diffusion models have enabled high-quality video generation, but the additional temporal dimension significantly increases computational costs, making training and inference on long videos prohibitively expensive. In this paper, we identify a phenomenon we term Spatiotemporal Energy Decay in video diffusion models: post-softmax attention scores diminish as spatial and temporal distance between tokens increase, akin to the physical decay of signal or waves over space and time in nature. Motivated by this, we propose Radial Attention, a scalable sparse attention mechanism with $O(n \log n)$ complexity that translates energy decay into exponentially decaying compute density, which is significantly more efficient than standard $O(n^2)$ dense attention and more expressive than linear attention. Specifically, Radial Attention employs a simple, static attention mask where each token attends to spatially nearby tokens, with the attention window size shrinking with temporal distance. Moreover, it allows pre-trained video diffusion models to extend their generation length with efficient LoRA-based fine-tuning. Extensive experiments show that Radial Attention maintains video quality across Wan2.1-14B, HunyuanVideo, and Mochi 1, achieving up to a 1.9$\times$ speedup over the original dense attention. With minimal tuning, it enables video generation up to 4$\times$ longer while reducing training costs by up to 4.4$\times$ compared to direct fine-tuning and accelerating inference by up to 3.7$\times$ compared to dense attention inference. 

---
# Mem4Nav: Boosting Vision-and-Language Navigation in Urban Environments with a Hierarchical Spatial-Cognition Long-Short Memory System 

**Authors**: Lixuan He, Haoyu Dong, Zhenxing Chen, Yangcheng Yu, Jie Feng, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.19433)  

**Abstract**: Vision-and-Language Navigation (VLN) in large-scale urban environments requires embodied agents to ground linguistic instructions in complex scenes and recall relevant experiences over extended time horizons. Prior modular pipelines offer interpretability but lack unified memory, while end-to-end (M)LLM agents excel at fusing vision and language yet remain constrained by fixed context windows and implicit spatial reasoning. We introduce \textbf{Mem4Nav}, a hierarchical spatial-cognition long-short memory system that can augment any VLN backbone. Mem4Nav fuses a sparse octree for fine-grained voxel indexing with a semantic topology graph for high-level landmark connectivity, storing both in trainable memory tokens embedded via a reversible Transformer. Long-term memory (LTM) compresses and retains historical observations at both octree and graph nodes, while short-term memory (STM) caches recent multimodal entries in relative coordinates for real-time obstacle avoidance and local planning. At each step, STM retrieval sharply prunes dynamic context, and, when deeper history is needed, LTM tokens are decoded losslessly to reconstruct past embeddings. Evaluated on Touchdown and Map2Seq across three backbones (modular, state-of-the-art VLN with prompt-based LLM, and state-of-the-art VLN with strided-attention MLLM), Mem4Nav yields 7-13 pp gains in Task Completion, sufficient SPD reduction, and >10 pp nDTW improvement. Ablations confirm the indispensability of both the hierarchical map and dual memory modules. Our codes are open-sourced via this https URL. 

---
# Video-XL-2: Towards Very Long-Video Understanding Through Task-Aware KV Sparsification 

**Authors**: Minghao Qin, Xiangrui Liu, Zhengyang Liang, Yan Shu, Huaying Yuan, Juenjie Zhou, Shitao Xiao, Bo Zhao, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19225)  

**Abstract**: Multi-modal large language models (MLLMs) models have made significant progress in video understanding over the past few years. However, processing long video inputs remains a major challenge due to high memory and computational costs. This makes it difficult for current models to achieve both strong performance and high efficiency in long video understanding. To address this challenge, we propose Video-XL-2, a novel MLLM that delivers superior cost-effectiveness for long-video understanding based on task-aware KV sparsification. The proposed framework operates with two key steps: chunk-based pre-filling and bi-level key-value decoding. Chunk-based pre-filling divides the visual token sequence into chunks, applying full attention within each chunk and sparse attention across chunks. This significantly reduces computational and memory overhead. During decoding, bi-level key-value decoding selectively reloads either dense or sparse key-values for each chunk based on its relevance to the task. This approach further improves memory efficiency and enhances the model's ability to capture fine-grained information. Video-XL-2 achieves state-of-the-art performance on various long video understanding benchmarks, outperforming existing open-source lightweight models. It also demonstrates exceptional efficiency, capable of processing over 10,000 frames on a single NVIDIA A100 (80GB) GPU and thousands of frames in just a few seconds. 

---
# eccDNAMamba: A Pre-Trained Model for Ultra-Long eccDNA Sequence Analysis 

**Authors**: Zhenke Liu, Jien Li, Ziqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18940)  

**Abstract**: Extrachromosomal circular DNA (eccDNA) plays key regulatory roles and contributes to oncogene overexpression in cancer through high-copy amplification and long-range interactions. Despite advances in modeling, no pre-trained models currently support full-length circular eccDNA for downstream analysis. Existing genomic models are either limited to single-nucleotide resolution or hindered by the inefficiency of the quadratic attention mechanism. Here, we introduce eccDNAMamba, the first bidirectional state-space encoder tailored for circular DNA sequences. It combines forward and reverse passes for full-context representation learning with linear-time complexity, and preserves circular structure through a novel augmentation strategy. Tested on two real-world datasets, eccDNAMamba achieves strong classification performance and scales to sequences up to 200 Kbp, offering a robust and efficient framework for modeling circular genomes. Our codes are available at this https URL. 

---
# Accurate, fast, cheap: Choose three. Replacing Multi-Head-Attention with Bidirectional Recurrent Attention for Long-Form ASR 

**Authors**: Martin Ratajczak, Jean-Philippe Robichaud, Jennifer Drexler Fox  

**Link**: [PDF](https://arxiv.org/pdf/2506.19761)  

**Abstract**: Long-form speech recognition is an application area of increasing research focus. ASR models based on multi-head attention (MHA) are ill-suited to long-form ASR because of their quadratic complexity in sequence length. We build on recent work that has investigated linear complexity recurrent attention (RA) layers for ASR. We find that bidirectional RA layers can match the accuracy of MHA for both short- and long-form applications. We present a strong limited-context attention (LCA) baseline, and show that RA layers are just as accurate while being more efficient. We develop a long-form training paradigm which further improves RA performance, leading to better accuracy than LCA with 44% higher throughput. We also present Direction Dropout, a novel regularization method that improves accuracy, provides fine-grained control of the accuracy/throughput trade-off of bidirectional RA, and enables a new alternating directions decoding mode with even higher throughput. 

---
# Personality Prediction from Life Stories using Language Models 

**Authors**: Rasiq Hussain, Jerry Ma, Rithik Khandelwal, Joshua Oltmanns, Mehak Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.19258)  

**Abstract**: Natural Language Processing (NLP) offers new avenues for personality assessment by leveraging rich, open-ended text, moving beyond traditional questionnaires. In this study, we address the challenge of modeling long narrative interview where each exceeds 2000 tokens so as to predict Five-Factor Model (FFM) personality traits. We propose a two-step approach: first, we extract contextual embeddings using sliding-window fine-tuning of pretrained language models; then, we apply Recurrent Neural Networks (RNNs) with attention mechanisms to integrate long-range dependencies and enhance interpretability. This hybrid method effectively bridges the strengths of pretrained transformers and sequence modeling to handle long-context data. Through ablation studies and comparisons with state-of-the-art long-context models such as LLaMA and Longformer, we demonstrate improvements in prediction accuracy, efficiency, and interpretability. Our results highlight the potential of combining language-based features with long-context modeling to advance personality assessment from life narratives. 

---
