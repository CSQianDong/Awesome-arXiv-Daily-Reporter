# User Long-Term Multi-Interest Retrieval Model for Recommendation 

**Authors**: Yue Meng, Cheng Guo, Xiaohui Hu, Honghu Deng, Yi Cao, Tong Liu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.10097)  

**Abstract**: User behavior sequence modeling, which captures user interest from rich historical interactions, is pivotal for industrial recommendation systems. Despite breakthroughs in ranking-stage models capable of leveraging ultra-long behavior sequences with length scaling up to thousands, existing retrieval models remain constrained to sequences of hundreds of behaviors due to two main challenges. One is strict latency budget imposed by real-time service over large-scale candidate pool. The other is the absence of target-aware mechanisms and cross-interaction architectures, which prevent utilizing ranking-like techniques to simplify long sequence modeling. To address these limitations, we propose a new framework named User Long-term Multi-Interest Retrieval Model(ULIM), which enables thousand-scale behavior modeling in retrieval stages. ULIM includes two novel components: 1)Category-Aware Hierarchical Dual-Interest Learning partitions long behavior sequences into multiple category-aware subsequences representing multi-interest and jointly optimizes long-term and short-term interests within specific interest cluster. 2)Pointer-Enhanced Cascaded Category-to-Item Retrieval introduces Pointer-Generator Interest Network(PGIN) for next-category prediction, followed by next-item retrieval upon the top-K predicted categories. Comprehensive experiments on Taobao dataset show that ULIM achieves substantial improvement over state-of-the-art methods, and brings 5.54% clicks, 11.01% orders and 4.03% GMV lift for Taobaomiaosha, a notable mini-app of Taobao. 

---
# Enhancing Clinical Text Classification via Fine-Tuned DRAGON Longformer Models 

**Authors**: Mingchuan Yang, Ziyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09470)  

**Abstract**: This study explores the optimization of the DRAGON Longformer base model for clinical text classification, specifically targeting the binary classification of medical case descriptions. A dataset of 500 clinical cases containing structured medical observations was used, with 400 cases for training and 100 for validation. Enhancements to the pre-trained joeranbosma/dragon-longformer-base-mixed-domain model included hyperparameter tuning, domain-specific preprocessing, and architectural adjustments. Key modifications involved increasing sequence length from 512 to 1024 tokens, adjusting learning rates from 1e-05 to 5e-06, extending training epochs from 5 to 8, and incorporating specialized medical terminology. The optimized model achieved notable performance gains: accuracy improved from 72.0% to 85.2%, precision from 68.0% to 84.1%, recall from 75.0% to 86.3%, and F1-score from 71.0% to 85.2%. Statistical analysis confirmed the significance of these improvements (p < .001). The model demonstrated enhanced capability in interpreting medical terminology, anatomical measurements, and clinical observations. These findings contribute to domain-specific language model research and offer practical implications for clinical natural language processing applications. The optimized model's strong performance across diverse medical conditions underscores its potential for broad use in healthcare settings. 

---
# Context-Aware Regularization with Markovian Integration for Attention-Based Nucleotide Analysis 

**Authors**: Mohammadsaleh Refahi, Mahdi Abavisani, Bahrad A. Sokhansanj, James R. Brown, Gail Rosen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09378)  

**Abstract**: Transformers have revolutionized nucleotide sequence analysis, yet capturing long-range dependencies remains challenging. Recent studies show that autoregressive transformers often exhibit Markovian behavior by relying on fixed-length context windows for next-token prediction. However, standard self-attention mechanisms are computationally inefficient for long sequences due to their quadratic complexity and do not explicitly enforce global transition consistency.
We introduce CARMANIA (Context-Aware Regularization with Markovian Integration for Attention-Based Nucleotide Analysis), a self-supervised pretraining framework that augments next-token (NT) prediction with a transition-matrix (TM) loss. The TM loss aligns predicted token transitions with empirically derived n-gram statistics from each input sequence, encouraging the model to capture higher-order dependencies beyond local context. This integration enables CARMANIA to learn organism-specific sequence structures that reflect both evolutionary constraints and functional organization.
We evaluate CARMANIA across diverse genomic tasks, including regulatory element prediction, functional gene classification, taxonomic inference, antimicrobial resistance detection, and biosynthetic gene cluster classification. CARMANIA outperforms the previous best long-context model by at least 7 percent, matches state-of-the-art on shorter sequences (exceeding prior results on 20 out of 40 tasks while running approximately 2.5 times faster), and shows particularly strong improvements on enhancer and housekeeping gene classification tasks, including up to a 34 percent absolute gain in Matthews correlation coefficient (MCC) for enhancer prediction. The TM loss boosts accuracy in 33 of 40 tasks, especially where local motifs or regulatory patterns drive prediction. 

---
# Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation 

**Authors**: Jialong Mai, Xiaofen Xing, Yawei Li, Zhipeng Li, Jingyuan Xing, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09076)  

**Abstract**: Recent research has focused on applying speech large language model (SLLM) to improve speech emotion recognition (SER). However, the inherently high frame rate in speech modality severely limits the signal processing and understanding capabilities of SLLM. For example, a SLLM with a 4K context window can only process 80 seconds of audio at 50Hz feature sampling rate before reaching its capacity limit. Input token compression methods used in SLLM overlook the continuity and inertia of emotions across multiple conversation turns. This paper proposes a Dynamic Parameter Memory (DPM) mechanism with contextual semantics and sentence-level emotion encoding, enabling processing of unlimited-length audio with limited context windows in SLLM. Specifically, DPM progressively encodes sentence-level information and emotions into a temporary LoRA module during inference to effectively "memorize" the contextual information. We trained an emotion SLLM as a backbone and incorporated our DPM into inference for emotion recognition in conversation (ERC). Experimental results on the IEMOCAP dataset show that DPM significantly improves the emotion recognition capabilities of SLLM when processing long audio sequences, achieving state-of-the-art performance. 

---
# Ref-Long: Benchmarking the Long-context Referencing Capability of Long-context Language Models 

**Authors**: Junjie Wu, Gefei Gu, Yanan Zheng, Dit-Yan Yeung, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2507.09506)  

**Abstract**: Long-context language models (LCLMs) have exhibited impressive capabilities in long-context understanding tasks. Among these, long-context referencing -- a crucial task that requires LCLMs to attribute items of interest to specific parts of long-context data -- remains underexplored. To bridge this gap, this paper proposes Referencing Evaluation for Long-context Language Models (Ref-Long), a novel benchmark designed to assess the long-context referencing capability of LCLMs. Specifically, Ref-Long requires LCLMs to identify the indexes of documents that reference a specific key, emphasizing contextual relationships between the key and the documents over simple retrieval. Based on the task design, we construct three subsets ranging from synthetic to realistic scenarios to form the Ref-Long benchmark. Experimental results of 13 LCLMs reveal significant shortcomings in long-context referencing, even among advanced models like GPT-4o. To further investigate these challenges, we conduct comprehensive analyses, including human evaluations, task format adjustments, fine-tuning experiments, and error analyses, leading to several key insights. Our data and code can be found in https://github. com/wujunjie1998/Ref-Long. 

---
# Lizard: An Efficient Linearization Framework for Large Language Models 

**Authors**: Chien Van Nguyen, Ruiyi Zhang, Hanieh Deilamsalehy, Puneet Mathur, Viet Dac Lai, Haoliang Wang, Jayakumar Subramanian, Ryan A. Rossi, Trung Bui, Nikos Vlassis, Franck Dernoncourt, Thien Huu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09025)  

**Abstract**: We propose Lizard, a linearization framework that transforms pretrained Transformer-based Large Language Models (LLMs) into flexible, subquadratic architectures for infinite-context generation. Transformer-based LLMs face significant memory and computational bottlenecks as context lengths increase, due to the quadratic complexity of softmax attention and the growing key-value (KV) cache. Lizard addresses these limitations by introducing a subquadratic attention mechanism that closely approximates softmax attention while preserving the output quality. Unlike previous linearization methods, which are often limited by fixed model structures and therefore exclude gating mechanisms, Lizard incorporates a gating module inspired by recent state-of-the-art linear models. This enables adaptive memory control, supports constant-memory inference, offers strong length generalization, and allows more flexible model design. Lizard combines gated linear attention for global context compression with sliding window attention enhanced by meta memory, forming a hybrid mechanism that captures both long-range dependencies and fine-grained local interactions. Moreover, we introduce a hardware-aware algorithm that accelerates the training speed of our models. Extensive experiments show that Lizard achieves near-lossless recovery of the teacher model's performance across standard language modeling tasks, while significantly outperforming previous linearization methods. On the 5-shot MMLU benchmark, Lizard improves over prior models by 18 points and shows significant improvements on associative recall tasks. 

---
