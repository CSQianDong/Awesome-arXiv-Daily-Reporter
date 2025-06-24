# Context-Aware Scientific Knowledge Extraction on Linked Open Data using Large Language Models 

**Authors**: Sajratul Y. Rubaiat, Hasan M. Jamil  

**Link**: [PDF](https://arxiv.org/pdf/2506.17580)  

**Abstract**: The exponential growth of scientific literature challenges researchers extracting and synthesizing knowledge. Traditional search engines return many sources without direct, detailed answers, while general-purpose LLMs may offer concise responses that lack depth or omit current information. LLMs with search capabilities are also limited by context window, yielding short, incomplete answers. This paper introduces WISE (Workflow for Intelligent Scientific Knowledge Extraction), a system addressing these limits by using a structured workflow to extract, refine, and rank query-specific knowledge. WISE uses an LLM-powered, tree-based architecture to refine data, focusing on query-aligned, context-aware, and non-redundant information. Dynamic scoring and ranking prioritize unique contributions from each source, and adaptive stopping criteria minimize processing overhead. WISE delivers detailed, organized answers by systematically exploring and synthesizing knowledge from diverse sources. Experiments on HBB gene-associated diseases demonstrate WISE reduces processed text by over 80% while achieving significantly higher recall over baselines like search engines and other LLM-based approaches. ROUGE and BLEU metrics reveal WISE's output is more unique than other systems, and a novel level-based metric shows it provides more in-depth information. We also explore how the WISE workflow can be adapted for diverse domains like drug discovery, material science, and social science, enabling efficient knowledge extraction and synthesis from unstructured scientific papers and web sources. 

---
# CommVQ: Commutative Vector Quantization for KV Cache Compression 

**Authors**: Junyan Li, Yang Zhang, Muhammad Yusuf Hassan, Talha Chafekar, Tianle Cai, Zhile Ren, Pengsheng Guo, Foroozan Karimzadeh, Colorado Reed, Chong Wang, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18879)  

**Abstract**: Large Language Models (LLMs) are increasingly used in applications requiring long context lengths, but the key-value (KV) cache often becomes a memory bottleneck on GPUs as context grows. To address this, we propose Commutative Vector Quantization (CommVQ) to significantly reduce memory usage for long-context LLM inference. We first introduce additive quantization with a lightweight encoder and codebook to compress the KV cache, which can be decoded via simple matrix multiplication. To further reduce computational costs during decoding, we design the codebook to be commutative with Rotary Position Embedding (RoPE) and train it using an Expectation-Maximization (EM) algorithm. This enables efficient integration of decoding into the self-attention mechanism. Our approach achieves high accuracy with additive quantization and low overhead via the RoPE-commutative codebook. Experiments on long-context benchmarks and GSM8K show that our method reduces FP16 KV cache size by 87.5% with 2-bit quantization, while outperforming state-of-the-art KV cache quantization methods. Notably, it enables 1-bit KV cache quantization with minimal accuracy loss, allowing a LLaMA-3.1 8B model to run with a 128K context length on a single RTX 4090 GPU. The source code is available at: this https URL. 

---
# LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning 

**Authors**: Yuhao Wu, Yushi Bai, Zhiqiang Hu, Roy Ka-Wei Lee, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.18841)  

**Abstract**: Ultra-long generation by large language models (LLMs) is a widely demanded scenario, yet it remains a significant challenge due to their maximum generation length limit and overall quality degradation as sequence length increases. Previous approaches, exemplified by LongWriter, typically rely on ''teaching'', which involves supervised fine-tuning (SFT) on synthetic long-form outputs. However, this strategy heavily depends on synthetic SFT data, which is difficult and costly to construct, often lacks coherence and consistency, and tends to be overly artificial and structurally monotonous. In this work, we propose an incentivization-based approach that, starting entirely from scratch and without relying on any annotated or synthetic data, leverages reinforcement learning (RL) to foster the emergence of ultra-long, high-quality text generation capabilities in LLMs. We perform RL training starting from a base model, similar to R1-Zero, guiding it to engage in reasoning that facilitates planning and refinement during the writing process. To support this, we employ specialized reward models that steer the LLM towards improved length control, writing quality, and structural formatting. Experimental evaluations show that our LongWriter-Zero model, trained from Qwen2.5-32B, consistently outperforms traditional SFT methods on long-form writing tasks, achieving state-of-the-art results across all metrics on WritingBench and Arena-Write, and even surpassing 100B+ models such as DeepSeek R1 and Qwen3-235B. We open-source our data and model checkpoints under this https URL 

---
# PDF Retrieval Augmented Question Answering 

**Authors**: Thi Thu Uyen Hoang, Viet Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18027)  

**Abstract**: This paper presents an advancement in Question-Answering (QA) systems using a Retrieval Augmented Generation (RAG) framework to enhance information extraction from PDF files. Recognizing the richness and diversity of data within PDFs--including text, images, vector diagrams, graphs, and tables--poses unique challenges for existing QA systems primarily designed for textual content. We seek to develop a comprehensive RAG-based QA system that will effectively address complex multimodal questions, where several data types are combined in the query. This is mainly achieved by refining approaches to processing and integrating non-textual elements in PDFs into the RAG framework to derive precise and relevant answers, as well as fine-tuning large language models to better adapt to our system. We provide an in-depth experimental evaluation of our solution, demonstrating its capability to extract accurate information that can be applied to different types of content across PDFs. This work not only pushes the boundaries of retrieval-augmented QA systems but also lays a foundation for further research in multimodal data integration and processing. 

---
# QueueEDIT: Structural Self-Correction for Sequential Model Editing in LLMs 

**Authors**: Taolin Zhang, Haidong Kang, Dongyang Li, Qizhou Chen, Chengyu Wang Xiaofeng He, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2506.17864)  

**Abstract**: Recently, large language models (LLMs) have demonstrated impressive results but still suffer from hallucinations. Model editing has been proposed to correct factual inaccuracies in LLMs. A challenging case is sequential model editing (SME), which aims to rectify errors continuously rather than treating them as a one-time task. During SME, the general capabilities of LLMs can be negatively affected due to the introduction of new parameters. In this paper, we propose a queue-based self-correction framework (QueueEDIT) that not only enhances SME performance by addressing long-sequence dependency but also mitigates the impact of parameter bias on the general capabilities of LLMs. Specifically, we first introduce a structural mapping editing loss to map the triplets to the knowledge-sensitive neurons within the Transformer layers of LLMs. We then store the located parameters for each piece of edited knowledge in a queue and dynamically align previously edited parameters. In each edit, we select queue parameters most relevant to the currently located parameters to determine whether previous knowledge needs realignment. Irrelevant parameters in the queue are frozen, and we update the parameters at the queue head to the LLM to ensure they do not harm general abilities. Experiments show that our framework significantly outperforms strong baselines across various SME settings and maintains competitiveness in single-turn editing. The resulting LLMs also preserve high capabilities in general NLP tasks throughout the SME process. 

---
# TPTT: Transforming Pretrained Transformer into Titans 

**Authors**: Fabien Furfaro  

**Link**: [PDF](https://arxiv.org/pdf/2506.17671)  

**Abstract**: Recent advances in large language models (LLMs) have led to remarkable progress in natural language processing, but their computational and memory demands remain a significant challenge, particularly for long-context inference. We introduce TPTT (Transforming Pretrained Transformer into Titans), a novel framework for enhancing pretrained Transformer models with efficient linearized attention mechanisms and advanced memory management. TPTT employs techniques such as Memory as Gate (MaG) and mixed linearized attention (LiZA). It is fully compatible with the Hugging Face Transformers library, enabling seamless adaptation of any causal LLM through parameter-efficient fine-tuning (LoRA) without full retraining. We show the effectiveness of TPTT on the MMLU benchmark with models of approximately 1 billion parameters, observing substantial improvements in both efficiency and accuracy. For instance, Titans-Llama-3.2-1B achieves a 20% increase in Exact Match (EM) over its baseline. Statistical analyses and comparisons with recent state-of-the-art methods confirm the practical scalability and robustness of TPTT. Code is available at this https URL . Python package at this https URL . 

---
# GTA: Grouped-head latenT Attention 

**Authors**: Luoyang Sun, Jiwen Jiang, Cheng Deng, Xinjian Wu, Haifeng Zhang, Lei Chen, Lionel Ni, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17286)  

**Abstract**: Attention mechanisms underpin the success of large language models (LLMs), yet their substantial computational and memory overhead poses challenges for optimizing efficiency and performance. A critical bottleneck arises as KV cache and attention computations scale rapidly with text length, challenging deployment on hardware with limited computational and memory resources. We observe that attention mechanisms exhibit substantial redundancy, since the KV cache can be significantly compressed and attention maps across heads display high similarity, revealing that much of the computation and storage is unnecessary. Leveraging these insights, we propose \textbf{G}rouped-Head Laten\textbf{T} \textbf{A}ttention (GTA), a novel attention mechanism that reduces memory usage and computational complexity while maintaining performance. GTA comprises two components: (1) a shared attention map mechanism that reuses attention scores across multiple heads, decreasing the key cache size; and (2) a nonlinear value decoder with learned projections that compresses the value cache into a latent space, further cutting memory needs. GTA cuts attention computation FLOPs by up to \emph{62.5\%} versus Grouped-Query Attention and shrink the KV cache by up to \emph{70\%}, all while avoiding the extra overhead of Multi-Head Latent Attention to improve LLM deployment efficiency. Consequently, GTA models achieve a \emph{2x} increase in end-to-end inference speed, with prefill benefiting from reduced computational cost and decoding benefiting from the smaller cache footprint. 

---
# PaceLLM: Brain-Inspired Large Language Models for Long-Context Understanding 

**Authors**: Kangcong Li, Peng Ye, Chongjun Tu, Lin Zhang, Chunfeng Song, Jiamin Wu, Tao Yang, Qihao Zheng, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17310)  

**Abstract**: While Large Language Models (LLMs) demonstrate strong performance across domains, their long-context capabilities are limited by transient neural activations causing information decay and unstructured feed-forward network (FFN) weights leading to semantic fragmentation. Inspired by the brain's working memory and cortical modularity, we propose PaceLLM, featuring two innovations: (1) a Persistent Activity (PA) Mechanism that mimics prefrontal cortex (PFC) neurons' persistent firing by introducing an activation-level memory bank to dynamically retrieve, reuse, and update critical FFN states, addressing contextual decay; and (2) Cortical Expert (CE) Clustering that emulates task-adaptive neural specialization to reorganize FFN weights into semantic modules, establishing cross-token dependencies and mitigating fragmentation. Extensive evaluations show that PaceLLM achieves 6% improvement on LongBench's Multi-document QA and 12.5-17.5% performance gains on Infinite-Bench tasks, while extending measurable context length to 200K tokens in Needle-In-A-Haystack (NIAH) tests. This work pioneers brain-inspired LLM optimization and is complementary to other works. Besides, it can be generalized to any model and enhance their long-context performance and interpretability without structural overhauls. 

---
# Leveraging Large Language Model for Intelligent Log Processing and Autonomous Debugging in Cloud AI Platforms 

**Authors**: Cheng Ji, Huaiying Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17900)  

**Abstract**: With the increasing complexity and rapid expansion of the scale of AI systems in cloud platforms, the log data generated during system operation is massive, unstructured, and semantically ambiguous, which brings great challenges to fault location and system self-repair. In order to solve this problem, this paper proposes an intelligent log processing and automatic debugging framework based on Large Language Model (LLM), named Intelligent Debugger (LLM-ID). This method is extended on the basis of the existing pre-trained Transformer model, and integrates a multi-stage semantic inference mechanism to realize the context understanding of system logs and the automatic reconstruction of fault chains. Firstly, the system log is dynamically structured, and the unsupervised clustering and embedding mechanism is used to extract the event template and semantic schema. Subsequently, the fine-tuned LLM combined with the multi-round attention mechanism to perform contextual reasoning on the log sequence to generate potential fault assumptions and root cause paths. Furthermore, this paper introduces a reinforcement learning-based policy-guided recovery planner, which is driven by the remediation strategy generated by LLM to support dynamic decision-making and adaptive debugging in the cloud environment. Compared with the existing rule engine or traditional log analysis system, the proposed model has stronger semantic understanding ability, continuous learning ability and heterogeneous environment adaptability. Experiments on the cloud platform log dataset show that LLM-ID improves the fault location accuracy by 16.2%, which is significantly better than the current mainstream methods 

---
