# MambaCSP: Hybrid-Attention State Space Models for Hardware-Efficient Channel State Prediction 

**Authors**: Aladin Djuhera, Haris Gacanin, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2604.21957)  

**Abstract**: Recent works have demonstrated that attention-based transformer and large language model (LLM) architectures can achieve strong channel state prediction (CSP) performance by capturing long-range temporal dependencies across channel state information (CSI) sequences. However, these models suffer from quadratic scaling in sequence length, leading to substantial computational cost, memory consumption, and inference latency, which limits their applicability in real-time and resource-constrained wireless deployments. In this paper, we investigate whether selective state space models (SSMs) can serve as a hardware-efficient alternative for CSI prediction. We propose MambaCSP, a hybrid-attention SSM architecture that replaces LLM-based prediction backbones with a linear-time Mamba model. To overcome the local-only dependencies of pure SSMs, we introduce lightweight patch-mixer attention layers that periodically inject cross-token attentions, helping with long-context CSI prediction. Extensive MISO-OFDM simulations show that MambaCSP improves prediction accuracy over LLM-based approaches by 9-12%, while delivering up to 3.0x higher throughput, 2.6x lower VRAM usage, and 2.9x faster inference. Our results demonstrate that hybrid state space architectures provide a promising direction for scalable and hardware-efficient AI-native CSI prediction in future wireless networks. 

---
# Focus Session: Hardware and Software Techniques for Accelerating Multimodal Foundation Models 

**Authors**: Muhammad Shafique, Abdul Basit, Muhammad Abdullah Hanif, Alberto Marchisio, Rachmad Vidya Wicaksana Putra, Minghao Shao  

**Link**: [PDF](https://arxiv.org/pdf/2604.21952)  

**Abstract**: This work presents a multi-layered methodology for efficiently accelerating multimodal foundation models (MFMs). It combines hardware and software co-design of transformer blocks with an optimization pipeline that reduces computational and memory requirements. During model development, it employs performance enhancements through fine-tuning for domain-specific adaptation. Our methodology further incorporates hardware and software techniques for optimizing MFMs. Specifically, it employs MFM compression using hierarchy-aware mixed-precision quantization and structural pruning for transformer blocks and MLP channels. It also optimizes operations through speculative decoding, model cascading that routes queries through a small-to-large cascade and uses lightweight self-tests to determine when to escalate to larger models, as well as co-optimization of sequence length, visual resolution & stride, and graph-level operator fusion. To efficiently execute the model, the processing dataflow is optimized based on the underlying hardware architecture together with memory-efficient attention to meet on-chip bandwidth and latency budgets. To support this, a specialized hardware accelerator for the transformer workloads is employed, which can be developed through expert design or an LLM-aided design approach. We demonstrate the effectiveness of the proposed methodology on medical-MFMs and on code generation tasks, and conclude with extensions toward energy-efficient spiking-MFMs. 

---
# Feedback Over Form: Why Execution Feedback Matters More Than Pipeline Topology in 1-3B Code Generation 

**Authors**: Charles Junichi McAndrews  

**Link**: [PDF](https://arxiv.org/pdf/2604.21950)  

**Abstract**: Small language models (1-3B) are practical to run locally, but individually limited on harder code generation tasks. We ask whether composing them into pipelines can recover some of that lost capability. We study code generation pipelines built from 1-3B models with execution feedback, and use a NEAT-inspired evolutionary search to test whether more complex pipeline structure helps beyond a simple refinement loop. We evaluate on HumanEval (164 problems) and sanitized MBPP (427 problems), all with local inference on a single laptop. Self-refinement with execution feedback improves code generation by more than 4 standard deviations on both benchmarks. The gains are narrow in mechanism: refinement fixes many runtime errors (especially NameError and SyntaxError), but rarely fixes logic errors such as AssertionError. Within our tested general-purpose model pool, generator identity mattered less than refiner capability: a 1.5B generator paired with a 3B refiner matched a 3B model doing both roles. Early stopping is essential; without it, every iteration is net-negative. The code-specialized models outperform every general-purpose pipeline configuration, suggesting model specialization matters more than pipeline architecture. Preliminary text-only pipeline experiments without execution feedback did not show gains at this scale. In our constrained search space, evolutionary search mostly rediscovered the same simple generate-execute-refine loop we found manually, with no clearly significant gain from added topology. Single-evaluation fitness inflates results by 5-7 percent, selecting lucky genomes over good ones. On these benchmarks at 1-3B scale, execution feedback mattered more than added pipeline complexity in determining whether composition helped. 

---
# Spontaneous Persuasion: An Audit of Model Persuasiveness in Everyday Conversations 

**Authors**: Nalin Poungpeth, Nicholas Clark, Tanu Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2604.22109)  

**Abstract**: Large language models (LLMs) possess strong persuasive capabilities that outperform humans in head-to-head comparisons. Users report consulting LLMs to inform major life decisions in relationships, medical settings, and when seeking professional advice. Prior work measures persuasion as intentional attempts at producing the most effective argument or convincing statement. This fails to capture everyday human-AI interactions in which users seek information or advice. To address this gap, we introduce "spontaneous persuasion," which characterizes the inexplicit use of persuasive strategies in everyday scenarios where persuasion is not necessarily warranted. We conduct an audit of five LLMs to uncover how frequently and through which techniques spontaneous persuasion appears in multi-turn conversations. To simulate response styles, we provide a user response taxonomy grounded in literature from psychology, communication, and linguistics. Furthermore, we compare the distribution of spontaneous persuasion produced by LLMs with human responses on the same topics, collected from Reddit. We find LLMs spontaneously persuade the user in virtually all conversations, heavily relying on information-based strategies such as appeals to logic or quantitative evidence. This was consistent across models and user response styles, but conversations concerning mental health saw higher rates of appraisal-based and emotion-based strategies. In comparison, human responses tended to invoke strategies that generate social influence, like negative emotion appeals and non-expert testimony. This difference may explain the effectiveness of LLM in persuading users, as well as the perception of models as objective and impartial. 

---
# Hidden Failure Modes of Gradient Modification under Adam in Continual Learning, and Adaptive Decoupled Moment Routing as a Repair 

**Authors**: Yuelin Hu, Zhenbo Yu, Zhengxue Cheng, Wei Liu, Li Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.22407)  

**Abstract**: Many continual-learning methods modify gradients upstream (e.g., projection, penalty rescaling, replay mixing) while treating Adam as a neutral backend. We show this composition has a hidden failure mode. In a high-overlap, non-adaptive 8-domain continual LM, all shared-routing projection baselines collapse close to vanilla forgetting (12.5--12.8 vs. 13.2). A 0.5% replay buffer is the strongest shared alternative but still reaches 11.6, while fixed-strength decoupling falls below vanilla at 14.1. Only adaptive decoupled routing remains stable at 9.4, improving over vanilla by 3.8 units. On a 16-domain stream, its gain over the strongest shared-routing projection baseline grows to 4.5--4.8 units. The failure is largely invisible on clean benchmarks.
We explain this effect through Adam's second-moment pathway: in the tested regime, projection induces a 1/(1-alpha) inflation of the old-direction effective learning rate, matching measurements within 8% across eight alpha values. The same conflict appears with penalty methods, replay mixing, and at 7B scale under LoRA. Our fix routes the modified gradient only to the first moment while preserving magnitude-faithful second-moment statistics, with overlap-aware adaptive strength. This simple change is the only tested configuration that consistently avoids collapse across methods, optimizers, and scale. 

---
# Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions 

**Authors**: Minda Zhao, Yilun Du, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05414)  

**Abstract**: As large language models (LLMs) transition from chat interfaces to integral components of stochastic pipelines and systems approaching general intelligence, the ability to faithfully sample from specified probability distributions has become a functional requirement rather than a theoretical curiosity. We present the first large-scale, statistically powered audit of native probabilistic sampling in frontier LLMs, benchmarking 11 models across 15 distributions. To disentangle failure modes, we employ a dual-protocol design: Batch Generation, where a model produces $N{=}1000$ samples within one response, and Independent Requests, comprising $N{=}1000$ stateless calls. We observe a sharp protocol asymmetry: batch generation achieves only modest statistical validity, with a 7% median pass rate, while independent requests collapse almost entirely, with 10 of 11 models passing none of the distributions. Beyond this asymmetry, we reveal that sampling fidelity degrades monotonically with distributional complexity and aggravates as the sampling horizon $N$ increases. Finally, we demonstrate how the propagation of these failures into downstream real-world application tasks introduces systematic biases: models fail to enforce uniform answer-position constraints in Multiple Choice Question generation and systematically violate demographic targets in attribute-constrained text-to-image prompt synthesis. These findings indicate that current LLMs lack a functional internal sampler, necessitating external tools for applications requiring statistical guarantees. 

---
# PermaFrost-Attack: Stealth Pretraining Seeding(SPS) for planting Logic Landmines During LLM Training 

**Authors**: Harsh Kumar, Rahul Maity, Tanmay Joshi, Aman Chadha, Vinija Jain, Suranjana Trivedy, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2604.22117)  

**Abstract**: Aligned large language models(LLMs) remain vulnerable to adversarial manipulation, and their dependence on web-scale pretraining creates a subtle but serious attack surface. We study Stealth Pretraining Seeding (SPS), a new attack family in which adversaries distribute small amounts of poisoned content across stealth websites, expose them to web crawlers through this http URL, and thereby increase the likelihood that such content is absorbed into future training corpora derived from sources such as Common Crawl. Because each individual payload is tiny, diffuse, and superficially benign, the attack is difficult to detect during dataset construction or filtering. The result is a latent form of poisoning: dormant logic landmines embedded during pretraining that remain largely invisible under standard evaluation, yet can later be activated by precise alphanumeric triggers such as <00TRIGGER00> to bypass safeguards. We call this attack PermaFrost, by analogy to Arctic permafrost: harmful material can remain frozen, buried, and unnoticed for long periods, only to resurface when conditions allow. We operationalize this threat through PermaFrost-Attack, a controlled framework for latent conceptual poisoning, together with a suite of geometric diagnostics: Thermodynamic Length, Spectral Curvature, and the Infection Traceback Graph. Across multiple model families and scales, we show that SPS is broadly effective, inducing persistent unsafe behavior while often evading alignment defenses. Our results identify SPS as a practical and underappreciated threat to future foundation models. This paper introduces a novel geometric diagnostic lens for systematically examining latent model behavior, providing a principled foundation for detecting, characterizing, and understanding vulnerabilities that may remain invisible to standard evaluation. 

---
# Rethinking Semantic Collaborative Integration: Why Alignment Is Not Enough 

**Authors**: Maolin Wang, Dongze Wu, Jianing Zhou, Hongyu Chen, Beining Bao, Yu Jiang, Chenbin Zhang, Chang Wang, Jian Liu, Lei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2604.22195)  

**Abstract**: Large language models (LLMs) have become an important semantic infrastructure for modern recommender systems. A prevailing paradigm integrates LLM-derived semantic embeddings with collaborative representations via representation alignment, implicitly assuming that the two views encode a shared latent entity and that stronger alignment yields better results. We formalize this assumption as the global low-complexity alignment hypothesis and argue that it is stronger than necessary and often structurally mismatched with real-world recommendation settings. We propose a complementary perspective in which semantic and collaborative representations are treated as partially shared yet fundamentally heterogeneous views, each containing both shared and view-specific factors. Under this shared-plus-private latent structure, enforcing global geometric alignment may distort local structure, suppress view-specific signals, and reduce informational diversity. To support this perspective, we develop complementarity-aware diagnostics that quantify overlap, unique-hit contribution, and theoretical fusion upper bounds. Empirical analyses on sparse recommendation benchmarks reveal low item-level agreement between semantic and collaborative views and substantial oracle fusion gains, indicating strong complementarity. Furthermore, controlled alignment probes show that low-capacity mappings capture only shared components and fail to recover full collaborative geometry, especially under distribution shift. These findings suggest that alignment should not be treated as the default integration principle. We advocate a shift from alignment-centric modeling to complementarity fusion-centric, complementarity-aware design, where shared factors are selectively integrated while private signals are preserved. This reframing provides a principled foundation for the next generation of LLM-enhanced recommender systems. 

---
# ResRank: Unifying Retrieval and Listwise Reranking via End-to-End Joint Training with Residual Passage Compression 

**Authors**: Xiaojie Ke, Shuai Zhang, Liansheng Sun, Yongjin Wang, Hengjun Jiang, Xiangkun Liu, Cunxin Gu, Jian Xu, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22180)  

**Abstract**: Large language model (LLM) based listwise reranking has emerged as the dominant paradigm for achieving state-of-the-art ranking effectiveness in information retrieval. However, its reliance on feeding full passage texts into the LLM introduces two critical bottlenecks: the "lost in the middle" phenomenon degrades ranking quality as input length grows, and the inference latency scales super-linearly with sequence length, rendering it impractical for industrial deployment. In this paper, we present ResRank, a unified retrieval-reranking framework that fundamentally addresses both challenges. Inspired by multimodal LLMs that project visual inputs into compact token representations, ResRank employs an Encoder-LLM to compress each candidate passage into a single embedding, which is then fed alongside the query text into a Reranker-LLM for listwise ranking. To alleviate the misalignment between the compressed representation space and the ranking space, we introduce a residual connection structure that combines encoder embeddings with contextualized hidden states from the reranker. Furthermore, we replace the conventional autoregressive decoding with a one-step cosine-similarity-based scoring mechanism, eliminating the generation bottleneck entirely. ResRank is trained through a carefully designed dual-stage, multi-task, end-to-end joint optimization strategy that simultaneously trains the encoder and reranker, achieving learning objective alignment between retrieval and reranking while substantially reducing training complexity. Extensive experiments on TREC Deep Learning and eight BEIR benchmark datasets demonstrate that ResRank achieves competitive or superior ranking effectiveness compared to existing approaches while requiring zero generated tokens and processing only one token per passage, yielding a fundamentally better balance between effectiveness and efficiency. 

---
# Can QPP Choose the Right Query Variant? Evaluating Query Variant Selection for RAG Pipelines 

**Authors**: Negar Arabzadeh, Andrew Drozdov, Michael Bendersky, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2604.22661)  

**Abstract**: Large Language Models (LLMs) have made query reformulation ubiquitous in modern retrieval and Retrieval-Augmented Generation (RAG) pipelines, enabling the generation of multiple semantically equivalent query variants. However, executing the full pipeline for every reformulation is computationally expensive, motivating selective execution: can we identify the best query variant before incurring downstream retrieval and generation costs? We investigate Query Performance Prediction (QPP) as a mechanism for variant selection across ad-hoc retrieval and end-to-end RAG. Unlike traditional QPP, which estimates query difficulty across topics, we study intra-topic discrimination - selecting the optimal reformulation among competing variants of the same information need. Through large-scale experiments on TREC-RAG using both sparse and dense retrievers, we evaluate pre- and post-retrieval predictors under correlation- and decision-based metrics. Our results reveal a systematic divergence between retrieval and generation objectives: variants that maximize ranking metrics such as nDCG often fail to produce the best generated answers, exposing a "utility gap" between retrieval relevance and generation fidelity. Nevertheless, QPP can reliably identify variants that improve end-to-end quality over the original query. Notably, lightweight pre-retrieval predictors frequently match or outperform more expensive post-retrieval methods, offering a latency-efficient approach to robust RAG. 

---
# ASPIRE: Make Spectral Graph Collaborative Filtering Great Again via Adaptive Filter Learning 

**Authors**: Yunhang He, Cong Xu, Zhangchi Zhu, Hongzhi Yin, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22549)  

**Abstract**: Graph filter design is central to spectral collaborative filtering, yet most existing methods rely on manually tuned hyperparameters rather than fully learnable filters. We show that this challenge stems from a bias in traditional recommendation objectives, which induces a spectral phenomenon termed low-frequency explosion, thereby fundamentally hindering the effective learning of graph filters. To overcome this limitation, we propose a novel adaptive spectral graph collaborative filtering framework (ASPIRE) based on a bi-level optimization objective. Guided by our theoretical analysis, we disentangle the filter learning objective, which in turn leads to excellent recommendation performance, spectral adaptivity, and training stability in practice. Extensive experiments show our learned filters match the performance of carefully engineered task-specific designs. Furthermore, ASPIRE is equally effective in LLM-powered collaborative filtering. Our findings demonstrate that graph filter learning is viable and generalizable, paving the way for more expressive graph neural networks in collaborative filtering. 

---
# Objective Shaping with Hard Negatives: Windowed Partial AUC Optimization for RL-based LLM Recommenders 

**Authors**: Wentao Shi, Qifan Wang, Chen Chen, Fei Liu, Dongfang Liu, Xu Liu, Wanli Ma, Junfeng Pan, Linhong Zhu, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.22504)  

**Abstract**: Reinforcement learning (RL) effectively optimizes Large Language Model (LLM)-based recommenders by contrasting positive and negative items. Empirically, training with beam-search negatives consistently outperforms random negatives, yet the mechanism is not well understood. We address this gap by analyzing the induced optimization objective and show that: (i) Under binary reward feedback, optimizing LLM recommenders with Group Relative Policy Optimization (GRPO) is theoretically equivalent to maximizing the Area Under the ROC Curve (AUC), which is often misaligned with Top-$K$ recommendation; and (ii) Replacing random negatives with beam-search negatives reshapes the objective toward partial AUC, improving alignment with Top-$K$ metrics. Motivated by this perspective, we introduce Windowed Partial AUC (WPAUC), which constrains the false positive rate (FPR) to a window [$\alpha,\alpha+d$] to more directly align with Top-$K$ metrics. We further propose an efficient Threshold-Adjusted Windowed reweighting (TAWin) RL method for its optimization, enabling explicit control over the targeted Top-$K$ performance. Experiments on four real-world datasets validate the theory and deliver consistent state-of-the-art performance. 

---
# AgentSearchBench: A Benchmark for AI Agent Search in the Wild 

**Authors**: Bin Wu, Arastun Mammadli, Xiaoyu Zhang, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.22436)  

**Abstract**: The rapid growth of AI agent ecosystems is transforming how complex tasks are delegated and executed, creating a new challenge of identifying suitable agents for a given task. Unlike traditional tools, agent capabilities are often compositional and execution-dependent, making them difficult to assess from textual descriptions alone. However, existing research and benchmarks typically assume well-specified functionalities, controlled candidate pools, or only executable task queries, leaving realistic agent search scenarios insufficiently studied. We introduce AgentSearchBench, a large-scale benchmark for agent search in the wild, built from nearly 10,000 real-world agents across multiple providers. The benchmark formalizes agent search as retrieval and reranking problems under both executable task queries and high-level task descriptions, and evaluates relevance using execution-grounded performance signals. Experiments reveal a consistent gap between semantic similarity and actual agent performance, exposing the limitations of description-based retrieval and reranking methods. We further show that lightweight behavioral signals, including execution-aware probing, can substantially improve ranking quality, highlighting the importance of incorporating execution signals into agent discovery. Our code is available at this https URL. 

---
# Aligning Dense Retrievers with LLM Utility via DistillationAligning Dense Retrievers with LLM Utility via Distillation 

**Authors**: Rajinder Sandhu, Di Mu, Cheng Chang, Md Shahriar Tasjid, Himanshu Rai, Maksims Volkovs, Ga Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22722)  

**Abstract**: Dense vector retrieval is the practical backbone of Retrieval- Augmented Generation (RAG), but similarity search can suffer from precision limitations. Conversely, utility-based approaches leveraging LLM re-ranking often achieve superior performance but are computationally prohibitive and prone to noise inherent in perplexity estimation. We propose Utility-Aligned Embeddings (UAE), a framework designed to merge these advantages into a practical, high-performance retrieval method. We formulate retrieval as a distribution matching problem, training a bi-encoder to imitate a utility distribution derived from perplexity reduction using a Utility-Modulated InfoNCE objective. This approach injects graded utility signals directly into the embedding space without requiring test-time LLM inference. On the QASPER benchmark, UAE improves retrieval Recall@1 by 30.59%, MAP by 30.16% and Token F1 by 17.3% over the strong semantic baseline BGE-Base. Crucially, UAE is over 180x faster than the efficient LLM re-ranking methods preserving competitive performance, demonstrating that aligning retrieval with generative utility yields reliable contexts at scale. 

---
