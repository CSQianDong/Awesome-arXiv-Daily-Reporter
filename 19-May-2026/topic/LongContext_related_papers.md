# Episodic-Semantic Memory Architecture for Long-Horizon Scientific Agents 

**Authors**: Nikola Milosevic  

**Link**: [PDF](https://arxiv.org/pdf/2605.17625)  

**Abstract**: As Large Language Models (LLMs) evolve into persistent scientific collaborators, context window saturation has emerged as a critical bottleneck. Scientific workflows involving iterative data analysis and hypothesis refinement rapidly saturate even extended contexts with dense technical content, while monolithic approaches suffer from quadratic cost scaling and cognitive degradation. We evaluate a Dual Process Memory Architecture that decouples immediate episodic needs (constant 10-message window) from long-term consolidated knowledge (growing at approximately 3 tokens/message). Unlike prior social agent memory systems, our domain-specific consolidation addresses contradictory parameter evolution, multi-hop reasoning across experimental phases, and precise technical fact retention. Through large-scale evaluation spanning 15,000 messages with cross-model validation across six LLMs from three families (OpenAI, Anthropic, Google), totaling 1,440 queries, we establish three key findings. First, while full-context models fail at 10,000 messages due to context overflow, our system maintains 70-85% accuracy with 1-2 second latency using 62% fewer tokens (45,434 vs 120,000+ limit). Second, cross-model validation reveals architecture-level trade-offs independent of specific LLMs: Dual Process excels at numeric/temporal queries (65-90% accuracy) while RAG excels at historical retrieval (60-85%), suggesting complementary deployment strategies. Third, we identify a "Sim-to-Real" gap where synthetic tests maintain constant memory but realistic workflows exhibit linear growth (about 3 tokens/message), with consolidation quality emerging as the primary scalability bottleneck. The architecture successfully manages profiles with 14,000+ scientific facts (125k tokens), demonstrating that domain-specific memory consolidation enables sustained operation beyond full-context limits. 

---
# Causal Intervention-Based Memory Selection for Long-Horizon LLM Agents 

**Authors**: Saksham Sahai Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2605.17641)  

**Abstract**: Long-horizon LLM agents rely on persistent memory to support interactions across sessions, yet existing memory systems often retrieve context using semantic similarity or broad history inclusion, treating retrieved memories as uniformly useful. This assumption is fragile because memories may be topically related while remaining irrelevant, stale, or misleading. We propose Causal Memory Intervention (CMI), a causal memory-selection technique that estimates how candidate memories affect the model's answer under controlled interventions, selecting memories that improve task performance while suppressing unstable, irrelevant, or harmful ones. To evaluate this setting, we introduce Causal-LoCoMo, a causally annotated benchmark derived from long conversational data, where each example contains a user request, a structured memory bank, useful memories, irrelevant distractors, and synthetic harmful memories. We compare CMI against vector, graph, reflection, summary, full-history, and no-memory baselines. Results show that CMI achieves a stronger balance between answer quality and robustness to misleading memory, suggesting that reliable long-term memory requires selecting context based on causal usefulness rather than relevance alone. The full framework, benchmark construction code, and experimental pipeline are available at this https URL. 

---
# Towards Human-Level Book-Writing Capability 

**Authors**: Jan Zierstek, Matteo Batelic, Maya Medjad, Tim Schönenberger  

**Link**: [PDF](https://arxiv.org/pdf/2605.17064)  

**Abstract**: Large language models optimized for instruction following and agentic tasks remain poorly aligned with the requirements of high-quality creative writing. Fiction frequently depends on behaviors that assistant-tuned models are explicitly trained to avoid, particularly deception, moral ambiguity, and unreliable narration. As a result, generated stories often appear structurally correct while remaining stylistically generic, overly explanatory, or weakly grounded in human literary behavior. We present a dataset construction and training framework for book-scale creative writing that reframes supervised fine-tuning as a prompt-to-book generation task grounded in human-authored fiction. Starting from public-domain novels, we derive a multi-resolution planning scaffold by summarizing each book at progressively finer levels, from a high-level premise to chapter- and scene-level structure. We then invert this hierarchy during training: the model learns to expand a prompt into increasingly detailed plans and finally into the original human-authored book text. This formulation preserves human prose as the final supervised target while using intermediate summaries to make book-scale generation learnable. We train a long-context language model on these prompt-to-book trajectories and study whether this objective shifts generation away from assistant-style prose and toward human literary writing. 

---
# Recall Isn't Enough: Bounding Commitments in Personalized Language Systems 

**Authors**: Rui Tang, Yichi Zhang, Xi Chen, Chen Dong, Youwei Yang, Yumeng Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.16712)  

**Abstract**: Long-context and memory systems usually treat personalization as a recall problem. In practice, many failures occur later, when a system commits: it turns noisy hints into hard constraints, drops rare witnesses, forgets downstream obligations, or answers despite infeasibility. We introduce Contract-Bounded Evidence Activation (CBEA) with Lexicographic Commitment Validation (LCV). CBEA activates a bounded evidence set using typed coverage, tail witnesses, and consequence debt; LCV validates structured commitments before prose and routes infeasible states to repair, abstention, or recontract. Across 360 fixtures and three generation backends, CBEA+LCV reaches zero failures within validator scope at 0.49-0.60 availability over attempted runs. Raw and long-context baselines with the same LCV gate reach zero only at 0.003-0.092. A shadow oracle diagnostic marks the limit: CBEA+LCV recalls 0.012 of uncompiled visible facts, while raw recalls 0.53. The result is a bounded operating point: explicit commitment control and 74-75% lower median input payload, not universal memory dominance. 

---
# DashAttention: Differentiable and Adaptive Sparse Hierarchical Attention 

**Authors**: Yuxiang Huang, Nuno M. T. Gonçalves, Federico Alvetreti, Lei Li, Xu Han, Edoardo M. Ponti, André F. T. Martins, Marcos V. Treviso  

**Link**: [PDF](https://arxiv.org/pdf/2605.18753)  

**Abstract**: Current hierarchical attention methods, such as NSA and InfLLMv2, select the top-k relevant key-value (KV) blocks based on coarse attention scores and subsequently apply fine-grained softmax attention on the selected tokens. However, the top-k operation assumes the number of relevant tokens for any query is fixed and it precludes the gradient flow between the sparse and dense stages. In this work, we propose DashAttention (Differentiable and Adaptive Sparse Hierarchical Attention), which leverages the adaptively sparse $\alpha$-entmax transformation to select a variable number of blocks according to the current query in the first stage. This in turn provides a prior for the second-stage softmax attention, keeping the entire hierarchy fully differentiable. Contrary to other hierarchical attention methods, we show that DashAttention is non-dispersive, translating to better long-context modeling ability. Experiments with large language models (LLMs) show that DashAttention achieves comparable accuracy as full attention with 75% sparsity and a better Pareto frontier than NSA and InfLLMv2, especially in high-sparsity regimes. We also provide an efficient, GPU-aware implementation of DashAttention in Triton, which achieves a speedup of up to over FlashAttention-3 at inference time. Overall, DashAttention offers a cost-effective strategy to model long contexts. 

---
# LongMINT: Evaluating Memory under Multi-Target Interference in Long-Horizon Agent Systems 

**Authors**: Hyunji Lee, Justin Chih-Yao Chen, Joykirat Singh, Zaid Khan, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2605.18565)  

**Abstract**: Real-world agents operate over long and evolving horizons, where information is repeatedly updated and may interfere across memories, requiring accurate recall and aggregated reasoning over multiple pieces of information. However, existing benchmarks focus on static, independent recall and fail to capture these dynamic interactions between evolving memories. In this paper, we study how current memory-augmented agents perform in realistic, interference-heavy, long-horizon settings across diverse domains and question types. We introduce LongMINT (Long-Horizon Memory under INTerference), a benchmark featuring (1) long, highly interconnected contexts with frequently updated information that induces substantial interference, (2) diverse domains (state tracking, multi-turn dialogue, Wikipedia revisions, and GitHub commits), enabling evaluation of domain generalization, and (3) diverse question types that assess robustness to interference, including (i) single-target recall tasks requiring retrieval of a specific target from long contexts, and (ii) multi-target aggregation tasks requiring reasoning over multiple relevant pieces of information. Overall, LongMINT has 15.6k question-answering pairs over long-horizon contexts averaging 138.8k tokens and extending up to 1.8M tokens per instance. We evaluate 7 representative systems, including vanilla long-context LLMs, RAG, and memory-augmented agent frameworks. Across all systems, we observe consistently low performance (avg. 27.9% accuracy), especially on questions requiring aggregated reasoning over multiple pieces of evidence. Our analysis shows that performance is primarily limited by retrieval and memory construction. Furthermore, current memory systems struggle to recall and reason over earlier facts that are later revised or interfered with by subsequent context, with performance degrading as the number of intervening updates increases. 

---
# Context Memorization for Efficient Long Context Generation 

**Authors**: Yasuyuki Okoshi, Hao Mark Chen, Guanxi Lu, Hongxiang Fan, Masato Motomura, Daichi Fujiki  

**Link**: [PDF](https://arxiv.org/pdf/2605.18226)  

**Abstract**: Modern large language model (LLM) applications increasingly rely on long conditioning prefixes to control model behavior at inference time. While prefix-augmented inference is effective, it incurs two structural limitations: i) the prefix's influence fades as generation proceeds, and ii) attention computation over the prefix scales linearly with its length. Existing approaches either keep the prefix in attention while compressing it, or internalize it into model parameters through gradient-based training. The former still attends to the prefix at inference, while the latter is training-intensive and ill-suited to prefix updates. To address these issues, we propose attention-state memory, a training-free approach that externalizes the prefix into a lightweight, lookup-based memory of precomputed attention states between prefix and query tokens. On ManyICLBench with LLaMA-3.1-8B, our method improves accuracy over in-context learning at 1K-8K memory budgets while reducing attention latency by 1.36x at 8K, and surpasses full-attention RAG performance on NBA benchmark using only 20% of its memory footprint. 

---
# MARS: Technical Report for the CASTLE Challenge at EgoVis 2026 

**Authors**: Haoyu Zhang, Qiaohui Chu, Yisen Feng, Meng Liu, Weili Guan, Yaowei Wang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2605.18176)  

**Abstract**: This report presents MARS, short for Multimodal Agentic Reasoning with Source selection, our system for the CASTLE Challenge at EgoVis 2026. Participants must answer 185 closed-form questions over the CASTLE 2024 dataset. In contrast to prior single-video egocentric benchmarks, CASTLE requires reasoning over four days of activity, 15 synchronized perspectives, official transcripts, and multiple auxiliary modalities, including personal photos, auxiliary videos, gaze, thermal imagery, and heartrate measurements. MARS therefore treats the task as an agentic evidence-selection problem over multimodal sources rather than a purely text-only pipeline. MARS first follows the official CASTLE directory organization to build evidence memories from two primary sources, videos and transcripts, and four auxiliary sources, gaze, heartrate, photos, and thermal imagery. Long videos are converted into captions and DeepSeek-based summaries only because CASTLE videos are too long to fit directly into the model context for every question; this step compresses temporal evidence while keeping photos and other auxiliary media available as source-specific evidence. At inference time, a GPT-5.4 decision agent repeatedly chooses whether to continue reasoning, request a specific missing modality, produce an answer, or fall back to a random option when the evidence remains insufficient. The resulting system achieved second place on the final CASTLE Challenge leaderboard. Our codes are available at this https URL. 

---
# Prompt Compression in Diffusion Large Language Models: Evaluating LLMLingua-2 on LLaDA 

**Authors**: Sterling Huang, Abigayle Brown, Jiyoo Noh, Jiakang Xu, Wantong Huo, Kaung Myat Kyaw, Jonathan Chan  

**Link**: [PDF](https://arxiv.org/pdf/2605.17932)  

**Abstract**: Prompt compression reduces inference cost and context length in large language models, but prior evaluations focus primarily on autoregressive architectures. This study investigates whether prompt compression transfers effectively to diffusion large language models (DLLMs) using LLMLingua-2, specifically the 8B-parameter DLLM LLaDA. We evaluate compression performance on GSM8K, DUC2004, and ShareGPT using 250 prompts per dataset at an approximate 2$\times$ compression ratio, across mathematical reasoning, prompt reconstruction, and summarization tasks. Outputs generated from original prompts, compressed prompts, reconstructed prompts, and reconstructed-prompt reasoning were compared using exact-match accuracy, BLEU, ROUGE, and BERTScore. Results show that semantic preservation does not necessarily imply stable downstream behavior in diffusion models. Summarization tasks remained comparatively robust under compression, while mathematical reasoning degraded substantially despite high semantic similarity scores. Reconstruction experiments further showed that semantically similar prompts may still omit reasoning-critical information required for stable denoising. Across tasks, BERTScore recall was consistently lower than precision, suggesting that compression failures are primarily driven by information omission rather than semantic drift. These findings indicate that prompt compression methods designed for autoregressive models do not transfer uniformly to diffusion large language models and motivate the development of diffusion-aware compression strategies. 

---
# OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization 

**Authors**: Zhongzhu Zhou, Donglin Zhuang, Jisen Li, Ziyan Chen, Shuaiwen Leon Song, Ben Athiwaratkun, Xiaoxia Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.17757)  

**Abstract**: INT2 KV-cache quantization is attractive for long-context LLM serving, but it remains difficult to make both accurate and deployable. Simple rotations such as Hadamard transforms reduce outliers, but still degrade at INT2 because they are not aligned with downstream attention. We propose OSCAR, an Ultra-low-bit KV Cache quantization method that estimates attention-aware covariance structures offline and uses them to derive fixed rotations and clipping thresholds for quantization. In this way, it aligns KV quantization with the covariance structures that attention actually consumes. More importantly, we not only provide theoretical justification but also develop a fully deployable OSCAR system with a custom INT2 attention kernel that remains compatible with paged KV-cache serving and fused kernel pipelines, enabling seamless integration into modern LLM serving frameworks such as SGLang and vLLM.
We evaluate our methods on recent reasoning models with reasoning traces of up to 32k tokens across 5 tasks. On Qwen3-4B-Thinking-2507 and Qwen3-8B, OSCAR reduces the BF16 accuracy gap to 3.78 and 1.42 points, respectively, while naive rotation INT2 collapses to nearly zero. We further scale OSCAR to Qwen3-32B and GLM-4.7 (358B params), where it remains effectively on par with BF16. On long context - RULER-NIAH up to 128K, OSCAR remains robust on both Qwen3 models, while naive rotation INT2 collapses. System-wise, OSCAR reduces KV-cache memory by approximately 8x, improves throughput by up to 7x at large batch sizes under the same memory budget, and accelerates batch-size-1 decoding by up to 3x over BF16 due to reduced memory bandwidth overhead. 

---
# Full Attention Strikes Back: Transferring Full Attention into Sparse within Hundred Training Steps 

**Authors**: Yanke Zhou, Yiduo Li, Hanlin Tang, Maohua Li, Kan Liu, Lan Tao, Lin Qu, Yuan Yao, Xiaoxing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.16928)  

**Abstract**: Long-context inference in large language models is bottlenecked by the quadratic cost of full attention. Existing efficient alternatives often rely either on native sparse training or on heuristic token eviction, creating an undesirable trade-off among efficiency, training cost, and accuracy. In this work, we show that full-attention LLMs are already intrinsically sparse and can be transformed into highly sparse models with only minimal adaptation. Our approach is built on three observations: (1) only a small subset of attention heads truly requires full long-context processing; (2) long-range retrieval is governed primarily by a low-dimensional subspace, allowing relevant tokens to be retrieved efficiently with a 16-dimensional indexer; and (3) the useful token budget is strongly query-dependent, making dynamic top-$p$ selection more suitable than fixed top-$k$ sparsification. Based on these insights, we propose RTPurbo, which retains the full KV cache only for retrieval heads and introduces a lightweight token indexer for sparse attention. By exploiting the model's intrinsic sparsity, RTPurbo achieves sparsification with only a few hundred training steps. Experiments on long-context benchmarks and reasoning tasks show that RTPurbo preserves near-lossless accuracy while delivering substantial efficiency gains, including up to a 9.36$\times$ prefill speedup at 1M context and about a 2.01$\times$ decode speedup. These results suggest that strong sparse inference can be obtained from standard full-attention training without expensive native sparse pretraining. 

---
# Visual Agentic Memory: Enabling Online Long Video Understanding via Online Indexing, Hierarchical Memory, and Agentic Retrieval 

**Authors**: Aiden Yiliu Li, Nels Numan, Anthony Steed  

**Link**: [PDF](https://arxiv.org/pdf/2605.16481)  

**Abstract**: Long video understanding requires more than large context windows. It also needs a memory mechanism that decides what visual evidence to retain, keeps it searchable over long horizons, and grounds later reasoning in recoverable observations rather than compressed latent state alone. We propose Visual Agentic Memory (VAM), a training-free framework with three components. Online Indexing supports selective evidence retention under streaming constraints. Hierarchical Memory organises retained evidence in a Parallel Representation that aligns temporal context with spatial observations. Agentic Retrieval searches, inspects, and verifies candidate evidence before producing a grounded answer. On OVO-Bench, VAM achieves the highest RT+BT average (68.41) across all reported baselines, improving over end-to-end use of the same underlying MLLM (Gemini 3 Flash, 67.46). On the month-scale split of MM-Lifelong train@month (105.6 hours over 51 days), VAM reaches 17.11%, second only to ReMA with GPT-5 (17.62%). These results suggest that long-horizon video understanding benefits from treating visual memory as an explicit, inspectable, and queryable substrate. Code is available at this https URL. 

---
# KVCapsule: Efficient Sequential KV Cache Compression for Vision-Language Models with Asymmetric Redundancy 

**Authors**: Yingbing Huang, Tharun Adithya Srikrishnan, Steven K. Reinhardt, Deming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.16439)  

**Abstract**: Vision-Language Models (VLMs) have emerged as a critical and fast-growing extension of Large Language Models (LLMs) that enable multimodal reasoning through both text and image inputs. Although VLMs enrich the capabilities of language models, they also inherit and amplify key computational bottlenecks: the memory overhead caused by the large key-value (KV) cache during autoregressive decoding. This challenge is particularly severe in VLMs, where images produce longer token sequences and denser feature representations compared to text. Moreover, the spatial and information-rich nature of vision tokens introduces structured attention patterns that make many LLM-oriented KV cache compression techniques ineffective when applied directly to VLMs.
In this work, we conduct a detailed empirical analysis of the behavior of vision tokens, highlighting the critical differences from purely text-based models. Based on these insights, we propose KVCapsule, a novel KV cache compression framework for vision tokens. KVCapsule keeps the pretrained VLM backbone frozen, requires no modification to the attention computation modules, and can be integrated into existing VLMs through lightweight compression and reconstruction components. We evaluate KVCapsule on multiple VLMs and benchmark tasks, demonstrating up to 2x improvement in TPS and 2.4x reduction in KV cache memory at a 60% compression ratio, with negligible degradation in accuracy or response quality. Our findings offer practical pathways to scale VLM inference under constrained memory budgets and inspire further research into structure-aware cache compression for multimodal models. 

---
# ProxyKV: Cross-Model Proxy Pruning for Efficient Long-Context LLM Inference 

**Authors**: Junjie Li, Jiong Lou, Jie Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.16360)  

**Abstract**: Efficient long-context inference in Large Language Models (LLMs) is severely constrained by the Key-Value (KV) cache memory wall, yet existing pruning methods force a choice between
low-latency heuristics that sacrifice precision and high-precision reconstruction methods that incur prohibitive prefilling overhead. To bridge this scoring-cost--accuracy gap, we propose
ProxyKV, a cross-model proxy pruning framework that offloads importance scoring to a lightweight intra-family Small-Model Proxy executed asynchronously to the Large-Model Target. To bridge
the architectural gap between heterogeneous models, we design the HybridAxialMapper, which disentangles temporal feature extraction from cross-head alignment, together with a
Multi-Granularity Hybrid Loss that shifts the learning objective from rigid regression to relative ranking consistency. Across the Llama-3.1, Qwen-2.5, and Qwen-3 families spanning targets
from 7B up to 32B parameters on LongBench, SCBench, and RULER, ProxyKV matches KVZip on aggregate (recovering $\sim$$98.7\%$ of its mean accuracy) while delivering up to a $3.21\times$
prefilling speedup on Llama-3.1-8B (dual-GPU; $\sim$$1.5\times$ shared single-GPU) and sustaining the speedup at contexts up to 170k tokens on Qwen-2.5-7B. 

---
# KVDrive: A Holistic Multi-Tier KV Cache Management System for Long-Context LLM Inference 

**Authors**: Jian Lin, Jiazhi Mi, Zicong Hong, Haodong Wang, Qianli Liu, Haodyue Zhang, Peng Li, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.18071)  

**Abstract**: Supporting long-context LLMs is challenging due to the substantial memory demands of the key-value (KV) cache. Existing offloading systems store the full cache in host memory and selectively fetch critical entries during decoding, but this strategy quickly hits a ceiling: sparsity cannot be pushed further without degrading accuracy. As a result, when context length and batch size grow, the volume of KV transfers rises sharply and becomes the dominant source of decoding latency. We present KVDrive, a holistic multi-tier KV cache management system spanning GPU memory, host DRAM, and SSD. Unlike prior work that pursues greater sparsity through algorithmic refinements, KVDrive tackles the problem from a systems perspective - jointly orchestrating cache placement, pipeline scheduling, and cross-tier coordination to sustain high-throughput inference under tight GPU budgets. KVDrive advances three fundamental capabilities: it adapts cache management to attention behavior to maximize reuse and minimize redundant data movement; it restructures the decoding pipeline to overlap I/O- and CPU/GPU compute-bound stages, eliminating stalls across heterogeneous resources; and it harmonizes data movement across memory tiers to unlock scalable long-context inference far beyond GPU and DRAM limits. We have implemented a fully functional prototype of KVDrive and evaluated it on long-context benchmarks with popular LLMs. The system achieves up to 1.74x higher throughput compared to state-of-the-art works while preserving accuracy. 

---
# CompactAttention: Accelerating Chunked Prefill with Block-Union KV Selection 

**Authors**: Jiwon Song, Dongwon Jo, Beomseok Kang, Jae-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.16839)  

**Abstract**: Chunked prefill has become a widely adopted serving strategy for long-context large language models, but efficient attention computation in this regime remains challenging. Existing sparse attention methods are primarily designed for one-shot prefill and do not translate efficiently to chunked prefill: block-sparse kernels lose efficiency when the query length is limited by the chunk size, while fine-grained pattern search becomes costly when repeated over the accumulated KV cache at every chunk. QUOKA, a recent method that directly targets chunked prefill, avoids sparse-kernel overhead but relies on query-subsampled, token-level KV selection, which can miss query-specific KV entries and introduce explicit KV-copy overhead. To address these limitations, we propose CompactAttention, a chunked-prefill attention mechanism based on Block-Union KV Selection. CompactAttention treats 2D block-sparse masks as KV-selection signals rather than direct sparse-kernel execution plans, and converts them into GQA-aware per-group KV block tables through Q-block union and intra-group union. This construction produces the minimal block tables that preserve all KV blocks selected by the input masks under paged execution constraints, enabling selected KV blocks to be accessed in place without explicit KV compaction. On LLaMA-3.1-8B-Instruct, CompactAttention maintains accuracy close to dense attention on the RULER benchmark while delivering up to 2.72$\times$ attention speedup at 128K context length under chunked prefill. 

---
# CHI-Bench: Can AI Agents Automate End-to-End, Long-Horizon, Policy-Rich Healthcare Workflows? 

**Authors**: Haolin Chen, Deon Metelski, Leon Qi, Tao Xia, Joonyul Lee, Steve Brown, Kevin Riley, Frank Wang, T. Y. Alvin Liu, Hank Capps MD, Zeyu Tang, Xiangchen Song, Lingjing Kong, Fan Feng, Tianyi Zeng, Zhiwei Liu, Zixian Ma, Hang Jiang, Fangli Geng, Yuan Yuan, Chenyu You, Qingsong Wen, Hua Wei, Yanjie Fu, Yue Zhao, Carl Yang, Biwei Huang, Kun Zhang, Caiming Xiong, Sanmi Koyejo, Eric P. Xing, Philip S. Yu, Weiran Yao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16679)  

**Abstract**: End-to-end automation of realistic healthcare operations stresses three capabilities underrepresented in current benchmarks: policy density, decisions must be grounded in a large library of medical, insurance, and operational rules; Multi-role composition: a single task requires the agent to play multiple roles with handoffs; and multilateral interaction: intermediate workflow steps are multi-turn dialogs, such as peer-to-peer review and patient outreach. We introduce $\chi$-Bench, a benchmark of long-horizon healthcare workflows across three domains: provider prior authorization, payer utilization management, and care management. Each task hands the agent a clinical case in a high-fidelity simulator of 20 healthcare apps exposed via 87 MCP tools, which it must drive to a terminal status through tool calls and writing the role's artifacts, guided by a 1,290+ document managed-care operations handbook skill. Across 30 agent harness/models configurations, the best agent resolves only 28.0% of tasks, no agent clears 20% on strict pass^3, and executing all tasks in a single session slumps the performance to 3.8%. These results raise the hypothesis that similar gaps are likely to surface in other policy-dense, role-composed, irreversible enterprise domains. 

---
# Protection Is (Nearly) All You Need: Structural Protection Dominates Scoring in Globally Capped KV Eviction 

**Authors**: Gabriel Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2605.18053)  

**Abstract**: We study KV cache eviction under a shared globally capped decode-time harness. Seven policies (LRU, H2O, SnapKV, StreamingLLM, Ada-KV, QUEST, Random) share a prompt-boundary vulnerability: without structural protection, they collapse to near-zero quality on six pure-transformer models (F1$\leq$0.064). Reserving 10\% of cache at each boundary recovers 69--90\% of the $C{=}2{,}048$ reference-ceiling quality on seven LongBench models at $C{=}256$ (13\% retention); a ten-model panel spans 68--98\%. An attention-mass pilot (Qwen2.5-3B, $N{=}30$) suggests why: the position-0 sink holds ${\sim}75\%$ of prefix mass, while other boundary tokens sit near ${\sim}0.41{\times}$ uniform expectation, so attention scorers retain the sink but still drop structurally critical tokens. With protection, simplified score-isolation variants are TOST-equivalent to LRU at $K{=}32$ ($\Delta{=}0.02$); at $K{=}8$, attention policies pairwise converge yet beat LRU by 0.011--0.021 F1 across $C{=}256$ and $C{=}512$. Faithful Ada-KV/QUEST add ${\sim}0.03$--$0.04$ F1 on Mistral-7B and Phi-3.5 beyond simplified variants. A NIAH-32K regime-transfer pilot on Qwen3-4B (decode vs.\ prefill, $C{\in}\{512,2048\}$) shows near-identical protection lifts (ratio 0.99--1.00). At 64K, protection helps but recovery is modest; faithful per-head scoring matches full-cache ceiling on Gemma-3-4B at 6.3\% retention only when the model already supports strong 64K retrieval without eviction. Overall: protection dominates; scoring differences are secondary once boundaries are guarded; per-head allocation gives a further modest gain. 

---
# Compress the Context, Keep the Commitments: A Formal Framework for Verifiable LLM Context Compression 

**Authors**: Natalia Trukhina, Vadim Vashkelis  

**Link**: [PDF](https://arxiv.org/pdf/2605.17304)  

**Abstract**: LLM context is not just tokens; it is a set of commitments. Long-running conversations accumulate goals, constraints, decisions, preferences, tool results, retrieved evidence, artifacts, and safety boundaries that future responses must preserve. Existing context-management methods reduce length through truncation, retrieval, summarization, memory systems, or token-level prompt compression, but they rarely specify which semantic commitments must survive compression or how their preservation should be measured. We propose Context Codec, a commitment-level framework for compressing prompts and chat histories.
Context Codec represents dialogue state as typed, source-grounded semantic atoms with canonical identity, equivalence, conflict, confidence, risk, and evidence spans. It separates five concerns - extraction, normalization, representation, rendering, and verification - and introduces metrics for Critical Atom Recall, Weighted Atom Recall, Commitment Density, and round-trip recoverability. It also defines a taxonomy of semantic compression errors, a concrete normalization procedure, conservative fallback rules for low-confidence and safety-critical atoms, and Context Compression Language (CCL), an ASCII-first compact rendering of canonical JSON atoms. In a small diagnostic study, CCL-Core occupies a useful middle ground between structured prose and JSON: more explicit and auditable than prose, usually more compact than JSON, and less risky than heavily minified notation. The result is not a claim that shorthand solves compression, but a framework for making context compression verifiable: compress the conversation, keep the commitments. 

---
# HEED: Density-Weighted Residual Alignment for Hybrid Vision-Language Model Distillation 

**Authors**: Yihao Liang, Niraj K. Jha  

**Link**: [PDF](https://arxiv.org/pdf/2605.17093)  

**Abstract**: Distilling vision-language models into faster hybrid architectures, such as 3:1 Mamba-2/attention mixes, is now standard practice for making inference efficient. Aggregate benchmarks suggest that this works but they hide selective failures. When we distill Qwen3-VL-8B-Instruct into a 3:1 Mamba-2/attention hybrid, student model stays within 2 points of the teacher across visual reasoning benchmarks like MMStar, MMBench, and MMMU-Pro, while dropping 13 points on optical-character-recognition and document tasks. The student can still understand the scene but loses the fine-grained text needed to answer. We localize much of the failure to a specific kind of position. In a high-resolution image, most patches are sky, wall, or smooth texture, while a small fraction carries text, edges, object boundaries, or other local details. In a token-level diagnostic, the top 10% highest-density patches have 3.6$\times$ larger residual drift than the bottom 10% lowest-density patches and 3.5$\times$ larger teacher-masking answer contribution. Uniform weighting devotes many loss terms to low-information background patches, whereas sparse answer-bearing patches receive no special protection. The required intervention is minimal: we replace uniform residual alignment with density-weighted residual alignment, using patch self-dissimilarity as a training-free proxy for position importance. We call this HEED. Compared with normal end-to-end distillation, HEED increases performance by 8.7 points on OCRBench v2 and 5.13 points on a 10-benchmark average. The gain is realized on different teacher models and hybrid architectures. After standard post-training, the student reaches teacher-level performance on the 10-benchmark average with a 4.12$\times$ throughput and a 68% memory saving at 128k context, with no additional parameters and no inference-time cost. 

---
