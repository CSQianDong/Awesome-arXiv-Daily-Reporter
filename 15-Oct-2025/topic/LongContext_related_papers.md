# Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks 

**Authors**: Yuxiang Zhang, Jiangming Shu, Ye Ma, Xueyuan Lin, Shangxi Wu, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12635)  

**Abstract**: Large Language Models face challenges in long-horizon agentic tasks as their constrained memory is easily overwhelmed by distracting or irrelevant context. Existing working memory methods typically rely on external, heuristic mechanisms that are decoupled from the agent's core policy. In this work, we reframe working memory management as a learnable, intrinsic capability. We propose a novel framework, Memory-as-Action, where an agent actively manages its working memory by executing explicit editing operations as part of a unified policy. This formulation allows an agent, trained via reinforcement learning, to balance memory curation against long-term task objectives under given resource constraints. However, such memory editing actions break the standard assumption of a continuously growing prefix in LLM interactions, leading to what we call trajectory fractures. These non-prefix changes disrupt the causal continuity required by standard policy gradient methods, making those methods inapplicable. To address this, we propose a new algorithm, Dynamic Context Policy Optimization, which enables stable end-to-end reinforcement learning by segmenting trajectories at memory action points and applying trajectory-level advantages to the resulting action segments. Our results demonstrate that jointly optimizing for task reasoning and memory management in an end-to-end fashion not only reduces overall computational consumption but also improves task performance, driven by adaptive context curation strategies tailored to the model's intrinsic capabilities. 

---
# Chinese ModernBERT with Whole-Word Masking 

**Authors**: Zeyu Zhao, Ningtao Wang, Xing Fu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12285)  

**Abstract**: Encoder-only Transformers have advanced along three axes -- architecture, data, and systems -- yielding Pareto gains in accuracy, speed, and memory efficiency. Yet these improvements have not fully transferred to Chinese, where tokenization and morphology differ markedly from English. We introduce Chinese ModernBERT, a from-scratch Chinese encoder that couples: (i) a hardware-aware 32k BPE vocabulary tailored to frequent Chinese affixes/compounds, lowering the embedding budget; (ii) whole-word masking (WWM) with a dynamic masking curriculum (30% -> 15%) to align task difficulty with training progress; (iii) a two-stage pre-training pipeline that extends the native context from 1,024 to 8,192 tokens using RoPE and alternating local/global attention; and (iv) a damped-cosine learning-rate schedule for stable long-horizon optimization. We pre-train on ~1.2T Chinese tokens from CCI3-HQ, CCI4 (Chinese), and Cosmopedia-Chinese. On CLUE, Chinese ModernBERT is competitive with strong Chinese encoders under a unified fine-tuning protocol. Under bf16 it achieves high long-sequence throughput while maintaining strong short-sequence speed, reflecting benefits from budget allocation and attention design. To probe retrieval-oriented quality, we add a small amount of open contrastive data: fine-tuning on SimCLUE (~3M pairs) improves further when adding T2Ranking (~2M), reaching 0.505 (Pearson) / 0.537 (Spearman) on the SimCLUE test set. Under this open-data setting, Chinese ModernBERT surpasses Qwen-0.6B-embedding on SimCLUE, suggesting a clear scaling path for STS with additional curated pairs. We will release tokenizer and weights to facilitate reproducible research. 

---
# APCE: Adaptive Progressive Context Expansion for Long Context Processing 

**Authors**: Baisub Lee, Sanghyun Byun, Mohanad Odema, Jung Guack, Jacob Song, Woo Seong Chung  

**Link**: [PDF](https://arxiv.org/pdf/2510.12051)  

**Abstract**: Deploying useful Long-Context Transformer Models (LCTMs) requires addressing two key challenges: (1) A growing memory footprint due to quadratic self-attention and linear KV-cache scaling in memory as sequence length increases; (2) the ContextRot phenomena where empirical evidence suggests that transformer architecture's performance degrades with increasing context length. Given the shared dependency on the input, a natural question arises: Can we surgically select the most important input chunks for processing to synergistically (a) reduce the memory footprint, and (b) mitigate the ContextRot effects? In this paper, we answer this question in the affirmative for long-context summarization tasks. We propose APCE as a context-aware solution to select the most important input chunks through low-dimensional semantic similarity matching with the current query. By directly operating on the input, APCE decouples from strict dependency on underlying hardware or CUDA environments, promising a compatible solution scalable to different deployment systems. Our empirical evaluations have demonstrated superior or on-par summarization performance for APCE compared to the full dense baseline using a fraction (50%-70%) of the input sequence resulting in KV-cache and self-attention memory efficiency improvements. We hope our findings inspire further research on context-aware efficiency solutions for LCTMs geared towards other relevant long-context tasks. 

---
# Leveraging LLMs, IDEs, and Semantic Embeddings for Automated Move Method Refactoring 

**Authors**: Fraol Batole, Abhiram Bellur, Malinda Dilhara, Mohammed Raihan Ullah, Yaroslav Zharov, Timofey Bryksin, Kai Ishikawa, Haifeng Chen, Masaharu Morimoto, Shota Motoura, Takeo Hosomi, Tien N. Nguyen, Hridesh Rajan, Nikolaos Tsantalis, Danny Dig  

**Link**: [PDF](https://arxiv.org/pdf/2503.20934)  

**Abstract**: MOVEMETHOD is a hallmark refactoring. Despite a plethora of research tools that recommend which methods to move and where, these recommendations do not align with how expert developers perform MOVEMETHOD. Given the extensive training of Large Language Models and their reliance upon naturalness of code, they should expertly recommend which methods are misplaced in a given class and which classes are better hosts. Our formative study of 2016 LLM recommendations revealed that LLMs give expert suggestions, yet they are unreliable: up to 80% of the suggestions are hallucinations. We introduce the first LLM fully powered assistant for MOVEMETHOD refactoring that automates its whole end-to-end lifecycle, from recommendation to execution. We designed novel solutions that automatically filter LLM hallucinations using static analysis from IDEs and a novel workflow that requires LLMs to be self-consistent, critique, and rank refactoring suggestions. As MOVEMETHOD refactoring requires global, projectlevel reasoning, we solved the limited context size of LLMs by employing refactoring-aware retrieval augment generation (RAG). Our approach, MM-assist, synergistically combines the strengths of the LLM, IDE, static analysis, and semantic relevance. In our thorough, multi-methodology empirical evaluation, we compare MM-assist with the previous state-of-the-art approaches. MM-assist significantly outperforms them: (i) on a benchmark widely used by other researchers, our Recall@1 and Recall@3 show a 1.7x improvement; (ii) on a corpus of 210 recent refactorings from Open-source software, our Recall rates improve by at least 2.4x. Lastly, we conducted a user study with 30 experienced participants who used MM-assist to refactor their own code for one week. They rated 82.8% of MM-assist recommendations positively. This shows that MM-assist is both effective and useful. 

---
# ACADATA: Parallel Dataset of Academic Data for Machine Translation 

**Authors**: Iñaki Lacunza, Javier Garcia Gilabert, Francesca De Luca Fornaciari, Javier Aula-Blasco, Aitor Gonzalez-Agirre, Maite Melero, Marta Villegas  

**Link**: [PDF](https://arxiv.org/pdf/2510.12621)  

**Abstract**: We present ACADATA, a high-quality parallel dataset for academic translation, that consists of two subsets: ACAD-TRAIN, which contains approximately 1.5 million author-generated paragraph pairs across 96 language directions and ACAD-BENCH, a curated evaluation set of almost 6,000 translations covering 12 directions. To validate its utility, we fine-tune two Large Language Models (LLMs) on ACAD-TRAIN and benchmark them on ACAD-BENCH against specialized machine-translation systems, general-purpose, open-weight LLMs, and several large-scale proprietary models. Experimental results demonstrate that fine-tuning on ACAD-TRAIN leads to improvements in academic translation quality by +6.1 and +12.4 d-BLEU points on average for 7B and 2B models respectively, while also improving long-context translation in a general domain by up to 24.9% when translating out of English. The fine-tuned top-performing model surpasses the best propietary and open-weight models on academic translation domain. By releasing ACAD-TRAIN, ACAD-BENCH and the fine-tuned models, we provide the community with a valuable resource to advance research in academic domain and long-context translation. 

---
# DSAS: A Universal Plug-and-Play Framework for Attention Optimization in Multi-Document Question Answering 

**Authors**: Jiakai Li, Rongzheng Wang, Yizhuo Ma, Shuang Liang, Guangchun Luo, Ke Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.12251)  

**Abstract**: While large language models (LLMs) show considerable promise across various fields, they have notable limitations in handling multi-document question answering (Multi-doc QA) tasks. The first challenge is long-range dependency modeling, where LLMs struggle to focus on key information in long texts, which weakens important semantic connections. Second, most LLMs suffer from the ''lost-in-the-middle'' issue, where they have difficulty processing information in the middle of long inputs. Current solutions either truncate global dependencies or demand costly finetuning, ultimately lacking a universal and simple solution for these challenges. To resolve these limitations, we propose Dual-Stage Adaptive Sharpening (DSAS) containing two modules. (i) The Contextual Gate Weighting (CGW) module alleviates ''lost-in-the-middle'' by assessing paragraph relevance through layer-wise attention tracking and position-aware weighting. (ii) The Reciprocal Attention Suppression (RAS) module enhances focus on critical paragraphs by suppressing information exchange between key and irrelevant texts, thus mitigating the limitations in long-range dependency modeling. Notably, DSAS functions as a plug-and-play solution requiring no architectural modifications or extra training parameters. Extensive experiments on four benchmarks demonstrate DSAS's efficacy across mainstream LLMs (Llama, Qwen, Mistral, and Deepseek), with an average F1-score improvement of 4.2% in Multi-doc QA tasks on Llama-3.1-8B-Instruct and Qwen2.5-14B-Instruct. Ablation studies confirm the essential contributions of both the CGW and RAS modules. In addition, detailed discussions in the Appendix further validate the robustness and scalability of DSAS. 

---
# Scaling Long-Horizon LLM Agent via Context-Folding 

**Authors**: Weiwei Sun, Miao Lu, Zhan Ling, Kang Liu, Xuesong Yao, Yiming Yang, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.11967)  

**Abstract**: Large language model (LLM) agents are fundamentally constrained by context length on long-horizon tasks. We introduce Context-Folding, a framework that empowers agents to actively manage their working context. An agent can procedurally branch into a sub-trajectory to handle a subtask and then fold it upon completion, collapsing the intermediate steps while retaining a concise summary of the outcome. To make this behavior learnable, we develop an end-to-end reinforcement learning framework FoldGRPO with specific process rewards to encourage effective task decomposition and context management. On complex long-horizon tasks (Deep Research and SWE), our folding agent matches or outperforms the ReAct baselines while using an active context 10$\times$ smaller and significantly outperforms models that rely on summarization-based context management. 

---
