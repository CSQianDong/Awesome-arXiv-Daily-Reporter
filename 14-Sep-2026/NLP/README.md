# Type Diversity Enables Transformers to Generalise Compositionally 

**Authors**: Anssi Moisio, Mathias Creutz, Mikko Kurimo  

**Link**: [PDF](https://arxiv.org/pdf/2609.13144)  

**Abstract**: Compositional generalisation has been divided into lexical and structural generalisation. Previous work has found that structural generalisation is harder than lexical for Transformers. We propose that this difference is not inherent to Transformers, but due to the high diversity of lexical types and low diversity of structural types in the specific datasets of these previous works. By type diversity we mean the number of different constructors of that type, instead of, for example, the specific word combinations that might populate the structure. To test this, we vary the amounts of type diversity of lexical and structural types in previously published datasets. We create linguistically diverse variants of the COGS and SLOG datasets using Grammatical Framework. We find that type diversity correlates with compositional generalisation equally in lexical and structural test cases, supporting our hypothesis. We note a contradiction with the proposition in previous work that compound divergence explains the difficulty in compositional generalisation tasks. We further investigate the effects of other dataset properties on compositional generalisation, such as the diversity of types other than the novel test structure, and surface properties of the logical semantics format. 

---
# SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking 

**Authors**: Zhiwei Li, Lei Zhu, Hao Gu, Xiang Hu, Yan Wang, Haitao Mi, Sirui Han, Leo Liang, Zhijiang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.13141)  

**Abstract**: Post-training attention sparsification reduces the quadratic cumulative attention cost of pretrained Transformers by selecting a small set of context units (tokens or blocks) for each query. Existing trainable methods usually use a lightweight selector to score context units, followed by hard Top-K selection that blocks gradients from the language modeling loss. Consequently, these methods commonly distill layer-wise dense attention distributions. Although this encourages the selector to rank context units by dense attention weights in the original model, the ranking is not directly aligned with their impact on predictions under a fixed attention budget (i.e., the number of attended context units per query), potentially wasting the limited budget on less useful units. To address this misalignment, we propose Simple Attention Sparsification (SAS), a gated sparse attention mechanism that optimizes context ranking end-to-end with the language modeling loss. The key idea is to inject the selector's continuous scores into attention logits during training, allowing the loss to update the selector through standard backpropagation. We identify several choices crucial for this simple design to work well in practice: placing the gate inside the attention softmax in log form, using normalized softmax gates to calibrate historical context against the always-retained current block, and preserving continuous selector scores so the model learns relative priorities rather than only hard selections. To support long-sequence training, we implement a memory-efficient Triton kernel that integrates SAS into FlashAttention-style computation. Across reasoning, long-context understanding, and agentic tasks, SAS consistently outperforms trainable sparse attention baselines across attention budgets, with especially large gains under tight budgets, demonstrating more effective context ranking for downstream tasks. 

---
# Continue, Adapt, or Yield: In-Turn Adaptation to Overlapping Speech in Full-Duplex Agents 

**Authors**: Yunqi Lu, Tyler Baumgartner, Nikhil Johri, Brandon Tai, Candice Fan, Luc Debaupte, Ruben Aguilar, Bill Wang, Yi Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2609.13117)  

**Abstract**: Full-duplex evaluation often emphasizes whether an agent keeps speaking or stops. That binary cannot express a third response humans use routinely: continuing to speak while incorporating what the listener just contributed. The contribution may be a missing word, a correction or a clarification. We introduce Duplex Cue, an evaluation of this \emph{in-turn adaptation} in full-duplex voice agents. Duplex Cue separates listener intent (backchannel, collaboration, or interruption) from speaker behavior: continuing unchanged, adapting within the turn, or yielding. Adaptation includes acknowledgment as well as content revision. In a single-model case study using 300 human-confirmed cues from unscripted English conversations, we compare recorded human responses with PersonaPlex continuations generated while replaying the listener's audio. We retain 208 pairs with the ongoing speaker active at cue onset and a scorable response in each condition. On the 66 collaborative pairs, recorded speakers adapt in 68.2\% of cases, compared with 34.8\% for PersonaPlex. The model otherwise continues unchanged (42.4\%) or yields (22.7\%). These findings show why evaluating natural voice interaction requires measuring how an agent responds to a listener's contribution as well as whether it keeps speaking. 

---
# Expert-Space Exploration in MoE Reinforcement Learning 

**Authors**: Hongyi He, Zhenghao Lin, Xiao Liu, Peng Cheng, Yan Lu, Yeyun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2609.13058)  

**Abstract**: Reinforcement learning (RL) has become central to post-training of large language models. Recent advances in RL for Mixture-of-Experts (MoE) models have primarily focused on improving optimization stability and training efficiency, while treating the expert selection as a fixed component. Since routing determines the sparse computation paths that induce output distributions, expert selection offers an additional source of rollout diversity. Through empirical analysis, we find that perturbing expert routing effectively alters model output and increases rollout diversity, which is similar to increasing the decoding temperature. However, direct perturbation can activate unsuitable experts and substantially degrade rollout quality. Motivated by these observations, we introduce Expert-Space Exploration Reinforcement Learning (ESRL), an architecture-aware framework that explicitly explores the expert-routing space of MoE models. ESRL preserves high-confidence experts as anchors, and restricts stochastic routing to a plausible candidate pool, thereby retaining reliable computation paths. The perturbation strength is further adapted according to router entropy to avoid over-perturbation. To mitigate the routing mismatch introduced by perturbation, ESRL records the expert paths used during rollout and replays them during policy optimization. Experiments demonstrate that ESRL achieves the best performance across MoE backbones with top-K, top-1, and shared-expert routing, as well as across mathematics, science, and code tasks without additional sampling or computational cost. Specifically, ESRL on Qwen3-30B-A3B achieves the best among all compared methods, improving average Pass@1 and Pass@8 over GRPO by 3.2 and 4.5 percentage points, respectively. Further analyses of expert utilization and training dynamics provide insights into how exploiting MoE-specific routing structure benefits RL training. 

---
# Kraken: LLM-based Speech-to-Speech Translation via Low-bitrate VQ and Dual-path Source Conditioning 

**Authors**: Hayato Futami, Hassan Shahmohammadi, Tushar Dhyani, Alkis Koudounas, Raphaël Lafargue, Yosuke Kashiwagi, Quentin Jodelet, Emiru Tsunoo  

**Link**: [PDF](https://arxiv.org/pdf/2609.13045)  

**Abstract**: Speech-to-speech translation (S2ST) has advanced significantly with speech LLMs, offering the potential for joint optimization and preserving non-linguistic information. However, these models struggle with predicting high-bitrate speech tokens in LLMs, and face the challenge of relying on S2ST training data with ideally aligned speaker identity and prosody. We propose using low-bitrate tokens based on single-layer vector quantization, trained to reconstruct self-supervised learning (SSL) features. We also employ a separate token-to-waveform decoder named Autowave-X, which is also conditioned on the source speech to improve non-linguistic transfer, thereby relaxing the training data constraints. With the integration of these techniques, we propose an S2ST model named Kraken, which augments a pre-trained LLM with speech feature inputs and the low-bitrate token outputs, followed by Autowave-X vocoder. We built the model upon Qwen3-8B and trained it using 150k hours of multilingual and multitask speech data. We demonstrated that our model exhibited better translation quality than SeamlessM4T-Large v2 and Qwen2.5-Omni, along with improved speaker and prosody transfer capabilities. 

---
# Tasks over Application Manuals: Revealing Gaps in Long-Horizon Procedural Reasoning for Language Models 

**Authors**: Utkarsh Soni, Syed Shariyar Murtaza, Yifan Nie, Sachin Chandrasekhar, Eugene Wen  

**Link**: [PDF](https://arxiv.org/pdf/2609.13005)  

**Abstract**: Large language models (LLMs) have achieved strong performance on a wide range of natural language tasks, and recent benchmarks suggest that they are increasingly adept at multi-hop reasoning. However, these benchmarks are typically short-horizon, requiring only a small number of retrieval or inference steps, and provide limited evidence of reliability on real-world tasks that involve following manuals spanning hundreds of pages with complex, interdependent guidelines. In this paper, we introduce Tasks over Application Manuals (TAM), a benchmark for evaluating long-horizon procedural reasoning. We construct TAM by curating real-world tasks from two domains: ICD-10-CM clinical coding (mapping medical conditions to diagnostic codes) and U.S. federal sentencing (computing crime sentencing guideline outcomes, specifically offense levels), with human-validated labels. Each task requires following an authoritative manual with tens of thousands of rules and executing a sequence of interdependent steps across different sections to produce an exact answer. We evaluate general-purpose prompting approaches, including retrieval-augmented generation, ReAct-style prompting, and an agent-harness baseline on GPT-5, and find that the best exact-match performance remains extremely low: 1% on ICD-10-CM coding and 15.5% on sentencing tasks. These results show that current benchmarks may overestimate LLM reasoning ability and miss a key challenge: reliably following long, rule-based procedures. The complete TAM data and code are publicly available. 

---
# Judging by the Cover: Cleaning LLM Truthfulness Benchmarks to Avoid Surface-Level Feature Leakage 

**Authors**: Foad Namjoo, Remy Ogasawara, Amirali Abdullah, Cullen Anderson, Narmeen Fatimah Oozeer, Jeff M. Phillips  

**Link**: [PDF](https://arxiv.org/pdf/2609.13003)  

**Abstract**: Binary-choice truth benchmarks ask models to choose between a correct and an incorrect answer, but if the two answers differ systematically in surface-level features, models can exceed chance without performing the intended reasoning. We show that this failure mode is detectable and can be exploited by downstream classifiers. In TruthfulQA, a simple six-feature logistic classifier achieves substantial accuracy in separating correct from incorrect answers. We further show that similar surface-level artifacts are present in additional benchmarks. To counteract this, we developed a general mechanism to clean them by removing the most leakage-reinforcing pairs. We release a version of TruthfulQA with surface-feature leakage reduced close to chance and provide a mechanism, Audit-Prune, so that the datasets can be cleaned before release. 

---
# Fewer Words, Not Fewer Tokens: Measuring the Sanskrit Tokenization Penalty per Proposition 

**Authors**: Devansh Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2609.12960)  

**Abstract**: Sanskrit fuses case, number, person and tense into word endings and chains clauses into compounds, so it is information-dense per word. Whether that density survives subword tokenization is a separate question, to be asked per unit of meaning rather than per word. On identical FLORES-200 devtest content, Sanskrit costs 1.774-2.187 times the English tokens under deployed tokenizers with vocabularies of 200,019 ids or more, but only 1.325-1.353 times the Hindi tokens. Against a deployed English tokenizer, Sanskrit-trained BPE arms then look cheaper per proposition than English on contemporary prose (0.887). Against a matched English control, the same algorithm and vocabulary trained on the English side of the same corpus, that flip disappears: at 32,000 and 64,000 pieces all 8 matched pairs, each size-matched arm against both a pair-matched and a byte-matched control, sit above 1.0 on prose with 95% intervals excluding it. The gap closes as the vocabulary grows: at 128,000 pieces the BPE pair reads 0.983 in domain while staying above parity out of domain (1.025) and on FLORES (1.116). The ratio factorises into a character-length ratio and a tokens-per-character ratio, the second near 1 throughout: what survives matched tokenization is character-level length, which Sanskrit prose lacks over English in SLP1 (1.028) and Sanskrit verse has (0.596). The robust statement is about deployed practice: on contemporary prose and on FLORES, with the Sanskrit side in SLP1 against the deployed o200k English pivot, Sanskrit costs 1.831-2.899 English tokens per proposition under the tokenizers people actually ship. Code, the results snapshot and every table here are public. 

---
# LLM-Enhanced Dual-Branch Learning for Large-Scale Multi-Label Text Classification 

**Authors**: Hui Ye, Jing Zhang, Xiulong Yang, Rajshekhar Sunderraman  

**Link**: [PDF](https://arxiv.org/pdf/2609.12915)  

**Abstract**: Large-scale multi-label text classification assigns a small subset of relevant labels to each document from a vocabulary containing thousands or tens of thousands of candidate labels. Although pretrained language models have improved semantic text representations, most representation-based approaches center their prediction pipelines on a primary encoder or combine auxiliary features within a single ranker. The complementarity between heterogeneous language models therefore remains insufficiently explored. We propose DualMLC, a dual-branch framework that processes the same document through an autoregressive decoder-only language model and a bidirectional encoder. Each branch maintains its own representation pathway and independently estimates relevance scores over the shared label space. DualMLC combines the two score vectors through late logit fusion, allowing shared evidence to reinforce relevant labels and branch-specific evidence to compensate for limitations in the other branch's representation. DualMLC achieves state-of-the-art results on three widely used large-scale multi-label text classification benchmarks. Ablation results further confirm that integrating the heterogeneous predictors produces stronger rankings than either branch alone. The source code is publicly available at this https URL. 

---
# Parameter-Efficient Retrievers for Polish and European Languages 

**Authors**: Sławomir Dadas, Rafał Poświata, Małgorzata Grębowiec, Michał Perełkiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2609.12913)  

**Abstract**: Dense retrieval systems increasingly rely on multi-billion-parameter language models, whose memory and computational requirements make large-scale indexing, frequent corpus updates, and low-latency serving costly. We present a three-stage training pipeline for developing compact and efficient retrievers that remain competitive with substantially larger models. The pipeline combines cross-lingual alignment, relational knowledge distillation, and contrastive fine-tuning. It requires no original ground-truth relevance labels, relying exclusively on supervision generated by strong embedding models and rerankers utilised as teachers. Using this pipeline, we develop PolDense and EuroDense, both supporting contexts of up to 8,192 tokens. PolDense is a family of six Polish retrievers ranging from 17M to 1B parameters. EuroDense is a 435M-parameter retriever supporting nine European languages. We conduct an extensive evaluation covering 41 Polish and 150 multilingual retrieval tasks. The results demonstrate strong quality-efficiency trade-offs. PolDense-1B outperforms the evaluated retrievers with up to 9B parameters, while the PolDense family forms the Pareto frontier across model sizes. Among the evaluated models below 1B parameters, EuroDense ranks first in both task-averaged and language-averaged performance and leads in seven of nine languages. We release all models publicly. 

---
# MedSNIP: Building and Benchmarking Snippet-Level Granularity for Medical Fact Verification 

**Authors**: Hasan Iqbal, Sarfraz Ahmad, Hyunjae Kim, Sihyeon Park, Junjie Liao, Qingyu Chen, Preslav Nakov, Yuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12884)  

**Abstract**: A medical claim's correctness often depends not on the claim alone, but on the clinical structure around it. A claim may require a lab reference range, a causal or conditional link, or patient-specific details to be judged correctly, and atom-level decomposition can fragment these dependencies, leaving the verifier with clinically incomplete claims. We reformulate medical fact-checking around snippet-level verification, where clause-grouped units preserve local clinical structure. We introduce MedSNIP-Bench, a human-annotated benchmark for snippet-level medical fact verification, and MedSNIP, an automatic snippet-generation pipeline. MedSNIP-Bench covers 276 consumer-health and clinical-vignette responses, segmented into 2,524 snippets with dual in-general and in-patient-context labels and six structural pattern codes. MedSNIP is evaluated against human snippet boundaries on MedSNIP-Bench and then used to generate snippet-level units for external corpora. Across MedSNIP-Bench, HealthFC, and MedHallu, snippet-level verification preserves or improves false-class F1, with gains concentrated where answers are long enough to fragment and where the verifier is strong enough to exploit the recovered structure. The largest merge-pattern gain is on causal-conditional clinical chains. It also reduces verifier calls by 24-73%, though the saving survives end-to-end only when decomposition is cheap, which an open-weight decomposer makes possible at no loss of chunking fidelity. 

---
# DuplexDrama: A Synthesized Dialogue Dataset with Scenarios, Full-Duplex Behaviors, Expressive Speech, and Sound Events 

**Authors**: Qingxiang Guo, Wenke Fan, Shuofeng Zhao, Dawei Yang, Zhiyang Zhou, Yingxin Shang, Hongwei Cai, Zhou Wang, Weixu Wang, Lin Yang, Shuran Zhou, Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2609.12872)  

**Abstract**: We present DuplexDrama, the first synthesized spoken dialogue dataset that simultaneously covers four dimensions: (i) complete persona and scenario settings; (ii) three full-duplex behaviors (interruption, backchannel, incomplete); (iii) expressive speech with persona-aligned emotion labels; and (iv) script-aware sound events. DuplexDrama is built via a 4-stage pipeline; quality validation on both scripts and synthesized audio confirms its quality. We have produced more than 2,000 hours audio data with a 64-voice timbre pool spanning 13 personas and 5 age buckets; 3.8% of all turns carry at least one full-duplex behavior. This data has been validated through internal full-duplex model training. We will release a curated subset of 6,400 bilingual dialogues (800 h, Chinese ~500 h + English ~300 h) to advance full-duplex spoken dialogue model research. Data samples are available at our demo page and LLM-judge evaluation prompts will be released with the dataset. 

---
# Cognition on Graph: Navigating Massive Knowledge Space via Cognitive Cycles and Bidirectional Graph-Text Synergy 

**Authors**: Gengxian Zhou, Jian Xu, Zichen Tang, Shiming Xiang, Haihong E, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12791)  

**Abstract**: Retrieval-Augmented Generation (RAG) has empowered Large Language Models (LLMs) to tackle knowledge-intensive tasks. However, navigating global, heterogeneous knowledge bases (large-scale knowledge graphs and text corpora) for complex reasoning remains a challenge. Existing methods typically employ reactive, graph-driven exploration strategies, which blindly follow graph topology without adapting to the question context or evolving exploration progress, and lack deep bidirectional synergy between graph and text. To address these limitations, we propose CoG (Cognition on Graph), a cognitive-inspired, training-free framework for adaptive knowledge exploration. Drawing inspiration from human problem-solving, CoG performs a continuous plan-explore-reflect cycle, where it proactively formulates investigation plans, performs dual-source retrieval, and dynamically reflects on progress to adjust strategies. Crucially, it establishes deep bidirectional synergy between structured graph and unstructured text, where entities extracted from text dynamically guide graph exploration to bridge knowledge gaps. Extensive experiments on seven multi-hop QA benchmarks demonstrate that CoG significantly outperforms state-of-the-art methods while achieving superior exploration efficiency. Our code and datasets are available at this https URL. 

---
# Doc2FRC: Length-Consistent Document-Level Machine Translation via Fixed-Range Chunking 

**Authors**: Xiaotian Wang, Youyuan Lin, Zhan Shen, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2609.12674)  

**Abstract**: Advanced large language models (LLMs) with long context windows can substantially reduce input truncation in document-level machine translation (DocMT). However, direct Doc2Doc translation remains prone to n-gram repetition and progressive quality degradation. A common remedy is to segment the document into finer-grained chunks. Nonetheless, conventional rule-based chunking approaches fail to handle the length distribution mismatch between training and inference. To address this, we introduce Fixed-Range Chunking (FRC), utilizing dynamic programming to partition documents into chunks within a predefined length interval. By consistently applying FRC during training and inference, the input documents of any length are mapped to the same length distribution, substantially reducing train-test length mismatch. Centered on FRC, we propose a lightweight dual-boundary matching algorithm for chunk alignment, alongside four distinct training strategies. Experimental results show that FRC-based fine-tuning substantially improves 7B LLMs over direct Doc2Doc fine-tuning and outperforms existing DocMT methods on IWSLT2017. We further construct GlobVDoc, a 10-language test set independent of mainstream DocMT training sources, and show that FRC improves out-of-distribution document translation. 

---
# LifeMem: Enabling Lifelong Experience Reuse for LLM Agents 

**Authors**: Yuli Qiu, Yutong Li, Wei Su, Zeming Liu, Wanxiang Che, Heyan Huang, Haifeng Wang, Yuang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.12655)  

**Abstract**: Large language model agents are expected to continuously adapt to new tasks and environments over their lifetime by reusing past experience. However, existing memory-based agents struggle to transfer reusable experience across environments and suffer from catastrophic forgetting as experience accumulated. To address these challenges, we propose LifeMem, a lifelong learning framework that enables agents to transfer knowledge across multiple environments. During learning, LifeMem clusters accumulated interaction trajectories based on underlying workflows to extract reusable skills. When solving a new task at inference time, the agent recalls relevant skills and trajectories to guide actions. To validate our method, we conduct experiments across 10 environments and over 13k tasks with 2k newly annotated interaction trajectories. Results show that LifeMem enables effective experience reuse in lifelong learning, achieving both reduced forgetting on learned tasks and superior cross-task transfer. Further analysis reveals that task streaming impacts learning, while consolidating structurally similar trajectories within memory boosts performance. 

---
# SWARM: A Multilingual Human-Annotated Dataset for Russian Propaganda Detection in Search Engine Results 

**Authors**: Manuel Tonneau, Abhinav Dubey, Farhan Shaikh, Ilaria Vitulano, Martha Stolze, Hale Dedeoglu, Clara Riechert, Ella Kuka, Maryna Sydorova, Mykola Makhortykh, Elizaveta Kuznetsova  

**Link**: [PDF](https://arxiv.org/pdf/2609.12653)  

**Abstract**: Russian state propaganda spreads across many languages and online spaces. Yet, most computational work examines only one such space, usually social media, in one or two languages, and analyses sources rather than content. We introduce SWARM (Search-Web documents Annotated for Russian propaganda, Multilingual), a dataset of 2,183 search engine results across nine languages and diverse web domains (e.g., news, blogs, government sites), each annotated by trained coders for whether it supports a recurring Russian propaganda narrative. We benchmark a source-based blocklist, supervised classifiers, and zero-shot LLMs against these labels. The blocklist misses most propaganda-supporting documents, because such content is not confined to flagged "propaganda" outlets but also appears on mainstream ones. Content-level analysis helps, though how much depends on the model: the strongest LLM reaches a positive-class F1 of 0.73, whereas the supervised classifiers reach only about 0.5, with the smaller LLMs over-predicting support, mistaking topical relevance for endorsement. Detecting search-borne propaganda thus requires per-language, content-level evaluation, which we hope SWARM and our evaluation code enable. 

---
# Calibrated Ambiguity in Multimodal Language Models: Humans reach for cultural references, while models describe the picture 

**Authors**: Cody Kommers, Mingrui Ye, Evelyn Gius, Daniela Mihai, Hoyt Long, Zheng Yuan, Drew Hemment  

**Link**: [PDF](https://arxiv.org/pdf/2609.12575)  

**Abstract**: Ambiguity is often treated as a bug for AI systems to resolve---but in human communication and culture, ambiguity can also be a generative resource. From humour to politics to art, people express themselves in words and images that are open enough to invite different interpretations, yet constrained enough to be interpretable. We operationalise this notion of calibrated ambiguity with a task drawn from the parlour game Dixit. We compare differences in clues generated by human vs multimodal language models, based on a novel coding rubric for calibrated ambiguity, and find that models consistently exhibit ambiguity collapse (i.e., their outputs are over-specified, leaving no room for multiple legitimate interpretations). Unlike human clues, AI-generated clues also exhibit cultural flattening; they almost never make reference to culturally-situated knowledge, even when prompted to use allusion and figurative language. 

---
# Meddies-PII: A Multilingual Framework for Personally Identifiable Information Extraction in Clinical De-identification 

**Authors**: Linh Uyen Le, Christian Hoang, Huy Hoang Ha  

**Link**: [PDF](https://arxiv.org/pdf/2609.12544)  

**Abstract**: Clinical de-identification relies on accurately identifying personally identifiable information (PII). However, manually annotated datasets are costly to construct, while existing synthetic alternatives often provide limited details about their generation process or rely on relatively simple synthesis strategies. We introduce Meddies-PII-Dataset, a corpus of one million synthetic clinical documents spanning seventeen languages and nine PII labels. The documents are generated using attribute-conditioned prompts and validated through thirteen deterministic gates that enforce structural and annotation consistency. To evaluate the dataset's utility, we train Meddies-PII-Model, a BIOES token classifier, and compare it with existing PII extraction systems using exact-match entity-level F1. Meddies-PII-Model achieves the highest performance among the evaluated systems on all reported benchmarks, with a mean F1 of 0.827 across fifteen external benchmarks, compared with 0.658 for the strongest baseline. Upon acceptance, we will publicly release the dataset, benchmark suite, model, generation framework, and evaluation code to support research on multilingual clinical de-identification. 

---
# Agent as Policy for Robotic Manipulation 

**Authors**: Mengzhao Jia, Yang Lin, Xixin Zhang, Zhihan Zhang, Xiaobai Liu, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12541)  

**Abstract**: We demonstrate that a general-purpose agent can directly drive a physical robot throughout task execution without any task-specific or environment-specific training. We introduce Agent as Policy (AGP), which places task planning and execution under the agent's control. Given a task and a robot interface, the agent interprets visual evidence, writes executable programs, issues motion commands, and revises its actions in response to physical outcomes. This brings the agent's reasoning and programming capabilities into continuous interaction with the physical world. We study AGP across multiple real-world manipulation tasks spanning precision manipulation, dynamic motions, and deformable objects. These include assembly from human videos, block construction from goal images, die reorientation, targeted throwing, and bimanual towel folding. AGP achieves success rates of 100%, 100%, and 80% on three block construction configurations. These findings establish a path for general-purpose agents to act as robotic policies, extending their autonomy to physical manipulation through runtime reasoning, programming, and interaction. 

---
# The House with a Million Windows: Interactive Fiction for Narrative Restorying 

**Authors**: Cody Kommers, Sarah G Immel, Drew Hemment, Mina Lee  

**Link**: [PDF](https://arxiv.org/pdf/2609.12537)  

**Abstract**: AI-assisted writing can flatten meaning in human storytelling, enabling the production of homogeneous outputs without the intentional effort and sense-making writing entails. To address this challenge, we present The House with a Million Windows (HWAMW), an LLM-based interactive fiction system designed to help users explore both the breadth and depth of potential meanings within their personal stories -- drawing on a psychological paradigm called the restorying intervention. In HWAMW, users play through a text-based narrative in which they tell a story, then encounter a set of LLM-generated "windows" reframing it according to different literary styles. Empirical evidence shows that HWAMW increases users' sense of narrative identity, while an expert review explores how this effect is achieved. Our findings suggest that HWAMW facilitates restorying and offers a valuable paradigm for AI-assisted writing, wherein LLMs do not tell our stories but rather help us see greater potential in the stories we tell. 

---
# Zipbench: Low-Cost Framework for Compressing Comprehensive Benchmarks of Large Language Models 

**Authors**: Zhongzhan Huang, Junxin Li, Guoming Ling, Yupei Lin, Shanshan Zhong, Hefeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12475)  

**Abstract**: Comprehensive benchmark suites are essential for improving large language models (LLMs), but many widely used benchmarks are redundant, making evaluation unnecessarily expensive. Although recent benchmark compression methods (BCMs) can mitigate this cost, many strong BCMs rely on large collections of per-sample evaluation results from numerous LLMs to identify representative samples. Building such collections is also expensive unless they are already public, making these methods difficult to extend to newly released benchmarks. To address this challenge, we present ZipBench, a simple and low-cost BCM with theoretical error and rank-consistency guarantees. ZipBench evaluates only a small set of anchor LLMs, synthesizes pseudo evaluation results to broaden coverage, learns compact sample representations, and selects a small yet representative subset. Building on it, we create ZipBench Zoo, a collection of compact versions of 100+ benchmark proxies spanning text, multimodal, and agent tasks. These benchmark achieve mean absolute errors of 0.002--0.02 and average Spearman correlations of ~0.98 with the full benchmarks. Overall, ZipBench reduces the cost of both LLM evaluation and compact benchmark construction, lowering the barrier to broad LLM research for compute-constrained researchers. The code has been released in this https URL. 

---
# AMDKernelVault: Large-Scale Datasets and Agentic Training for AMD GPU Kernel Optimization 

**Authors**: Ji Liu, Saptarshi Majumder, Yiqing Huang, Wenwen Ouyang, Umang Pandey, Zeping Li, Chushi Chen, Zihao An, Puyuan Yang, Zekai Li, Sina Rafati, Ziqiong Liu, Pratik Prabhanjan Brahma, Dong Li, Zicheng Liu, Sharon Zhou, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2609.12471)  

**Abstract**: We introduce AMDKernelVault, an open HIP and Triton kernel corpus and training framework for recent AMD CDNA GPUs. Existing LLM-based kernel agents are largely CUDA/NVIDIA-centric and often depend on repeated frontier-LLM calls for generation, reflection, and optimization. To address this gap, we develop HIPKernelGen and TritonKernelGen, agent-driven pipelines that transform PyTorch references into HIP or Triton kernels, compile and validate candidates under ROCm, and latency-profile them on AMD hardware. The corpus contains 62,153 execution-verified HIP kernel samples, 2,377 production-grounded ROCm Libraries QA entries, and 39,893 Triton kernels. We further train Qwen3-8B with supervised fine-tuning and execution-aware reinforcement learning as a demonstration of the corpus's utility. Under fixed evaluation budgets, it achieves the highest correctness among the compared models on PyTorch-to-HIP (34.0% Pass@1), TritonBench-G (33.2% Corr@3), and ROCmBench (41.94% Corr@3), but does not uniformly lead compilation or speed metrics. The corpus and documentation are available at this https URL, and the associated training and kernel-generation code is available at this https URL. 

---
# Not All Speech Is Intent: Adaptive Self-Correcting Inference Layer for Post-ASR False Wake-Up 

**Authors**: Preeti Saraswat, Divya Neelagiri, Anil Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2609.12469)  

**Abstract**: False wake-up activations remain a persistent challenge in conversational AI. Speech phonetically similar to a device's wake word can produce a syntactically valid and semantically coherent ASR transcript that the assistant incorrectly executes. Most existing systems make a single intent decision in isolation, without a mechanism to learn from recurring errors over time or adapt to individual users through personalized learning. We introduce the Feedback-Driven Adaptive Self-Correcting Inference Layer (ASCIL), a complementary post-ASR correction framework that re-evaluates wake-up intent before response generation by fusing acoustic embeddings, linguistic cues, device context, and patterns from past misclassifications. ASCIL interprets implicit signals, including hesitation, disengagement, and silence, and explicit signals, including cancellation and repetition, as automatically inferred, noisy behavioral indicators of potential misclassification. These signals drive online pattern updates without manual annotation, whereas the intentional/unintentional reference labels used for offline evaluation are human-annotated. It generalizes from prior errors, applies corrective adjustments at inference time, and continuously updates in parallel with natural-language execution. Evaluated on a proprietary dataset of 3,667 interactions with human-annotated intentional/unintentional reference labels spanning 14 acoustic and contextual conditions, ASCIL achieves 54.27% relative error reduction on a session-disjoint subset constructed from baseline failures, and up to 24.39% relative error reduction at threshold 0.90 on the issue-tagged evaluation slice. These gains are achieved while improving intentional acceptance rates, with a median added latency below 60 ms in the reported benchmark. 

---
# GraphProfiler: Source-Linked Sensitive Attribute Inference via Personal Knowledge Graphs 

**Authors**: Ahmed Sohair Khan, Estrid He, Chenglong Ma, Monica Wachowicz, Elham Naghizade  

**Link**: [PDF](https://arxiv.org/pdf/2609.12448)  

**Abstract**: Sensitive attributes such as age, income, and occupation can be inferred from user-generated content by aggregating indirect cues across many ordinary posts. LLM-based profilers can perform this aggregation automatically and with high accuracy, which makes large-scale personal attribute inference a major privacy threat. Existing LLM-based profilers, however, offer limited insight into which specific posts, concepts, and relationships made an inference possible, which is key to targeted privacy mitigation, i.e., redacting or rewriting only the few posts that actually leak an attribute, rather than perturbing entire histories. We introduce GraphProfiler, an auditable LLM-based profiler that represents each user's post history as a source-linked personal knowledge graph where nodes and edges trace back to the originating post and resolves attribute predictions to cited graph records and source texts. GraphProfiler reaches 86.7% attack success rate on the eight-attribute SynthPAI benchmark, within two points of strong text-only baselines, and 84.6% on PANDORA, while citing supporting evidence for over 98% of predictions. Our controlled ablation experiments provide evidence that the cited posts contribute to attack success, as removing them reduces the attack success rate substantially more than removing an equal number of random posts. 

---
# Representation-based Masked Diffusion Model 

**Authors**: Yangrong Hu, Ding Huang, Xueyu Zhou, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12382)  

**Abstract**: Masked Diffusion Models (MDMs) have emerged as a compelling paradigm for language modeling, offering the capability for efficient parallel text generation. However, existing parallel sampling methods typically update multiple masked tokens independently and ignore the complex mutual dependencies among the masked tokens. This independent updating mechanism lacks global coordination and might lead to incoherent outputs. To address this limitation, we propose Representation-based Masked Diffusion Model (RMDM), a framework that leverages the text representation to explicitly encode global semantics and help to parallel update tokens more precisely. Specifically, we first encode text into a continuous semantic space using a pretrained encoder and learn an invertible transformation that normalizes the representation distribution to a Gaussian prior, facilitating efficient sampling during generation. Conditioned on this latent semantic representation, we train a masked diffusion model to learn the conditional text distribution, where the representation serves as global semantic guidance to coordinate parallel token updates and faithfully approximate the target distribution. Empirical results demonstrate that RMDM significantly improves generation quality, particularly in aggressive few-step sampling regimes. 

---
# ORQA: An Occupation-Realistic Question and Answer Framework for LLM Professional Knowledge 

**Authors**: Shreyas Krishnan, Serina Chang, Abhishek Nagaraj  

**Link**: [PDF](https://arxiv.org/pdf/2609.12366)  

**Abstract**: We present ORQA, a method for testing occupation-level knowledge in large language models. Prior methods either map abstract LLM skills to occupations via task definitions or utilize expert knowledge which is difficult to obtain at scale and expensive. ORQA complements both of these methods by connecting O*NET occupations to trusted occupation-specific websites (such as regulatory agencies, licensing bodies, professional organizations, and government publications) and converting these into source-traceable question-answer pairs. A combination of an automated pipeline and human review produces a set of high quality questions about occupations. The question set created via our method covers 116 occupations from all 21 major groups in the SOC, with 480 questions sourced from 187 different websites. Each question is designed to probe a real-world skill question that is relevant to the occupation in question. We test 15 state-of-the-art frontier and open-weight models via this method. Claude Opus 4.6, GPT-5.4 and Claude Sonnet 4.6 all perform the best at approximately 58-62% while smaller open-weight models achieve approximately 33-41% performance. Performance varies significantly across occupations. Healthcare-related occupations achieve the highest performance (78%) while Office and Administrative Support achieve approximately 40%. Performance on individual occupations (e.g. Sheet Metal Workers and Fish and Game Wardens) is essentially zero. We also find that open-ended questions and weighting by wage bill do not significantly affect the ranking of models on this benchmark. We believe that leveraging existing trusted occupation-specific information to test LLM knowledge in professional domains may be a scalable and useful method for evaluating occupation-level AI performance in the future. Results and data are available at this http URL. 

---
# CueMem: Cue-Guided Context Reconstruction for Long-Term Conversational Memory 

**Authors**: Changjian Wang, Rongzhen Li, Weili Guan, Shuming Shi, Quan Lu, Ning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2609.12354)  

**Abstract**: Long-term conversational agents must answer user queries by recalling information from extended dialogue histories, yet directly using the full history is costly and often unreliable, while compressed memory units may lose fine-grained evidence needed for question answering. Motivated by the reconstructive view of autobiographical memory, we propose CueMem, a cue-guided framework that treats extracted memory records as retrieval cues rather than self-contained evidence and reconstructs query-relevant dialogue context from their source turns. During memory construction, CueMem extracts fine-grained memory cues from dialogue turns and links each cue to its source turn. At query time, it retrieves query-relevant cues, maps them to source-turn anchors, and expands from these anchors over a turn graph that captures temporal proximity and semantic relatedness, reconstructing a compact evidence context from the original dialogue for LLM answer generation. Experiments on LoCoMo and LongMemEval show that CueMem consistently outperforms representative long-term memory baselines. Further analyses show that graph-based context reconstruction helps recover supporting dialogue evidence while reducing query-time input tokens and latency compared with the full-history LLM setting. These results highlight retrieval cues as an effective alternative to self-contained memory evidence for long-term conversational question answering. 

---
# SynthSentry: Detecting Synthetic Data Contamination in Language Model Training Data 

**Authors**: Praveen Kumar Myakala, Ravichandra Namburi, Sowmya Keragodu Jayaramu, Sooraj George Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2609.12353)  

**Abstract**: Large language models trained recursively on their own or other models' outputs undergo model collapse, in which distributional tails and factual accuracy deteriorate while fluency survives. Prior work diagnoses collapse after training; the actionable problem is screening a corpus of unknown provenance before training. We introduce SynthSentry, a corpus-level, model-agnostic contamination signal requiring no access to the generating model, no generation history, and no synthetic labels. The score is a distributional divergence over three statistics: lexical diversity collapse, n-gram tail truncation, and perplexity variance across reference models. We evaluate on corpora contaminated by small open-weight generators and an instruction-tuned open-weight model under a leave-one-generator-out protocol. A domain-stratified study measures false positives on naturally repetitive human text (legal, clinical, source code). The score ranks corpora by severity with little loss when whole generator families are held out. Per-domain calibration holds near its nominal false-positive budget once covariance shrinkage and a bootstrap threshold replace a naive quantile, which runs four times over budget. A downstream fine-tuning check showed no contamination-driven accuracy deficit at our scale, so whether pruning recovers one remains open; the same run shows over-pruning risk once pruning exceeds the true contamination fraction. We frame screening as a data-curation defense rather than a post-hoc diagnosis and release the scoring toolkit. All results are small-scale; scope is English-language, batch-mode corpus screening. Contamination sources are single-generation or hand-authored rather than recursively generated, so results speak to synthetic contamination generally and not to recursion depth. 

---
# I Am No One: Style-Aware Paraphrasing for Text Anonymization 

**Authors**: Ahmed Sohair Khan, Estrid He, Monica Wachowicz, Elham Naghizade  

**Link**: [PDF](https://arxiv.org/pdf/2609.12341)  

**Abstract**: Authorship attribution models can re-identify users from seemingly anonymized text by exploiting stable stylistic fingerprints, even after explicit identifiers are removed, posing a growing privacy risk for text publishing and analytics. This risk extends to speech-derived text such as ASR transcripts of meetings and call-center conversations, where stylometric leakage can persist even after acoustic anonymization. Differential privacy-based anonymization often severely degrades text quality and utility. We propose a style-aware, prompt-driven anonymization approach that uses pretrained large language models to construct compact stylistic profiles from minimal samples and rewrite text to suppress identifiable style markers while preserving meaning. Across blog and review datasets, our approach reduces authorship attribution F1 by 60-70% while maintaining content quality and readability, substantially outperforming DP-based and non-DP baselines. 

---
# ESTS at WMT26: Routing-Informed Expert Pruning for Model Compression 

**Authors**: Liu O. Martin, Lucas Bandarkar, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2609.12310)  

**Abstract**: We describe six submissions under the team name ESTS to the unconstrained WMT26 Model Compression Shared Task for English--Simplified Chinese and English--Egyptian Arabic. We submit three compression operating points per translation direction, all derived from GPT-OSS-20B. We use task-specific routing mass to rank experts and cross-lingual routing divergence to allocate retained capacity across layers, then physically remove low-importance experts. The resulting specialists are recovery-tuned on GPT-5.1-generated synthetic translation data and further compressed by applying MXFP4 quantization to the retained expert projection weights. We additionally implement a robust inference system for the instruction-conditioned WMT26 setting, including category inference, output validation, retries, segmented fallback, and source-owned JSON reconstruction. Across our six submissions, parameter counts range from 4.186B to 7.770B and packed artifact sizes from 4.55 to 6.33~GiB. Internal xCOMET-XL evaluation using GPT-5.1 pseudo-references provides an internal comparison across the submitted compression operating points. 

---
# Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models 

**Authors**: Kalyani Marathe, Artidoro Pagnoni, Tomasz Limisiewicz, Margaret Li, Mike Lewis, Luke Zettlemoyer, Srinivasan Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2609.12303)  

**Abstract**: Small models are made more capable through distillation from a larger one that shares their tokenization scheme. However, do distilled byte and token models behave similarly in terms of scaling trends as compute and data increases? To enable this comparison, we introduce two variants to efficiently convert token logits to Byte Logits: 1) approximate: Marginalize-It, and 2) exact: End-Of-Token. We then present the first large scale study of overtraining decoder-only dense transformer models varying two dimensions simultaneously: the tokenization scheme (Tokens, Bytes, Bytes w/ eot) and the training objective (Distillation vs. Cross-Entropy), sweeping layer-parameter-matched models with roughly 1 billion parameters up to 1 trillion bytes of data. Across eight benchmarks spanning three categories: Multiple Choice QA, Language Generation, and Machine Translation, we find that Token-1B models outperform byte models (End-Of-Token-1B and Bytes-1B) in the low-FLOP regime but eventually plateau; byte models start worse yet surpass Token-1B models with more compute, reaching a higher downstream task performance ceiling. Extrapolating the average top-1 error vs. validation BPB scaling laws predicts that, asymptotically, distilled End-Of-Token-1B outperforms distilled Token-1B by up to 4%. They are also far more data efficient, matching the performance of distilled Token-1B using only one-sixth of the training data. Moreover, by operating over a small vocabulary of 256 bytes instead of on the order of 100K tokens, they circumvent the need for top-k truncation during logit dumping, while also reducing logit storage costs to roughly one-fifth. Finally, our downstream performance scaling laws predict that our distilled End-Of-Token-1B models asymptotically surpass the Llama 3.2-1B, Gemma-3-1B-pt, and Gemma 2B models on averaged downstream tasks by up to 6.5%, 8.1%, and 2.1%, respectively. 

---
# EAR: Entity-Aware Partitioning Approach for Retrieval-Augmented Generation Development 

**Authors**: Cenab Batu Bora, Oylum Alatlı, Sebnem Bora, Oguz Dikenelli  

**Link**: [PDF](https://arxiv.org/pdf/2609.12268)  

**Abstract**: Retrieval-augmented generation (RAG) can improve knowledge-intensive question answering, but the first design choice is easy to overlook: how should the source corpus be partitioned into retrievable units? Fixed-size chunks often return long passages whose relation to the question is only implicit. We introduce EAR, an Entity-Aware Partitioning approach for multiple-choice question answering (MCQA). EAR extracts normalized surface anchors from the question, answer options, and corpus; retrieves local windows around matching corpus anchors; and can attach a larger parent passage through an extractive summary. We evaluate EAR on a cleaned Massive Multitask Language Understanding (MMLU)-style subset of 153 questions selected by an automatic corpus-support heuristic and using decontaminated public textbook text. Across same-protocol top-k = 3 and top-k = 8 sweeps with Mistral, Gemma, and DeepSeek, EAR entity-window reduces retrieved words by 37.5-40.2% relative to chunks. Observed accuracy changes are +5.2, +1.3, and -3.9 points at top-k = 3, and +5.9, -3.3, and -4.6 points at top-k = 8; none of the entity-window differences is statistically significant. The scoped contribution is methodological: EAR provides a compact and inspectable retrieval unit, while its rule-based anchor extractor remains domain-specific and requires separate validation before transfer. 

---
# HypoKG: Evidence-Disciplined Biomedical Hypothesis Generation Beyond Endpoint Knowledge 

**Authors**: Dominic Okonkwo, Adetayo Okunoye, Ismailcem Budak Arpinar  

**Link**: [PDF](https://arxiv.org/pdf/2609.12260)  

**Abstract**: Large language models (LLMs) can generate biomedical hypotheses, but it remains unclear whether they truly reason from scientific evidence or simply produce convincing-sounding ideas. To study this, we combine three major biological databases: the Kyoto Encyclopedia of Genes and Genomes (KEGG), Rhea, and UniProt, into a unified biochemical knowledge graph and construct a benchmark of 550 paths connecting enzyme sources to rare disease endpoints, yielding 13,200 hypotheses from six LLMs under four conditions varying the biological information each model receives: source enzyme only, full biological path, or source and disease endpoint only. Hypotheses are scored using an expert-derived five-criterion rubric on a 1-5 scale per criterion. We find that models given both the source and disease endpoint often produce the highest-scoring hypotheses, showing that LLMs can generate compelling ideas from minimal information. However, these hypotheses are less grounded in the evidence. In contrast, models given the full biological path generate hypotheses more consistent with known mechanistic relationships. We call this evidence-disciplined reasoning. To confirm this effect, we shuffled intermediate path steps while keeping endpoints fixed. Evidence grounding dropped significantly (delta = -0.793, p < 0.001), confirming models genuinely used path structure during reasoning. Our findings show that knowledge graphs support hypothesis generation in two ways: they identify biological endpoint pairs absent from the literature, and their mechanistic paths guide how LLMs reason between them. 

---
# Automated Detection and Structuring of Social Tipping Point Evidence in Climate related Documents: A Modular AI Framework 

**Authors**: Kavindu Perera, Mohammad Abaeiani, Ekaterina Gilman, Lauri Loven, Mourad Oussalah, Tassos Kanellos, Beatrice Gobbo, Dante Adami, Nicolò Ferriani, Maximiliano Romero, Pierre Rossel, Marc Bonazountas, Christina Deligianni, Nikos Xyderis, Artur Bogucki, Lampros Argyriou, Prasasthy Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2609.12254)  

**Abstract**: The climate literature has grown faster than review teams can read it. That gap matters most for a concept like the environmental social tipping point, the threshold at which a small change triggers rapid, self-reinforcing change in a social system. Evidence of this kind of shift is usually contained in one or two paragraphs within a longer document. As a result, existing text mining tools-which categorize entire documents by topic or highlight isolated claims-leave an expanding set of important evidence without any systematic method for discovery or organization. This paper presents an open and modular transformer-based framework that detects and structures social tipping point evidence at the passage level. The framework joins five components into a single deployable workflow: a DistilBERT boundary splitter for segmentation, an iteratively augmented RoBERTa classifier for detection, a Mistral 7B model that rewrites each detected passage for clarity, a LLaMA 3.2 3B model that rates the passage against five published social tipping point criteria, and a Milvus vector store for semantic retrieval. The system is wrapped in a Streamlit interface backed by MinIO object storage. Evaluated on a 163-passage benchmark labelled by GPT-4.1 and a 51-passage set reviewed by experts, the splitter surpassed three competing methods on a nine-metric composite score (6.137). The tuned RoBERTa model achieved 71.4 percent accuracy with a Cohen's kappa of 0.337 on the full benchmark, and 87.5 percent accuracy with a kappa of 0.742 on passages with labels, outperforming both a climate-focused model and untuned language models. 

---
# Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding 

**Authors**: Minoo Ahmadi, Seyedarmin Azizi, Erfan Baghaei Potraghloo, Mehdi Kamal, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2609.12243)  

**Abstract**: Inference-time power sampling via Sequential Monte Carlo (SMC) can substantially improve large language model (LLM) reasoning without requiring post-training. However, many existing SMC approaches rely on equal-weight resampling, which can aggressively prune low-weight trajectories, discarding potentially correct reasoning paths and degrading the genealogical diversity of the search space. To address this, we introduce Chopthin-Consensus Power Sampling (CCPS). Our method applies the Chopthin resampler to LLM decoding: rather than equalizing weights and forcing unnecessary particle duplication, it enforces an upper bound on the ratio between the largest and smallest weights and carries the unequal weights forward. This targeted intervention preserves a richer set of distinct reasoning paths, keeps the weighted SMC approximation unchanged in conditional expectation, and guarantees a lower bound on the post-resampling effective sample size (ESS). To fully exploit this enriched population, we employ a semantic-majority selection mechanism that merges token-identical final trajectories, clusters semantically equivalent answers, and returns the answer supported by the largest number of distinct trajectories. Evaluating across three open-weight models and five reasoning benchmarks, we show that Chopthin increases oracle coverage in 13 of 15 settings. Combined with semantic-majority selection, CCPS matches or exceeds the final-answer accuracy of the Power-SMC baseline in 14 of 15 settings, delivering absolute gains of up to 10.6 percentage points. These findings demonstrate that diversity-preserving resampling and diversity-aware selection are complementary mechanisms for training-free LLM reasoning. Code is available at this http URL. 

---
# Repair Before Reinforce: Context-Augmented Knowledge Graph Reasoning for Multi-Hop Question Answering 

**Authors**: Tharaka D. Fonseka, Niraj K. Jha  

**Link**: [PDF](https://arxiv.org/pdf/2609.12230)  

**Abstract**: Question-answering often requires reasoning across multiple connected facts rather than retrieving a single isolated relation. Knowledge graphs (KGs) provide a structured way to represent such facts, but training large language models (LLMs) only on isolated KG head-relation-tail triples may limit their ability to learn the surrounding context needed for multi-hop reasoning. In this work, we propose a context-augmented training framework for multi-hop question-answering. Although generally applicable, we validate the framework in the context of disease-specific KGs, extracted using a reliable KG extraction framework called GraphMERT, for Gastroparesis and Diabetes. For each primary KG triple, we attach supporting triples extracted from the same source text chunk to form a context graph (CG). This creates two supervision settings: KG-grounded supervision, which uses only the target KG triple or path, and CG-grounded supervision, which uses the target KG triple or path together with supporting context triples. We train the Qwen3-14B model using supervised fine-tuning (SFT) under both settings, producing KGModel and CGModel variants. To strengthen the lower-hop factual foundation of the models, we introduce an LLM-judged, history-aware adaptive repair pipeline that identifies unresolved one-hop failures, continually fine-tunes on targeted repair examples, and removes or quarantines problematic noisy triples. This repair stage enables the models to reach 100% accuracy on the cleaned retained one-hop validation sets. Finally, we employ reinforcement learning (RL) using lower-hop question-answer items and evaluate generalization on harder 3-hop, 4-hop, and 5-hop tasks. Across both diseases, context-augmented supervision consistently improves multi-hop performance over KG-only supervision. RL initialized from repaired SFT checkpoints yields larger and more stable gains. 

---
# GAUGE: When Not to Trust LLM-as-a-Judge in User-Simulated Evaluation of Task-Oriented Agents 

**Authors**: Umesh Bodhwani, Thanh Tran, Kai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2609.12191)  

**Abstract**: Comparing and selecting task-oriented LLM agents increasingly relies on a low-cost offline evaluation gate: persona-driven LLM user-simulators converse with each candidate, an LLM-as-a-judge scores the transcripts, and the higher-scoring agent is promoted. We introduce GAUGE, a reusable offline protocol that measures whether this gate's ranking matches a grounded verifiable reward across 25 agents from six providers on the $\tau^2$-bench and SimulatorArena benchmarks, separating two kinds of evaluation validity that release practices conflate: ranking validity and construct validity. First, a satisfaction-success gap: satisfaction carries essentially no information about task success, as conversations rated satisfied by our blind panel are decorrelated from actual success, with 57.5% of them failing the customer's task, a pattern consistent across five rater populations, both benchmarks, and every subjective dimension we rated. Second, while the gate's ranking is robust across the broad capability span, it loses resolution among the near-equal strong agents: this decision-disagreement rate jumps from $<$1% on wide-reward pairs to 31% on close pairs. The gate is thus human-validated yet mis-anchored. As a remedy, we propose a calibrate-then-trust cadence in which a judge-free completion bit is a zero-cost tripwire for truncation regressions. 

---
# Population-level measures of perceived food access reveal barriers beyond geographic proximity 

**Authors**: Teresa Groton, Benjamin rachunok  

**Link**: [PDF](https://arxiv.org/pdf/2609.12132)  

**Abstract**: Food access is multidimensional, but population-level measurement still relies heavily on geography because perceived dimensions of access are difficult to measure at scale. Here, we use 25,125 Google Maps reviews from 49 grocery stores in Raleigh, North Carolina, to measure five dimensions of food access: availability, accessibility, affordability, accommodation, and acceptability. We identify review topics with unsupervised topic modeling and assign them to access dimensions using zero-shot classification, with 85.4% agreement against manual coding. The resulting store-level measures capture distinct aspects of food access and reveal barriers that geographic proximity alone does not capture. Comparisons between nearby stores in the same chain further show that identical store policies can be perceived very differently across locations, consistent with food access reflecting the fit between residents and their food environment. Perceived food access also follows systematic socioeconomic and demographic patterns that broadly parallel, but do not replicate, those observed for geographic access. These results show that online grocery reviews can provide a scalable complement to geographic measures of food access. 

---
# Local Edits, Global Ripples: Replay-Informed Policy Adaptation for Workflow Synthesis 

**Authors**: Manqing Mao, Hong Wang, Samson Koelle, Jie Yuan, Zhuoer Wang, James Feng, Yanjun Lin, Daniel Edmiston, Nikki Lijing Kuang, Zhecheng Sheng, Wei Niu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12127)  

**Abstract**: Prompt-policy editing offers a practical way to improve agents that synthesize executable workflows without updating the underlying model. However, persistent prompt editing has two coupled properties. First, edit locality does not imply effect locality: an edit confined to one policy segment can ripple through downstream execution, altering behavior beyond the edited segment. Second, edit effects are composition-sensitive: edits that work in isolation can interfere after composition, causing one or both to lose their benefit or become harmful. Persistent adaptation must therefore support two distinct decisions: identifying where the policy should change from execution feedback, and determining whether the resulting edit remains safe to persist after composition.
To address these challenges, we introduce RIPPLE (Replay-Informed Persistent Policy Localization and Editing), which separates where an edit is made from whether it remains safe after composition. It diagnoses failed trajectories, maps each actionable failure to a predefined policy segment, and restricts the correction to that part of the policy. RIPPLE then evaluates candidates against the same iteration-start policy to compare their isolated gains, before replaying promising edits after previously accepted updates to expose downstream effects and interactions. Only edits that remain safe under composition are retained.
We evaluate RIPPLE on Flow-HO, a synthetic held-out benchmark for executable workflow synthesis. RIPPLE improves validation success by up to 23.1% and yields positive gains on two additional frozen language-model backbones, while maintaining edit efficiency and low execution cost. Targeted interaction analysis further demonstrates both properties: a segment-local tool-use edit changes downstream resource resolution and validation, while an edit beneficial in isolation becomes harmful after composition. 

---
# Quantifying Consonant Contributions to Word Intelligibility via Acoustic Masking 

**Authors**: Eunjung Yeo, Kwanghee Choi, Krupaben Kothadia, Visar Berisha, Julie M. Liss, David R. Mortensen, David Harwath  

**Link**: [PDF](https://arxiv.org/pdf/2609.12122)  

**Abstract**: Consonants contribute unequally to whether a word is understood. Given the limited time available for therapy, ranking consonants by contribution to intelligibility helps prioritize intervention targets in motor speech disorders. However, measuring this contribution relies on perceptual studies that are difficult to scale. This paper presents a scalable method that measures consonant contribution using acoustic masking. We silence one consonant at a time in an isolated word and test whether an automatic speech recognition (ASR) model still recognizes the word. We define a consonant's contribution score as the proportion of its masked instances for which the word becomes misrecognized, which we refer to as the mask-induced misrecognition rate (MMR). We validate MMR against two linguistic factors previously reported to correlate with consonant contribution, namely phoneme frequency and functional load. We apply this analysis across four languages, English, Spanish, German, and Czech, using three ASR architectures, MMS (encoder-only), Whisper (encoder-decoder), and Qwen3-ASR (LLM-based). Using partial Spearman correlations, we find that phoneme frequency correlates negatively with MMR while functional load correlates positively. In other words, more frequent consonants are less disruptive when masked, whereas consonants carrying more lexical contrast are more disruptive. Further cross-language analysis shows that consonant rankings are not consistent, indicating that consonant contribution is language-dependent. 

---
# The Cost of Compression: A Rate-Distortion Limit on Factual Hallucination 

**Authors**: Xi Wang, Shijia Xu, Rongfeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2609.12111)  

**Abstract**: Factual hallucination in closed-book question answering is often treated as a coverage problem: a model fails because the relevant fact is absent from its internal memory. This view misses a second source of error. Even when a fact has been observed, finite memory may force it to be stored only approximately. We study this effect through a simple coverage--compression model of factual recall. We consider an unstructured question-answering task with $N$ possible queries and $K$ possible answers. A learner observes $M$ training facts, compresses them into at most $B$ bits, and answers uniformly drawn test queries without retrieval. For a uniformly random ground-truth mapping, we prove $\mathcal{E} \geq \frac{M}{N}\delta^\star\!\left(\frac{B}{M}\right) + \left(1-\frac{M}{N}\right)\left(1-\frac{1}{K}\right)$, where $\delta^\star(r)$ is the inverse rate-distortion function of a uniform $K$-ary source under zero-one loss. The two terms separate compression distortion on observed facts from missing coverage on unobserved facts. The bound gives a compact way to reason about selective memory, forced compression, structure, retrieval, abstention, and long-context organization. We study the predicted signatures with theory-implied simulations and controlled fact-injection probes in modern language models that vary fact load and effective trainable memory. The result is not a complete theory of hallucination, but an information-theoretic account of a separable failure mode: lossy recall of observed facts under finite memory. 

---
# Extracting Dataset Mentions in Forced Displacement and FCV Documents: A Weakly Supervised Framework with LLM-Based Label Refinement 

**Authors**: Rafael Macalaba, Aivin V. Solatorio, Patrick Michael Brock, Olivier Dupriez  

**Link**: [PDF](https://arxiv.org/pdf/2609.12107)  

**Abstract**: Development and humanitarian organizations produce and support surveys, administrative registries, and other data resources to inform research, policy, and operations, yet systematically identifying where these datasets are referenced remains difficult. Such references are dispersed across research papers, project documents, humanitarian reports, and other unstructured text, limiting both the ability to trace data use and to identify potential gaps in data availability or dissemination. We present a weakly supervised framework for adapting dataset extraction to forced displacement and Fragile, Conflict, and Violence (FCV) documents without first constructing a large manually labeled training corpus. A lightweight model trained on general research literature generates candidate dataset mentions from unlabeled domain documents, which a frontier large language model (LLM) reviews in context, validating or rejecting candidates and correcting their extraction boundaries. The resulting annotations are supplemented with targeted synthetic and contrastive examples and used to fine-tune the lightweight model for large-scale extraction. We evaluate the resulting model on an independent gold-standard benchmark of 1,706 text passages spanning research, humanitarian, and operational documents. Across the full benchmark, the model achieves 74.1\% precision and 70.5\% recall at the mention level; among passages containing dataset references, precision reaches 89.5\%. At the passage level, the model achieves 88.2\% accuracy and 88.6\% specificity in distinguishing passages with dataset references from those without them. These results demonstrate a practical approach for constructing domain-specific supervision when labeled data are limited, and provide a technical foundation for larger-scale analysis of data use and potential gaps in the displacement data landscape. 

---
# What Counts as a Mistake? Annotating Recitation Events in Quran Memorization Transcripts 

**Authors**: Mohamad Al Mdfaa, Nursultan Askarbekuly, Ahmed Helaly, Ubai Sandouk, Manuel Mazzara  

**Link**: [PDF](https://arxiv.org/pdf/2609.12085)  

**Abstract**: Checking Quran recitation from an ASR transcript requires distinguishing unresolved mistakes from repetitions, repairs, opening formulas and accepted spelling differences. We report a completed human annotation of 100 production recording cases: 348 scored units and 162 localized events across ten combined labels. An executable evaluator scores labels and word positions together. A plain diff reaches label-aware F1 0.525 and localization F1 0.826; adapted production cleaner/alignment components reach 0.518 and 0.786, with exact-span F1 0.505 for both. Correcting the adapter's word coordinates recovers all five annotated repetition events, showing why annotation interfaces must be checked before interpreting baseline failures. In a preliminary pilot, eight single 20-minute runs across three coding agents and eight models span label-aware F1 0.143 to 0.892: seven land far above every baseline, and one collapses below the naive diff from a missing normalization step. Across the six, 970 of 972 gold-event instances draw an overlapping prediction, so what remains is not detection but convention: span extent, and the labels whose boundary is stipulated by adjudication rather than visible in the text. Seven of 162 events defeat all six same-day runs, five of them one orthographic rule, and the strongest run still misses the same ones. No run annotated before building, so the pilot measures the algorithm half of the task only. 

---
# R2VC: Modular Fact-Checking with Retrieval, Verification, and Confidence Calibration 

**Authors**: Dhruv Dixit, Paritosh Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2609.11955)  

**Abstract**: Large language models are increasingly used for automated fact checking, but end-to-end prompting often entangles evidence retrieval, reasoning, and uncertainty estimation, making failures difficult to diagnose and confidence difficult to trust. We present R2VC, a modular retrieve, reason, verify, calibrate architecture for evidence-grounded fact checking with citations and abstention. R2VC combines hybrid sparse+dense retrieval over Wikipedia, a supervised fine-tuned and DPO-aligned generator that produces diverse structured verdict candidates, an external NLI cross-encoder for evidence-based candidate selection, and a lightweight sequence-level calibrator for confidence estimation and selective abstention. On FEVER, an 8B backbone with R2VC achieves 13.74% higher accuracy than baseline. Ablation studies show that verifier-based candidate selection and confidence calibration are the largest contributors to performance. Removing candidate selection drops FEVER accuracy to 76.24%, while removing calibration nearly doubles the Brier score to 0.161. A manual analysis of 250 errors further shows that retrieval failures, especially wrong-entity evidence, remain the dominant bottleneck. Together, these results show that modular fact-checking pipelines can substantially improve both predictive accuracy and confidence reliability in open-domain verification. 

---
# MP-Bench: Evaluating Voice Agents as a Multiparty Conversation Participant 

**Authors**: Yi-Jen Shih, Shih-Yun Shan Kuan, Guan-Ting Lin, Kai-Wei Chang, Siddhant Arora, Shu-wen Yang, Abdelrahman Mohamed, Shinji Watanabe, Hung-yi Lee, David Harwath  

**Link**: [PDF](https://arxiv.org/pdf/2609.13076)  

**Abstract**: Conversational voice agents have advanced significantly, offering increasingly natural human-machine interactions through both cascaded and end-to-end architectures. However, while recent benchmarks extensively evaluate dyadic interactions and passive audio comprehension, they largely overlook a prevalent real-world scenario: multi-party conversations. Evaluating agents in these settings is fundamentally more challenging than in dyadic interactions due to the exponentially greater conversational complexity. For voice agents to integrate seamlessly into human group dynamics, they must not only generate contextually appropriate responses but also demonstrate a nuanced understanding of open turn-taking. To address this gap, we introduce Multiparty Bench (MP-Bench), the first benchmark specifically designed to objectively evaluate conversational speech systems as active participants within multi-party contexts. MP-Bench assesses agent behavior along two primary dimensions: turn-taking awareness and response appropriateness. Additionally, we incorporate comprehension-based question-answering tasks as a complementary evaluation. By benchmarking 12 voice agents, we find that real-time voice agents stay at or below 22% on multiparty comprehension and remain near chance on multiparty turn-taking, exposing an open challenge for real-time voice agents under multiparty scenario. 

---
# MAxBench: A Multinomial Concept Recovery Benchmark 

**Authors**: Divya Appapogu, Freya Behrens, Yonatan Belinkov, Aaron Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2609.13072)  

**Abstract**: Fine-grained control of language model behaviors (e.g., steering) is among the more actionable outcomes of interpretability research. For binary concepts such as refusal, a single direction in activation space often suffices for steering. However, many concepts are not binary: Animals and Countries contain many subcategories, each with multiple instances. For these concepts, the search space over possible representation geometries is far larger than for binary concepts; it is thus not clear what geometries are most appropriate, nor what methods are most effective at recovering them. In this work, we introduce MAxBench, a geometry-agnostic evaluation framework for multinomial concept representations based on sampling from the recovered concept representation. We use MAxBench to compare 10 localization methods (covering 5 geometry types) across 6 concepts and 4 models. Using this framework, we find that (i) affine subspaces steer more reliably and have greater recall than rank-one or linear subspaces; (ii) much of this advantage is due to better non-zero offsets rather than the choice of bases; (iii) manifold steering is competitive with the best methods when applicable; and (iv) no method consistently outperforms prompting, in alignment with prior findings on binary concepts. These findings underscore the importance of expanding the scope of interpretability research and meta-evaluation to concepts with more varied structure. 

---
# Investigating Temporal Motion Features for Pose-to-Text Indian Sign Language Translation 

**Authors**: Manav Dhamecha, Praveen Kumar Chandaliya, Pruthwik Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2609.12993)  

**Abstract**: We investigate the effect of pretrained T5 model scale and explicit motion features on pose-to-text Indian Sign Language Translation (SLT) for the WSLP 2026 Shared Task. Pose sequences are projected into the embedding space of T5 through a lightweight pose encoder, with the complete model fine-tuned to generate English text. The shared task data used for this work consists of a test set with 5,334 examples and a validation set with 5,257 examples. We compare T5-small, T5-base, and T5-large, and additionally introduce a motion-augmented variant, T5-small + Motion, that adds explicit frame-to-frame pose differences to the input representation. T5-small achieves the best BLEU and ROUGE scores among the spatial-only models, while T5-large obtains the highest chrF score. Augmenting T5-small with motion features yields the largest single improvement observed in our study, substantially improving BLEU over the spatial-only baseline and making it the strongest model overall on this metric. Our submitted system ranked 5th on the official WSLP 2026 SLT testing leaderboard. The source code and trained models are publicly available on GitHub and HuggingFace. 

---
# PA-CDM: Position-Aware Character Detection Matching for Evaluating Handwritten Mathematical Expression Recognition 

**Authors**: Shiliang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2609.12917)  

**Abstract**: Handwritten mathematical expression recognition (HMER) is conventionally scored by exact-match rates and string-similarity metrics that are blind to where an error occurs: two predictions with identical token-error counts receive identical scores whether they misplace a subscript or swap the operands of a fraction. Render-based character detection matching (CDM) aligns glyphs robustly but remains position-blind---on controlled fraction-operand swaps it scores 0.8595 where position-aware scoring yields 0.6253. Tree-edit metrics exhibit a complementary blind spot: rewrites outside the parser's normalization coverage are penalized as structural errors (0.8552 where render-based metrics score 1.0). We propose PA-CDM, a position-aware metric that couples character detection matching with position-forest encoding and divergence-level weighting; StructPerturb v2.0, a frozen benchmark of 1,340 controlled perturbation pairs across 15 type--intensity cells; and a cross-metric consistency protocol combining a sensitivity matrix, a human study, and LLM-judge calibration. In a six-annotator study, PA-CDM attains the highest correlation with human judgments among seven automatic metrics (Spearman rho=0.9535, n=990). A frontier LLM judge correlates slightly higher (rho=0.9613) but is costly, nondeterministic, and API-dependent; PA-CDM approaches it at zero marginal cost with deterministic, diagnosable behavior. 

---
# What Drives Recovery in Agentic Text-to-Cypher? LAST-CQ: An LLM Agent Self-Refinement Framework 

**Authors**: Ioannis Prokopiou, Athanasios Aidinis, Panagiotis-Christos Kyrmpatsos, Pantelis Vikatos  

**Link**: [PDF](https://arxiv.org/pdf/2609.12746)  

**Abstract**: Agentic pipelines for structured-query generation are rapidly expanding, but it is unclear which part of the loop produces the gain. We use LAST-CQ -- a five-agent, training-free, execution-grounded Text-to-Cypher framework -- as an instrumented testbed, running three counterfactuals over 2,471 live-database queries and six backbones spanning three vendor scale tiers. Removing correction is worth between 3.1% aggregate execution-BLEU against the single-pass system and 12.3% against a no-refinement counterfactual (up to 80.7% for the weakest backbone). Replacing schema-grounded, LLM-synthesised feedback with raw database error strings costs almost nothing (20.9% vs. 19.9% naive exact match; <0.2% end-to-end; equivalent within $\pm 0.075$ set-F1 by two one-sided tests). Spending the same call budget on parallel sampling degrades quality by 10-11%. What works is detecting failure and routing it to a retry, not the feedback sophistication or number of samples. LAST-CQ itself recovers 91.7% of queries that fail under single-pass generation, while a query that succeeds first time still costs exactly one LLM call. We also show that n-gram overlap on serialised results is not a bound in either direction: it over-scores against set equivalence on 65.9% of results while under-scoring against judged semantics. Finally, we calibrate our LLM judge against blind human labels and find it optimistic by 9 points. 

---
# Residual Vector-based Reconstruction as Long-Context Recall Regardless of Context Window Size 

**Authors**: MyungHoon Ryu, XinYu Piao, Jong-Kook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2609.12686)  

**Abstract**: Large language models (LLMs) process long contexts, including long documents and lengthy conversations, but face token-level memory usage that increases proportionally to input length. Although model optimization and lossy prompt compression are widely used, these methods still fail to solve the long-context recall problem beyond pretrained and size-constrained context windows. This paper proposes a long-context recall method that maintains near-constant GPU memory usage as context length increases, without additional training. The main idea is to reconstruct facts using parameter activations in the LLM's feed-forward layers, which store residual vectors representing facts from the source document. Utilizing residual vectors allows the LLM to deterministically reconstruct query relevant facts without referencing the original document, preserving high fidelity and reducing memory usage without fine-tuning weights. Experimental results show that the proposed method enables answering single-fact questions in two-million-token story contexts where previous methods fail. 

---
# SteerDuplex: Steerable Duplex Speech Dialogue Models 

**Authors**: Utkarsh Tyagi, Ramaneswaran Selvakumar, Advait Gosai, Sonal Kumar, Nikhil Barhate, Isabell Sagar, Steven Li, Miheer Bavare, Daniel Quigley, Fabiola Tapia Carrillo, Jose M Patron E, Diego Macías Gutiérrez, Paul Song, Ramani Duraiswami, Dinesh Manocha, Yunzhong He  

**Link**: [PDF](https://arxiv.org/pdf/2609.12623)  

**Abstract**: Full-duplex spoken dialogue models support low-latency turn taking, interruption handling, and backchanneling, yet a key capability remains underexplored: steerability, the ability to reliably shift conversational behavior along attributes such as tone, persona, speaking rate, and voice style in response to user instructions. We introduce a taxonomy of text- and audio-based steerability that identifies substantial gaps in current full-duplex models. To address this gap, we introduce SteerDuplex, a Moshi-based full-duplex speech model fine-tuned on natural conversations and synthetic dialogues targeting instruction following, vocal delivery, reasoning, and duplex interaction. We further apply two-stage reinforcement learning (RL) with hybrid rewards, combining verifiable interaction checks and judge-based semantic feedback to improve timing and response continuity. To evaluate full-duplex spoken steerability, we introduce SteerBench, a benchmark with 390 spoken prompts and 1,067 human-authored binary audio and text rubrics spanning tone, persona, style/accent, and speed/length. On SteerBench, supervised training improves audio-steering average pass rate by 44.5 percentage points over the strongest evaluated open baseline. On Audio MultiChallenge, task average pass rate improves by 7 points over its strongest evaluated open baseline. RL further raises source-clean interruption response from 72.5% to 82.5% and reduces synthetic pause barge-in from 26.5% to 9%. Steering and aggregate task scores remain comparable or higher, while reward probes reveal reward hacking through incomplete responses. Our model and benchmark support systematic research on spoken steerability, with reward analysis showing why timing gains must be evaluated alongside response completeness. 

---
# Earth-Agent-Pro: Towards Real-World Full-Chain Earth Observation with Agents 

**Authors**: Zhutao Lv, Chenhao Dang, Yi Feng, Yanpei Gong, Xiaolei Wang, Junyan Ye, Conghui He, Weijia Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.12533)  

**Abstract**: Real-world Earth observation (EO) agents must translate high-level scientific questions into executable workflows to acquire observations, prepare data, perform domain computations, and derive conclusions from runtime evidence. Existing EO agents typically start from supplied observations, while benchmarks typically provide prepared inputs or candidate answers, leaving full-chain open-world EO execution largely untested. We present Earth-Agent-Pro, an execution-adaptive Plan-and-Execute framework using expert-authored skills to constrain planning and runtime tool use. Workflow-centered structured memory records planned steps, accepted evidence, and their dependencies, enabling repair of only the affected workflow suffix when runtime evidence invalidates a step. Separate large language model adapters use sequence-level supervised fine-tuning for planner workflow composition and node-level group relative policy optimization with locally verifiable rewards for executor tool-argument grounding. Earth-Bench-Pro instantiates 248 expert-curated task cores as 744 questions under three matched regimes. Its 248 Open-World Execution questions span RGB imagery, spectral observations, and remote sensing products, pairing high-level requests with runtime data requirements, executable trajectories, and open-ended answers grounded in execution evidence. With a shared GPT-5 backbone, Earth-Agent-Pro achieves 66.13% LLM-as-Judge accuracy, exceeding ReAct by 20.95 points in this metric and 24.44 points in Tools-In-Order. Joint adapter tuning raises Qwen3.5-9B LLM-as-Judge accuracy from 38.31% to 50.00%, an 11.69-point gain over the untuned configuration. Planning-only evaluation and execution with the reference workflow show that the adapters improve workflow composition and argument grounding, respectively. Code and datasets will be released soon. 

---
# Information Specialization and Constrained Synthesis in Multi-Agent LLM Forecasting: A Prospective Live-Study of the 2026 FIFA World Cup 

**Authors**: Julian Varghese, Lucas Bickmann, Sarah Sandmann  

**Link**: [PDF](https://arxiv.org/pdf/2609.12495)  

**Abstract**: Large language models are being organized into multi-agent systems with specialized roles, but whether such specialization produces distinct forecasts and whether subsequent synthesis improves utility remains unclear. In this study, we carried out a live, prospective evaluation over the final 56 matches of the information-dense 2026 FIFA World Cup, keeping a frontier foundation model constant while assigning two primary forecasting agents contrasting specialist roles: a quantitative specialist focusing on structured performance statistics and a news specialist focusing on current injuries, tactics and information from press conferences. Their forecasts were then reviewed by a separate critic before being combined by a meta-agent, resulting in a sequential four-agent model. Forecasts from the betting market served as an external benchmark. The news specialist obtained the highest mean probability-weighted Top-3 utility and matched the betting market in Top-3 exact-score hits. Nevertheless, the two specialist forecasters agreed on at least two of the three scorelines in 50 out of 56 matches, and the meta-agent never generated more than one scoreline outside the specialists' forecast set. These findings show that rapidly changing, unstructured information can provide a valuable forecasting signal alongside structured statistics, whereas adding critic and meta-agent stages does not necessarily create complementary information or improve on the strongest specialist. 

---
# Confidence-Gated Transductive Test Generation for Code Reranking 

**Authors**: Sungjae Lee, Youngsik Yoon, Seockbean Song, Siwei Wang, Wei Chen, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2609.12489)  

**Abstract**: Test case synthesis is crucial for evaluating and ranking programs generated by large language models (LLMs). However, constructing high-quality test cases remains challenging because reliable expected outputs are often difficult to obtain. We propose Confidence-Gated Transductive Test Generation (CoTT), which first uses an efficient inductive procedure and invokes transductive generation only when inductive confidence is low. This adaptive design improves output reliability while allocating extra computation only when needed. On code reranking benchmarks, CoTT outperforms prior baselines across the reported metrics while reducing cost relative to applying transductive generation to every input. These results show that confidence-based allocation of test-time computation provides a favorable efficiency-effectiveness trade-off with a single efficient LLM. 

---
# Do LLMs Trust the Accuser or the Accusation? Measuring Belief Shifts in Werewolf 

**Authors**: Yu-Yu Yang, Ti-Rong Wu, Hung Guei, Hsing-Yu Chen, I-Chen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2609.12446)  

**Abstract**: Social-deduction games such as Werewolf are increasingly used to evaluate LLM agents, but existing evaluations often rely on final game outcomes. We propose a belief-shift evaluation benchmark in Werewolf for analyzing communication skills through belief updating. Using LLM-played games, we annotate suspicion and accusation messages and measure how an observing village-side model's beliefs change after each message. We evaluate 40 open-weight LLM configurations on 1,224 annotated messages. Our results show that larger models better distinguish true wolves from villagers based on game history, but accusations still strongly influence their beliefs. Models become more suspicious of the accused target and less suspicious of the accuser, especially when the accuser is trusted, even if the accuser is wolf-aligned. Larger models better resist accusations from accusers they already distrust. Overall, our findings suggest that current open-weight LLMs up to 120B parameters still struggle to integrate accusation content with source trust in strategic communication. Our benchmark and code are available at this https URL. 

---
# Diverse Minds, Divided Networks? Personality Composition, Polarization, and Collective Intelligence in LLM-Based Social Simulations 

**Authors**: Raad Bin Tareaf  

**Link**: [PDF](https://arxiv.org/pdf/2609.12444)  

**Abstract**: Simulated societies of large language model agents are used to study online polarization, and separately to study collective intelligence, but the two are rarely measured in the same system. It is therefore difficult to say whether a society's personality composition shapes both, or whether reducing polarization costs collective competence. We present TraitMix, an experimental design in which the Big Five composition of a simulated social network, both trait levels and trait heterogeneity, is a controlled experimental variable, and in which polarization and collective performance are measured in the same runs. Across 991 simulations of hundred-agent societies, spanning six contested topics and six language models, trait heterogeneity has the largest measured effects, acting in opposite directions on two faces of polarization: varied societies hold more dispersed opinions while being less segregated into camps, so homogeneous societies are not moderate but consensual echo chambers. Trait effects are not additive, as Agreeableness determines the sign of Openness, an interaction that replicates across models although the primary model's estimate is influence-driven. Contrary to the trade-off the study was designed to measure, no polarization measure predicts poorer collective performance, and cross-cutting interaction is the only one of four whose association with collective accuracy survives partialling on the aggregation identity. We report ablations removing two potential measurement circularities, an induction gate applied to every model, and the measures that failed them. 

---
# Beyond ID Embeddings: Process-Grounded Language Modeling for Cognitive Diagnosis 

**Authors**: Minghang Liu, Yuanzhuo Wang, Qiang Qiu, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2609.12403)  

**Abstract**: Cognitive Diagnosis Models (CDMs) play a pivotal role in personalized online learning. Traditional CDMs rely on discrete, ID-based embeddings to represent students, exercises, and concepts. This paradigm diverges from the nature of learner cognition, where knowledge is not stored and retrieved as isolated symbols. As a result, CDMs suffer from semantic limitations when new exercises or concepts appear. In this paper, we propose a Process-aware Language Cognitive Diagnosis (PLCD) framework that uses language-derived structures as cognitive priors and response records to calibrate student posterior states. PLCD leverages large language models (LLMs) to construct concept schemas and cognitive process graphs, and uses target-conditioned semantic memory to retrieve historical responses that are relevant to each target exercise. A process-grounded Language-to-Cognition Mapper with DA-MoE experts and process-level contrastive learning then maps the textual evidence into a unified cognitive space. Experimental results show that PLCD not only outperforms traditional baselines in predicting student performance but also exhibits strong cognitive transfer capabilities. These results connect the computational power of LLMs with the psychometric goal of measuring latent knowledge states, suggesting that structured language priors calibrated by response records can improve cold-start robustness and cognitive grounding. 

---
# Can LLMs in Draft-Verify-Revise Pipelines Resolve Deictic Ambiguity? 

**Authors**: Obinna I. Ekekezie  

**Link**: [PDF](https://arxiv.org/pdf/2609.12162)  

**Abstract**: Draft-verify-revise is a common LLM orchestration pattern for scaling inference-time compute. One LLM drafts, a second critiques the draft and provides feedback, and a third uses that feedback to revise the draft into the final output. As context cascades between stages, LLMs at different stages can resolve a context-dependent expression such as "previous" differently. When that happens, the expression undergoes a deictic shift, a change in what it refers to. This phenomenon was studied with a synthetic dataset of 10 base examples, each rendered in three conditions. Holding the shared components constant, the conditions varied whether the draft stage LLM (the assistant) or the verify stage LLM (the grader) resolved the expression correctly, and how much independent reasoning the revise stage LLM (the meta-evaluator) needed to determine which reading was correct. Six models from three providers were tested across 21 reasoning effort configurations using e-values for sequential testing, in a primary experiment and an ablation experiment that removed error classification labels from the grader's feedback. A separate LLM analyzed the meta-evaluator's stated rationale for each wrong verdict. Balanced accuracy (the unweighted mean of sensitivity and specificity) ranged from 0.156, below chance, to near-perfect. GPT-5.2 rose from 0.156 without reasoning to 0.942 at its highest reasoning effort level, while Gemini 3 Pro stayed above 0.94 at every level. Gemini 3 Pro at low reasoning effort outscored GPT-5.2 at xhigh reasoning effort for roughly 5% of the cost per trial. When the meta-evaluator erred, it tended to rely on surface cues rather than operational reasoning. Context engineers implementing draft-verify-revise pipelines should be wary of deictic shifts and make the intended referent explicit at each stage. 

---
# Creating an Atomic User Model for Personality-Aware Large Language Model Interaction 

**Authors**: B. Sankar, Deepthika S, Pawni Yadav, Amogh A S  

**Link**: [PDF](https://arxiv.org/pdf/2609.12086)  

**Abstract**: Assistants built on large language models are expected to write as their user would, and the dominant approach is single-channel: preferences summarised from conversation history and reinserted into context. This inverts the order of inference. Preferences are the task-dependent surface of a comparatively stable personality structure, so a system storing only preferences relearns the person whenever the task changes. First, we characterise personality seepage, where a prompt's linguistic surface carries a personality fingerprint the assistant mirrors without access to the personality behind it. Second, we propose the Atomic User Model (AUM), a human-readable representation organising a person as a stable identity nucleus with four interpretable shells (psychological, cognitive and experiential, behavioural, and social), plus cross-shell entries recording internal conflict and authenticity. Third, we treat AUM as a retrieval index over a person rather than a prompt prefix, with a pipeline where a task classifier, component-selection function and budgeted retriever return a small payload of fields at generation time. Fourth, we evaluate it with sixteen language-model-simulated participants, six style-sensitive tasks and three seeds, plus a synthetic scaling study of the retriever. Retrieving eight fields matched the style fidelity of the full user model on 23% of the context (211 tokens against 915), improved on flat preference notes by 0.24 points on a five-point scale (p < 0.001, dz = 0.50), and raised forced-choice identification of the participant's own voice from 14.9% to 42.7% (25% chance). Four pre-registered controls returned null, locating the effect in the representation rather than the search over it. The benefit is largest for participants the un-personalised assistant reproduces worst (rho = -0.61, p = 0.013): personalisation is worth most to those the default serves least. 

---
# Is Bash All You Need? An Empirical Study of Tool Interfaces for Enterprise Digital Worker Agents 

**Authors**: Hazel Mak, Susheel Suresh, Sahil Bhatnagar, Barry Wang, Chhaya Methani, Alejandro Gutierrez Munoz  

**Link**: [PDF](https://arxiv.org/pdf/2609.11999)  

**Abstract**: In this study, we examine whether a general shell can outperform specialized tools on enterprise tasks. Shell-based agents have shown strong results in coding, but enterprise work also involves moving between applications and services, coordinating with coworkers, and performing professional analysis. We compare five tool interfaces on TheAgentCompany and APEX-Agents using Opus-4.8 and GPT-5.5: typed tools, typed tools plus bash, bash alone, bash with persistent agent-synthesized tools, and programmatic tool calling (PTC), which runs programs whose actions are restricted to a typed tool catalog. Bash alone outperforms typed tools on both benchmarks, improving score by 21.8-24.5 pp on TheAgentCompany and 4.8-7.4 pp on APEX-Agents while using 19-72% fewer total tokens. Adding typed tools or persistent tool synthesis to bash produces no detectable pooled score gain. PTC uses fewer tokens than direct typed calls with broadly similar task performance, but generally underperforms bash alone in both quality and cost efficiency. For enterprise practitioners, these results favor bash alone when arbitrary execution can be isolated and PTC when security or compliance policies require a fixed tool catalog. 

---
# Harness or Model? Isolating the Harness Effect in Agentic Coding with a Contamination-Controlled Private Suite 

**Authors**: Mohsen Arjmandi  

**Link**: [PDF](https://arxiv.org/pdf/2609.11987)  

**Abstract**: An agentic coding system couples a language model to a harness: the tools, prompts and control flow that turn a chat model into an autonomous software engineer. Vendors ship harnesses tuned to their own models, and practitioners assume the vendor-native pairing solves more tasks. We measure that assumption with paired same-model contrasts on a private, contamination-controlled suite of 256 repository and post-cutoff contest tasks. The same 80 tasks ran under claude-agent-sdk and under deepagents on claude-opus-4-8, and under the openai-codex SDK and deepagents on gpt-5.5, with gemini-3.5-flash and deepseek-v3.2 as side cells. 792 of 800 planned runs were graded by an isolated oracle. Neither contrast resolves an average advantage for either harness: -1.25 pp for Opus 4.8 (48.8% vs 50.0%, task-bootstrap 95% CI [-10.0, +7.5]) and +1.25 pp for GPT-5.5 (55.6% vs 54.4%, CI [-4.4, +6.9]). The Opus average combines opposite strata: the native harness trails by 9.0 pp on the 61 repository tasks and leads by 23.7 pp on the 19 contest tasks (label-permutation p = 0.003). The partition was chosen after seeing the data and needs a designed replication. Correctness and completion also separate: 22 of 81 runs cancelled at the wall-clock ceiling had produced a passing patch. Re-priced from raw per-turn usage at frozen list prices, the neutral harness cost 1.3 to 1.6 times as much per solved task on Opus 4.8 and 1.2 times on GPT-5.5. These are observed-usage estimates. On the Anthropic account 58 runs left no usage record, and allocating that spend to either cell would move the Opus ratio between 0.7 and 2.3, so the billed ordering is unresolved. This revision corrects an August 2026 manuscript whose cost figures rested on a usage-semantics defect in our own telemetry (Section 5.1). We release the orchestrator, grading oracle, reanalysis code and derived aggregates. The tasks stay private. 

---
# Cortex: Content Analysis Support Software, a Resource for Qualitative Research 

**Authors**: Ana Julia da Silva Soares, Rafael Coimbra Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2609.11970)  

**Abstract**: Qualitative research is widely used in the human and social sciences, characterized by a deep understanding of phenomena through the interpretation of meanings and contexts. Among qualitative data analysis methods, content analysis stands out as a consolidated technique, which allows for the systematic description and interpretation of textual contents. However, as data volume increases, the time required for organization, reading, and categorization becomes a significant challenge, potentially delaying research development. Therefore, this work aimed to develop a web application to support content analysis, based on Bardin's methodology, targeted at academic researchers. The methodology adopted a mixed approach, combining bibliographic research on content analysis with semi-structured interviews with four experienced researchers, aiming to identify real needs and requirements. Based on these inputs, the Cortex software was developed to assist the researcher in the pre-analysis and material exploration stages, generating suggestions for indices, indicators, and categories with full traceability to original documents. The results demonstrate that Cortex is capable of guiding the researcher through the entire methodological workflow, from corpus configuration to results exportation, acting as a methodological collaborator without replacing the researcher's interpretative autonomy. 

---
# Space as an Interventional Invariant: Cross-Modal Predictive Geometry for Stratified Cities and Em-Spaced Intelligence 

**Authors**: Tao Yang, Xuhui Lin, Kunyao Li, Haijiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2609.11959)  

**Abstract**: Space is a foundational concept across mathematics, physics, spatial cognition, urban science, and embodied intelligence, yet these fields often treat spatial structure either as a shared geometric container or as a collection of disconnected representations. Such approaches struggle to explain how heterogeneous sensory and urban processes can jointly reveal a common spatial structure, particularly when different modalities do not share the same metric or representation. This paper addresses this gap by defining space as an interventional invariant: the minimal relational structure that preserves local compatibility and the conditional laws of future observations under admissible actions. We develop a cross-modal predictive geometry that integrates local state spaces, modality-specific observation maps, an action groupoid, and a canonical predictive-state quotient, with explicit causal conditions for identifying interventional rather than merely observational structure. The key theoretical result shows that, under joint point separation, equivariance, and interventional faithfulness, the latent space is identifiable up to the centraliser of the intervention group, thereby reducing representational ambiguity to residual coordinate freedom. The framework is further extended to stratified urban systems using sheaf-valued representations, allowing geometric, physical, mobility, social, and economic layers to coexist without being reduced to a single metric. Synthetic experiments under noise evaluate equivariance, predictive sufficiency, holonomy, restriction-map recovery, cross-scale consistency, and context saturation. The resulting framework provides a unified and falsifiable foundation for spatial cognition, urban science, embodied AI, and em-spaced intelligence. 

---
# PRISMA-LLM: An Empirical Reporting Framework for AI-Assisted Systematic Reviews 

**Authors**: Miguel Zabaleta, Baihan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2609.11559)  

**Abstract**: Large language models (LLMs) and AI-enabled software increasingly participate in systematic-review decisions, yet the information needed to audit these workflows is reported inconsistently. We analyze SciLitBench, a corpus of 888 review-automation papers with 14,726 annotations, to characterize changes in methods, review-stage use, evaluation and reported limitations. Automation has shifted toward LLM- and software-facing workflows, including stages that can alter the evidence base. Since 2023, 38.0% of software/product papers reported no evaluation, compared with 9.3% of LLM papers. Reporting coverage increased with LLM workflow complexity, yet 52% of positive-only LLM evaluations still reported an unmet reliability or performance requirement. From these patterns, we introduce PRISMA-LLM, an empirically grounded framework separating implementation disclosure from consequence-sensitive evaluation and limitation reporting. 

---
