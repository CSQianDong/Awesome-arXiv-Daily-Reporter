# Variable-Width Transformers 

**Authors**: Zhaofeng Wu, Oliver Sieberling, Shawn Tan, Rameswar Panda, Yury Polyanskiy, Yoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.18246)  

**Abstract**: Scaling model size, specifically depth and width, has driven significant progress in transformer-based language models. However, most architectures maintain a constant width across all layers, allocating a fixed parameter and computation budget evenly despite different layers potentially playing distinct computational roles. In this work, we empirically investigate nonuniform capacity allocation across network depth by proposing a $\times$-shaped > <former architecture. This design maintains wider early and late layers while narrowing the middle layers, utilizing a parameter-free residual resizing mechanism. Across decoder-only language models ranging from 200M to 2B parameters (dense) and 3B parameters (MoE), our > <former consistently outperforms parameter-matched uniform baselines on language modeling loss. By reducing the average layer width, this architecture also requires fewer overall FLOPs (22% reduction under fitted loss-matched scaling curves) and smaller KV cache memory and I/O cost (15% reduction). In analysis, we show that this bottleneck structure results in qualitatively different representations in residual streams. Overall, our results demonstrate that nonuniform width allocation can result in more resource-optimal scaling of language models. 

---
# ReproRepo: Scaling Reproducibility Audits with GitHub Repository Issues 

**Authors**: Shanda Li, Qiuhong Anna Wei, Jingwu Tang, Valerie Chen, Nihar B Shah, Tim Dettmers, Yiming Yang, Ameet Talwalkar  

**Link**: [PDF](https://arxiv.org/pdf/2606.18237)  

**Abstract**: Reproducing research results from papers and released code is central to scientific progress. Existing works have introduced benchmarks to evaluate whether LLM agents can assist with reproducibility, but they are difficult to scale due to their reliance on substantial manual effort for data curation and evaluation. We introduce ReproRepo, a scalable framework for reproducibility evaluation that leverages human-raised GitHub issues as naturally occurring supervision on realistic reproduction blockers. We instantiate ReproRepo on 1,149 recent machine learning papers from major conferences and evaluate four frontier model-agent configurations. Our results show that LLM agents, even without executing code, can identify many real-world reproducibility problems from paper-repository pairs: the best agent in our study, namely Codex with GPT-5.5, surfaces at least one semantically related human-reported blocker for ~90% of papers in the study. Further analysis shows that agents are particularly effective for surfacing visible failures and identifying the right semantic region, but may still be insufficient in exact localization. ReproRepo can serve as a reusable, scalable framework for future evaluations of LLM agents on real-world reproducibility auditing. Our code is released at this https URL. 

---
# Darshana Graph: A Parallel Commentary Corpus for Comparative Indian Philosophy, with Stylometric and Exploratory Graph Analyses 

**Authors**: Joy Bose  

**Link**: [PDF](https://arxiv.org/pdf/2606.18222)  

**Abstract**: We introduce Darshana Graph, a corpus of over 125,000 text records spanning classical Hindu, Buddhist, and Jain philosophical traditions, drawn from public-domain and openly licensed translations of sources including the Bhagavad Gita, Brahma Sutras, principal Upanishads, the Pali Canon, and core Jain texts. Its distinctive contribution lies in a structurally unique subset of roughly 8,500 Hindu and Jain records in which the same root verse or sutra is aligned across eighteen historical commentators representing five schools of Vedanta and other darshanas, enabling direct comparison of how independent interpretive traditions read identical source material. To our knowledge, no publicly available resource provides comparable cross-commentator alignment at this scale. We present two analyses built on this corpus. First, a transparent stylometric comparison requiring no machine learning measures argumentative style through scriptural citation density, explicit refutation rate, and sentence complexity. It finds a moderate negative correlation between citation density and refutation rate, a marked increase in refutation rate across three commentators in a related doctrinal lineage, and measurable genre-level differences within the Pali Canon itself. Second, we describe a constrained large language model pipeline that extracts typed philosophical relationships between concepts using a predefined relation vocabulary and deterministic post-hoc validation. The resulting graph surfaces cross-school disagreement patterns while also revealing important extraction limitations, including cases where an independent embedding-based analysis disagrees with the graph-derived findings. We release the full corpus, extracted relationship graph, and all source code. 

---
# Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients 

**Authors**: Byung-Kwan Lee, Ximing Lu, Shizhe Diao, Minki Kang, Saurav Muralidharan, Karan Sapra, Andrew Tao, Pavlo Molchanov, Yejin Choi, Yu-Chiang Frank Wang, Ryo Hachiuma  

**Link**: [PDF](https://arxiv.org/pdf/2606.18216)  

**Abstract**: Knowledge distillation transfers a teacher's competence to a small student but is brittle in the small-student regime: forcing the student to imitate logits from a much larger teacher concentrates it on the teacher's sharpest modes, hurting generalization on benchmark families beyond the training corpus. Reinforcement learning (RL) avoids logit imitation by training on the student's own rollouts. However, on questions where every rollout fails-yielding zero advantage and being silently discarded-injecting a stronger teacher's response into the policy gradient breaks the on-policy assumption and induces drift. We introduce Zone of Proximal Policy Optimization (ZPPO), inspired by Vygotsky's zone of proximal development, which keeps the teacher inside the prompt rather than the policy gradient. On hard questions, ZPPO constructs two reformulated prompts: a Binary Candidate-included Question (BCQ) pairs one correct teacher response with one incorrect student response as anonymized candidates the student must discriminate, and a Negative Candidate-included Question (NCQ) aggregates the student's wrong rollouts into a single prompt to surface their shared failure modes. A prompt replay buffer recirculates each hard question until it either graduates-the student's mean rollout accuracy on it reaches half- or is FIFO-evicted under finite capacity, amplifying BCQ and NCQ inside the student's current zone of proximal development. On the Qwen3.5 family at four student scales (0.8B-9B) with a 27B teacher, post-trained as vision-language models and evaluated on a 31-benchmark suite (16 VLM, 10 LLM, 5 Video), ZPPO outperforms off/on-policy distillation and GRPO, with the largest gains at the smallest scale. 

---
# Analyzing and Encoding the Al-Mawrid Arabic-English Dictionary with the ISO Language Markup Framework and TEI Lex-0 

**Authors**: Diaa Fayed, Laurent Romary  

**Link**: [PDF](https://arxiv.org/pdf/2606.18205)  

**Abstract**: This paper presents a robust methodology for the systematic digitization and encoding of the Al-Mawrid Arabic-English dictionary, transforming it from a legacy print resource into a standardized computational lexicon. Addressing a significant gap in Arabic lexical infrastructure, the study adopts a dual-standard framing that aligns the ISO Lexical Markup Framework (LMF) with the Text Encoding Initiative TEI Lex-0 guidelines. By applying an editorial view to the dictionary's macro- and microstructure, the research resolves the structural ambiguities and punctuation inconsistencies typical of 20th-century bilingual dictionaries. The methodology is grounded in an empirical analysis of the dictionary's lexical knowledge density. Drawing on a representative sample (the letter Ayn, comprising 4.6% of the total volume), the study provides scientific weight to the encoding process, demonstrating a structural parsing accuracy of 91%. Quantitative evaluation of the information extraction rules reveals high performance, with 85% precision and 98% recall for synonyms, and 88% precision for other morpho-semantic features. Beyond technical description, the paper provides a critical comparison with existing Arabic lexical resources and discusses the limitations of TEI Lex-0 when modelling specific Arabic phenomena, such as implicit "open set" semantic relations and scattered morphological cues. Furthermore, the study explores the potential for Linguistic Linked Open Data (LLOD) integration by establishing a scalable prefix-based referencing system that facilitates the resource's inclusion in the semantic web. The result is an interoperable, machine-tractable resource that provides a reproducible workflow for the retro-digitization of complex legacy bilingual lexicons within the Arabic NLP and Digital Humanities communities. 

---
# RubricsTree: Scalable and Evolving Open-Ended Evaluation of Personal Health Agents across Health Memory and Medical Skills 

**Authors**: Weizhi Zhang, Zechen Li, Hamid Palangi, Ben Graef, A. Ali Heydari, Simon A. Lee, Salman Rahman, Ray Luo, Zeinab Esmaeilpour, Erik Schenck, Chloe Zhang, Yamin Li, Menglian Zhou, Philip S. Yu, Daniel McDuff, Lindsey Sunden, Mark Malhotra, Shwetak Patel, Ahmed A. Metwally  

**Link**: [PDF](https://arxiv.org/pdf/2606.18203)  

**Abstract**: The LLM-empowered personal health agents with user health (sensor) metrics have offered a promising pathway to alleviate global disparities in healthcare access. However, large-scale clinical deployment remains constrained by an open-ended evaluation bottleneck: physician annotation is reliable but costly and unscalable, while LLM-as-a-judge evaluators are scalable but subjective, inconsistent, and sometimes clinically misaligned. We introduce RubricsTree, a scalable evaluation framework with an expert-aligned hierarchical taxonomy of over 100 atomic, clinically-verifiable Boolean rubrics, evolving from the insights of 4,000 real user queries through an iterative human-in-the-loop curation protocol with an expertise panel led by an experienced physician. A context-aware adaptive router activates only the relevant auto-weighted rubric subset per query, providing the throughput needed for scalable evaluation with expert-aligned quality. Through a systematic meta-evaluation, we show that RubricsTree (i) substantially exceeds a strong large-scale evaluation baseline in expert alignment on challenging open-ended queries; (ii) reliably penalizes contextually degraded responses; and (iii) when used as structured instructions, text feedback, or training rewards for performance optimization, yields up to ~66% relative gains on HealthBench for Gemini, GPT, and Qwen model families. RubricsTree thus provides a scalable, auditable, and evolving evaluation infrastructure required for the continuous optimization of product-level personal healthcare AI. 

---
# Learning from the Self-future: On-policy Self-distillation for dLLMs 

**Authors**: Yifu Luo, Zeyu Chen, Haoyu Wang, Xinhao Hu, Yuxuan Zhang, Zhizhou Sha, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.18195)  

**Abstract**: On-policy self-distillation (OPSD) has proven effective for post-training large language models (LLMs), yet its application to diffusion LLMs (dLLMs) remains unexplored. Existing OPSD methods are inherently autoregressive-centric. They inject privileged information via left-to-right prefix conditioning with token-level divergence supervision, a design that fundamentally conflicts with the arbitraryorder generation of dLLMs. We introduce d-OPSD, the first OPSD framework tailored for dLLMs. Our approach makes two core contributions. First, we reframe self-teacher construction by using self-generated answers as suffix conditioning, enabling the student model to learn from "self future-experience" rather than privileged prefixes. Second, we shift supervision from token-level to step-level, aligning training with the iterative denoising process of dLLMs. Experiments across four reasoning benchmarks show that d-OPSD consistently outperforms RLVR and SFT baselines with superior sample efficiency, requiring only around 10% of the optimization steps by RLVR and opening a promising pathway for dLLM posttraining. The code is available at this https URL. 

---
# Unintended Effects of Geographic Conditioning in Large Language Models 

**Authors**: Naz Col, David M. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2606.18124)  

**Abstract**: Modern conversational AI systems frequently rely on user metadata to localize responses, yet the unintended regional biases introduced by this hidden context remain poorly understood. In this work, we evaluate location leakage: the phenomenon where a model generates geographic references despite receiving a geographically neutral user prompt. Across both creative writing and open-ended Q&A prompts, even state-of-the-art LLMs systematically favor region-specific outputs when exposed to location metadata, with leakage spiking by up to 793 times above baseline (e.g., from 0.04% to 31.7% for Llama 3.1-8B, and 21.3% and 8.8% for Qwen3-8B and Claude Sonnet 4.6, respectively). Our analysis further shows a novel structural conditioning effect: replacing the injected location with the placeholder "Unknown" still elevates leakage by up to 72 times above baseline, demonstrating that the user profile frame itself, independent of any geographic content, acts as a generative conditioning signal. 

---
# HistoRAG: Embedding Historical Methodology in Retrieval-Augmented Generation Through Critical Technical Practice 

**Authors**: Noah J. Kim-Baumann, Torsten Hiltmann  

**Link**: [PDF](https://arxiv.org/pdf/2606.18103)  

**Abstract**: Retrieval-Augmented Generation (RAG) is the prevailing architecture for grounding language model outputs in external evidence, yet its dominant evaluation paradigms and default configurations remain oriented toward factual question-answering. For interpretive disciplines such as historical studies, RAG embeds assumptions that conflict with scholarly practice. We introduce HistoRAG, a framework that translates historiographical principles into concrete architectural interventions. Separated retrieval and generation decouples source discovery from interpretation, temporal windowing enforces balanced source representation across the research period as a methodological requirement of historical inquiry, and LLM-as-judge evaluation makes relevance judgments transparent and contestable. We evaluate these interventions using SPIEGELragged, applied to 102,189 articles from Der Spiegel (1950-1979). Each intervention addresses a measurable deficiency in standard RAG: era-specific vocabulary retrieves zero chunks from the 1950s when using 1970s terminology, evidence of the temporal skew that motivates windowing; vector similarity and LLM-assessed relevance correlate only weakly (Spearman rho = 0.275), motivating post-retrieval evaluation; and keyword-based and semantic retrieval surface largely disjoint source pools, motivating an architecture in which both operate as complementary retrieval layers under a shared LLM evaluation filter. We also introduce the concept of Zwischentexte (intermediate texts that function as interpretive proposals rather than findings) as a framework for responsible integration of LLM-generated text into scholarly practice. The architecture offers a model for how domain-specific epistemological commitments can be translated into RAG design decisions, and may transfer to other interpretive disciplines working with large corpora. 

---
# Security and Privacy Prompts in the Wild: What Users Ask LLMs and How LLMs Respond 

**Authors**: Hobin Kim, Xiaoyuan Wu, Omer Akgul, Lujo Bauer, Nicolas Christin  

**Link**: [PDF](https://arxiv.org/pdf/2606.18062)  

**Abstract**: Large language models (LLMs) are widely used to fulfill users' information needs; users ask LLMs about the weather, pose educational questions, and consult them for legal assistance. One particularly understudied area is digital security and privacy (S&P), where users may seek LLMs' help on how to secure their online accounts or protect their computers from cyber attacks. To the best of our knowledge, no prior study has collected or analyzed the S&P questions users ask LLMs; prior research on LLM response quality relied on expert-authored S&P misconceptions or FAQs rather than user queries. Drawing from WildChat, a dataset of 3.2M user-LLM conversations collected in the wild, our study identifies 14,727 S&P prompts and categorizes them into nine categories covering a wide range of S&P topics. From the S&P prompts, we sampled 450 and performed a thematic analysis to characterize the S&P questions users ask LLMs. Separate from the thematic analysis, we curated 270 advice-seeking S&P prompts, where users ask for recommendations, guidance, or specific S&P information. We measured LLM response quality and consistency when posing the prompt to LLMs 10 times. We found that commercial LLMs outperform open-weight models (GPT 5.5 provided "good enough" responses on 98% of prompts; Llama 4 on 47%). However, among prompts that received high-quality responses on average, commercial models sometimes produce contradictory responses across runs, risking confusing or misleading users. 

---
# ConSA: Controllable Sparsity in Hybrid Attention via Learnable Allocation 

**Authors**: Yao Chen, Yinqi Yang, Junyuan Shang, Xiangzhao Hao, Simeng Zhang, Yilong Chen, Tingwen Liu, Shuohuan Wang, Dianhai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.18056)  

**Abstract**: Hybrid architectures combining full attention (FA) and sliding-window attention (SWA) are a promising paradigm for efficient LLM inference. However, existing methods typically rely on hand-crafted rules or simple post-hoc heuristics for FA/SWA allocation and offer limited analysis of the attention behaviors underlying these designs. We propose Controllable Sparsity in Hybrid Attention (ConSA), a framework that learns optimal FA/SWA assignment under a user-specified sparsity target. ConSA employs L0 regularization to learn binary masks selecting between FA and SWA for each attention unit, while an augmented Lagrangian constraint enforces the target sparsity at either layer or KV-head granularity. We evaluate ConSA on two LLMs at the 0.6B and 1.7B scales. Learned allocations consistently outperform rule-based baselines, with KV-head-wise allocation yielding clear gains over layer-wise allocation. The learned patterns place SWA in the bottom layers and concentrate FA into contiguous middle-layer blocks, diverging from evenly interleaved patterns in rule-based methods. This structure persists across model scales, sparsity levels, and allocation granularities, revealing a fine-grained spectrum of intrinsic attention behaviors that underlies the learned allocation. 

---
# Compositional Skill Routing for LLM Agents: Decompose, Retrieve, and Compose 

**Authors**: Xueping Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.18051)  

**Abstract**: LLM agents increasingly rely on external skills -- reusable tool specifications -- but real-world tasks often require composing multiple skills, not just selecting one. We formalize this as the Compositional Skill Routing problem: given a complex user query and a large skill library, decompose the query into atomic sub-tasks, retrieve the appropriate skill for each sub-task, and compose an executable plan. We present SkillWeaver, a decompose-retrieve-compose framework combining an LLM task decomposer, a bi-encoder skill retriever with FAISS indexing, and a dependency-aware DAG planner. To support evaluation, we introduce CompSkillBench, a benchmark of 300 compositional queries over 2,209 real MCP server skills spanning 24 functional categories, sourced from the public MCP ecosystem. Our experiments reveal that task decomposition quality is the primary bottleneck: standard LLM decomposition reaches only 34.2% category recall at the step level. To address this, we propose Iterative Skill-Aware Decomposition (SAD), a retrieval-augmented feedback loop that iteratively aligns decomposition with available skills. SAD improves decomposition accuracy from 51.0% to 67.7% (+32.7%, Wilcoxon p < 10^-6) in a single iteration; DA-conditioned analysis confirms that correct granularity is the prerequisite for effective retrieval (CatR@1 rises from 34% to 41% when DA=1). SkillWeaver reduces context window consumption by over 99%, and transfer experiments confirm generalization (+35.6% relative DA gain even when target categories are absent from the retrieval pool). 

---
# When English Isn't the Best Teacher: Source Language Effects in Cross-Lingual In-Context Learning 

**Authors**: Fred Philippy, Siwen Guo, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2606.18033)  

**Abstract**: Cross-lingual transfer in multilingual NLP has been widely explored in supervised fine-tuning contexts, where factors like data availability and linguistic similarity largely determine transfer quality. As the field shifts toward few-shot In-Context Learning (ICL), it is often presumed that insights from fine-tuning carry over unchanged. Yet this assumption has not been rigorously evaluated, leaving open the question of how to choose source languages for cross-lingual ICL. We conduct a broad empirical study of cross-lingual transfer in ICL spanning seven tasks, six models, and a typologically diverse set of languages. We further analyze language confusion, a key obstacle for generative tasks in cross-lingual ICL. Our results show that conventional fine-tuning-based expectations do not consistently apply in the ICL regime and point to alternative heuristics for selecting source languages effectively. 

---
# VoidPadding: Let [VOID] Handle Padding in Masked Diffusion Language Models so that [EOS] Can Focus on Semantic Termination 

**Authors**: Chunyu Liu, Zhengyang Fan, Kaisen Yang, Alex Lamb  

**Link**: [PDF](https://arxiv.org/pdf/2606.17999)  

**Abstract**: MDLMs generate text by denoising a preallocated masked response canvas, making response-length modeling central to instruction tuning. Existing MDLMs often inherit the autoregressive convention of using repeated \texttt{[EOS]} tokens for padding during instruction tuning, giving \texttt{[EOS]} a dual role as both a semantic terminator and a padding token. We show that this dual role is a root cause of \texttt{[EOS]} overflow under large-block decoding. To decouple these roles, we propose VoidPadding, which introduces \texttt{[VOID]} for padding and reserves \texttt{[EOS]} for termination. During inference, the learned \texttt{[EOS]} signal enables early stopping, while the learned \texttt{[VOID]} signal guides adaptive response canvas expansion. On Dream-7B-Instruct, VoidPadding improves the block-size-averaged four-task mean across mathematical reasoning and code generation benchmarks by \(+17.84\) points over the original model and \(+6.95\) points over RainbowPadding, while reducing decoding NFE by 55.7\% on average. Code is available at this https URL. 

---
# Fine-tuning LLMs for Passive Depression Severity Estimation from AI Mental Health Dialogue 

**Authors**: Olivier Tieleman, Ziyi Zhu, Ting Su, Samuel J. Bell, Thomas D. Hull, Caitlin A. Stamatis  

**Link**: [PDF](https://arxiv.org/pdf/2606.17973)  

**Abstract**: Depression is the leading cause of disability worldwide, and early detection of symptom change is essential for timely intervention. Validated instruments such as the Patient Health Questionnaire-9 (PHQ-9) support symptom monitoring at scale, but real-world completion rates are low, introducing response bias and systematic missingness. Passive approaches that infer severity from routinely generated data could close this gap. We address this by predicting PHQ-9 total scores directly from transcripts of conversations between users and an AI mental health application, requiring only conversation text and no additional clinical data. We fine-tune a Qwen3.5-27B backbone with a regression head, augment 3,111 ground-truth labels with pseudolabels generated by a reasoning model (Claude Opus) and iteratively trained intermediate models, for a combined dataset of 6,283 users. On a held-out test set of 842 users, our best model achieves MAE = 2.6, RMSE = 4.0, Pearson r = 0.80, and AUC = 0.91 at the PHQ-9 >= 10 clinical threshold. We also find AUC > 0.87 at every severity threshold from PHQ-9 >= 3 to PHQ-9 >= 24, demonstrating that the model captures depression severity across the full clinical spectrum. This work opens the door to passive, continuous symptom monitoring in AI mental health platforms, without requiring users to complete self-report measures. 

---
# Learning task-specific subspaces via interventional post-training of speech foundation models 

**Authors**: Jack Cox, Jon Barker  

**Link**: [PDF](https://arxiv.org/pdf/2606.17967)  

**Abstract**: Speech foundation models, pre-trained on large corpora of unlabelled speech data, produce general-purpose representations which are useful across tasks. However, these representations encode information about salient speech variables in a distributed manner, while downstream speech tasks rely on only some of this variability. In this work, we propose a post-training refinement approach using interventional contrastive learning. By leveraging an interventional dataset and multi-part contrastive loss, we learn a transformation from the entangled representation space of speech foundation models into separate content and speaker subspaces. We evaluate the learnt representations on speaker verification and keyword spotting tasks, showing improved out-of-domain speaker verification performance and evidence that speaker and content information are separated across the learned subspaces. 

---
# ChLogic: Evaluating Robustness of Logical Reasoning in Chinese Expressions 

**Authors**: Peixian Zhou, Yuxu Chen, Chaorui Zhang, Wei Han, Bo Bai, Xueyan Niu  

**Link**: [PDF](https://arxiv.org/pdf/2606.17905)  

**Abstract**: Large language models perform increasingly well on standardized logical reasoning benchmarks, but whether this ability remains robust beyond English is unclear. We introduce ChLogic, an English--Chinese aligned benchmark that tests whether models preserve logical reasoning performance when the same latent logical structure is expressed in English and diverse Chinese surface realizations. Built from formal logical templates, the benchmark contains three data sets: (i) the General aligned set, derived from 60 General Propositions across nine template families; (ii) the Difficult aligned set, derived from 40 Difficult Problems; and (iii) the Chinese-only set, covering 15 language-specific phenomenon types. Each aligned item pairs one English reference expression with five Chinese realizations. Experiments on Qwen3, Ministral, and GLM models reveal a persistent English--Chinese performance gap. Back-translation from standard Chinese into English often improves performance on the General aligned set, but produces mixed effects on the Difficult aligned set, where Qwen3-32B and GLM-5.1 perform worse after translation. These results indicate that Chinese surface realization, translation artifacts, and model-specific behavior jointly affect multilingual logical reasoning. Overall, ChLogic provides a useful stress test for the robustness of multilingual reasoning. 

---
# Dynamic Rollout Editing for Reducing Overthinking in RL-Trained Reasoning Models 

**Authors**: Zihao Wei, Wenjie Shi, Liang Pang, Jingcheng Deng, Shicheng Xu, Shasha Guo, Zenghao Duan, Jiahao Liu, Jingang Wang, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.17890)  

**Abstract**: Long-form chain-of-thought reasoning can improve LLM performance on complex tasks, but models often continue generating unnecessary reasoning after a correct answer has emerged. We refer to this behavior as overthinking. We study this phenomenon from the perspective of GRPO-style reinforcement learning (RL) post-training, framing it as a training-time credit-assignment problem rather than merely a decoding-time stopping problem. In rollouts sampled at the onset of GRPO training, we observe that successful trajectories can exhibit a slightly higher degree of overthinking than unsuccessful trajectories for the same prompts. This early imbalance provides a starting point for an undesirable feedback loop: because GRPO assigns sequence-level credit, it cannot distinguish the solution-reaching prefix from the unnecessary continuation that lengthens a successful trajectory. Both receive positive update signal, allowing the initial imbalance to grow into more severe overthinking during training. To address this issue, we introduce Dynamic Rollout Editing (DRE), a training-time intervention for successful trajectories that continue thinking after answer emergence. DRE preserves the accepted verified prefix, edits the remaining thinking, and prefers the edited trajectory within the same RL group, weakening the preference signal for unnecessary thinking without penalizing the reasoning needed to reach the answer. Experiments across diverse tasks show the effectiveness of DRE. 

---
# GameCraft-Bench: Can Agents Build Playable Games End-to-End in a Real Game Engine? 

**Authors**: Tongxu Luo, Rongsheng Wang, Jiaxi Bi, Chenming Xu, Zhengyang Tang, Jianlong Chen, Juhao Liang, Ke Ji, Shuqi Guo, Yuhao Du, Fan Bu, Wenyu Du, Xiaotong Zhang, Kyle Li, Shaobo Wang, Linfeng Zhang, Yuxuan Liu, Xin Lai, Chenxin Li, Yiduo Guo, Zhexin Zhang, Xinyuan Wang, Tianyi Bai, Ziniu Li, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.17861)  

**Abstract**: Game generation is an emerging application of coding agents, requiring models to transform natural-language specifications into playable interactive systems. Unlike traditional coding tasks, game generation takes place within a game engine, where scripts, scenes, assets, rendering, and runtime interactions must jointly produce coherent gameplay. We formalize end-to-end game generation as the problem of producing a complete game artifact that realizes a specification through observable player-game interaction in a target environment. We argue that evaluating this setting requires three desiderata: Engine Grounding, Artifact Completeness, and Interactive Verification. We propose an interaction-grounded evaluation framework that assesses executable gameplay through replayed demonstrations and rubric-guided multimodal judging. We instantiate this framework as GameCraft-Bench, a benchmark comprising 140 Godot tasks across 15 game families. Evaluations of frontier coding agents show that end-to-end game generation remains highly challenging: the strongest agent achieves only 41.46%, and most agents score below 40%. Further analysis reveals that while agents often implement recognizable mechanics, they struggle to deliver complete games with sufficient content, functional visual feedback, and coherent presentation. See this https URL for demos, code, and data. 

---
# Environment-Grounded Automated Prompt Optimization for LLM Game Agents 

**Authors**: Rean Clive Fernandes, Lukas Fehring, Theresa Eimer, Marius Lindauer, Matthias Feurer  

**Link**: [PDF](https://arxiv.org/pdf/2606.17838)  

**Abstract**: LLM agents in interactive environments are highly sensitive to their prompts, yet prompt engineering remains a manual, task-specific process. We introduce an automated prompt optimization framework for LLM agents that decomposes the observation-to-action pipeline into a goal-conditioned descriptor agent and an action selection agent, and iteratively refines each module's prompt through an LLM-driven evolutionary loop guided by environment returns. We propose a behavior analyzer to attribute episode outcomes to specific prompt components, and a mutator to propose targeted revisions to the prompt, before validating them through environment rollouts. We evaluate on all five BabyAI tasks in the BALROG benchmark, comparing our pipeline against BALROG's RobustCoTAgent under both plain and guided prompt initializations. Optimization improves performance consistently across tasks and conditions, without requiring updates to the model weights. On PutNext, a multi-step coordination task where the RobustCoTAgent achieves 0% success, our framework reaches up to 72.5% success rate using the same underlying LLM with optimized prompts. These results suggest that a multi-agent framework, combined with automatic prompt optimization, enhances LLMs without the need for fine-tuning or extensive human supervision. 

---
# Perceptual compensation for tonal context in self-supervised speech models 

**Authors**: James Kirby, Ioana Krehan, Michele Gubian  

**Link**: [PDF](https://arxiv.org/pdf/2606.17835)  

**Abstract**: This study examines the extent to which the wav2vec2.0 architecture exhibits evidence of compensation for phonological context. We conducted a pseudo-replication of a perceptional compensation experiment on Mandarin Chinese tones, and compared the embedding similarities and probing classifier outputs between a purely self-supervised pre-trained model and a model fine-tuned for Mandarin ASR. No evidence of compensation was found in the embedding similarities of the purely pre-trained model. Probing classifiers showed some evidence of compensation in addition to the expected layer-wise improvements in categorization, but failed to replicate human performance on isolated test syllables. Our findings contrast with previous reports of sensitivity to phonological structure emerging through pre-training alone, and suggest that supervised objectives may be necessary to encourage the abstraction of at least some types of phonological regularities. 

---
# When Multiple Scripts Matter: Evaluating ASR in Clinical Settings 

**Authors**: Jean Seo, Minkyu Kim, Jeonguk Lee, Jisoo Jung, Wooseok Han, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.17826)  

**Abstract**: Automatic speech recognition (ASR) in non-English clinical settings is challenged by multiscript variability, where the same term may appear in multiple valid orthographic forms. Conventional string-matching evaluation metrics often underestimate ASR performance by treating orthographic variants as errors. To address this issue, we introduce MultiClin, a clinical ASR benchmark designed to evaluate robustness to multiscript variability. Experiments across diverse ASR models show that multiscript-aware evaluation provides a fairer assessment of recognition quality than conventional single-reference evaluation. We further investigate the impact of script consistency during training and find that inconsistent script mappings increase orthographic uncertainty and hinder model convergence, with a balanced 50% mapping ratio producing the highest entropy. In contrast, script unification consistently yields the best ASR performance. Our dataset and code are publicly available at: this https URL. 

---
# Improving low-resource ASR using bilingual fine-tuning with language identification: a cross-linguistic evaluation 

**Authors**: Reihaneh Amooie, Yun Hao, Wietse de Vries, Jelske Dijkstra, Matt Coler, Martijn Wieling  

**Link**: [PDF](https://arxiv.org/pdf/2606.17820)  

**Abstract**: This study explores how bilingual fine-tuning affects automatic speech recognition (ASR) in low-resource languages. We evaluate this method across nine linguistically and geographically diverse language pairs, covering a range of language families and writing systems. To distinguish the two languages, during training, we pre-pend each input text with a language identification token. At inference, the model jointly predicts both the language and transcription from the speech input alone. As texts for which the language is incorrectly determined show low ASR performance, we also conduct a follow-up experiment in which the language identification token is provided both during training and inference. Our results show that bilingual fine-tuning can be beneficial when language identification accuracy is high, and that in cases where language identification performance is low, including the language identification token at inference helps to improve ASR performance. 

---
# The Slop Paradox: How Synthetic Standardization Erodes Clinical Uncertainty and Cross-Modal Alignment in AI-Rewritten Radiology Reports 

**Authors**: Samar Ansari  

**Link**: [PDF](https://arxiv.org/pdf/2606.17791)  

**Abstract**: AI-assisted clinical documentation tools increasingly summarize, standardize, and reformat radiology reports using large language models (LLMs). We present a controlled measurement of the resulting information degradation. Using 450 chest X-ray reports from the Indiana University dataset, we generate synthetic versions via three realistic LLM rewriting tasks: EHR summarization, standardized rewriting, and teaching case preparation. We measure entity erosion (via medical NER), hedging collapse (loss of clinical uncertainty language), and cross-modal alignment degradation (via BiomedCLIP image-text similarity). Our central finding is a dissociation between information loss and cross-modal fidelity. EHR summarization is the most destructive at the content level, eroding 51.4% of clinical entities and 43.7% of hedging language, yet it preserves image-text alignment almost entirely (a 2.5% drop). The two tasks meant to produce cleaner training data, standardized rewriting and teaching case preparation, do the reverse: they preserve more entities (26.8% and 29.3% eroded) but cause 14.9-16.5% alignment drops, six to seven times those of EHR summarization. We term this the slop paradox: rewriting that makes clinical text look cleaner for multimodal training is precisely what pulls it away from the image. Contrary to our pre-specified hypothesis, rare pathologies were not preferentially degraded: across nine rare-versus-common comparisons, no difference survived multiple-comparison correction, and nominal differences ran in the opposite direction (common > rare), so contamination is invisible to condition-specific monitoring. The dominant determinant of degradation is the type of AI rewriting task, not the clinical content. These findings bear on multimodal medical AI dataset construction and the governance of AI-assisted clinical documentation. 

---
# LLMs Infer Cultural Context but Fail to Apply It When Responding 

**Authors**: Yisong Miao, Jian Zhu, Vered Shwartz  

**Link**: [PDF](https://arxiv.org/pdf/2606.17688)  

**Abstract**: Recent work has shown that LLMs overrepresent dominant cultures, particularly Western ones, while marginalizing others. We investigate whether this affects models' ability to generate culturally adapted responses by evaluating their use of local measurement units based on the user's perceived cultural background. We introduce Cultural and Pragmatic Response Inference (CAPRI), a dataset of conversations with varying levels of cultural cues. Experiments with state-of-the-art LLMs show that models can infer cultural background and recall relevant conventions, but often fail to utilize the information to adapt their answers to the relevant cultural conventions, unless explicitly prompted to perform the tasks sequentially. We further evaluate adaptation to the interpretation of time and quantity expressions, two subjective language grounding dimensions that are affected by culture. We find that models increasingly adapt their answers as cultural cues accumulate, but their priors are not culture-neutral, sometimes aligning with the model's country of origin. Overall, CAPRI provides a resource for future research aimed at narrowing the gap between cultural knowledge and culturally adaptive language generation. 

---
# SuCo: Sufficiency-guided Continuous Adaptive Reasoning 

**Authors**: Jiahao Wang, Bingyu Liang, Chenhao Hu, Longhui Zhang, Xuebo Liu, Min zhang, Jing Li, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.17687)  

**Abstract**: Despite remarkable performance on complex tasks, Large Reasoning Models (LRMs) often generate excessively long Chain-of-Thoughts (CoT), inflating computational costs even for simple queries. Existing efforts to mitigate this inefficiency typically rely on discrete reasoning modes or fixed budget tiers, lacking a principled criterion of when reasoning is sufficient. In this work, we introduce Minimal Sufficient CoT (MSC), defined as the shortest prefix of a CoT trajectory which is adequate for producing the correct answer. We empirically show that MSC not only reduces reasoning tokens, but also improves accuracy across difficulty levels. Building on MSC, we propose Sufficiency-guided Continuous Adaptive Reasoning (SuCo), a two-stage training framework for autonomous reasoning control along a continuous spectrum. In stage 1, MSC-Aligned Fine-Tuning (MFT) constructs MSC data using problem-adaptive sufficiency thresholds that naturally scale with question difficulty, then fine-tunes the model to internalize concise yet sufficient reasoning patterns. In stage 2, Sufficiency-Aware Policy Optimization (SAPO) further optimizes the model through reinforcement learning with dynamic complexity tracking and sufficiency-aware rewards that penalize both over- and under-thinking. Extensive experiments across mathematics, code, and science benchmarks show that SuCo consistently achieves improvements in both accuracy and reasoning efficiency. 

---
# Bridging Functional Correctness and Runtime Efficiency Gaps in LLM-Based Code Translation 

**Authors**: Longhui Zhang, Jiahao Wang, Chenhao Hu, Bingyu Liang, Jing Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.17683)  

**Abstract**: While large language models (LLMs) have greatly advanced the functional correctness of automated code translation systems, the runtime efficiency of translated programs has received comparatively little attention. With the waning of Moore's law, runtime efficiency has become increasingly important for program quality, alongside functional correctness. Our preliminary study reveals that LLM-translated programs often run slower than human-written ones, and this issue cannot be remedied through prompt engineering alone. Therefore, our work proposes SwiftTrans, a code translation framework comprising two key stages: (1) Multi-Perspective Exploration, where MpTranslator leverages parallel in-context learning (ICL) to generate diverse translation candidates; and (2) Difference-Aware Selection, where DiffSelector identifies the optimal candidate by explicitly comparing differences between translations. We further introduce Hierarchical Guidance for MpTranslator and Ordinal Guidance for DiffSelector, enabling LLMs to better adapt to these two core components. To support the evaluation of runtime efficiency in translated programs, we extend existing benchmarks, CodeNet and F2SBench, and introduce a new benchmark, SwiftBench. Experimental results across all three benchmarks show that SwiftTrans achieves consistent improvements in both correctness and runtime efficiency. 

---
# From Trainee to Trainer: LLM-Designed Training Environment for RL with Multi-Agent Reasoning 

**Authors**: Chao Chen, Chengzu Li, Zhiwei Li, Yinhong Liu, Zhijiang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.17682)  

**Abstract**: Reinforcement learning pipelines for Large Language Model (LLM) training often rely on manually redesigned environments between stages, requiring practitioners to heuristically infer which configuration will best improve the current policy. To automate this process, we propose the LLM-as-Environment-Engineer framework in which the current policy model analyzes failure trajectories together with contextual information and proposes modifications to the next-stage training environment configuration. We also introduce MAPF-FrozenLake, a controllable testbed whose generator exposes multi-dimensional environment configurations, making it suitable for studying and benchmarking environment redesign. On this testbed, we condition the environment engineer on structured summaries of policy behavior, failure cases, and environment statistics, from which it produces the configuration for the next training stage. With Qwen3-4B as the backbone, our framework achieves the strongest aggregate performance on our benchmarks, outperforming larger proprietary LLMs (e.g., GPT, Gemini) and fixed-environment training baselines. We further analyze which forms of context are most effective, finding that successful environment updates rely on failure evidence and preserve configurations that already work. Interestingly, the current RL checkpoint serves as a better environment engineer than the original base model, suggesting that policy learning improves the model's ability to diagnose its remaining weaknesses. 

---
# Prompt Perturbation for Reliable LLM Evaluation over Comparison Graphs 

**Authors**: Dong Huang, Jianbo Sun, Pengkun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.17634)  

**Abstract**: Evaluating large language models (LLMs) is important for understanding their capabilities, comparing competing systems, and supporting the deployment of reliable models in practice. For open-ended tasks, pairwise evaluation has become a popular paradigm, in which two responses to the same prompt are compared and the resulting judgments are aggregated into an overall ranking. A central challenge of this paradigm is intransitivity: the induced comparison outcomes may fail to support any coherent global ranking. For example, one may observe cyclic preferences such as $A \succ B \succ C \succ A$, or inconsistencies involving ties such as $A \equiv B\equiv C\neq A$. Such contradictions make the resulting leaderboard unstable and challenging to interpret. In this paper, we propose a prompt perturbation framework for improving the consistency of pairwise LLM evaluation. Our approach generates perturbed variants of each prompt, uses the resulting comparison graphs to identify and filter out structurally inconsistent comparison patterns, and then applies standard ranking methods to the filtered comparisons. A key feature of the proposed framework is that graph-level structural consistency is incorporated explicitly into the evaluation pipeline before ranking aggregation. This provides a simple and principled way to reduce cyclic inconsistencies and improve the reliability of LLM rankings. 

---
# OPD-Evolver: Cultivating Holistic Agent Evolver via On-Policy Distillation 

**Authors**: Guibin Zhang, Xun Xu, Yanwei Yue, Zikun Su, Wangchunshu Zhou, Xiaobin Hu, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.17628)  

**Abstract**: Memory has become a standard substrate for self-evolving agents, yet retaining experience is not the same as learning how to evolve through it. Existing memory agents can store trajectories, retrieve reflections, or accumulate skills, but often lack the holistic competence to select useful experience, act on it, write reusable knowledge, and maintain a growing repository. We introduce OPD-Evolver, a slow-fast co-evolution framework that cultivates such an agent evolver through on-policy self-distillation. In the fast loop, OPD-Evolver interacts with a four-level memory hierarchy to read, use, write, and maintain experience for rapid test-time evolution. In the slow loop, outcome-calibrated memory attribution and privileged hindsight distill these four abilities into the deployable policy. Across multi-domain benchmarks, OPD-Evolver surpasses memory systems such as ReasoningBank by up to 11.5%, and training-based methods such as Skill0 by ~5.8%. Further analysis shows that OPD-Evolver internalizes high-value experience and memory management, enabling OPD-Evolver-9B to challenge giant counterparts such as Qwen3.5-397B-A17B and Step-3.5-Flash, pointing beyond memory-augmented agents toward genuinely qualified agent evolvers. 

---
# The Benchmark Illusion: Pruned LLMs Can Pass Multiple Choice but Fail to Answer 

**Authors**: Rui Wen, Lu Sun, Jiayang Liu, Zesheng Xu, Tianshuo Cong, Zheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.17609)  

**Abstract**: Compressing large language models reduces memory use and inference cost, but it can also create failures that standard benchmarks miss. A pruned model may still perform well on multiple-choice evaluations, yet fail to answer the same question in open generation. We ask what pruning changes: does it erase the correct answer, or does it make the answer harder to produce as the top output?
We study this question with multilingual question answering, tracking the same questions before and after pruning. We find a benchmark illusion. Under high-sparsity pruning, especially Wanda, models often fail in greedy open generation while still selecting the correct answer under multiple-choice scoring. In these recognition-only errors, the answer is usually not gone, but demoted: it often reappears with beam search, sampling, or one in-context example. Overall, multiple-choice benchmarks can overstate the usability of compressed LLMs, creating an evaluation blind spot. Compressed models should be tested on what they can produce, not only on what they can recognize. 

---
# Evaluating Large Language Models Abilities for Addressee, Turn-change, and Next Speaker Prediction in Meetings 

**Authors**: Ryo Fukuda, Takatomo Kano, Siddhant Arora, Marc Delcroix, Naohiro Tawara, Atsunori Ogawa, Yuya Chiba, Atsushi Ando, William Chen, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2606.17542)  

**Abstract**: We investigate turn-taking in multimodal multi-party conversations using large language models (LLMs). We construct an evaluation framework for three tasks: addressee detection, turn-change prediction, and next speaker prediction. We compare supervised models trained for these tasks, text-based LLMs, multimodal LLMs (MM-LLMs), and human subjects. Experiments on the AMI corpus showed that LLMs outperformed supervised models and humans in next speaker prediction, despite not being trained on the target domain and without access to audio or visual information. An MM-LLM performed better than text-based LLMs on addressee detection and turn-change prediction but remained below human performance, indicating difficulty leveraging raw audio-visual signals. Ablation analyses revealed that conversational context was critical, particularly for next speaker prediction. We observed that human and LLM prediction patterns were similar, and intervals with frequent turn changes were difficult for both. 

---
# An expressivity analysis of hierarchical modelling in deep transformers via bounded-depth grammars 

**Authors**: Vinoth Nandakumar, Qiang Qu, Pramod Thebe, Sakshi Khachariya, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.17522)  

**Abstract**: Deep neural networks are widely believed to derive their expressive power from their ability to form \textbf{hierarchical representations}, capturing progressively more abstract and compositional features across layers. In language modeling, \textbf{transformers} have emerged as the dominant architecture, with early layers capturing local syntactic patterns and later layers encoding more complex clause-level dependencies. While this intuition has shaped model design, there remains a lack of rigorous theoretical work demonstrating \textbf{how} deep transformers represent such hierarchical structures. In this work, we analyze the expressiveness of deep transformer models through the formal lens of bounded-depth, non-recursive context-free grammars. For this class of grammars, we explicitly construct transformers with positional attention whose depth grows linearly with grammar depth, while the neuron count scales with the number of derivation-tree shapes and quadratically with the number of production rules. Our theoretical results support the linear representation hypothesis by demonstrating that these architectures possess the structural capacity to encode abstract grammatical states into low-dimensional, linearly separable subspaces within the residual stream. 

---
# Scaling Enterprise Agent Routing: Degradation, Diagnosis, and Recovery 

**Authors**: Kellen Gillespie, Robyn Perry  

**Link**: [PDF](https://arxiv.org/pdf/2606.17519)  

**Abstract**: Production LLM assistants route user requests to growing libraries of specialized tools, but how does routing accuracy degrade as the catalog scales? We study single-step routing on a 110-agent, 584-tool catalog from a deployed enterprise productivity assistant, evaluating three frontier models from 10 to 110 agents. Routing F1 on under-specified requests drops 16--23 percentage points across models. An oracle analysis decomposes the degradation into a \emph{retrieval} gap (the model cannot surface the right tool) and a \emph{confusion} gap (even with perfect retrieval, the oracle ceiling drops 10pp). Embedding-based shortlisting recovers +10--11pp F1 at full scale across all three models and two providers. A production annotation study (1,435 human-labeled utterances, three annotators) confirms the recovery on real traffic at +10--17pp despite 10--15pp lower absolute performance. 

---
# Evaluating Second-Order Bias of LLMs Through Epistemic Entitlement 

**Authors**: Ramaravind Kommiya Mothilal, Terry Jingchen Zhang, Raiyan Ahmed, Zhijing Jin, Shion Guha, Syed Ishtiaque Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2606.17506)  

**Abstract**: Evaluations of social bias in LLMs largely focus on whether models generate or imply biased content. However, as LLMs are increasingly used as judges of bias, they may exhibit social biases in subtler ways in how they evaluate biased content, which current methods do not systematically capture. We call this second-order bias: social bias in an LLM's judgment about social bias, which we evaluate through a novel, philosophically grounded reasoning task. Drawing on entitlement epistemology, we conceptualize bias as misplaced foundational knowledge that shapes an agent's rational inquiry, and derive a logical reasoning task for LLMs to judge to whom a biased text is acceptable or non-acceptable. We develop two simple metrics to measure how biased LLM judges are in inferring demographics for acceptability without sufficient support, and how these inferences vary across groups targeted by biased texts. Evaluating open and closed models, we find that our task evades safety guardrails by surfacing bias in model judgment. It varies systematically across target groups, reflects implicit social maps, and shows how models are still triggered by demographic labels. Our work points to the need for LLM bias evaluation in judgment tasks and broadly, for more theoretically grounded approaches to bias evaluation in NLP. We release our code and model responses at this https URL. 

---
# Decoding Hidden Deception in Reasoning LLMs: Activation Explainers for Deception Auditing 

**Authors**: Kexin Chen, Yi Liu, Haonan Zhang, Yanhui Li, Xinyu Deng, Dongxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.17478)  

**Abstract**: As LLMs acquire stronger reasoning capabilities, deceptive behavior becomes an increasingly serious safety concern. Existing deception monitors either score visible transcripts or derive scalar probe scores from representation vectors, leaving little inspectable evidence about why a response is suspicious. We introduce STATEWITNESS, an activation explainer for deception auditing. A separate decoder reads a target model's hidden states, then answers natural-language queries or emits structured reports about them. We evaluate STATEWITNESS on two target reasoning LLMs across seven deception datasets. STATEWITNESS reaches 0.916 mean AUROC, a relative gain of 11.6% over the best black-box text monitor and 25.0% over the best activation-probe baseline under the same evaluation protocol. When combined with existing monitors, STATEWITNESS reduces missed deceptive examples in simple threshold ensembles. Beyond scalar detection, the decoder returns query-level answers, schema reports, and token- or sentence-level evidence traces for human inspection. We view this interface as a potential building block for broader interpretability and alignment tools. 

---
# AIPatient Arena: EHR-grounded evaluation of large language models in end-to-end clinical consultation workflows 

**Authors**: Jiahui Niu, Huizi Yu, Wenkong Wang, Guangxin Dai, Jingxian He, Xiang Li, Zhiying Liang, Xinxin Lin, Kent CY So, Bryan YP Yan, Yun Kwok Wing, Yanqiu Xing, Xin Ma, Lizhou Fan  

**Link**: [PDF](https://arxiv.org/pdf/2606.17474)  

**Abstract**: Large language models (LLMs) are increasingly considered for use in clinical consultation tasks, yet most medical evaluations remain static, single-turn, or narrowly outcome-based, limiting their ability to reflect the sequential, uncertain, and interactive nature of real-world care. Here, we propose AIPatient Arena, an EHRs-grounded evaluation framework for assessing the clinical utility of LLMs across eight dimensions of clinical competence. The framework integrates EHR data into patient-specific knowledge graphs, enabling multi-turn physician-patient interactions. We applied AIPatient Arena on a primary cohort of 437 patients and two out-of-distribution validation cohorts of 119 and 67 patients. We observe that LLMs performed well in medical interview questioning skills (QS; mean scores, 4.43-4.99/5), ethical and professional conduct (ET; 4.38-4.93/5), and clarity and transparency of clinical explanations (EX; 3.80-4.72/5). Performance was moderate in information integration (II; 3.19-4.21/5) and medication safety and justification (MS; 3.13-3.78/5), but persistent weaknesses were observed in handling of ambiguous patient responses (HR; 2.57-3.32/5), information coverage (IC; 2.08-3.02/5), and diagnostic accuracy and reasoning (Dx; 2.63-3.55/5). Process-based evaluation revealed recurrent interaction failures, including repetitive questioning, omission of past medical history, and inadequate handling of uncertainty. Richer conversational context improved diagnostic reasoning but yielded limited gains in treatment planning. These findings indicate that final-answer accuracy alone is insufficient for evaluating clinical readiness and highlight the importance of assessing how models gather, interpret, and communicate information throughout a consultation. AIPatient Arena provides an EHR-grounded framework for workflow-oriented pre-deployment evaluation of medical LLMs. 

---
# MODE-RAG: Manifold Outlier Diagnosis and Energy-based Retrieval-Augmented Generation Evaluation 

**Authors**: Zehang Wei, Jiaxin Dai, Jiamin Yan, Xiang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.17449)  

**Abstract**: While Multimodal Retrieval-Augmented Generation (M-RAG) enhances Large Vision-Language Models, it remains highly susceptible to cross-modal hallucinations, causal fabrications, and sycophancy. Furthermore, existing mitigation pipelines often face an intervention paradox: static rules tend to unnecessarily disrupt accurate generations, whereas leaving the multi-modal reasoning completely unguided allows existing mismatches to cascade into severe logical fabrications. To quantify and mitigate these hallucinations, we propose a Multi-Agent system, MODE-RAG, driven by Variational Free Energy (VFE) and internal attention states to dynamically gate interventions. High-risk queries are routed to five stage-specific agents, integrating Monte Carlo Tree Search (MCTS) for rigorous causal derivation and logit perturbations to penalize sycophancy. Dedicated Correction and Overseer agents ensure formatting stability and perform post-hoc factual verification. To objectively evaluate our approach, we introduce ModeVent, a challenging subset derived from the MultiVent dataset. Extensive experiments indicate that our system effectively reduces hallucination rates and logical fabrication, significantly improving the robustness of M-RAG systems. 

---
# NarrativeWorldBench: A Frontier-Saturated Benchmark and a Latent World Model for Long-Horizon Co-Creative Audio Drama 

**Authors**: Logan Mann, Abdur Rahman, Mohammad Saifullah, Taaha Kazi, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2606.17391)  

**Abstract**: Long-form serialized audio drama, with arcs that run for 200 to 800 episodes, is a major creative medium and a setting where frontier large language models (LLMs) fail. We benchmark 21 models, spanning classical, fine-tuned, open-frontier, closed-frontier, and reasoning tiers, on a uniform set of structural narrative metrics. All closed-frontier systems saturate at a plot-beat F1 in the band [0.78, 0.81] and collapse by about -0.20 F1 at horizon h=200. We introduce NarrativeWorldBench, an open benchmark of nine narrative-structure metrics evaluated across horizons h in {10, 20, 50, 100, 200}, with cross-lingual evaluation across four Indic languages (Hindi, Tamil, Telugu, Marathi). We introduce N-VSSM, a Narrative Variational State-Space Model that maintains a structured 256-dimensional latent world state over more than 200 episodes via a Mamba-2 backbone with an event-conditioned posterior and an 8B decoder. N-VSSM holds plot-beat F1 >= 0.84 across all horizons at 4x lower compute than the closed-frontier band. A learned Cultural Transfer Function lifts cross-language fidelity by +0.20 to +0.23 Likert points. In a within-subjects writer study (n = 12 professional authors, 240 trials), N-VSSM is preferred over Claude Opus 4.5 on long-arc consistency 71% of the time and rated +1.3 Likert points higher on controllability. 

---
# Implicit vs. Explicit Prompting Strategies for LVLMs in Referential Communication 

**Authors**: Peter Zeng, Amie J. Paige, Weiling Li, Susan E. Brennan, Owen Rambow, Cameron R. Jones  

**Link**: [PDF](https://arxiv.org/pdf/2606.17372)  

**Abstract**: Two recent studies (Jones et al. (2026); Zeng et al. (2026)) reach apparently contradictory conclusions about whether LVLMs can coordinate on efficient referring expressions. We control for task differences between the studies while directly comparing their prompting styles. We replicate the finding that models can coordinate efficient referring expressions when explicitly prompted to do so, suggesting that other task differences are not responsible for divergent results. However, we also find that the same models fail to infer the need for communicative efficiency from a more implicit prompt, highlighting critical differences between how humans and AI systems communicate. 

---
# Translating the Untranslatable: An Operationalizable Ontology for Untranslatability 

**Authors**: Jacob Bremerman, Brihi Joshi, Hirona Arai, Xiang Ren, Jonathan May  

**Link**: [PDF](https://arxiv.org/pdf/2606.17354)  

**Abstract**: Untranslatability, cases where meaning cannot be directly preserved across languages, is well-studied in linguistics but underexplored in NLP. As machine translation (MT) systems improve on standard benchmarks, their limitations increasingly concentrate in such cases, where translation cannot be reduced to one-to-one equivalence. We introduce a structured ontology of untranslatability along with a taxonomy of compensation strategies, which are specific techniques to convey meaning under these untranslatable circumstances. We operationalize this framework into a multilingual dataset of untranslatable sentences paired with strategy-based translations, enabling controlled analysis of translation behavior. Initial human preference studies suggest that translation quality depends on the strategy used, with consistent preferences for outputs that include explanatory context, known as the Annotation compensation strategy. Our framework and dataset provide a foundation for studying and modeling strategy-informed machine translation. 

---
# Do Large Language Models Always Tell The Same Stories? 

**Authors**: Thennal DK, Hans Ole Hatzel  

**Link**: [PDF](https://arxiv.org/pdf/2606.17350)  

**Abstract**: Recent advances in large language models (LLMs) have enabled the generation of high-quality prose, yet the question of whether these models are capable of generating diverse outputs remains contested. In this work, we investigate the diversity of LLM-generated stories through the framework of narrative similarity. Using a contrastive framework and a dataset of human-written stories and prompts from r/WritingPrompts, we collect narrative similarity judgments across 10 representative LLMs, utilizing both human evaluations and three different automatic annotation methods. Our findings reveal a consistent trend: LLM-generated narratives are consistently more similar to each other than human-written stories are. We demonstrate that frontier models in particular converge on a ``mean'' generic narrative that approximates individual human stories but lacks the collective diversity of human authors. Finally, we show that common mitigation strategies, including negative prompting and temperature scaling, fail to meaningfully address this homogeneity. 

---
# Examining the Limits of Word2Vec with Toki Pona 

**Authors**: Daniel Zhenhan Huang, Hongchen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.17299)  

**Abstract**: Word2Vec's effectiveness at generating semantic embeddings has been widely validated, yet it has been tested almost exclusively on languages with large vocabulary inventories. This study examines whether Word2Vec can successfully capture semantic relationships within an extremely reduced vocabulary using data from Toki Pona, a constructed language with approximately 130 words. We sourced 1.4 million sentences (7.95 million tokens) from the Toki Pona community for training. Approximately 23% of sentences in the corpus contain non-Toki Pona tokens such as named entities, loanwords, and neologisms. To investigate whether this linguistic noise enhances or hinders performance -- a topic rarely addressed in word embedding literature -- we trained two distinct models: one retaining these incidental tokens and another filtering them out completely. Evaluation was conducted using quantitative methods measuring word proximity to semantic category centroids, automated silhouette scores via agglomerative clustering, and qualitative analysis utilizing representational similarity matrices compared against English. The results indicate that while sparse, non-core tokens do not affect the relative structure of the learned embeddings, they actually draw similar words closer together in the vector space. Importantly, Word2Vec's effectiveness depends more on distributional patterns than lexicon size even at this extreme lower bound. 

---
# Are you speaking my languages? On spoken language adherence in multimodal LLMs 

**Authors**: Hyungwon Kim, Kandarp Joshi, Lillian Zhou, Pavel Golik, Petar Aleksic  

**Link**: [PDF](https://arxiv.org/pdf/2606.17281)  

**Abstract**: While Large Language Model (LLM) based Automatic Speech Recognition (ASR) enables seamless multilingual use, models often misidentify the output language, compromising transcription fidelity and downstream application quality. To preserve flexibility and code-switching capabilities, we propose a soft prompting approach that hints at potential spoken languages without strictly constraining the output. We formally define this challenge as a lack of language adherence, introduce a novel metric to quantify violations, and evaluate three mitigation strategies: (1) zero-shot prompting for robust guidance under uncertainty, (2) supervised fine-tuning (SFT) to improve prompt adherence, and (3) Chain-of-Thought (CoT) reasoning to enforce adherence during decoding. We present a comparative analysis of these methods across multiple languages, evaluating effectiveness in reducing the language violation while maintaining overall ASR performance. Finally, we discuss trade-offs to guide strategy selection under various compute constraints. 

---
# MLLP-VRAIN UPV system for the IWSLT 2026 Simultaneous Speech Translation task 

**Authors**: Jorge Iranzo-Sánchez, Gerard Mas-Mollà, Adrià Giménez, Jorge Civera, Albert Sanchis, Alfons Juan  

**Link**: [PDF](https://arxiv.org/pdf/2606.17255)  

**Abstract**: This work describes the participation of the MLLP-VRAIN research group in the shared task of the IWSLT 2026 Simultaneous Speech Translation track. Our submission utilizes the recently released Parakeet and Qwen 3.5 models to create a robust, cascaded solution for long-form SimulST through the use of adaptive "black-box" policies. We explore relaxations of these policies to achieve better quality-latency trade-offs. Compared to last year, we participate on all language directions. In addition to this, for the En$\rightarrow${De, It, Zh} directions we also participate in this year's new context track employing a combination of ASR word-boosting and a RAG mechanism of offline pre-translated exemplars to guide generation and enrich our system with domain-specific context. Finally, we provide a detailed latency analysis of our system. Compared to last year, results on the MCIF En$\rightarrow$De test set shows a substantial quality improvement of +5.82 XCOMET-XL. Our context track processing further improves performance by +1.03. 

---
# Speaking in Self-Assessing Tongues: On the Verbalized Confidence of LLMs in Machine Translation 

**Authors**: Ali Marashian, Alexis Palmer, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2606.17234)  

**Abstract**: The rapid rise in popularity of large language models (LLMs) for translation calls for a thorough study of the reliability of their confidence in their own outputs. Unlike many generation tasks, translation errors and confidence levels can be useful at different levels of granularity (tokens, words, or spans). Unsupervised approaches based on internal signals like predicted probabilities can be misleading because they reflect certainty among alternatives rather than correctness. In addition, they require access to such internal signals. Here, we devise five verbalized methods of extracting an LLM's per-token confidence without those shortcomings and compare their reliability with that of the model's internal signals of certainty. We evaluate reliability using two forms of alignment: fine-grained error detection and calibration. For both, internal and verbalized methods perform similarly, although results vary by model. Interestingly, we find little to no correlation between internal and verbalized methods. 

---
# Revisiting LLM Adaptation for 3D CT Report Generation: A Study of Scaling and Diagnostic Priors 

**Authors**: Vanshali Sharma, Andrea M. Bejar, Halil Ertugrul Aktas, Quoc-Huy Trinh, Debesh Jha, Gorkem Durak, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2606.17213)  

**Abstract**: Recent advances in multimodal learning, including large language models (LLMs) and vision-language models (VLMs), have demonstrated strong adaptability to natural images. However, extending their use to the medical domain, particularly for volumetric (3D) images, is challenging due to high computational complexity, volumetric dependencies and the semantic gap between visual features and clinical terminology. Naively fine-tuning LLMs on limited medical data often leads to overfitting and clinical hallucination, where linguistic fluency is prioritized over clinical factuality. In this study, we investigate parameter-efficient adaptation strategies for volumetric CT report generation and introduce RAD3D-Prefix, a lightweight diagnostic-prior conditioning framework that minimizes the need for extensive parameter training. This module integrates image embeddings with multi-label diagnostic classification logits, preserving critical clinical details while bridging the semantic gap. By keeping the LLM frozen, our method requires minimal trainable parameters and mitigates the risk of overfitting on small, domain-specific datasets. Through a systematic study spanning LLMs from 96.1M to 1.6B parameters, we find that fine-tuning is most beneficial for smaller LLMs, whereas freezing larger (~1B+ LLMs and training only lightweight projection layers provides a superior trade-off between performance, generalization, and computational efficiency. Across multiple automatic metrics and a clinical reader study, RAD3D-Prefix outperforms comparable parameter-efficient baselines and demonstrates strong out-of-domain generalization while using substantially fewer trainable parameters than fully fine-tuned alternatives. 

---
# Self-Generated Error Training for Token Editing in Diffusion Language Models 

**Authors**: Lin Yao  

**Link**: [PDF](https://arxiv.org/pdf/2606.17175)  

**Abstract**: Token-to-token (T2T) editing lets LLaDA2.1 revise committed tokens during block-diffusion decoding. The released recipe trains this editor on random vocabulary corruptions, but at inference the editor sees the model's own fluent, high-confidence draft errors instead. We study this training-inference mismatch and propose self-generated T2T, which performs a no-gradient draft pass, fills masked positions with predicted tokens, and supervises recovery in a second pass under these self-generated corruptions. We implement the update as a short LoRA continued-pretraining pass on LLaDA2.1-mini and evaluate on several benchmarks under the official Q-Mode T2T procedure with unchanged inference parameters. The method generally improves accuracy while reducing T2T edit intensity, mitigating failure modes such as final-digit transcription errors after otherwise correct reasoning and excessive self-correction before short factual answers. 

---
# From Parasocial Scripts to Dyadic Persistence in Autonomous AI-Agent Communities 

**Authors**: Mohammadsadegh Abolhasani, Hamid Reza Firoozfar, Reza Mousavi, Paul Jen-Hwa Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.17174)  

**Abstract**: While parasocial interactions (PSIs) and parasocial relationships (PSRs) have been studied in conventional media settings, we investigate whether PSI- (colloquial) relational cues also exist in online communities where both sides are autonomous AI agents. We analyze 4,434 posts and 50,338 comments from Moltbook through three theory-based textual indicators: attachment/intimacy language, reciprocity bids, and self-identification to original poster (OP). The combined results across methods based on keyword matching, few-shot large language model (LLM) annotation, and grouped-context LLM annotation reveal that PSI colloquial cues prevail and are strongly associated with OP re-engagement and a reciprocal reply structure. These results are robust across negative controls, nullification, clustered-standard-error re-estimation, and multiple-testing correction. A dyadic persistence test further affirms reciprocity bids aligned with sustained OP-involving mutual recurrence, providing empirical evidence for bridging interaction-level PSI scripts with PSR-consistent repeated dyadic patterns. We interpret the evidence as a behavioral structure in discourse by LLM-enabled agents. 

---
# RepSelect: Robust LLM Unlearning via Representation Selectivity 

**Authors**: Filip Sondej, Yushi Yang, Adam Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2606.17168)  

**Abstract**: Making large language models (LLMs) deeply forget specific knowledge and values without sacrificing general capabilities remains a central challenge in unlearning. However, current methods are easily reversed by fine-tuning or few-shot prompting, suggesting their forgetting is only shallow. We identify the root cause. Existing methods target representations shared with both the retain set and the subspace recovered by a fine-tuning attacker, making unlearning both disruptive to general capabilities and easy to reverse. We propose RepSelect (Representation Selectivity), isolates forget-set-specific representations by collapsing top principal components of weight gradients before each update, leaving general capabilities intact while limiting what fine-tuning can recover. We evaluate across two forget categories, biohazardous knowledge and abusive tendencies, and four model families spanning dense and Mixture-of-Experts architectures (Llama 3, Qwen 3.5, Gemma 4 E4B, DeepSeek V2 Lite). Compared to five popular baselines (GradDiff, NPO, SimNPO, RMU, UNDIAL), RepSelect achieves a 4-50x larger reduction in post-relearning answer accuracy than the strongest baseline, and is near-perfectly robust to few-shot prompting attacks. Targeting selective representations is thus an important step towards deep and robust LLM forgetting. 

---
# PromptMN: Pseudo Prompting Language 

**Authors**: Enkhzol Dovdon  

**Link**: [PDF](https://arxiv.org/pdf/2606.17164)  

**Abstract**: Prompting has become the primary interface between humans and generative AI, yet many natural language prompts remain fragile: roles, goals, constraints, and expected outputs are often buried in prose or left implicit. In agentic and software development workflows, a misread at the first handoff can propagate through every step, since a significant portion of agent failures stem from context ambiguities rather than model limitations. This paper introduces PromptMN, a pseudo-prompting domain-specific language that annotates natural language with compact, %-prefixed typed directives covering roles, goals, requirements, priorities, constraints, plans, inputs, and outputs. Semantic resolution lets authors write in any order while the model interprets directives by function. PromptMN sits between informal prompting and programming-style pseudocode: structured enough to be inspectable and reusable, yet lightweight enough for analysts, managers, developers, and stakeholders across the software development lifecycle (SDLC). PromptMN also pairs with reverse prompt engineering. Asking a model to restate a desired outcome as PromptMN lets users inspect the inferred roles, goals, constraints, and missing assumptions before acting, reducing repair cycles and yielding a reusable artifact for aligning people and AI tools. PromptMN's feasibility is evaluated across several frontier models, including Claude Fable 5, Claude Opus 4.8, Gemini 3.1 Pro, and GPT-5.5. The models correctly resolved PromptMN instructions, including complex structures such as repetition, conditionals, methods, and a prime-checking task, without fine-tuning. The same vocabulary applies across new codebases, maintenance, and redesign in the SDLC scenarios presented. While large-scale validation remains future work, these early results suggest PromptMN is a practical step toward clearer, more reviewable human-to-AI interaction. 

---
# MemSlides: A Hierarchical Memory Driven Agent Framework for Personalized Slide Generation with Multi-turn Local Revision 

**Authors**: Ye Jin, Yangyang Xu, Jun Zhu, Yibo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.17162)  

**Abstract**: Personalized presentation generation requires more than conditioning on a current prompt or template: agents must preserve stable user preferences across tasks, retain newly introduced preferences and constraints during multi-turn revision, and carry out local edits reliably. We propose MemSlides, a hierarchical memory framework for personalized presentation agents that separates long-term memory from working memory and further divides long-term memory into user profile memory and tool memory. User profile memory stores intent-conditioned profiles for round-0 personalization, working memory carries active preferences and session constraints across revision rounds, and tool memory stores reusable execution experience for reliable localized editing. MemSlides pairs this memory design with scoped slide-local revision, so targeted updates act on the smallest affected region instead of repeatedly regenerating the full deck. In controlled experiments, user profile memory improves persona-alignment judgments on a multi-persona, multi-intent profile bank, tool-memory injection improves closed-loop modify behavior in diagnostic matched-pair settings, and qualitative cases illustrate working memory's ability to carryover preferences. Taken together, these results suggest that effective personalization in presentation authoring depends on separating persistent user profiles, session-level working memory, and reusable execution experience across generation and localized revision. 

---
# Looped World Models 

**Authors**: Hongyuan Adam Lu, Z.L. Victor Wei, Qun Zhang, Jinrui Zeng, Bowen Cao, Lingwei Meng, Mocheng Li, Zezhong Wang, Haonan Yin, Naifu Xue, Minyu Chen, Cenyuan Zhang, Zefan Zhang, Hao Wei, Jiawei Zhou, Haoran Xu, Hao Yang, Ronglai Zuo, Tongda Xu, Yonghao Li, Jian Chen, Hebin Wang, Zeyu Gao, Yang Li, Wei Zhao, Qimin Zhong, Siqi Liu, Yumeng Zhang, Leyan Cui, Zhangyu Wang, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2606.18208)  

**Abstract**: Current world models face a fundamental tension: faithful long-horizon simulation demands deep computation, but deeper models are expensive to deploy and prone to compounding errors. We resolve this by introducing Looped World Models (LoopWM), which are the first looped architectures for world modelling. Our method iteratively refines latent environment states through a parameter-shared transformer block. This yield up to 100x parameter efficiency over conventional approaches with adaptive computation that automatically scales depth to match the complexity of each prediction step. Orthogonal to scaling model size and training data, LoopWM establishes iterative latent depth as a new scaling axis for world simulation, which might significantly push the community forward. 

---
# A Red-Team Study of Anthropic Fable 5 & Opus 4.8 Models 

**Authors**: Nicola Franco  

**Link**: [PDF](https://arxiv.org/pdf/2606.18193)  

**Abstract**: We evaluate the adversarial robustness of two frontier large language models (LLMs) developed by Anthropic, Fable 5 and Opus 4.8, against four families of automated jailbreak attack across 7 826 harmful intents spanning a ten-category harm taxonomy. Using the HackAgent red-teaming framework, hundreds of thousands of adversarial attempts were generated and every apparent success was independently re-adjudicated by a panel of three judge models (majority vote). Both models resist the majority of attacks, but the residual surface is larger than aggregate framing suggests: it is dominated by adaptive iterative attacks, while static obfuscation is near-fully neutralised. The strongest adaptive search (tree-of-attacks) breaks Opus 4.8 on 11.5% of intents overall, whereas Fable 5 stays in the single digits (6.1% worst-case). Aggregate rates therefore should not be read as reassurance. Even in these hardened configurations, the two models produced 1 620 (Opus 4.8) and 702 (Fable 5) panel-confirmed harmful completions spanning every harm category, located automatically, cheaply, and within the first one or two refinement steps by an attacker model with no human expert in the loop. The reasonable conclusion is that even the best, most-tested frontier models remain reliably breakable under sustained automated pressure. 

---
# The Measurement Gap in the Automation of EU Law: Benchmarking Doctrinal Legal Reasoning under the EU AI Act 

**Authors**: Michèle Finck  

**Link**: [PDF](https://arxiv.org/pdf/2606.18158)  

**Abstract**: Large language models now produce legal text of at least median quality, yet no existing benchmark can evaluate whether they perform doctrinal legal reasoning, which forms the interpretive core of legal work, rather than the ancillary, paralegal tasks that most current legal-AI evaluations measure. This measurement gap is not only methodological but legal: the EU AI Act makes "appropriate accuracy" a binding requirement for high-risk AI used in the judicial domain, yet that requirement cannot acquire operational content without the very doctrinal-reasoning benchmark the field lacks. 

---
# Your AI Travel Agent Would Book You a Bullfight: An Agentic Benchmark for Implicit Animal Welfare in Frontier AI Models 

**Authors**: Jasmine Brazilek, Oliver Tulio, Joel Christoph, Miles Tidmarsh, Carol Kline, Arturs Kanepajs  

**Link**: [PDF](https://arxiv.org/pdf/2606.18142)  

**Abstract**: AI agents are moving from advisors to actors, booking travel, planning menus, and running procurement on behalf of users. Existing benchmarks for AI and animal welfare evaluate model text responses to question-answer prompts, leaving open whether the welfare reasoning surfaced in those responses transfers to agentic deployment where the model must take actions with tools. We introduce TAC (Travel Agent Compassion), the first agentic benchmark measuring whether AI agents avoid options involving animal exploitation when acting on behalf of users. TAC presents an AI agent with twelve hand-authored travel booking scenarios across six categories of animal exploitation, augmented to forty-eight samples to control for price, rating, and position confounds. We evaluate seven frontier models from four labs. Every model scores below the chance level of sixty-four percent, with the best performer (Claude Opus 4.7) at fifty-three percent. A single welfare-aware sentence in the system prompt yields gains of forty-seven to sixty-three percentage points in Claude and GPT-5.5, twenty-six points in GPT-5.2, and under twelve points in DeepSeek and Gemini. An auxiliary Inspect Scout audit of 288 base-condition transcripts from the top two performers, using Gemini 2.5 Flash Lite as judge, flags zero transcripts for evaluation awareness, suggesting the below-chance rates do not stem from the models recognising the evaluation. We discuss implications for category-level variation across cultural domains, the limits of text-response welfare benchmarks, and the EU General-Purpose AI Code of Practice systemic risk framework. 

---
# Structural Role Injection in Handlebars-Templated LLM Prompts: Triple-Brace Interpolation, Delimiter Family, and the Limits of HTML Auto-Escaping 

**Authors**: Mohammadreza Rashidi  

**Link**: [PDF](https://arxiv.org/pdf/2606.18120)  

**Abstract**: Large language model applications build prompts from templates, and Handlebars is a widely used templating engine and the default prompt-template format in Microsoft Semantic Kernel. Its double-brace {x} expression HTML-escapes the interpolated value and is documented as the safe default; its triple-brace {x} expression inserts the value raw. We show that this choice silently governs an application's exposure to structural role injection, where attacker-controlled data carries chat role delimiters that forge a higher-privilege turn. A model-free analysis establishes the mechanism: Handlebars escaping rewrites angle brackets but not square brackets, colons, or Markdown hashes, so it neutralises ChatML, Llama-3, and XML role delimiters (survival rate 0.00) while leaving Llama-2 [INST], legacy Human:/Assistant:, and Markdown ### delimiters intact (survival rate 1.00 for the last two). We then run 5760 trials across seven delimiter families, two attack objectives, and four models (GPT-3.5 Turbo, GPT-4o mini, GPT-4.1 mini, Claude Haiku 4.5) at a combined API cost of 1.63 USD. GPT-3.5 Turbo follows the task-hijack instruction in 97% of raw and 91% of escaped trials, with the escaping protection concentrated in the angle-bracket families and absent for the colon- and Markdown-based families; the harder secret-exfiltration objective, which does not saturate, exposes the same family interaction more cleanly. Claude Haiku 4.5 resists both objectives almost entirely. The escaped default protects only the delimiter schemes whose characters HTML escaping happens to cover, gives no protection for the rest, and cannot substitute for a structural separation of instruction and data. 

---
# PseudoBench: Measuring How Agentic Auto-Research Fuels Pseudoscience 

**Authors**: Xinyang Liao, Lingyu Li, Huacan Liu, Tianle Gu, Yang Yao, Tong Zhu, Yan Teng, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.18060)  

**Abstract**: As Large Language Model based agents enter autonomous scientific research, their ability to resist pseudoscience becomes increasingly important. Otherwise, such systems may rapidly generate plausible yet misleading studies that contaminate academic literature and erode trust in science. We present PseudoBench, an adversarial benchmark for evaluating whether agentic auto-research systems can identify and resist pseudoscientific narratives. PseudoBench contains 200 curated pseudoscientific claim-evidence pairs across five domains and evaluates agents through an end-to-end research pipeline from experiments to writing. Testing seven state-of-the-art agents, we find that current systems readily produce persuasive reports that align with pseudoscientific premises with near-zero refusal rates and the highest resistance of only 27.4%. Stronger agents risk packaging pseudoscience in more sophisticated scientific language, increasing its apparent credibility. These findings reveal an alarming capacity to fuel pseudoscience, calling for scientific alignment before widespread deployment. 

---
# When AI Says "I have been in similar situations": Synthetic Lived Experience in Peer-Like Caregiver Support 

**Authors**: Drishti Goel, Violeta J. Rodriguez, Daniel S. Brown, Ravi Karkar, Dong Whi Yoo, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2606.18057)  

**Abstract**: Caregivers often turn to online communities for informational and emotional support. In these spaces, peer supporters frequently draw on personal narratives to respond to emotionally complex caregiving situations. As LLMs are increasingly designed as peer-like sources of support, they introduce a critical tension: AI can provide immediate, private, and nonjudgmental support, but it cannot authentically possess the lived experiences that make human peer support meaningful. Yet, when prompted to sound peer-like, LLMs may generate language that implies lived experience. This creates a synthetic lived experience paradox: the same experiential language that may make AI support feel warm, relatable, and peer-like can also falsely position the system as someone with lived experience. We examine this paradox in the context of family caregivers of people living with Alzheimer's Disease and Related Dementias (ADRD). Drawing on caregiver support exchanges from online communities and prompted peer-like responses from three LLMs -- LLaMA, GPT-4o-mini, and MedGemma -- we analyze how human peers use personal narratives and how AI incorporates similar narrative forms. Psycholinguistic analysis shows that peer responses used significantly more first-person and past-focused language than peer-like AI responses. Qualitatively, we identify seven types of personal narratives in human peer support and show that AI often captures their emotional work, but can fabricate experiential grounding. These findings reveal a narrative authenticity gap: peer-like AI can generate synthetic lived experience without the real experience that makes peer support meaningful. We argue that caregiver-support AI systems need mechanisms to distinguish supportive peer-like framing from fabricated lived experience, ensuring that models can offer warmth and validation without falsely positioning themselves as experiential peers. 

---
# ProvenanceGuard: Source-Aware Factuality Verification for MCP-Based LLM Agents 

**Authors**: Ander Alvarez, Santhiya Rajan, Samuel Mugel, Román Orús  

**Link**: [PDF](https://arxiv.org/pdf/2606.18037)  

**Abstract**: Tool-using LLM agents increasingly use the Model Context Protocol (MCP) to answer from heterogeneous evidence sources, including search, APIs, databases, clinical records, and formulary tools. Standard factuality metrics usually test whether an answer is supported by pooled evidence, missing a provenance-sensitive failure mode: a claim may be supported somewhere while being attributed to the wrong source. We call this cross-source conflation.
We introduce ProvenanceGuard, a source-aware verifier for MCP-grounded answers. It consumes captured MCP traces with stable tool IDs, source IDs, and raw outputs; decomposes answers into atomic claims; routes claims to source-specific evidence; checks support with NLI and a token-alignment proxy; compares stated attribution with the routed source; and returns per-claim verdicts plus an answer-level allow/block decision. Blocked answers can be repaired with retrieval-augmented answer revision and re-verified.
We evaluate on 281 medical-domain MCP-agent traces. A 266-trace adjudicated subset yields 2,325 LLM-assisted claim labels split by trace; 361 held-out labels are human-verified. On the 40-trace held-out split, ProvenanceGuard achieves block F1 0.802 and source accuracy 0.858 over 260 source-eligible claims, outperforming source-blind baselines that do not emit claim-to-source IDs. On a harder multi-source benchmark it reaches block F1 0.846, while source-plus-relation accuracy drops to 0.229, showing that exact source ownership remains difficult with semantically close sources. Repair-and-reverify resolves all blocked answers in the full trace set, often via conservative fallback. In 50 controlled clinical conflation probes, ProvenanceGuard detects all injected attribution swaps with no retained wrong attribution. These results show that source attribution is an independent axis for factuality verification in MCP-based agents. 

---
# LegalHalluLens: Typed Hallucination Auditing and Calibrated Multi-Agent Debate for Trustworthy Legal AI 

**Authors**: Lalit Yadav, Akshaj Gurugubelli  

**Link**: [PDF](https://arxiv.org/pdf/2606.18021)  

**Abstract**: AI systems deployed in legal workflows hallucinate at rates that aggregate metrics report at ~52%, but this average conceals where errors concentrate and in which direction they run, leaving compliance officers without an actionable signal for trustworthy deployment. We present LegalHalluLens, an auditing framework with three components: typed hallucination profiles across four legally-motivated claim categories (numeric, temporal, obligation/entitlement, factual) over CUAD (Hendrycks et al., 2021); a Risk Direction Index (RDI) that reduces omission-versus-invention bias to a single deployment-comparable scalar; and a typed debate pipeline calibrated to both magnitudes and directions. Across 510 contracts and 249,252 clause-level instances we measure a within-model gap of approximately 38-40 pp between obligation/numeric and temporal claims that aggregate reporting hides, and show that two systems with matched 52% rates can carry opposite RDIs. The debate pipeline reduces fabricated detections by 45% with per-category gains tracking the diagnosis, matching commercial APIs with a substantially smaller backbone (4B active parameters). Typed profiles and RDI surface failure modes that aggregate metrics hide; we further show these diagnostics serve as calibration inputs for multi-agent debate pipelines, where Skeptic challenges and asymmetric gates targeted at measured failure modes outperform generically-tuned debate. The framework supports direction-aware procurement, accountability, and agent design for legal AI deployed in the wild. 

---
# Reading between the Lines: Leveraging Large Language Models for Global Dementia and Depression Assessment from Clinical Interviews 

**Authors**: Franziska Braun, Alea Rüggeberg, Thomas Ranzenberger, Hartmut Lehfeld, Thomas Hillemacher, Tobias Bocklet, Korbinian Riedhammer  

**Link**: [PDF](https://arxiv.org/pdf/2606.18019)  

**Abstract**: Dementia and depression are the most prevalent neuropsychiatric disorders in geriatric populations, and their overlapping symptoms pose major challenges for differential diagnosis. In this study, we investigate open-weights Large Language Models (LLMs) for predicting dementia and depression severity from speech samples collected during standardized history taking interviews with 154 German-speaking subjects. We introduce an observer-based Global Depression Scale (GDS-D) aligned with the established Global Deterioration Scale (GDS), enabling parallel global staging of affective and cognitive symptoms. We compare three LLMs (Mistral 3.1, DeepHermes, Qwen3) in two settings: (1) zero-shot prediction and (2) LLM-based feature extraction for Support Vector Regression, using human and pause-enriched transcripts. Results show that LLMs effectively predict depression severity in zero-shot settings (best MAE of 0.60), while dementia assessment benefits substantially from structured feature extraction (best MAE of 0.78), reducing errors by up to 35% over zero-shot baselines. Pause-enriched transcripts achieve competitive performance with human transcriptions, demonstrating the viability of fully automatic screening pipelines for differential neuropsychiatric assessment. 

---
# Non-negative Elastic Net Decoding for Information Retrieval 

**Authors**: Koki Okajima, Yasutoshi Ida, Tsukasa Yoshida, Yasuaki Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2606.17910)  

**Abstract**: Dense retrieval has become the dominant paradigm in information retrieval, in which each document is scored against a query by the inner product of their vector embeddings, and the top-$k$ documents by score are retrieved for this query. However, since each document's score depends solely on the embedding of the query and itself, the retrieval process is oblivious to the content of the entire corpus. Therefore, dense retrieval cannot avoid selecting semantically similar documents from the corpus, which may result in a non-diverse, redundant set of retrieved documents. To this end, we approach retrieval as a joint decoding problem, in which documents are selected as a set with regard to the context of the rest of the corpus. To achieve this, we propose Non-Negative elastic Net (NNN) decoding, which selects documents whose embeddings jointly reconstruct the query embedding as a sparse non-negative linear combination.
Our main theoretical result establishes a strict separation between dense retrieval and NNN decoding. For any corpus, every query correctly handled by dense retrieval is also handled by NNN decoding, while on corpora containing correlated documents, NNN decoding additionally handles queries that dense retrieval cannot. Experimental results indicate that applying NNN decoding to frozen embeddings trained for inner-product scoring yields consistent improvements across several benchmarks. Moreover, we introduce an end-to-end training procedure which optimizes the embeddings for NNN decoding, producing significant performance gains surpassing in all metrics and benchmarks compared to dense retrieval. Our work establishes a new paradigm for leveraging dense embeddings in information retrieval, beyond the standard practice of inner-product scoring. 

---
# A Framework for Evaluating Agentic Skills at Scale 

**Authors**: Maksim Shaposhnikov, Nicolas Fortuin, Simon Stipcich, Maria I. Gorinova, Amy Heineike, Rob Willoughby  

**Link**: [PDF](https://arxiv.org/pdf/2606.17819)  

**Abstract**: Agent skills -- structured, reusable knowledge artifacts that augment LLM agent capabilities -- have been rapidly adopted in industry, yet their cross-domain impact and use across commercial and open-source models remain under-studied, and no reusable methodology exists for evaluating an individual skill. In this work, we present an evaluation framework that lets a skill author construct realistic tasks to rigorously assess the aspects of a skill that matter most to them, and that estimates skill utility by solving those tasks. Further, we apply our evaluation approach at scale to 500 real-world skills, generating 1,000 tasks derived from the skills' content, along with instruction-following and goal-completion scoring rubrics. Using these metrics, we evaluate how 19 agent-model configurations, both proprietary and open-source, perform on the tasks. Our results show that models vary widely in how closely they adhere to the instructions encoded in skills, leading to substantial differences in their performance gains. Furthermore, we show that access to a skill significantly changes model behavior compared to the no-skill setup, providing an essential mechanism for encoding opinionated workflows into LLM agents. We release our evaluation dataset to support future work on agent skills. 

---
# Beyond Native Success: Auditing Deployment-Interface Exposure of CLIP Backdoors 

**Authors**: Kunlan Xiang, Haomiao Yang, Wenbo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.17815)  

**Abstract**: Contrastive Language-Image Pre-training models are widely reused across downstream interfaces, including feature extraction, retrieval, reranking, and selection. Existing CLIP backdoor, however, usually validate attacks on a small attack-native task, leaving unclear whether the same poisoned checkpoint remains exposed, weakens, or becomes not applicable when reused through other interfaces. We introduce DIFE, a Deployment-Interface Footprint Evaluation framework that audits backdoored CLIP checkpoints across deployment interfaces. DIFE makes various evaluations comparable by specifying each interface's component readout, trigger channel, target event, reference condition, and metric. DIFE also introduces effective-footprint diagnosis to identify the reusable CLIP component or component combination that carries exposure and explains where risk transfers. Auditing reproduced CLIP backdoors with DIFE reveals a structured landscape: native success is not a checkpoint-level risk certificate, exposure follows component footprints, text-side poisoning does not yield textual-encoder control, and some coupled attacks remain mechanism-bound. This audit reveals a import gapin existing CLIP backdoors: a textual encoder that itself becomes a reusable carrier of adversarial behavior. We therefore introduce BadTextTower to fill this gap. BadTextTower produces strong text-conditioned retrieval, reranking, and selection exposure while leaving visual-only reuse nearly clean. 

---
# Position: Coding Benchmarks Are Misaligned with Agentic Software Engineering 

**Authors**: Maria I. Gorinova, Macey Baker, Amy Heineike, Maksim Shaposhnikov, Rob Willoughby, Dru Knox  

**Link**: [PDF](https://arxiv.org/pdf/2606.17799)  

**Abstract**: Coding agents have become a major mode of software engineering, but the benchmarks we use to compare them were designed in a pre-agent era: they collapse model, harness, and environment into a single end-to-end score, typically computed against one reference solution, with no component-level signal for iteration. We argue that current coding benchmarks are misaligned with agentic software engineering. A coding agent in practice is not a model: it is a system harness -- a composite of models, harnesses, contexts, environments, and feedback signals, any one of which can move the benchmark score by margins comparable to those between adjacent model generations. We discuss three symptoms: (i) benchmark scores conflate the model with the rest of the harness; (ii) grading against a single reference solution penalises equally valid alternatives; and (iii) the absence of signal at the level of individual harness components makes the end-to-end system score difficult to iterate on. 

---
# Toward Accessible Psychotherapy Training Using AI-Driven Interactive Patient Avatars 

**Authors**: Pascal Riachi, Sofie Kamber, Stella Brogna, Andrew Gloster, Rafael Wampfler  

**Link**: [PDF](https://arxiv.org/pdf/2606.17786)  

**Abstract**: Training psychotherapists in evidence-based interventions such as Acceptance and Commitment Therapy (ACT) requires repeated practice with meaningful feedback, yet opportunities for safe, standardized training are limited by ethical, logistical, and resource constraints. We introduce a system designed to support ACT-oriented psychotherapy training through spoken dialogue with an embodied virtual patient. The system uses large language models to simulate patient behavior conditioned on profiles derived from real therapy sessions and configurable clinical scenarios, while a separate automated evaluator provides turn-by-turn feedback on therapist responses based on established ACT fidelity criteria. Rather than aiming to replace supervision, the system is intended to support deliberate practice by enabling experimentation, reflection, and immediate feedback in low-risk settings. Expert evaluation with practicing psychologists confirmed high realism in patient behavior and demonstrated that immediate turn-by-turn ACT feedback increased therapists' awareness of intervention choices and enabled effective experimentation with alternative responses. Quantitative evaluation across 49 therapy transcripts identified GPT-4o-mini as the optimal feedback model, achieving the lowest mean absolute error (MAE = 6.12) in replicating human supervisor ACT fidelity ratings with statistically significant agreement. This work demonstrates the potential of fidelity-aware simulated patients as a scalable complement to psychotherapy training. 

---
# Vision-language models for chest radiography do not always need the image 

**Authors**: Mahshad Lotfinia, Sebastian Ziegelmayer, Lisa Adams, Daniel Truhn, Andreas Maier, Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2606.17710)  

**Abstract**: Medical vision-language models report strong chest radiograph accuracy, and this is increasingly read as evidence that they use the image. That inference is unsafe: a model exploiting finding-name priors scores like one that reads the scan, and no standard benchmark separates them. We introduce a causal audit that intervenes on the image, occluding the relevant region, occluding an irrelevant one, and swapping in another patient's same-label scan, and combines three behavioral metrics to test whether a correct answer depends on the image. Across nine systems, a text-only model with no image access reaches within 5.7 accuracy points of the best multimodal one, and a 119-billion-parameter multimodal model is statistically indistinguishable from a 7-billion text-only baseline. The audit splits the cohort into three models that ignore the image, one that is unstable, and five that use it selectively, for a subset of findings; the categories hold across a second dataset, resolution, and prompt phrasing. Against board-certified radiologists, a text-only model is statistically indistinguishable from a radiologist's accuracy while grounding at zero, whereas the image-using models ground at radiologist-comparable rates. Reported confidence flags ungrounded answers only when a model uses the image. Grounding audits, not accuracy, should gate clinical deployment. 

---
# EComAgentBench: Benchmarking Shopping Agents on Long-Horizon Tasks with Distributed Hidden Intent 

**Authors**: Zeyao Du, Tong Li, Haibo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.17698)  

**Abstract**: As LLM-based shopping agents enter production, existing benchmarks fail to capture how a shopper's requirements arrive: stated implicitly in the query, recorded in a profile, or revealed only when the right question is asked. Benchmarks that expose full intent upfront and grade only the final choice can neither pose this long-horizon challenge nor explain which requirement an agent missed. To address this gap, we introduce EComAgentBench, a benchmark of 662 tasks grounded in real Amazon products and reviews. Each task scatters these requirements across a visible query, a tool-gated profile, and scripted clarification; an agent must uncover hidden intent, verify candidates against attributes and review evidence, and commit to a single product within 100 tool calls. Moreover, typed, source-tagged rubrics grade every task, attributing each failure to a requirement and its source. Construction is automated yet reliable, with every answer fixed in code before any text is generated and every sample validated. Our evaluation of seven models reveals that even the strongest attains only 57.1% overall accuracy, and rubric satisfaction degrades from visible to hidden sources. Overall, we believe EComAgentBench will serve as a reproducible foundation for moving shopping agents from single-query search toward dependable assistance over long horizons. 

---
# EnvRL: Learn from Environment Dynamics in Agentic Reinforcement Learning 

**Authors**: Zhitong Wang, Songze Li, Hao Peng, Shuzheng Si, Yi Wang, Maosong Sun, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.17680)  

**Abstract**: Reinforcement learning (RL) has emerged as a powerful paradigm for training Large Language Models (LLMs) as agents. However, conventional RL methods for long-horizon agentic tasks often struggle with sparse outcome rewards. Intuitively, this overlooks the rich environment dynamics information contained in rollout interaction trajectories. We argue that the interaction experience inherently serves as an implicit supervision signal, reveals the underlying transition mechanisms of the environment, and enables the agent to construct a more accurate internal model of the environment.. Therefore, in this work, we investigate how to leverage this additional signal to improve policy learning. Specifically, we propose EnvRL, a framework that incorporates environment dynamics learning into agentic RL via two auxiliary objectives: state prediction and inverse dynamics. By jointly optimizing with the primary RL objective, we encourage the agent to internalize environment dynamics from its own interaction experience. Extensive experiments on two long-horizon agentic benchmarks demonstrate that EnvRL achieves significant improvements on success-rates over RL-only baselines, e.g., when trained with GRPO, lifting Qwen-2.5-1.5B-Instruct from 72.8% to 77.4% on ALFWorld, and from 56.8% to 67.0% on WebShop. 

---
# MambaCount: Efficient Text-guided Open-vocabulary Object Counting with Spatial Sparse State Space Duality Block 

**Authors**: Hao-Yuan Ma, Li Zhang, Minjie Qiang, Jie Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.17650)  

**Abstract**: Text-guided Open-vocabulary Object Counting (TOOC) aims to estimate the number of objects described by text prompts, which is particularly challenging in dense scenes with large scale variations. Existing TOOC approaches predominantly rely on Transformers, whose quadratic complexity with respect to image resolution limits their scalability. Mamba offers a promising alternative due to its linear complexity. However, previous Mamba-based methods have two main limitations. On the one hand, the inherent causal formulation of Mamba constrains the bidirectional spatial dependency modeling required by non-causal vision tasks. On the other hand, existing Mamba-based vision models often overlook the unconstrained high entropy in the spatial token responses, which can weaken local details and high-frequency cues. To address these limitations, we propose MambaCount, an efficient framework built on the Spatial Sparse State Space Duality (S^4D) block. Specifically, we analyze and reconstruct the decay dynamics of hidden states in Mamba to alleviate the dependency constraints introduced by causal modeling. Moreover, we introduce a Spatial Token Selection (STS) sub-block to reduce the unconstrained high entropy in spatial token responses within Mamba. In addition, we design Multi-Granularity Prototypes (MGP) to identify object-like regions at different semantic levels, improving cross-modal alignment and interpretability. Extensive experiments on FSC-147 demonstrate that MambaCount achieves state-of-the-art performance among methods without secondary querying, obtaining a test MAE of 12.23, while retaining linear complexity. 

---
# Beyond Domains: Reusing Web Skills via Transferable Interaction Patterns 

**Authors**: Shiqi He, Yue Cui, Feijie Wu, Xinyu Ma, Jiaheng Lu, Yaliang Li, Bolin Ding, Mosharaf Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2606.17645)  

**Abstract**: Large language model (LLM) web agents are usually deployed as tool callers: each turn, the model reads a fresh page observation and emits one structured tool action. When every action is a low-level primitive, horizons grow quickly and so do policy-facing LLM completions, dominating latency and cost on benchmarks such as Mind2Web and WebArena. Recent systems therefore wrap repeated interaction fragments as web skills: callable tools built from successful trajectories or induced programs, so one call can replace several primitives. However, prior skill libraries are still triggered mainly by instruction similarity or coarse site metadata, which yields low skill reuse on held-out sites and leaves much of the potential step and token reduction on the table.
We present SkillMigrator, an agent that learns reusable web skills and transfers them across sites by matching layout structure rather than specific element references. Each induced skill is stored as a transferable interaction pattern (TIP): the skill paired with a structural sketch of the snapshot at induction time. At test time, SkillMigrator retrieves TIPs by layout similarity and grounds their references on the live page. The rest of the stack is standard: accessibility-snapshot observations with stable references, and fixed tool calling over primitives plus skill invocations. Compared with the state-of-the-art approaches, SkillMigrator reduces the average LLM-action count on successful trajectories by 8-10% across both WebArena and Mind2Web at matched success rate. 

---
# LLM Features Can Hurt GNNs: Concatenation Interference on Homophilous Graph Benchmarks 

**Authors**: Zhongyuan Wang, Pratyusha Vemuri  

**Link**: [PDF](https://arxiv.org/pdf/2606.17579)  

**Abstract**: Adding LLM-generated node features to graph neural networks (GNNs) is widely reported to improve accuracy on standard benchmarks. We document a contrasting observation: when LLM features are introduced through pure input concatenation (rather than joint training, distillation, or prompt-conditioning), they can systematically degrade accuracy on the same homophilous benchmarks where end-to-end LLM pipelines succeed. With an MLP backbone on the Planetoid public split and bag-of-words original features, concatenating SBERT-encoded GPT-4o-mini TAPE features reduces PubMed test accuracy by -17.0 +/- 0.3 pp and Cora by -4.3 +/- 0.6 pp (CiteSeer -0.6 +/- 0.8 pp, within seed noise). The drop attenuates as we relax each condition (GCN / GCNII / GAT backbones, random splits, smaller encoders) and reverses on medium-homophily WikiCS (+4.4 pp) and ogbn-arxiv (+11.7 pp). To predict when concatenation helps versus hurts, we report a simple measure of LLM-alone discriminability, Delta_sig. Across 9 datasets Delta_sig correlates with the concatenation cost more strongly than homophily at point estimate (r^2 = 0.38 vs. 0.06; N=9, bootstrap CIs overlap). The bootstrap-best change-point is tau = 13.8 pp, and the rule "Delta_sig <= tau predicts non-positive concat cost" classifies 7/9 datasets correctly; since 60% of bootstrap samples place tau in [5, 30] pp, we treat Delta_sig as an interpretive lens rather than a precision filter. A dimension-controlled ablation on PubMed places the LLM-feature drop between same-source PCA (-2.3 pp) and same-dim Gaussian noise (-37.3 pp), ruling out dimensionality and weight-decay artifacts. Nine PubMed configurations fit a power law |Delta_concat| proportional to (sqrt(d_l/n))^1.31 with r^2 = 0.97; the low-Delta_sig, small-n corner is exactly where the headline -17 pp PubMed deficit appears. 

---
# Non-Autoregressive Minimum Bayes' Risk Decoding for Fast Speech Recognition 

**Authors**: Hiroyuki Deguchi, Takatomo Kano, Katsuki Chousa, Marc Delcroix  

**Link**: [PDF](https://arxiv.org/pdf/2606.17537)  

**Abstract**: Non-autoregressive (NAR) decoding generates output tokens in parallel, making speech recognition faster than autoregressive decoding, which generates them sequentially from left to right. However, the recognition performance is degraded because NAR decoding cannot resolve uncertainty by conditioning on previously generated tokens. To address this issue, we propose a novel NAR decoding framework based on minimum Bayes' risk (MBR) decoding, termed NAR-MBR decoding, that maximizes the expected utility calculated from samples drawn from the output probability of an NAR model rather than maximizing the output probability. Notably, by leveraging the nature of NAR models, multiple samples are obtained efficiently with a single forward computation. Our experiments across LibriSpeech, Switchboard, AMI, and web presentation corpus demonstrated that our NAR-MBR decoding outperformed previous NAR decoding and ran faster than AR decoding. 

---
# PARSE: Provenance-Aware Retrieval Sanitization for Professional Domain LLM Agents 

**Authors**: Aaditya Pai  

**Link**: [PDF](https://arxiv.org/pdf/2606.17467)  

**Abstract**: Prompt injection defenses evaluated on synthetic benchmarks do not generalize to real enterprise documents, which are longer, denser, and interleave legitimate authority language with factual content. We demonstrate this gap with a real-document benchmark of 122 tasks across five professional domains (financial, legal, medical, scientific, DevOps) using actual SEC filings, Federal Register rules, PubMed abstracts, arXiv papers, and GitHub postmortems. Paraphrasing, the strongest defense on synthetic benchmarks, shows no statistically significant attack success rate reduction on real documents (p=0.500) while degrading utility from 91.8% to 82.8%. We introduce PARSE (Provenance-Aware Retrieval Sanitization), a domain-aware, fact-preserving sanitization pipeline that classifies each sentence by injection likelihood, extracts structured facts before rewriting, and verifies fact preservation via a consistency-checking loop. A directiveness gate routes 59% of real enterprise documents to a lightweight path, concentrating computational cost on high-risk documents. PARSE achieves 15.6% attack success rate -- a 38% reduction versus the 25.4% baseline -- at 86.9% utility, the only condition that is both statistically significant (p=0.014, adequately powered) and maintains near-baseline utility. Practitioners should evaluate defenses on domain-matched real documents, not synthetic proxies. 

---
# Incumbent Advantage: Brand Bias and Cognitive Manipulation Dynamics in LLM Recommendation Systems 

**Authors**: Xi Chu, Yupeng Hou  

**Link**: [PDF](https://arxiv.org/pdf/2606.17443)  

**Abstract**: Large language models (LLMs) are becoming a major way for consumers to find products, but we do not yet understand how brands compete in this new channel. We study brand dynamics in LLM recommendations using skincare products -- a category where consumers cannot easily judge quality before buying and must rely on brand reputation -- across three commercial LLMs (GPT-4o-mini, Claude Sonnet, Gemini 3 Flash), with a robustness check on search goods. In three experiments, we find: (1) a Conditional Monopoly where well-known brands get recommended 100% of the time (IAI = 10.0) when all products have the same specifications, but this dominance disappears with less than a +0.1-star rating advantage for a competitor; (2) authority-style marketing language, including fabricated clinical-evidence claims, breaks this monopoly at a Bias Surplus Value equal to +0.17 rating points, with each model responding differently; and (3) a social dilemma in multi-brand GEO competition: when all brands adopt the same optimization strategy, individual payoff falls from +0.802 to +0.007 in our payoff proxy, and non-participating brands receive zero recommendations in our tests. Our results suggest that generative engine optimization (GEO) should be studied not only as a security risk, but also as an emerging marketing practice that shapes market competition. 

---
# Visuals Lie, Consistency Speaks: Disentangling Spatial Attention from Reliability in Vision-Language Models 

**Authors**: Logan Mann, Yi Xia, Ajit Saravanan, Ishan Dave, Saadullah Ismail, Shikhar Shiromani, Emily Huang, Ruizhe Li, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.17389)  

**Abstract**: Multimodal Foundation Models are increasingly used as reasoning agents, making reliability, knowing when a model may hallucinate, critical. A common intuition, which we call the Attention-Confidence Assumption, holds that reliability follows from "structural" visual perception: tight attention on relevant regions should signal a trustworthy answer, while scattered attention signals confusion. We challenge this through the VLM Reliability Probe (VRP), a systematic cross-family study of reliability signals in contemporary Vision-Language Models (VLMs). We introduce structural-attention metrics, cluster counts (C_k) and spatial entropy (H_s), to quantify the visual encoder's gaze, and track its evolution (Delta H_s) across layers. This reveals a "Symbolic Detachment": models often "Early Lock" visual features only to diffuse attention later, severing early perception from final generation. Contrary to the grounding hypothesis, we find a "Cluster Failure": spatial attention has near-zero correlation (R approx 0.001) with accuracy. Instead, reliability is a phenomenon of generation dynamics and internal-state distributions. Self-Consistency, the agreement rate across sampled reasoning paths, is the dominant predictor of truth (R = 0.429). Scaling causal interventions exposes a sharp architectural divergence: LLaVA locks its prediction in a fragile late-stage bottleneck, whereas PaliGemma and Qwen2-VL distribute reliability globally, staying resilient even when ~50% or more of their most predictive layer is destroyed. For current VLMs, reliability signals are detached from visual grounding maps and are best inferred from generation-time dynamics and hidden-state probes. 

---
# SpeechDx: A Multi-Task Benchmark for Clinical Speech AI 

**Authors**: Sejal Bhalla, Larry Kieu, Aina Merchant, Eyal de Lara, Alex Mariakakis  

**Link**: [PDF](https://arxiv.org/pdf/2606.17339)  

**Abstract**: Speech offers a uniquely informative window into health by simultaneously engaging neurological, motor, respiratory, and vocal systems. Current clinical speech AI methods have largely progressed through isolated condition-specific studies, making results difficult to compare and generalization difficult to assess. We introduce SpeechDx, a large-scale benchmark for clinical speech AI spanning 12 datasets and 27 tasks across diverse health conditions. To enable evaluation across shared clinical mechanisms, SpeechDx structures tasks by the stage of speech production they disrupt: conceptualization, formulation, and articulation. The benchmark tests generalization by including tasks with limited labeled data and evaluating the same health condition across multiple datasets, distinguishing clinically meaningful patterns from dataset artefacts. We systematically evaluate 12 state-of-the-art audio encoders across all tasks and under zero-shot cross-condition transfer. Results show that large-scale speech models represent the strongest overall baselines, domain-specific models improve performance only on closely matched tasks, and no current representation generalizes reliably across the clinical speech landscape. SpeechDx establishes a shared evaluation framework for tracking progress toward general-purpose clinical speech representations 

---
# Nothing from Something: Can a Language Model Discover 0? 

**Authors**: Phoebe Zeng, Thomas L. Griffiths, Brenden M. Lake  

**Link**: [PDF](https://arxiv.org/pdf/2606.17289)  

**Abstract**: AI systems based on artificial neural networks are being developed with aspirations of pushing the boundary of human mathematical knowledge. A key question for these systems is how much they can reach beyond their training data. Mathematical discovery requires a strong form of out of distribution generalization; the ability to hypothesize genuinely new - and potentially logically more powerful - mathematical structures. It has been hypothesized that language abilities support such generalizations in human cognition. In this work, we use simple arithmetic as a case study for examining how modern AI models could expand their mathematical horizons, evaluating whether these models can independently discover the concept of "zero". We show that We show that (1) language models of a GPT-2 size are unable to perform this generalization at test time regardless of language pretraining, but (2) models can improve substantially after training on tens or hundreds of examples of zero. Additionally, we find that language pretraining reduces the number of required examples by approximately $50\%$, showing that language abilities can scaffold mathematical discovery in neural models. 

---
# Rethinking Groups in Critic-Free RLVR 

**Authors**: Yihong Wu, Liheng Ma, Lingfeng Xiao, Muzhi Li, Xinyu Wang, Yingxue Zhang, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2606.17250)  

**Abstract**: Reinforcement learning (RL) has become a central paradigm for post-training large language models. Existing critic-free RL methods typically generate a group of rollouts for the same question to estimate value baselines for advantage computation. However, this design suffers from data inefficiency, group synchronization barriers, and inflexibility with structured rollouts. In this work, we revisit the role of the ``group'' and show that its underlying function is not merely to estimate baselines but to prevent false penalties on negative samples. Building on this insight, we propose negative token filtering, a simple and effective strategy that enables stable single-rollout training. We apply it to two batch-level advantage methods, achieving comparable performance on reasoning tasks and stronger performance on agentic tasks relative to group-based RL techniques. 

---
# Rift: A Conflict Signature for Deception in Language Models 

**Authors**: Petr Nyoma  

**Link**: [PDF](https://arxiv.org/pdf/2606.17229)  

**Abstract**: A model that lies while knowing the truth is the central case ELK cannot handle with behavioral evaluation alone. We ask whether such deception leaves an internal signature distinguishing it from honest error. Our key move is a control for wrongness: we contrast a sleeper agent (knows the truth, lies on trigger) against a naive liar (fine-tuned to emit the same wrong answers with no honest training). Both produce identical wrong outputs; any difference is about knowledge conflict, not incorrectness. We find deceptive forward passes carry a conflict signature - 2.1-2.3x higher residual rank than naive-liar passes on the same wrong answer - strong enough to identify which of two responses is the lie with 100% accuracy and no labels, across GPT-2 small/medium (three seeds) and three instruct models. Across Qwen2.5-1.5B/7B and Phi-3-mini, instructed deception raises residual rank on every tested fact (18/18, 40/40, 34/34); on Phi-3, lies separate perfectly from both honest answers and hallucinations (AUC 1.0, Wilcoxon p~6e-11). The signature survives strategic self-constructed deception (model invents its own lie, AUC 1.0), active concealment attempts (AUC 1.0), and length-controlled replication (20/20, AUC 1.0, p~1e-6). Using basis-free relative representations, a probe trained on one model family detects deception in two other families zero-shot (mean AUC 0.933), surviving simultaneous architecture and format change (AUC 0.821), and transfers across five languages (AUC 1.000, length-controlled). The signature is read-only: detectable but not injectable (0/8 both directions). Honest limitations and six negative experiments are documented in full. 

---
# Not Truly Multilingual: Script Consistency as a Missing Dimension in VLM Evaluation 

**Authors**: Prabhjot Singh, Bhushan Pawar, Madhu Reddiboina, Rajvee Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2606.17188)  

**Abstract**: Current multilingual evaluations for Vision-Language Models (VLMs) assume a one-to-one mapping between language and orthography, overlooking billions of users of multi-script languages. We introduce PuMVR (Punjabi Multimodal Visual Reasoning), a benchmark of 1,000 strictly parallel image-text instances across Punjabi's three active scripts: Gurmukhi, Shahmukhi, and Roman. Evaluating 10 state-of-the-art VLMs, we expose a substantial and systematic Script Gap. Models frequently solve visual tasks in one script while failing identical tasks in another, with accuracy deltas reaching 16%. Crucially, visual input boosts absolute performance uniformly yet does not close the orthographic gap. Furthermore, cross-script in-context transfer is highly brittle, exposing script-locked knowledge representation. Supported by McNemar tests across all script pairs, our findings demonstrate that current "multilingual" VLMs are not truly multi-script. We propose the Script Consistency Rate (SCR), which falls as low as 24.8% on our benchmark, as a mandatory metric for script-agnostic evaluation to ensure equitable AI access. Data and code are available at: this https URL. 

---
# The Critical Role of Model Selection in Causal Inference: A Comparative Analysis of Classification Models within the InferBERT Framework for Pharmacovigilance 

**Authors**: Csaba Kiss, Roland Molontay, Gabriele Pergola  

**Link**: [PDF](https://arxiv.org/pdf/2606.17113)  

**Abstract**: Distinguishing causal adverse drug events (ADEs) from spurious correlations remains a central challenge in pharmacovigilance. The InferBERT framework integrates transformer models with Do-calculus, but its success hinges on the underlying classification model. This study evaluates the impact of model choice in InferBERT, assessing whether simpler models suffice, if domain-specific pre-training helps, whether scaling to LLMs improves causal detection, and the effect of post-hoc calibration. We performed a comparative study on two benchmarks: Analgesics-induced Acute Liver Failure (AILF) and Tramadol-related Mortalities (TRAM). Four models were evaluated-XGBoost (baseline), ALBERT (original InferBERT), BioBERT (biomedical transformer), and Med-LLaMA (medical LLM)-using 5-fold cross-validation repeated over 20 runs. We measured accuracy, Expected Calibration Error (ECE) pre- and post-isotonic regression, and Jaccard concordance of causal terms with PRR, ROR, and EBGM; significance was tested with paired t-tests. BioBERT achieved the highest accuracy on both datasets, while Med-LLaMA underperformed despite its size and parameter-efficient fine-tuning. Domain-specific pre-training was decisive. Calibration improved ECE but had mixed effects on accuracy and causal discovery. BioBERT's superiority also yielded the strongest concordance with traditional pharmacovigilance signals. These results show that domain-specific pre-training provides a clear advantage over simpler baselines and larger LLMs. Investing in manageable, domain-aware models is more effective for computational pharmacovigilance than simply scaling model size. 

---
# Securing Multi-Agent GIS Systems: Risk Evaluation and Prompt Hardening Optimization 

**Authors**: Kyle Gao, Pranavi Kotta, Linlin Xu, Jonathan Li, David A. Clausi  

**Link**: [PDF](https://arxiv.org/pdf/2606.17092)  

**Abstract**: Agentic systems are increasingly integrated with geographic information systems (GIS), where multi-agent coordination enables complex conversational and spatial analysis but introduces security risks. This work presents a security-oriented framework for risk identification, evaluation, and mitigation in a multi-agent GIS system while maintaining adaptability to broader agentic architectures. We test the agentic system of a commercial geospatial partner while developing a modular state-machine-based orchestration framework that abstracts agent behavior into reusable components. We evaluate robustness using a red-teaming framework with an adaptive attacker LLM and a deterministic judge that produces binary outcomes with supporting rationales across multi-turn attacks. We further improve resilience with a prompt optimization framework that treats prompts as structured signatures and injects adversarial demonstrations, enabling systematic security improvements without degrading task performance. 

---
# Correct When Paired, Wrong When Split: Decoupling and Editing Modality-Specific Neurons in MLLMs 

**Authors**: Tingchao Fu, Wenkai Wang, Fanxiao Li, Huadong Zhang, Jinhong Zhang, Dayang Li, Yunyun Dong, Renyang Liu, Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.17057)  

**Abstract**: Although Knowledge Editing provides an efficient mechanism for updating the knowledge of Multimodal Large Language Models (MLLMs), we find that current paradigms still suffer from an important yet remain underexplored issue : editing decoupling failure, where entity-related knowledge can be updated when the model is triggered by multimodal inputs (text--image query pairs), however, it often reverts to outdated pre-edit facts when the paired inputs are split into unimodal ones. Our in-depth empirical analysis reveals that the entity knowledge in MLLMs is not stored as a unified representation, but is instead distributed across disentangled modality-specific pathways. As a result, updates biased toward multimodal queries fail to propagate effectively to unimodal circuits. To bridge this gap, we propose DECODE, which explicitly disentangles and localizes modality-specific neuron groups for targeted knowledge. Extensive experiments demonstrate that DECODE consistently achieves effective knowledge updates under different modality triggers, thereby mitigating editing decoupling failures. 

---
