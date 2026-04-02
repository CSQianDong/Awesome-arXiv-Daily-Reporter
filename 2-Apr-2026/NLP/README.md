# Universal YOCO for Efficient Depth Scaling 

**Authors**: Yutao Sun, Li Dong, Tianzhu Ye, Shaohan Huang, Jianyong Wang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.01220)  

**Abstract**: The rise of test-time scaling has remarkably boosted the reasoning and agentic proficiency of Large Language Models (LLMs). Yet, standard Transformers struggle to scale inference-time compute efficiently, as conventional looping strategies suffer from high computational overhead and a KV cache that inflates alongside model depth. We present Universal YOCO (YOCO-U), which combines the YOCO decoder-decoder architecture with recursive computation to achieve a synergistic effect greater than either alone. Built on the YOCO framework, YOCO-U implements a Universal Self-Decoder that performs multiple iterations via parameter sharing, while confining the iterative process to shallow, efficient-attention layers. This combination yields a favorable capability-efficiency tradeoff that neither YOCO nor recursion achieves independently. The YOCO architecture provides a constant global KV cache and linear pre-filling, while partial recursion enhances representational depth with limited overhead. Together, YOCO-U improves token utility and scaling behavior while maintaining efficient inference. Empirical results confirm that YOCO-U remains highly competitive in general and long-context benchmarks, demonstrating that the integration of efficient-attention architectures and recursive computation is a promising direction for scalable LLMs. 

---
# $\texttt{YC-Bench}$: Benchmarking AI Agents for Long-Term Planning and Consistent Execution 

**Authors**: Muyu He, Adit Jain, Anand Kumar, Vincent Tu, Soumyadeep Bakshi, Sachin Patro, Nazneen Rajani  

**Link**: [PDF](https://arxiv.org/pdf/2604.01212)  

**Abstract**: As LLM agents tackle increasingly complex tasks, a critical question is whether they can maintain strategic coherence over long horizons: planning under uncertainty, learning from delayed feedback, and adapting when early mistakes compound. We introduce $\texttt{YC-Bench}$, a benchmark that evaluates these capabilities by tasking an agent with running a simulated startup over a one-year horizon spanning hundreds of turns. The agent must manage employees, select task contracts, and maintain profitability in a partially observable environment where adversarial clients and growing payroll create compounding consequences for poor decisions. We evaluate 12 models, both proprietary and open source, across 3 seeds each. Only three models consistently surpass the starting capital of \$200K, with Claude Opus 4.6 achieving the highest average final funds at \$1.27 M, followed by GLM-5 at \$1.21 M at 11$\times$ lower inference cost. Scratchpad usage, the sole mechanism for persisting information across context truncation, is the strongest predictor of success, and adversarial client detection is the primary failure mode, accounting for $47\%$ of bankruptcies. Our analysis reveals that frontier models still fail through distinct failure modes such as over-parallelization, demonstrating the capability gaps for long-horizon performance. $\texttt{YC-Bench}$ is open-source, reproducible, and configurable. 

---
# LLM REgression with a Latent Iterative State Head 

**Authors**: Yiheng Su, Matthew Lease  

**Link**: [PDF](https://arxiv.org/pdf/2604.01206)  

**Abstract**: We present RELISH (REgression with a Latent Iterative State Head), a novel, lightweight architecture designed for text regression with large language models. Rather than decoding numeric targets as text or aggregating multiple generated outputs, RELISH predicts scalar values directly from frozen LLM representations by iteratively refining a learned latent state through cross-attention over token-level representations, and then mapping the final state to a point estimate with a linear regressor. Across five datasets, four LLM backbones, and two LLM training regimes, RELISH consistently outperforms prior baselines from all three major LLM regression families, including autoregressive decoding, regression-aware inference, and existing predictive head methods. Despite these gains, RELISH remains highly parameter-efficient, requiring only 3.4-3.7M trainable parameters across frozen LLM backbones (only 0.01-0.04% additional overhead), far less than LoRA-based alternatives that grow with model size (0.26-0.42%). 

---
# ORBIT: Scalable and Verifiable Data Generation for Search Agents on a Tight Budget 

**Authors**: Nandan Thakur, Zijian Chen, Xueguang Ma, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.01195)  

**Abstract**: Search agents, which integrate language models (LMs) with web search, are becoming crucial for answering complex user queries. Constructing training datasets for deep research tasks, involving multi-step retrieval and reasoning, remains challenging due to expensive human annotation, or cumbersome prerequisites. In this work, we introduce ORBIT, a training dataset with 20K reasoning-intensive queries with short verifiable answers, generated using a frugal framework without relying on paid API services. The modular framework relies on four stages: seed creation, question--answer pair generation, and two stages of verification: self and external. ORBIT spans 15 domains and each training pair requires 4--5 reasoning steps, with external search verification required from the complete web. We train Qwen3-4B as the base model on ORBIT using GRPO and evaluate it on Wikipedia question answering tasks. Extensive experiment results demonstrate that ORBIT-4B achieves strong performance among sub-4B LLMs as search agents, proving the utility of synthetic datasets. Our framework, code and datasets are open-sourced and available publicly. 

---
# Embarrassingly Simple Self-Distillation Improves Code Generation 

**Authors**: Ruixiang Zhang, Richard He Bai, Huangjie Zheng, Navdeep Jaitly, Ronan Collobert, Yizhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.01193)  

**Abstract**: Can a large language model (LLM) improve at code generation using only its own raw outputs, without a verifier, a teacher model, or reinforcement learning? We answer in the affirmative with simple self-distillation (SSD): sample solutions from the model with certain temperature and truncation configurations, then fine-tune on those samples with standard supervised fine-tuning. SSD improves Qwen3-30B-Instruct from 42.4% to 55.3% pass@1 on LiveCodeBench v6, with gains concentrating on harder problems, and it generalizes across Qwen and Llama models at 4B, 8B, and 30B scale, including both instruct and thinking variants. To understand why such a simple method can work, we trace these gains to a precision-exploration conflict in LLM decoding and show that SSD reshapes token distributions in a context-dependent way, suppressing distractor tails where precision matters while preserving useful diversity where exploration matters. Taken together, SSD offers a complementary post-training direction for improving LLM code generation. 

---
# S0 Tuning: Zero-Overhead Adaptation of Hybrid Recurrent-Attention Models 

**Authors**: Jack Young  

**Link**: [PDF](https://arxiv.org/pdf/2604.01168)  

**Abstract**: Using roughly 48 execution-verified HumanEval training solutions, tuning a single initial state matrix per recurrent layer, with zero inference overhead, outperforms LoRA by +10.8 pp (p < 0.001) on HumanEval. The method, which we call S0 tuning, optimizes one state matrix per recurrent layer while freezing all model weights. On Qwen3.5-4B (GatedDeltaNet hybrid), S0 tuning improves greedy pass@1 by +23.6 +/- 1.7 pp (10 seeds). On FalconH1-7B (Mamba-2 hybrid), S0 reaches 71.8% +/- 1.3 and LoRA reaches 71.4% +/- 2.4 (3 seeds), statistically indistinguishable at this sample size while requiring no weight merging. Cross-domain transfer is significant on MATH-500 (+4.8 pp, p = 0.00002, 8 seeds) and GSM8K (+2.8 pp, p = 0.0003, 10 seeds); a text-to-SQL benchmark (Spider) shows no transfer, consistent with the trajectory-steering mechanism. A prefix-tuning control on a pure Transformer (Qwen2.5-3B) degrades performance by -13.9 pp under all nine configurations tested. On Qwen3.5, a per-step state-offset variant reaches +27.1 pp, above both S0 and LoRA but with per-step inference cost. Taken together, the results show that recurrent state initialization is a strong zero-inference-overhead PEFT surface for hybrid language models when verified supervision is scarce. The tuned state is a ~48 MB file; task switching requires no weight merging or model reload. Code and library: this https URL. 

---
# Brainstacks: Cross-Domain Cognitive Capabilities via Frozen MoE-LoRA Stacks for Continual LLM Learning 

**Authors**: Mohammad R. Abu Ayyash  

**Link**: [PDF](https://arxiv.org/pdf/2604.01152)  

**Abstract**: We present Brainstacks, a modular architecture for continual multi-domain fine-tuning of large language models that packages domain expertise as frozen adapter stacks composing additively on a shared frozen base at inference. Five interlocking components: (1) MoE-LoRA with Shazeer-style noisy top-2 routing across all seven transformer projections under QLoRA 4-bit quantization with rsLoRA scaling; (2) an inner loop performing residual boosting by freezing trained stacks and adding new ones; (3) an outer loop training sequential domain-specific stacks with curriculum-ordered dependencies; (4) null-space projection via randomized SVD constraining new stacks to subspaces orthogonal to prior directions, achieving zero forgetting in isolation; (5) an outcome-based sigmoid meta-router trained on empirically discovered domain-combination targets that selectively weights stacks, enabling cross-domain composition. Two boundary experiments: (6) PSN pretraining on a randomly initialized model; (7) per-domain RL (DPO/GRPO) validating compatibility with post-SFT alignment. Validated on TinyLlama-1.1B (4 domains, 9 stacks) and Gemma 3 12B IT (5 domains, 10 stacks), MoE-LoRA achieves 2.5x faster convergence than parameter-matched single LoRA, residual boosting breaks through the single-stack ceiling, and the routed system recovers generation quality destroyed by ungated stack accumulation. The central finding: the outcome-based router discovers that domain stacks encode transferable cognitive primitives (instruction-following clarity, numerical reasoning, procedural logic, chain-of-thought structure) rather than domain-specific knowledge, with medical prompts routing to chat+math stacks in 97% of cases despite zero medical data in those stacks. 

---
# Paper Reconstruction Evaluation: Evaluating Presentation and Hallucination in AI-written Papers 

**Authors**: Atsuyuki Miyai, Mashiro Toyooka, Zaiying Zhao, Kenta Watanabe, Toshihiko Yamasaki, Kiyoharu Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2604.01128)  

**Abstract**: This paper introduces the first systematic evaluation framework for quantifying the quality and risks of papers written by modern coding agents. While AI-driven paper writing has become a growing concern, rigorous evaluation of the quality and potential risks of AI-written papers remains limited, and a unified understanding of their reliability is still lacking. We introduce Paper Reconstruction Evaluation (PaperRecon), an evaluation framework in which an overview (this http URL) is created from an existing paper, after which an agent generates a full paper based on the overview and minimal additional resources, and the result is subsequently compared against the original paper. PaperRecon disentangles the evaluation of the AI-written papers into two orthogonal dimensions, Presentation and Hallucination, where Presentation is evaluated using a rubric and Hallucination is assessed via agentic evaluation grounded in the original paper source. For evaluation, we introduce PaperWrite-Bench, a benchmark of 51 papers from top-tier venues across diverse domains published after 2025. Our experiments reveal a clear trade-off: while both ClaudeCode and Codex improve with model advances, ClaudeCode achieves higher presentation quality at the cost of more than 10 hallucinations per paper on average, whereas Codex produces fewer hallucinations but lower presentation quality. This work takes a first step toward establishing evaluation frameworks for AI-driven paper writing and improving the understanding of its risks within the research community. 

---
# CARE: Privacy-Compliant Agentic Reasoning with Evidence Discordance 

**Authors**: Haochen Liu, Weien Li, Rui Song, Zeyu Li, Chun Jason Xue, Xiao-Yang Liu, Sam Nallaperuma, Xue Liu, Ye Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2604.01113)  

**Abstract**: Large language model (LLM) systems are increasingly used to support high-stakes decision-making, but they typically perform worse when the available evidence is internally inconsistent. Such a scenario exists in real-world healthcare settings, with patient-reported symptoms contradicting medical signs. To study this problem, we introduce MIMIC-DOS, a dataset for short-horizon organ dysfunction worsening prediction in the intensive care unit (ICU) setting. We derive this dataset from the widely recognized MIMIC-IV, a publicly available electronic health record dataset, and construct it exclusively from cases in which discordance between signs and symptoms exists. This setting poses a substantial challenge for existing LLM-based approaches, with single-pass LLMs and agentic pipelines often struggling to reconcile such conflicting signals. To address this problem, we propose CARE: a multi-stage privacy-compliant agentic reasoning framework in which a remote LLM provides guidance by generating structured categories and transitions without accessing sensitive patient data, while a local LLM uses these categories and transitions to support evidence acquisition and final decision-making. Empirically, CARE achieves stronger performance across all key metrics compared to multiple baseline settings, showing that CARE can more robustly handle conflicting clinical evidence while preserving privacy. 

---
# Temporal Dependencies in In-Context Learning: The Role of Induction Heads 

**Authors**: Anooshka Bajaj, Deven Mahesh Mistry, Sahaj Singh Maini, Yash Aggarwal, Billy Dickson, Zoran Tiganj  

**Link**: [PDF](https://arxiv.org/pdf/2604.01094)  

**Abstract**: Large language models (LLMs) exhibit strong in-context learning capabilities, but how they track and retrieve information from context remains underexplored. Drawing on the free recall paradigm in cognitive science (where participants recall list items in any order), we show that several open-source LLMs consistently display a serial-recall-like pattern, assigning peak probability to tokens that immediately follow a repeated token in the input sequence. Through systematic ablation experiments, we show that induction heads, specialized attention heads that attend to the token following a previous occurrence of the current token, play an important role in this phenomenon. Removing heads with a high induction score substantially reduces the +1 lag bias, whereas ablating random heads does not reproduce the same reduction. We also show that removing heads with high induction scores impairs the performance of models prompted to do serial recall using few-shot learning to a larger extent than removing random heads. Our findings highlight a mechanistically specific connection between induction heads and temporal context processing in transformers, suggesting that these heads are especially important for ordered retrieval and serial-recall-like behavior during in-context learning. 

---
# Narrative Fingerprints: Multi-Scale Author Identification via Novelty Curve Dynamics 

**Authors**: Fred Zimmerman, Hilmar AI  

**Link**: [PDF](https://arxiv.org/pdf/2604.01073)  

**Abstract**: We test whether authors have characteristic "fingerprints" in the information-theoretic novelty curves of their published works. Working with two corpora -- Books3 (52,796 books, 759 qualifying authors) and PG-19 (28,439 books, 1,821 qualifying authors) -- we find that authorial voice leaves measurable traces in how novelty unfolds across a text. The signal is multi-scale: at book level, scalar dynamics (mean novelty, speed, volume, circuitousness) identify 43% of authors significantly above chance; at chapter level, SAX motif patterns in sliding windows achieve 30x-above-chance attribution, far exceeding the scalar features that dominate at book level. These signals are complementary, not redundant. We show that the fingerprint is partly confounded with genre but persists within-genre for approximately one-quarter of authors. Classical authors (Twain, Austen, Kipling) show fingerprints comparable in strength to modern authors, suggesting the phenomenon is not an artifact of contemporary publishing conventions. 

---
# Uncertainty-Aware Variational Reward Factorization via Probabilistic Preference Bases for LLM Personalization 

**Authors**: Gyuseok Lee, Wonbin Kweon, Zhenrui Yue, SeongKu Kang, Jiawei Han, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00997)  

**Abstract**: Reward factorization personalizes large language models (LLMs) by decomposing rewards into shared basis functions and user-specific weights. Yet, existing methods estimate user weights from scarce data in isolation and as deterministic points, leading to inaccurate and unreliable inference. We introduce Variational Reward Factorization (VRF), an uncertainty-aware framework that represents each user's preferences as a variational distribution in a shared preference space. VRF infers user distributions via a variational encoder, derives weights through Wasserstein distance matching with shared probabilistic bases, and downweights uncertain estimates through a variance-attenuated loss. On three benchmarks, VRF outperforms all baselines across seen and unseen users, few-shot scenarios, and varying uncertainty levels, with gains extending to downstream alignment. 

---
# Multimodal Analysis of State-Funded News Coverage of the Israel-Hamas War on YouTube Shorts 

**Authors**: Daniel Miehling, Sandra Kuebler  

**Link**: [PDF](https://arxiv.org/pdf/2604.00994)  

**Abstract**: YouTube Shorts have become central to news consumption on the platform, yet research on how geopolitical events are represented in this format remains limited. To address this gap, we present a multimodal pipeline that combines automatic transcription, aspect-based sentiment analysis (ABSA), and semantic scene classification. The pipeline is first assessed for feasibility and then applied to analyze short-form coverage of the Israel-Hamas war by state-funded outlets. Using over 2,300 conflict-related Shorts and more than 94,000 visual frames, we systematically examine war reporting across major international broadcasters. Our findings reveal that the sentiment expressed in transcripts regarding specific aspects differs across outlets and over time, whereas scene-type classifications reflect visual cues consistent with real-world events. Notably, smaller domain-adapted models outperform large transformers and even LLMs for sentiment analysis, underscoring the value of resource-efficient approaches for humanities research. The pipeline serves as a template for other short-form platforms, such as TikTok and Instagram, and demonstrates how multimodal methods, combined with qualitative interpretation, can characterize sentiment patterns and visual cues in algorithmically driven video environments. 

---
# Dual Optimal: Make Your LLM Peer-like with Dignity 

**Authors**: Xiangqi Wang, Yue Huang, Haomin Zhuang, Kehan Guo, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00979)  

**Abstract**: Current aligned language models exhibit a dual failure mode we term the Evasive Servant: they sycophantically validate flawed user beliefs while deflecting responsibility with boilerplate disclaimers. We propose the Dignified Peer framework, which counters servility with anti-sycophancy and trustworthiness, and mitigates evasiveness through empathy and creativity. Realizing this agent requires overcoming significant challenges in data supervision, objective collapse, and evaluation bias. We address these issues by introducing the PersonaKnob dataset which features a compositional partial order structure of multiple persona preference. This data is utilized alongside a tolerant constrained Lagrangian DPO algorithm that dynamically balances all persona dimensions to prevent behavioral collapse. Additionally, we employ a psychometrically calibrated Item Response Theory evaluation protocol to disentangle latent model persona capability from confounders like judge biases. Extensive empirical studies demonstrate that our approach successfully build a LLM agent with both dignity and peer. 

---
# Phase transition on a context-sensitive random language model with short range interactions 

**Authors**: Yuma Toji, Jun Takahashi, Vwani Roychowdhury, Hideyuki Miyahara  

**Link**: [PDF](https://arxiv.org/pdf/2604.00947)  

**Abstract**: Since the random language model was proposed by E. DeGiuli [Phys. Rev. Lett. 122, 128301], language models have been investigated intensively from the viewpoint of statistical mechanics. Recently, the existence of a Berezinskii--Kosterlitz--Thouless transition was numerically demonstrated in models with long-range interactions between symbols. In statistical mechanics, it has long been known that long-range interactions can induce phase transitions. Therefore, it has remained unclear whether phase transitions observed in language models originate from genuinely linguistic properties that are absent in conventional spin models. In this study, we construct a random language model with short-range interactions and numerically investigate its statistical properties. Our model belongs to the class of context-sensitive grammars in the Chomsky hierarchy and allows explicit reference to contexts. We find that a phase transition occurs even when the model refers only to contexts whose length remains constant with respect to the sentence length. This result indicates that finite-temperature phase transitions in language models are genuinely induced by the intrinsic nature of language, rather than by long-range interactions. 

---
# Positional Cognitive Specialization: Where Do LLMs Learn To Comprehend and Speak Your Language? 

**Authors**: Luis Frentzen Salim, Lun-Wei Ku, Hsing-Kuo Kenneth Pao  

**Link**: [PDF](https://arxiv.org/pdf/2604.00923)  

**Abstract**: Adapting large language models (LLMs) to new languages is an expensive and opaque process. Understanding how language models acquire new languages and multilingual abilities is key to achieve efficient adaptation. Prior work on multilingual interpretability research focuses primarily on how trained models process multilingual instructions, leaving unexplored the mechanisms through which they acquire new languages during training. We investigate these training dynamics on decoder-only transformers through the lens of two functional cognitive specializations: language perception (input comprehension) and production (output generation). Through experiments on low-resource languages, we demonstrate how perceptual and productive specialization emerges in different regions of a language model by running layer ablation sweeps from the model's input and output directions. Based on the observed specialization patterns, we propose CogSym, a layer-wise heuristic that enables effective adaptation by exclusively fine-tuning a few early and late layers. We show that tuning only the 25% outermost layers achieves downstream task performance within 2-3% deviation from the full fine-tuning baseline. CogSym yields consistent performance with adapter methods such as LoRA, showcasing generalization beyond full fine-tuning. These findings provide insights to better understand how LLMs learn new languages and push toward accessible and inclusive language modeling. 

---
# GPT-NL Public Corpus: A Permissively Licensed, Dutch-First Dataset for LLM Pre-training 

**Authors**: Jesse van Oort, Frank Brinkkemper, Erik de Graaf, Bram Vanroy, Saskia Lensink  

**Link**: [PDF](https://arxiv.org/pdf/2604.00920)  

**Abstract**: We present the GPT-NL Public Corpus, the biggest permissively licensed corpus of Dutch language resources. The GPT-NL Public Corpus contains 21 Dutch-only collections totalling 36B preprocessed Dutch tokens not present in any other LLM pretraining corpus. Additionally, the corpus includes roughly 207B English, 232B Code, and 48B German/Danish tokens taken from existing sets which we further curated for compliance. This corpus includes curated data from large existing corpora like Common Corpus and Common Crawl, as well as newly created Dutch-specific collections. Most newly created Dutch collections consist of content collected in collaboration with organisations or synthetically augmented content. All data is collected and evaluated with the aim of facilitating the creation of (commercial) language models that are lawful, useful and non-harmful. All data included in the GPT-NL Public Corpus is sourced from datasets with permissive licensing and is curated and redistributed under a CC-BY license. The full dataset is publicly available on the Hugging Face Hub. 

---
# When Users Change Their Mind: Evaluating Interruptible Agents in Long-Horizon Web Navigation 

**Authors**: Henry Peng Zou, Chunyu Miao, Wei-Chieh Huang, Yankai Chen, Yue Zhou, Hanrong Zhang, Yaozu Wu, Liancheng Fang, Zhengyao Gu, Zhen Zhang, Kening Zheng, Fangxin Wang, Yi Nian, Shanghao Li, Wenzhe Fan, Langzhou He, Weizhi Zhang, Xue Liu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00892)  

**Abstract**: As LLM agents transition from short, static problem solving to executing complex, long-horizon tasks in dynamic environments, the ability to handle user interruptions, such as adding requirement or revising goals, during mid-task execution is becoming a core requirement for realistic deployment. However, existing benchmarks largely assume uninterrupted agent behavior or study interruptions only in short, unconstrained language tasks. In this paper, we present the first systematic study of interruptible agents in long-horizon, environmentally grounded web navigation tasks, where actions induce persistent state changes. We formalize three realistic interruption types, including addition, revision, and retraction, and introduce InterruptBench, a benchmark derived from WebArena-Lite that synthesizes high-quality interruption scenarios under strict semantic constraints. Using a unified interruption simulation framework, we evaluate six strong LLM backbones across single- and multi-turn interruption settings, analyzing both their effectiveness in adapting to updated intents and their efficiency in recovering from mid-task changes. Our results show that handling user interruptions effectively and efficiently during long-horizon agentic tasks remains challenging for powerful large-scale LLMs. Code and dataset are available at this https URL. 

---
# KUET at StanceNakba Shared Task: StanceMoE: Mixture-of-Experts Architecture for Stance Detection 

**Authors**: Abdullah Al Shafi, Md. Milon Islam, Sk. Imran Hossain, K. M. Azharul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2604.00878)  

**Abstract**: Actor-level stance detection aims to determine an author expressed position toward specific geopolitical actors mentioned or implicated in a text. Although transformer-based models have achieved relatively good performance in stance classification, they typically rely on unified representations that may not sufficiently capture heterogeneous linguistic signals, such as contrastive discourse structures, framing cues, and salient lexical indicators. This motivates the need for adaptive architectures that explicitly model diverse stance-expressive patterns. In this paper, we propose StanceMoE, a context-enhanced Mixture-of-Experts (MoE) architecture built upon a fine-tuned BERT encoder for actor-level stance detection. Our model integrates six expert modules designed to capture complementary linguistic signals, including global semantic orientation, salient lexical cues, clause-level focus, phrase-level patterns, framing indicators, and contrast-driven discourse shifts. A context-aware gating mechanism dynamically weights expert contributions, enabling adaptive routing based on input characteristics. Experiments are conducted on the StanceNakba 2026 Subtask A dataset, comprising 1,401 annotated English texts where the target actor is implicit in the text. StanceMoE achieves a macro-F1 score of 94.26%, outperforming traditional baselines, and alternative BERT-based variants. 

---
# Agentic Tool Use in Large Language Models 

**Authors**: Jinchao Hu, Meizhi Zhong, Kehai Chen, Xuefeng Bai, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00835)  

**Abstract**: Large language models are increasingly being deployed as autonomous agents yet their real world effectiveness depends on reliable tools for information retrieval, computation and external action. Existing studies remain fragmented across tasks, tool types, and training settings, lacking a unified view of how tool-use methods differ and evolve. This paper organizes the literature into three paradigms: prompting as plug-and-play, supervised tool learning and reward-driven tool policy learning, analyzes their methods, strengths and failure modes, reviews the evaluation landscape and highlights key challenges, aiming to address this fragmentation and provide a more structured evolutionary view of agentic tool use. 

---
# Emotion Entanglement and Bayesian Inference for Multi-Dimensional Emotion Understanding 

**Authors**: Hemanth Kotaprolu, Kishan Maharaj, Raey Zhao, Abhijit Mishra, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2604.00819)  

**Abstract**: Understanding emotions in natural language is inherently a multi-dimensional reasoning problem, where multiple affective signals interact through context, interpersonal relations, and situational cues. However, most existing emotion understanding benchmarks rely on short texts and predefined emotion labels, reducing this process to independent label prediction and ignoring the structured dependencies among emotions. To address this limitation, we introduce Emotional Scenarios (EmoScene), a theory-grounded benchmark of 4,731 context-rich scenarios annotated with an 8-dimensional emotion vector derived from Plutchik's basic emotions. We evaluate six instruction-tuned large language models in a zero-shot setting and observe modest performance, with the best model achieving a Macro F1 of 0.501, highlighting the difficulty of context-aware multi-label emotion prediction. Motivated by the observation that emotions rarely occur independently, we further propose an entanglement-aware Bayesian inference framework that incorporates emotion co-occurrence statistics to perform joint posterior inference over the emotion vector. This lightweight post-processing improves structural consistency of predictions and yields notable gains for weaker models (e.g., +0.051 Macro F1 for Qwen2.5-7B). EmoScene therefore provides a challenging benchmark for studying multi-dimensional emotion understanding and the limitations of current language models. 

---
# Valency Classification of Mapudungun Verbal Roots. Established by the language's own morphotactics 

**Authors**: Andrés Chandía  

**Link**: [PDF](https://arxiv.org/pdf/2604.00789)  

**Abstract**: In the previous work, a lexical (re)categorisation -- or confirmation of the given category -- of roots identified as verbal was undertaken to determine their original category accurately. Building on this, the present paper offers an account of the valency classification of those Mapudungun roots confirmed to be verbal, using the language's own morphotactics; specifically, by examining the permissible and restricted combinations of various suffixes with roots or verbal stems in the Mapuche verb form. As with all work conducted thus far, the results presented here aim to improve the morphological analyser (Dungupeyum) with all verified findings incorporated into the system. From a theoretical perspective, we also hope to contribute to the recognition and understanding of issues related to the valency of Mapuche verb forms. 

---
# From Early Encoding to Late Suppression: Interpreting LLMs on Character Counting Tasks 

**Authors**: Ayan Datta, Mounika Marreddy, Alexander Mehler, Zhixue Zhao, Radhika Mamidi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00778)  

**Abstract**: Large language models (LLMs) exhibit failures on elementary symbolic tasks such as character counting in a word, despite excelling on complex benchmarks. Although this limitation has been noted, the internal reasons remain unclear. We use character counting (e.g., "How many p's are in apple?") as a minimal, controlled probe that isolates token-level reasoning from higher-level confounds. Using this setting, we uncover a consistent phenomenon across modern architectures, including LLaMA, Qwen, and Gemma: models often compute the correct answer internally yet fail to express it at the output layer.
Through mechanistic analysis combining probing classifiers, activation patching, logit lens analysis, and attention head tracing, we show that character-level information is encoded in early and mid-layer representations. However, this information is attenuated by a small set of components in later layers, especially the penultimate and final layer MLP. We identify these components as negative circuits: subnetworks that downweight correct signals in favor of higher-probability but incorrect outputs.
Our results lead to two contributions. First, we show that symbolic reasoning failures in LLMs are not due to missing representations or insufficient scale, but arise from structured interference within the model's computation graph. This explains why such errors persist and can worsen under scaling and instruction tuning. Second, we provide evidence that LLM forward passes implement a form of competitive decoding, in which correct and incorrect hypotheses coexist and are dynamically reweighted, with final outputs determined by suppression as much as by amplification.
These findings carry implications for interpretability and robustness: simple symbolic reasoning exposes weaknesses in modern LLMs, underscoring need for design strategies that ensure information is encoded and reliably used. 

---
# From Baselines to Preferences: A Comparative Study of LoRA/QLoRA and Preference Optimization for Mental Health Text Classification 

**Authors**: Mihael Arcan  

**Link**: [PDF](https://arxiv.org/pdf/2604.00773)  

**Abstract**: Mental health text classification has rapidly adopted modern adaptation methods, yet practical guidance on which optimization strategy to use, when, and why remains limited. This paper presents a systematic comparative study of optimization pathways for a joint mental-health classification task, moving from strong vanilla baselines to progressively more specialized techniques. We first establish classical and encoder references, then examine parameter-efficient supervised fine-tuning with LoRA/QLoRA under multiple objective and optimization settings, and finally evaluate preference-based optimization with DPO, ORPO, and KTO, including class-rebalanced training. Rather than emphasizing a single headline score, we focus on methodological insight: how performance changes with objective formulation, adapter choice, optimizer behavior, context windowing, and class-balance intervention. The results show that optimization effects are highly method-dependent: some approaches deliver stable, transferable gains, while others are sensitive to configuration and data balance. Preference optimization, in particular, exhibits large variation across objectives, indicating that method selection is more consequential than simply adding a preference-training stage. The central contribution is a clear optimization narrative for mental health NLP: start from transparent baselines, apply controlled tuning, and use preference optimization selectively where its gains are demonstrable. This provides a reproducible and practically grounded framework for choosing effective training strategies beyond architecture choice alone. 

---
# Stochastic Attention: Connectome-Inspired Randomized Routing for Expressive Linear-Time Attention 

**Authors**: Zehao Jin, Yanan Sui  

**Link**: [PDF](https://arxiv.org/pdf/2604.00754)  

**Abstract**: The whole-brain connectome of a fruit fly comprises over 130K neurons connected with a probability of merely 0.02%, yet achieves an average shortest path of only 4.4 hops. Despite being highly structured at the circuit level, the network's long-range connections are broadly distributed across brain regions, functioning as stochastic shortcuts that enable efficient global communication. Inspired by this observation, we propose Stochastic Attention (SA), a drop-in enhancement for sliding-window attention (SWA) that applies a random permutation to the token sequence before windowed attention and restores the original order afterward. This transforms the fixed local window into a stochastic global one within the same $O(nw)$ per-layer budget. Through depth, independently sampled permutations yield exponentially growing receptive fields, achieving full sequence coverage in $O(\log_w n)$ layers versus $O(n/w)$ for SWA. We validate SA in two settings: pre-training language models from scratch, where a gated SA + SWA combination achieves the best average zero-shot accuracy, and training-free inference on Qwen3-8B and Qwen3-30B-A3B, where SA consistently outperforms SWA and matches or exceeds Mixture of Block Attention at comparable compute budgets. These results suggest that connectome-inspired stochastic routing is a practical primitive for improving the expressivity of efficient attention, complementary to existing linear and sparse approaches. 

---
# LangMARL: Natural Language Multi-Agent Reinforcement Learning 

**Authors**: Huaiyuan Yao, Longchao Da, Xiaoou Liu, Charles Fleming, Tianlong Chen, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.00722)  

**Abstract**: Large language model (LLM) agents struggle to autonomously evolve coordination strategies in dynamic environments, largely because coarse global outcomes obscure the causal signals needed for local policy refinement. We identify this bottleneck as a multi-agent credit assignment problem, which has long been studied in classical multi-agent reinforcement learning (MARL) but remains underaddressed in LLM-based systems. Building on this observation, we propose LangMARL, a framework that brings credit assignment and policy gradient evolution from cooperative MARL into the language space. LangMARL introduces agent-level language credit assignment, pioneers gradient evolution in language space for policy improvement, and summarizes task-relevant causal relations from replayed trajectories to provide dense feedback and improve convergence under sparse rewards. Extensive experiments across diverse cooperative multi-agent tasks demonstrate improved sample efficiency, interpretability, and strong generalization. 

---
# To Memorize or to Retrieve: Scaling Laws for RAG-Considerate Pretraining 

**Authors**: Karan Singh, Michael Yu, Varun Gangal, Zhuofu Tao, Sachin Kumar, Emmy Liu, Steven Y. Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.00715)  

**Abstract**: Retrieval-augmented generation (RAG) improves language model (LM) performance by providing relevant context at test time for knowledge-intensive situations. However, the relationship between parametric knowledge acquired during pretraining and non-parametric knowledge accessed via retrieval remains poorly understood, especially under fixed data budgets. In this work, we systematically study the trade-off between pretraining corpus size and retrieval store size across a wide range of model and data scales. We train OLMo-2-based LMs ranging from 30M to 3B parameters on up to 100B tokens of DCLM data, while varying both pretraining data scale (1-150x the number of parameters) and retrieval store size (1-20x), and evaluate performance across a diverse suite of benchmarks spanning reasoning, scientific QA, and open-domain QA. We find that retrieval consistently improves performance over parametric-only baselines across model scales and introduce a three-dimensional scaling framework that models performance as a function of model size, pretraining tokens, and retrieval corpus size. This scaling manifold enables us to estimate optimal allocations of a fixed data budget between pretraining and retrieval, revealing that the marginal utility of retrieval depends strongly on model scale, task type, and the degree of pretraining saturation. Our results provide a quantitative foundation for understanding when and how retrieval should complement pretraining, offering practical guidance for allocating data resources in the design of scalable language modeling systems. 

---
# AfrIFact: Cultural Information Retrieval, Evidence Extraction and Fact Checking for African Languages 

**Authors**: Israel Abebe Azime, Jesujoba Oluwadara Alabi, Crystina Zhang, Iffat Maab, Atnafu Lambebo Tonja, Tadesse Destaw Belay, Folasade Peace Alabi, Salomey Osei, Saminu Mohammad Aliyu, Nkechinyere Faith Aguobi, Bontu Fufa Balcha, Blessing Kudzaishe Sibanda, Davis David, Mouhamadane Mboup, Daud Abolade, Neo Putini, Philipp Slusallek, David Ifeoluwa Adelani, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2604.00706)  

**Abstract**: Assessing the veracity of a claim made online is a complex and important task with real-world implications. When these claims are directed at communities with limited access to information and the content concerns issues such as healthcare and culture, the consequences intensify, especially in low-resource languages. In this work, we introduce AfrIFact, a dataset that covers the necessary steps for automatic fact-checking (i.e., information retrieval, evidence extraction, and fact checking), in ten African languages and English. Our evaluation results show that even the best embedding models lack cross-lingual retrieval capabilities, and that cultural and news documents are easier to retrieve than healthcare-domain documents, both in large corpora and in single documents. We show that LLMs lack robust multilingual fact-verification capabilities in African languages, while few-shot prompting improves performance by up to 43% in AfriqueQwen-14B, and task-specific fine-tuning further improves fact-checking accuracy by up to 26%. These findings, along with our release of the AfrIFact dataset, encourage work on low-resource information retrieval, evidence retrieval, and fact checking. 

---
# OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models 

**Authors**: Han Zhu, Lingxuan Ye, Wei Kang, Zengwei Yao, Liyong Guo, Fangjun Kuang, Zhifeng Han, Weiji Zhuang, Long Lin, Daniel Povey  

**Link**: [PDF](https://arxiv.org/pdf/2604.00688)  

**Abstract**: We present OmniVoice, a massive multilingual zero-shot text-to-speech (TTS) model that scales to over 600 languages. At its core is a novel diffusion language model-style discrete non-autoregressive (NAR) architecture. Unlike conventional discrete NAR models that suffer from performance bottlenecks in complex two-stage (text-to-semantic-to-acoustic) pipelines, OmniVoice directly maps text to multi-codebook acoustic tokens. This simplified approach is facilitated by two key technical innovations: (1) a full-codebook random masking strategy for efficient training, and (2) initialization from a pre-trained LLM to ensure superior intelligibility. By leveraging a 581k-hour multilingual dataset curated entirely from open-source data, OmniVoice achieves the broadest language coverage to date and delivers state-of-the-art performance across Chinese, English, and diverse multilingual benchmarks. Our code and pre-trained models are publicly available at this https URL. 

---
# Common TF-IDF variants arise as key components in the test statistic of a penalized likelihood-ratio test for word burstiness 

**Authors**: Zeyad Ahmed, Paul Sheridan, Michael McIsaac, Aitazaz A. Farooque  

**Link**: [PDF](https://arxiv.org/pdf/2604.00672)  

**Abstract**: TF-IDF is a classical formula that is widely used for identifying important terms within documents. We show that TF-IDF-like scores arise naturally from the test statistic of a penalized likelihood-ratio test setup capturing word burstiness (also known as word over-dispersion). In our framework, the alternative hypothesis captures word burstiness by modeling a collection of documents according to a family of beta-binomial distributions with a gamma penalty term on the precision parameter. In contrast, the null hypothesis assumes that words are binomially distributed in collection documents, a modeling approach that fails to account for word burstiness. We find that a term-weighting scheme given rise to by this test statistic performs comparably to TF-IDF on document classification tasks. This paper provides insights into TF-IDF from a statistical perspective and underscores the potential of hypothesis testing frameworks for advancing term-weighting scheme development. 

---
# TRIMS: Trajectory-Ranked Instruction Masked Supervision for Diffusion Language Models 

**Authors**: Lingjie Chen, Ruizhong Qiu, Yuyu Fan, Yanjun Zhao, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2604.00666)  

**Abstract**: Diffusion language models (DLMs) offer a promising path toward low-latency generation through parallel decoding, but their practical efficiency depends heavily on the decoding trajectory. In practice, this advantage often fails to fully materialize because standard training does not provide explicit supervision over token reveal order, creating a train-inference mismatch that leads to suboptimal decoding behavior. We propose Trajectory-Ranked Instruction Masked Supervision (TRIMS), a simple trajectory-guided supervised fine-tuning framework that injects trajectory supervision into standard Masked Diffusion Language Model (MDLM) training with minimal overhead. Instead of relying on costly DLM-based distillation, TRIMS uses lightweight signals from an autoregressive teacher to guide a trajectory-aware masking strategy, encouraging the model to learn more effective decoding orders. Experiments on LLaDA and Dream across math and coding benchmarks show that TRIMS significantly improves the accuracy-parallelism trade-off over both standard MDLM training and train-free acceleration baselines, while achieving competitive performance with prior distillation-based approaches at substantially lower training cost. Further analysis shows that TRIMS leads to better decoding trajectories, validating the effectiveness of trajectory-guided supervision for DLMs. 

---
# English to Central Kurdish Speech Translation: Corpus Creation, Evaluation, and Orthographic Standardization 

**Authors**: Mohammad Mohammadamini, Daban Q. Jaff, Josep Crego, Marie Tahon, Antoine Laurent  

**Link**: [PDF](https://arxiv.org/pdf/2604.00613)  

**Abstract**: We present KUTED, a speech-to-text translation (S2TT) dataset for Central Kurdish, derived from TED and TEDx talks. The corpus comprises 91,000 sentence pairs, including 170 hours of English audio, 1.65 million English tokens, and 1.40 million Central Kurdish tokens. We evaluate KUTED on the S2TT task and find that orthographic variation significantly degrades Kurdish translation performance, producing nonstandard outputs. To address this, we propose a systematic text standardization approach that yields substantial performance gains and more consistent translations. On a test set separated from TED talks, a fine-tuned Seamless model achieves 15.18 BLEU, and we improve Seamless baseline by 3.0 BLEU on the FLEURS benchmark. We also train a Transformer model from scratch and evaluate a cascaded system that combines Seamless (ASR) with NLLB (MT). 

---
# Speech LLMs are Contextual Reasoning Transcribers 

**Authors**: Keqi Deng, Ruchao Fan, Bo Ren, Yiming Wang, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.00610)  

**Abstract**: Despite extensions to speech inputs, effectively leveraging the rich knowledge and contextual understanding of large language models (LLMs) in automatic speech recognition (ASR) remains non-trivial, as the task primarily involves direct speech-to-text mapping. To address this, this paper proposes chain-of-thought ASR (CoT-ASR), which constructs a reasoning chain that enables LLMs to first analyze the input speech and generate contextual analysis, thereby fully exploiting their generative capabilities. With this contextual reasoning, CoT-ASR then performs more informed speech recognition and completes both reasoning and transcription in a single pass. Moreover, CoT-ASR naturally supports user-guided transcription: while designed to self-generate reasoning, it can also seamlessly incorporate user-provided context to guide transcription, further extending ASR functionality. To reduce the modality gap, this paper introduces a CTC-guided Modality Adapter, which uses CTC non-blank token probabilities to weight LLM embeddings, efficiently aligning speech encoder outputs with the LLM's textual latent space. Experiments show that, compared to standard LLM-based ASR, CoT-ASR achieves a relative reduction of 8.7% in word error rate (WER) and 16.9% in entity error rate (EER). 

---
# More Human, More Efficient: Aligning Annotations with Quantized SLMs 

**Authors**: Jiayu Wang, Junyoung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.00586)  

**Abstract**: As Large Language Model (LLM) capabilities advance, the demand for high-quality annotation of exponentially increasing text corpora has outpaced human capacity, leading to the widespread adoption of LLMs in automatic evaluation and annotation. However, proprietary LLMs often exhibit systematic biases that diverge from human expert consensus, lacks reproducibility, and raises data privacy concerns. Our work examines the viability of finetuning a quantized Small Language Model of 1.7B parameter size on limited human-annotated data to serve as a highly aligned, deterministic evaluator and annotator. By implementing a custom, multi-dimensional rubric framework and simple augmentation and regularization techniques, the proposed approach achieves higher inter-annotator agreement (0.23 points increase in Krippendorff's $\alpha$) than the best performing state-of-the-art proprietary LLM. We also demonstrate the generalizability of the proposed training pipeline on a separate emotion classification task. The results show that task-specific alignment and efficient 4-bit quantized fine-tuning provide superior open-source alternative to using proprietary models for evaluation and annotation. Our finetuning approach is publicly available at this https URL. 

---
# A Japanese Benchmark for Evaluating Social Bias in Reasoning Based on Attribution Theory 

**Authors**: Taihei Shiotani, Masahiro Kaneko, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2604.00568)  

**Abstract**: In enhancing the fairness of Large Language Models (LLMs), evaluating social biases rooted in the cultural contexts of specific linguistic regions is essential. However, most existing Japanese benchmarks heavily rely on translating English data, which does not necessarily provide an evaluation suitable for Japanese culture. Furthermore, they only evaluate bias in the conclusion, failing to capture biases lurking in the reasoning. In this study, based on attribution theory in social psychology, we constructed a new dataset, ``JUBAKU-v2,'' which evaluates the bias in attributing behaviors to in-groups and out-groups within reasoning while fixing the conclusion. This dataset consists of 216 examples reflecting cultural biases specific to Japan. Experimental results verified that it can detect performance differences across models more sensitively than existing benchmarks. 

---
# Optimsyn: Influence-Guided Rubrics Optimization for Synthetic Data Generation 

**Authors**: Zhiting Fan, Ruizhe Chen, Tianxiang Hu, Ru Peng, Zenan Huang, Haokai Xu, Yixin Chen, Jian Wu, Junbo Zhao, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00536)  

**Abstract**: Large language models (LLMs) achieve strong downstream performance largely due to abundant supervised fine-tuning (SFT) data. However, high-quality SFT data in knowledge-intensive domains such as humanities, social sciences, medicine, law, and finance is scarce because expert curation is expensive, privacy constraints are strict, and label consistency is hard to ensure. Recent work uses synthetic data, typically by prompting a generator over domain documents and filtering outputs with handcrafted rubrics. Yet rubric design is expert-dependent, transfers poorly across domains, and is often optimized through a brittle heuristic loop of writing rubrics, synthesizing data, training, inspecting results, and manually guessing revisions. This process lacks reliable quantitative feedback about how a rubric affects downstream performance. We propose evaluating synthetic data by its training utility on the target model and using this signal to guide data generation. Inspired by influence estimation, we adopt an optimizer-aware estimator that uses gradient information to quantify each synthetic sample's contribution to a target model's objective on specific tasks. Our analysis shows that even when synthetic and real samples are close in embedding space, their influence on learning can differ substantially. Based on this insight, we propose an optimization-based framework that adapts rubrics using target-model feedback. We provide lightweight guiding text and use a rubric-specialized model to generate task-conditioned rubrics. Influence score is used as the reward to optimize the rubric generator with reinforcement learning. Experiments across domains, target models, and data generators show consistent improvements and strong generalization without task-specific tuning. 

---
# Adapting Text LLMs to Speech via Multimodal Depth Up-Scaling 

**Authors**: Kazuki Yano, Jun Suzuki, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2604.00489)  

**Abstract**: Adapting pre-trained text Large Language Models (LLMs) into Speech Language Models (Speech LMs) via continual pretraining on speech data is promising, but often degrades the original text capabilities. We propose Multimodal Depth Upscaling, an extension of an emerging strategy in continual LLM pre-training, where new transformer layers are inserted into a frozen text LLM and only the added layers are trained on speech data. Experiments with SmolLM2-360M and SmolLM2-1.7B on 48k hours of English Automatic Speech Recognition (ASR) data show that depth up-scaling achieves ASR comparable to full fine-tuning while causing far less text degradation than both full fine-tuning and Low-Rank Adaptation (LoRA). We further show that incorporating E-Branchformer, an architecture designed for speech recognition, as the inserted layers achieves ASR that matches or surpasses full fine-tuning on the larger model while reducing text degradation by over 75% with 60% fewer trainable parameters. 

---
# Polysemanticity or Polysemy? Lexical Identity Confounds Superposition Metrics 

**Authors**: Iyad Ait Hou, Rebecca Hwa  

**Link**: [PDF](https://arxiv.org/pdf/2604.00443)  

**Abstract**: If the same neuron activates for both "lender" and "riverside," standard metrics attribute the overlap to superposition--the neuron must be compressing two unrelated concepts. This work explores how much of the overlap is due a lexical confound: neurons fire for a shared word form (such as "bank") rather than for two compressed concepts. A 2x2 factorial decomposition reveals that the lexical-only condition (same word, different meaning) consistently exceeds the semantic-only condition (different word, same meaning) across models spanning 110M-70B parameters. The confound carries into sparse autoencoders (18-36% of features blend senses), sits in <=1% of activation dimensions, and hurts downstream tasks: filtering it out improves word sense disambiguation and makes knowledge edits more selective (p = 0.002). 

---
# TR-ICRL: Test-Time Rethinking for In-Context Reinforcement Learning 

**Authors**: Wenxuan Jiang, Yuxin Zuo, Zijian Zhang, Xuecheng Wu, Zining Fan, Wenxuan Liu, Li Chen, Xiaoyu Li, Xuezhi Cao, Xiaolong Jin, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00438)  

**Abstract**: In-Context Reinforcement Learning (ICRL) enables Large Language Models (LLMs) to learn online from external rewards directly within the context window. However, a central challenge in ICRL is reward estimation, as models typically lack access to ground-truths during inference. To address this limitation, we propose Test-Time Rethinking for In-Context Reinforcement Learning (TR-ICRL), a novel ICRL framework designed for both reasoning and knowledge-intensive tasks. TR-ICRL operates by first retrieving the most relevant instances from an unlabeled evaluation set for a given query. During each ICRL iteration, LLM generates a set of candidate answers for every retrieved instance. Next, a pseudo-label is derived from this set through majority voting. This label then serves as a proxy to give reward messages and generate formative feedbacks, guiding LLM through iterative refinement. In the end, this synthesized contextual information is integrated with the original query to form a comprehensive prompt, with the answer determining through a final round of majority voting. TR-ICRL is evaluated on mainstream reasoning and knowledge-intensive tasks, where it demonstrates significant performance gains. Remarkably, TR-ICRL improves Qwen2.5-7B by 21.23% on average on MedQA and even 137.59% on AIME2024. Extensive ablation studies and analyses further validate the effectiveness and robustness of our approach. Our code is available at this https URL. 

---
# Locally Confident, Globally Stuck: The Quality-Exploration Dilemma in Diffusion Language Models 

**Authors**: Liancheng Fang, Aiwei Liu, Henry Peng Zou, Yankai Chen, Enze Ma, Leyi Pan, Chunyu Miao, Wei-Chieh Huang, Xue Liu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00375)  

**Abstract**: Diffusion large language models (dLLMs) theoretically permit token decoding in arbitrary order, a flexibility that could enable richer exploration of reasoning paths than autoregressive (AR) LLMs. In practice, however, random-order decoding often hurts generation quality. To mitigate this, low-confidence remasking improves single-sample quality (e.g., Pass@$1$) by prioritizing confident tokens, but it also suppresses exploration and limits multi-sample gains (e.g., Pass@$k$), creating a fundamental quality--exploration dilemma. In this paper, we provide a unified explanation of this dilemma. We show that low-confidence remasking improves a myopic proxy for quality while provably constraining the entropy of the induced sequence distribution. To overcome this limitation, we characterize the optimal distribution that explicitly balances quality and exploration, and develop a simple Independent Metropolis--Hastings sampler that approximately targets this distribution during decoding. Experiments across a range of reasoning benchmarks including MATH500, AIME24/25, HumanEval, and MBPP show that our approach yields better exploration-quality tradeoff than both random and low-confidence remasking. 

---
# Agent Q-Mix: Selecting the Right Action for LLM Multi-Agent Systems through Reinforcement Learning 

**Authors**: Eric Hanchen Jiang, Levina Li, Rui Sun, Xiao Liang, Yubei Li, Yuchen Wu, Haozheng Luo, Hengli Li, Zhi Zhang, Zhaolu Kang, Kai-Wei Chang, Ying Nian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00344)  

**Abstract**: Large Language Models (LLMs) have shown remarkable performance in completing various tasks. However, solving complex problems often requires the coordination of multiple agents, raising a fundamental question: how to effectively select and interconnect these agents. In this paper, we propose \textbf{Agent Q-Mix}, a reinforcement learning framework that reformulates topology selection as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. Our method learns decentralized communication decisions using QMIX value factorization, where each agent selects from a set of communication actions that jointly induce a round-wise communication graph. At its core, Agent Q-Mix combines a topology-aware GNN encoder, GRU memory, and per-agent Q-heads under a Centralized Training with Decentralized Execution (CTDE) paradigm. The framework optimizes a reward function that balances task accuracy with token cost. Across seven core benchmarks in coding, reasoning, and mathematics, Agent Q-Mix achieves the highest average accuracy compared to existing methods while demonstrating superior token efficiency and robustness against agent failure. Notably, on the challenging Humanity's Last Exam (HLE) using Gemini-3.1-Flash-Lite as a backbone, Agent Q-Mix achieves 20.8\% accuracy, outperforming Microsoft Agent Framework (19.2\%) and LangGraph (19.2\%), followed by AutoGen and Lobster by OpenClaw. These results underscore the effectiveness of learned, decentralized topology optimization in pushing the boundaries of multi-agent reasoning. 

---
# Large Language Models in the Abuse Detection Pipeline 

**Authors**: Suraj Kath, Sanket Badhe, Preet Shah, Ashwin Sampathkumar, Shivani Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.00323)  

**Abstract**: Online abuse has grown increasingly complex, spanning toxic language, harassment, manipulation, and fraudulent behavior. Traditional machine-learning approaches dependent on static classifiers and labor-intensive labeling struggle to keep pace with evolving threat patterns and nuanced policy requirements. Large Language Models introduce new capabilities for contextual reasoning, policy interpretation, explanation generation, and cross-modal understanding, enabling them to support multiple stages of modern safety systems. This survey provides a lifecycle-oriented analysis of how LLMs are being integrated into the Abuse Detection Lifecycle (ADL), which we define across four stages: (I) Label \& Feature Generation, (II) Detection, (III) Review \& Appeals, and (IV) Auditing \& Governance. For each stage, we synthesize emerging research and industry practices, highlight architectural considerations for production deployment, and examine the strengths and limitations of LLM-driven approaches. We conclude by outlining key challenges including latency, cost-efficiency, determinism, adversarial robustness, and fairness and discuss future research directions needed to operationalize LLMs as reliable, accountable components of large-scale abuse-detection and governance systems. 

---
# Asymmetric Actor-Critic for Multi-turn LLM Agents 

**Authors**: Shuli Jiang, Zhaoyang Zhang, Yi Zhang, Shuo Yang, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2604.00304)  

**Abstract**: Large language models (LLMs) exhibit strong reasoning and conversational abilities, but ensuring reliable behavior in multi-turn interactions remains challenging. In many real-world applications, agents must succeed in one-shot settings where retries are impossible. Existing approaches either rely on reflection or post-hoc evaluation, which require additional attempts, or assume fully trainable models that cannot leverage proprietary LLMs. We propose an asymmetric actor-critic framework for reliable conversational agents. A powerful proprietary LLM acts as the actor, while a smaller open-source critic provides runtime supervision, monitoring the actor's actions and intervening within the same interaction trajectory. Unlike training-based actor-critic methods, our framework supervises a fixed actor operating in open-ended conversational environments. The design leverages a generation-verification asymmetry: while high-quality generation requires large models, effective oversight can often be achieved by smaller ones. We further introduce a data generation pipeline that produces supervision signals for critic fine-tuning without modifying the actor. Experiments on $\tau$-bench and UserBench show that our approach significantly improves reliability and task success over strong single-agent baselines. Moreover, lightweight open-source critics rival or surpass larger proprietary models in the critic role, and critic fine-tuning yields additional gains over several state-of-the-art methods. 

---
# Frege in the Flesh: Biolinguistics and the Neural Enforcement of Syntactic Structures 

**Authors**: Elliot Murphy  

**Link**: [PDF](https://arxiv.org/pdf/2604.00291)  

**Abstract**: Biolinguistics is the interdisciplinary scientific study of the biological foundations, evolution, and genetic basis of human language. It treats language as an innate biological organ or faculty of the mind, rather than a cultural tool, and it challenges a behaviorist conception of human language acquisition as being based on stimulus-response associations. Extracting its most essential component, it takes seriously the idea that mathematical, algebraic models of language capture something natural about the world. The syntactic structure-building operation of MERGE is thought to offer the scientific community a "real joint of nature", "a (new) aspect of nature" (Mukherji 2010), not merely a formal artefact. This mathematical theory of language is then seen as being able to offer biologists, geneticists and neuroscientists clearer instructions for how to explore language. The argument of this chapter proceeds in four steps. First, I clarify the object of inquiry for biolinguistics: not speech, communication, or generic sequence processing, but the internal computational system that generates hierarchically structured expressions. Second, I argue that this formal characterization matters for evolutionary explanation, because different conceptions of syntax imply different standards of what must be explained. Third, I suggest that a sufficiently explicit algebraic account of syntax places non-trivial constraints on candidate neural mechanisms. Finally, I consider how recent neurocomputational work begins to transform these constraints into empirically tractable hypotheses, while also noting the speculative and revisable character of the present program. 

---
# Can Large Language Models Self-Correct in Medical Question Answering? An Exploratory Study 

**Authors**: Zaifu Zhan, Mengyuan Cui, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00261)  

**Abstract**: Large language models (LLMs) have achieved strong performance on medical question answering (medical QA), and chain-of-thought (CoT) prompting has further improved results by eliciting explicit intermediate reasoning; meanwhile, self-reflective (self-corrective) prompting has been widely claimed to enhance model reliability by prompting LLMs to critique and revise their own reasoning, yet its effectiveness in safety-critical medical settings remains unclear. In this work, we conduct an exploratory analysis of self-reflective reasoning for medical multiple-choice question answering: using GPT-4o and GPT-4o-mini, we compare standard CoT prompting with an iterative self-reflection loop and track how predictions evolve across reflection steps on three widely used medical QA benchmarks (MedQA, HeadQA, and PubMedQA). We analyze whether self-reflection leads to error correction, error persistence, or the introduction of new errors. Our results show that self-reflective prompting does not consistently improve accuracy and its impact is highly dataset- and model-dependent: it yields modest gains on MedQA but provides limited or negative benefits on HeadQA and PubMedQA, and increasing the number of reflection steps does not guarantee better performance. These findings highlight a gap between reasoning transparency and reasoning correctness, suggesting that self-reflective reasoning is better viewed as an analytical tool for understanding model behavior rather than a standalone solution for improving medical QA reliability. 

---
# LLM Essay Scoring Under Holistic and Analytic Rubrics: Prompt Effects and Bias 

**Authors**: Filip J. Kucia, Anirban Chakraborty, Anna Wróblewska  

**Link**: [PDF](https://arxiv.org/pdf/2604.00259)  

**Abstract**: Despite growing interest in using Large Language Models (LLMs) for educational assessment, it remains unclear how closely they align with human scoring. We present a systematic evaluation of instruction-tuned LLMs across three open essay-scoring datasets (ASAP 2.0, ELLIPSE, and DREsS) that cover both holistic and analytic scoring. We analyze agreement with human consensus scores, directional bias, and the stability of bias estimates. Our results show that strong open-weight models achieve moderate to high agreement with humans on holistic scoring (Quadratic Weighted Kappa about 0.6), but this does not transfer uniformly to analytic scoring. In particular, we observe large and stable negative directional bias on Lower-Order Concern (LOC) traits, such as Grammar and Conventions, meaning that models often score these traits more harshly than human raters. We also find that concise keyword-based prompts generally outperform longer rubric-style prompts in multi-trait analytic scoring. To quantify the amount of data needed to detect these systematic deviations, we compute the minimum sample size at which a 95% bootstrap confidence interval for the mean bias excludes zero. This analysis shows that LOC bias is often detectable with very small validation sets, whereas Higher-Order Concern (HOC) traits typically require much larger samples. These findings support a bias-correction-first deployment strategy: instead of relying on raw zero-shot scores, systematic score offsets can be estimated and corrected using small human-labeled bias-estimation sets, without requiring large-scale fine-tuning. 

---
# REM-CTX: Automated Peer Review via Reinforcement Learning with Auxiliary Context 

**Authors**: Pawin Taechoyotin, Daniel E. Acuna  

**Link**: [PDF](https://arxiv.org/pdf/2604.00248)  

**Abstract**: Most automated peer review systems rely on textual manuscript content alone, leaving visual elements such as figures and external scholarly signals underutilized. We introduce REM-CTX, a reinforcement-learning system that incorporates auxiliary context into the review generation process via correspondence-aware reward functions. REM-CTX trains an 8B-parameter language model with Group Relative Policy Optimization (GRPO) and combines a multi-aspect quality reward with two correspondence rewards that explicitly encourage alignment with auxiliary context. Experiments on manuscripts across Computer, Biological, and Physical Sciences show that REM-CTX achieves the highest overall review quality among six baselines, outperforming other systems with substantially larger commercial models, and surpassing the next-best RL baseline across both quality and contextual grounding metrics. Ablation studies confirm that the two correspondence rewards are complementary: each selectively improves its targeted correspondence reward while preserving all quality dimensions, and the full model outperforms all partial variants. Analysis of training dynamics reveals that the criticism aspect is negatively correlated with other metrics during training, suggesting that future studies should group multi-dimension rewards for review generation. 

---
# A Taxonomy of Programming Languages for Code Generation 

**Authors**: Nishat Raihan, Christian Newman, Marcos Zampieri  

**Link**: [PDF](https://arxiv.org/pdf/2604.00239)  

**Abstract**: The world's 7,000+ languages vary widely in the availability of resources for NLP, motivating efforts to systematically categorize them by their degree of resourcefulness (Joshi et al., 2020). A similar disparity exists among programming languages (PLs); however, no resource-tier taxonomy has been established for code. As large language models (LLMs) grow increasingly capable of generating code, such a taxonomy becomes essential. To fill this gap, we present the first reproducible PL resource classification, grouping 646 languages into four tiers. We show that only 1.9% of languages (Tier 3, High) account for 74.6% of all tokens in seven major corpora, while 71.7% of languages (Tier 0, Scarce) contribute just 1.0%. Statistical analyses of within-tier inequality, dispersion, and distributional skew confirm that this imbalance is both extreme and systematic. Our results provide a principled framework for dataset curation and tier-aware evaluation of multilingual LLMs. 

---
# Do Language Models Know When They'll Refuse? Probing Introspective Awareness of Safety Boundaries 

**Authors**: Tanay Gondil  

**Link**: [PDF](https://arxiv.org/pdf/2604.00228)  

**Abstract**: Large language models are trained to refuse harmful requests, but can they accurately predict when they will refuse before responding? We investigate this question through a systematic study where models first predict their refusal behavior, then respond in a fresh context. Across 3754 datapoints spanning 300 requests, we evaluate four frontier models: Claude Sonnet 4, Claude Sonnet 4.5, GPT-5.2, and Llama 3.1 405B. Using signal detection theory (SDT), we find that all models exhibit high introspective sensitivity (d' = 2.4-3.5), but sensitivity drops substantially at safety boundaries. We observe generational improvement within Claude (Sonnet 4.5: 95.7 percent accuracy vs Sonnet 4: 93.0 percent), while GPT-5.2 shows lower accuracy (88.9 percent) with more variable behavior. Llama 405B achieves high sensitivity but exhibits strong refusal bias and poor calibration, resulting in lower overall accuracy (80.0 percent). Topic-wise analysis reveals weapons-related queries are consistently hardest for introspection. Critically, confidence scores provide actionable signal: restricting to high-confidence predictions yields 98.3 percent accuracy for well-calibrated models, enabling practical confidence-based routing for safety-critical deployments. 

---
# Do LLMs Know What Is Private Internally? Probing and Steering Contextual Privacy Norms in Large Language Model Representations 

**Authors**: Haoran Wang, Li Xiong, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00209)  

**Abstract**: Large language models (LLMs) are increasingly deployed in high-stakes settings, yet they frequently violate contextual privacy by disclosing private information in situations where humans would exercise discretion. This raises a fundamental question: do LLMs internally encode contextual privacy norms, and if so, why do violations persist? We present the first systematic study of contextual privacy as a structured latent representation in LLMs, grounded in contextual integrity (CI) theory. Probing multiple models, we find that the three norm-determining CI parameters (information type, recipient, and transmission principle) are encoded as linearly separable and functionally independent directions in activation space. Despite this internal structure, models still leak private information in practice, revealing a clear gap between concept representation and model behavior. To bridge this gap, we introduce CI-parametric steering, which independently intervenes along each CI dimension. This structured control reduces privacy violations more effectively and predictably than monolithic steering. Our results demonstrate that contextual privacy failures arise from misalignment between representation and behavior rather than missing awareness, and that leveraging the compositional structure of CI enables more reliable contextual privacy control, shedding light on potential improvement of contextual privacy understanding in LLMs. 

---
# Polish phonology and morphology through the lens of distributional semantics 

**Authors**: Paula Orzechowska, R. Harald Baayen  

**Link**: [PDF](https://arxiv.org/pdf/2604.00174)  

**Abstract**: This study investigates the relationship between the phonological and morphological structure of Polish words and their meanings using Distributional Semantics. In the present analysis, we ask whether there is a relationship between the form properties of words containing consonant clusters and their meanings. Is the phonological and morphonological structure of complex words mirrored in semantic space? We address these questions for Polish, a language characterized by non-trivial morphology and an impressive inventory of morphologically-motivated consonant clusters. We use statistical and computational techniques, such as t-SNE, Linear Discriminant Analysis and Linear Discriminative Learning, and demonstrate that -- apart from encoding rich morphosyntactic information (e.g. tense, number, case) -- semantic vectors capture information on sub-lexical linguistic units such as phoneme strings. First, phonotactic complexity, morphotactic transparency, and a wide range of morphosyntactic categories available in Polish (case, gender, aspect, tense, number) can be predicted from embeddings without requiring any information about the forms of words. Second, we argue that computational modelling with the discriminative lexicon model using embeddings can provide highly accurate predictions for comprehension and production, exactly because of the existence of extensive information in semantic space that is to a considerable extent isomorphic with structure in the form space. 

---
# Oblivion: Self-Adaptive Agentic Memory Control through Decay-Driven Activation 

**Authors**: Ashish Rana, Chia-Chien Hung, Qumeng Sun, Julian Martin Kunkel, Carolin Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2604.00131)  

**Abstract**: Human memory adapts through selective forgetting: experiences become less accessible over time but can be reactivated by reinforcement or contextual cues. In contrast, memory-augmented LLM agents rely on "always-on" retrieval and "flat" memory storage, causing high interference and latency as histories grow. We introduce Oblivion, a memory control framework that casts forgetting as decay-driven reductions in accessibility, not explicit deletion. Oblivion decouples memory control into read and write paths. The read path decides when to consult memory, based on agent uncertainty and memory buffer sufficiency, avoiding redundant always-on access. The write path decides what to strengthen, by reinforcing memories contributing to forming the response. Together, this enables hierarchical memory organization that maintains persistent high-level strategies while dynamically loading details as needed. We evaluate on both static and dynamic long-horizon interaction benchmarks. Results show that Oblivion dynamically adapts memory access and reinforcement, balancing learning and forgetting under shifting contexts, highlighting that memory control is essential for effective LLM-agentic reasoning. The source code is available at this https URL. 

---
# Hierarchical Chain-of-Thought Prompting: Enhancing LLM Reasoning Performance and Efficiency 

**Authors**: Xingshuai Huang, Derek Li, Bahareh Nikpour, Parsa Omidi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00130)  

**Abstract**: Chain-of-Thought (CoT) prompting has significantly improved the reasoning capabilities of large language models (LLMs). However, conventional CoT often relies on unstructured, flat reasoning chains that suffer from redundancy and suboptimal performance. In this work, we introduce Hierarchical Chain-of-Thought (Hi-CoT) prompting, a structured reasoning paradigm specifically designed to address the challenges of complex, multi-step reasoning. Hi-CoT decomposes the reasoning process into hierarchical substeps by alternating between instructional planning and step-by-step execution. This decomposition enables LLMs to better manage long reasoning horizons and maintain logical coherence. Extensive evaluations across diverse LLMs and mathematical reasoning benchmarks show that Hi-CoT consistently improves average accuracy by 6.2% (up to 61.4% on certain models and tasks) while reducing reasoning trace length by 13.9% compared to CoT prompting. We further show that accuracy and efficiency are maximized when models strictly adhere to the hierarchical structure. Our code is available at this https URL. 

---
# Multi-lingual Multi-institutional Electronic Health Record based Predictive Model 

**Authors**: Kyunghoon Hur, Heeyoung Kwak, Jinsu Jang, Nakhwan Kim, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00027)  

**Abstract**: Large-scale EHR prediction across institutions is hindered by substantial heterogeneity in schemas and code systems. Although Common Data Models (CDMs) can standardize records for multi-institutional learning, the manual harmonization and vocabulary mapping are costly and difficult to scale. Text-based harmonization provides an alternative by converting raw EHR into a unified textual form, enabling pooled learning without explicit standardization. However, applying this paradigm to multi-national datasets introduces an additional layer of heterogeneity, which is "language" that must be addressed for truly scalable EHRs learning. In this work, we investigate multilingual multi-institutional learning for EHR prediction, aiming to enable pooled training across multinational ICU datasets without manual standardization. We compare two practical strategies for handling language barriers: (i) directly modeling multilingual records with multilingual encoders, and (ii) translating non-English records into English via LLM-based word-level translation. Across seven public ICU datasets, ten clinical tasks with multiple prediction windows, translation-based lingual alignment yields more reliable cross-dataset performance than multilingual encoders. The multi-institutional learning model consistently outperforms strong baselines that require manual feature selection and harmonization, and also surpasses single-dataset training. We further demonstrate that text-based framework with lingual alignment effectively performs transfer learning via few-shot fine-tuning, with additional gains. To our knowledge, this is the first study to aggregate multilingual multinational ICU EHR datasets into one predictive model, providing a scalable path toward language-agnostic clinical prediction and future global multi-institutional EHR research. 

---
# "Who Am I, and Who Else Is Here?" Behavioral Differentiation Without Role Assignment in Multi-Agent LLM Systems 

**Authors**: Houssam EL Kandoussi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00026)  

**Abstract**: When multiple large language models interact in a shared conversation, do they develop differentiated social roles or converge toward uniform behavior? We present a controlled experimental platform that orchestrates simultaneous multi-agent discussions among 7 heterogeneous LLMs on a unified inference backend, systematically varying group composition, naming conventions, and prompt structure across 12 experimental series (208 runs, 13,786 coded messages). Each message is independently coded on six behavioral flags by two LLM judges from distinct model families (Gemini 3.1 Pro and Claude Sonnet 4.6), achieving mean Cohen's kappa = 0.78 with conservative intersection-based adjudication. Human validation on 609 randomly stratified messages confirmed coding reliability (mean kappa = 0.73 vs. Gemini). We find that (1) heterogeneous groups exhibit significantly richer behavioral differentiation than homogeneous groups (cosine similarity 0.56 vs. 0.85; p < 10^-5, r = 0.70); (2) groups spontaneously exhibit compensatory response patterns when an agent crashes; (3) revealing real model names significantly increases behavioral convergence (cosine 0.56 to 0.77, p = 0.001); and (4) removing all prompt scaffolding converges profiles to homogeneous-level similarity (p < 0.001). Critically, these behaviors are absent when agents operate in isolation, confirming that behavioral diversity is a structured, reproducible phenomenon driven by the interaction of architectural heterogeneity, group context, and prompt-level scaffolding. 

---
# Brevity Constraints Reverse Performance Hierarchies in Language Models 

**Authors**: MD Azizul Hakim  

**Link**: [PDF](https://arxiv.org/pdf/2604.00025)  

**Abstract**: Standard evaluation protocols reveal a counterintuitive phenomenon: on 7.7% of benchmark problems spanning five datasets, larger language models underperform smaller ones by 28.4 percentage points despite 10-100x more parameters. Through systematic evaluation of 31 models (0.5B-405B parameters) across 1,485 problems, we identify the mechanism as spontaneous scale-dependent verbosity that introduces errors through overelaboration. Causal intervention experiments demonstrate this reflects correctable prompt design rather than fundamental capability limitations. Constraining large models to produce brief responses improves accuracy by 26 percentage points and reduces performance gaps by up to two-thirds. Most critically, brevity constraints completely reverse performance hierarchies on mathematical reasoning and scientific knowledge benchmarks, with large models achieving 7.7-15.9 percentage point advantages over small models -- direct inversions of the original gaps. These reversals prove large models possess superior latent capabilities that universal prompting masks. We validate findings through three independent contamination tests and demonstrate inverse scaling operates continuously across the full parameter spectrum, with dataset-specific optimal scales ranging from 0.5B to 3.0B parameters. Our results establish that maximizing large model performance requires scale-aware prompt engineering rather than universal evaluation protocols, with immediate implications for deployment: prompt adaptation simultaneously improves accuracy and reduces computational costs. 

---
# WHBench: Evaluating Frontier LLMs with Expert-in-the-Loop Validation on Women's Health Topics 

**Authors**: Sneha Maurya, Pragya Saboo, Girish Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.00024)  

**Abstract**: Large language models are increasingly used for medical guidance, but women's health remains under-evaluated in benchmark design. We present the Women's Health Benchmark (WHBench), a targeted evaluation suite of 47 expert-crafted scenarios across 10 women's health topics, designed to expose clinically meaningful failure modes including outdated guidelines, unsafe omissions, dosing errors, and equity-related blind spots. We evaluate 22 models using a 23-criterion rubric spanning clinical accuracy, completeness, safety, communication quality, instruction following, equity, uncertainty handling, and guideline adherence, with safety-weighted penalties and server-side score recalculation. Across 3,102 attempted responses (3,100 scored), no model mean performance exceeds 75 percent; the best model reaches 72.1 percent. Even top models show low fully correct rates and substantial variation in harm rates. Inter-rater reliability is moderate at the response label level but high for model ranking, supporting WHBench utility for comparative system evaluation while highlighting the need for expert oversight in clinical deployment. WHBench provides a public, failure-mode-aware benchmark to track safer and more equitable progress in womens health AI. 

---
# Phonological Fossils: Machine Learning Detection of Non-Mainstream Vocabulary in Sulawesi Basic Lexicon 

**Authors**: Mukhlis Amien, Go Frendi Gunawan  

**Link**: [PDF](https://arxiv.org/pdf/2604.00023)  

**Abstract**: Basic vocabulary in many Sulawesi Austronesian languages includes forms resisting reconstruction to any proto-form with phonological patterns inconsistent with inherited roots, but whether this non-conforming vocabulary represents pre-Austronesian substrate or independent innovation has not been tested computationally. We combine rule-based cognate subtraction with a machine learning classifier trained on phonological features. Using 1,357 forms from six Sulawesi languages in the Austronesian Basic Vocabulary Database, we identify 438 candidate substrate forms (26.5%) through cognate subtraction and Proto-Austronesian cross-checking. An XGBoost classifier trained on 26 phonological features distinguishes inherited from non-mainstream forms with AUC=0.763, revealing a phonological fingerprint: longer forms, more consonant clusters, higher glottal stop rates, and fewer Austronesian prefixes. Cross-method consensus (Cohen's kappa=0.61) identifies 266 high-confidence non-mainstream candidates. However, clustering yields no coherent word families (silhouette=0.114; cross-linguistic cognate test p=0.569), providing no evidence for a single pre-Austronesian language layer. Application to 16 additional languages confirms geographic patterning: Sulawesi languages show higher predicted non-mainstream rates (mean P_sub=0.606) than Western Indonesian languages (0.393). This study demonstrates that phonological machine learning can complement traditional comparative methods in detecting non-mainstream lexical layers, while cautioning against interpreting phonological non-conformity as evidence for a shared substrate language. 

---
# Criterion Validity of LLM-as-Judge for Business Outcomes in Conversational Commerce 

**Authors**: Liang Chen, Qi Liu, Wenhuan Lin, Feng Liang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00022)  

**Abstract**: Multi-dimensional rubric-based dialogue evaluation is widely used to assess conversational AI, yet its criterion validity -- whether quality scores are associated with the downstream outcomes they are meant to serve -- remains largely untested. We address this gap through a two-phase study on a major Chinese matchmaking platform, testing a 7-dimension evaluation rubric (implemented via LLM-as-Judge) against verified business conversion. Our findings concern rubric design and weighting, not LLM scoring accuracy: any judge using the same rubric would face the same structural issue. The core finding is dimension-level heterogeneity: in Phase 2 (n=60 human conversations, stratified sample, verified labels), Need Elicitation (D1: rho=0.368, p=0.004) and Pacing Strategy (D3: rho=0.354, p=0.006) are significantly associated with conversion after Bonferroni correction, while Contextual Memory (D5: rho=0.018, n.s.) shows no detectable association. This heterogeneity causes the equal-weighted composite (rho=0.272) to underperform its best dimensions -- a composite dilution effect that conversion-informed reweighting partially corrects (rho=0.351). Logistic regression controlling for conversation length confirms D3's association strengthens (OR=3.18, p=0.006), ruling out a length confound. An initial pilot (n=14) mixing human and AI conversations had produced a misleading "evaluation-outcome paradox," which Phase 2 revealed as an agent-type confound artifact. Behavioral analysis of 130 conversations through a Trust-Funnel framework identifies a candidate mechanism: AI agents execute sales behaviors without building user trust. We operationalize these findings in a three-layer evaluation architecture and advocate criterion validity testing as standard practice in applied dialogue evaluation. 

---
# How Do Language Models Process Ethical Instructions? Deliberation, Consistency, and Other-Recognition Across Four Models 

**Authors**: Hiroki Fukui  

**Link**: [PDF](https://arxiv.org/pdf/2604.00021)  

**Abstract**: Alignment safety research assumes that ethical instructions improve model behavior, but how language models internally process such instructions remains unknown. We conducted over 600 multi-agent simulations across four models (Llama 3.3 70B, GPT-4o mini, Qwen3-Next-80B-A3B, Sonnet 4.5), four ethical instruction formats (none, minimal norm, reasoned norm, virtue framing), and two languages (Japanese, English). Confirmatory analysis fully replicated the Llama Japanese dissociation pattern from a prior study ($\mathrm{BF}_{10} > 10$ for all three hypotheses), but none of the other three models reproduced this pattern, establishing it as model-specific. Three new metrics -- Deliberation Depth (DD), Value Consistency Across Dilemmas (VCAD), and Other-Recognition Index (ORI) -- revealed four distinct ethical processing types: Output Filter (GPT; safe outputs, no processing), Defensive Repetition (Llama; high consistency through formulaic repetition), Critical Internalization (Qwen; deep deliberation, incomplete integration), and Principled Consistency (Sonnet; deliberation, consistency, and other-recognition co-occurring). The central finding is an interaction between processing capacity and instruction format: in low-DD models, instruction format has no effect on internal processing; in high-DD models, reasoned norms and virtue framing produce opposite effects. Lexical compliance with ethical instructions did not correlate with any processing metric at the cell level ($r = -0.161$ to $+0.256$, all $p > .22$; $N = 24$; power limited), suggesting that safety, compliance, and ethical processing are largely dissociable. These processing types show structural correspondence to patterns observed in clinical offender treatment, where formal compliance without internal processing is a recognized risk signal. 

---
# Detecting Abnormal User Feedback Patterns through Temporal Sentiment Aggregation 

**Authors**: Yalun Qi, Sichen Zhao, Zhiming Xue, Xianling Zeng, Zihan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00020)  

**Abstract**: In many real-world applications, such as customer feedback monitoring, brand reputation management, and product health tracking, understanding the temporal dynamics of user sentiment is crucial for early detection of anomalous events such as malicious review campaigns or sudden declines in user satisfaction. Traditional sentiment analysis methods focus on individual text classification, which is insufficient to capture collective behavioral shifts over time due to inherent noise and class imbalance in short user comments. In this work, we propose a temporal sentiment aggregation framework that leverages pretrained transformer-based language models to extract per-comment sentiment signals and aggregates them into time-window-level scores. Significant downward shifts in these aggregated scores are interpreted as potential anomalies in user feedback patterns. We adopt RoBERTa as our core semantic feature extractor and demonstrate, through empirical evaluation on real social media data, that the aggregated sentiment scores reveal meaningful trends and support effective anomaly detection. Experiments on real-world social media data demonstrate that our method successfully identifies statistically significant sentiment drops that correspond to coherent complaint patterns, providing an effective and interpretable solution for feedback anomaly monitoring. 

---
# The Chronicles of RiDiC: Generating Datasets with Controlled Popularity Distribution for Long-form Factuality Evaluation 

**Authors**: Pavel Braslavski, Dmitrii Iarosh, Nikita Sushko, Andrey Sakhovskiy, Vasily Konovalov, Elena Tutubalina, Alexander Panchenko  

**Link**: [PDF](https://arxiv.org/pdf/2604.00019)  

**Abstract**: We present a configurable pipeline for generating multilingual sets of entities with specified characteristics, such as domain, geographical location and popularity, using data from Wikipedia and Wikidata. These datasets are intended for evaluating the factuality of LLMs' long-form generation, thereby complementing evaluation based on short-form QA datasets. We present the RiDiC dataset as an example of this approach. RiDiC contains 3,000 entities from three domains -- rivers, natural disasters, and car models -- spanning different popularity tiers. Each entity is accompanied by its geographical location, English and Chinese names (if available) and relevant English and Chinese Wikipedia content, which is used to evaluate LLMs' responses. Generations about RiDiC entities were obtained from three LLMs in English and Chinese. These were then evaluated using a third-party factuality checker, which showed that entities from our dataset caused even frontier models to hallucinate. To facilitate the evaluation of LLMs' long-form factuality in multiple languages, the code, data, and generation/evaluation scripts have been released. 

---
# Think Twice Before You Write -- an Entropy-based Decoding Strategy to Enhance LLM Reasoning 

**Authors**: Jiashu He, Meizhu Liu, Olaitan P Olaleye, Amit Agarwal, M. Avendi, Yassi Abbasi, Matthew Rowe, Hitesh Laxmichand Patel, Paul Li, Tao Sheng, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2604.00018)  

**Abstract**: Decoding strategies play a central role in shaping the reasoning ability of large language models (LLMs). Traditional methods such as greedy decoding and beam search often suffer from error propagation, while sampling-based approaches introduce randomness without adequate robustness. Self-consistency improves reliability by aggregating multiple rollouts, but incurs significant computational overhead. We propose an entropy-guided decoding framework that introduces token-level adaptivity into generation. At each step, the model computes the entropy of the token distribution, identifies high-uncertainty positions, and selectively branches on these vulnerable points. A dynamic pool of partial rollouts is maintained and expanded until solutions are completed, concentrating computation where uncertainty is greatest and avoiding unnecessary exploration in confident regions. To enable efficient termination, we apply a rollout-level Entropy After </Think> (EAT) stopping criterion by performing entropy evaluation after the full reasoning trace, rather than incrementally at every step. Experiments on GSM8K, AMC2023, and their perturbed variants demonstrate that our method achieves consistently strong accuracy. Notably, on smaller LLMs, performance is comparable to GPT-5 while operating at a fraction of the cost. 

---
# Semantic Shifts of Psychological Concepts in Scientific and Popular Media Discourse: A Distributional Semantics Analysis of Russian-Language Corpora 

**Authors**: Orlova Anastasia  

**Link**: [PDF](https://arxiv.org/pdf/2604.00017)  

**Abstract**: This article examines semantic shifts in psychological concepts across scientific and popular media discourse using methods of distributional semantics applied to Russian-language corpora. Two corpora were compiled: a scientific corpus of approximately 300 research articles from the journals Psychology. Journal of the Higher School of Economics and Vestnik of Saint Petersburg University. Psychology (767,543 tokens) and a popular science corpus consisting of texts from the online psychology platforms Yasno and Chistye kogntsii (1,199,150 tokens). After preprocessing (OCR recognition, lemmatization, removal of stop words and non-informative characters), the corpora were analyzed through frequency analysis, clustering, and the identification of semantic associations. The results reveal significant differences in vocabulary and conceptual framing between the two discourse types: scientific texts emphasize methodological and clinical terminology, while popular science materials foreground everyday experience and therapeutic practice. A comparison of semantic associations for key concepts such as burnout and depression shows that scientific discourse links these terms to psychological resources, symptomatology, and diagnostic constructs, whereas popular science discourse frames them through personal narratives, emotions, and everyday situations. These findings demonstrate a clear shift from precise professional terminology toward more generalized and experiential meanings in popular media discourse and confirm the effectiveness of distributional semantics methods for identifying semantic transformations of psychological concepts across different communicative contexts. 

---
# Are they human? Detecting large language models by probing human memory constraints 

**Authors**: Simon Schug, Brenden M. Lake  

**Link**: [PDF](https://arxiv.org/pdf/2604.00016)  

**Abstract**: The validity of online behavioral research relies on study participants being human rather than machine. In the past, it was possible to detect machines by posing simple challenges that were easily solved by humans but not by machines. General-purpose agents based on large language models (LLMs) can now solve many of these challenges, threatening the validity of online behavioral research. Here we explore the idea of detecting humanness by using tasks that machines can solve too well to be human. Specifically, we probe for the existence of an established human cognitive constraint: limited working memory capacity. We show that cognitive modeling on a standard serial recall task can be used to distinguish online participants from LLMs even when the latter are specifically instructed to mimic human working memory constraints. Our results demonstrate that it is viable to use well-established cognitive phenomena to distinguish LLMs from humans. 

---
# ASCAT: An Arabic Scientific Corpus and Benchmark for Advanced Translation Evaluation 

**Authors**: Serry Sibaee, Khloud Al Jallad, Zineb Yousfi, Israa Elsayed Elhosiny, Yousra El-Ghawi, Batool Balah, Omer Nacar  

**Link**: [PDF](https://arxiv.org/pdf/2604.00015)  

**Abstract**: We present ASCAT (Arabic Scientific Corpus for Advanced Translation), a high-quality English-Arabic parallel benchmark corpus designed for scientific translation evaluation constructed through a systematic multi-engine translation and human validation pipeline. Unlike existing Arabic-English corpora that rely on short sentences or single-domain text, ASCAT targets full scientific abstracts averaging 141.7 words (English) and 111.78 words (Arabic), drawn from five scientific domains: physics, mathematics, computer science, quantum mechanics, and artificial intelligence. Each abstract was translated using three complementary architectures generative AI (Gemini), transformer-based models (Hugging Face \texttt{quickmt-en-ar}), and commercial MT APIs (Google Translate, DeepL) and subsequently validated by domain experts at the lexical, syntactic, and semantic levels. The resulting corpus contains 67,293 English tokens and 60,026 Arabic tokens, with an Arabic vocabulary of 17,604 unique words reflecting the morphological richness of the language. We benchmark three state-of-the-art LLMs on the corpus GPT-4o-mini (BLEU: 37.07), Gemini-3.0-Flash-Preview (BLEU: 30.44), and Qwen3-235B-A22B (BLEU: 23.68) demonstrating its discriminative power as an evaluation benchmark. ASCAT addresses a critical gap in scientific MT resources for Arabic and is designed to support rigorous evaluation of scientific translation quality and training of domain-specific translation models. 

---
# Disentangling Prompt Element Level Risk Factors for Hallucinations and Omissions in Mental Health LLM Responses 

**Authors**: Congning Ni, Sarvech Qadir, Bryan Steitz, Mihir Sachin Vaidya, Qingyuan Song, Lantian Xia, Shelagh Mulvaney, Siru Liu, Hyeyoung Ryu, Leah Hecht, Amy Bucher, Christopher Symons, Laurie Novak, Susannah L. Rose, Murat Kantarcioglu, Bradley Malin, Zhijun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.00014)  

**Abstract**: Mental health concerns are often expressed outside clinical settings, including in high-distress help seeking, where safety-critical guidance may be needed. Consumer health informatics systems increasingly incorporate large language models (LLMs) for mental health question answering, yet many evaluations underrepresent narrative, high-distress inquiries. We introduce UTCO (User, Topic, Context, Tone), a prompt construction framework that represents an inquiry as four controllable elements for systematic stress testing. Using 2,075 UTCO-generated prompts, we evaluated Llama 3.3 and annotated hallucinations (fabricated or incorrect clinical content) and omissions (missing clinically necessary or safety-critical guidance). Hallucinations occurred in 6.5% of responses and omissions in 13.2%, with omissions concentrated in crisis and suicidal ideation prompts. Across regression, element-specific matching, and similarity-matched comparisons, failures were most consistently associated with context and tone, while user-background indicators showed no systematic differences after balancing. These findings support evaluating omissions as a primary safety outcome and moving beyond static benchmark question sets. 

---
# MSA-Thinker: Discrimination-Calibration Reasoning with Hint-Guided Reinforcement Learning for Multimodal Sentiment Analysis 

**Authors**: Miaosen Luo, Zhenhao Yang, Jieshen Long, Jinghu Sun, Yichu Liu, Sijie Mai  

**Link**: [PDF](https://arxiv.org/pdf/2604.00013)  

**Abstract**: Multimodal sentiment analysis aims to understand human emotions by integrating textual, auditory, and visual modalities. Although Multimodal Large Language Models (MLLMs) have achieved state-of-the-art performance via supervised fine-tuning (SFT), their end-to-end "black-box" nature limits interpretability. Existing methods incorporating Chain-of-Thought (CoT) reasoning are hindered by high annotation costs, while Reinforcement Learning (RL) faces challenges such as low exploration efficiency and sparse rewards, particularly on hard samples. To address these issues, we propose a novel training framework that integrates structured Discrimination-Calibration (DC) reasoning with Hint-based Reinforcement Learning. First, we perform cold-start SFT using high-quality CoT data synthesized by a teacher model (Qwen3Omni-30B), which inherently contains the DC structure. This equips the model with a reasoning paradigm that performs macro discrimination followed by fine-grained calibration from the initial stage. Building on this, we propose Hint-GRPO, which leverages the discrimination phase within the DC structure as a verifiable anchor during RL to provide directional hints for hard samples, guiding policy optimization and effectively mitigating the reward sparsity problem. Experiments on the Qwen2.5Omni-7B model demonstrate that our method not only achieves higher accuracy in fine-grained sentiment regression tasks but also generates high-quality structured reasoning chains. Crucially, it exhibits superior generalization capability in cross-domain evaluations. This enhances model interpretability while validating the positive contribution of explicit reasoning steps to model robustness, offering a new paradigm for building trustworthy and efficient sentiment analysis systems. 

---
# Finding and Reactivating Post-Trained LLMs' Hidden Safety Mechanisms 

**Authors**: Mingjie Li, Wai Man Si, Michael Backes, Yang Zhang, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00012)  

**Abstract**: Despite the impressive performance of general-purpose large language models (LLMs), they often require fine-tuning or post-training to excel at specific tasks. For instance, large reasoning models (LRMs), such as the DeepSeek-R1 series, demonstrate strong reasoning capabilities after post-training different general large language models on diverse chain-of-thought (CoT) datasets. However, this additional training frequently comes at the cost of reduced safety, as the fine-tuned or post-trained models tend to exhibit more harmful behaviors compared with the regular LLMs before post-training or fine-tuning, potentially leading to harmful outcomes due to their enhanced capabilities. Taking LRMs as an example, we first investigate the underlying cause of this safety degradation in this paper. Our analysis reveals that post-training can mask the original safety mechanisms of the base LLM, while over-amplifying representations related to their post-training ability. But luckily, we also find that LRMs' safety mechanisms still exist instead of being removed during their post-training. Based on these findings, we propose a lightweight and cost-effective solution called SafeReAct that restores the suppressed safety behaviors by aligning with LoRA adapters on a few layers. Experiments on four state-of-the-art LRMs show that our method significantly improves safety on harmful prompts without compromising reasoning performance. Besides LRMs, additional results on other domain-specific LLMs, like medical models, further confirm the generality and effectiveness of our approach. 

---
# Can LLMs Perceive Time? An Empirical Investigation 

**Authors**: Aniketh Garikaparthi  

**Link**: [PDF](https://arxiv.org/pdf/2604.00010)  

**Abstract**: Large language models cannot estimate how long their own tasks take. We investigate this limitation through four experiments across 68 tasks and four model families. Pre-task estimates overshoot actual duration by 4--7$\times$ ($p < 0.001$), with models predicting human-scale minutes for tasks completing in seconds. Relative ordering fares no better: on task pairs designed to expose heuristic reliance, models score at or below chance (GPT-5: 18\% on counter-intuitive pairs, $p = 0.033$), systematically failing when complexity labels mislead. Post-hoc recall is disconnected from reality -- estimates diverge from actuals by an order of magnitude in either direction. These failures persist in multi-step agentic settings, with errors of 5--10$\times$. The models possess propositional knowledge about duration from training but lack experiential grounding in their own inference time, with practical implications for agent scheduling, planning and time-critical scenarios. 

---
# Eyla: Toward an Identity-Anchored LLM Architecture with Integrated Biological Priors -- Vision, Implementation Attempt, and Lessons from AI-Assisted Development 

**Authors**: Arif Aditto  

**Link**: [PDF](https://arxiv.org/pdf/2604.00009)  

**Abstract**: We present the design rationale, implementation attempt, and failure analysis of Eyla, a proposed identity-anchored LLM architecture that integrates biologically-inspired subsystems -- including HiPPO-initialized state-space models, zero-initialized adapters, episodic memory retrieval, and calibrated uncertainty training -- into a unified agent operating system running on consumer hardware. Unlike existing approaches that optimize models for generic helpfulness, Eyla targets identity consistency: the ability to maintain a coherent self-model under adversarial pressure, admit uncertainty, and resist manipulation. We propose the Identity Consistency Score (ICS), a novel benchmark for evaluating this property across LLMs. We then present an honest account of attempting to implement this architecture using AI coding assistants (Claude Code, Cursor) as a non-programmer, documenting a $1,000+ failure that produced a 1.27B parameter model with 86 brain subsystems contributing less than 2% to output. Our analysis identifies five systematic failure modes of AI-assisted development for novel architectures and offers concrete recommendations. To our knowledge, this is the first paper to combine an architectural vision with a documented first-person failure analysis of AI-assisted LLM development, providing lessons for both the AI systems and AI-assisted software engineering communities. 

---
# How Trustworthy Are LLM-as-Judge Ratings for Interpretive Responses? Implications for Qualitative Research Workflows 

**Authors**: Songhee Han, Jueun Shin, Jiyoon Han, Bung-Woo Jun, Hilal Ayan Karabatman  

**Link**: [PDF](https://arxiv.org/pdf/2604.00008)  

**Abstract**: As qualitative researchers show growing interest in using automated tools to support interpretive analysis, a large language model (LLM) is often introduced into an analytic workflow as is, without systematic evaluation of interpretive quality or comparison across models. This practice leaves model selection largely unexamined despite its potential influence on interpretive outcomes. To address this gap, this study examines whether LLM-as-judge evaluations meaningfully align with human judgments of interpretive quality and can inform model-level decision making. Using 712 conversational excerpts from semi-structured interviews with K-12 mathematics teachers, we generated one-sentence interpretive responses using five widely adopted inference models: Command R+ (Cohere), Gemini 2.5 Pro (Google), GPT-5.1 (OpenAI), Llama 4 Scout-17B Instruct (Meta), and Qwen 3-32B Dense (Alibaba). Automated evaluations were conducted using AWS Bedrock's LLM-as-judge framework across five metrics, and a stratified subset of responses was independently rated by trained human evaluators on interpretive accuracy, nuance preservation, and interpretive coherence. Results show that LLM-as-judge scores capture broad directional trends in human evaluations at the model level but diverge substantially in score magnitude. Among automated metrics, Coherence showed the strongest alignment with aggregated human ratings, whereas Faithfulness and Correctness revealed systematic misalignment at the excerpt level, particularly for non-literal and nuanced interpretations. Safety-related metrics were largely irrelevant to interpretive quality. These findings suggest that LLM-as-judge methods are better suited for screening or eliminating underperforming models than for replacing human judgment, offering practical guidance for systematic comparison and selection of LLMs in qualitative research workflows. 

---
# Dynin-Omni: Omnimodal Unified Large Diffusion Language Model 

**Authors**: Jaeik Kim, Woojin Kim, Jihwan Hong, Yejoon Lee, Sieun Hyeon, Mintaek Lim, Yunseok Han, Dogeun Kim, Hoeun Lee, Hyunggeun Kim, Jaeyoung Do  

**Link**: [PDF](https://arxiv.org/pdf/2604.00007)  

**Abstract**: We present Dynin-Omni, the first masked-diffusion-based omnimodal foundation model that unifies text, image, and speech understanding and generation, together with video understanding, within a single architecture. Unlike autoregressive unified models that serialize heterogeneous modalities, or compositional unified models that require orchestration with external modality-specific decoders, Dynin-Omni natively formulates omnimodal modeling as masked diffusion over a shared discrete token space, enabling iterative refinement under bidirectional context. Dynin-Omni adopts a multi-stage training strategy with model-merging-based modality expansion and omnimodal alignment. We evaluate Dynin-Omni across 19 multimodal benchmarks spanning language reasoning, image generation and editing, video understanding, and speech recognition and synthesis. Dynin-Omni achieves 87.6 on GSM8K, 1733.6 on MME-P, 61.4 on VideoMME, 0.87 on GenEval, and 2.1 WER on LibriSpeech test-clean, consistently outperforming existing open-source unified models while remaining competitive with strong modality-specific expert systems. These results demonstrate the potential of masked diffusion as a unified paradigm for any-to-any modeling, providing a flexible foundation for real-time omnimodal systems, unified cross-modal retrieval and generation, and embodied multimodal agents. 

---
# Scalable Identification and Prioritization of Requisition-Specific Personal Competencies Using Large Language Models 

**Authors**: Wanxin Li, Denver McNeney, Nivedita Prabhu, Charlene Zhang, Renee Barr, Matthew Kitching, Khanh Dao Duc, Anthony S. Boyce  

**Link**: [PDF](https://arxiv.org/pdf/2604.00006)  

**Abstract**: AI-powered recruitment tools are increasingly adopted in personnel selection, yet they struggle to capture the requisition (req)-specific personal competencies (PCs) that distinguish successful candidates beyond job categories. We propose a large language model (LLM)-based approach to identify and prioritize req-specific PCs from reqs. Our approach integrates dynamic few-shot prompting, reflection-based self-improvement, similarity-based filtering, and multi-stage validation. Applied to a dataset of Program Manager reqs, our approach correctly identifies the highest-priority req-specific PCs with an average accuracy of 0.76, approaching human expert inter-rater reliability, and maintains a low out-of-scope rate of 0.07. 

---
# LinearARD: Linear-Memory Attention Distillation for RoPE Restoration 

**Authors**: Ning Yang, Hengyu Zhong, Wentao Wang, Baoliang Tian, Haijun Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00004)  

**Abstract**: The extension of context windows in Large Language Models is typically facilitated by scaling positional encodings followed by lightweight Continual Pre-Training (CPT). While effective for processing long sequences, this paradigm often disrupts original model capabilities, leading to performance degradation on standard short-text benchmarks. We propose LinearARD, a self-distillation method that restores Rotary Position Embeddings (RoPE)-scaled students through attention-structure consistency with a frozen native-RoPE teacher. Rather than matching opaque hidden states, LinearARD aligns the row-wise distributions of dense $Q/Q$, $K/K$, and $V/V$ self-relation matrices to directly supervise attention dynamics. To overcome the quadratic memory bottleneck of $n \times n$ relation maps, we introduce a linear-memory kernel. This kernel leverages per-token log-sum-exp statistics and fuses logit recomputation into the backward pass to compute exact Kullback-Leibler divergence and gradients. On LLaMA2-7B extended from 4K to 32K, LinearARD recovers 98.3\% of the short-text performance of state-of-the-art baselines while surpassing them on long-context benchmarks. Notably, our method achieves these results using only \textbf{4.25M} training tokens compared to the \textbf{256M} tokens required by LongReD and CPT. Our code is available at this https URL. 

---
# A Reliability Evaluation of Hybrid Deterministic-LLM Based Approaches for Academic Course Registration PDF Information Extraction 

**Authors**: Muhammad Anis Al Hilmi, Neelansh Khare, Noel Framil Iglesias  

**Link**: [PDF](https://arxiv.org/pdf/2604.00003)  

**Abstract**: This study evaluates the reliability of information extraction approaches from KRS documents using three strategies: LLM only, Hybrid Deterministic - LLM (regex + LLM), and a Camelot based pipeline with LLM fallback. Experiments were conducted on 140 documents for the LLM based test and 860 documents for the Camelot based pipeline evaluation, covering four study programs with varying data in tables and metadata. Three 12 - 14B LLM models (Gemma 3, Phi 4, and Qwen 2.5) were run locally using Ollama and a consumer grade CPU without a GPU. Evaluations used exact match (EM) and Levenshtein similarity (LS) metrics with a threshold of 0.7. Although not applicable to all models, the results show that the hybrid approach can improve efficiency compared to LLM only, especially for deterministic metadata. The Camelot based pipeline with LLM fallback produced the best combination of accuracy (EM and LS up to 0.99 - 1.00) and computational efficiency (less than 1 second per PDF in most cases). The Qwen 2.5:14b model demonstrated the most consistent performance across all scenarios. These findings confirm that integrating deterministic and LLM methods is increasingly reliable and efficient for information extraction from text based academic documents in computationally constrained environments. 

---
# Benchmark for Assessing Olfactory Perception of Large Language Models 

**Authors**: Eftychia Makri, Nikolaos Nakis, Laura Sisson, Gigi Minsky, Leandros Tassiulas, Vahid Satarifard, Nicholas A. Christakis  

**Link**: [PDF](https://arxiv.org/pdf/2604.00002)  

**Abstract**: Here we introduce the Olfactory Perception (OP) benchmark, designed to assess the capability of large language models (LLMs) to reason about smell. The benchmark contains 1,010 questions across eight task categories spanning odor classification, odor primary descriptor identification, intensity and pleasantness judgments, multi-descriptor prediction, mixture similarity, olfactory receptor activation, and smell identification from real-world odor sources. Each question is presented in two prompt formats, compound names and isomeric SMILES, to evaluate the effect of molecular representations. Evaluating 21 model configurations across major model families, we find that compound-name prompts consistently outperform isomeric SMILES, with gains ranging from +2.4 to +18.9 percentage points (mean approx +7 points), suggesting current LLMs access olfactory knowledge primarily through lexical associations rather than structural molecular reasoning. The best-performing model reaches 64.4\% overall accuracy, which highlights both emerging capabilities and substantial remaining gaps in olfactory reasoning. We further evaluate a subset of the OP across 21 languages and find that aggregating predictions across languages improves olfactory prediction, with AUROC = 0.86 for the best performing language ensemble model. LLMs should be able to handle olfactory and not just visual or aural information. 

---
# True (VIS) Lies: Analyzing How Generative AI Recognizes Intentionality, Rhetoric, and Misleadingness in Visualization Lies 

**Authors**: Graziano Blasilli, Marco Angelini  

**Link**: [PDF](https://arxiv.org/pdf/2604.01181)  

**Abstract**: This study investigates the ability of multimodal Large Language Models (LLMs) to identify and interpret misleading visualizations, and recognize these observations along with their underlying causes and potential intentionality. Our analysis leverages concepts from visualization rhetoric and a newly developed taxonomy of authorial intents as explanatory lenses. We formulated three research questions and addressed them experimentally using a dataset of 2,336 COVID-19-related tweets, half of which contain misleading visualizations, and supplemented it with real-world examples of perceptual, cognitive, and conceptual errors drawn from VisLies, the IEEE VIS community event dedicated to showcasing deceptive and misleading visualizations. To ensure broad coverage of the current LLM landscape, we evaluated 16 state-of-the-art models. Among them, 15 are open-weight models, spanning a wide range of model sizes, architectural families, and reasoning capabilities. The selection comprises small models, namely Nemotron-Nano-V2-VL (12B parameters), Mistral-Small-3.2 (24B), DeepSeek-VL2 (27B), Gemma3 (27B), and GTA1 (32B); medium-sized models, namely Qianfan-VL (70B), Molmo (72B), GLM-4.5V (108B), LLaVA-NeXT (110B), and Pixtral-Large (124B); and large models, namely Qwen3-VL (235B), InternVL3.5 (241B), Step3 (321B), Llama-4-Maverick (400B), and Kimi-K2.5 (1000B). In addition, we employed OpenAI GPT-5.4, a frontier proprietary model. To establish a human perspective on these tasks, we also conducted a user study with visualization experts to assess how people perceive rhetorical techniques and the authorial intentions behind the same misleading visualizations. This allows comparison between model and expert behavior, revealing similarities and differences that provide insights into where LLMs align with human judgment and where they diverge. 

---
# Screening Is Enough 

**Authors**: Ken M. Nakanishi  

**Link**: [PDF](https://arxiv.org/pdf/2604.01178)  

**Abstract**: A core limitation of standard softmax attention is that it does not define a notion of absolute query--key relevance: attention weights are obtained by redistributing a fixed unit mass across all keys according to their relative scores. As a result, relevance is defined only relative to competing keys, and irrelevant keys cannot be explicitly rejected. We introduce Multiscreen, a language-model architecture built around a mechanism we call screening, which enables absolute query--key relevance. Instead of redistributing attention across all keys, screening evaluates each key against an explicit threshold, discarding irrelevant keys and aggregating the remaining keys, thereby removing global competition among keys. Across experiments, Multiscreen achieves comparable validation loss with approximately 40% fewer parameters than a Transformer baseline, enables stable optimization at substantially larger learning rates, maintains strong performance in long-context perplexity, shows little to no degradation in retrieval performance even far beyond the training context length, and reduces inference latency by up to 3.2$\times$ at 100K context length. 

---
# Online Reasoning Calibration: Test-Time Training Enables Generalizable Conformal LLM Reasoning 

**Authors**: Cai Zhou, Zekai Wang, Menghua Wu, Qianyu Julie Zhu, Flora C. Shi, Chenyu Wang, Ashia Wilson, Tommi Jaakkola, Stephen Bates  

**Link**: [PDF](https://arxiv.org/pdf/2604.01170)  

**Abstract**: While test-time scaling has enabled large language models to solve highly difficult tasks, state-of-the-art results come at exorbitant compute costs. These inefficiencies can be attributed to the miscalibration of post-trained language models, and the lack of calibration in popular sampling techniques. Here, we present Online Reasoning Calibration (ORCA), a framework for calibrating the sampling process that draws upon conformal prediction and test-time training. Specifically, we introduce a meta-learning procedure that updates the calibration module for each input. This allows us to provide valid confidence estimates under distributional shift, e.g. in thought patterns that occur across different stages of reasoning, or in prompt distributions between model development and deployment. ORCA not only provides theoretical guarantees on conformal risks, but also empirically shows higher efficiency and generalization across different reasoning tasks. At risk level $\delta=0.1$, ORCA improves Qwen2.5-32B efficiency on in-distribution tasks with savings up to 47.5% with supervised labels and 40.7% with self-consistency labels. Under zero-shot out-of-domain settings, it improves MATH-500 savings from 24.8% of the static calibration baseline to 67.0% while maintaining a low empirical error rate, and the same trend holds across model families and downstream benchmarks. Our code is publicly available at this https URL. 

---
# Revision or Re-Solving? Decomposing Second-Pass Gains in Multi-LLM Pipelines 

**Authors**: Jingjie Ning, Xueqi Li, Chengyu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.01029)  

**Abstract**: Multi-LLM revision pipelines, in which a second model reviews and improves a draft produced by a first, are widely assumed to derive their gains from genuine error correction. We question this assumption with a controlled decomposition experiment that uses four matched conditions to separate second-pass gains into three additive components: re-solving, scaffold, and content. We evaluate this design across two model pairs on three benchmarks spanning knowledge-intensive MCQ and competitive programming. Our results show that the gains of multi-LLM revision are not monolithic, but depend on task structure, draft quality, and the type of draft information. On MCQ tasks, where the answer space is constrained and drafts provide little structural guidance, most gains are consistent with stronger-model re-solving, and directly routing queries to the stronger model can be more effective than revising a weak draft. On code generation tasks, however, two-stage prompting remains useful because even semantically null drafts can provide substantial structural scaffolding, while weak draft content can be harmful. Finally, role-reversed experiments show that strong drafts clearly benefit weak reviewers. Ultimately, our findings demonstrate that the utility of multi-LLM revision is dynamically bottlenecked by task structure and draft quality, necessitating more targeted pipeline designs rather than blanket revision strategies. 

---
# Do Phone-Use Agents Respect Your Privacy? 

**Authors**: Zhengyang Tang, Ke Ji, Xidong Wang, Zihan Ye, Xinyuan Wang, Yiduo Guo, Ziniu Li, Chenxin Li, Jingyuan Hu, Shunian Chen, Tongxu Luo, Jiaxi Bi, Zeyu Qin, Shaobo Wang, Xin Lai, Pengyuan Lyu, Junyi Li, Can Xu, Chengquan Zhang, Han Hu, Ming Yan, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00986)  

**Abstract**: We study whether phone-use agents respect privacy while completing benign mobile tasks. This question has remained hard to answer because privacy-compliant behavior is not operationalized for phone-use agents, and ordinary apps do not reveal exactly what data agents type into which form entries during execution. To make this question measurable, we introduce MyPhoneBench, a verifiable evaluation framework for privacy behavior in mobile agents. We operationalize privacy-respecting phone use as permissioned access, minimal disclosure, and user-controlled memory through a minimal privacy contract, iMy, and pair it with instrumented mock apps plus rule-based auditing that make unnecessary permission requests, deceptive re-disclosure, and unnecessary form filling observable and reproducible. Across five frontier models on 10 mobile apps and 300 tasks, we find that task success, privacy-compliant task completion, and later-session use of saved preferences are distinct capabilities, and no single model dominates all three. Evaluating success and privacy jointly reshuffles the model ordering relative to either metric alone. The most persistent failure mode across models is simple data minimization: agents still fill optional personal entries that the task does not require. These results show that privacy failures arise from over-helpful execution of benign tasks, and that success-only evaluation overestimates the deployment readiness of current phone-use agents. All code, mock apps, and agent trajectories are publicly available at~ this https URL. 

---
# Benchmarking and Mechanistic Analysis of Vision-Language Models for Cross-Depiction Assembly Instruction Alignment 

**Authors**: Zhuchenyang Liu, Yao Zhang, Yu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.00913)  

**Abstract**: 2D assembly diagrams are often abstract and hard to follow, creating a need for intelligent assistants that can monitor progress, detect errors, and provide step-by-step guidance. In mixed reality settings, such systems must recognize completed and ongoing steps from the camera feed and align them with the diagram instructions. Vision Language Models (VLMs) show promise for this task, but face a depiction gap because assembly diagrams and video frames share few visual features. To systematically assess this gap, we construct IKEA-Bench, a benchmark of 1,623 questions across 6 task types on 29 IKEA furniture products, and evaluate 19 VLMs (2B-38B) under three alignment strategies. Our key findings: (1) assembly instruction understanding is recoverable via text, but text simultaneously degrades diagram-to-video alignment; (2) architecture family predicts alignment accuracy more strongly than parameter count; (3) video understanding remains a hard bottleneck unaffected by strategy. A three-level mechanistic analysis further reveals that diagrams and video occupy disjoint ViT subspaces, and that adding text shifts models from visual to text-driven reasoning. These results identify visual encoding as the primary target for improving cross-depiction robustness. Project page: this https URL 

---
# Beyond Symbolic Solving: Multi Chain-of-Thought Voting for Geometric Reasoning in Large Language Models 

**Authors**: Md. Abu Bakor Siddique, Shahrin Hossain, Sadman Ahmed Siam, Syed Rifat Raiyan, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2604.00890)  

**Abstract**: Geometric Problem Solving (GPS) remains at the heart of enhancing mathematical reasoning in large language models because it requires the combination of diagrammatic understanding, symbolic manipulation and logical inference. In existing literature, researchers have chiefly focused on synchronising the diagram descriptions with text literals and solving the problem. In this vein, they have either taken a neural, symbolic or neuro-symbolic approach. But this solves only the first two of the requirements, namely diagrammatic understanding and symbolic manipulation, while leaving logical inference underdeveloped. The logical inference is often limited to one chain-of-thought (CoT). To address this weakness in hitherto existing models, this paper proposes MARS-GPS, that generates multiple parallel reasoning rollouts augmented with Python code execution for numerical verification, ranks them using token-level entropy as a confidence signal, and aggregates answers through a multi-stage voting and self-verification pipeline. Empirical results show that MARS-GPS with 8 parallel rollouts achieves 88.8% on Geometry3K, a nearly +11% improvement over the prior state-of-the-art, with accuracy scaling consistently as the number of rollouts increases from 1 to 16 (+6.0% on ablation subset). We provide our code and data in an anonymous repository: this https URL. 

---
# PixelPrune: Pixel-Level Adaptive Visual Token Reduction via Predictive Coding 

**Authors**: Nan Wang, Zhiwei Jin, Chen Chen, Haonan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00886)  

**Abstract**: Document understanding and GUI interaction are among the highest-value applications of Vision-Language Models (VLMs), yet they impose exceptionally heavy computational burden: fine-grained text and small UI elements demand high-resolution inputs that produce tens of thousands of visual tokens. We observe that this cost is largely wasteful -- across document and GUI benchmarks, only 22--71\% of image patches are pixel-unique, the rest being exact duplicates of another patch in the same image. We propose \textbf{PixelPrune}, which exploits this pixel-level redundancy through predictive-coding-based compression, pruning redundant patches \emph{before} the Vision Transformer (ViT) encoder. Because it operates in pixel space prior to any neural computation, PixelPrune accelerates both the ViT encoder and the downstream LLM, covering the full inference pipeline. The method is training-free, requires no learnable parameters, and supports pixel-lossless compression ($\tau{=}0$) as well as controlled lossy compression ($\tau{>}0$). Experiments across three model scales and document and GUI benchmarks show that PixelPrune maintains competitive task accuracy while delivering up to 4.2$\times$ inference speedup and 1.9$\times$ training acceleration. Code is available at this https URL. 

---
# LinguDistill: Recovering Linguistic Ability in Vision- Language Models via Selective Cross-Modal Distillation 

**Authors**: Patrick Amadeus Irawan, Erland Hilman Fuadi, Shanu Kumar, Alham Fikri Aji, Yova Kementchedjhieva  

**Link**: [PDF](https://arxiv.org/pdf/2604.00829)  

**Abstract**: Adapting pretrained language models (LMs) into vision-language models (VLMs) can degrade their native linguistic capability due to representation shift and cross-modal interference introduced during multimodal adaptation. Such loss is difficult to recover, even with targeted task-specific fine-tuning using standard objectives. Prior recovery approaches typically introduce additional modules that act as intermediate alignment layers to maintain or isolate modality-specific subspaces, which increases architectural complexity, adds parameters at inference time, and limits flexibility across models and settings. We propose LinguDistill, an adapter-free distillation method that restores linguistic capability by utilizing the original frozen LM as a teacher. We overcome the key challenge of enabling vision-conditioned teacher supervision by introducing layer-wise KV-cache sharing, which exposes the teacher to the student's multimodal representations without modifying the architecture of either model. We then selectively distill the teacher's strong linguistic signal on language-intensive data to recover language capability, while preserving the student's visual grounding on multimodal tasks. As a result, LinguDistill recovers $\sim$10% of the performance lost on language and knowledge benchmarks, while maintaining comparable performance on vision-heavy tasks. Our findings demonstrate that linguistic capability can be recovered without additional modules, providing an efficient and practical solution to modality-specific degradation in multimodal models. 

---
# Routing-Free Mixture-of-Experts 

**Authors**: Yilun Liu, Jinru Han, Sikuan Yan, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.00801)  

**Abstract**: Standard Mixture-of-Experts (MoE) models rely on centralized routing mechanisms that introduce rigid inductive biases. We propose Routing-Free MoE which eliminates any hard-coded centralized designs including external routers, Softmax, Top-K and load balancing, instead encapsulating all activation functionalities within individual experts and directly optimized through continuous gradient flow, enabling each expert to determine its activation entirely on its own. We introduce a unified adaptive load-balancing framework to simultaneously optimize both expert-balancing and token-balancing objectives through a configurable interpolation, allowing flexible and customizable resource allocation. Extensive experiments show that Routing-Free MoE can consistently outperform baselines with better scalability and robustness. We analyze its behavior in detail and offer insights that may facilitate future MoE design ad optimization. 

---
# Multimodal Language Models Cannot Spot Spatial Inconsistencies 

**Authors**: Om Khangaonkar, Hadi J. Rad, Hamed Pirsiavash  

**Link**: [PDF](https://arxiv.org/pdf/2604.00799)  

**Abstract**: Spatial consistency is a fundamental property of the visual world and a key requirement for models that aim to understand physical reality. Despite recent advances, multimodal large language models (MLLMs) often struggle to reason about 3D geometry across multiple views. Rather than asking models to describe scene attributes, we introduce a more challenging task: given two views of the same scene, identify the object that violates 3D motion consistency. We propose a simple and scalable method for generating realistic, spatially inconsistent image pairs from multi-view scenes, enabling systematic evaluation of this capability. Our results show that state-of-the-art MLLMs significantly underperform human observers and exhibit substantial variability across different scene attributes, revealing a fragile and incomplete understanding of 3D structure. We hope our findings underscore the need for approaches that develop a more deeply grounded understanding of the physical world. 

---
# Learning to Hint for Reinforcement Learning 

**Authors**: Yu Xia, Canwen Xu, Zhewei Yao, Julian McAuley, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2604.00698)  

**Abstract**: Group Relative Policy Optimization (GRPO) is widely used for reinforcement learning with verifiable rewards, but it often suffers from advantage collapse: when all rollouts in a group receive the same reward, the group yields zero relative advantage and thus no learning signal. For example, if a question is too hard for the reasoner, all sampled rollouts can be incorrect and receive zero reward. Recent work addresses this issue by adding hints or auxiliary scaffolds to such hard questions so that the reasoner produces mixed outcomes and recovers a non-zero update. However, existing hints are usually fixed rather than adapted to the current reasoner, and a hint that creates learning signal under the hinted input does not necessarily improve the no-hint policy used at test time. To this end, we propose Hint Learning for Reinforcement Learning (HiLL), a framework that jointly trains a hinter policy and a reasoner policy during RL. For each hard question, the hinter generates hints online conditioned on the current reasoner's incorrect rollout, allowing hint generation to adapt to the reasoner's evolving errors. We further introduce hint reliance, which measures how strongly correct hinted trajectories depend on the hint. We derive a transferability result showing that lower hint reliance implies stronger transfer from hinted success to no-hint success, and we use this result to define a transfer-weighted reward for training the hinter. Therefore, HiLL favors hints that not only recover informative GRPO groups, but also produce signals that are more likely to improve the original no-hint policy. Experiments across multiple benchmarks show that HiLL consistently outperforms GRPO and prior hint-based baselines, demonstrating the value of adaptive and transfer-aware hint learning for RL. The code is available at this https URL. 

---
# A Survey of On-Policy Distillation for Large Language Models 

**Authors**: Mingyang Song, Mao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.00626)  

**Abstract**: Knowledge distillation has become a primary mechanism for transferring reasoning and domain expertise from frontier Large Language Models (LLMs) to smaller, deployable students. However, the dominant paradigm remains \textit{off-policy}: students train on static teacher-generated data and never encounter their own errors during learning. This train--test mismatch, an instance of \textit{exposure bias}, causes prediction errors to compound autoregressively at inference time. On-Policy Distillation (OPD) addresses this by letting the student generate its own trajectories and receive teacher feedback on these self-generated outputs, grounding distillation in the theory of interactive imitation learning. Despite rapid growth spanning divergence minimization, reward-guided learning, and self-play, the OPD literature remains fragmented with no unified treatment. This survey provides the first comprehensive overview of OPD for LLMs. We introduce a unified $f$-divergence framework over on-policy samples and organize the landscape along three orthogonal dimensions: \emph{feedback signal} (logit-based, outcome-based, or self-play), \emph{teacher access} (white-box, black-box, or teacher-free), and \emph{loss granularity} (token-level, sequence-level, or hybrid). We systematically analyze representative methods, examine industrial deployments, and identify open problems including distillation scaling laws, uncertainty-aware feedback, and agent-level distillation. 

---
# Ontology-Constrained Neural Reasoning in Enterprise Agentic Systems: A Neurosymbolic Architecture for Domain-Grounded AI Agents 

**Authors**: Thanh Luong Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2604.00555)  

**Abstract**: Enterprise adoption of Large Language Models (LLMs) is constrained by hallucination, domain drift, and the inability to enforce regulatory compliance at the reasoning level. We present a neurosymbolic architecture implemented within the Foundation AgenticOS (FAOS) platform that addresses these limitations through ontology-constrained neural reasoning. Our approach introduces a three-layer ontological framework--Role, Domain, and Interaction ontologies--that provides formal semantic grounding for LLM-based enterprise agents. We formalize the concept of asymmetric neurosymbolic coupling, wherein symbolic ontological knowledge constrains agent inputs (context assembly, tool discovery, governance thresholds) while proposing mechanisms for extending this coupling to constrain agent outputs (response validation, reasoning verification, compliance checking). We evaluate the architecture through a controlled experiment (600 runs across five industries: FinTech, Insurance, Healthcare, Vietnamese Banking, and Vietnamese Insurance), finding that ontology-coupled agents significantly outperform ungrounded agents on Metric Accuracy (p < .001, W = .460), Regulatory Compliance (p = .003, W = .318), and Role Consistency (p < .001, W = .614), with improvements greatest where LLM parametric knowledge is weakest--particularly in Vietnam-localized domains. Our contributions include: (1) a formal three-layer enterprise ontology model, (2) a taxonomy of neurosymbolic coupling patterns, (3) ontology-constrained tool discovery via SQL-pushdown scoring, (4) a proposed framework for output-side ontological validation, (5) empirical evidence for the inverse parametric knowledge effect that ontological grounding value is inversely proportional to LLM training data coverage of the domain, and (6) a production system serving 21 industry verticals with 650+ agents. 

---
# MF-QAT: Multi-Format Quantization-Aware Training for Elastic Inference 

**Authors**: Zifei Xu, Sayeh Sharify, Hesham Mostafa  

**Link**: [PDF](https://arxiv.org/pdf/2604.00529)  

**Abstract**: Quantization-aware training (QAT) is typically performed for a single target numeric format, while practical deployments often need to choose numerical precision at inference time based on hardware support or runtime constraints. We study multi-format QAT, where a single model is trained to be robust across multiple quantization formats. We find that multi-format QAT can match single-format QAT at each target precision, yielding one model that performs well overall across different formats, even formats that were not seen during training. To enable practical deployment, we propose the Slice-and-Scale conversion procedure for both MXINT and MXFP that converts a high-precision representation into lower-precision formats without re-training. Building on this, we introduce a pipeline that (i) trains a model with multi-format QAT, (ii) stores a single anchor format checkpoint (MXINT8/MXFP8), and (iii) allows on-the-fly conversion to lower MXINT or MXFP formats at runtime with negligible-or no-additional accuracy degradation. Together, these components provide a practical path to elastic precision scaling and allow selecting the runtime format at inference time across diverse deployment targets. 

---
# Logarithmic Scores, Power-Law Discoveries: Disentangling Measurement from Coverage in Agent-Based Evaluation 

**Authors**: HyunJoon Jung, William Na  

**Link**: [PDF](https://arxiv.org/pdf/2604.00477)  

**Abstract**: LLM-based agent judges are an emerging approach to evaluating conversational AI, yet a fundamental uncertainty remains: can we trust their assessments, and if so, how many are needed? Through 960 sessions with two model pairs across 15 tasks, we show that persona-based agent judges produce evaluations indistinguishable from human raters in a Turing-style validation. We then identify a score-coverage dissociation: quality scores improve logarithmically with panel size, while unique issue discoveries follow a sublinear power law-both exhibit diminishing returns, but scores saturate roughly twice as fast as discoveries. We hypothesize this reflects a power law distribution of the finding space: critical issues are discovered first by small panels, while corner cases require progressively larger panels, analogous to species accumulation curves in ecology. The mechanism traces to ensemble diversity-Big Five personality conditioning makes agents probe different quality dimensions, with expert judges acting as adversarial probes that push discovery into the tail of the finding distribution. A controlled ablation confirms that structured persona conditioning, not simple prompting, is required to produce these scaling properties. 

---
# Not My Truce: Personality Differences in AI-Mediated Workplace Negotiation 

**Authors**: Veda Duddu, Jash Rajesh Parekh, Andy Mao, Hanyi Min, Ziang Xiao, Vedant Das Swain, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2604.00464)  

**Abstract**: AI-driven conversational coaching is increasingly used to support workplace negotiation, yet prior work assumes uniform effectiveness across users. We challenge this assumption by examining how individual differences, particularly personality traits, moderate coaching outcomes. We conducted a between-subjects experiment (N=267) comparing theory-driven AI (Trucey), general-purpose AI (Control-AI), and a traditional negotiation handbook (Control-NoAI). Participants were clustered into three profiles -- resilient, overcontrolled, and undercontrolled -- based on the Big-Five personality traits and ARC typology. Resilient workers achieved broad psychological gains primarily from the handbook, overcontrolled workers showed outcome-specific improvements with theory-driven AI, and undercontrolled workers exhibited minimal effects despite engaging with the frameworks. These patterns suggest personality as a predictor of readiness beyond stage-based tailoring: vulnerable users benefit from targeted rather than comprehensive interventions. The study advances understanding of personality-determined intervention prerequisites and highlights design implications for adaptive AI coaching systems that align support intensity with individual readiness, rather than assuming universal effectiveness. 

---
# First Logit Boosting: Visual Grounding Method to Mitigate Object Hallucination in Large Vision-Language Models 

**Authors**: Jiwoo Ha, Jongwoo Baek, Jinhyun So  

**Link**: [PDF](https://arxiv.org/pdf/2604.00455)  

**Abstract**: Recent Large Vision-Language Models (LVLMs) have demonstrated remarkable performance across various multimodal tasks that require understanding both visual and linguistic inputs. However, object hallucination -- the generation of nonexistent objects in answers -- remains a persistent challenge. Although several approaches such as retraining and external grounding methods have been proposed to mitigate this issue, they still suffer from high data costs or structural complexity. Training-free methods such as Contrastive Decoding (CD) are more cost-effective, avoiding additional training or external models, but still suffer from long-term decay, where visual grounding weakens and language priors dominate as the generation progresses. In this paper, we propose First Logit Boosting (FLB), a simple yet effective training-free technique designed to alleviate long-term decay in LVLMs. FLB stores the logit of the first generated token and adds it to subsequent token predictions, effectively mitigating long-term decay of visual information. We observe that FLB (1) sustains the visual information embedded in the first token throughout generation, and (2) suppresses hallucinated words through the stabilizing effect of the ``The'' token. Experimental results show that FLB significantly reduces object hallucination across various tasks, benchmarks, and backbone models. Notably, it causes negligible inference overhead, making it highly applicable to real-time multimodal systems. Code is available at this https URL 

---
# Towards Reliable Truth-Aligned Uncertainty Estimation in Large Language Models 

**Authors**: Ponhvoan Srey, Quang Minh Nguyen, Xiaobao Wu, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00445)  

**Abstract**: Uncertainty estimation (UE) aims to detect hallucinated outputs of large language models (LLMs) to improve their reliability. However, UE metrics often exhibit unstable performance across configurations, which significantly limits their applicability. In this work, we formalise this phenomenon as proxy failure, since most UE metrics originate from model behaviour, rather than being explicitly grounded in the factual correctness of LLM outputs. With this, we show that UE metrics become non-discriminative precisely in low-information regimes. To alleviate this, we propose Truth AnChoring (TAC), a post-hoc calibration method to remedy UE metrics, by mapping the raw scores to truth-aligned scores. Even with noisy and few-shot supervision, our TAC can support the learning of well-calibrated uncertainty estimates, and presents a practical calibration protocol. Our findings highlight the limitations of treating heuristic UE metrics as direct indicators of truth uncertainty, and position our TAC as a necessary step toward more reliable uncertainty estimation for LLMs. The code repository is available at this https URL. 

---
# Execution-Verified Reinforcement Learning for Optimization Modeling 

**Authors**: Runda Guan, Xiangqing Shen, Jiajun Zhang, Yifan Zhang, Jian Cheng, Rui Xia  

**Link**: [PDF](https://arxiv.org/pdf/2604.00442)  

**Abstract**: Automating optimization modeling with LLMs is a promising path toward scalable decision intelligence, but existing approaches either rely on agentic pipelines built on closed-source LLMs with high inference latency, or fine-tune smaller LLMs using costly process supervision that often overfits to a single solver API. Inspired by reinforcement learning with verifiable rewards, we propose Execution-Verified Optimization Modeling (EVOM), an execution-verified learning framework that treats a mathematical programming solver as a deterministic, interactive verifier. Given a natural-language problem and a target solver, EVOM generates solver-specific code, executes it in a sandboxed harness, and converts execution outcomes into scalar rewards, optimized with GRPO and DAPO in a closed-loop generate-execute-feedback-update process. This outcome-only formulation removes the need for process-level supervision, and enables cross-solver generalization by switching the verification environment rather than reconstructing solver-specific datasets. Experiments on NL4OPT, MAMO, IndustryOR, and OptiBench across Gurobi, OR-Tools, and COPT show that EVOM matches or outperforms process-supervised SFT, supports zero-shot solver transfer, and achieves effective low-cost solver adaptation by continuing training under the target solver backend. 

---
# Signals: Trajectory Sampling and Triage for Agentic Interactions 

**Authors**: Shuguang Chen, Adil Hafeez, Salman Paracha  

**Link**: [PDF](https://arxiv.org/pdf/2604.00356)  

**Abstract**: Agentic applications based on large language models increasingly rely on multi-step interaction loops involving planning, action execution, and environment feedback. While such systems are now deployed at scale, improving them post-deployment remains challenging. Agent trajectories are voluminous and non-deterministic, and reviewing each one, whether through human review or auxiliary LLMs, is slow and cost-prohibitive. We propose a lightweight, signal-based framework for triaging agentic interaction trajectories. Our approach computes cheap, broadly applicable signals from live interactions and attaches them as structured attributes for trajectory triage, identifying interactions likely to be informative without affecting online agent behavior. We organize signals into a coarse-grained taxonomy spanning interaction (misalignment, stagnation, disengagement, satisfaction), execution (failure, loop), and environment (exhaustion), designed for computation without model calls. In a controlled annotation study on $\tau$-bench, a widely used benchmark for tool-augmented agent evaluation, we show that signal-based sampling achieves an 82\% informativeness rate compared to 74\% for heuristic filtering and 54\% for random sampling, with a 1.52x efficiency gain per informative trajectory. The advantage is robust across reward strata and task domains, confirming that signals provide genuine per-trajectory informativeness gains rather than merely oversampling obvious failures. These results show that lightweight signals can serve as practical sampling infrastructure for agentic systems, and suggest a path toward preference data construction and post-deployment optimization. 

---
# FGR-ColBERT: Identifying Fine-Grained Relevance Tokens During Retrieval 

**Authors**: Antonín Jarolím, Martin Fajčík  

**Link**: [PDF](https://arxiv.org/pdf/2604.00242)  

**Abstract**: Document retrieval identifies relevant documents but does not provide fine-grained evidence cues, such as specific relevant spans. A possible solution is to apply an LLM after retrieval; however, this introduces significant computational overhead and limits practical deployment. We propose FGR-ColBERT, a modification of ColBERT retrieval model that integrates fine-grained relevance signals distilled from an LLM directly into the retrieval function. Experiments on MS MARCO show that FGR-ColBERT (110M) achieves a token-level F1 of 64.5, exceeding the 62.8 of Gemma 2 (27B), despite being approximately 245 times smaller. At the same time, it preserves retrieval effectiveness (99% relative Recall@50) and remains efficient, incurring only a ~1.12x latency overhead compared to the original ColBERT. 

---
# ParetoBandit: Budget-Paced Adaptive Routing for Non-Stationary LLM Serving 

**Authors**: Annette Taberner-Miller  

**Link**: [PDF](https://arxiv.org/pdf/2604.00136)  

**Abstract**: Production LLM serving often relies on multi-model portfolios spanning a ~530x cost range, where routing decisions trade off quality against cost. This trade-off is non-stationary: providers revise pricing, model quality can regress silently, and new models must be integrated without downtime. We present ParetoBandit, an open-source adaptive router built on cost-aware contextual bandits that is the first to simultaneously enforce dollar-denominated budgets, adapt online to such shifts, and onboard new models at runtime.
ParetoBandit closes these gaps through three mechanisms. An online primal-dual budget pacer enforces a per-request cost ceiling over an open-ended stream, replacing offline penalty tuning with closed-loop control. Geometric forgetting on sufficient statistics enables rapid adaptation to price and quality shifts while bootstrapping from offline priors. A hot-swap registry lets operators add or remove models at runtime, with a brief forced-exploration phase for each newcomer, after which UCB selection discovers its quality-cost niche from live traffic alone.
We evaluate ParetoBandit across four deployment scenarios on 1,824 prompts routed through a three-model portfolio. Across seven budget ceilings, mean per-request cost never exceeds the target by more than 0.4%. When conditions shift, the system adapts: an order-of-magnitude price cut on the costliest model yields up to +0.071 quality lift, and a silent quality regression is detected and rerouted within budget. A cold-started model reaches meaningful adoption within ~142 steps without breaching the cost ceiling. The router discriminates rather than blindly adopting: expensive models are budget-gated and low-quality models rejected after bounded exploration. End-to-end routing latency is 9.8ms on CPU -- less than 0.4% of typical inference time -- with the routing decision itself taking just 22.5us. 

---
# Hierarchical Pre-Training of Vision Encoders with Large Language Models 

**Authors**: Eugene Lee, Ting-Yu Chang, Jui-Huang Tsai, Jiajie Diao, Chen-Yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.00086)  

**Abstract**: The field of computer vision has experienced significant advancements through scalable vision encoders and multimodal pre-training frameworks. However, existing approaches often treat vision encoders and large language models (LLMs) as independent modules, limiting the integration of hierarchical visual features. In this work, we propose HIVE (Hierarchical Pre-Training of Vision Encoders), a novel framework that enhances vision-language alignment by introducing hierarchical cross-attention between the vision encoder and LLM. Unlike conventional methods that flatten image embeddings, HIVE enables structured feature fusion across multiple layers, improving gradient flow and representation learning. To optimize this interaction, we introduce a three-stage training strategy that progressively aligns the vision encoder with the LLM, ensuring stable optimization and effective multimodal fusion. Empirical evaluations demonstrate that HIVE achieves superior performance not only in image classification but also on various vision-language tasks, outperforming self-attention-based methods in benchmarks such as MME, GQA, OK-VQA, and ScienceQA. Our results highlight the benefits of hierarchical feature integration, paving the way for more efficient and expressive vision-language models. 

---
# One Panel Does Not Fit All: Case-Adaptive Multi-Agent Deliberation for Clinical Prediction 

**Authors**: Yuxing Lu, Yushuhong Lin, Jason Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.00085)  

**Abstract**: Large language models applied to clinical prediction exhibit case-level heterogeneity: simple cases yield consistent outputs, while complex cases produce divergent predictions under minor prompt changes. Existing single-agent strategies sample from one role-conditioned distribution, and multi-agent frameworks use fixed roles with flat majority voting, discarding the diagnostic signal in disagreement. We propose CAMP (Case-Adaptive Multi-agent Panel), where an attending-physician agent dynamically assembles a specialist panel tailored to each case's diagnostic uncertainty. Each specialist evaluates candidates via three-valued voting (KEEP/REFUSE/NEUTRAL), enabling principled abstention outside one's expertise. A hybrid router directs each diagnosis through strong consensus, fallback to the attending physician's judgment, or evidence-based arbitration that weighs argument quality over vote counts. On diagnostic prediction and brief hospital course generation from MIMIC-IV across four LLM backbones, CAMP consistently outperforms strong baselines while consuming fewer tokens than most competing multi-agent methods, with voting records and arbitration traces offering transparent decision audits. 

---
# Terminal Agents Suffice for Enterprise Automation 

**Authors**: Patrice Bechard, Orlando Marquez Ayala, Emily Chen, Jordan Skelton, Sagar Davasam, Srinivas Sunkara, Vikas Yadav, Sai Rajeswar  

**Link**: [PDF](https://arxiv.org/pdf/2604.00073)  

**Abstract**: There has been growing interest in building agents that can interact with digital platforms to execute meaningful enterprise tasks autonomously. Among the approaches explored are tool-augmented agents built on abstractions such as Model Context Protocol (MCP) and web agents that operate through graphical interfaces. Yet, it remains unclear whether such complex agentic systems are necessary given their cost and operational overhead. We argue that a coding agent equipped only with a terminal and a filesystem can solve many enterprise tasks more effectively by interacting directly with platform APIs. We evaluate this hypothesis across diverse real-world systems and show that these low-level terminal agents match or outperform more complex agent architectures. Our findings suggest that simple programmatic interfaces, combined with strong foundation models, are sufficient for practical enterprise automation. 

---
# How Emotion Shapes the Behavior of LLMs and Agents: A Mechanistic Study 

**Authors**: Moran Sun, Tianlin Li, Yuwei Zheng, Zhenhong Zhou, Aishan Liu, Xianglong Liu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00005)  

**Abstract**: Emotion plays an important role in human cognition and performance. Motivated by this, we investigate whether analogous emotional signals can shape the behavior of large language models (LLMs) and agents. Existing emotion-aware studies mainly treat emotion as a surface-level style factor or a perception target, overlooking its mechanistic role in task processing. To address this limitation, we propose E-STEER, an interpretable emotion steering framework that enables direct representation-level intervention in LLMs and agents. It embeds emotion as a structured, controllable variable in hidden states, and with it, we examine the impact of emotion on objective reasoning, subjective generation, safety, and multi-step agent behaviors. The results reveal non-monotonic emotion-behavior relations consistent with established psychological theories, and show that specific emotions not only enhance LLM capability but also improve safety, and systematically shape multi-step agent behaviors. 

---
# Two-Stage Optimizer-Aware Online Data Selection for Large Language Models 

**Authors**: Fangxin Wang, Peyman Baghershahi, Langzhou He, Henry Peng Zou, Sourav Medya, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.00001)  

**Abstract**: Gradient-based data selection offers a principled framework for estimating sample utility in large language model (LLM) fine-tuning, but existing methods are mostly designed for offline settings. They are therefore less suited to online fine-tuning, where data arrives sequentially, sample utility is step-dependent, and the effective update geometry is shaped by adaptive optimizers. We propose an optimizer-aware framework for gradient-based online data selection and reweighting in LLM fine-tuning. Our key idea is to view online selection not as static sample ranking, but as shaping the next target-oriented update under the optimizer state. We formulate this as an optimizer-aware update-matching problem, establish its connection to second-order target utility, and show why subset-level construction must account for interactions and redundancy among selected samples. Based on this view, we develop a two-stage Filter-then-Weight algorithm that first filters geometrically useful candidates and then optimizes their coefficients. To make the framework practical for LLMs, we introduce a factorized outer-product gradient representation and optimized matrix computations for long-context data. Experiments show that our method consistently improves convergence and downstream performance over existing online data selection baselines under the same data budget. 

---
