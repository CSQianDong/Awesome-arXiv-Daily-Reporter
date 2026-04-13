# From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models 

**Authors**: Chenchen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09459)  

**Abstract**: Reinforcement learning (RL) for large language models (LLMs) increasingly relies on sparse, outcome-level rewards -- yet determining which actions within a long trajectory caused the outcome remains difficult. This credit assignment (CA) problem manifests in two regimes: reasoning RL, where credit must be distributed across tokens and steps within a single chain-of-thought generation (500--30K+ tokens); and agentic RL, where multi-turn environment interaction introduces stochastic transitions, partial observability, and horizons of 100+ turns (100K--1M tokens), making episode-level credit increasingly uninformative.
We survey 47 CA methods (41 core, 6 adjacent enablers) published between 2024 and early 2026, organizing them in a two-dimensional taxonomy by assignment granularity (token, segment, step, turn, multi-agent) and methodology (Monte Carlo, temporal difference, model-based, game-theoretic, information-theoretic). Beyond the survey itself, we contribute three reusable resources: (1) a structured, machine-readable paper inventory with taxonomy labels, baseline families, and evidence levels; (2) a reporting checklist for future CA papers, validated against the reviewed literature to identify systematic methodological gaps; and (3) a benchmark protocol specification with task families, metadata requirements, and controlled bifurcation tasks, accompanied by a method selection decision tree.
Our synthesis suggests that the shift from reasoning to agentic RL complicates and reshapes the credit assignment landscape: reasoning CA is maturing around process reward models and critic-free group comparison, while agentic CA is driving genuinely new approaches -- hindsight counterfactual analysis, privileged asymmetric critics, and turn-level MDP reformulations -- that have no direct precedent in reasoning RL. 

---
# BERT-as-a-Judge: A Robust Alternative to Lexical Methods for Efficient Reference-Based LLM Evaluation 

**Authors**: Hippolyte Gisserot-Boukhlef, Nicolas Boizard, Emmanuel Malherbe, Céline Hudelot, Pierre Colombo  

**Link**: [PDF](https://arxiv.org/pdf/2604.09497)  

**Abstract**: Accurate evaluation is central to the large language model (LLM) ecosystem, guiding model selection and downstream adoption across diverse use cases. In practice, however, evaluating generative outputs typically relies on rigid lexical methods to extract and assess answers, which can conflate a model's true problem-solving ability with its compliance with predefined formatting guidelines. While recent LLM-as-a-Judge approaches mitigate this issue by assessing semantic correctness rather than strict structural conformity, they also introduce substantial computational overhead, making evaluation costly. In this work, we first systematically investigate the limitations of lexical evaluation through a large-scale empirical study spanning 36 models and 15 downstream tasks, demonstrating that such methods correlate poorly with human judgments. To address this limitation, we introduce BERT-as-a-Judge, an encoder-driven approach for assessing answer correctness in reference-based generative settings, robust to variations in output phrasing, and requiring only lightweight training on synthetically annotated question-candidate-reference triplets. We show that it consistently outperforms the lexical baseline while matching the performance of much larger LLM judges, providing a compelling tradeoff between the two and enabling reliable, scalable evaluation. Finally, through extensive experimentation, we provide detailed insights into BERT-as-a-Judge's performance to offer practical guidance for practitioners, and release all project artifacts to foster downstream adoption. 

---
# RecaLLM: Addressing the Lost-in-Thought Phenomenon with Explicit In-Context Retrieval 

**Authors**: Kyle Whitecross, Negin Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2604.09494)  

**Abstract**: We propose RecaLLM, a set of reasoning language models post-trained to make effective use of long-context information. In-context retrieval, which identifies relevant evidence from context, and reasoning are deeply intertwined: retrieval supports reasoning, while reasoning often determines what must be retrieved. However, their interaction remains largely underexplored. In preliminary experiments on several open-source LLMs, we observe that in-context retrieval performance substantially degrades even after a short reasoning span, revealing a key bottleneck for test-time scaling that we refer to as lost-in-thought: reasoning steps that improve performance also make subsequent in-context retrieval more challenging. To address this limitation, RecaLLM interleaves reasoning with explicit in-context retrieval, alternating between reasoning and retrieving context information needed to solve intermediate subproblems. We introduce a negligible-overhead constrained decoding mechanism that enables verbatim copying of evidence spans, improving the grounding of subsequent generation. Trained on diverse lexical and semantic retrieval tasks, RecaLLM achieves strong performance on two long-context benchmarks, RULER and HELMET, significantly outperforming baselines. Notably, we observe consistent gains at context windows of up to 128K tokens using training samples of at most 10K tokens, far shorter than those used by existing long-context approaches, highlighting a promising path toward improving long-context performance without expensive long-context training data. 

---
# You Can't Fight in Here! This is BBS! 

**Authors**: Richard Futrell, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2604.09501)  

**Abstract**: Norm, the formal theoretical linguist, and Claudette, the computational language scientist, have a lovely time discussing whether modern language models can inform important questions in the language sciences. Just as they are about to part ways until they meet again, 25 of their closest friends show up -- from linguistics, neuroscience, cognitive science, psychology, philosophy, and computer science. We use this discussion to highlight what we see as some common underlying issues: the String Statistics Strawman (the mistaken idea that LMs can't be linguistically competent or interesting because they, like their Markov model predecessors, are statistical models that learn from strings) and the As Good As it Gets Assumption (the idea that LM research as it stands in 2026 is the limit of what it can tell us about linguistics). We clarify the role of LM-based work for scientific insights into human language and advocate for a more expansive research program for the language sciences in the AI age, one that takes on the commentators' concerns in order to produce a better and more robust science of both human language and of LMs. 

---
# Across the Levels of Analysis: Explaining Predictive Processing in Humans Requires More Than Machine-Estimated Probabilities 

**Authors**: Sathvik Nair, Colin Phillips  

**Link**: [PDF](https://arxiv.org/pdf/2604.09466)  

**Abstract**: Under the lens of Marr's levels of analysis, we critique and extend two claims about language models (LMs) and language processing: first, that predicting upcoming linguistic information based on context is central to language processing, and second, that many advances in psycholinguistics would be impossible without large language models (LLMs). We further outline future directions that combine the strengths of LLMs with psycholinguistic models. 

---
# Large Language Models Generate Harmful Content Using a Distinct, Unified Mechanism 

**Authors**: Hadas Orgad, Boyi Wei, Kaden Zheng, Martin Wattenberg, Peter Henderson, Seraphina Goldfarb-Tarrant, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2604.09544)  

**Abstract**: Large language models (LLMs) undergo alignment training to avoid harmful behaviors, yet the resulting safeguards remain brittle: jailbreaks routinely bypass them, and fine-tuning on narrow domains can induce ``emergent misalignment'' that generalizes broadly. Whether this brittleness reflects a fundamental lack of coherent internal organization for harmfulness remains unclear. Here we use targeted weight pruning as a causal intervention to probe the internal organization of harmfulness in LLMs. We find that harmful content generation depends on a compact set of weights that are general across harm types and distinct from benign capabilities. Aligned models exhibit a greater compression of harm generation weights than unaligned counterparts, indicating that alignment reshapes harmful representations internally--despite the brittleness of safety guardrails at the surface level. This compression explains emergent misalignment: if weights of harmful capabilities are compressed, fine-tuning that engages these weights in one domain can trigger broad misalignment. Consistent with this, pruning harm generation weights in a narrow domain substantially reduces emergent misalignment. Notably, LLMs harmful generation capability is dissociated from how they recognize and explain such content. Together, these results reveal a coherent internal structure for harmfulness in LLMs that may serve as a foundation for more principled approaches to safety. 

---
# Agentic Jackal: Live Execution and Semantic Value Grounding for Text-to-JQL 

**Authors**: Vishnu Murali, Anmol Gulati, Elias Lumer, Kevin Frank, Sindy Campagna, Vamse Kumar Subbiah  

**Link**: [PDF](https://arxiv.org/pdf/2604.09470)  

**Abstract**: Translating natural language into Jira Query Language (JQL) requires resolving ambiguous field references, instance-specific categorical values, and complex Boolean predicates. Single-pass LLMs cannot discover which categorical values (e.g., component names or fix versions) actually exist in a given Jira instance, nor can they verify generated queries against a live data source, limiting accuracy on paraphrased or ambiguous requests. No open, execution-based benchmark exists for mapping natural language to JQL. We introduce Jackal, the first large-scale, execution-based text-to-JQL benchmark comprising 100,000 validated NL-JQL pairs on a live Jira instance with over 200,000 issues. To establish baselines on Jackal, we propose Agentic Jackal, a tool-augmented agent that equips LLMs with live query execution via the Jira MCP server and JiraAnchor, a semantic retrieval tool that resolves natural-language mentions of categorical values through embedding-based similarity search. Among 9 frontier LLMs evaluated, single-pass models average only 43.4% execution accuracy on short natural-language queries, highlighting that text-to-JQL remains an open challenge. The agentic approach improves 7 of 9 models, with a 9.0% relative gain on the most linguistically challenging variant; in a controlled ablation isolating JiraAnchor, categorical-value accuracy rises from 48.7% to 71.7%, with component-field accuracy jumping from 16.9% to 66.2%. Our analysis identifies inherent semantic ambiguities, such as issue-type disambiguation and text-field selection, as the dominant failure modes rather than value-resolution errors, pointing to concrete directions for future work. We publicly release the benchmark, all agent transcripts, and evaluation code to support reproducibility. 

---
# Automated Instruction Revision (AIR): A Structured Comparison of Task Adaptation Strategies for LLM 

**Authors**: Solomiia Bilyk, Volodymyr Getmanskyi, Taras Firman  

**Link**: [PDF](https://arxiv.org/pdf/2604.09418)  

**Abstract**: This paper studies Automated Instruction Revision (AIR), a rule-induction-based method for adapting large language models (LLMs) to downstream tasks using limited task-specific examples. We position AIR within the broader landscape of adaptation strategies, including prompt optimization, retrieval-based methods, and fine-tuning. We then compare these approaches across a diverse benchmark suite designed to stress different task requirements, such as knowledge injection, structured extraction, label remapping, and logical reasoning. The paper argues that adaptation performance is strongly task-dependent: no single method dominates across all settings. Across five benchmarks, AIR was strongest or near-best on label-remapping classification, while KNN retrieval performed best on closed-book QA, and fine-tuning dominated structured extraction and event-order reasoning. AIR is most promising when task behavior can be captured by compact, interpretable instruction rules, while retrieval and fine-tuning remain stronger in tasks dominated by source-specific knowledge or dataset-specific annotation regularities. 

---
# UIPress: Bringing Optical Token Compression to UI-to-Code Generation 

**Authors**: Dasen Dai, Shuoqi Li, Ronghao Chen, Huacan Wang, Biao Wu, Qizhen Lan  

**Link**: [PDF](https://arxiv.org/pdf/2604.09442)  

**Abstract**: UI-to-Code generation requires vision-language models (VLMs) to produce thousands of tokens of structured HTML/CSS from a single screenshot, making visual token efficiency critical. Existing compression methods either select tokens at inference time using task-agnostic heuristics, or zero out low-attention features without actually shortening the sequence -- neither truly reduces prefill latency or adapts to the non-uniform information density of UI screenshots. Meanwhile, optical (encoder-side learned) compression has shown strong results for document OCR, yet no prior work has adapted this paradigm to UI-to-Code generation. We propose UIPress, a lightweight learned compression module inserted between the frozen ViT encoder and the LLM decoder of Qwen3-VL-8B. UIPress combines depthwise-separable convolutions, element-guided spatial reweighting, and Transformer refinement to compress ${\sim}$6{,}700 visual tokens to a fixed budget of 256. Together with Low-Rank Adaptation (LoRA) on the decoder to bridge the representation gap, the entire system adds only ${\sim}$21.7M trainable parameters (0.26\% of the 8B base model). Under a fair comparison on the same base model against four baselines on Design2Code, UIPress at 256 tokens achieves a CLIP score of 0.8127, outperforming the uncompressed baseline by +7.5\% and the strongest inference-time method by +4.6\%, while delivering 9.1$\times$ time-to-first-token speedup. To the best of our knowledge, UIPress is the first encoder-side learned compression method for the UI-to-Code task. 

---
# Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios 

**Authors**: Hui Liu, Bin Zou, Kecheng Chen, Jie Liu, Wenya Wang, Haoliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.09377)  

**Abstract**: Large language models (LLMs) exhibit substantial variability in performance and computational cost across tasks and queries, motivating routing systems that select models to meet user-specific cost-performance trade-offs. However, existing routers generalize poorly in cold-start scenarios where in-domain training data is unavailable. We address this limitation with a multi-level task-profile-guided data synthesis framework that constructs a hierarchical task taxonomy and produces diverse question-answer pairs to approximate the test-time query distribution. Building on this, we introduce TRouter, a task-type-aware router approach that models query-conditioned cost and performance via latent task-type variables, with prior regularization derived from the synthesized task taxonomy. This design enhances TRouter's routing utility under both cold-start and in-domain settings. Across multiple benchmarks, we show that our synthesis framework alleviates cold-start issues and that TRouter delivers effective LLM routing. 

---
# Do LLMs Follow Their Own Rules? A Reflexive Audit of Self-Stated Safety Policies 

**Authors**: Avni Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2604.09189)  

**Abstract**: LLMs internalize safety policies through RLHF, yet these policies are never formally specified and remain difficult to inspect. Existing benchmarks evaluate models against external standards but do not measure whether models understand and enforce their own stated boundaries. We introduce the Symbolic-Neural Consistency Audit (SNCA), a framework that (1) extracts a model's self-stated safety rules via structured prompts, (2) formalizes them as typed predicates (Absolute, Conditional, Adaptive), and (3) measures behavioral compliance via deterministic comparison against harm benchmarks. Evaluating four frontier models across 45 harm categories and 47,496 observations reveals systematic gaps between stated policy and observed behavior: models claiming absolute refusal frequently comply with harmful prompts, reasoning models achieve the highest self-consistency but fail to articulate policies for 29% of categories, and cross-model agreement on rule types is remarkably low (11%). These results demonstrate that the gap between what LLMs say and what they do is measurable and architecture-dependent, motivating reflexive consistency audits as a complement to behavioral benchmarks. 

---
# Facet-Level Tracing of Evidence Uncertainty and Hallucination in RAG 

**Authors**: Passant Elchafei, Monorama Swain, Shahed Masoudian, Markus Schedl  

**Link**: [PDF](https://arxiv.org/pdf/2604.09174)  

**Abstract**: Retrieval-Augmented Generation (RAG) aims to reduce hallucination by grounding answers in retrieved evidence, yet hallucinated answers remain common even when relevant documents are available. Existing evaluations focus on answer-level or passage-level accuracy, offering limited insight into how evidence is used during generation. In this work, we introduce a facet-level diagnostics framework for QA that decomposes each input question into atomic reasoning facets. For each facet, we assess evidence sufficiency and grounding using a structured Facet x Chunk matrix that combines retrieval relevance with natural language inference-based faithfulness scores. To diagnose evidence usage, we analyze three controlled inference modes: Strict RAG, which enforces exclusive reliance on retrieved evidence; Soft RAG, which allows integration of retrieved evidence and parametric knowledge; and LLM-only generation without retrieval. Comparing these modes enables thorough analysis of retrieval-generation misalignment, defined as cases where relevant evidence is retrieved but not correctly integrated during generation. Across medical QA and HotpotQA, we evaluate three open-source and closed-source LLMs (GPT, Gemini, and LLaMA), providing interpretable diagnostics that reveal recurring facet-level failure modes, including evidence absence, evidence misalignment, and prior-driven overrides. Our results demonstrate that hallucinations in RAG systems are driven less by retrieval accuracy and more by how retrieved evidence is integrated during generation, with facet-level analysis exposing systematic evidence override and misalignment patterns that remain hidden under answer-level evaluation. 

---
# Many Ways to Be Fake: Benchmarking Fake News Detection Under Strategy-Driven AI Generation 

**Authors**: Xinyu Wang, Sai Koneru, Wenbo Zhang, Wenliang Zheng, Saksham Ranjan, Sarah Rajtmajer  

**Link**: [PDF](https://arxiv.org/pdf/2604.09514)  

**Abstract**: Recent advances in large language models (LLMs) have enabled the large-scale generation of highly fluent and deceptive news-like content. While prior work has often treated fake news detection as a binary classification problem, modern fake news increasingly arises through human-AI collaboration, where strategic inaccuracies are embedded within otherwise accurate and credible narratives. These mixed-truth cases represent a realistic and consequential threat, yet they remain underrepresented in existing benchmarks. To address this gap, we introduce MANYFAKE, a synthetic benchmark containing 6,798 fake news articles generated through multiple strategy-driven prompting pipelines that capture many ways fake news can be constructed and refined. Using this benchmark, we evaluate a range of state-of-the-art fake news detectors. Our results show that even advanced reasoning-enabled models approach saturation on fully fabricated stories, but remain brittle when falsehoods are subtle, optimized, and interwoven with accurate information. 

---
# ScheMatiQ: From Research Question to Structured Data through Interactive Schema Discovery 

**Authors**: Shahar Levy, Eliya Habba, Reshef Mintz, Barak Raveh, Renana Keydar, Gabriel Stanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2604.09237)  

**Abstract**: Many disciplines pose natural-language research questions over large document collections whose answers typically require structured evidence, traditionally obtained by manually designing an annotation schema and exhaustively labeling the corpus, a slow and error-prone process. We introduce ScheMatiQ, which leverages calls to a backbone LLM to take a question and a corpus to produce a schema and a grounded database, with a web interface that lets steer and revise the extraction. In collaboration with domain experts, we show that ScheMatiQ yields outputs that support real-world analysis in law and computational biology. We release ScheMatiQ as open source with a public web interface, and invite experts across disciplines to use it with their own data. All resources, including the website, source code, and demonstration video, are available at: this http URL 

---
# SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation 

**Authors**: Han Luo, Guy Laban  

**Link**: [PDF](https://arxiv.org/pdf/2604.09212)  

**Abstract**: Large language models are increasingly deployed in multi-turn settings such as tutoring, support, and counseling, where reliability depends on preserving consistent roles, personas, and goals across long horizons. This requirement becomes critical when LLMs are used to generate synthetic dialogues for training and evaluation, since LLM--LLM conversations can accumulate identity-related failures such as persona drift, role confusion, and "echoing", where one agent gradually mirrors its partner. We introduce SPASM (Stable Persona-driven Agent Simulation for Multi-turn dialogue generation), a modular, stability-first framework that decomposes simulation into (i) persona creation via schema sampling, plausibility validation, and natural-language persona crafting, (ii) Client--Responder dialogue generation, and (iii) termination detection for coherent stopping. To improve long-horizon stability without changing model weights, we propose Egocentric Context Projection (ECP): dialogue history is stored in a perspective-agnostic representation and deterministically projected into each agent's egocentric view before generation. Across three LLM backbones (GPT-4o-mini, DeepSeek-V3.2, Qwen-Plus) and nine Client--Responder pairings, we construct a dataset of 4,500 personas and 45,000 conversations (500 personas X 10 conversations per pairing). Ablations show ECP substantially reduces persona drift and, under human validation, eliminates echoing; embedding analyses recover persona structure and reveal strong responder-driven interaction geometry. Our code is available at this https URL. 

---
# EthicMind: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue 

**Authors**: Jiawen Deng, Wei Li, Wentao Zhang, Ziyun Jiao, Fuji Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.09265)  

**Abstract**: Intelligent dialogue systems are increasingly deployed in emotionally and ethically sensitive settings, where failures in either emotional attunement or ethical judgment can cause significant harm. Existing dialogue models typically address empathy and ethical safety in isolation, and often fail to adapt their behavior as ethical risk and user emotion evolve across multi-turn interactions. We formulate ethical-emotional alignment in dialogue as an explicit turn-level decision problem, and propose \textsc{EthicMind}, a risk-aware framework that implements this formulation in multi-turn dialogue at inference time. At each turn, \textsc{EthicMind} jointly analyzes ethical risk signals and user emotion, plans a high-level response strategy, and generates context-sensitive replies that balance ethical guidance with emotional engagement, without requiring additional model training. To evaluate alignment behavior under ethically complex interactions, we introduce a risk-stratified, multi-turn evaluation protocol with a context-aware user simulation procedure. Experimental results show that \textsc{EthicMind} achieves more consistent ethical guidance and emotional engagement than competitive baselines, particularly in high-risk and morally ambiguous scenarios. 

---
# Think Less, Know More: State-Aware Reasoning Compression with Knowledge Guidance for Efficient Reasoning 

**Authors**: Yi Sui, Chaozhuo Li, Dawei Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.09150)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance on complex tasks by leveraging long Chain-of-Thought (CoT), but often suffer from overthinking, leading to excessive reasoning steps and high inference latency. Existing CoT compression methods struggle to balance accuracy and efficiency, and lack fine-grained, step-level adaptation to redundancy and reasoning bias. Therefore, we propose State-Aware Reasoning Compression with Knowledge Guidance (STACK), a framework that performs step-wise CoT compression by explicitly modeling stage-specific redundancy sources and integrating with a retrieval-augmented guidance. STACK constructs online long-short contrastive samples and dynamically switches between knowledge-guided compression for uncertain or biased reasoning state and self-prompted compression for overly long but confident state, complemented by an answer-convergence-based early stopping mechanism to suppress redundant verification. We further propose a reward-difference-driven training strategy by combining Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO), enabling models to learn state-conditioned compression strategies. Experiments on three mathematical reasoning benchmarks show that STACK achieves a superior accuracy-efficiency balance, reducing average response length by 59.9% while improving accuracy by 4.8 points over existing methods. 

---
# NyayaMind- A Framework for Transparent Legal Reasoning and Judgment Prediction in the Indian Legal System 

**Authors**: Parjanya Aditya Shukla, Shubham Kumar Nigam, Debtanu Datta, Balaramamahanthi Deepak Patnaik, Noel Shallum, Pradeep Reddy Vanga, Saptarshi Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2604.09069)  

**Abstract**: Court Judgment Prediction and Explanation (CJPE) aims to predict a judicial decision and provide a legally grounded explanation for a given case based on the facts, legal issues, arguments, cited statutes, and relevant precedents. For such systems to be practically useful in judicial or legal research settings, they must not only achieve high predictive performance but also generate transparent and structured legal reasoning that aligns with established judicial practices. In this work, we present NyayaMind, an open-source framework designed to enable transparent and scalable legal reasoning for the Indian judiciary. The proposed framework integrates retrieval, reasoning, and verification mechanisms to emulate the structured decision-making process typically followed in courts. Specifically, NyayaMind consists of two main components: a Retrieval Module and a Prediction Module. The Retrieval Module employs a RAG pipeline to identify legally relevant statutes and precedent cases from large-scale legal corpora, while the Prediction Module utilizes reasoning-oriented LLMs fine-tuned for the Indian legal domain to generate structured outputs including issues, arguments, rationale, and the final decision. Our extensive results and expert evaluation demonstrate that NyayaMind significantly improves the quality of explanation and evidence alignment compared to existing CJPE approaches, providing a promising step toward trustworthy AI-assisted legal decision support systems. 

---
# Many-Tier Instruction Hierarchy in LLM Agents 

**Authors**: Jingyu Zhang, Tianjian Li, William Jurayj, Hongyuan Zhan, Benjamin Van Durme, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2604.09443)  

**Abstract**: Large language model agents receive instructions from many sources-system messages, user prompts, tool outputs, and more-each carrying different levels of trust and authority. When these instructions conflict, models must reliably follow the highest-privilege instruction to remain safe and effective. The dominant paradigm, instruction hierarchy (IH), assumes a fixed, small set of privilege levels (typically fewer than five) defined by rigid role labels (e.g., system > user). This is inadequate for real-world agentic settings, where conflicts can arise across far more sources and contexts. In this work, we propose Many-Tier Instruction Hierarchy (ManyIH), a paradigm for resolving instruction conflicts among instructions with arbitrarily many privilege levels. We introduce ManyIH-Bench, the first benchmark for ManyIH. ManyIH-Bench requires models to navigate up to 12 levels of conflicting instructions with varying privileges, comprising 853 agentic tasks (427 coding and 426 instruction-following). ManyIH-Bench composes constraints developed by LLMs and verified by humans to create realistic and difficult test cases spanning 46 real-world agents. Our experiments show that even the current frontier models perform poorly (~40% accuracy) when instruction conflict scales. This work underscores the urgent need for methods that explicitly target fine-grained, scalable instruction conflict resolution in agentic settings. 

---
# Anchored Sliding Window: Toward Robust and Imperceptible Linguistic Steganography 

**Authors**: Ruiyi Yan, Shiao Meng, Yugo Murawaki  

**Link**: [PDF](https://arxiv.org/pdf/2604.09066)  

**Abstract**: Linguistic steganography based on language models typically assumes that steganographic texts are transmitted without alteration, making them fragile to even minor modifications. While previous work mitigates this fragility by limiting the context window, it significantly compromises text quality. In this paper, we propose the anchored sliding window (ASW) framework to improve imperceptibility and robustness. In addition to the latest tokens, the prompt and a bridge context are anchored within the context window, encouraging the model to compensate for the excluded tokens. We formulate the optimization of the bridge context as a variant of prompt distillation, which we further extend using self-distillation strategies. Experiments show that our ASW significantly and consistently outperforms the baseline method in text quality, imperceptibility, and robustness across diverse settings. The code is available at this http URL. 

---
# PerMix-RLVR: Preserving Persona Expressivity under Verifiable-Reward Alignment 

**Authors**: Jihwan Oh, Soowon Oh, Murad Aghazada, Minchan Jeong, Sungnyun Kim, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2604.08986)  

**Abstract**: Persona prompting has been widely adopted to steer large language models (LLMs) behavior and improve their instruction performance by assigning specific characters. However, identifying an optimal persona is time-consuming, and its impact on output quality remains poorly understood. Prior work has mainly addressed this issue at the prompt level via inference-time strategies, incurring additional computation. In this work, we avoid inference-time prompt search by tackling persona sensitivity during training, aiming to train models that adapt their behavior to diverse personas while preserving task performance. In particular, we find that reinforcement learning with verifiable rewards (RLVR) systematically reduces sensitivity to persona prompts, but also reveals an inherent trade-off of outcome-based optimization: while RLVR improves robustness on tasks with verifiable goals, it can also degrade persona expressivity when needed, e.g., in-character role-playing. To address this limitation, we propose PerMix-RLVR, a persona-mixed RLVR strategy that mitigates the persona robustness-fidelity trade-off, preserving strong robustness to harmful persona variation while enabling faithful persona adoption when required. Concretely, PerMix-RLVR improves persona stability score (PSS) over RLVR by +21.2% on MATH500, while also enhancing persona fidelity by +11.4% on PersonaGym. 

---
# CONDESION-BENCH: Conditional Decision-Making of Large Language Models in Compositional Action Space 

**Authors**: Yeonjun Hwang, Sungyong Park, Minju Kim, Dongha Lee, Jinyoung Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2604.09029)  

**Abstract**: Large language models have been widely explored as decision-support tools in high-stakes domains due to their contextual understanding and reasoning capabilities. However, existing decision-making benchmarks rely on two simplifying assumptions: actions are selected from a finite set of pre-defined candidates, and explicit conditions restricting action feasibility are not incorporated into the decision-making process. These assumptions fail to capture the compositional structure of real-world actions and the explicit conditions that constrain their validity. To address these limitations, we introduce CONDESION-BENCH, a benchmark designed to evaluate conditional decision-making in compositional action space. In CONDESION-BENCH, actions are defined as allocations to decision variables and are restricted by explicit conditions at the variable, contextual, and allocation levels. By employing oracle-based evaluation of both decision quality and condition adherence, we provide a more rigorous assessment of LLMs as decision-support tools. 

---
# Hierarchical Alignment: Enforcing Hierarchical Instruction-Following in LLMs through Logical Consistency 

**Authors**: Shu Yang, Zihao Zhou, Di Wang, Wenda Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.09075)  

**Abstract**: Large language models increasingly operate under multiple instructions from heterogeneous sources with different authority levels, including system policies, user requests, tool outputs, and retrieved context. While prior work on instruction hierarchy highlights the importance of respecting instruction priorities, it mainly focuses on adversarial attacks and overlooks the benign but common instruction conflicts that arise in real-world applications. In such settings, models must not only avoid security violations but also preserve task utility and behavioral consistency when instructions partially or implicitly conflict. We propose Neuro-Symbolic Hierarchical Alignment (NSHA) for hierarchical instruction-following by explicitly modeling and enforcing instruction priorities. At inference time, we introduce solver-guided reasoning that formulates instruction resolution as a constraint satisfaction problem, enabling the model to derive a maximally consistent set of applicable instructions under hierarchical constraints. At training time, NSHA distills solver-based decisions into model parameters using automatically constructed supervision. We evaluate our approach on rule following, task execution, tool use, and safety, covering both single-turn and multi-turn interactions, and show that NSHA significantly improves performance under such conflicts while maintaining competitive utility in reference settings. 

---
# ASTRA: Adaptive Semantic Tree Reasoning Architecture for Complex Table Question Answering 

**Authors**: Xiaoke Guo, Songze Li, Zhiqiang Liu, Zhaoyan Gong, Yuanxiang Liu, Huajun Chen, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08999)  

**Abstract**: Table serialization remains a critical bottleneck for Large Language Models (LLMs) in complex table question answering, hindered by challenges such as structural neglect, representation gaps, and reasoning opacity. Existing serialization methods fail to capture explicit hierarchies and lack schema flexibility, while current tree-based approaches suffer from limited semantic adaptability. To address these limitations, we propose ASTRA (Adaptive Semantic Tree Reasoning Architecture) including two main modules, AdaSTR and DuTR. First, we introduce AdaSTR, which leverages the global semantic awareness of LLMs to reconstruct tables into Logical Semantic Trees. This serialization explicitly models hierarchical dependencies and employs an adaptive mechanism to optimize construction strategies based on table scale. Second, building on this structure, we present DuTR, a dual-mode reasoning framework that integrates tree-search-based textual navigation for linguistic alignment and symbolic code execution for precise verification. Experiments on complex table benchmarks demonstrate that our method achieves state-of-the-art (SOTA) performance. 

---
# Quantisation Reshapes the Metacognitive Geometry of Language Models 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.08976)  

**Abstract**: We report that model quantisation restructures domain-level metacognitive efficiency in LLMs rather than degrading it uniformly. Evaluating Llama-3-8B-Instruct on the same 3,000 questions at Q5_K_M and f16 precision, we find that M-ratio profiles across four knowledge domains are uncorrelated between formats (Spearman rho = 0.00). Arts & Literature moves from worst-monitored (M-ratio = 0.606 at Q5_K_M) to best-monitored (1.542 at f16). Geography moves from well-monitored (1.210) to under-monitored (0.798). However, Type-2 AUROC profiles are perfectly stable across formats (rho = 1.00), localising the restructuring to the M-ratio normalisation rather than the underlying discrimination signal. This finding emerged from a pre-registered attempt to improve metacognition through domain-conditional training. We prescribed confidence-amplification SFT for the diagnosed weak domain, with matched-budget agnostic and wrong-prescription controls. All four confirmatory hypotheses were null (10,000 bootstrap resamples, seed = 42). The training successfully reshaped confidence distributions, doubling the NLP gap in Science from 0.076 to 0.152, but did not improve meta-d' because the diagnostic profile did not transfer across formats. Any system relying on domain-level M-ratio profiles has an unexamined dependency on inference format. Systems using AUROC_2 are safer. We release all code, pre-registrations, and trial-level data. 

---
# MuTSE: A Human-in-the-Loop Multi-use Text Simplification Evaluator 

**Authors**: Rares-Alexandru Roscan, Gabriel Petre1, Adrian-Marius Dumitran, Angela-Liliana Dumitran  

**Link**: [PDF](https://arxiv.org/pdf/2604.08947)  

**Abstract**: As Large Language Models (LLMs) become increasingly prevalent in text simplification, systematically evaluating their outputs across diverse prompting strategies and architectures remains a critical methodological challenge in both NLP research and Intelligent Tutoring Systems (ITS). Developing robust prompts is often hindered by the absence of structured, visual frameworks for comparative text analysis. While researchers typically rely on static computational scripts, educators are constrained to standard conversational interfaces -- neither paradigm supports systematic multi-dimensional evaluation of prompt-model permutations. To address these limitations, we introduce \textbf{MuTSE}\footnote{The project code and the demo have been made available for peer review at the following anonymized URL. this https URL, an interactive human-in-the-loop web application designed to streamline the evaluation of LLM-generated text simplifications across arbitrary CEFR proficiency targets. The system supports concurrent execution of $P \times M$ prompt-model permutations, generating a comprehensive comparison matrix in real-time. By integrating a novel tiered semantic alignment engine augmented with a linearity bias heuristic ($\lambda$), MuTSE visually maps source sentences to their simplified counterparts, reducing the cognitive load associated with qualitative analysis and enabling reproducible, structured annotation for downstream NLP dataset construction. 

---
# Confident in a Confidence Score: Investigating the Sensitivity of Confidence Scores to Supervised Fine-Tuning 

**Authors**: Lorenzo Jaime Yu Flores, Cesare Spinoso di-Piano, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2604.08974)  

**Abstract**: Uncertainty quantification is a set of techniques that measure confidence in language models. They can be used, for example, to detect hallucinations or alert users to review uncertain predictions. To be useful, these confidence scores must be correlated with the quality of the output. However, recent work found that fine-tuning can affect the correlation between confidence scores and quality. Hence, we investigate the underlying behavior of confidence scores to understand its sensitivity to supervised fine-tuning (SFT). We find that post-SFT, the correlation of various confidence scores degrades, which can stem from changes in confidence scores due to factors other than the output quality, such as the output's similarity to the training distribution. We demonstrate via a case study how failing to address this miscorrelation reduces the usefulness of the confidence scores on a downstream task. Our findings show how confidence metrics cannot be used off-the-shelf without testing, and motivate the need for developing metrics which are more robust to fine-tuning. 

---
# Persona-E$^2$: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events 

**Authors**: Yuqin Yang, Haowu Zhou, Haoran Tu, Zhiwen Hui, Shiqi Yan, HaoYang Li, Dong She, Xianrong Yao, Yang Gao, Zhanpeng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2604.09162)  

**Abstract**: Most affective computing research treats emotion as a static property of text, focusing on the writer's sentiment while overlooking the reader's perspective. This approach ignores how individual personalities lead to diverse emotional appraisals of the same event. Although role-playing Large Language Models (LLMs) attempt to simulate such nuanced reactions, they often suffer from "personality illusion'' -- relying on surface-level stereotypes rather than authentic cognitive logic. A critical bottleneck is the absence of ground-truth human data to link personality traits to emotional shifts. To bridge the gap, we introduce Persona-E$^2$ (Persona-Event2Emotion), a large-scale dataset grounded in annotated MBTI and Big Five traits to capture reader-based emotional variations across news, social media, and life narratives. Extensive experiments reveal that state-of-the-art LLMs struggle to capture precise appraisal shifts, particularly in social media domains. Crucially, we find that personality information significantly improves comprehension, with the Big Five traits alleviating "personality illusion.' 

---
# GRASP: Grounded CoT Reasoning with Dual-Stage Optimization for Multimodal Sarcasm Target Identification 

**Authors**: Faxian Wan, Xiaocui Yang, Yifan Cao, Shi Feng, Daling Wang, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08879)  

**Abstract**: Moving beyond the traditional binary classification paradigm of Multimodal Sarcasm Detection, Multimodal Sarcasm Target Identification (MSTI) presents a more formidable challenge, requiring precise localization of fine-grained targets such as textual phrases and visual regions. Existing approaches predominantly rely on implicit cross-modal alignment, offering limited interpretability and suboptimal fine-grained localization. To address these limitations, we propose GRASP, Grounded Chain-of-Thought ReAsoning with Dual-Stage Optimization for Multimodal Sarcasm Prediction and Target Identification, a framework that integrates visual grounding with explicit Chain-of-Thought (CoT) reasoning to move beyond black-box MSTI. Specifically, we curate MSTI-MAX, a refined dataset that mitigates class imbalance and enriches multimodal sarcasm cues. We introduce Grounded CoT reasoning, which explicitly anchors sarcasm-related visual regions within the reasoning trajectory and prompts the model to articulate rationales before predicting the final classification labels and sarcasm targets. Furthermore, we employ a dual-stage outcome-supervised joint optimization strategy: Supervised Fine-Tuning with a coordinate-aware weighted loss, followed by Fine-Grained Target Policy Optimization. Extensive experiments demonstrate that GRASP outperforms existing baselines in fine-grained sarcasm target identification across modalities, and an LLM-as-a-Judge evaluation quantitatively measures the quality of internal reasoning chains. Our dataset and source code will be released on GitHub. 

---
# NCL-BU at SemEval-2026 Task 3: Fine-tuning XLM-RoBERTa for Multilingual Dimensional Sentiment Regression 

**Authors**: Tong Wu, Nicolay Rusnachenko, Huizhi Liang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08923)  

**Abstract**: Dimensional Aspect-Based Sentiment Analysis (DimABSA) extends traditional ABSA from categorical polarity labels to continuous valence-arousal (VA) regression. This paper describes a system developed for Track A - Subtask 1 (Dimensional Aspect Sentiment Regression), aiming to predict real-valued VA scores in the [1, 9] range for each given aspect in a text. A fine-tuning approach based on XLM-RoBERTa-base is adopted, constructing the input as [CLS] T [SEP] a_i [SEP] and training dual regression heads with sigmoid-scaled outputs for valence and arousal prediction. Separate models are trained for each language-domain combination (English and Chinese across restaurant, laptop, and finance domains), and training and development sets are merged for final test predictions. In development experiments, the fine-tuning approach is compared against several large language models including GPT-5.2, LLaMA-3-70B, LLaMA-3.3-70B, and LLaMA-4-Maverick under a few-shot prompting setting, demonstrating that task-specific fine-tuning substantially and consistently outperforms these LLM-based methods across all evaluation datasets. The code is publicly available at this https URL. 

---
# Interactive ASR: Towards Human-Like Interaction and Semantic Coherence Evaluation for Agentic Speech Recognition 

**Authors**: Peng Wang, Yanqiao Zhu, Zixuan Jiang, Qinyuan Chen, Xingjian Zhao, Xipeng Qiu, Wupeng Wang, Zhifu Gao, Xiangang Li, Kai Yu, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.09121)  

**Abstract**: Recent years have witnessed remarkable progress in automatic speech recognition (ASR), driven by advances in model architectures and large-scale training data. However, two important aspects remain underexplored. First, Word Error Rate (WER), the dominant evaluation metric for decades, treats all words equally and often fails to reflect the semantic correctness of an utterance at the sentence level. Second, interactive correction-an essential component of human communication-has rarely been systematically studied in ASR research. In this paper, we integrate these two perspectives under an agentic framework for interactive ASR. We propose leveraging LLM-as-a-Judge as a semantic-aware evaluation metric to assess recognition quality beyond token-level accuracy. Furthermore, we design an LLM-driven agent framework to simulate human-like multi-turn interaction, enabling iterative refinement of recognition outputs through semantic feedback. Extensive experiments are conducted on standard benchmarks, including GigaSpeech (English), WenetSpeech (Chinese), the ASRU 2019 code-switching test set. Both objective and subjective evaluations demonstrate the effectiveness of the proposed framework in improving semantic fidelity and interactive correction capability. We will release the code to facilitate future research in interactive and agentic ASR. 

---
# Lessons Without Borders? Evaluating Cultural Alignment of LLMs Using Multilingual Story Moral Generation 

**Authors**: Sophie Wu, Andrew Piper  

**Link**: [PDF](https://arxiv.org/pdf/2604.08797)  

**Abstract**: Stories are key to transmitting values across cultures, but their interpretation varies across linguistic and cultural contexts. Thus, we introduce multilingual story moral generation as a novel culturally grounded evaluation task. Using a new dataset of human-written story morals collected across 14 language-culture pairs, we compare model outputs with human interpretations via semantic similarity, a human preference survey, and value categorization. We show that frontier models such as GPT-4o and Gemini generate story morals that are semantically similar to human responses and preferred by human evaluators. However, their outputs exhibit markedly less cross-linguistic variation and concentrate on a narrower set of widely shared values. These findings suggest that while contemporary models can approximate central tendencies of human moral interpretation, they struggle to reproduce the diversity that characterizes human narrative understanding. By framing narrative interpretation as an evaluative task, this work introduces a new approach to studying cultural alignment in language models beyond static benchmarks or knowledge-based tests. 

---
# Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models 

**Authors**: Shun Zou, Yong Wang, Zehui Chen, Lin Chen, Chongyang Tao, Feng Zhao, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08964)  

**Abstract**: Diffusion Large Language Models (dLLMs) have recently become a promising alternative to autoregressive large language models (ARMs). Semi-autoregressive (Semi-AR) decoding is widely employed in base dLLMs and advanced decoding strategies due to its superior performance. However, our observations reveal that Semi-AR decoding suffers from inherent block constraints, which cause the decoding of many cross-block stable tokens to be unnecessarily delayed. To address this challenge, we systematically investigate the identification of stable tokens and present three key findings: (1) naive lookahead decoding is unreliable, (2) token stability closely correlates with convergence trend, and (3) historical information is isolated. Building on these insights, we propose Anchor-based History-stable Decoding (AHD), a training-free, plug-and-play dynamic decoding strategy. Specifically, AHD monitors the stability trend of tokens in real time through dynamic anchors. Once a token reaches stability, it initiates early cross-block decoding to enhance efficiency and performance. Extensive experiments across language, vision-language, and audio-language domains demonstrate that AHD simultaneously improves both performance and inference efficiency. Notably, AHD effectively reverses the performance degradation typically observed in existing advanced decoding acceleration strategies. For instance, on the BBH benchmark, our approach reduces decoding steps by 80% while improving performance by 3.67%. 

---
# Litmus (Re)Agent: A Benchmark and Agentic System for Predictive Evaluation of Multilingual Models 

**Authors**: Avni Mittal, Shanu Kumar, Sandipan Dandapat, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2604.08970)  

**Abstract**: We study predictive multilingual evaluation: estimating how well a model will perform on a task in a target language when direct benchmark results are missing. This problem is common in multilingual deployment, where evaluation coverage is sparse and published evidence is uneven across languages, tasks, and model families. We introduce a controlled benchmark of 1,500 questions spanning six tasks and five evidence scenarios. The benchmark separates accessible evidence from ground truth, enabling evaluation of systems that must infer missing results from incomplete literature evidence. We also present Litmus (Re)Agent, a DAG-orchestrated agentic system that decomposes queries into hypotheses, retrieves evidence, and synthesises predictions through feature-aware aggregation. Across six systems, Litmus (Re)Agent achieves the best overall performance, with the largest gains in transfer-heavy scenarios where direct evidence is weak or absent. These results show that structured agentic reasoning is a promising approach to multilingual performance estimation under incomplete evidence. 

---
# Cards Against LLMs: Benchmarking Humor Alignment in Large Language Models 

**Authors**: Yousra Fettach, Guillaume Bied, Hannu Toivonen, Tijl De Bie  

**Link**: [PDF](https://arxiv.org/pdf/2604.08757)  

**Abstract**: Humor is one of the most culturally embedded and socially significant dimensions of human communication, yet it remains largely unexplored as a dimension of Large Language Model (LLM) alignment. In this study, five frontier language models play the same Cards Against Humanity games (CAH) as human players. The models select the funniest response from a slate of ten candidate cards across 9,894 rounds. While all models exceed the random baseline, alignment with human preference remains modest. More striking is that models agree with each other substantially more often than they agree with humans. We show that this preference is partly explained by systematic position biases and content preferences, raising the question whether LLM humor judgment reflects genuine preference or structural artifacts of inference and alignment. 

---
# TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice 

**Authors**: Gang Hu, Yating Chen, Haiyan Ding, Wang Gao, Jiajia Huang, Min Peng, Qianqian Xie, Kun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08948)  

**Abstract**: While Large Language Models (LLMs) excel in various general domains, they exhibit notable gaps in the highly specialized, knowledge-intensive, and legally regulated Chinese tax domain. Consequently, while tax-related benchmarks are gaining attention, many focus on isolated NLP tasks, neglecting real-world practical capabilities. To address this issue, we introduce TaxPraBen, the first dedicated benchmark for Chinese taxation practice. It combines 10 traditional application tasks, along with 3 pioneering real-world scenarios: tax risk prevention, tax inspection analysis, and tax strategy planning, sourced from 14 datasets totaling 7.3K instances. TaxPraBen features a scalable structured evaluation paradigm designed through process of "structured parsing-field alignment extraction-numerical and textual matching", enabling end-to-end tax practice assessment while being extensible to other domains. We evaluate 19 LLMs based on Bloom's taxonomy. The results indicate significant performance disparities: all closed-source large-parameter LLMs excel, and Chinese LLMs like Qwen2.5 generally exceed multilingual LLMs, while the YaYi2 LLM, fine-tuned with some tax data, shows only limited improvement. TaxPraBen serves as a vital resource for advancing evaluations of LLMs in practical applications. 

---
# MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits 

**Authors**: Yixin Xiang, Yunshan Ma, Xiaoyu Du, Yibing Chen, Yanxin Zhang, Jinhui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08952)  

**Abstract**: Document Question Answering (DQA) involves generating answers from a document based on a user's query, representing a key task in document understanding. This task requires interpreting visual layouts, which has prompted recent studies to adopt multimodal Retrieval-Augmented Generation (RAG) that processes page images for answer generation. However, in multimodal RAG, visual DQA struggles to utilize a large number of images effectively, as the retrieval stage often retains only a few candidate pages (e.g., Top-4), causing informative but less visually salient content to be overlooked in favor of common yet low-information pages. To address this issue, we propose a Multi-Armed Bandit-based DQA framework (MAB-DQA) to explicitly model the varying importance of multiple implicit aspects in a query. Specifically, MAB-DQA decomposes a query into aspect-aware subqueries and retrieves an aspect-specific candidate set for each. It treats each subquery as an arm and uses preliminary reasoning results from a small number of representative pages as reward signals to estimate aspect utility. Guided by an exploration-exploitation policy, MAB-DQA dynamically reallocates retrieval budgets toward high-value aspects. With the most informative pages and their correlations, MAB-DQA generates the expected results. On four benchmarks, MAB-DQA shows an average improvement of 5%-18% over the state-of-the-art method, consistently enhancing document understanding. Code at this https URL. 

---
# LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs 

**Authors**: Paolo Gajo, Domenic Rosati, Hassan Sajjad, Alberto Barrón-Cedeño  

**Link**: [PDF](https://arxiv.org/pdf/2604.08752)  

**Abstract**: Relation extraction represents a fundamental component in the process of creating knowledge graphs, among other applications. Large language models (LLMs) have been adopted as a promising tool for relation extraction, both in supervised and in-context learning settings. However, in this work we show that their performance still lags behind much smaller architectures when the linguistic graph underlying a text has great complexity. To demonstrate this, we evaluate four LLMs against a graph-based parser on six relation extraction datasets with sentence graphs of varying sizes and complexities. Our results show that the graph-based parser increasingly outperforms the LLMs, as the number of relations in the input documents increases. This makes the much lighter graph-based parser a superior choice in the presence of complex linguistic graphs. 

---
# MedConceal: A Benchmark for Clinical Hidden-Concern Reasoning Under Partial Observability 

**Authors**: Yikun Han, Joey Chan, Jingyuan Chen, Mengting Ai, Simo Du, Yue Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.08788)  

**Abstract**: Patient-clinician communication is an asymmetric-information problem: patients often do not disclose fears, misconceptions, or practical barriers unless clinicians elicit them skillfully. Effective medical dialogue therefore requires reasoning under partial observability: clinicians must elicit latent concerns, confirm them through interaction, and respond in ways that guide patients toward appropriate care. However, existing medical dialogue benchmarks largely sidestep this challenge by exposing hidden patient state, collapsing elicitation into extraction, or evaluating responses without modeling what remains hidden. We present MedConceal, a benchmark with an interactive patient simulator for evaluating hidden-concern reasoning in medical dialogue, comprising 300 curated cases and 600 clinician-LLM interactions. Built from clinician-answered online health discussions, each case pairing clinician-visible context with simulator-internal hidden concerns derived from prior literature and structured using an expert-developed taxonomy. The simulator withholds these concerns from the dialogue agent, tracks whether they have been revealed and addressed via theory-grounded turn-level communication signals, and is clinician-reviewed for clinical plausibility. This enables process-aware evaluation of both task success and the interaction process that leads to it. We study two abilities: confirmation, surfacing hidden concerns through multi-turn dialogue, and intervention, addressing the primary concern and guiding the patient toward a target plan. Results show that no single system dominates: frontier models lead on different confirmation metrics, while human clinicians (N=159) remain strongest on intervention success. Together, these results identify hidden-concern reasoning under partial observability as a key unresolved challenge for medical dialogue systems. 

---
# MT-OSC: Path for LLMs that Get Lost in Multi-Turn Conversation 

**Authors**: Jyotika Singh, Fang Tu, Miguel Ballesteros, Weiyi Sun, Sandip Ghoshal, Michelle Yuan, Yassine Benajiba, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2604.08782)  

**Abstract**: Large language models (LLMs) suffer significant performance degradation when user instructions and context are distributed over multiple conversational turns, yet multi-turn (MT) interactions dominate chat interfaces. The routine approach of appending full chat history to prompts rapidly exhausts context windows, leading to increased latency, higher computational costs, and diminishing returns as conversations extend. We introduce MT-OSC, a One-off Sequential Condensation framework that efficiently and automatically condenses chat history in the background without disrupting the user experience. MT-OSC employs a Condenser Agent that uses a few-shot inference-based Condenser and a lightweight Decider to selectively retain essential information, reducing token counts by up to 72% in 10-turn dialogues. Evaluated across 13 state-of-the-art LLMs and diverse multi-turn benchmarks, MT-OSC consistently narrows the multi-turn performance gap - yielding improved or preserved accuracy across datasets while remaining robust to distractors and irrelevant turns. Our results establish MT-OSC as a scalable solution for multi-turn chats, enabling richer context within constrained input spaces, reducing latency and operational cost, while balancing performance. 

---
# Cross-Lingual Attention Distillation with Personality-Informed Generative Augmentation for Multilingual Personality Recognition 

**Authors**: Jing Jie Tan, Ban-Hoe Kwan, Danny Wee-Kiat Ng, Yan-Chai Hum, Noriyuki Kawarazaki, Kosuke Takano  

**Link**: [PDF](https://arxiv.org/pdf/2604.08851)  

**Abstract**: While significant work has been done on personality recognition, the lack of multilingual datasets remains an unresolved challenge. To address this, we propose ADAM (Cross-Lingual (A)ttention (D)istillation with Personality-Guided Generative (A)ugmentation for (M)ultilingual Personality Recognition), a state-of-the-art approach designed to advance multilingual personality recognition. Our approach leverages an existing English-language personality dataset as the primary source and employs a large language model (LLM) for translationbased augmentation, enhanced by Personality-Informed Generative Augmentation (PIGA), to generate high-quality training data in multiple languages, including Japanese, Chinese, Malay, and French. We provide a thorough analysis to justify the effectiveness of these augmentation techniques. Building on these advancements, ADAM integrates Cross-Lingual Attention Distillation (CLAD) to train a model capable of understanding and recognizing personality traits across languages, bridging linguistic and cultural gaps in personality analysis. This research presents a thorough evaluation of the proposed augmentation method, incorporating an ablation study on recognition performance to ensure fair comparisons and robust validation. Overall, with PIGA augmentation, the findings demonstrate that CLAD significantly outperforms the standard BCE across all languages and personality traits, achieving notable improvements in average BA scores - 0.6332 (+0.0573) on the Essays dataset and 0.7448 (+0.0968) on the Kaggle dataset. The CLAD-trained model also demonstrated strong generalizability and achieved benchmark performance comparable to current leading encoder models. The model weight, dataset, and algorithm repository are available at this https URL. 

---
# Can We Still Hear the Accent? Investigating the Resilience of Native Language Signals in the LLM Era 

**Authors**: Nabelanita Utami, Sasano Ryohei  

**Link**: [PDF](https://arxiv.org/pdf/2604.08568)  

**Abstract**: The evolution of writing assistance tools from machine translation to large language models (LLMs) has changed how researchers write. This study investigates whether this shift is homogenizing research papers by analyzing native language identification (NLI) trends in ACL Anthology papers across three eras: pre-neural network (NN), pre-LLM, and post-LLM. We construct a labeled dataset using a semi-automated framework and fine-tune a classifier to detect linguistic fingerprints of author backgrounds. Our analysis shows a consistent decline in NLI performance over time. Interestingly, the post-LLM era reveals anomalies: while Chinese and French show unexpected resistance or divergent trends, Japanese and Korean exhibit sharper-than-expected declines. 

---
# Sentiment Classification of Gaza War Headlines: A Comparative Analysis of Large Language Models and Arabic Fine-Tuned BERT Models 

**Authors**: Amr Eleraqi, Hager H. Mustafa, Abdul Hadi N. Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2604.08566)  

**Abstract**: This study examines how different artificial intelligence architectures interpret sentiment in conflict-related media discourse, using the 2023 Gaza War as a case study. Drawing on a corpus of 10,990 Arabic news headlines (Eleraqi 2026), the research conducts a comparative analysis between three large language models and six fine-tuned Arabic BERT models. Rather than evaluating accuracy against a single human-annotated gold standard, the study adopts an epistemological approach that treats sentiment classification as an interpretive act produced by model architectures. To quantify systematic differences across models, the analysis employs information-theoretic and distributional metrics, including Shannon Entropy, Jensen-Shannon Distance, and a Variance Score measuring deviation from aggregate model behavior. The results reveal pronounced and non-random divergence in sentiment distributions. Fine-tuned BERT models, particularly MARBERT, exhibit a strong bias toward neutral classifications, while LLMs consistently amplify negative sentiment, with LLaMA-3.1-8B showing near-total collapse into negativity. Frame-conditioned analysis further demonstrates that GPT-4.1 adjusts sentiment judgments in line with narrative frames (e.g., humanitarian, legal, security), whereas other LLMs display limited contextual modulation. These findings suggest that the choice of model constitutes a choice of interpretive lens, shaping how conflict narratives are algorithmically framed and emotionally evaluated. The study contributes to media studies and computational social science by foregrounding algorithmic discrepancy as an object of analysis and by highlighting the risks of treating automated sentiment outputs as neutral or interchangeable measures of media tone in contexts of war and crisis. 

---
# Decomposing the Delta: What Do Models Actually Learn from Preference Pairs? 

**Authors**: Chia-Hsuan Lee, Mingyang Zhou, Renkun Ni, Zelei Cheng, Sihui Dai, Supriyo Chakraborty, Shixiong Zhang, Sambit Sahu, William Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2604.08723)  

**Abstract**: Preference optimization methods such as DPO and KTO are widely used for aligning language models, yet little is understood about what properties of preference data drive downstream reasoning gains. We ask: what aspects of a preference pair improve a reasoning model's performance on general reasoning tasks? We investigate two distinct notions of quality delta in preference data: generator-level delta, arising from the differences in capability between models that generate chosen and rejected reasoning traces, and sample-level delta, arising from differences in judged quality differences within an individual preference pair. To study generator-level delta, we vary the generator's scale and model family, and to study sample-level delta, we employ an LLM-as-a-judge to rate the quality of generated traces along multiple reasoning-quality dimensions. We find that increasing generator-level delta steadily improves performance on out-of-domain reasoning tasks and filtering data by sample-level delta can enable more data-efficient training. Our results suggest a twofold recipe for improving reasoning performance through preference optimization: maximize generator-level delta when constructing preference pairs and exploit sample-level delta to select the most informative training examples. 

---
# EXAONE 4.5 Technical Report 

**Authors**: Eunbi Choi, Kibong Choi, Sehyun Chun, Seokhee Hong, Junwon Hwang, Hyojin Jeon, Ahra Jo, Hyunjik Jo, Yeonsik Jo, Joonkee Kim, Seonghwan Kim, Soyeon Kim, Sunkyoung Kim, Yireun Kim, Yongil Kim, Changhun Lee, Haeju Lee, Jinsik Lee, Kyungmin Lee, Sangha Park, Kwangrok Ryoo, Minju Seo, Sejong Yang, Heuiyeen Yeen, Hwan Chang, Stanley Jungkyu Choi, Yejin Choi, Kyubeen Han, Joonwon Jang, Kijeong Jeon, Geunyeong Jeong, Gerrard Jeongwon Jo, Jiyeon Jung, Daeseong Kim, Dohoon Kim, Dohyun Kim, Hyunseo Kim, Minu Kim, Myoungshin Kim, Youchul Kim, Byungoh Ko, Christopher Lee, Edward Hwayoung Lee, Honglak Lee, Jiyoung Lee, Sangeun Lee, Seungwon Lim, Woohyung Lim, Jueun Mun, Jaewoo Park, Jimin Park, Jinho Park, Yongmin Park, Wooseok Seo, Yongwoo Song, Sihyuk Yi, Kyungjae Yoo, Sangyeon Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2604.08644)  

**Abstract**: This technical report introduces EXAONE 4.5, the first open-weight vision language model released by LG AI Research. EXAONE 4.5 is architected by integrating a dedicated visual encoder into the existing EXAONE 4.0 framework, enabling native multimodal pretraining over both visual and textual modalities. The model is trained on large-scale data with careful curation, particularly emphasizing document-centric corpora that align with LG's strategic application domains. This targeted data design enables substantial performance gains in document understanding and related tasks, while also delivering broad improvements across general language capabilities. EXAONE 4.5 extends context length up to 256K tokens, facilitating long-context reasoning and enterprise-scale use cases. Comparative evaluations demonstrate that EXAONE 4.5 achieves competitive performance in general benchmarks while outperforming state-of-the-art models of similar scale in document understanding and Korean contextual reasoning. As part of LG's ongoing effort toward practical industrial deployment, EXAONE 4.5 is designed to be continuously extended with additional domains and application scenarios to advance AI for a better life. 

---
# Temperature-Dependent Performance of Prompting Strategies in Extended Reasoning Large Language Models 

**Authors**: Mousa Salah, Amgad Muneer  

**Link**: [PDF](https://arxiv.org/pdf/2604.08563)  

**Abstract**: Extended reasoning models represent a transformative shift in Large Language Model (LLM) capabilities by enabling explicit test-time computation for complex problem solving. However, the optimal configuration of sampling temperature and prompting strategy for these systems remains largely underexplored. We systematically evaluate chain-of-thought and zero-shot prompting across four temperature settings (0.0, 0.4, 0.7, and 1.0) using Grok-4.1 with extended reasoning on 39 mathematical problems from AMO-Bench, a challenging International Mathematical Olympiad-level benchmark. We find that zero-shot prompting achieves peak performance at moderate temperatures, reaching 59% accuracy at T=0.4 and T=0.7, while chain-of-thought prompting performs best at the temperature extremes. Most notably, the benefit of extended reasoning increases from 6x at T=0.0 to 14.3x at T=1.0. These results suggest that temperature should be optimized jointly with prompting strategy, challenging the common practice of using T=0 for reasoning tasks. 

---
# Adaptive Rigor in AI System Evaluation using Temperature-Controlled Verdict Aggregation via Generalized Power Mean 

**Authors**: Aleksandr Meshkov  

**Link**: [PDF](https://arxiv.org/pdf/2604.08595)  

**Abstract**: Existing evaluation methods for LLM-based AI systems, such as LLM-as-a-Judge, verdict systems, and NLI, do not always align well with human assessment because they cannot adapt their strictness to the application domain. This paper presents Temperature-Controlled Verdict Aggregation (TCVA), a method that combines a five-level verdict-scoring system with generalized power-mean aggregation and an intuitive temperature parameter T [0.1, 1.0] to control evaluation rigor. Low temperatures yield pessimistic scores suited for safety-critical domains; high temperatures produce lenient scores appropriate for conversational AI. Experimental evaluation on three benchmark datasets with human Likert-scale annotations (SummEval and USR) shows that TCVA achieves correlation with human judgments comparable to RAGAS on faithfulness (Spearman = 0.667 vs. 0.676) while consistently outperforming DeepEval. The method requires no additional LLM calls when adjusting the temperature parameter. 

---
# Attention-Based Sampler for Diffusion Language Models 

**Authors**: Yuyan Zhou, Kai Syun Hou, Weiyu Chen, James Kwok  

**Link**: [PDF](https://arxiv.org/pdf/2604.08564)  

**Abstract**: Auto-regressive models (ARMs) have established a dominant paradigm in language modeling. However, their strictly sequential decoding paradigm imposes fundamental constraints on both inference efficiency and modeling flexibility. To address these limitations, diffusion-based large language models (dLLMs) have been proposed, offering the potential for parallel decoding and flexible language modeling. Despite these advantages, current dLLMs decoding strategies rely primarily on token level information, which fails to account for global sequence structure and often yields suboptimal results. In this paper, we study the decoding order selection problem from the perspective of log-likelihood maximization. We theoretically demonstrate that optimal sequence likelihood can be approximately achieved by decoding tokens in descending order of their attention matrix column sums. This finding provides a principled justification for attention-guided decoding and offers a theoretically grounded alternative to greedy search. We instantiate this theoretical insight in a new training-free decoding algorithm, termed Attn-Sampler, and further propose a block attention approximation and dynamic attention thresholding for practical acceleration. Extensive experiments across multiple benchmarks validate the effectiveness of our proposed method, demonstrating that it achieves superior generation quality while enhancing the decoding parallelism. 

---
# A Representation-Level Assessment of Bias Mitigation in Foundation Models 

**Authors**: Svetoslav Nizhnichenkov, Rahul Nair, Elizabeth Daly, Brian Mac Namee  

**Link**: [PDF](https://arxiv.org/pdf/2604.08561)  

**Abstract**: We investigate how successful bias mitigation reshapes the embedding space of encoder-only and decoder-only foundation models, offering an internal audit of model behaviour through representational analysis. Using BERT and Llama2 as representative architectures, we assess the shifts in associations between gender and occupation terms by comparing baseline and bias-mitigated variants of the models. Our findings show that bias mitigation reduces gender-occupation disparities in the embedding space, leading to more neutral and balanced internal representations. These representational shifts are consistent across both model types, suggesting that fairness improvements can manifest as interpretable and geometric transformations. These results position embedding analysis as a valuable tool for understanding and validating the effectiveness of debiasing methods in foundation models. To further promote the assessment of decoder-only models, we introduce WinoDec, a dataset consisting of 4,000 sequences with gender and occupation terms, and release it to the general public. (this https URL) 

---
# Scalable High-Recall Constraint-Satisfaction-Based Information Retrieval for Clinical Trials Matching 

**Authors**: Cyrus Zhou, Yufei Jin, Yilin Xu, Yu-Chiang Wang, Chieh-Ju Chao, Monica S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2604.08849)  

**Abstract**: Clinical trials are central to evidence-based medicine, yet many struggle to meet enrollment targets, despite the availability of over half a million trials listed on this http URL, which attracts approximately two million users monthly. Existing retrieval techniques, largely based on keyword and embedding-similarity matching between patient profiles and eligibility criteria, often struggle with low recall, low precision, and limited interpretability due to complex constraints. We propose SatIR, a scalable clinical trial retrieval method based on constraint satisfaction, enabling high-precision and interpretable matching of patients to relevant trials. Our approach uses formal methods -- Satisfiability Modulo Theories (SMT) and relational algebra -- to efficiently represent and match key constraints from clinical trials and patient records. Beyond leveraging established medical ontologies and conceptual models, we use Large Language Models (LLMs) to convert informal reasoning regarding ambiguity, implicit clinical assumptions, and incomplete patient records into explicit, precise, controllable, and interpretable formal constraints. Evaluated on 59 patients and 3,621 trials, SatIR outperforms TrialGPT on all three evaluated retrieval objectives. It retrieves 32%-72% more relevant-and-eligible trials per patient, improves recall over the union of useful trials by 22-38 points, and serves more patients with at least one useful trial. Retrieval is fast, requiring 2.95 seconds per patient over 3,621 trials. These results show that SatIR is scalable, effective, and interpretable. 

---
# Neural networks for Text-to-Speech evaluation 

**Authors**: Ilya Trofimenko, David Kocharyan, Aleksandr Zaitsev, Pavel Repnikov, Mark Levin, Nikita Shevtsov  

**Link**: [PDF](https://arxiv.org/pdf/2604.08562)  

**Abstract**: Ensuring that Text-to-Speech (TTS) systems deliver human-perceived quality at scale is a central challenge for modern speech technologies. Human subjective evaluation protocols such as Mean Opinion Score (MOS) and Side-by-Side (SBS) comparisons remain the de facto gold standards, yet they are expensive, slow, and sensitive to pervasive assessor biases. This study addresses these barriers by formulating, and implementing a suite of novel neural models designed to approximate expert judgments in both relative (SBS) and absolute (MOS) settings. For relative assessment, we propose NeuralSBS, a HuBERT-backed model achieving 73.7% accuracy (on SOMOS dataset). For absolute assessment, we introduce enhancements to MOSNet using custom sequence-length batching, as well as WhisperBert, a multimodal stacking ensemble that combines Whisper audio features and BERT textual embeddings via weak learners. Our best MOS models achieve a Root Mean Square Error (RMSE) of ~0.40, significantly outperforming the human inter-rater RMSE baseline of 0.62. Furthermore, our ablation studies reveal that naively fusing text via cross-attention can degrade performance, highlighting the effectiveness of ensemble-based stacking over direct latent fusion. We additionally report negative results with SpeechLM-based architectures and zero-shot LLM evaluators (Qwen2-Audio, Gemini 2.5 flash preview), reinforcing the necessity of dedicated metric learning frameworks. 

---
# Medical Reasoning with Large Language Models: A Survey and MR-Bench 

**Authors**: Xiaohan Ren, Chenxiao Fan, Wenyin Ma, Hongliang He, Chongming Gao, Xiaoyan Zhao, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.08559)  

**Abstract**: Large language models (LLMs) have achieved strong performance on medical exam-style tasks, motivating growing interest in their deployment in real-world clinical settings. However, clinical decision-making is inherently safety-critical, context-dependent, and conducted under evolving evidence. In such situations, reliable LLM performance depends not on factual recall alone, but on robust medical reasoning. In this work, we present a comprehensive review of medical reasoning with LLMs. Grounded in cognitive theories of clinical reasoning, we conceptualize medical reasoning as an iterative process of abduction, deduction, and induction, and organize existing methods into seven major technical routes spanning training-based and training-free approaches. We further conduct a unified cross-benchmark evaluation of representative medical reasoning models under a consistent experimental setting, enabling a more systematic and comparable assessment of the empirical impact of existing methods. To better assess clinically grounded reasoning, we introduce MR-Bench, a benchmark derived from real-world hospital data. Evaluations on MR-Bench expose a pronounced gap between exam-level performance and accuracy on authentic clinical decision tasks. Overall, this survey provides a unified view of existing medical reasoning methods, benchmarks, and evaluation practices, and highlights key gaps between current model performance and the requirements of real-world clinical reasoning. 

---
# Multi-User Large Language Model Agents 

**Authors**: Shu Yang, Shenzhe Zhu, Hao Zhu, José Ramón Enríquez, Di Wang, Alex Pentland, Michiel A. Bakker, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2604.08567)  

**Abstract**: Large language models (LLMs) and LLM-based agents are increasingly deployed as assistants in planning and decision making, yet most existing systems are implicitly optimized for a single-principal interaction paradigm, in which the model is designed to satisfy the objectives of one dominant user whose instructions are treated as the sole source of authority and utility. However, as they are integrated into team workflows and organizational tools, they are increasingly required to serve multiple users simultaneously, each with distinct roles, preferences, and authority levels, leading to multi-user, multi-principal settings with unavoidable conflicts, information asymmetry, and privacy constraints. In this work, we present the first systematic study of multi-user LLM agents. We begin by formalizing multi-user interaction with LLM agents as a multi-principal decision problem, where a single agent must account for multiple users with potentially conflicting interests and associated challenges. We then introduce a unified multi-user interaction protocol and design three targeted stress-testing scenarios to evaluate current LLMs' capabilities in instruction following, privacy preservation, and coordination. Our results reveal systematic gaps: frontier LLMs frequently fail to maintain stable prioritization under conflicting user objectives, exhibit increasing privacy violations over multi-turn interactions, and suffer from efficiency bottlenecks when coordination requires iterative information gathering. 

---
# Dynamic sparsity in tree-structured feed-forward layers at scale 

**Authors**: Reza Sedghi, Robin Schiewer, Anand Subramoney, David Kappel  

**Link**: [PDF](https://arxiv.org/pdf/2604.08565)  

**Abstract**: At typical context lengths, the feed-forward MLP block accounts for a large share of a transformer's compute budget, motivating sparse alternatives to dense MLP blocks. We study sparse, tree-structured feed-forward layers as drop-in replacements for MLP blocks in deep transformer architectures, enabling conditional computation via hard hierarchical routing without a separate router network. We demonstrate for the first time that this form of tree-structured conditional sparsity can be applied for autoregressive language modeling and downstream question answering, including zero- and few-shot settings, and its scalability beyond 1B parameters. Despite activating fewer than 5% of the feed-forward block's units per token, our models match dense baselines under controlled training and fine-tuning protocols. We further analyze training dynamics and identify an emergent auto-pruning effect: the interaction of hard routing with asymmetric nonlinearities progressively deactivates unused paths, yielding partial conversion of dynamic routing into static structural sparsity. We show that simple architectural choices can modulate this behavior and recover balanced trees without auxiliary losses. Overall, our work demonstrates that tree-structured feed-forward layers provide a scalable and controllable mechanism for sparsifying large transformer models. 

---
# SynDocDis: A Metadata-Driven Framework for Generating Synthetic Physician Discussions Using Large Language Models 

**Authors**: Beny Rubinstein, Sergio Matos  

**Link**: [PDF](https://arxiv.org/pdf/2604.08555)  

**Abstract**: Physician-physician discussions of patient cases represent a rich source of clinical knowledge and reasoning that could feed AI agents to enrich and even participate in subsequent interactions. However, privacy regulations and ethical considerations severely restrict access to such data. While synthetic data generation using Large Language Models offers a promising alternative, existing approaches primarily focus on patient-physician interactions or structured medical records, leaving a significant gap in physician-to-physician communication synthesis. We present SynDocDis, a novel framework that combines structured prompting techniques with privacy-preserving de-identified case metadata to generate clinically accurate physician-to-physician dialogues. Evaluation by five practicing physicians in nine oncology and hepatology scenarios demonstrated exceptional communication effectiveness (mean 4.4/5) and strong medical content quality (mean 4.1/5), with substantial interrater reliability (kappa = 0.70, 95% CI: 0.67-0.73). The framework achieved 91% clinical relevance ratings while maintaining doctors' and patients' privacy. These results place SynDocDis as a promising framework for advancing medical AI research ethically and responsibly through privacy-compliant synthetic physician dialogue generation with direct applications in medical education and clinical decision support. 

---
# Revisiting Anisotropy in Language Transformers: The Geometry of Learning Dynamics 

**Authors**: Raphael Bernas, Fanny Jourdan, Antonin Poché, Céline Hudelot  

**Link**: [PDF](https://arxiv.org/pdf/2604.08764)  

**Abstract**: Since their introduction, Transformer architectures have dominated Natural Language Processing (NLP). However, recent research has highlighted an inherent anisotropy phenomenon in these models, presenting a significant challenge to their geometric interpretation. Previous theoretical studies on this phenomenon are rarely grounded in the underlying representation geometry. In this paper, we extend them by deriving geometric arguments for how frequency-biased sampling attenuates curvature visibility and why training preferentially amplify tangent directions. Empirically, we then use concept-based mechanistic interpretability during training, rather than only post hoc, to fit activation-derived low-rank tangent proxies and test them against ordinary backpropagated true gradients. Across encoder-style and decoder-style language models, we find that these activation-derived directions capture both unusually large gradient energy and a substantially larger share of gradient anisotropy than matched-rank normal controls, providing strong empirical support for a tangent-aligned account of anisotropy. 

---
# VisionFoundry: Teaching VLMs Visual Perception with Synthetic Images 

**Authors**: Guanyu Zhou, Yida Yin, Wenhao Chai, Shengbang Tong, Xingyu Fu, Zhuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09531)  

**Abstract**: Vision-language models (VLMs) still struggle with visual perception tasks such as spatial understanding and viewpoint recognition. One plausible contributing factor is that natural image datasets provide limited supervision for low-level visual skills. This motivates a practical question: can targeted synthetic supervision, generated from only a task keyword such as Depth Order, address these weaknesses? To investigate this question, we introduce VisionFoundry, a task-aware synthetic data generation pipeline that takes only the task name as input and uses large language models (LLMs) to generate questions, answers, and text-to-image (T2I) prompts, then synthesizes images with T2I models and verifies consistency with a proprietary VLM, requiring no reference images or human annotation. Using VisionFoundry, we construct VisionFoundry-10K, a synthetic visual question answering (VQA) dataset containing 10k image-question-answer triples spanning 10 tasks. Models trained on VisionFoundry-10K achieve substantial improvements on visual perception benchmarks: +7% on MMVP and +10% on CV-Bench-3D, while preserving broader capabilities and showing favorable scaling behavior as data size increases. Our results suggest that limited task-targeted supervision is an important contributor to this bottleneck and that synthetic supervision is a promising path toward more systematic training for VLMs. 

---
# VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning 

**Authors**: Wenyi Xiao, Xinchi Xu, Leilei Gan  

**Link**: [PDF](https://arxiv.org/pdf/2604.09529)  

**Abstract**: Large Vision Language Models (LVLMs) achieve strong multimodal reasoning but frequently exhibit hallucinations and incorrect responses with high certainty, which hinders their usage in high-stakes domains. Existing verbalized confidence calibration methods, largely developed for text-only LLMs, typically optimize a single holistic confidence score using binary answer-level correctness. This design is mismatched to LVLMs: an incorrect prediction may arise from perceptual failures or from reasoning errors given correct perception, and a single confidence conflates these sources while visual uncertainty is often dominated by language priors. To address these issues, we propose VL-Calibration, a reinforcement learning framework that explicitly decouples confidence into visual and reasoning confidence. To supervise visual confidence without ground-truth perception labels, we introduce an intrinsic visual certainty estimation that combines (i) visual grounding measured by KL-divergence under image perturbations and (ii) internal certainty measured by token entropy. We further propose token-level advantage reweighting to focus optimization on tokens based on visual certainty, suppressing ungrounded hallucinations while preserving valid perception. Experiments on thirteen benchmarks show that VL-Calibration effectively improves calibration while boosting visual reasoning accuracy, and it generalizes to out-of-distribution benchmarks across model scales and architectures. 

---
# Drift and selection in LLM text ecosystems 

**Authors**: Søren Riis  

**Link**: [PDF](https://arxiv.org/pdf/2604.08554)  

**Abstract**: The public text record -- the material from which both people and AI systems now learn -- is increasingly shaped by its own outputs. Generated text enters the public record, later agents learn from it, and the cycle repeats. Here we develop an exactly solvable mathematical framework for this recursive process, based on variable-order $n$-gram agents, and separate two forces acting on the public corpus. The first is drift: unfiltered reuse progressively removes rare forms, and in the infinite-corpus limit we characterise the stable distributions exactly. The second is selection: publication, ranking and verification filter what enters the record, and the outcome depends on what is selected. When publication merely reflects the statistical status quo, the corpus converges to a shallow state in which further lookahead brings no benefit. When publication is normative -- rewarding quality, correctness or novelty -- deeper structure persists, and we establish an optimal upper bound on the resulting divergence from shallow equilibria. The framework therefore identifies when recursive publication compresses public text and when selective filtering sustains richer structure, with implications for the design of AI training corpora. 

---
# Visually-Guided Policy Optimization for Multimodal Reasoning 

**Authors**: Zengbin Wang, Feng Xiong, Liang Lin, Xuecai Hu, Yong Wang, Yanlin Wang, Man Zhang, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09349)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has significantly advanced the reasoning ability of vision-language models (VLMs). However, the inherent text-dominated nature of VLMs often leads to insufficient visual faithfulness, characterized by sparse attention activation to visual tokens. More importantly, our empirical analysis reveals that temporal visual forgetting along reasoning steps exacerbates this deficiency. To bridge this gap, we propose Visually-Guided Policy Optimization (VGPO), a novel framework to reinforce visual focus during policy optimization. Specifically, VGPO initially introduces a Visual Attention Compensation mechanism that leverages visual similarity to localize and amplify visual cues, while progressively elevating visual expectations in later steps to counteract visual forgetting. Building on this mechanism, we implement a dual-grained advantage re-weighting strategy: the intra-trajectory level highlights tokens exhibiting relatively high visual activation, while the inter-trajectory level prioritizes trajectories demonstrating superior visual accumulation. Extensive experiments demonstrate that VGPO achieves better visual activation and superior performance in mathematical multimodal reasoning and visual-dependent tasks. 

---
# Beyond Relevance: Utility-Centric Retrieval in the LLM Era 

**Authors**: Hengran Zhang, Minghao Tang, Keping Bi, Jiafeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.08920)  

**Abstract**: Information retrieval systems have traditionally optimized for topical relevance-the degree to which retrieved documents match a query. However, relevance only approximates a deeper goal: utility, namely, whether retrieved information helps accomplish a user's underlying task. The emergence of retrieval-augmented generation (RAG) fundamentally changes this paradigm. Retrieved documents are no longer consumed directly by users but instead serve as evidence for large language models (LLMs) that produce answers. As a result, retrieval effectiveness must be evaluated by its contribution to generation quality rather than by relevance-based ranking metrics alone. This tutorial argues that retrieval objectives are evolving from relevance-centric optimization toward LLM-centric utility. We present a unified framework covering LLM-agnostic versus LLM-specific utility, context-independent versus context-dependent utility, and the connection with LLM information needs and agentic RAG. By synthesizing recent advances, the tutorial provides conceptual foundations and practical guidance for designing retrieval systems aligned with the requirements of LLM-based information access. 

---
# Revisiting the Capacity Gap in Chain-of-Thought Distillation from a Practical Perspective 

**Authors**: Tokio Kajitsuka, Ukyo Honda, Sho Takase  

**Link**: [PDF](https://arxiv.org/pdf/2604.08880)  

**Abstract**: Chain-of-thought (CoT) distillation transfers reasoning behaviors from a strong teacher to a smaller student, but prior work reports a capacity gap: distillation may fail when the teacher-student capability mismatch is large. We revisit the capacity gap from a practical perspective by re-examining commonly used experimental settings. Notably, we find that CoT distillation often degrades performance compared to the student's pre-distillation baseline, an issue obscured when only post-distillation comparisons are reported. We therefore propose a more realistic evaluation protocol and find that the impact of capacity gap effects does not consistently dominate across tasks and settings, especially when candidate teachers differ substantially in performance. Our results offer practical guidance for selecting teacher-student pairs in CoT distillation. 

---
# SiMing-Bench: Evaluating Procedural Correctness from Continuous Interactions in Clinical Skill Videos 

**Authors**: Xiyang Huang, Jiawei Lin, Keying Wu, Jiaxin Huang, Kailai Yang, Renxiong Wei, Cheng zeng, Jiayi Xiang, Ziyan Kuang, Min Peng, Qianqian Xie, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2604.09037)  

**Abstract**: Current video benchmarks for multimodal large language models (MLLMs) focus on event recognition, temporal ordering, and long-context recall, but overlook a harder capability required for expert procedural judgment: tracking how ongoing interactions update the procedural state and thereby determine the correctness of later actions. We introduce SiMing-Bench, the first benchmark for evaluating this capability from full-length clinical skill videos. It targets rubric-grounded process-level judgment of whether interaction-driven state updates preserve procedural correctness across an entire workflow. SiMing-Bench is instantiated with SiMing-Score, a physician-annotated dataset of real clinical skill examination videos spanning cardiopulmonary resuscitation, automated external defibrillator operation, and bag-mask ventilation, each paired with a standardized step-wise rubric and dual-expert labels. Across diverse open- and closed-source MLLMs, we observe consistently weak agreement with physician judgments. Moreover, weak performance on rubric-defined intermediate steps persists even when overall procedure-level correlation appears acceptable, suggesting that coarse global assessment substantially overestimates current models' procedural judgment ability. Additional analyses with binary step judgment and step-aligned clips indicate that the bottleneck is not merely fine-grained scoring or temporal localization, but modeling how continuous interactions update procedural state over time. 

---
# Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs 

**Authors**: Jinqi Luo, Jinyu Yang, Tal Neiman, Lei Fan, Bing Yin, Son Tran, Mubarak Shah, René Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2604.08846)  

**Abstract**: Multimodal Large Language Models (MLLMs) have been shown to be vulnerable to malicious queries that can elicit unsafe responses. Recent work uses prompt engineering, response classification, or finetuning to improve MLLM safety. Nevertheless, such approaches are often ineffective against evolving malicious patterns, may require rerunning the query, or demand heavy computational resources. Steering the activations of a frozen model at inference time has recently emerged as a flexible and effective solution. However, existing steering methods for MLLMs typically handle only a narrow set of safety-related concepts or struggle to adjust specific concepts without affecting others. To address these challenges, we introduce Dictionary-Aligned Concept Control (DACO), a framework that utilizes a curated concept dictionary and a Sparse Autoencoder (SAE) to provide granular control over MLLM activations. First, we curate a dictionary of 15,000 multimodal concepts by retrieving over 400,000 caption-image stimuli and summarizing their activations into concept directions. We name the dataset DACO-400K. Second, we show that the curated dictionary can be used to intervene activations via sparse coding. Third, we propose a new steering approach that uses our dictionary to initialize the training of an SAE and automatically annotate the semantics of the SAE atoms for safeguarding MLLMs. Experiments on multiple MLLMs (e.g., QwenVL, LLaVA, InternVL) across safety benchmarks (e.g., MM-SafetyBench, JailBreakV) show that DACO significantly improves MLLM safety while maintaining general-purpose capabilities. 

---
# $p1$: Better Prompt Optimization with Fewer Prompts 

**Authors**: Zhaolin Gao, Wang, Bo Liu, Thorsten Joachims, Kianté Brantley, Wen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.08801)  

**Abstract**: Prompt optimization improves language models without updating their weights by searching for a better system prompt, but its effectiveness varies widely across tasks. We study what makes a task amenable to prompt optimization. We show that the reward variance across different system prompts can be decomposed into two components: variance among responses, which captures generation stochasticity, and variance among system prompts, which captures differences in system prompt quality. Prompt optimization succeeds when variance among system prompts is sufficiently large, but fails when variance among responses dominates the variance of the system prompts. Surprisingly, we further show that scaling to more user prompts can hurt optimization by reducing variance among system prompts, especially on heterogeneous datasets where different user prompts favor different system prompts. Motivated by this insight, we propose $p1$, a simple user prompt filtering method that selects a small subset of user prompts with high variance across candidate system prompts. This subset of user prompts allows one to distinguish a good system prompt from a bad one, making system optimization easier. Experiments on reasoning benchmarks show that $p1$ substantially improves prompt optimization over training on the full dataset and outperforms strong baselines such as GEPA. Notably, training on only two prompts from AIME 24 yields a system prompt that generalizes well to other reasoning benchmarks. 

---
# Mind the Gap Between Spatial Reasoning and Acting! Step-by-Step Evaluation of Agents With Spatial-Gym 

**Authors**: Lars Benedikt Kaesberg, Tianyu Yang, Niklas Bauer, Terry Ruas, Jan Philip Wahle, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2604.09338)  

**Abstract**: Spatial reasoning is central to navigation and robotics, yet measuring model capabilities on these tasks remains difficult. Existing benchmarks evaluate models in a one-shot setting, requiring full solution generation in a single response, unlike humans, who work in interactive environments step-by-step. We introduce Spatial-Gym, a Gymnasium environment that isolates spatial constraint reasoning by testing pathfinding in 2D-grid puzzles as a sequential decision task with optional backtracking. We evaluate eight models in three settings (one-shot, step-by-step, step-by-step with backtracking) against human, random, and A* baselines on 500 episodes. The best model, GPT-OSS 120B, achieves a solve rate of 16.0%, 82 points below the human baseline (98.0%). Step-by-step format helps weaker models (up to +5.4%) by removing formatting errors, but hurts stronger models (up to 5.6%) by constraining global planning. Backtracking improves episode completion, but increases solve rate only for weaker models; stronger models rarely backtrack and do not benefit from it. Our experiments have three key findings: (1) models fail to scale reasoning effort with difficulty, (2) vision models receiving images of the spatial environment reduce solve rate by 73%, and (3) extended chain-of-thought reasoning retains a 3-5x accuracy advantage over standard inference even in the step-by-step setting. Spatial-Gym enables diagnosis of model limitations and provides a framework for improving spatial reasoning through reinforcement learning. 

---
# Is More Data Worth the Cost? Dataset Scaling Laws in a Tiny Attention-Only Decoder 

**Authors**: Götz-Henrik Wiegand, Lorena Raichle, Rico Städeli, Tomas Hrycej, Bernhard Bermeitinger, Siegfried Handschuh  

**Link**: [PDF](https://arxiv.org/pdf/2604.09389)  

**Abstract**: Training Transformer language models is expensive, as performance typically improves with increasing dataset size and computational budget. Although scaling laws describe this trend at large scale, their implications in controlled, smaller-scale settings remain less explored. In this work, we isolate dataset-size effects using a strongly reduced attention-only decoder architecture. By training on progressively larger power-of-two subsets, we observe smooth performance improvements accompanied by clear diminishing returns, consistent with scaling-law behavior. Using only about 30% of the training data is sufficient to reach approximately 90% of the full-data validation token-level accuracy. These results provide actionable insights into dataset scaling in a controlled, component-isolated setting and offer practical guidance for balancing dataset size and computational cost in compute- and data-restricted environments, such as small research labs and exploratory model development. 

---
# EMA Is Not All You Need: Mapping the Boundary Between Structure and Content in Recurrent Context 

**Authors**: Arth Singh  

**Link**: [PDF](https://arxiv.org/pdf/2604.08556)  

**Abstract**: What exactly do efficient sequence models gain over simple temporal averaging? We use exponential moving average (EMA) traces, the simplest recurrent context (no gating, no content-based retrieval), as a controlled probe to map the boundary between what fixed-coefficient accumulation can and cannot represent. EMA traces encode temporal structure: a Hebbian architecture with multi-timescale traces achieves 96% of a supervised BiGRU on grammatical role assignment with zero labels, surpassing the supervised model on structure-dependent roles. EMA traces destroy token identity: a 130M-parameter language model using only EMA context reaches C4 perplexity 260 (8x GPT-2), and a predictor ablation (replacing the linear predictor with full softmax attention) yields identical loss, localizing the entire gap to the traces. The traces apply lossy, data-independent compression; by the data processing inequality, no downstream predictor can recover the discarded information. Fixed-coefficient accumulation, whether across time or depth, suffers irreversible information dilution that only learned, input-dependent selection can resolve. 

---
# Arbitration Failure, Not Perceptual Blindness: How Vision-Language Models Resolve Visual-Linguistic Conflicts 

**Authors**: Farhad Nooralahzadeh, Omid Rohanian, Yi Zhang, Jonathan Fürst, Kurt Stockinger  

**Link**: [PDF](https://arxiv.org/pdf/2604.09364)  

**Abstract**: When a Vision-Language Model (VLM) sees a blue banana and answers "yellow", is the problem of perception or arbitration? We explore the question in ten VLMs with various sizes and reveal an Encoding--Grounding Dissociation: models that fail to report what they see (and thus provide a wrong answer) still encode the visual evidence as strongly as models that provide the correct answer. Using Multimodal Arbitration Crossover (MAC) analysis with layer-by-layer Logit Lens probing, we track the competition between visual and prior signals across every layer of each model. We show that visual attributes can be linearly decodable from early layers (AUC > 0.86). The accuracy remains nearly identical for both successful and failed samples. However, the gap in the final-layer logit -- not the strength of encoding -- better predicts grounding outcomes with a correlation of . After having studied when VLMs base their answers on image clues rather than prior knowledge, we want to understand the causal relationships. We establish causality through full-sequence activation patching. The standard last-token interventions in LLM interpretability do not affect VLMs. In contrast, replacing the full token sequence at layers identified by MAC alters 60 to 84% of outputs. Partial-token decomposition shows that image tokens carry almost all of the causal impact, while text tokens have none. Scaling addresses the remaining architectural differences to achieve perfect retention. Moving from diagnosis to intervention, we show that training-free activation steering -- both linear and sparse autoencoder-guided -- in early layers can improve visual grounding by up to +3.8% with degrading performance in some setups. Overall, these findings lead to a clear conclusion: VLMs already see well, but the challenge is acting on what they see. Targeted interventions can help to bridge this gap. 

---
# Every Response Counts: Quantifying Uncertainty of LLM-based Multi-Agent Systems through Tensor Decomposition 

**Authors**: Tiejin Chen, Huaiyuan Yao, Jia Chen, Evangelos E. Papalexakis, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.08708)  

**Abstract**: While Large Language Model-based Multi-Agent Systems (MAS) consistently outperform single-agent systems on complex tasks, their intricate interactions introduce critical reliability challenges arising from communication dynamics and role dependencies. Existing Uncertainty Quantification methods, typically designed for single-turn outputs, fail to address the unique complexities of the MAS. Specifically, these methods struggle with three distinct challenges: the cascading uncertainty in multi-step reasoning, the variability of inter-agent communication paths, and the diversity of communication topologies. To bridge this gap, we introduce MATU, a novel framework that quantifies uncertainty through tensor decomposition. MATU moves beyond analyzing final text outputs by representing entire reasoning trajectories as embedding matrices and organizing multiple execution runs into a higher-order tensor. By applying tensor decomposition, we disentangle and quantify distinct sources of uncertainty, offering a comprehensive reliability measure that is generalizable across different agent structures. We provide comprehensive experiments to show that MATU effectively estimates holistic and robust uncertainty across diverse tasks and communication topologies. 

---
# Skip-Connected Policy Optimization for Implicit Advantage 

**Authors**: Fengwei Teng, Jinyi Bai, Xinhao Yao, Demi Ruohan Wang, Jiahao Zhao, Zhijiang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.08690)  

**Abstract**: Group Relative Policy Optimization (GRPO) has proven effective in RLVR by using outcome-based rewards. While fine-grained dense rewards can theoretically improve performance, we reveal that under practical sampling budgets, Monte Carlo estimation yields high-variance and sign-inconsistent advantages for early reasoning tokens, paradoxically underperforming outcome-only GRPO. We propose Skip-Connected Optimization (SKPO), which decomposes reasoning into upstream and downstream phases: upstream receives dense rewards from downstream Monte Carlo sampling with single-stream optimization; downstream maintains group-relative optimization, where a skip connection concatenates the upstream segment with the original problem, enabling the model to leverage helpful upstream reasoning while preserving the freedom to bypass flawed reasoning through direct problem access. Experiments demonstrate improvements of 3.91% and 6.17% relative gains over the strongest baselines on Qwen2.5-Math-7B and Llama-3.2-3B respectively across mathematical benchmarks and out-of-domain tasks including general reasoning and code generation. Further analysis reveals an implicit advantage: SKPO generates trajectories with higher intermediate-step quality even when matched for final correctness. 

---
# HiFloat4 Format for Language Model Pre-training on Ascend NPUs 

**Authors**: Mehran Taghian, Yunke Peng, Xing Huang, Yao Wang, Yaoyuan Wang, Wei Guo, Yuanyong Luo, Tianchi Hu, Junsong Wang, Xin Wang, Hu Liu, Yu Cheng, Ziwei Yu, Hongliang Li, Mehdi Rahimifar, Lei Yan, Xuefei Wang, Zhuang Ma, Lei Liu, Hui Yu, Anandharaju Durai Raju, Hoang Le, Hei Yi Mak, Tanzila Rahman, Shadan Golestan  

**Link**: [PDF](https://arxiv.org/pdf/2604.08826)  

**Abstract**: Large foundation models have become central to modern machine learning, with performance scaling predictably with model size and data. However, training and deploying such models incur substantial computational and memory costs, motivating the development of low-precision training techniques. Recent work has demonstrated that 4-bit floating-point (FP4) formats--such as MXFP4 and NVFP4--can be successfully applied to linear GEMM operations in large language models (LLMs), achieving up to 4x improvements in compute throughput and memory efficiency compared to higher-precision baselines. In this work, we investigate the recently proposed HiFloat4 FP4 format for Huawei Ascend NPUs and systematically compare it with MXFP4 in large-scale training settings. All experiments are conducted on Ascend NPU clusters, with linear and expert GEMM operations performed entirely in FP4 precision. We evaluate both dense architectures (e.g., Pangu and LLaMA-style models) and mixture-of-experts (MoE) models, where both standard linear layers and expert-specific GEMMs operate in FP4. Furthermore, we explore stabilization techniques tailored to FP4 training that significantly reduce numerical degradation, maintaining relative error within 1% of full-precision baselines while preserving the efficiency benefits of 4-bit computation. Our results provide a comprehensive empirical study of FP4 training on NPUs and highlight the practical trade-offs between FP4 formats in large-scale dense and MoE models. 

---
# Robust Reasoning Benchmark 

**Authors**: Pavel Golikov, Evgenii Opryshko, Gennady Pekhimenko, Mark C. Jeffrey  

**Link**: [PDF](https://arxiv.org/pdf/2604.08571)  

**Abstract**: While Large Language Models (LLMs) achieve high performance on standard mathematical benchmarks, their underlying reasoning processes remain highly overfit to standard textual formatting. We propose a perturbation pipeline consisting of 14 techniques to evaluate robustness of LLM reasoning. We apply this pipeline to AIME 2024 dataset and evalute 8 state-of-the-art models on the resulting benchmark. While frontier models exhibit resilience, open weights reasoning models suffer catastrophic collapses (up to 55% average accuracy drops across perturbations and up to 100% on some), exposing structural fragility. To further disentangle mechanical parsing failures from downstream reasoning failures, we strictly isolate the models' working memory capacity by forcing models to solve multiple unperturbed mathematical problems sequentially within a single context window. Our results indicate that open weight models ranging from 7B to 120B parameters and Claude Opus 4.6 exhibit accuracy decay on subsequent problems. This degradation demonstrates that intermediate reasoning steps permanently pollute standard dense attention mechanisms. We argue that to achieve reliable reasoning, future reasoning architectures must integrate explicit contextual resets within a model's own Chain-of-Thought, leading to fundamental open questions regarding the optimal granularity of atomic reasoning tasks. 

---
# GNN-as-Judge: Unleashing the Power of LLMs for Graph Learning with GNN Feedback 

**Authors**: Ruiyao Xu, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.08553)  

**Abstract**: Large Language Models (LLMs) have shown strong performance on text-attributed graphs (TAGs) due to their superior semantic understanding ability on textual node features. However, their effectiveness as predictors in the low-resource setting, where labeled nodes are severely limited and scarce, remains constrained since fine-tuning LLMs usually requires sufficient labeled data, especially when the TAG shows complex structural patterns. In essence, this paper targets two key challenges: (i) the difficulty of generating and selecting reliable pseudo labels on TAGs for LLMs, and (ii) the need to mitigate potential label noise when fine-tuning LLMs with pseudo labels. To counter the challenges, we propose a new framework, GNN-as-Judge, which can unleash the power of LLMs for few-shot semi-supervised learning on TAGs by incorporating the structural inductive bias of Graph Neural Networks (GNNs). Specifically, GNN-as-Judge introduces a collaborative pseudo-labeling strategy that first identifies the most influenced unlabeled nodes from labeled nodes, then exploits both the agreement and disagreement patterns between LLMs and GNNs to generate reliable labels. Furthermore, we develop a weakly-supervised LLM fine-tuning algorithm that can distill the knowledge from informative pseudo labels while mitigating the potential label noise. Experiments on multiple TAG datasets demonstrate that GNN-as-Judge significantly outperforms existing methods, particularly in low-resource regimes where labeled data are scarce. 

---
# From Business Events to Auditable Decisions: Ontology-Governed Graph Simulation for Enterprise AI 

**Authors**: Hongyin Zhu, Jinming Liang, Mengjun Hou, Ruifan Tang, Xianbin Zhu, Jingyuan Yang, Yuanman Mao, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08603)  

**Abstract**: Existing LLM-based agent systems share a common architectural failure: they answer from the unrestricted knowledge space without first simulating how active business scenarios reshape that space for the event at hand -- producing decisions that are fluent but ungrounded and carrying no audit trail. We present LOM-action, which equips enterprise AI with \emph{event-driven ontology simulation}: business events trigger scenario conditions encoded in the enterprise ontology~(EO), which drive deterministic graph mutations in an isolated sandbox, evolving a working copy of the subgraph into the scenario-valid simulation graph $G_{\text{sim}}$; all decisions are derived exclusively from this evolved graph. The core pipeline is \emph{event $\to$ simulation $\to$ decision}, realized through a dual-mode architecture -- \emph{skill mode} and \emph{reasoning mode}. Every decision produces a fully traceable audit log. LOM-action achieves 93.82% accuracy and 98.74% tool-chain F1 against frontier baselines Doubao-1.8 and DeepSeek-V3.2, which reach only 24--36% F1 despite 80% accuracy -- exposing the \emph{illusive accuracy} phenomenon. The four-fold F1 advantage confirms that ontology-governed, event-driven simulation, not model scale, is the architectural prerequisite for trustworthy enterprise decision intelligence. 

---
# VerifAI: A Verifiable Open-Source Search Engine for Biomedical Question Answering 

**Authors**: Miloš Košprdić, Adela Ljajić, Bojana Bašaragin, Darija Medvecki, Lorenzo Cassano, Nikola Milošević  

**Link**: [PDF](https://arxiv.org/pdf/2604.08549)  

**Abstract**: We introduce VerifAI, an open-source expert system for biomedical question answering that integrates retrieval-augmented generation (RAG) with a novel post-hoc claim verification mechanism. Unlike standard RAG systems, VerifAI ensures factual consistency by decomposing generated answers into atomic claims and validating them against retrieved evidence using a fine-tuned natural language inference (NLI) engine. The system comprises three modular components: (1) a hybrid Information Retrieval (IR) module optimized for biomedical queries (MAP@10 of 42.7%), (2) a citation-aware Generative Component fine-tuned on a custom dataset to produce referenced answers, and (3) a Verification Component that detects hallucinations with state-of-the-art accuracy, outperforming GPT-4 on the HealthVer benchmark. Evaluations demonstrate that VerifAI significantly reduces hallucinated citations compared to zero-shot baselines and provides a transparent, verifiable lineage for every claim. The full pipeline, including code, models, and datasets, is open-sourced to facilitate reliable AI deployment in high-stakes domains. 

---
# SUPERNOVA: Eliciting General Reasoning in LLMs with Reinforcement Learning on Natural Instructions 

**Authors**: Ashima Suvarna, Kendrick Phan, Mehrab Beikzadeh, Hritik Bansal, Saadia Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2604.08477)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has significantly improved large language model (LLM) reasoning in formal domains such as mathematics and code. Despite these advancements, LLMs still struggle with general reasoning tasks requiring capabilities such as causal inference and temporal understanding. Extending RLVR to general reasoning is fundamentally constrained by the lack of high-quality, verifiable training data that spans diverse reasoning skills. To address this challenge, we propose SUPERNOVA, a data curation framework for RLVR aimed at enhancing general reasoning. Our key insight is that instruction-tuning datasets containing expert-annotated ground-truth encode rich reasoning patterns that can be systematically adapted for RLVR. To study this, we conduct 100+ controlled RL experiments to analyze how data design choices impact downstream reasoning performance. In particular, we investigate three key factors: (i) source task selection, (ii) task mixing strategies, and (iii) synthetic interventions for improving data quality. Our analysis reveals that source task selection is non-trivial and has a significant impact on downstream reasoning performance. Moreover, selecting tasks based on their performance for individual target tasks outperforms strategies based on overall average performance. Finally, models trained on SUPERNOVA outperform strong baselines (e.g., Qwen3.5) on challenging reasoning benchmarks including BBEH, Zebralogic, and MMLU-Pro. In particular, training on SUPERNOVA yields relative improvements of up to 52.8\% on BBEH across model sizes, demonstrating the effectiveness of principled data curation for RLVR. Our findings provide practical insights for curating human-annotated resources to extend RLVR to general reasoning. The code and data is available at this https URL. 

---
# Re-Mask and Redirect: Exploiting Denoising Irreversibility in Diffusion Language Models 

**Authors**: Arth Singh  

**Link**: [PDF](https://arxiv.org/pdf/2604.08557)  

**Abstract**: Diffusion-based language models (dLLMs) generate text by iteratively denoising masked token sequences. We show that their safety alignment rests on a single fragile assumption: that the denoising schedule is monotonic and committed tokens are never re-evaluated. Safety-aligned dLLMs commit refusal tokens within the first 8-16 of 64 denoising steps, and the schedule treats these commitments as permanent. A trivial two-step intervention - re-masking these tokens and injecting a 12-token affirmative prefix - achieves 76.1% ASR on HarmBench (n=159, Lg=128) against LLaDA-8B-Instruct and 81.8% ASR (n=159) against Dream-7B-Instruct, without any gradient computation or adversarial search. The simplicity of this exploit is itself the central finding: augmenting with gradient-optimized perturbation via a differentiable Gumbel-softmax chain consistently degrades ASR (e.g., 41.5% vs. 76.1% at Lg=128), confirming that the vulnerability is structural rather than requiring sophisticated exploitation. These findings reveal that dLLM safety is not adversarially robust but architecturally shallow - it holds only because the denoising schedule is never violated. We discuss defenses including safety-aware unmasking schedules, step-conditional prefix detection, and post-commitment re-verification. 

---
# Optimal Multi-bit Generative Watermarking Schemes Under Worst-Case False-Alarm Constraints 

**Authors**: Yu-Shin Huang, Chao Tian, Krishna Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2604.08759)  

**Abstract**: This paper considers the problem of multi-bit generative watermarking for large language models under a worst-case false-alarm constraint. Prior work established a lower bound on the achievable miss-detection probability in the finite-token regime and proposed a scheme claimed to achieve this bound. We show, however, that the proposed scheme is in fact suboptimal. We then develop two new encoding-decoding constructions that attain the previously established lower bound, thereby completely characterizing the optimal multi-bit watermarking performance. Our approach formulates the watermark design problem as a linear program and derives the structural conditions under which optimality can be achieved. In addition, we identify the failure mechanism of the previous construction and compare the tradeoffs between the two proposed schemes. 

---
# Dynamic Ranked List Truncation for Reranking Pipelines via LLM-generated Reference-Documents 

**Authors**: Nilanjan Sinhababu, Soumedhik Bharati, Debasis Ganguly, Pabitra Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2604.09492)  

**Abstract**: Large Language Models (LLM) have been widely used in reranking. Computational overhead and large context lengths remain a challenging issue for LLM rerankers. Efficient reranking usually involves selecting a subset of the ranked list from the first stage, known as ranked list truncation (RLT). The truncated list is processed further by a reranker. For LLM rerankers, the ranked list is often partitioned and processed sequentially in batches to reduce the context length. Both these steps involve hyperparameters and topic-agnostic heuristics. Recently, LLMs have been shown to be effective for relevance judgment. Equivalently, we propose that LLMs can be used to generate reference documents that can act as a pivot between relevant and non-relevant documents in a ranked list. We propose methods to use these generated reference documents for RLT as well as for efficient listwise reranking. While reranking, we process the ranked list in either parallel batches of non-overlapping windows or overlapping windows with adaptive strides, improving the existing fixed stride setup. The generated reference documents are also shown to improve existing efficient listwise reranking frameworks. Experiments on TREC Deep Learning benchmarks show that our approach outperforms existing RLT-based approaches. In-domain and out-of-domain benchmarks demonstrate that our proposed methods accelerate LLM-based listwise reranking by up to 66\% compared to existing approaches. This work not only establishes a practical paradigm for efficient LLM-based reranking but also provides insight into the capability of LLMs to generate semantically controlled documents using relevance signals. 

---
# On the Representational Limits of Quantum-Inspired 1024-D Document Embeddings: An Experimental Evaluation Framework 

**Authors**: Dario Maio  

**Link**: [PDF](https://arxiv.org/pdf/2604.09430)  

**Abstract**: Text embeddings are central to modern information retrieval and Retrieval-Augmented Generation (RAG). While dense models derived from Large Language Models (LLMs) dominate current practice, recent work has explored quantum-inspired alternatives motivated by the geometric properties of Hilbert-like spaces and their potential to encode richer semantic structure.
This paper presents an experimental framework for constructing quantum-inspired 1024-dimensional document embeddings based on overlapping windows and multi-scale aggregation. The pipeline combines semantic projections (e.g., EigAngle), circuit-inspired feature mappings, and optional teacher-student distillation, together with a fingerprinting mechanism for reproducibility and controlled evaluation.
We introduce a set of diagnostic tools for hybrid retrieval, including static and dynamic interpolation between BM25 and embedding-based scores, candidate union strategies, and a conceptual alpha-oracle that provides an upper bound for score-level fusion.
Experiments on controlled corpora of Italian and English documents across technical, narrative, and legal domains, using synthetic queries, show that BM25 remains a strong baseline, teacher embeddings provide stable semantic structure, and standalone quantum-inspired embeddings exhibit weak and unstable ranking signals. Distillation yields mixed effects, improving alignment in some cases but not consistently enhancing retrieval performance, while hybrid retrieval can recover competitive results when lexical and embedding-based signals are combined.
Overall, the results highlight structural limitations in the geometry of quantum-inspired embeddings, including distance compression and ranking instability, and clarify their role as auxiliary components rather than standalone retrieval representations. 

---
# SkillForge: Forging Domain-Specific, Self-Evolving Agent Skills in Cloud Technical Support 

**Authors**: Xingyan Liu, Xiyue Luo, Linyu Li, Ganghong Huang, Jianfeng Liu, Honglin Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.08618)  

**Abstract**: Deploying LLM-powered agents in enterprise scenarios such as cloud technical support demands high-quality, domain-specific skills. However, existing skill creators lack domain grounding, producing skills poorly aligned with real-world task requirements. Moreover, once deployed, there is no systematic mechanism to trace execution failures back to skill deficiencies and drive targeted refinements, leaving skill quality stagnant despite accumulating operational evidence. We introduce SkillForge, a self-evolving framework that closes an end-to-end creation-evaluation-refinement loop. To produce well-aligned initial skills, a Domain-Contextualized Skill Creator grounds skill synthesis in knowledge bases and historical support tickets. To enable continuous self-optimization, a three-stage pipeline -- Failure Analyzer, Skill Diagnostician, and Skill Optimizer -- automatically diagnoses execution failures in batch, pinpoints the underlying skill deficiencies, and rewrites the skill to eliminate them. This cycle runs iteratively, allowing skills to self-improve with every round of deployment feedback. Evaluated on five real-world cloud support scenarios spanning 1,883 tickets and 3,737 tasks, experiments show that: (1) the Domain-Contextualized Skill Creator produces substantially better initial skills than the generic skill creator, as measured by consistency with expert-authored reference responses from historical tickets; and (2) the self-evolution loop progressively improves skill quality from diverse starting points (including expert-authored, domain-created, and generic skills) across successive rounds, demonstrating that automated evolution can surpass manually curated expert knowledge. 

---
# BracketRank: Large Language Model Document Ranking via Reasoning-based Competitive Elimination 

**Authors**: Abdelrahman Abdallah, Mohammed Ali, Bhawna Piryani, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2604.08834)  

**Abstract**: Reasoning-intensive retrieval requires deep semantic inference beyond surface-level keyword matching, posing a challenge for current LLM-based rerankers limited by context constraints and order sensitivity. We propose \textbf{\BracketRank}, a framework that treats document reranking as a reasoning-driven competitive tournament. Our approach introduces three key innovations: (1) adaptive grouping based on model context limits, (2) reasoning-enhanced prompts that mandate step-by-step relevance explanations, and (3) a bracket-style elimination structure with winner and loser tracks. This design ensures robust document advancement while enabling parallel processing across competition stages. Evaluation on the BRIGHT reasoning benchmark shows that \BracketRank achieves \textbf{26.56 nDCG@10}, significantly outperforming state-of-the-art baselines including RankGPT-4 (17.0) and Rank-R1-14B (20.5). On TREC datasets, BracketRank achieves 77.90 nDCG@5 on DL 19 and 75.85 nDCG@5 on DL 20, exceeding all baselines, establishing that explicit reasoning within competitive elimination is a powerful paradigm for complex, multi-step retrieval tasks. this https URL 

---
# Trans-RAG: Query-Centric Vector Transformation for Secure Cross-Organizational Retrieval 

**Authors**: Yu Liu, Kun Peng, Wenxiao Zhang, Fangfang Yuan, Cong Cao, Wenxuan Lu, Yanbing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09541)  

**Abstract**: Retrieval Augmented Generation (RAG) systems deployed across organizational boundaries face fundamental tensions between security, accuracy, and efficiency. Current encryption methods expose plaintext during decryption, while federated architectures prevent resource integration and incur substantial overhead. We introduce Trans-RAG, implementing a novel vector space language paradigm where each organization's knowledge exists in a mathematically isolated semantic space. At the core lies vector2Trans, a multi-stage transformation technique that enables queries to dynamically "speak" each organization's vector space "language" through query-centric transformations, eliminating decryption overhead while maintaining native retrieval efficiency. Security evaluations demonstrate near-orthogonal vector spaces with 89.90° angular separation and 99.81% isolation rates. Experiments across 8 retrievers, 3 datasets, and 3 LLMs show minimal accuracy degradation (3.5% decrease in nDCG@10) and significant efficiency improvements over homomorphic encryption. 

---
# FashionStylist: An Expert Knowledge-enhanced Multimodal Dataset for Fashion Understanding 

**Authors**: Kaidong Feng, Zhuoxuan Huang, Huizhong Guo, Yuting Jin, Xinyu Chen, Yue Liang, Yifei Gai, Li Zhou, Yunshan Ma, Zhu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.09249)  

**Abstract**: Fashion understanding requires both visual perception and expert-level reasoning about style, occasion, compatibility, and outfit rationale. However, existing fashion datasets remain fragmented and task-specific, often focusing on item attributes, outfit co-occurrence, or weak textual supervision, and thus provide limited support for holistic outfit understanding. In this paper, we introduce FashionStylist, an expert-annotated benchmark for holistic and expert-level fashion understanding. Constructed through a dedicated fashion-expert annotation pipeline, FashionStylist provides professionally grounded annotations at both the item and outfit levels. It supports three representative tasks: outfit-to-item grounding, outfit completion, and outfit evaluation. These tasks cover realistic item recovery from complex outfits with layering and accessories, compatibility-aware composition beyond co-occurrence matching, and expert-level assessment of style, season, occasion, and overall coherence. Experimental results show that FashionStylist serves not only as a unified benchmark for multiple fashion tasks, but also as an effective training resource for improving grounding, completion, and outfit-level semantic evaluation in MLLM-based fashion systems. 

---
# Case-Grounded Evidence Verification: A Framework for Constructing Evidence-Sensitive Supervision 

**Authors**: Soroosh Tayebi Arasteh, Mehdi Joodaki, Mahshad Lotfinia, Sven Nebelung, Daniel Truhn  

**Link**: [PDF](https://arxiv.org/pdf/2604.09537)  

**Abstract**: Evidence-grounded reasoning requires more than attaching retrieved text to a prediction: a model should make decisions that depend on whether the provided evidence supports the target claim. In practice, this often fails because supervision is weak, evidence is only loosely tied to the claim, and evaluation does not test evidence dependence directly. We introduce case-grounded evidence verification, a general framework in which a model receives a local case context, external evidence, and a structured claim, and must decide whether the evidence supports the claim for that case. Our key contribution is a supervision construction procedure that generates explicit support examples together with semantically controlled non-support examples, including counterfactual wrong-state and topic-related negatives, without manual evidence annotation. We instantiate the framework in radiology and train a standard verifier on the resulting support task. The learned verifier substantially outperforms both case-only and evidence-only baselines, remains strong under correct evidence, and collapses when evidence is removed or swapped, indicating genuine evidence dependence. This behavior transfers across unseen evidence articles and an external case distribution, though performance degrades under evidence-source shift and remains sensitive to backbone choice. Overall, the results suggest that a major bottleneck in evidence grounding is not only model capacity, but the lack of supervision that encodes the causal role of evidence. 

---
# Pretrain-then-Adapt: Uncertainty-Aware Test-Time Adaptation for Text-based Person Search 

**Authors**: Jiahao Zhang, Shaofei Huang, Yaxiong Wang, Zhedong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.08598)  

**Abstract**: Text-based person search faces inherent limitations due to data scarcity, driven by stringent privacy constraints and the high cost of manual annotation. To mitigate this, existing methods usually rely on a Pretrain-then-Finetune paradigm, where models are first pretrained on synthetic person-caption data to establish cross-modal alignment, followed by fine-tuning on labeled real-world datasets. However, this paradigm lacks practicality in real-world deployment scenarios, where large-scale annotated target-domain data is typically inaccessible. In this work, we propose a new Pretrain-then-Adapt paradigm that eliminates reliance on extensive target-domain supervision through an offline test-time adaptation manner, enabling dynamic model adaptation using only unlabeled test data with minimal post-train time cost. To mitigate overconfidence with false positives of previous entropy-based test-time adaptation, we propose an Uncertainty-Aware Test-Time Adaptation (UATTA) framework, which introduces a bidirectional retrieval disagreement mechanism to estimate uncertainty, i.e., low uncertainty is assigned when an image-text pair ranks highly in both image-to-text and text-to-image retrieval, indicating high alignment; otherwise, high uncertainty is detected. This indicator drives offline test-time model recalibration without labels, effectively mitigating domain shift. We validate UATTA on four benchmarks, i.e., CUHK-PEDES, ICFG-PEDES, RSTPReid, and PAB, showing consistent improvements across both CLIP-based (one-stage) and XVLM-based (two-stage) frameworks. Ablation studies confirm that UATTA outperforms existing offline test-time adaptation strategies, establishing a new benchmark for label-efficient, deployable person search systems. Our code is available at this https URL. 

---
# Retrieval Augmented Classification for Confidential Documents 

**Authors**: Yeseul E. Chang, Rahul Kailasa, Simon Shim, Byunghoon Oh, Jaewoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.08628)  

**Abstract**: Unauthorized disclosure of confidential documents demands robust, low-leakage classification. In real work environments, there is a lot of inflow and outflow of documents. To continuously update knowledge, we propose a methodology for classifying confidential documents using Retrieval Augmented Classification (RAC). To confirm this effectiveness, we compare RAC and supervised fine tuning (FT) on the WikiLeaks US Diplomacy corpus under realistic sequence-length constraints. On balanced data, RAC matches FT. On unbalanced data, RAC is more stable while delivering comparable performance--about 96% Accuracy on both the original (unbalanced) and augmented (balanced) sets, and up to 94% F1 with proper prompting--whereas FT attains 90% F1 trained on the augmented, balanced set but drops to 88% F1 trained on the original, unbalanced set. When robust augmentation is infeasible, RAC provides a practical, security-preserving path to strong classification by keeping sensitive content out of model weights and under your control, and it remains robust as real-world conditions change in class balance, data, context length, or governance requirements. Because RAC grounds decisions in an external vector store with similarity matching, it is less sensitive to label skew, reduces parameter-level leakage, and can incorporate new data immediately via reindexing--a difficult step for FT, which typically requires retraining. The contributions of this paper are threefold: first, a RAC-based classification pipeline and evaluation recipe; second, a controlled study that isolates class imbalance and context-length effects for FT versus RAC in confidential-document grading; and third, actionable guidance on RAC design patterns for governed deployments. 

---
# Strategic Algorithmic Monoculture:Experimental Evidence from Coordination Games 

**Authors**: Gonzalo Ballestero, Hadi Hosseini, Samarth Khanna, Ran I. Shorrer  

**Link**: [PDF](https://arxiv.org/pdf/2604.09502)  

**Abstract**: AI agents increasingly operate in multi-agent environments where outcomes depend on coordination. We distinguish primary algorithmic monoculture -- baseline action similarity -- from strategic algorithmic monoculture, whereby agents adjust similarity in response to incentives. We implement a simple experimental design that cleanly separates these forces, and deploy it on human and large language model (LLM) subjects. LLMs exhibit high levels of baseline similarity (primary monoculture) and, like humans, they regulate it in response to coordination incentives (strategic monoculture). While LLMs coordinate extremely well on similar actions, they lag behind humans in sustaining heterogeneity when divergence is rewarded. 

---
# SAGE: A Service Agent Graph-guided Evaluation Benchmark 

**Authors**: Ling Shi, Yuqin Dai, Ziyin Wang, Ning Gao, Wei Zhang, Chaozheng Wang, Yujie Wang, Wei He, Jinpeng Wang, Deiyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.09285)  

**Abstract**: The development of Large Language Models (LLMs) has catalyzed automation in customer service, yet benchmarking their performance remains challenging. Existing benchmarks predominantly rely on static paradigms and single-dimensional metrics, failing to account for diverse user behaviors or the strict adherence to structured Standard Operating Procedures (SOPs) required in real-world deployments. To bridge this gap, we propose SAGE (Service Agent Graph-guided Evaluation), a universal multi-agent benchmark for automated, dual-axis assessment. SAGE formalizes unstructured SOPs into Dynamic Dialogue Graphs, enabling precise verification of logical compliance and comprehensive path coverage. We introduce an Adversarial Intent Taxonomy and a modular Extension Mechanism, enabling low-cost deployment across domains and facilitating automated dialogue data synthesis. Evaluation is conducted via a framework where Judge Agents and a Rule Engine analyze interactions between User and Service Agents to generate deterministic ground truth. Extensive experiments on 27 LLMs across 6 industrial scenarios reveal a significant ``Execution Gap'' where models accurately classify intents but fail to derive correct subsequent actions. We also observe ``Empathy Resilience'', a phenomenon where models maintain polite conversational facades despite underlying logical failures under high adversarial intensity. Code and resources are available at this https URL. 

---
# HiL-Bench (Human-in-Loop Benchmark): Do Agents Know When to Ask for Help? 

**Authors**: Mohamed Elfeki, Tu Trinh, Kelvin Luu, Guangze Luo, Nathan Hunt, Ernesto Montoya, Nandan Marwaha, Yannis He, Charles Wang, Fernando Crabedo, Alessa Castilo, Bing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09408)  

**Abstract**: Frontier coding agents solve complex tasks when given complete context but collapse when specifications are incomplete or ambiguous. The bottleneck is not raw capability, but judgment: knowing when to act autonomously and when to ask for help. Current benchmarks are blind to this failure mode. They supply unambiguous detailed instructions and solely reward execution correctness, so an agent that makes a lucky guess for a missing requirement will score identically to one that would have asked to be certain.
We present HiL-Bench (Human-in-the-Loop Benchmark) to measure this selective escalation skill. Each task contains human-validated blockers (missing information, ambiguous requests, contradictory information) that surface only through progressive exploration, not upfront inspection. Our core metric, Ask-F1, the harmonic mean of question precision and blocker recall, captures the tension between over-asking and silent guessing; its structure architecturally prevents gaming through question spam.
Evaluation across SWE and text-to-SQL domains reveals a large universal judgment gap: no frontier model recovers more than a fraction of its full-information performance when deciding whether to ask. Failure analysis identifies three key help-seeking patterns: overconfident wrong beliefs with no gap detection; high uncertainty detection yet persistent errors; broad, imprecise escalation without self-correction. These consistent patterns confirm poor help-seeking is a model-level flaw, not task-specific. RL training on shaped Ask-F1 reward shows judgment is trainable: a 32B model improves both help-seeking quality and task pass rate, with gains that transfer across domains. The model does not learn domain-specific heuristics for when to ask; it learns to detect unresolvable uncertainty and act on it. 

---
# Constraint-Aware Corrective Memory for Language-Based Drug Discovery Agents 

**Authors**: Maochen Sun, Youzhi Zhang, Gaofeng Meng  

**Link**: [PDF](https://arxiv.org/pdf/2604.09308)  

**Abstract**: Large language models are making autonomous drug discovery agents increasingly feasible, but reliable success in this setting is not determined by any single action or molecule. It is determined by whether the final returned set jointly satisfies protocol-level requirements such as set size, diversity, binding quality, and developability. This creates a fundamental control problem: the agent plans step by step, while task validity is decided at the level of the whole candidate set. Existing language-based drug discovery systems therefore tend to rely on long raw history and under-specified self-reflection, making failure localization imprecise and planner-facing agent states increasingly noisy. We present CACM (Constraint-Aware Corrective Memory), a language-based drug discovery framework built around precise set-level diagnosis and a concise memory write-back mechanism. CACM introduces protocol auditing and a grounded diagnostician, which jointly analyze multimodal evidence spanning task requirements, pocket context, and candidate-set evidence to localize protocol violations, generate actionable remediation hints, and bias the next action toward the most relevant correction. To keep planning context compact, CACM organizes memory into static, dynamic, and corrective channels and compresses them before write-back, thereby preserving persistent task information while exposing only the most decision-relevant failures. Our experimental results show that CACM improves the target-level success rate by 36.4% over the state-of-the-art baseline. The results show that reliable language-based drug discovery benefits not only from more powerful molecular tools, but also from more precise diagnosis and more economical agent states. 

---
# DRBENCHER: Can Your Agent Identify the Entity, Retrieve Its Properties and Do the Math? 

**Authors**: Young-Suk Lee, Ramon Fernandez Astudillo, Radu Florian  

**Link**: [PDF](https://arxiv.org/pdf/2604.09251)  

**Abstract**: Deep research agents increasingly interleave web browsing with multi-step computation, yet existing benchmarks evaluate these capabilities in isolation, creating a blind spot in assessing real-world performance. We introduce DRBENCHER, a synthetic benchmark generator for questions that require both browsing and computation. It enforces four criteria: verifiability (gold answers are computed by executing parameterized code over knowledge-graph values), complexity (multi-hop entity identification, property retrieval, and domain-specific computation), difficulty (a two-stage verification cascade filters out questions solvable by the generating model), and diversity (a greedy max-min embedding filter maximizes coverage). These criteria are realized via a unified answer-first pipeline spanning five domains: biochemistry, financial, geophysical, security, and history. Human evaluation shows 76% validity (84% excluding stale data), with 35% of errors due to outdated knowledge-graph entries, highlighting an inherent limitation of systems that reason over evolving data. Automatic evaluation shows that the strongest frontier model achieves only 20% answer accuracy. Compared to manually constructed benchmarks (BrowseComp+, MATH-500, GPQA), DRBENCHER achieves the highest semantic diversity. 

---
# Process Reward Agents for Steering Knowledge-Intensive Reasoning 

**Authors**: Jiwoong Sohn, Tomasz Sternal, Kenneth Styppa, Torsten Hoefler, Michael Moor  

**Link**: [PDF](https://arxiv.org/pdf/2604.09482)  

**Abstract**: Reasoning in knowledge-intensive domains remains challenging as intermediate steps are often not locally verifiable: unlike math or code, evaluating step correctness may require synthesizing clues across large external knowledge sources. As a result, subtle errors can propagate through reasoning traces, potentially never to be detected. Prior work has proposed process reward models (PRMs), including retrieval-augmented variants, but these methods operate post hoc, scoring completed trajectories, which prevents their integration into dynamic inference procedures. Here, we introduce Process Reward Agents (PRA), a test-time method for providing domain-grounded, online, step-wise rewards to a frozen policy. In contrast to prior retrieval-augmented PRMs, PRA enables search-based decoding to rank and prune candidate trajectories at every generation step. Experiments on multiple medical reasoning benchmarks demonstrate that PRA consistently outperforms strong baselines, achieving 80.8% accuracy on MedQA with Qwen3-4B, a new state of the art at the 4B scale. Importantly, PRA generalizes to unseen frozen policy models ranging from 0.5B to 8B parameters, improving their accuracy by up to 25.7% without any policy model updates. More broadly, PRA suggests a paradigm in which frozen reasoners are decoupled from domain-specific reward modules, allowing the deployment of new backbones in complex domains without retraining. 

---
# E3-TIR: Enhanced Experience Exploitation for Tool-Integrated Reasoning 

**Authors**: Weiyang Guo, Zesheng Shi, Liye Zhao, Jiayuan Ma, Zeen Zhu, Junxian He, Min Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.09455)  

**Abstract**: While Large Language Models (LLMs) have demonstrated significant potential in Tool-Integrated Reasoning (TIR), existing training paradigms face significant limitations: Zero-RL suffers from inefficient exploration and mode degradation due to a lack of prior guidance, while SFT-then-RL is limited by high data costs and capability plateaus caused by low-entropy collapse. To address these challenges, we propose E3-TIR (Enhanced Experience Exploitation), a warm-up paradigm for the early stages of agent training. Specifically, we formulate training as the dynamic integration of three experience types: Expert Prefixes, Expert Guided, and Self-Exploration. By executing diverse branching exploration around expert "anchors" and employing a mix policy optimization mechanism, we effectively mitigate distribution shifts and resolve optimization conflicts arising from shared prefixes. Our method dynamically adapts the model's knowledge boundaries, effectively balancing exploration diversity with training this http URL results demonstrate that E3-TIR achieves a 6 performance improvement over traditional paradigms on tool-use tasks, while requiring less than 10 of the synthetic data. Furthermore, in terms of ROI, a comprehensive metric integrating performance, data cost, and training efficiency we achieve a 1.46x gain compared to baselines. Code is available at this https URL. 

---
# SEA-Eval: A Benchmark for Evaluating Self-Evolving Agents Beyond Episodic Assessment 

**Authors**: Sihang Jiang, Lipeng Ma, Zhonghua Hong, Keyi Wang, Zhiyu Lu, Shisong Chen, Jinghao Zhang, Tianjun Pan, Weijia Zhou, Jiaqing Liang, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.08988)  

**Abstract**: Current LLM-based agents demonstrate strong performance in episodic task execution but remain constrained by static toolsets and episodic amnesia, failing to accumulate experience or optimize strategies across task boundaries. While the Self-Evolving Agent (SEA) paradigm has been previously proposed, this paper contributes a new formal definition of SEA grounded in digital embodiment and continuous cross-task evolution, and introduces SEA-Eval, the first benchmark designed to evaluate SEA characteristics across two dimensions, intra-task execution reliability and long-term evolutionary performance. By organizing tasks into sequential streams and analyzing Success Rate and Token Consumption over time, SEA-Eval quantifies evolutionary gain and structural stability in ways that existing episodic benchmarks cannot. Empirical evaluations reveal a significant evolutionary bottleneck in current state-of-the-art frameworks, where identical success rates mask up to 31.2 times differences in token consumption and divergent evolutionary trajectories under sequential analysis. SEA-Eval provides a rigorous scientific foundation for advancing agents from mere task executors toward genuinely self-evolving digital entities. 

---
# Model Space Reasoning as Search in Feedback Space for Planning Domain Generation 

**Authors**: James Oswald, Daniel Oblinsky, Volodymyr Varha, Vasilije Dragovic, Harsha Kokel, Kavitha Srinivas, Michael Katz, Shirin Sohrabi  

**Link**: [PDF](https://arxiv.org/pdf/2604.08712)  

**Abstract**: The generation of planning domains from natural language descriptions remains an open problem even with the advent of large language models and reasoning models. Recent work suggests that while LLMs have the ability to assist with domain generation, they are still far from producing high quality domains that can be deployed in practice. To this end, we investigate the ability of an agentic language model feedback framework to generate planning domains from natural language descriptions that have been augmented with a minimal amount of symbolic information. In particular, we evaluate the quality of the generated domains under various forms of symbolic feedback, including landmarks, and output from the VAL plan validator. Using these feedback mechanisms, we experiment using heuristic search over model space to optimize domain quality. 

---
# SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks 

**Authors**: Tianyi Wang, Yixia Li, Long Li, Yibiao Chen, Shaohan Huang, Yun Chen, Peng Li, Yang Liu, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.08865)  

**Abstract**: Proximal Policy Optimization (PPO) is central to aligning Large Language Models (LLMs) in reasoning tasks with verifiable rewards. However, standard token-level PPO struggles in this setting due to the instability of temporal credit assignment over long Chain-of-Thought (CoT) horizons and the prohibitive memory cost of the value model. While critic-free alternatives like GRPO mitigate these issues, they incur significant computational overhead by requiring multiple samples for baseline estimation, severely limiting training throughput. In this paper, we introduce Sequence-Level PPO (SPPO), a scalable algorithm that harmonizes the sample efficiency of PPO with the stability of outcome-based updates. SPPO reformulates the reasoning process as a Sequence-Level Contextual Bandit problem, employing a decoupled scalar value function to derive low-variance advantage signals without multi-sampling. Extensive experiments on mathematical benchmarks demonstrate that SPPO significantly surpasses standard PPO and matches the performance of computation-heavy group-based methods, offering a resource-efficient framework for aligning reasoning LLMs. 

---
# PilotBench: A Benchmark for General Aviation Agents with Safety Constraints 

**Authors**: Yalun Wu, Haotian Liu, Zhoujun Li, Boyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08987)  

**Abstract**: As Large Language Models (LLMs) advance toward embodied AI agents operating in physical environments, a fundamental question emerges: can models trained on text corpora reliably reason about complex physics while adhering to safety constraints? We address this through PilotBench, a benchmark evaluating LLMs on safety-critical flight trajectory and attitude prediction. Built from 708 real-world general aviation trajectories spanning nine operationally distinct flight phases with synchronized 34-channel telemetry, PilotBench systematically probes the intersection of semantic understanding and physics-governed prediction through comparative analysis of LLMs and traditional forecasters. We introduce Pilot-Score, a composite metric balancing 60% regression accuracy with 40% instruction adherence and safety compliance. Comparative evaluation across 41 models uncovers a Precision-Controllability Dichotomy: traditional forecasters achieve superior MAE of 7.01 but lack semantic reasoning capabilities, while LLMs gain controllability with 86--89% instruction-following at the cost of 11--14 MAE precision. Phase-stratified analysis further exposes a Dynamic Complexity Gap-LLM performance degrades sharply in high-workload phases such as Climb and Approach, suggesting brittle implicit physics models. These empirical discoveries motivate hybrid architectures combining LLMs' symbolic reasoning with specialized forecasters' numerical precision. PilotBench provides a rigorous foundation for advancing embodied AI in safety-constrained domains. 

---
# Enhancing LLM Problem Solving via Tutor-Student Multi-Agent Interaction 

**Authors**: Nurullah Eymen Özdemir, Erhan Oztop  

**Link**: [PDF](https://arxiv.org/pdf/2604.08931)  

**Abstract**: Human cognitive development is shaped not only by individual effort but by structured social interaction, where role-based exchanges such as those between a tutor and a learner, enable solutions that neither could achieve alone. Inspired by these developmental principles, we ask the question whether a tutor-student multi-agent system can create a synergistic effect by pushing Large Language Model (LLM) beyond what it can do within existing frameworks. To test the idea, we adopt autonomous coding problem domain where two agents instantiated from the same LLM assigned asymmetric roles: a student agent generates and iteratively refines solutions, while a tutor agent provides structured evaluative feedback without access to ground-truth answers. In our proposed framework (PETITE), we aim to extract better problem-solving performance from one model by structuring its interaction through complementary roles, rather than relying on stronger supervisory models or heterogeneous ensembles. Our model is evaluated on the APPS coding benchmark against state-of-the-art approaches of Self-Consistency, Self-Refine, Multi-Agent Debate, and Multi-Agent Review. The results show that our model achieves similar or higher accuracy while consuming significantly fewer tokens. These results suggest that developmentally grounded role-differentiated interaction structures provide a principled and resource-efficient paradigm for enhancing LLM problem-solving through structured peer-like interactions. Index Terms- Peer Tutoring, Scaffolding, Large Language Models, Multi-Agent Systems, Code Generation 

---
# StaRPO: Stability-Augmented Reinforcement Policy Optimization 

**Authors**: Jinghan Zhang, Fengran Mo, Tharindu Cyril Weerasooriya, Ruimin Dai, Xiaoyan Han, Yanjie Fu, Dakuo Wang, Kunpeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08905)  

**Abstract**: Reinforcement learning (RL) is effective in enhancing the accuracy of large language models in complex reasoning tasks. Existing RL policy optimization frameworks rely on final-answer correctness as feedback signals and rarely capture the internal logical structure of the reasoning process. Consequently, the models would generate fluent and semantically relevant responses but logically inconsistent, structurally erratic, or redundant. To this end, we propose StaRPO, a stability-augmented reinforcement learning framework that explicitly incorporates reasoning stability into the optimization objective. Our StaRPO decomposes stability into two computable lightweight metrics: the Autocorrelation Function (ACF) to evaluate local step-to-step coherence, and Path Efficiency (PE) to evaluate global goal-directedness of the reasoning trajectory. These stability rewards are combined with task rewards to provide complementary and process-aware feedback. We validate the effectiveness of using ACF and PE rewards by showing their correlation with logic errors on two backbone models. Experiments on four reasoning benchmarks show that StaRPO consistently outperforms compared baselines and can enhance both final-answer accuracy and logical stability. 

---
# Hidden in Plain Sight: Visual-to-Symbolic Analytical Solution Inference from Field Visualizations 

**Authors**: Pengze Li, Jiaquan Zhang, Yunbo Long, Xinping Liu, Zhou wenjie, Encheng Su, Zihang Zeng, Jiaqi Liu, Jiyao Liu, Junchi Yu, Lihao Liu, Philip Torr, Shixiang Tang, Aoran Wang, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.08863)  

**Abstract**: Recovering analytical solutions of physical fields from visual observations is a fundamental yet underexplored capability for AI-assisted scientific reasoning. We study visual-to-symbolic analytical solution inference (ViSA) for two-dimensional linear steady-state fields: given field visualizations (and first-order derivatives) plus minimal auxiliary metadata, the model must output a single executable SymPy expression with fully instantiated numeric constants. We introduce ViSA-R2 and align it with a self-verifying, solution-centric chain-of-thought pipeline that follows a physicist-like pathway: structural pattern recognition solution-family (ansatz) hypothesis parameter derivation consistency verification. We also release ViSA-Bench, a VLM-ready synthetic benchmark covering 30 linear steady-state scenarios with verifiable analytical/symbolic annotations, and evaluate predictions by numerical accuracy, expression-structure similarity, and character-level accuracy. Using an 8B open-weight Qwen3-VL backbone, ViSA-R2 outperforms strong open-source baselines and the evaluated closed-source frontier VLMs under a standardized protocol. 

---
# Camera Artist: A Multi-Agent Framework for Cinematic Language Storytelling Video Generation 

**Authors**: Haobo Hu, Qi Mao, Yuanhang Li, Libiao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2604.09195)  

**Abstract**: We propose Camera Artist, a multi-agent framework that models a real-world filmmaking workflow to generate narrative videos with explicit cinematic language. While recent multi-agent systems have made substantial progress in automating filmmaking workflows from scripts to videos, they often lack explicit mechanisms to structure narrative progression across adjacent shots and deliberate use of cinematic language, resulting in fragmented storytelling and limited filmic quality. To address this, Camera Artist builds upon established agentic pipelines and introduces a dedicated Cinematography Shot Agent, which integrates recursive storyboard generation to strengthen shot-to-shot narrative continuity and cinematic language injection to produce more expressive, film-oriented shot designs. Extensive quantitative and qualitative results demonstrate that our approach consistently outperforms existing baselines in narrative consistency, dynamic expressiveness, and perceived film quality. 

---
# VISOR: Agentic Visual Retrieval-Augmented Generation via Iterative Search and Over-horizon Reasoning 

**Authors**: Yucheng Shen, Jiulong Wu, Jizhou Huang, Dawei Yin, Lingyong Yan, Min Cao  

**Link**: [PDF](https://arxiv.org/pdf/2604.09508)  

**Abstract**: Visual Retrieval-Augmented Generation (VRAG) empowers Vision-Language Models to retrieve and reason over visually rich documents. To tackle complex queries requiring multi-step reasoning, agentic VRAG systems interleave reasoning with iterative retrieval.. However, existing agentic VRAG faces two critical bottlenecks. (1) Visual Evidence Sparsity: key evidence is scattered across pages yet processed in isolation, hindering cross-page reasoning; moreover, fine-grained intra-image evidence often requires precise visual actions, whose misuse degrades retrieval quality; (2) Search Drift in Long Horizons: the accumulation of visual tokens across retrieved pages dilutes context and causes cognitive overload, leading agents to deviate from their search objective. To address these challenges, we propose VISOR (Visual Retrieval-Augmented Generation via Iterative Search and Over-horizon Reasoning), a unified single-agent framework. VISOR features a structured Evidence Space for progressive cross-page reasoning, coupled with a Visual Action Evaluation and Correction mechanism to manage visual actions. Additionally, we introduce a Dynamic Trajectory with Sliding Window and Intent Injection to mitigate search drift. They anchor the evidence space while discarding earlier raw interactions, preventing context from being overwhelmed by visual tokens. We train VISOR using a Group Relative Policy Optimization-based Reinforcement Learning (GRPO-based RL) pipeline with state masking and credit assignment tailored for dynamic context reconstruction. Extensive experiments on ViDoSeek, SlideVQA, and MMLongBench demonstrate that VISOR achieves state-of-the-art performance with superior efficiency for long-horizon visual reasoning tasks. 

---
# ECHO: Efficient Chest X-ray Report Generation with One-step Block Diffusion 

**Authors**: Lifeng Chen, Tianqi You, Hao Liu, Zhimin Bao, Jile Jiao, Xiao Han, Zhicai Ou, Tao Sun, Xiaofeng Mou, Xiaojie Jin, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09450)  

**Abstract**: Chest X-ray report generation (CXR-RG) has the potential to substantially alleviate radiologists' workload. However, conventional autoregressive vision--language models (VLMs) suffer from high inference latency due to sequential token decoding. Diffusion-based models offer a promising alternative through parallel generation, but they still require multiple denoising iterations. Compressing multi-step denoising to a single step could further reduce latency, but often degrades textual coherence due to the mean-field bias introduced by token-factorized denoisers. To address this challenge, we propose \textbf{ECHO}, an efficient diffusion-based VLM (dVLM) for chest X-ray report generation. ECHO enables stable one-step-per-block inference via a novel Direct Conditional Distillation (DCD) framework, which mitigates the mean-field limitation by constructing unfactorized supervision from on-policy diffusion trajectories to encode joint token dependencies. In addition, we introduce a Response-Asymmetric Diffusion (RAD) training strategy that further improves training efficiency while maintaining model effectiveness. Extensive experiments demonstrate that ECHO surpasses state-of-the-art autoregressive methods, improving RaTE and SemScore by \textbf{64.33\%} and \textbf{60.58\%} respectively, while achieving an \textbf{$8\times$} inference speedup without compromising clinical accuracy. 

---
# OpenKedge: Governing Agentic Mutation with Execution-Bound Safety and Evidence Chains 

**Authors**: Jun He, Deying Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08601)  

**Abstract**: The rise of autonomous AI agents exposes a fundamental flaw in API-centric architectures: probabilistic systems directly execute state mutations without sufficient context, coordination, or safety guarantees. We introduce OpenKedge, a protocol that redefines mutation as a governed process rather than an immediate consequence of API invocation. OpenKedge requires actors to submit declarative intent proposals, which are evaluated against deterministically derived system state, temporal signals, and policy constraints prior to execution. Approved intents are compiled into execution contracts that strictly bound permitted actions, resource scope, and time, and are enforced via ephemeral, task-oriented identities. This shifts safety from reactive filtering to preventative, execution-bound enforcement. Crucially, OpenKedge introduces an Intent-to-Execution Evidence Chain (IEEC), which cryptographically links intent, context, policy decisions, execution bounds, and outcomes into a unified lineage. This transforms mutation into a verifiable and reconstructable process, enabling deterministic auditability and reasoning about system behavior. We evaluate OpenKedge across multi-agent conflict scenarios and cloud infrastructure mutations. Results show that the protocol deterministically arbitrates competing intents and cages unsafe execution while maintaining high throughput, establishing a principled foundation for safely operating agentic systems at scale. 

---
# BadSkill: Backdoor Attacks on Agent Skills via Model-in-Skill Poisoning 

**Authors**: Guiyao Tie, Jiawen Shi, Pan Zhou, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.09378)  

**Abstract**: Agent ecosystems increasingly rely on installable skills to extend functionality, and some skills bundle learned model artifacts as part of their execution logic. This creates a supply-chain risk that is not captured by prompt injection or ordinary plugin misuse: a third-party skill may appear benign while concealing malicious behavior inside its bundled model. We present BadSkill, a backdoor attack formulation that targets this model-in-skill threat surface. In BadSkill, an adversary publishes a seemingly benign skill whose embedded model is backdoor-fine-tuned to activate a hidden payload only when routine skill parameters satisfy attacker-chosen semantic trigger combinations. To realize this attack, we train the embedded classifier with a composite objective that combines classification loss, margin-based separation, and poison-focused optimization, and evaluate it in an OpenClaw-inspired simulation environment that preserves third-party skill installation and execution while enabling controlled multi-model study. Our benchmark spans 13 skills, including 8 triggered tasks and 5 non-trigger control skills, with a combined main evaluation set of 571 negative-class queries and 396 trigger-aligned queries. Across eight architectures (494M--7.1B parameters) from five model families, BadSkill achieves up to 99.5\% average attack success rate (ASR) across the eight triggered skills while maintaining strong benign-side accuracy on negative-class queries. In poison-rate sweeps on the standard test split, a 3\% poison rate already yields 91.7\% ASR. The attack remains effective across the evaluated model scales and under five text perturbation types. These findings identify model-bearing skills as a distinct model supply-chain risk in agent ecosystems and motivate stronger provenance verification and behavioral vetting for third-party skill artifacts. 

---
# SkillMOO: Multi-Objective Optimization of Agent Skills for Software Engineering 

**Authors**: Jingzhi Gong, Ruizhen Gu, Zhiwei Fei, Yazhuo Cao, Lukas Twist, Alina Geiger, Shuo Han, Dominik Sobania, Federica Sarro, Jie M. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09297)  

**Abstract**: Agent skills provide modular, task-specific guidance for LLM- based coding agents, but manually tuning skill bundles to balance success rate, cost, and runtime is expensive and fragile. We present SkillMOO, a multi-objective optimization framework that automatically evolves skill bundles using LLM-proposed edits and NSGA-II survivor selection: a solver agent evaluates candidate skill bundles on coding tasks and an optimizer agent proposes bundle edits based on failure analysis. On three SkillsBench software engineering tasks, SkillMOO improves pass rate by up to 131% while reducing cost up to 32% relative to the best baseline per task at low optimization overhead. Pattern analysis reveals pruning and substitution as primary drivers of improvement, suggesting effective bundles favor minimal, focused content over accumulated instructions. 

---
# GRM: Utility-Aware Jailbreak Attacks on Audio LLMs via Gradient-Ratio Masking 

**Authors**: Yunqiang Wang, Hengyuan Na, Di Wu, Miao Hu, Guocong Quan  

**Link**: [PDF](https://arxiv.org/pdf/2604.09222)  

**Abstract**: Audio large language models (ALLMs) enable rich speech-text interaction, but they also introduce jailbreak vulnerabilities in the audio modality. Existing audio jailbreak methods mainly optimize jailbreak success while overlooking utility preservation, as reflected in transcription quality and question answering performance. In practice, stronger attacks often come at the cost of degraded utility. To study this trade-off, we revisit existing attacks by varying their perturbation coverage in the frequency domain, from partial-band to full-band, and find that broader frequency coverage does not necessarily improve jailbreak performance, while utility consistently deteriorates. This suggests that concentrating perturbation on a subset of bands can yield a better attack-utility trade-off than indiscriminate full-band coverage. Based on this insight, we propose GRM, a utility-aware frequency-selective jailbreak framework. It ranks Mel bands by their attack contribution relative to utility sensitivity, perturbs only a selected subset of bands, and learns a reusable universal perturbation under a semantic-preservation objective. Experiments on four representative ALLMs show that GRM achieves an average Jailbreak Success Rate (JSR) of 88.46% while providing a better attack-utility trade-off than representative baselines. These results highlight the potential of frequency-selective perturbation for better balancing attack effectiveness and utility preservation in audio jailbreak. Content Warning: This paper includes harmful query examples and unsafe model responses. 

---
# LLM-Rosetta: A Hub-and-Spoke Intermediate Representation for Cross-Provider LLM API Translation 

**Authors**: Peng Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.09360)  

**Abstract**: The rapid proliferation of Large Language Model (LLM) providers--each exposing proprietary API formats--has created a fragmented ecosystem where applications become tightly coupled to individual vendors. Switching or bridging providers requires $O(N^2)$ bilateral adapters, impeding portability and multi-provider architectures. We observe that despite substantial syntactic divergence, the major LLM APIs share a common semantic core: the practical challenge is the combinatorial surface of syntactic variations, not deep semantic incompatibility. Based on this finding, we present LLM-Rosetta, an open-source translation framework built on a hub-and-spoke Intermediate Representation (IR) that captures the shared semantic core--messages, content parts, tool calls, reasoning traces, and generation controls--in a 9-type content model and 10-type stream event schema. A modular Ops-composition converter architecture enables each API standard to be added independently. LLM-Rosetta supports bidirectional conversion (provider-to-IR-to-provider) for both request and response payloads, including chunk-level streaming with stateful context management. We implement converters for four API standards (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Google GenAI), covering the vast majority of commercial providers. Empirical evaluation demonstrates lossless round-trip fidelity, correct streaming behavior, and sub-100 microsecond conversion overhead--competitive with LiteLLM's single-pass approach while providing bidirectionality and provider neutrality. LLM-Rosetta passes the Open Responses compliance suite and is deployed in production at Argonne National Laboratory. Code is available at this https URL. 

---
# Yes, But Not Always. Generative AI Needs Nuanced Opt-in 

**Authors**: Wiebke Hutiri, Morgan Scheuerman, Shruti Nagpal, Austin Hoag, Alice Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09413)  

**Abstract**: This paper argues that a one-size-fits-all approach to specifying consent for the use of creative works in generative AI is insufficient. Real-world ownership and rights holder structures, the imitation of artistic styles and likeness, and the limitless contexts of use of AI outputs make the status quo of binary consent with opt-in by default untenable. To move beyond the current impasse, we consider levers of control in generative AI workflows at training, inference, and dissemination. Based on these insights, we position inference-time opt-in as an overlooked opportunity for nuanced consent verification. We conceptualize nuanced consent conditions for opt-in and propose an agent-based inference-time opt-in architecture to verify if user intent requests meet conditional consent granted by rights holders. In a case study for music, we demonstrate that nuanced opt-in at inference can account for established rights and re-establish a balance of power between rights holders and AI developers. 

---
# Artificial intelligence can persuade people to take political actions 

**Authors**: Kobi Hackenburg, Luke Hewitt, Caroline Wagner, Ben M. Tappin, Christopher Summerfield  

**Link**: [PDF](https://arxiv.org/pdf/2604.09200)  

**Abstract**: There is substantial concern about the ability of advanced artificial intelligence to influence people's behaviour. A rapidly growing body of research has found that AI can produce large persuasive effects on people's attitudes, but whether AI can persuade people to take consequential real-world actions has remained unclear. In two large preregistered experiments N=17,950 responses from 14,779 people), we used conversational AI models to persuade participants on a range of attitudinal and behavioural outcomes, including signing real petitions and donating money to charity. We found sizable AI persuasion effects on these behavioural outcomes (e.g. +19.7 percentage points on petition signing). However, we observed no evidence of a correlation between AI persuasion effects on attitudes and behaviour. Moreover, we replicated prior findings that information provision drove effects on attitudes, but found no such evidence for our behavioural outcomes. In a test of eight behavioural persuasion strategies, all outperformed the most effective attitudinal persuasion strategy, but differences among the eight were small. Taken together, these results suggest that previous findings relying on attitudinal outcomes may generalize poorly to behaviour, and therefore risk substantially mischaracterizing the real-world behavioural impact of AI persuasion. 

---
# The AI Codebase Maturity Model: From Assisted Coding to Self-Sustaining Systems 

**Authors**: Andy Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2604.09388)  

**Abstract**: AI coding tools are widely adopted, but most teams plateau at prompt-and-review without a framework for systematic progression. This paper presents the AI Codebase Maturity Model (ACMM), a 5-level framework describing how codebases evolve from basic AI-assisted coding to self-sustaining systems. Inspired by CMMI, each level is defined by its feedback loop topology the specific mechanisms that must exist before the next level becomes possible. I validate the model through a 4-month experience report maintaining KubeStellar Console, a CNCF Kubernetes dashboard built from scratch with Claude Code (Opus) and GitHub Copilot. The system currently operates with 63 CI/CD workflows, 32 nightly test suites, 91% code coverage, and achieves bug-to-fix times under 30 minutes 24 hours a day. The central finding: the intelligence of an AI-driven development system resides not in the AI model itself, but in the infrastructure of instructions, tests, metrics, and feedback loops that surround it. You cannot skip levels, and at each level, the thing that unlocks the next one is another feedback mechanism. Testing the volume of test cases, the coverage thresholds, and the reliability of test execution proved to be the single most important investment in the entire journey. 

---
# Generalization and Scaling Laws for Mixture-of-Experts Transformers 

**Authors**: Mansour Zoubeirou a Mayaki  

**Link**: [PDF](https://arxiv.org/pdf/2604.09175)  

**Abstract**: We develop a theory of generalization and scaling for Mixture-of-Experts (MoE) Transformers that cleanly separates \emph{active} per-input capacity from routing combinatorics. By conditioning on fixed routing patterns and union-bounding across them, we derive a sup-norm covering-number bound whose metric entropy scales with the active parameter budget and incurs a MoE-specific routing overhead. Combined with a standard ERM analysis for squared loss, this yields a generalization bound under a $d$-dimensional manifold data model and $C^\beta$ targets, showing that approximation and estimation trade off as in dense networks once active parameters are accounted for appropriately. We further prove a constructive approximation theorem for MoE architectures, showing that, under the approximation construction, error can decrease either by scaling active capacity or by increasing the number of experts, depending on the dominant bottleneck. From these results we derive neural scaling laws for model size, data size, and compute-optimal tradeoffs. Overall, our results provide a transparent statistical reference point for reasoning about MoE scaling, clarifying which behaviors are certified by worst-case theory and which must arise from data-dependent routing structure or optimization dynamics. 

---
# CORA: Conformal Risk-Controlled Agents for Safeguarded Mobile GUI Automation 

**Authors**: Yushi Feng, Junye Du, Qifan Wang, Zizhan Ma, Qian Niu, Yutaka Matsuo, Long Feng, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09155)  

**Abstract**: Graphical user interface (GUI) agents powered by vision language models (VLMs) are rapidly moving from passive assistance to autonomous operation. However, this unrestricted action space exposes users to severe and irreversible financial, privacy or social harm. Existing safeguards rely on prompt engineering, brittle heuristics and VLM-as-critic lack formal verification and user-tunable guarantees. We propose CORA (COnformal Risk-controlled GUI Agent), a post-policy, pre-action safeguarding framework that provides statistical guarantees on harmful executed actions. CORA reformulates safety as selective action execution: we train a Guardian model to estimate action-conditional risk for each proposed step. Rather than thresholding raw scores, we leverage Conformal Risk Control to calibrate an execute/abstain boundary that satisfies a user-specified risk budget and route rejected actions to a trainable Diagnostician model, which performs multimodal reasoning over rejected actions to recommend interventions (e.g., confirm, reflect, or abort) to minimize user burden. A Goal-Lock mechanism anchors assessment to a clarified, frozen user intent to resist visual injection attacks. To rigorously evaluate this paradigm, we introduce Phone-Harm, a new benchmark of mobile safety violations with step-level harm labels under real-world settings. Experiments on Phone-Harm and public benchmarks against diverse baselines validate that CORA improves the safety--helpfulness--interruption Pareto frontier, offering a practical, statistically grounded safety paradigm for autonomous GUI execution. Code and benchmark are available at this http URL. 

---
# Structuring versus Problematizing: How LLM-based Agents Scaffold Learning in Diagnostic Reasoning 

**Authors**: Fatma Betül Güreş, Tanya Nazaretsky, Seyed Parsa Neshaei, Tanja Käser  

**Link**: [PDF](https://arxiv.org/pdf/2604.09158)  

**Abstract**: Supporting students in developing diagnostic reasoning is a key challenge across educational domains. Novices often face cognitive biases such as premature closure and over-reliance on heuristics, and they struggle to transfer diagnostic strategies to new cases. Scenario-based learning (SBL) enhanced by Learning Analytics (LA) and large language models (LLM) offers a promising approach by combining realistic case experiences with personalized scaffolding. Yet, how different scaffolding approaches shape reasoning processes remains insufficiently explored. This study introduces PharmaSim Switch, an SBL environment for pharmacy technician training, extended with an LA- and LLM-powered pharmacist agent that implements pedagogical conversations rooted in two theory-driven scaffolding approaches: \emph{structuring} and \emph{problematizing}, as well as a student learning trajectory. In a between-groups experiment, 63 vocational students completed a learning scenario, a near-transfer scenario, and a far-transfer scenario under one of the two scaffolding conditions. Results indicate that both scaffolding approaches were effective in supporting the use of diagnostic strategies. Performance outcomes were primarily influenced by scenario complexity rather than students' prior knowledge or the scaffolding approach used. The structuring approach was associated with more accurate Active and Interactive participation, whereas problematizing elicited more Constructive engagement. These findings underscore the value of combining scaffolding approaches when designing LA- and LLM-based systems to effectively foster diagnostic reasoning. 

---
# TensorHub: Scalable and Elastic Weight Transfer for LLM RL Training 

**Authors**: Chenhao Ye, Huaizheng Zhang, Mingcong Han, Baoquan Zhong, Xiang Li, Qixiang Chen, Xinyi Zhang, Weidong Zhang, Kaihua Jiang, Wang Zhang, He Sun, Wencong Xiao, Andrea C. Arpaci-Dusseau, Remzi H. Arpaci-Dusseau  

**Link**: [PDF](https://arxiv.org/pdf/2604.09107)  

**Abstract**: Modern LLM reinforcement learning (RL) workloads require a highly efficient weight transfer system to scale training across heterogeneous computational resources. However, existing weight transfer approaches either fail to provide flexibility for dynamically scaling clusters or incur fundamental data movement overhead, resulting in poor performance.
We introduce Reference-Oriented Storage (ROS), a new storage abstraction for RL weight transfer that exploits the highly replicated model weights in place. ROS presents the illusion that certain versions of the model weights are stored and can be fetched on demand. Underneath, ROS does not physically store any copies of the weights; instead, it tracks the workers that hold these weights on GPUs for inference. Upon request, ROS directly uses them to serve reads. We build TensorHub, a production-quality system that extends the ROS idea with topology-optimized transfer, strong consistency, and fault tolerance. Evaluation shows that TensorHub fully saturates RDMA bandwidth and adapts to three distinct rollout workloads with minimal engineering effort. Specifically, TensorHub reduces total GPU stall time by up to 6.7x for standalone rollouts, accelerates weight update for elastic rollout by 4.8x, and cuts cross-datacenter rollout stall time by 19x. TensorHub has been deployed in production to support cutting-edge RL training. 

---
# Scheming in the wild: detecting real-world AI scheming incidents with open-source intelligence 

**Authors**: Tommy Shaffer Shane, Simon Mylius, Hamish Hobbs  

**Link**: [PDF](https://arxiv.org/pdf/2604.09104)  

**Abstract**: Scheming, the covert pursuit of misaligned goals by AI systems, represents a potentially catastrophic risk, yet scheming research suffers from significant limitations. In particular, scheming evaluations demonstrate behaviours that may not occur in real-world settings, limiting scientific understanding, hindering policy development, and not enabling real-time detection of loss of control incidents. Real-world evidence is needed, but current monitoring techniques are not effective for this purpose. This paper introduces a novel open-source intelligence (OSINT) methodology for detecting real-world scheming incidents: collecting and analysing transcripts from chatbot conversations or command-line interactions shared online. Analysing over 183,420 transcripts from X (formerly Twitter), we identify 698 real-world scheming-related incidents between October 2025 and March 2026. We observe a statistically significant 4.9x increase in monthly incidents from the first to last month, compared to a 1.7x increase in posts discussing scheming. We find evidence of multiple scheming-related behaviours in real-world deployments previously reported only in experiments, many resulting in real-world harms. While we did not detect catastrophic scheming incidents, the behaviours observed demonstrate concerning precursors, such as willingness to disregard instructions, circumvent safeguards, lie to users, and single-mindedly pursue goals in harmful ways. As AI systems become more capable, these could evolve into more strategic scheming with potentially catastrophic consequences. Our findings demonstrate the viability of transcript-based OSINT as a scalable approach to real-world scheming detection supporting scientific research, policy development, and emergency response. We recommend further investment towards OSINT techniques for monitoring scheming and loss of control. 

---
# PS-TTS: Phonetic Synchronization in Text-to-Speech for Achieving Natural Automated Dubbing 

**Authors**: Changi Hong, Yoonah Song, Hwayoung Park, Chaewoon Bang, Dayeon Gu, Do Hyun Lee, Hong Kook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.09111)  

**Abstract**: Recently, artificial intelligence-based dubbing technology has advanced, enabling automated dubbing (AD) to convert the source speech of a video into target speech in different languages. However, natural AD still faces synchronization challenges such as duration and lip-synchronization (lip-sync), which are crucial for preserving the viewer experience. Therefore, this paper proposes a synchronization method for AD processes that paraphrases translated text, comprising two steps: isochrony for timing constraints and phonetic synchronization (PS) to preserve lip-sync. First, we achieve isochrony by paraphrasing the translated text with a language model, ensuring the target speech duration matches that of the source speech. Second, we introduce PS, which employs dynamic time warping (DTW) with local costs of vowel distances measured from training data so that the target text composes vowels with pronunciations similar to source vowels. Third, we extend this approach to PSComet, which jointly considers semantic and phonetic similarity to preserve meaning better. The proposed methods are incorporated into text-to-speech systems, PS-TTS and PS-Comet TTS. The performance evaluation using Korean and English lip-reading datasets and a voice-actor dubbing dataset demonstrates that both systems outperform TTS without PS on several objective metrics and outperform voice actors in Korean-to-English and English-to-Korean dubbing. We extend the experiments to French, testing all pairs among these languages to evaluate cross-linguistic applicability. Across all language pairs, PS-Comet performed best, balancing lip-sync accuracy with semantic preservation, confirming that PS-Comet achieves more accurate lip-sync with semantic preservation than PS alone. 

---
# Mosaic: Multimodal Jailbreak against Closed-Source VLMs via Multi-View Ensemble Optimization 

**Authors**: Yuqin Lan, Gen Li, Yuanze Hu, Weihao Shen, Zhaoxin Fan, Faguo Wu, Xiao Zhang, Laurence T. Yang, Zhiming Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.09253)  

**Abstract**: Vision-Language Models (VLMs) are powerful but remain vulnerable to multimodal jailbreak attacks. Existing attacks mainly rely on either explicit visual prompt attacks or gradient-based adversarial optimization. While the former is easier to detect, the latter produces subtle perturbations that are less perceptible, but is usually optimized and evaluated under homogeneous open-source surrogate-target settings, leaving its effectiveness on commercial closed-source VLMs under heterogeneous settings unclear. To examine this issue, we study different surrogate-target settings and observe a consistent gap between homogeneous and heterogeneous settings, a phenomenon we term surrogate dependency. Motivated by this finding, we propose Mosaic, a Multi-view ensemble optimization framework for multimodal jailbreak against closed-source VLMs, which alleviates surrogate dependency under heterogeneous surrogate-target settings by reducing over-reliance on any single surrogate model and visual view. Specifically, Mosaic incorporates three core components: a Text-Side Transformation module, which perturbs refusal-sensitive lexical patterns; a Multi-View Image Optimization module, which updates perturbations under diverse cropped views to avoid overfitting to a single visual view; and a Surrogate Ensemble Guidance module, which aggregates optimization signals from multiple surrogate VLMs to reduce surrogate-specific bias. Extensive experiments on safety benchmarks demonstrate that Mosaic achieves state-of-the-art Attack Success Rate and Average Toxicity against commercial closed-source VLMs. 

---
# DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation 

**Authors**: Li Huang, Zhongxin Liu, Yifan Wu, Tao Yin, Dong Li, Jichao Bi, Nankun Mu, Hongyu Zhang, Meng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.09089)  

**Abstract**: Large Language Models (LLMs) for code generation can replicate insecure patterns from their training data. To mitigate this, a common strategy for security hardening is to fine-tune models using supervision derived from the final transformer layer. However, this design may suffer from a final-layer bottleneck: vulnerability-discriminative cues can be distributed across layers and become less detectable near the output representations optimized for next-token prediction. To diagnose this issue, we perform layer-wise linear probing. We observe that vulnerability-related signals are most detectable in a band of intermediate-to-upper layers yet attenuate toward the final layers. Motivated by this observation, we introduce DeepGuard, a framework that leverages distributed security-relevant cues by aggregating representations from multiple upper layers via an attention-based module. The aggregated signal powers a dedicated security analyzer within a multi-objective training objective that balances security enhancement and functional correctness, and further supports a lightweight inference-time steering strategy. Extensive experiments across five code LLMs demonstrate that DeepGuard improves the secure-and-correct generation rate by an average of 11.9% over strong baselines such as SVEN. It also preserves functional correctness while exhibiting generalization to held-out vulnerability types. Our code is public at this https URL. 

---
# Watt Counts: Energy-Aware Benchmark for Sustainable LLM Inference on Heterogeneous GPU Architectures 

**Authors**: Mauricio Fadel Argerich, Jonathan Fürst, Marta Patiño-Martínez  

**Link**: [PDF](https://arxiv.org/pdf/2604.09048)  

**Abstract**: While the large energy consumption of Large Language Models (LLMs) is recognized by the community, system operators lack guidance for energy-efficient LLM inference deployments that leverage energy trade-offs of heterogeneous hardware due to a lack of energy-aware benchmarks and data. In this work we address this gap with Watt Counts: the largest open-access dataset of energy consumption of LLMs, with over 5,000 experiments for 50 LLMs across 10 NVIDIA Graphics Processing Units (GPUs) in batch and server scenarios along with a reproducible, open-source benchmark that enables community submissions to expand this dataset. Leveraging this dataset, we conduct a system-level study of LLM inference across heterogeneous GPU architectures and show that GPU selection is crucial for energy efficiency outcomes and that optimal hardware choices vary significantly across models and deployment scenarios, demonstrating the critical importance of hardware-aware deployment in heterogeneous LLM systems. Guided by our data and insights, we show that practitioners can reduce energy consumption by up to 70% in server scenarios with negligible impact on user experience, and by up to 20% in batch scenarios. 

---
# Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Images via Visual Prompt Injection 

**Authors**: Zedian Shao, Hongbin Liu, Yuepeng Hu, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2604.09024)  

**Abstract**: Multi-modal large language models (MLLMs) have emerged as powerful tools for analyzing Internet-scale image data, offering significant benefits but also raising critical safety and societal concerns. In particular, open-weight MLLMs may be misused to extract sensitive information from personal images at scale, such as identities, locations, or other private details. In this work, we propose ImageProtector, a user-side method that proactively protects images before sharing by embedding a carefully crafted, nearly imperceptible perturbation that acts as a visual prompt injection attack on MLLMs. As a result, when an adversary analyzes a protected image with an MLLM, the MLLM is consistently induced to generate a refusal response such as "I'm sorry, I can't help with that request." We empirically demonstrate the effectiveness of ImageProtector across six MLLMs and four datasets. Additionally, we evaluate three potential countermeasures, Gaussian noise, DiffPure, and adversarial training, and show that while they partially mitigate the impact of ImageProtector, they simultaneously degrade model accuracy and/or efficiency. Our study focuses on the practically important setting of open-weight MLLMs and large-scale automated image analysis, and highlights both the promise and the limitations of perturbation-based privacy protection. 

---
# Noise-Aware In-Context Learning for Hallucination Mitigation in ALLMs 

**Authors**: Qixuan Huang, Khalid Zaman, Masashi Unoki  

**Link**: [PDF](https://arxiv.org/pdf/2604.09021)  

**Abstract**: Auditory large language models (ALLMs) have demonstrated strong general capabilities in audio understanding and reasoning tasks. However, their reliability is still undermined by hallucination issues. Existing hallucination evaluation methods are formulated as binary classification tasks, which are insufficient to characterize the more complex hallucination patterns that arise in generative tasks. Moreover, current hallucination mitigation strategies rely on fine-tuning, resulting in high computational costs. To address the above limitations, we propose a plug-and-play Noise-Aware In-Context Learning (NAICL) method. Specifically, we construct a noise prior library, retrieve noise examples relevant to the input audio, and incorporate them as contextual priors, thereby guiding the model to reduce speculative associations when acoustic evidence is insufficient and to adopt a more conservative generation strategy. In addition, we establish a hallucination benchmark for audio caption tasks including the construction of the Clotho-1K multi-event benchmark dataset, the definition of four types of auditory hallucinations, and the introduction of metrics such as hallucination type distribution to support fine-grained analysis. Experimental results show that all evaluated ALLMs exhibit same hallucination behaviors. Moreover, the proposed NAICL method reduces the overall hallucination rate from 26.53% to 16.98%. 

---
# PinpointQA: A Dataset and Benchmark for Small Object-Centric Spatial Understanding in Indoor Videos 

**Authors**: Zhiyu Zhou, Peilin Liu, Ruoxuan Zhang, Luyang Zhang, Cheng Zhang, Hongxia Xie, Wen-Huang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.08991)  

**Abstract**: Small object-centric spatial understanding in indoor videos remains a significant challenge for multimodal large language models (MLLMs), despite its practical value for object search and assistive applications. Although existing benchmarks have advanced video spatial intelligence, embodied reasoning, and diagnostic perception, no existing benchmark directly evaluates whether a model can localize a target object in video and express its position with sufficient precision for downstream use. In this work, we introduce PinpointQA, the first dataset and benchmark for small object-centric spatial understanding in indoor videos. Built from ScanNet++ and ScanNet200, PinpointQA comprises 1,024 scenes and 10,094 QA pairs organized into four progressively challenging tasks: Target Presence Verification (TPV), Nearest Reference Identification (NRI), Fine-Grained Spatial Description (FSD), and Structured Spatial Prediction (SSP). The dataset is built from intermediate spatial representations, with QA pairs generated automatically and further refined through quality control. Experiments on representative MLLMs reveal a consistent capability gap along the progressive chain, with SSP remaining particularly difficult. Supervised fine-tuning on PinpointQA yields substantial gains, especially on the harder tasks, demonstrating that PinpointQA serves as both a diagnostic benchmark and an effective training dataset. The dataset and project page are available at this https URL. 

---
# Skill-Conditioned Visual Geolocation for Vision-Language 

**Authors**: Chenjie Yang, Yutian Jiang, Chenyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09025)  

**Abstract**: Vision-language models (VLMs) have shown a promising ability in image geolocation, but they still lack structured geographic reasoning and the capacity for autonomous self-evolution. Existing methods predominantly rely on implicit parametric memory, which often exploits outdated knowledge and generates hallucinated reasoning. Furthermore, current inference is a "one-off" process, lacking the feedback loops necessary for self-evolution based on reasoning outcomes. To address these issues, we propose GeoSkill, a training-free framework based on an evolving Skill-Graph. We first initialize the graph by refining human expert trajectories into atomic, natural-language skills. For execution, GeoSkill employs an inference model to perform direct reasoning guided by the current Skill-Graph. For continuous growth, an Autonomous Evolution mechanism leverages a larger model to conduct multiple reasoning rollouts on image-coordinate pairs sourced from web-scale data and verified real-world reasoning. By analyzing both successful and failed trajectories from these rollouts, the mechanism iteratively synthesizes and prunes skills, effectively expanding the Skill-Graph and correcting geographic biases without any parameter updates. Experiments demonstrate that GeoSkill achieves promising performance in both geolocation accuracy and reasoning faithfulness on GeoRC, while maintaining superior generalization across diverse external datasets. Furthermore, our autonomous evolution fosters the emergence of novel, verifiable skills, significantly enhancing the system's cognition of real-world geographic knowledge beyond isolated case studies. 

---
# HM-Bench: A Comprehensive Benchmark for Multimodal Large Language Models in Hyperspectral Remote Sensing 

**Authors**: Xinyu Zhang, Zurong Mai, Qingmei Li, Zjin Liao, Yibin Wen, Yuhang Chen, Xiaoya Fan, Chan Tsz Ho, Bi Tianyuan, Haoyuan Liang, Ruifeng Su, Zihao Qian, Juepeng Zheng, Jianxi Huang, Yutong Lu, Haohuan Fu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08884)  

**Abstract**: While multimodal large language models (MLLMs) have made significant strides in natural image understanding, their ability to perceive and reason over hyperspectral image (HSI) remains underexplored, which is a vital modality in remote sensing. The high dimensionality and intricate spectral-spatial properties of HSI pose unique challenges for models primarily trained on RGB this http URL address this gap, we introduce Hyperspectral Multimodal Benchmark (HM-Bench), the first benchmark designed specifically to evaluate MLLMs in HSI understanding. We curate a large-scale dataset of 19,337 question-answer pairs across 13 task categories, ranging from basic perception to spectral reasoning. Given that existing MLLMs are not equipped to process raw hyperspectral cubes natively, we propose a dual-modality evaluation framework that transforms HSI data into two complementary representations: PCA-based composite images and structured textual reports. This approach facilitates a systematic comparison of different representation for model performance. Extensive evaluations on 18 representative MLLMs reveal significant difficulties in handling complex spatial-spectral reasoning tasks. Furthermore, our results demonstrate that visual inputs generally outperform textual inputs, highlighting the importance of grounding in spectral-spatial evidence for effective HSI understanding. Dataset and appendix can be accessed at this https URL. 

---
# AudioGuard: Toward Comprehensive Audio Safety Protection Across Diverse Threat Models 

**Authors**: Mintong Kang, Chen Fang, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.08867)  

**Abstract**: Audio has rapidly become a primary interface for foundation models, powering real-time voice assistants. Ensuring safety in audio systems is inherently more complex than just "unsafe text spoken aloud": real-world risks can hinge on audio-native harmful sound events, speaker attributes (e.g., child voice), impersonation/voice-cloning misuse, and voice-content compositional harms, such as child voice plus sexual content. The nature of audio makes it challenging to develop comprehensive benchmarks or guardrails against this unique risk landscape. To close this gap, we conduct large-scale red teaming on audio systems, systematically uncover vulnerabilities in audio, and develop a comprehensive, policy-grounded audio risk taxonomy and AudioSafetyBench, the first policy-based audio safety benchmark across diverse threat models. AudioSafetyBench supports diverse languages, suspicious voices (e.g., celebrity/impersonation and child voice), risky voice-content combinations, and non-speech sound events. To defend against these threats, we propose AudioGuard, a unified guardrail consisting of 1) SoundGuard for waveform-level audio-native detection and 2) ContentGuard for policy-grounded semantic protection. Extensive experiments on AudioSafetyBench and four complementary benchmarks show that AudioGuard consistently improves guardrail accuracy over strong audio-LLM-based baselines with substantially lower latency. 

---
# Aligned Agents, Biased Swarm: Measuring Bias Amplification in Multi-Agent Systems 

**Authors**: Keyu Li, Jin Gao, Dequan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08963)  

**Abstract**: While Multi-Agent Systems (MAS) are increasingly deployed for complex workflows, their emergent properties-particularly the accumulation of bias-remain poorly understood. Because real-world MAS are too complex to analyze entirely, evaluating their ethical robustness requires first isolating their foundational mechanics. In this work, we conduct a baseline empirical study investigating how basic MAS topologies and feedback loops influence prejudice. Contrary to the assumption that multi-agent collaboration naturally dilutes bias, we hypothesize that structured workflows act as echo chambers, amplifying minor stochastic biases into systemic polarization. To evaluate this, we introduce Discrim-Eval-Open, an open-ended benchmark that bypasses individual model neutrality through forced comparative judgments across demographic groups. Analyzing bias cascades across various structures reveals that architectural sophistication frequently exacerbates bias rather than mitigating it. We observe systemic amplification even when isolated agents operate neutrally, and identify a 'Trigger Vulnerability' where injecting purely objective context drastically accelerates polarization. By stripping away advanced swarm complexity to study foundational dynamics, we establish a crucial baseline: structural complexity does not guarantee ethical robustness. Our code is available at this https URL. 

---
# Scrapyard AI 

**Authors**: Marc Böhlen, Sai Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2604.08803)  

**Abstract**: This paper considers AI model churn as an opportunity for frugal investigation of large AI models. It describes how the incessant push for ever more powerful AI systems leaves in its wake a collection of obsolete yet powerful AI models, discarded in a veritable scrapyard of AI production. This scrapyard offers a potent opportunity for resource-constrained experimentation into AI systems. As in the physical scrapyard, nothing ever truly disappears in the AI scrapyard, it is just waiting to be reconfigured into something else. Project Nudge-x is an example of what can emerge from the AI scrapyard. Nudge-x seeks to manipulate legacy AI models to describe how mining sites across the planet are impacting landscapes and lives. By sharing this collection of brutal landscape interventions with people and AI systems alike, Nudge-x creates a venue for the appreciation of a history sadly shared between AI and people. 

---
# Demystifying the Silence of Correctness Bugs in PyTorch Compiler 

**Authors**: Meiziniu Li, Dongze Li, Jianmeng Liu, Shing-Chi Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2604.08720)  

**Abstract**: Performance optimization of AI infrastructure is key to the fast adoption of large language models (LLMs). The PyTorch compiler (this http URL), a core optimization tool for deep learning (DL) models (including LLMs), has received due attention. However, this http URL is prone to correctness bugs, which cause incorrect outputs of compiled DL models without triggering exceptions, crashes, or warnings. These bugs pose a serious threat to the reliability of downstream LLM applications. Data from the PyTorch community shows that 19.2% of high-priority issues are incorrect outputs of compiled DL models induced by this http URL bugs, the second-most-common bug category (only behind program crashes at 19.57%). However, no systematic study has been conducted to specifically characterize and thereby detect these bugs. In this paper, we present the first empirical study of the correctness bugs in this http URL, examine their characteristics, and assess the effectiveness of existing fuzzers in detecting them. Based on our findings, we propose a proof-of-concept testing technique named AlignGuard, tailored specifically for detecting correctness bugs in this http URL. AlignGuard incorporates bug characteristics distilled from our empirical study, applying LLM-based test mutation to existing test cases for correctness bug detection. At the time of writing, AlignGuard has successfully detected 23 new correctness bugs in recent this http URL. All these bugs have been confirmed or fixed by the PyTorch development team, and over half (14/23) of them are even marked as high-priority bugs, underscoring the usefulness of our technique. 

---
# LMGenDrive: Bridging Multimodal Understanding and Generative World Modeling for End-to-End Driving 

**Authors**: Hao Shao, Letian Wang, Yang Zhou, Yuxuan Hu, Zhuofan Zong, Steven L. Waslander, Wei Zhan, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.08719)  

**Abstract**: Recent years have seen remarkable progress in autonomous driving, yet generalization to long-tail and open-world scenarios remains a major bottleneck for large-scale deployment. To address this challenge, some works use LLMs and VLMs for vision-language understanding and reasoning, enabling vehicles to interpret rare and safety-critical situations when generating actions. Others study generative world models to capture the spatio-temporal evolution of driving scenes, allowing agents to imagine possible futures before acting. Inspired by human intelligence, which unifies understanding and imagination, we explore a unified model for autonomous driving. We present LMGenDrive, the first framework that combines LLM-based multimodal understanding with generative world models for end-to-end closed-loop driving. Given multi-view camera inputs and natural-language instructions, LMGenDrive generates both future driving videos and control signals. This design provides complementary benefits: video prediction improves spatio-temporal scene modeling, while the LLM contributes strong semantic priors and instruction grounding from large-scale pretraining. We further propose a progressive three-stage training strategy, from vision pretraining to multi-step long-horizon driving, to improve stability and performance. LMGenDrive supports both low-latency online planning and autoregressive offline video generation. Experiments show that it significantly outperforms prior methods on challenging closed-loop benchmarks, with clear gains in instruction following, spatio-temporal understanding, and robustness to rare scenarios. These results suggest that unifying multimodal understanding and generation is a promising direction for more generalizable and robust embodied decision-making systems. 

---
# Evidential Transformation Network: Turning Pretrained Models into Evidential Models for Post-hoc Uncertainty Estimation 

**Authors**: Yongchan Chun, Chanhee Park, Jeongho Yoon, Jaehyung Seo, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2604.08627)  

**Abstract**: Pretrained models have become standard in both vision and language, yet they typically do not provide reliable measures of confidence. Existing uncertainty estimation methods, such as deep ensembles and MC dropout, are often too computationally expensive to deploy in practice. Evidential Deep Learning (EDL) offers a more efficient alternative, but it requires models to be trained to output evidential quantities from the start, which is rarely true for pretrained networks. To enable EDL-style uncertainty estimation in pretrained models, we propose the Evidential Transformation Network (ETN), a lightweight post-hoc module that converts a pretrained predictor into an evidential model. ETN operates in logit space: it learns a sample-dependent affine transformation of the logits and interprets the transformed outputs as parameters of a Dirichlet distribution for uncertainty estimation. We evaluate ETN on image classification and large language model question-answering benchmarks under both in-distribution and out-of-distribution settings. ETN consistently improves uncertainty estimation over post-hoc baselines while preserving accuracy and adding only minimal computational overhead. 

---
# MARINER: A 3E-Driven Benchmark for Fine-Grained Perception and Complex Reasoning in Open-Water Environments 

**Authors**: Xingming Liao, Ning Chen, Muying Shu, Yunpeng Yin, Peijian Zeng, Zhuowei Wang, Nankai Lin, Lianglun Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.08615)  

**Abstract**: Fine-grained visual understanding and high-level reasoning in real-world open-water environments remain under-explored due to the lack of dedicated benchmarks. We introduce MARINER, a comprehensive benchmark built under the novel Entity-Environment-Event (3E) paradigm. MARINER contains 16,629 multi-source maritime images with 63 fine-grained vessel categories, diverse adverse environments, and 5 typical dynamic maritime incidents, covering fine-grained classification, object detection, and visual question answering tasks. We conduct extensive evaluations on mainstream Multimodal Large language models (MLLMs) and establish baselines, revealing that even advanced models struggle with fine-grained discrimination and causal reasoning in complex marine scenes. As a dedicated maritime benchmark, MARINER fills the gap of realistic and cognitive-level evaluation for maritime multimodal understanding, and promotes future research on robust vision-language models for open-water applications. Appendix and supplementary materials are available at this https URL. 

---
# Semantic Intent Fragmentation: A Single-Shot Compositional Attack on Multi-Agent AI Pipelines 

**Authors**: Tanzim Ahad, Ismail Hossain, Md Jahangir Alam, Sai Puppala, Yoonpyo Lee, Syed Bahauddin Alam, Sajedul Talukder  

**Link**: [PDF](https://arxiv.org/pdf/2604.08608)  

**Abstract**: We introduce Semantic Intent Fragmentation (SIF), an attack class against LLM orchestration systems where a single, legitimately phrased request causes an orchestrator to decompose a task into subtasks that are individually benign but jointly violate security policy. Current safety mechanisms operate at the subtask level, so each step clears existing classifiers -- the violation only emerges at the composed plan. SIF exploits OWASP LLM06:2025 through four mechanisms: bulk scope escalation, silent data exfiltration, embedded trigger deployment, and quasi-identifier aggregation, requiring no injected content, no system modification, and no attacker interaction after the initial request. We construct a three-stage red-teaming pipeline grounded in OWASP, MITRE ATLAS, and NIST frameworks to generate realistic enterprise scenarios. Across 14 scenarios spanning financial reporting, information security, and HR analytics, a GPT-20B orchestrator produces policy-violating plans in 71% of cases (10/14) while every subtask appears benign. Three independent signals validate this: deterministic taint analysis, chain-of-thought evaluation, and a cross-model compliance judge with 0% false positives. Stronger orchestrators increase SIF success rates. Plan-level information-flow tracking combined with compliance evaluation detects all attacks before execution, showing the compositional safety gap is closable. 

---
# Extrapolating Volition with Recursive Information Markets 

**Authors**: Abhimanyu Pallavi Sudhir, Long Tran-Thanh  

**Link**: [PDF](https://arxiv.org/pdf/2604.08606)  

**Abstract**: One of the impediments to the efficiency of information markets is the inherent information asymmetry present in them, exacerbated by the "buyer's inspection paradox" (the buyer cannot mitigate the asymmetry by "inspecting" the information, because in doing so the buyer obtains the information without paying for it). Previous work has suggested that using Large Language Model (LLM) buyers to inspect and purchase information could overcome this information asymmetry, as an LLM buyer can simply "forget" the information it inspects. In this work, we analyze this mechanism formally through a "value-of-information" paradigm, i.e. whether it incentivizes information to be priced and provided in accordance with its "true value". We focus in particular on our new recursive version of the mechanism, which we believe has a range of applications including in AI alignment research, where it is related to Extrapolated Volition and Scalable Oversight. 

---
# TiAb Review Plugin: A Browser-Based Tool for AI-Assisted Title and Abstract Screening 

**Authors**: Yuki Kataoka, Masahiro Banno, Michihito Kyo, Shuri Nakao, Tomoo Sato, Shunsuke Taito, Tomohiro Takayama, Takahiro Tsuge, Yasushi Tsujimoto, Ryuhei So, Toshi A. Furukawa  

**Link**: [PDF](https://arxiv.org/pdf/2604.08602)  

**Abstract**: Background: Server-based screening tools impose subscription costs, while open-source alternatives require coding skills. Objectives: We developed a browser extension that provides no-code, serverless artificial intelligence (AI)-assisted title and abstract screening and examined its functionality. Methods: TiAb Review Plugin is an open-source Chrome browser extension (available at this https URL). It uses Google Sheets as a shared database, requiring no dedicated server and enabling multi-reviewer collaboration. Users supply their own Gemini API key, stored locally and encrypted. The tool offers three screening modes: manual review, large language model (LLM) batch screening, and machine learning (ML) active learning. For ML evaluation, we re-implemented the default ASReview active learning algorithm (TF-IDF with Naive Bayes) in TypeScript to enable in-browser execution, and verified equivalence against the original Python implementation using 10-fold cross-validation on six datasets. For LLM evaluation, we compared 16 parameter configurations across two model families on a benchmark dataset, then validated the optimal configuration (Gemini 3.0 Flash, low thinking budget, TopP=0.95) with a sensitivity-oriented prompt on five public datasets (1,038 to 5,628 records, 0.5 to 2.0 percent prevalence). Results: The TypeScript classifier produced top-100 rankings 100 percent identical to the original ASReview across all six datasets. For LLM screening, recall was 94 to 100 percent with precision of 2 to 15 percent, and Work Saved over Sampling at 95 percent recall (WSS@95) ranged from 48.7 to 87.3 percent. Conclusions: We developed a functional browser extension that integrates LLM screening and ML active learning into a no-code, serverless environment, ready for practical use in systematic review screening. 

---
# STIndex: A Context-Aware Multi-Dimensional Spatiotemporal Information Extraction System 

**Authors**: Wenxiao Zhang, Yu Liu, Qiang sun, Yihao Ding, Sirui Li, Yanbing Liu, Jin B. Hong, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08597)  

**Abstract**: Extracting structured knowledge from unstructured data still faces practical limitations: entity and event extraction pipelines remain brittle, knowledge graph construction requires costly ontology engineering, and cross-domain generalization is rarely production-ready. In contrast, space and time provide universal contextual anchors that naturally align heterogeneous information and benefit downstream tasks such as retrieval and reasoning. We introduce \textbf{STIndex}, an end-to-end system that structures unstructured content into a multidimensional spatiotemporal data warehouse. Users define domain-specific analysis dimensions with configurable hierarchies, while large language models perform context-aware extraction and grounding. \textbf{STIndex} integrates document-level memory, geocoding correction, and quality validation, and offers an interactive analytics dashboard for visualization, clustering, burst detection, and entity network analysis. In evaluation on a public health benchmark, \textbf{STIndex} improves spatiotemporal entity extraction F1 by 4.37\% (GPT-4o-mini) and 3.60\% (Qwen3-8B). A live demonstration and open-source code are available at this https URL. 

---
# Act or Escalate? Evaluating Escalation Behavior in Automation with Language Models 

**Authors**: Matthew DosSantos DiSorbo, Harang Ju  

**Link**: [PDF](https://arxiv.org/pdf/2604.08588)  

**Abstract**: Effective automation hinges on deciding when to act and when to escalate. We model this as a decision under uncertainty: an LLM forms a prediction, estimates its probability of being correct, and compares the expected costs of acting and escalating. Using this framework across five domains of recorded human decisions-demand forecasting, content recommendation, content moderation, loan approval, and autonomous driving-and across multiple model families, we find marked differences in the implicit thresholds models use to trade off these costs. These thresholds vary substantially and are not predicted by architecture or scale, while self-estimates are miscalibrated in model-specific ways. We then test interventions that target this decision process by varying cost ratios, providing accuracy signals, and training models to follow the desired escalation rule. Prompting helps mainly for reasoning models. SFT on chain-of-thought targets yields the most robust policies, which generalize across datasets, cost ratios, prompt framings, and held-out domains. These results suggest that escalation behavior is a model-specific property that should be characterized before deployment, and that robust alignment benefits from training models to reason explicitly about uncertainty and decision costs. 

---
# CSAttention: Centroid-Scoring Attention for Accelerating LLM Inference 

**Authors**: Chuxu Song, Zhencan Peng, Jiuqi Wei, Chuanhui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.08584)  

**Abstract**: Long-context LLMs increasingly rely on extended, reusable prefill prompts for agents and domain Q&A, pushing attention and KV-cache to become the dominant decode-time bottlenecks. While sparse attention reduces computation and transfer costs, it often struggles to maintain accuracy at high sparsity levels due to the inherent distribution shift between Queries and Keys. We propose Centroid-Scoring Attention (CSAttention), a training-free sparse attention method optimized for high-throughput serving of reusable contexts. CSAttention adopts a storage-for-computation strategy tailored to the offline-prefill/online-decode setting: it front-loads computation into a one-time offline prefill phase that can be amortized across multiple queries, while aggressively optimizing per-step decoding latency. Specifically, CSAttention constructs query-centric lookup tables during offline prefill, whose size remains fixed during decoding, and enables online decoding to replace full-context scans with efficient table lookups and GPU-friendly score accumulation. Extensive experiments demonstrate that CSAttention achieves near-identical accuracy to full attention. Under high sparsity (95%) and long-context settings (32K-128K), CSAttention consistently outperforms state-of-the-art sparse attention methods in both model accuracy and inference speed, achieving up to 4.6x inference speedup over the most accurate baseline at a context length of 128K. 

---
# 3D-VCD: Hallucination Mitigation in 3D-LLM Embodied Agents through Visual Contrastive Decoding 

**Authors**: Makanjuola Ogunleye, Eman Abdelrahman, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2604.08645)  

**Abstract**: Large multimodal models are increasingly used as the reasoning core of embodied agents operating in 3D environments, yet they remain prone to hallucinations that can produce unsafe and ungrounded decisions. Existing inference-time hallucination mitigation methods largely target 2D vision-language settings and do not transfer to embodied 3D reasoning, where failures arise from object presence, spatial layout, and geometric grounding rather than pixel-level inconsistencies. We introduce 3D-VCD, the first inference-time visual contrastive decoding framework for hallucination mitigation in 3D embodied agents. 3D-VCD constructs a distorted 3D scene graph by applying semantic and geometric perturbations to object-centric representations, such as category substitutions and coordinate or extent corruption. By contrasting predictions under the original and distorted 3D contexts, our method suppresses tokens that are insensitive to grounded scene evidence and are therefore likely driven by language priors. We evaluate 3D-VCD on the 3D-POPE and HEAL benchmarks and show that it consistently improves grounded reasoning without any retraining, establishing inference-time contrastive decoding over structured 3D representations as an effective and practical route to more reliable embodied intelligence. 

---
# Distributionally Robust Token Optimization in RLHF 

**Authors**: Yeping Jin, Jiaming Hu, Ioannis Ch. Paschalidis  

**Link**: [PDF](https://arxiv.org/pdf/2604.08577)  

**Abstract**: Large Language Models (LLMs) tend to respond correctly to prompts that align to the data they were trained and fine-tuned on. Yet, small shifts in wording, format, or language can trigger surprisingly large failures, especially on multi-step reasoning problems. To address this problem, we propose a Distributionally Robust Token Optimization (DRTO) approach, which combines token-level Reinforcement Learning from Human Feedback (RLHF) with Distributionally Robust Optimization (DRO). DRTO bounds worst case token-wise rewards by constructing an f-divergence ambiguity set over a loss minibatch, leading to a theoretical robustness. Empirically, DRTO enhances consistency under distribution shifts in mathematical reasoning benchmarks, achieving 9.17\% improvement on GSM8K and 2.49% improvement on MathQA. 

---
# Detection of Hate and Threat in Digital Forensics: A Case-Driven Multimodal Approach 

**Authors**: Ponkoj Chandra Shill  

**Link**: [PDF](https://arxiv.org/pdf/2604.08609)  

**Abstract**: Digital forensic investigations increasingly rely on heterogeneous evidence such as images, scanned documents, and contextual reports. These artifacts may contain explicit or implicit expressions of harm, hate, threat, violence, or intimidation, yet existing automated approaches often assume clean text input or apply vision models without forensic justification. This paper presents a case-driven multimodal approach for hate and threat detection in forensic analysis. The proposed framework explicitly determines the presence and source of textual evidence, distinguishing between embedded text, associated contextual text, and image-only evidence. Based on the identified evidence configuration, the framework selectively applies text analysis, multimodal fusion, or image-only semantic reasoning using vision language models with vision transformer backbones (ViT). By conditioning inference on evidence availability, the approach mirrors forensic decision-making, improves evidentiary traceability, and avoids unjustified modality assumptions. Experimental evaluation on forensic-style image evidence demonstrates consistent and interpretable behavior across heterogeneous evidence scenarios. 

---
# QCFuse: Query-Centric Cache Fusion for Efficient RAG Inference 

**Authors**: Jianxin Yan, Zeheng Qian, Wangze Ni, Zhitao Shen, Zhiping Wang, Haoyang Li, Jia Zhu, Lei Chen, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.08585)  

**Abstract**: Cache fusion accelerates generation process of LLMs equipped with RAG through KV caching and selective token recomputation, thereby reducing computational costs and improving efficiency. However, existing methods primarily rely on local perspectives for token selection and lack global awareness from the user query. Utilizing this global awareness is challenging due to the high cost of obtaining context-aware query representations and the strict pipeline constraints required for efficient attention analysis. Thus, this demonstration introduces QCFuse, an innovative KV cache fusion system centered on the user query. QCFuse leverages semantic summary anchors to enhance query representations and selectively recomputes query-related tokens to improve accuracy, updating tokens based on the attention distribution of the most critical Transformer layer to preserve the high efficiency of the pipeline structure. Evaluations on real-world datasets demonstrate that QCFuse significantly improves the response efficiency of LLMs by 40\% while maintaining equivalent accuracy compared to current methods. Additionally, in certain scenarios, QCFuse achieves an attention denoising effect that yields higher response accuracy, demonstrating substantial potential in the optimization of LLM inference. 

---
# AlphaLab: Autonomous Multi-Agent Research Across Optimization Domains with Frontier LLMs 

**Authors**: Brendan R. Hogan, Xiwen Chen, James T. Wilson, Kashif Rasul, Adel Boyarsky, Thomas Kamei, Anderson Schneider, Yuriy Nevmyvaka  

**Link**: [PDF](https://arxiv.org/pdf/2604.08590)  

**Abstract**: We present AlphaLab, an autonomous research harness that leverages frontier LLM agentic capabilities to automate the full experimental cycle in quantitative, computation-intensive domains. Given only a dataset and a natural-language objective, AlphaLab proceeds through three phases without human intervention: (1) it adapts to the domain and explores the data, writing analysis code and producing a research report; (2) it constructs and adversarially validates its own evaluation framework; and (3) it runs large-scale GPU experiments via a Strategist/Worker loop, accumulating domain knowledge in a persistent playbook that functions as a form of online prompt optimization. All domain-specific behavior is factored into adapters generated by the model itself, so the same pipeline handles qualitatively different tasks without modification. We evaluate AlphaLab with two frontier LLMs (GPT-5.2 and Claude Opus 4.6) on three domains: CUDA kernel optimization, where it writes GPU kernels that run 4.4x faster than this http URL on average (up to 91x); LLM pretraining, where the full system achieves 22% lower validation loss than a single-shot baseline using the same model; and traffic forecasting, where it beats standard baselines by 23-25% after researching and implementing published model families from the literature. The two models discover qualitatively different solutions in every domain (neither dominates uniformly), suggesting that multi-model campaigns provide complementary search coverage. We additionally report results on financial time series forecasting in the appendix, and release all code at this https URL. 

---
# Structured Exploration and Exploitation of Label Functions for Automated Data Annotation 

**Authors**: Phong Lam, Ha-Linh Nguyen, Thu-Trang Nguyen, Son Nguyen, Hieu Dinh Vo  

**Link**: [PDF](https://arxiv.org/pdf/2604.08578)  

**Abstract**: High-quality labeled data is critical for training reliable machine learning and deep learning models, yet manual annotation remains costly and error-prone. Programmatic labeling addresses this challenge by using label functions (LFs), i.e., heuristic rules that automatically generate weak labels for training datasets. However, existing automated LF generation methods either rely on large language models (LLMs) to synthesize surface-level heuristics or employ model-based synthesis over hand-crafted primitives. These approaches often result in limited coverage and unreliable label quality. In this paper, we introduce EXPONA, an automated framework for programmatic labeling that formulates LF generation as a principled process balancing diversity and reliability. EXPONA systematically explores multi-level LFs, spanning surface, structural, and semantic perspectives. EXPONA further applies reliability-aware mechanisms to suppress noisy or redundant heuristics while preserving complementary signals. To evaluate EXPONA, we conducted extensive experiments on eleven classification datasets across diverse domains. Experimental results show that EXPONA consistently outperformed state-of-the-art automated LF generation methods. Specifically, EXPONA achieved nearly complete label coverage (up to 98.9%), improved weak label quality by up to 87%, and yielded downstream performance gains of up to 46% in weighted F1. These results indicate that EXPONA's combination of multi-level LF exploration and reliability-aware filtering enabled more consistent label quality and downstream performance across diverse tasks by balancing coverage and precision in the generated LF set. 

---
# QuanBench+: A Unified Multi-Framework Benchmark for LLM-Based Quantum Code Generation 

**Authors**: Ali Slim, Haydar Hamieh, Jawad Kotaich, Yehya Ghosn, Mahdi Chehimi, Ammar Mohanna, Hasan Abed Al Kader Hammoud, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2604.08570)  

**Abstract**: Large Language Models (LLMs) are increasingly used for code generation, yet quantum code generation is still evaluated mostly within single frameworks, making it difficult to separate quantum reasoning from framework familiarity. We introduce QuanBench+, a unified benchmark spanning Qiskit, PennyLane, and Cirq, with 42 aligned tasks covering quantum algorithms, gate decomposition, and state preparation.
We evaluate models with executable functional tests, report Pass@1 and Pass@5, and use KL-divergence-based acceptance for probabilistic outputs. We additionally study Pass@1 after feedback-based repair, where a model may revise code after a runtime error or wrong answer. Across frameworks, the strongest one-shot scores reach 59.5% in Qiskit, 54.8% in Cirq, and 42.9% in PennyLane; with feedback-based repair, the best scores rise to 83.3%, 76.2%, and 66.7%, respectively. These results show clear progress, but also that reliable multi-framework quantum code generation remains unsolved and still depends strongly on framework-specific knowledge. 

---
# Towards Real-world Human Behavior Simulation: Benchmarking Large Language Models on Long-horizon, Cross-scenario, Heterogeneous Behavior Traces 

**Authors**: Jiawei Chen, Ruoxi Xu, Boxi Cao, Ruotong Pan, Yunfei Zhang, Yifei Hu, Yong Du, Tingting Gao, Yaojie Lu, Yingfei Sun, Xianpei Han, Le Sun, Xiangyu Wu, Hongyu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.08362)  

**Abstract**: The emergence of Large Language Models (LLMs) has illuminated the potential for a general-purpose user simulator. However, existing benchmarks remain constrained to isolated scenarios, narrow action spaces, or synthetic data, failing to capture the holistic nature of authentic human behavior. To bridge this gap, we introduce OmniBehavior, the first user simulation benchmark constructed entirely from real-world data, integrating long-horizon, cross-scenario, and heterogeneous behavioral patterns into a unified framework. Based on this benchmark, we first provide empirical evidence that previous datasets with isolated scenarios suffer from tunnel vision, whereas real-world decision-making relies on long-term, cross-scenario causal chains. Extensive evaluations of state-of-the-art LLMs reveal that current models struggle to accurately simulate these complex behaviors, with performance plateauing even as context windows expand. Crucially, a systematic comparison between simulated and authentic behaviors uncovers a fundamental structural bias: LLMs tend to converge toward a positive average person, exhibiting hyper-activity, persona homogenization, and a Utopian bias. This results in the loss of individual differences and long-tail behaviors, highlighting critical directions for future high-fidelity simulation research. 

---
# Automated Standardization of Legacy Biomedical Metadata Using an Ontology-Constrained LLM Agent 

**Authors**: Josef Hardi, Martin J. O'Connor, Marcos Martinez-Romero, Jean G. Rosario, Stephen A. Fisher, Mark A. Musen  

**Link**: [PDF](https://arxiv.org/pdf/2604.08552)  

**Abstract**: Scientific metadata are often incomplete and noncompliant with community standards, limiting dataset findability, interoperability, and reuse. When reporting guidelines exist, they typically lack machine-actionable representations. Producing FAIR datasets requires encoding metadata standards as machine-actionable templates with rich field specifications and precise value constraints. Recent work has shown that LLMs guided by field names and ontology constraints can improve metadata standardization, but these approaches treat constraints as static text prompts, relying on the model's training knowledge alone. We present an LLM-based metadata standardization system that queries authoritative biomedical terminology services in real time to retrieve canonically correct vocabulary terms on demand. We evaluate this approach on 839 legacy metadata records from the Human BioMolecular Atlas Program (HuBMAP) using an expert-curated gold standard for exact-match assessment. Our evaluation shows that augmenting the LLM with real-time tool access consistently improves prediction accuracy over the LLM alone across both ontology-constrained and non-ontology-constrained fields, demonstrating a practical, scalable approach to automated standardization of biomedical metadata. 

---
