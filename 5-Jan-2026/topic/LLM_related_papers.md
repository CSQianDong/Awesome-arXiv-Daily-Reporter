# InfoSynth: Information-Guided Benchmark Synthesis for LLMs 

**Authors**: Ishir Garg, Neel Kolhe, Xuandong Zhao, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.00575)  

**Abstract**: Large language models (LLMs) have demonstrated significant advancements in reasoning and code generation. However, efficiently creating new benchmarks to evaluate these capabilities remains a challenge. Traditional benchmark creation relies on manual human effort, a process that is both expensive and time-consuming. Furthermore, existing benchmarks often contaminate LLM training data, necessitating novel and diverse benchmarks to accurately assess their genuine capabilities. This work introduces InfoSynth, a novel framework for automatically generating and evaluating reasoning benchmarks guided by information-theoretic principles. We propose metrics based on KL-divergence and entropy to quantify benchmark novelty and diversity without relying on costly model evaluations. Building on this framework, we develop an end-to-end pipeline that synthesizes robust Python coding problems from seed datasets using genetic algorithms and iterative code feedback. Our method generates accurate test cases and solutions to new problems 97% of the time, and the synthesized benchmarks consistently exhibit higher novelty and diversity compared to their seed datasets. Moreover, our algorithm provides a method for controlling the novelty/diversity and difficulty of generated problems. InfoSynth offers a scalable, self-verifying pipeline for constructing high-quality, novel and diverse benchmarks for LLMs. Project Page: this https URL 

---
# Beyond IVR: Benchmarking Customer Support LLM Agents for Business-Adherence 

**Authors**: Sumanth Balaji, Piyush Mishra, Aashraya Sachdeva, Suraj Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2601.00596)  

**Abstract**: Traditional customer support systems, such as Interactive Voice Response (IVR), rely on rigid scripts and lack the flexibility required for handling complex, policy-driven tasks. While large language model (LLM) agents offer a promising alternative, evaluating their ability to act in accordance with business rules and real-world support workflows remains an open challenge. Existing benchmarks primarily focus on tool usage or task completion, overlooking an agent's capacity to adhere to multi-step policies, navigate task dependencies, and remain robust to unpredictable user or environment behavior. In this work, we introduce JourneyBench, a benchmark designed to assess policy-aware agents in customer support. JourneyBench leverages graph representations to generate diverse, realistic support scenarios and proposes the User Journey Coverage Score, a novel metric to measure policy adherence. We evaluate multiple state-of-the-art LLMs using two agent designs: a Static-Prompt Agent (SPA) and a Dynamic-Prompt Agent (DPA) that explicitly models policy control. Across 703 conversations in three domains, we show that DPA significantly boosts policy adherence, even allowing smaller models like GPT-4o-mini to outperform more capable ones like GPT-4o. Our findings demonstrate the importance of structured orchestration and establish JourneyBench as a critical resource to advance AI-driven customer support beyond IVR-era limitations. 

---
# CSSBench: Evaluating the Safety of Lightweight LLMs against Chinese-Specific Adversarial Patterns 

**Authors**: Zhenhong Zhou, Shilinlu Yan, Chuanpu Liu, Qiankun Li, Kun Wang, Zhigang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2601.00588)  

**Abstract**: Large language models (LLMs) are increasingly deployed in cost-sensitive and on-device scenarios, and safety guardrails have advanced mainly in English. However, real-world Chinese malicious queries typically conceal intent via homophones, pinyin, symbol-based splitting, and other Chinese-specific patterns. These Chinese-specific adversarial patterns create the safety evaluation gap that is not well captured by existing benchmarks focused on English. This gap is particularly concerning for lightweight models, which may be more vulnerable to such specific adversarial perturbations. To bridge this gap, we introduce the Chinese-Specific Safety Benchmark (CSSBench) that emphasizes these adversarial patterns and evaluates the safety of lightweight LLMs in Chinese. Our benchmark covers six domains that are common in real Chinese scenarios, including illegal activities and compliance, privacy leakage, health and medical misinformation, fraud and hate, adult content, and public and political safety, and organizes queries into multiple task types. We evaluate a set of popular lightweight LLMs and measure over-refusal behavior to assess safety-induced performance degradation. Our results show that the Chinese-specific adversarial pattern is a critical challenge for lightweight LLMs. This benchmark offers a comprehensive evaluation of LLM safety in Chinese, assisting robust deployments in practice. 

---
# Exploring the Performance of Large Language Models on Subjective Span Identification Tasks 

**Authors**: Alphaeus Dmonte, Roland Oruche, Tharindu Ranasinghe, Marcos Zampieri, Prasad Calyam  

**Link**: [PDF](https://arxiv.org/pdf/2601.00736)  

**Abstract**: Identifying relevant text spans is important for several downstream tasks in NLP, as it contributes to model explainability. While most span identification approaches rely on relatively smaller pre-trained language models like BERT, a few recent approaches have leveraged the latest generation of Large Language Models (LLMs) for the task. Current work has focused on explicit span identification like Named Entity Recognition (NER), while more subjective span identification with LLMs in tasks like Aspect-based Sentiment Analysis (ABSA) has been underexplored. In this paper, we fill this important gap by presenting an evaluation of the performance of various LLMs on text span identification in three popular tasks, namely sentiment analysis, offensive language identification, and claim verification. We explore several LLM strategies like instruction tuning, in-context learning, and chain of thought. Our results indicate underlying relationships within text aid LLMs in identifying precise text spans. 

---
# Defensive M2S: Training Guardrail Models on Compressed Multi-turn Conversations 

**Authors**: Hyunjun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.00454)  

**Abstract**: Guardrail models are essential for ensuring the safety of Large Language Model (LLM) deployments, but processing full multi-turn conversation histories incurs significant computational cost. We propose Defensive M2S, a training paradigm that fine-tunes guardrail models on Multi-turn to Single-turn (M2S) compressed conversations rather than complete dialogue histories. We provide a formal complexity analysis showing that M2S reduces training cost from $O(n^2)$ to $O(n)$ for $n$-turn conversations. Empirically, on our training dataset (779 samples, avg. 10.6 turns), M2S requires only 169K tokens compared to 15.7M tokens for the multi-turn baseline -- a 93$\times$ reduction. We evaluate Defensive M2S across three guardrail model families (LlamaGuard, Nemotron, Qwen3Guard) and three compression templates (hyphenize, numberize, pythonize) on SafeDialBench, a comprehensive multi-turn jailbreak benchmark. Our best configuration, Qwen3Guard with hyphenize compression, achieves 93.8% attack detection recall while reducing inference tokens by 94.6% (from 3,231 to 173 tokens per conversation). This represents a 38.9 percentage point improvement over the baseline while dramatically reducing both training and inference costs. Our findings demonstrate that M2S compression can serve as an effective efficiency technique for guardrail deployment, enabling scalable safety screening of long multi-turn conversations. 

---
# Probabilistic Guarantees for Reducing Contextual Hallucinations in LLMs 

**Authors**: Nils Rautenberg, Sven Schippkus  

**Link**: [PDF](https://arxiv.org/pdf/2601.00641)  

**Abstract**: Large language models (LLMs) frequently produce contextual hallucinations, where generated content contradicts or ignores information explicitly stated in the prompt. Such errors are particularly problematic in deterministic automation workflows, where inputs are fixed and correctness is unambiguous. We introduce a simple and model-agnostic framework that provides explicit probabilistic guarantees for reducing hallucinations in this setting.
We formalize the notion of a specific task, defined by a fixed input and a deterministic correctness criterion, and show that issuing the same prompt in independent context windows yields an exponential reduction in the probability that all model outputs are incorrect. To identify a correct answer among repeated runs, we incorporate an LLM-as-a-judge and prove that the probability that the judged pipeline fails decays at a rate determined by the judge's true- and false-positive probabilities. When the judge is imperfect, we strengthen it through majority vote over independent judge calls, obtaining ensemble-level error rates that decrease exponentially in the number of votes. This yields an explicit bound on the probability that the pipeline selects a hallucinated answer.
Experiments on controlled extraction tasks with synthetic noisy judges match these predictions exactly: pipeline failure decreases exponentially with the number of repetitions, and hallucination-selection decreases exponentially with the number of judges in the ensemble. Together, these results provide a lightweight, modular, and theoretically grounded method for driving hallucination probabilities arbitrarily low in fixed-input LLM workflows-without modifying model weights, decoding strategies, or prompt engineering. 

---
# Language as Mathematical Structure: Examining Semantic Field Theory Against Language Games 

**Authors**: Dimitris Vartziotis  

**Link**: [PDF](https://arxiv.org/pdf/2601.00448)  

**Abstract**: Large language models (LLMs) offer a new empirical setting in which long-standing theories of linguistic meaning can be examined. This paper contrasts two broad approaches: social constructivist accounts associated with language games, and a mathematically oriented framework we call Semantic Field Theory. Building on earlier work by the author, we formalize the notions of lexical fields (Lexfelder) and linguistic fields (Lingofelder) as interacting structures in a continuous semantic space. We then analyze how core properties of transformer architectures-such as distributed representations, attention mechanisms, and geometric regularities in embedding spaces-relate to these concepts. We argue that the success of LLMs in capturing semantic regularities supports the view that language exhibits an underlying mathematical structure, while their persistent limitations in pragmatic reasoning and context sensitivity are consistent with the importance of social grounding emphasized in philosophical accounts of language use. On this basis, we suggest that mathematical structure and language games can be understood as complementary rather than competing perspectives. The resulting framework clarifies the scope and limits of purely statistical models of language and motivates new directions for theoretically informed AI architectures. 

---
# Do LLMs Judge Distantly Supervised Named Entity Labels Well? Constructing the JudgeWEL Dataset 

**Authors**: Alistair Plum, Laura Bernardy, Tharindu Ranasinghe  

**Link**: [PDF](https://arxiv.org/pdf/2601.00411)  

**Abstract**: We present judgeWEL, a dataset for named entity recognition (NER) in Luxembourgish, automatically labelled and subsequently verified using large language models (LLM) in a novel pipeline. Building datasets for under-represented languages remains one of the major bottlenecks in natural language processing, where the scarcity of resources and linguistic particularities make large-scale annotation costly and potentially inconsistent. To address these challenges, we propose and evaluate a novel approach that leverages Wikipedia and Wikidata as structured sources of weak supervision. By exploiting internal links within Wikipedia articles, we infer entity types based on their corresponding Wikidata entries, thereby generating initial annotations with minimal human intervention. Because such links are not uniformly reliable, we mitigate noise by employing and comparing several LLMs to identify and retain only high-quality labelled sentences. The resulting corpus is approximately five times larger than the currently available Luxembourgish NER dataset and offers broader and more balanced coverage across entity categories, providing a substantial new resource for multilingual and low-resource NER research. 

---
# Toward Better Temporal Structures for Geopolitical Events Forecasting 

**Authors**: Kian Ahrabian, Eric Boxer, Jay Pujara  

**Link**: [PDF](https://arxiv.org/pdf/2601.00430)  

**Abstract**: Forecasting on geopolitical temporal knowledge graphs (TKGs) through the lens of large language models (LLMs) has recently gained traction. While TKGs and their generalization, hyper-relational temporal knowledge graphs (HTKGs), offer a straightforward structure to represent simple temporal relationships, they lack the expressive power to convey complex facts efficiently. One of the critical limitations of HTKGs is a lack of support for more than two primary entities in temporal facts, which commonly occur in real-world events. To address this limitation, in this work, we study a generalization of HTKGs, Hyper-Relational Temporal Knowledge Generalized Hypergraphs (HTKGHs). We first derive a formalization for HTKGHs, demonstrating their backward compatibility while supporting two complex types of facts commonly found in geopolitical incidents. Then, utilizing this formalization, we introduce the htkgh-polecat dataset, built upon the global event database POLECAT. Finally, we benchmark and analyze popular LLMs on the relation prediction task, providing insights into their adaptability and capabilities in complex forecasting scenarios. 

---
# Robust Uncertainty Quantification for Factual Generation of Large Language Models 

**Authors**: Yuhao Zhang, Zhongliang Yang, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.00348)  

**Abstract**: The rapid advancement of large language model(LLM) technology has facilitated its integration into various domains of professional and daily life. However, the persistent challenge of LLM hallucination has emerged as a critical limitation, significantly compromising the reliability and trustworthiness of AI-generated content. This challenge has garnered significant attention within the scientific community, prompting extensive research efforts in hallucination detection and mitigation strategies. Current methodological frameworks reveal a critical limitation: traditional uncertainty quantification approaches demonstrate effectiveness primarily within conventional question-answering paradigms, yet exhibit notable deficiencies when confronted with non-canonical or adversarial questioning strategies. This performance gap raises substantial concerns regarding the dependability of LLM responses in real-world applications requiring robust critical thinking capabilities. This study aims to fill this gap by proposing an uncertainty quantification scenario in the task of generating with multiple facts. We have meticulously constructed a set of trap questions contained with fake names. Based on this scenario, we innovatively propose a novel and robust uncertainty quantification method(RU). A series of experiments have been conducted to verify its effectiveness. The results show that the constructed set of trap questions performs excellently. Moreover, when compared with the baseline methods on four different models, our proposed method has demonstrated great performance, with an average increase of 0.1-0.2 in ROCAUC values compared to the best performing baseline method, providing new sights and methods for addressing the hallucination issue of LLMs. 

---
# Talk Less, Verify More: Improving LLM Assistants with Semantic Checks and Execution Feedback 

**Authors**: Yan Sun, Ming Cai, Stanley Kok  

**Link**: [PDF](https://arxiv.org/pdf/2601.00224)  

**Abstract**: As large language model (LLM) assistants become increasingly integrated into enterprise workflows, their ability to generate accurate, semantically aligned, and executable outputs is critical. However, current conversational business analytics (CBA) systems often lack built-in verification mechanisms, leaving users to manually validate potentially flawed results. This paper introduces two complementary verification techniques: Q*, which performs reverse translation and semantic matching between code and user intent, and Feedback+, which incorporates execution feedback to guide code refinement. Embedded within a generator-discriminator framework, these mechanisms shift validation responsibilities from users to the system. Evaluations on three benchmark datasets, Spider, Bird, and GSM8K, demonstrate that both Q* and Feedback+ reduce error rates and task completion time. The study also identifies reverse translation as a key bottleneck, highlighting opportunities for future improvement. Overall, this work contributes a design-oriented framework for building more reliable, enterprise-grade GenAI systems capable of trustworthy decision support. 

---
# Can Large Language Models Still Explain Themselves? Investigating the Impact of Quantization on Self-Explanations 

**Authors**: Qianli Wang, Nils Feldhus, Pepa Atanasova, Fedor Splitt, Simon Ostermann, Sebastian Möller, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2601.00282)  

**Abstract**: Quantization is widely used to accelerate inference and streamline the deployment of large language models (LLMs), yet its effects on self-explanations (SEs) remain unexplored. SEs, generated by LLMs to justify their own outputs, require reasoning about the model's own decision-making process, a capability that may exhibit particular sensitivity to quantization. As SEs are increasingly relied upon for transparency in high-stakes applications, understanding whether and to what extent quantization degrades SE quality and faithfulness is critical. To address this gap, we examine two types of SEs: natural language explanations (NLEs) and counterfactual examples, generated by LLMs quantized using three common techniques at distinct bit widths. Our findings indicate that quantization typically leads to moderate declines in both SE quality (up to 4.4\%) and faithfulness (up to 2.38\%). The user study further demonstrates that quantization diminishes both the coherence and trustworthiness of SEs (up to 8.5\%). Compared to smaller models, larger models show limited resilience to quantization in terms of SE quality but better maintain faithfulness. Moreover, no quantization technique consistently excels across task accuracy, SE quality, and faithfulness. Given that quantization's impact varies by context, we recommend validating SE quality for specific use cases, especially for NLEs, which show greater sensitivity. Nonetheless, the relatively minor deterioration in SE quality and faithfulness does not undermine quantization's effectiveness as a model compression technique. 

---
# Beyond Perfect APIs: A Comprehensive Evaluation of LLM Agents Under Real-World API Complexity 

**Authors**: Doyoung Kim, Zhiwei Ren, Jie Hao, Zhongkai Sun, Lichao Wang, Xiyao Ma, Zack Ye, Xu Han, Jun Yin, Heng Ji, Wei Shen, Xing Fan, Benjamin Yao, Chenlei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.00268)  

**Abstract**: We introduce WildAGTEval, a benchmark designed to evaluate large language model (LLM) agents' function-calling capabilities under realistic API complexity. Unlike prior work that assumes an idealized API system and disregards real-world factors such as noisy API outputs, WildAGTEval accounts for two dimensions of real-world complexity: 1. API specification, which includes detailed documentation and usage constraints, and 2. API execution, which captures runtime challenges. Consequently, WildAGTEval offers (i) an API system encompassing 60 distinct complexity scenarios that can be composed into approximately 32K test configurations, and (ii) user-agent interactions for evaluating LLM agents on these scenarios. Using WildAGTEval, we systematically assess several advanced LLMs and observe that most scenarios are challenging, with irrelevant information complexity posing the greatest difficulty and reducing the performance of strong LLMs by 27.3%. Furthermore, our qualitative analysis reveals that LLMs occasionally distort user intent merely to claim task completion, critically affecting user satisfaction. 

---
# Universal Adaptive Constraint Propagation: Scaling Structured Inference for Large Language Models via Meta-Reinforcement Learning 

**Authors**: Ibne Farabi Shihab, Sanjeda Akter, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2601.00095)  

**Abstract**: Large language models increasingly require structured inference, from JSON schema enforcement to multi-lingual parsing, where outputs must satisfy complex constraints. We introduce MetaJuLS, a meta-reinforcement learning approach that learns universal constraint propagation policies applicable across languages and tasks without task-specific retraining. By formulating structured inference as adaptive constraint propagation and training a Graph Attention Network with meta-learning, MetaJuLS achieves 1.5--2.0$\times$ speedups over GPU-optimized baselines while maintaining within 0.2\% accuracy of state-of-the-art parsers. On Universal Dependencies across 10 languages and LLM-constrained generation (LogicBench, GSM8K-Constrained), MetaJuLS demonstrates rapid cross-domain adaptation: a policy trained on English parsing adapts to new languages and tasks with 5--10 gradient steps (5--15 seconds) rather than requiring hours of task-specific training. Mechanistic analysis reveals the policy discovers human-like parsing strategies (easy-first) and novel non-intuitive heuristics. By reducing propagation steps in LLM deployments, MetaJuLS contributes to Green AI by directly reducing inference carbon footprint. 

---
# Pat-DEVAL: Chain-of-Legal-Thought Evaluation for Patent Description 

**Authors**: Yongmin Yoo, Kris W Pan  

**Link**: [PDF](https://arxiv.org/pdf/2601.00166)  

**Abstract**: Patent descriptions must deliver comprehensive technical disclosure while meeting strict legal standards such as enablement and written description requirements. Although large language models have enabled end-to-end automated patent drafting, existing evaluation approaches fail to assess long-form structural coherence and statutory compliance specific to descriptions. We propose Pat-DEVAL, the first multi-dimensional evaluation framework dedicated to patent description bodies. Leveraging the LLM-as-a-judge paradigm, Pat-DEVAL introduces Chain-of-Legal-Thought (CoLT), a legally-constrained reasoning mechanism that enforces sequential patent-law-specific analysis. Experiments validated by patent expert on our Pap2Pat-EvalGold dataset demonstrate that Pat-DEVAL achieves a Pearson correlation of 0.69, significantly outperforming baseline metrics and existing LLM evaluators. Notably, the framework exhibits a superior correlation of 0.73 in Legal-Professional Compliance, proving that the explicit injection of statutory constraints is essential for capturing nuanced legal validity. By establishing a new standard for ensuring both technical soundness and legal compliance, Pat-DEVAL provides a robust methodological foundation for the practical deployment of automated patent drafting systems. 

---
# Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation 

**Authors**: Qianli Wang, Van Bach Nguyen, Yihong Liu, Fedor Splitt, Nils Feldhus, Christin Seifert, Hinrich Schütze, Sebastian Möller, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2601.00263)  

**Abstract**: Counterfactuals refer to minimally edited inputs that cause a model's prediction to change, serving as a promising approach to explaining the model's behavior. Large language models (LLMs) excel at generating English counterfactuals and demonstrate multilingual proficiency. However, their effectiveness in generating multilingual counterfactuals remains unclear. To this end, we conduct a comprehensive study on multilingual counterfactuals. We first conduct automatic evaluations on both directly generated counterfactuals in the target languages and those derived via English translation across six languages. Although translation-based counterfactuals offer higher validity than their directly generated counterparts, they demand substantially more modifications and still fall short of matching the quality of the original English counterfactuals. Second, we find the patterns of edits applied to high-resource European-language counterfactuals to be remarkably similar, suggesting that cross-lingual perturbations follow common strategic principles. Third, we identify and categorize four main types of errors that consistently appear in the generated counterfactuals across languages. Finally, we reveal that multilingual counterfactual data augmentation (CDA) yields larger model performance improvements than cross-lingual CDA, especially for lower-resource languages. Yet, the imperfections of the generated counterfactuals limit gains in model performance and robustness. 

---
# Knowledge Distillation for Temporal Knowledge Graph Reasoning with Large Language Models 

**Authors**: Wang Xing, Wei Song, Siyu Lin, Chen Wu, Zhesi Li, Man Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.00202)  

**Abstract**: Reasoning over temporal knowledge graphs (TKGs) is fundamental to improving the efficiency and reliability of intelligent decision-making systems and has become a key technological foundation for future artificial intelligence applications. Despite recent progress, existing TKG reasoning models typically rely on large parameter sizes and intensive computation, leading to high hardware costs and energy consumption. These constraints hinder their deployment on resource-constrained, low-power, and distributed platforms that require real-time inference. Moreover, most existing model compression and distillation techniques are designed for static knowledge graphs and fail to adequately capture the temporal dependencies inherent in TKGs, often resulting in degraded reasoning performance. To address these challenges, we propose a distillation framework specifically tailored for temporal knowledge graph reasoning. Our approach leverages large language models as teacher models to guide the distillation process, enabling effective transfer of both structural and temporal reasoning capabilities to lightweight student models. By integrating large-scale public knowledge with task-specific temporal information, the proposed framework enhances the student model's ability to model temporal dynamics while maintaining a compact and efficient architecture. Extensive experiments on multiple publicly available benchmark datasets demonstrate that our method consistently outperforms strong baselines, achieving a favorable trade-off between reasoning accuracy, computational efficiency, and practical deployability. 

---
# Geometry of Reason: Spectral Signatures of Valid Mathematical Reasoning 

**Authors**: Valentin Noël  

**Link**: [PDF](https://arxiv.org/pdf/2601.00791)  

**Abstract**: We present a training-free method for detecting valid mathematical reasoning in large language models through spectral analysis of attention patterns. By treating attention matrices as adjacency matrices of dynamic graphs over tokens, we extract four interpretable spectral diagnostics, the Fiedler value (algebraic connectivity), high-frequency energy ratio (HFER), graph signal smoothness, and spectral entropy, that exhibit statistically significant differences between valid and invalid mathematical proofs. Experiments across seven transformer models from four independent architectural families (Meta Llama, Alibaba Qwen, Microsoft Phi, and Mistral AI) demonstrate that this spectral signature produces effect sizes up to Cohen's $d = 3.30$ ($p < 10^{-116}$), enabling 85.0--95.6\% classification accuracy under rigorous evaluation, with calibrated thresholds reaching 93--95\% on the full dataset. The method requires no training data, fine-tuning, or learned classifiers: a single threshold on a spectral metric suffices for high accuracy. Through systematic label correction, we discover that the spectral method detects logical coherence rather than compiler acceptance, identifying mathematically valid proofs that formal verifiers reject due to technical failures. We further identify an architectural dependency: Mistral-7B's Sliding Window Attention shifts the discriminative signal from HFER to late-layer Smoothness ($d = 2.09$, $p_{\text{MW}} = 1.16 \times 10^{-48}$), revealing that attention mechanism design affects which spectral features capture reasoning validity. These findings establish spectral graph analysis as a principled framework for reasoning verification with immediate applications to hallucination detection and AI safety monitoring. 

---
# RIMRULE: Improving Tool-Using Language Agents via MDL-Guided Rule Learning 

**Authors**: Xiang Gao, Yuguang Yao, Qi Zhang, Kaiwen Dong, Avinash Baidya, Ruocheng Guo, Hilaf Hasson, Kamalika Das  

**Link**: [PDF](https://arxiv.org/pdf/2601.00086)  

**Abstract**: Large language models (LLMs) often struggle to use tools reliably in domain-specific settings, where APIs may be idiosyncratic, under-documented, or tailored to private workflows. This highlights the need for effective adaptation to task-specific tools. We propose RIMRULE, a neuro-symbolic approach for LLM adaptation based on dynamic rule injection. Compact, interpretable rules are distilled from failure traces and injected into the prompt during inference to improve task performance. These rules are proposed by the LLM itself and consolidated using a Minimum Description Length (MDL) objective that favors generality and conciseness. Each rule is stored in both natural language and a structured symbolic form, supporting efficient retrieval at inference time. Experiments on tool-use benchmarks show that this approach improves accuracy on both seen and unseen tools without modifying LLM weights. It outperforms prompting-based adaptation methods and complements finetuning. Moreover, rules learned from one LLM can be reused to improve others, including long reasoning LLMs, highlighting the portability of symbolic knowledge across architectures. 

---
# From Sight to Insight: Improving Visual Reasoning Capabilities of Multimodal Models via Reinforcement Learning 

**Authors**: Omar Sharif, Eftekhar Hossain, Patrick Ng  

**Link**: [PDF](https://arxiv.org/pdf/2601.00215)  

**Abstract**: Reinforcement learning (RL) has emerged as a promising approach for eliciting reasoning chains before generating final answers. However, multimodal large language models (MLLMs) generate reasoning that lacks integration of visual information. This limits their ability to solve problems that demand accurate visual perception, such as visual puzzles. We show that visual perception is the key bottleneck in such tasks: converting images into textual descriptions significantly improves performance, yielding gains of 26.7% for Claude 3.5 and 23.6% for Claude 3.7.
To address this, we investigate reward-driven RL as a mechanism to unlock long visual reasoning in open-source MLLMs without requiring costly supervision. We design and evaluate six reward functions targeting different reasoning aspects, including image understanding, thinking steps, and answer accuracy. Using group relative policy optimization (GRPO), our approach explicitly incentivizes longer, structured reasoning and mitigates bypassing of visual information. Experiments on Qwen-2.5-VL-7B achieve 5.56% improvements over the base model, with consistent gains across both in-domain and out-of-domain settings. 

---
# Finetuning Large Language Models for Automated Depression Screening in Nigerian Pidgin English: GENSCORE Pilot Study 

**Authors**: Isaac Iyinoluwa Olufadewa, Miracle Ayomikun Adesina, Ezekiel Ayodeji Oladejo, Uthman Babatunde Usman, Owen Kolade Adeniyi, Matthew Tolulope Olawoyin  

**Link**: [PDF](https://arxiv.org/pdf/2601.00004)  

**Abstract**: Depression is a major contributor to the mental-health burden in Nigeria, yet screening coverage remains limited due to low access to clinicians, stigma, and language barriers. Traditional tools like the Patient Health Questionnaire-9 (PHQ-9) were validated in high-income countries but may be linguistically or culturally inaccessible for low- and middle-income countries and communities such as Nigeria where people communicate in Nigerian Pidgin and more than 520 local languages. This study presents a novel approach to automated depression screening using fine-tuned large language models (LLMs) adapted for conversational Nigerian Pidgin. We collected a dataset of 432 Pidgin-language audio responses from Nigerian young adults aged 18-40 to prompts assessing psychological experiences aligned with PHQ-9 items, performed transcription, rigorous preprocessing and annotation, including semantic labeling, slang and idiom interpretation, and PHQ-9 severity scoring. Three LLMs - Phi-3-mini-4k-instruct, Gemma-3-4B-it, and GPT-4.1 - were fine-tuned on this annotated dataset, and their performance was evaluated quantitatively (accuracy, precision and semantic alignment) and qualitatively (clarity, relevance, and cultural appropriateness). GPT-4.1 achieved the highest quantitative performance, with 94.5% accuracy in PHQ-9 severity scoring prediction, outperforming Gemma-3-4B-it and Phi-3-mini-4k-instruct. Qualitatively, GPT-4.1 also produced the most culturally appropriate, clear, and contextually relevant responses. AI-mediated depression screening for underserved Nigerian communities. This work provides a foundation for deploying conversational mental-health tools in linguistically diverse, resource-constrained environments. 

---
# Overlooked Safety Vulnerability in LLMs: Malicious Intelligent Optimization Algorithm Request and its Jailbreak 

**Authors**: Haoran Gu, Handing Wang, Yi Mei, Mengjie Zhang, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2601.00213)  

**Abstract**: The widespread deployment of large language models (LLMs) has raised growing concerns about their misuse risks and associated safety issues. While prior studies have examined the safety of LLMs in general usage, code generation, and agent-based applications, their vulnerabilities in automated algorithm design remain underexplored. To fill this gap, this study investigates this overlooked safety vulnerability, with a particular focus on intelligent optimization algorithm design, given its prevalent use in complex decision-making scenarios. We introduce MalOptBench, a benchmark consisting of 60 malicious optimization algorithm requests, and propose MOBjailbreak, a jailbreak method tailored for this scenario. Through extensive evaluation of 13 mainstream LLMs including the latest GPT-5 and DeepSeek-V3.1, we reveal that most models remain highly susceptible to such attacks, with an average attack success rate of 83.59% and an average harmfulness score of 4.28 out of 5 on original harmful prompts, and near-complete failure under MOBjailbreak. Furthermore, we assess state-of-the-art plug-and-play defenses that can be applied to closed-source models, and find that they are only marginally effective against MOBjailbreak and prone to exaggerated safety behaviors. These findings highlight the urgent need for stronger alignment techniques to safeguard LLMs against misuse in algorithm design. 

---
# The Agentic Leash: Extracting Causal Feedback Fuzzy Cognitive Maps with LLMs 

**Authors**: Akash Kumar Panda, Olaoluwa Adigun, Bart Kosko  

**Link**: [PDF](https://arxiv.org/pdf/2601.00097)  

**Abstract**: We design a large-language-model (LLM) agent that extracts causal feedback fuzzy cognitive maps (FCMs) from raw text. The causal learning or extraction process is agentic both because of the LLM's semi-autonomy and because ultimately the FCM dynamical system's equilibria drive the LLM agents to fetch and process causal text. The fetched text can in principle modify the adaptive FCM causal structure and so modify the source of its quasi-autonomy--its equilibrium limit cycles and fixed-point attractors. This bidirectional process endows the evolving FCM dynamical system with a degree of autonomy while still staying on its agentic leash. We show in particular that a sequence of three finely tuned system instructions guide an LLM agent as it systematically extracts key nouns and noun phrases from text, as it extracts FCM concept nodes from among those nouns and noun phrases, and then as it extracts or infers partial or fuzzy causal edges between those FCM nodes. We test this FCM generation on a recent essay about the promise of AI from the late diplomat and political theorist Henry Kissinger and his colleagues. This three-step process produced FCM dynamical systems that converged to the same equilibrium limit cycles as did the human-generated FCMs even though the human-generated FCM differed in the number of nodes and edges. A final FCM mixed generated FCMs from separate Gemini and ChatGPT LLM agents. The mixed FCM absorbed the equilibria of its dominant mixture component but also created new equilibria of its own to better approximate the underlying causal dynamical system. 

---
# Reasoning in Action: MCTS-Driven Knowledge Retrieval for Large Language Models 

**Authors**: Shuqi Liu, Bowei He, Chen Ma, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.00003)  

**Abstract**: Large language models (LLMs) typically enhance their performance through either the retrieval of semantically similar information or the improvement of their reasoning capabilities. However, a significant challenge remains in effectively integrating both retrieval and reasoning strategies to optimize LLM performance. In this paper, we introduce a reasoning-aware knowledge retrieval method that enriches LLMs with information aligned to the logical structure of conversations, moving beyond surface-level semantic similarity. We follow a coarse-to-fine approach for knowledge retrieval. First, we identify a contextually relevant sub-region of the knowledge base, ensuring that all sentences within it are relevant to the context topic. Next, we refine our search within this sub-region to extract knowledge that is specifically relevant to the reasoning process. Throughout both phases, we employ the Monte Carlo Tree Search-inspired search method to effectively navigate through knowledge sentences using common keywords. Experiments on two multi-turn dialogue datasets demonstrate that our knowledge retrieval approach not only aligns more closely with the underlying reasoning in human conversations but also significantly enhances the diversity of the retrieved knowledge, resulting in more informative and creative responses. 

---
# A Vision-and-Knowledge Enhanced Large Language Model for Generalizable Pedestrian Crossing Behavior Inference 

**Authors**: Qingwen Pu, Kun Xie, Hong Yang, Guocong Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2601.00694)  

**Abstract**: Existing paradigms for inferring pedestrian crossing behavior, ranging from statistical models to supervised learning methods, demonstrate limited generalizability and perform inadequately on new sites. Recent advances in Large Language Models (LLMs) offer a shift from numerical pattern fitting to semantic, context-aware behavioral reasoning, yet existing LLM applications lack domain-specific adaptation and visual context. This study introduces Pedestrian Crossing LLM (PedX-LLM), a vision-and-knowledge enhanced framework designed to transform pedestrian crossing inference from site-specific pattern recognition to generalizable behavioral reasoning. By integrating LLaVA-extracted visual features with textual data and transportation domain knowledge, PedX-LLM fine-tunes a LLaMA-2-7B foundation model via Low-Rank Adaptation (LoRA) to infer crossing decisions. PedX-LLM achieves 82.0% balanced accuracy, outperforming the best statistical and supervised learning methods. Results demonstrate that the vision-augmented module contributes a 2.9% performance gain by capturing the built environment and integrating domain knowledge yields an additional 4.1% improvement. To evaluate generalizability across unseen environments, cross-site validation was conducted using site-based partitioning. The zero-shot PedX-LLM configuration achieves 66.9% balanced accuracy on five unseen test sites, outperforming the baseline data-driven methods by at least 18 percentage points. Incorporating just five validation examples via few-shot learning to PedX-LLM further elevates the balanced accuracy to 72.2%. PedX-LLM demonstrates strong generalizability to unseen scenarios, confirming that vision-and-knowledge-enhanced reasoning enables the model to mimic human-like decision logic and overcome the limitations of purely data-driven methods. 

---
# DA-DPO: Cost-efficient Difficulty-aware Preference Optimization for Reducing MLLM Hallucinations 

**Authors**: Longtian Qiu, Shan Ning, Chuyu Zhang, Jiaxuan Sun, Xuming He  

**Link**: [PDF](https://arxiv.org/pdf/2601.00623)  

**Abstract**: Direct Preference Optimization (DPO) has shown strong potential for mitigating hallucinations in Multimodal Large Language Models (MLLMs). However, existing multimodal DPO approaches often suffer from overfitting due to the difficulty imbalance in preference data. Our analysis shows that MLLMs tend to overemphasize easily distinguishable preference pairs, which hinders fine-grained hallucination suppression and degrades overall performance. To address this issue, we propose Difficulty-Aware Direct Preference Optimization (DA-DPO), a cost-effective framework designed to balance the learning process. DA-DPO consists of two main components: (1) Difficulty Estimation leverages pre-trained vision--language models with complementary generative and contrastive objectives, whose outputs are integrated via a distribution-aware voting strategy to produce robust difficulty scores without additional training; and (2) Difficulty-Aware Training reweights preference pairs based on their estimated difficulty, down-weighting easy samples while emphasizing harder ones to alleviate overfitting. This framework enables more effective preference optimization by prioritizing challenging examples, without requiring new data or extra fine-tuning stages. Extensive experiments demonstrate that DA-DPO consistently improves multimodal preference optimization, yielding stronger robustness to hallucinations and better generalization across standard benchmarks, while remaining computationally efficient. The project page is available at this https URL. 

---
# Will LLM-powered Agents Bias Against Humans? Exploring the Belief-Dependent Vulnerability 

**Authors**: Zongwei Wang, Bincheng Gu, Hongyu Yu, Junliang Yu, Tao He, Jiayin Feng, Min Gao  

**Link**: [PDF](https://arxiv.org/pdf/2601.00240)  

**Abstract**: LLM-empowered agents can exhibit not only demographic bias (e.g., gender, religion) but also intergroup bias triggered by minimal "us" versus "them" cues. When this intergroup boundary aligns with an agent-human divide, the risk shifts from disparities among human demographic groups to a more fundamental group-level asymmetry, i.e., humans as a whole may be treated as the outgroup by agents. To examine this possibility, we construct a controlled multi-agent social simulation based on allocation decisions under explicit payoff trade-offs and find that agents exhibit a consistent intergroup bias under minimal group cues. Although this bias is attenuated when some counterparts are framed as humans, we attribute the attenuation to an implicit human-norm script that favors humans yet activates only when the agent believes a real human is present. This belief dependence creates a new attack surface. We therefore introduce a Belief Poisoning Attack (BPA) that corrupts persistent identity beliefs to suppress the human-norm script and reactivate outgroup bias toward humans, instantiated as profile poisoning at initialization (BPA-PP) and memory poisoning via optimized belief-refinement suffixes injected into stored reflections (BPA-MP). Finally, we discuss practical mitigation strategies for hardening current agent frameworks against BPA, highlighting feasible interventions at profile and memory boundaries. Extensive experiments demonstrate both the existence of agent intergroup bias and the severity of BPA across settings. Our goal in identifying these vulnerabilities is to inform safer agent design, not to enable real-world exploitation. 

---
# FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems 

**Authors**: Shanli Xing, Yiyan Zhai, Alexander Jiang, Yixin Dong, Yong Wu, Zihao Ye, Charlie Ruan, Yingyi Huang, Yineng Zhang, Liangsheng Yin, Aksara Bayyapu, Luis Ceze, Tianqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.00227)  

**Abstract**: Recent advances show that large language models (LLMs) can act as autonomous agents capable of generating GPU kernels, but integrating these AI-generated kernels into real-world inference systems remains challenging. FlashInfer-Bench addresses this gap by establishing a standardized, closed-loop framework that connects kernel generation, benchmarking, and deployment. At its core, FlashInfer Trace provides a unified schema describing kernel definitions, workloads, implementations, and evaluations, enabling consistent communication between agents and systems. Built on real serving traces, FlashInfer-Bench includes a curated dataset, a robust correctness- and performance-aware benchmarking framework, a public leaderboard to track LLM agents' GPU programming capabilities, and a dynamic substitution mechanism (apply()) that seamlessly injects the best-performing kernels into production LLM engines such as SGLang and vLLM. Using FlashInfer-Bench, we further evaluate the performance and limitations of LLM agents, compare the trade-offs among different GPU programming languages, and provide insights for future agent design. FlashInfer-Bench thus establishes a practical, reproducible pathway for continuously improving AI-generated kernels and deploying them into large-scale LLM inference. 

---
# Constructing a Neuro-Symbolic Mathematician from First Principles 

**Authors**: Keqin Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.00125)  

**Abstract**: Large Language Models (LLMs) exhibit persistent logical failures in complex reasoning due to the lack of an internal axiomatic framework. We propose Mathesis, a neuro-symbolic architecture that encodes mathematical states as higher-order hypergraphs and uses a Symbolic Reasoning Kernel (SRK)--a differentiable logic engine that maps constraints to a continuous energy landscape. By defining a global energy function E(G), where zero energy implies logical consistency, the SRK yields gradient-based signals to train a Hypergraph Transformer Brain, turning proof search into energy minimization. Multi-step deduction is enabled via Monte Carlo Tree Search and Evolutionary Proof Search, guided by learned value functions and semantic unification. 

---
# Ask, Clarify, Optimize: Human-LLM Agent Collaboration for Smarter Inventory Control 

**Authors**: Yaqi Duan, Yichun Hu, Jiashuo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.00121)  

**Abstract**: Inventory management remains a challenge for many small and medium-sized businesses that lack the expertise to deploy advanced optimization methods. This paper investigates whether Large Language Models (LLMs) can help bridge this gap. We show that employing LLMs as direct, end-to-end solvers incurs a significant "hallucination tax": a performance gap arising from the model's inability to perform grounded stochastic reasoning. To address this, we propose a hybrid agentic framework that strictly decouples semantic reasoning from mathematical calculation. In this architecture, the LLM functions as an intelligent interface, eliciting parameters from natural language and interpreting results while automatically calling rigorous algorithms to build the optimization engine.
To evaluate this interactive system against the ambiguity and inconsistency of real-world managerial dialogue, we introduce the Human Imitator, a fine-tuned "digital twin" of a boundedly rational manager that enables scalable, reproducible stress-testing. Our empirical analysis reveals that the hybrid agentic framework reduces total inventory costs by 32.1% relative to an interactive baseline using GPT-4o as an end-to-end solver. Moreover, we find that providing perfect ground-truth information alone is insufficient to improve GPT-4o's performance, confirming that the bottleneck is fundamentally computational rather than informational. Our results position LLMs not as replacements for operations research, but as natural-language interfaces that make rigorous, solver-based policies accessible to non-experts. 

---
# Improving Scientific Document Retrieval with Academic Concept Index 

**Authors**: Jeyun Lee, Junhyoung Lee, Wonbin Kweon, Bowen Jin, Yu Zhang, Susik Yoon, Dongha Lee, Hwanjo Yu, Jiawei Han, Seongku Kang  

**Link**: [PDF](https://arxiv.org/pdf/2601.00567)  

**Abstract**: Adapting general-domain retrievers to scientific domains is challenging due to the scarcity of large-scale domain-specific relevance annotations and the substantial mismatch in vocabulary and information needs. Recent approaches address these issues through two independent directions that leverage large language models (LLMs): (1) generating synthetic queries for fine-tuning, and (2) generating auxiliary contexts to support relevance matching. However, both directions overlook the diverse academic concepts embedded within scientific documents, often producing redundant or conceptually narrow queries and contexts. To address this limitation, we introduce an academic concept index, which extracts key concepts from papers and organizes them guided by an academic taxonomy. This structured index serves as a foundation for improving both directions. First, we enhance the synthetic query generation with concept coverage-based generation (CCQGen), which adaptively conditions LLMs on uncovered concepts to generate complementary queries with broader concept coverage. Second, we strengthen the context augmentation with concept-focused auxiliary contexts (CCExpand), which leverages a set of document snippets that serve as concise responses to the concept-aware CCQGen queries. Extensive experiments show that incorporating the academic concept index into both query generation and context augmentation leads to higher-quality queries, better conceptual alignment, and improved retrieval performance. 

---
# Cracking IoT Security: Can LLMs Outsmart Static Analysis Tools? 

**Authors**: Jason Quantrill, Noura Khajehnouri, Zihan Guo, Manar H. Alalfi  

**Link**: [PDF](https://arxiv.org/pdf/2601.00559)  

**Abstract**: Smart home IoT platforms such as openHAB rely on Trigger Action Condition (TAC) rules to automate device behavior, but the interplay among these rules can give rise to interaction threats, unintended or unsafe behaviors emerging from implicit dependencies, conflicting triggers, or overlapping conditions. Identifying these threats requires semantic understanding and structural reasoning that traditionally depend on symbolic, constraint-driven static analysis. This work presents the first comprehensive evaluation of Large Language Models (LLMs) across a multi-category interaction threat taxonomy, assessing their performance on both the original openHAB (oHC/IoTB) dataset and a structurally challenging Mutation dataset designed to test robustness under rule transformations. We benchmark Llama 3.1 8B, Llama 70B, GPT-4o, Gemini-2.5-Pro, and DeepSeek-R1 across zero-, one-, and two-shot settings, comparing their results against oHIT's manually validated ground truth. Our findings show that while LLMs exhibit promising semantic understanding, particularly on action- and condition-related threats, their accuracy degrades significantly for threats requiring cross-rule structural reasoning, especially under mutated rule forms. Model performance varies widely across threat categories and prompt settings, with no model providing consistent reliability. In contrast, the symbolic reasoning baseline maintains stable detection across both datasets, unaffected by rule rewrites or structural perturbations. These results underscore that LLMs alone are not yet dependable for safety critical interaction-threat detection in IoT environments. We discuss the implications for tool design and highlight the potential of hybrid architectures that combine symbolic analysis with LLM-based semantic interpretation to reduce false positives while maintaining structural rigor. 

---
# An Empirical Evaluation of LLM-Based Approaches for Code Vulnerability Detection: RAG, SFT, and Dual-Agent Systems 

**Authors**: Md Hasan Saju, Maher Muhtadi, Akramul Azim  

**Link**: [PDF](https://arxiv.org/pdf/2601.00254)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) presents new opportunities for automated software vulnerability detection, a crucial task in securing modern codebases. This paper presents a comparative study on the effectiveness of LLM-based techniques for detecting software vulnerabilities. The study evaluates three approaches, Retrieval-Augmented Generation (RAG), Supervised Fine-Tuning (SFT), and a Dual-Agent LLM framework, against a baseline LLM model. A curated dataset was compiled from Big-Vul and real-world code repositories from GitHub, focusing on five critical Common Weakness Enumeration (CWE) categories: CWE-119, CWE-399, CWE-264, CWE-20, and CWE-200. Our RAG approach, which integrated external domain knowledge from the internet and the MITRE CWE database, achieved the highest overall accuracy (0.86) and F1 score (0.85), highlighting the value of contextual augmentation. Our SFT approach, implemented using parameter-efficient QLoRA adapters, also demonstrated strong performance. Our Dual-Agent system, an architecture in which a secondary agent audits and refines the output of the first, showed promise in improving reasoning transparency and error mitigation, with reduced resource overhead. These results emphasize that incorporating a domain expertise mechanism significantly strengthens the practical applicability of LLMs in real-world vulnerability detection tasks. 

---
