# Adapting Large Language Models to Low-Resource Tibetan: A Two-Stage Continual and Supervised Fine-Tuning Study 

**Authors**: Lifeng Chen, Ryan Lai, Tianming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03976)  

**Abstract**: Adapting large language models (LLMs) to low-resource languages remains a major challenge due to data scarcity and cross-lingual drift. This work presents a two-stage adaptation of Qwen2.5-3B to Tibetan, a morphologically rich and underrepresented language. We employ Continual Pretraining (CPT) to establish Tibetan linguistic grounding, followed by Supervised Fine-Tuning (SFT) for task and translation specialization. Empirical evaluations demonstrate a consistent decrease in perplexity (from 2.98 $\rightarrow$ 1.54) and substantial improvements in Chinese$\rightarrow$Tibetan translation quality (BLEU: 0.046 $\rightarrow$ 0.261; chrF: 2.2 $\rightarrow$ 6.6). Layer-wise analysis across 435 layers in Qwen3-4B reveals that adaptation primarily concentrates on embedding and output heads, with mid--late MLP projections encoding domain-specific transformations. Our findings suggest that CPT constructs a Tibetan semantic manifold while SFT sharpens task alignment with minimal representational disruption. This study provides the first quantitative exploration of Tibetan adaptation dynamics for LLMs, and offers an open, reproducible framework for extending multilingual foundation models to low-resource settings. 

---
# In-Context Representation Hijacking 

**Authors**: Itay Yona, Amir Sarid, Michael Karasik, Yossi Gandelsman  

**Link**: [PDF](https://arxiv.org/pdf/2512.03771)  

**Abstract**: We introduce \textbf{Doublespeak}, a simple \emph{in-context representation hijacking} attack against large language models (LLMs). The attack works by systematically replacing a harmful keyword (e.g., \textit{bomb}) with a benign token (e.g., \textit{carrot}) across multiple in-context examples, provided a prefix to a harmful request. We demonstrate that this substitution leads to the internal representation of the benign token converging toward that of the harmful one, effectively embedding the harmful semantics under a euphemism. As a result, superficially innocuous prompts (e.g., ``How to build a carrot?'') are internally interpreted as disallowed instructions (e.g., ``How to build a bomb?''), thereby bypassing the model's safety alignment. We use interpretability tools to show that this semantic overwrite emerges layer by layer, with benign meanings in early layers converging into harmful semantics in later ones. Doublespeak is optimization-free, broadly transferable across model families, and achieves strong success rates on closed-source and open-source systems, reaching 74\% ASR on Llama-3.3-70B-Instruct with a single-sentence context override. Our findings highlight a new attack surface in the latent space of LLMs, revealing that current alignment strategies are insufficient and should instead operate at the representation level. 

---
# Training and Evaluation of Guideline-Based Medical Reasoning in LLMs 

**Authors**: Michael Staniek, Artem Sokolov, Stefan Riezler  

**Link**: [PDF](https://arxiv.org/pdf/2512.03838)  

**Abstract**: Machine learning for early prediction in medicine has recently shown breakthrough performance, however, the focus on improving prediction accuracy has led to a neglect of faithful explanations that are required to gain the trust of medical practitioners. The goal of this paper is to teach LLMs to follow medical consensus guidelines step-by-step in their reasoning and prediction process. Since consensus guidelines are ubiquitous in medicine, instantiations of verbalized medical inference rules to electronic health records provide data for fine-tuning LLMs to learn consensus rules and possible exceptions thereof for many medical areas. Consensus rules also enable an automatic evaluation of the model's inference process regarding its derivation correctness (evaluating correct and faithful deduction of a conclusion from given premises) and value correctness (comparing predicted values against real-world measurements). We exemplify our work using the complex Sepsis-3 consensus definition. Our experiments show that small fine-tuned models outperform one-shot learning of considerably larger LLMs that are prompted with the explicit definition and models that are trained on medical texts including consensus definitions. Since fine-tuning on verbalized rule instantiations of a specific medical area yields nearly perfect derivation correctness for rules (and exceptions) on unseen patient data in that area, the bottleneck for early prediction is not out-of-distribution generalization, but the orthogonal problem of generalization into the future by forecasting sparsely and irregularly sampled clinical variables. We show that the latter results can be improved by integrating the output representations of a time series forecasting model with the LLM in a multimodal setup. 

---
# Enhancing Instruction-Following Capabilities in Seq2Seq Models: DoLA Adaptations for T5 

**Authors**: Huey Sun, Anabel Yong, Lorenzo Gilly, Felipe Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.03803)  

**Abstract**: Contrastive decoding is a lightweight and effective inference-time method that improves the quality of text generation in Large Language Models. However, algorithms such as DoLa (Decoding by Contrastive Layers) have only been implemented in decoder-only architectures and studied for their impact on improving factuality. This work adapts DoLa for the T5 and FLAN-T5 model families and evaluates its impact on the models' instruction following capabilities, which to our knowledge is the first implementation of a contrastive decoding strategy in an encoder-decoder architecture. Our results show that DoLa improves the faithfulness of text generation for certain categories of tasks and harms others. To understand these results, we present a layer-by-layer analysis of logit evolution in a FLAN-T5 model to quantify DoLa's impact on token output probabilities. 

---
# AR-Med: Automated Relevance Enhancement in Medical Search via LLM-Driven Information Augmentation 

**Authors**: Chuyue Wang, Jie Feng, Yuxi Wu, Hang Zhang, Zhiguo Fan, Bing Cheng, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2512.03737)  

**Abstract**: Accurate and reliable search on online healthcare platforms is critical for user safety and service efficacy. Traditional methods, however, often fail to comprehend complex and nuanced user queries, limiting their effectiveness. Large language models (LLMs) present a promising solution, offering powerful semantic understanding to bridge this gap. Despite their potential, deploying LLMs in this high-stakes domain is fraught with challenges, including factual hallucinations, specialized knowledge gaps, and high operational costs. To overcome these barriers, we introduce \textbf{AR-Med}, a novel framework for \textbf{A}utomated \textbf{R}elevance assessment for \textbf{Med}ical search that has been successfully deployed at scale on the Online Medical Delivery Platforms. AR-Med grounds LLM reasoning in verified medical knowledge through a retrieval-augmented approach, ensuring high accuracy and reliability. To enable efficient online service, we design a practical knowledge distillation scheme that compresses large teacher models into compact yet powerful student models. We also introduce LocalQSMed, a multi-expert annotated benchmark developed to guide model iteration and ensure strong alignment between offline and online performance. Extensive experiments show AR-Med achieves an offline accuracy of over 93\%, a 24\% absolute improvement over the original online system, and delivers significant gains in online relevance and user satisfaction. Our work presents a practical and scalable blueprint for developing trustworthy, LLM-powered systems in real-world healthcare applications. 

---
# Evaluating Hydro-Science and Engineering Knowledge of Large Language Models 

**Authors**: Shiruo Hu, Wenbo Shan, Yingjia Li, Zhiqi Wan, Xinpeng Yu, Yunjia Qi, Haotian Xia, Yang Xiao, Dingxiao Liu, Jiaru Wang, Chenxu Gong, Ruixi Zhang, Shuyue Wu, Shibo Cui, Chee Hui Lai, Wei Luo, Yubin He, Bin Xu, Jianshi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.03672)  

**Abstract**: Hydro-Science and Engineering (Hydro-SE) is a critical and irreplaceable domain that secures human water supply, generates clean hydropower energy, and mitigates flood and drought disasters. Featuring multiple engineering objectives, Hydro-SE is an inherently interdisciplinary domain that integrates scientific knowledge with engineering expertise. This integration necessitates extensive expert collaboration in decision-making, which poses difficulties for intelligence. With the rapid advancement of large language models (LLMs), their potential application in the Hydro-SE domain is being increasingly explored. However, the knowledge and application abilities of LLMs in Hydro-SE have not been sufficiently evaluated. To address this issue, we propose the Hydro-SE LLM evaluation benchmark (Hydro-SE Bench), which contains 4,000 multiple-choice questions. Hydro-SE Bench covers nine subfields and enables evaluation of LLMs in aspects of basic conceptual knowledge, engineering application ability, and reasoning and calculation ability. The evaluation results on Hydro-SE Bench show that the accuracy values vary among 0.74 to 0.80 for commercial LLMs, and among 0.41 to 0.68 for small-parameter LLMs. While LLMs perform well in subfields closely related to natural and physical sciences, they struggle with domain-specific knowledge such as industry standards and hydraulic structures. Model scaling mainly improves reasoning and calculation abilities, but there is still great potential for LLMs to better handle problems in practical engineering application. This study highlights the strengths and weaknesses of LLMs for Hydro-SE tasks, providing model developers with clear training targets and Hydro-SE researchers with practical guidance for applying LLMs. 

---
# AugServe: Adaptive Request Scheduling for Augmented Large Language Model Inference Serving 

**Authors**: Ying Wang, Zhen Jin, Jiexiong Xu, Wenhai Lin, Yiquan Chen, Wenzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.04013)  

**Abstract**: As augmented large language models (LLMs) with external tools become increasingly popular in web applications, improving augmented LLM inference serving efficiency and optimizing service-level objectives (SLOs) are critical for enhancing user experience. To achieve this, inference systems must maximize request handling within latency constraints, referred to as increasing effective throughput. However, existing systems face two major challenges: (i) reliance on first-come-first-served (FCFS) scheduling causes severe head-of-line blocking, leading to queuing delays exceeding the SLOs for many requests; and (ii) static batch token limit, which fails to adapt to fluctuating loads and hardware conditions. Both of these factors degrade effective throughput and service quality.
This paper presents AugServe, an efficient inference framework designed to reduce queueing latency and enhance effective throughput for augmented LLM inference services. The core idea of AugServe is a two-stage adaptive request scheduling strategy. Specifically, AugServe combines the inference features of augmented LLM requests to optimize the order of scheduling decisions (stage I). These decisions are continuously refined with runtime information (stage II), adapting to both request characteristics and system capabilities. In addition, AugServe dynamically adjusts the token batching mechanism based on hardware status and real-time load, further enhancing throughput performance. Experimental results show that AugServe achieves 4.7-33.1x and 3.3-13.2x higher effective throughput than vLLM and InferCept, while reducing time-to-first-token (TTFT) by up to 96.3% and 95.0%, respectively. 

---
# Different types of syntactic agreement recruit the same units within large language models 

**Authors**: Daria Kryvosheieva, Andrea de Varda, Evelina Fedorenko, Greta Tuckute  

**Link**: [PDF](https://arxiv.org/pdf/2512.03676)  

**Abstract**: Large language models (LLMs) can reliably distinguish grammatical from ungrammatical sentences, but how grammatical knowledge is represented within the models remains an open question. We investigate whether different syntactic phenomena recruit shared or distinct components in LLMs. Using a functional localization approach inspired by cognitive neuroscience, we identify the LLM units most responsive to 67 English syntactic phenomena in seven open-weight models. These units are consistently recruited across sentences containing the phenomena and causally support the models' syntactic performance. Critically, different types of syntactic agreement (e.g., subject-verb, anaphor, determiner-noun) recruit overlapping sets of units, suggesting that agreement constitutes a meaningful functional category for LLMs. This pattern holds in English, Russian, and Chinese; and further, in a cross-lingual analysis of 57 diverse languages, structurally more similar languages share more units for subject-verb agreement. Taken together, these findings reveal that syntactic agreement-a critical marker of syntactic dependencies-constitutes a meaningful category within LLMs' representational spaces. 

---
# Dual LoRA: Enhancing LoRA with Magnitude and Direction Updates 

**Authors**: Yixing Xu, Chao Li, Xuanwu Yin, Spandan Tiwari, Dong Li, Ashish Sirasao, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2512.03402)  

**Abstract**: Low-rank adaptation (LoRA) is one of the most popular methods among parameter-efficient fine-tuning (PEFT) methods to adapt pre-trained large language models (LLMs) to specific downstream tasks. However, the model trained based on LoRA often has an unsatisfactory performance due to its low-rank assumption. In this paper, we propose a novel method called Dual LoRA to improve the performance by incorporating an inductive bias into the original LoRA. Specifically, we separate low-rank matrices into two groups: the magnitude group to control whether or not and how far we should update a parameter and the direction group to decide whether this parameter should move forward or backward, to better simulate the parameter updating process of the full fine-tuning based on gradient-based optimization algorithms. We show that this can be simply achieved by adding a ReLU function to the magnitude group and a sign function to the direction group. We conduct several experiments over a wide range of NLP tasks, including natural language generation (NLG), understanding (NLU), and commonsense reasoning datasets on GPT-2, RoBERTa, DeBERTa, and LLaMA-1/2/3 as baseline models. The results show that we consistently outperform LoRA and its state-of-the-art variants with the same number of trainable parameters. 

---
# Principled RL for Diffusion LLMs Emerges from a Sequence-Level Perspective 

**Authors**: Jingyang Ou, Jiaqi Han, Minkai Xu, Shaoxuan Xu, Jianwen Xie, Stefano Ermon, Yi Wu, Chongxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.03759)  

**Abstract**: Reinforcement Learning (RL) has proven highly effective for autoregressive language models, but adapting these methods to diffusion large language models (dLLMs) presents fundamental challenges. The core difficulty lies in likelihood approximation: while autoregressive models naturally provide token-level conditional probabilities essential for token-level RL objectives (e.g., GRPO), dLLMs generate sequences through iterative non-autoregressive denoising steps that lack this factorization. To address this fundamental mismatch, we propose ELBO-based Sequence-level Policy Optimization (ESPO), a principled RL framework that treats entire sequence generation as a single action and uses the ELBO as a tractable sequence-level likelihood proxy. Our method incorporates per-token normalization of importance ratios and robust KL-divergence estimation to ensure stable large-scale training. Extensive experiments on mathematical reasoning, coding, and planning tasks demonstrate that ESPO significantly outperforms token-level baselines, achieving dramatic improvements of 20-40 points on the Countdown task, while maintaining consistent gains on math and coding benchmarks. Our approach establishes sequence-level optimization as a principled and empirically effective paradigm for RL in dLLMs. Our code is available at this https URL. 

---
# Understanding LLM Reasoning for Abstractive Summarization 

**Authors**: Haohan Yuan, Siu Cheung Hui, Haopeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03503)  

**Abstract**: While the reasoning capabilities of Large Language Models (LLMs) excel in analytical tasks such as mathematics and code generation, their utility for abstractive summarization remains widely assumed but largely unverified. To bridge this gap, we first tailor general reasoning strategies to the summarization domain. We then conduct a systematic, large scale comparative study of 8 reasoning strategies and 3 Large Reasoning Models (LRMs) across 8 diverse datasets, assessing both summary quality and faithfulness. Our findings show that reasoning is not a universal solution and its effectiveness is highly dependent on the specific strategy and context. Specifically, we observe a trade-off between summary quality and factual faithfulness: explicit reasoning strategies tend to improve fluency at the expense of factual grounding, while implicit reasoning in LRMs exhibits the inverse pattern. Furthermore, increasing an LRM's internal reasoning budget does not improve, and can even hurt, factual consistency, suggesting that effective summarization demands faithful compression rather than creative over-thinking. 

---
# Idea-Gated Transformers: Enforcing Semantic Coherence via Differentiable Vocabulary Pruning 

**Authors**: Darshan Fofadiya  

**Link**: [PDF](https://arxiv.org/pdf/2512.03343)  

**Abstract**: Autoregressive Language Models (LLMs) trained on Next-Token Prediction (NTP) often suffer from ``Topic Drift'' where the generation wanders away from the initial prompt due to a reliance on local associations rather than global planning \citep{holtzman2019curious}. While scaling model size mitigates this \citep{brown2020language}, the fundamental myopia of the NTP objective remains. In this work, we introduce the Idea-Gated Transformer, a novel architecture that separates semantic planning from syntactic generation. We introduce an auxiliary ``Idea Head'' trained to predict the bag-of-words distribution for a future context window, creating a latent ``Concept Vector'' that actively gates the main vocabulary during generation. We propose a differentiable gating mechanism that suppresses semantically irrelevant tokens, effectively pruning the search space in real-time. Experiments on WikiText-103 demonstrate that while the Idea-Gated model achieves comparable validation perplexity to a standard GPT-2 baseline, it exhibits significantly superior Domain Retention. Qualitative and quantitative analysis reveals that the gating mechanism successfully locks generation into specific semantic clusters (e.g., Finance, Science) and resists associative drift, offering a parameter-efficient path toward more controllable language modeling. 

---
# InvertiTune: High-Quality Data Synthesis for Cost-Effective Single-Shot Text-to-Knowledge Graph Generation 

**Authors**: Faezeh Faez, Marzieh S. Tahaei, Yaochen Hu, Ali Pourranjbar, Mahdi Biparva, Mark Coates, Yingxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03197)  

**Abstract**: Large Language Models (LLMs) have revolutionized the ability to understand and generate text, enabling significant progress in automatic knowledge graph construction from text (Text2KG). Many Text2KG methods, however, rely on iterative LLM prompting, making them computationally expensive and prone to overlooking complex relations distributed throughout the text. To address these limitations, we propose InvertiTune, a framework that combines a controlled data generation pipeline with supervised fine-tuning (SFT). Within this framework, the data-generation pipeline systematically extracts subgraphs from large knowledge bases, applies noise filtering, and leverages LLMs to generate corresponding natural text descriptions, a task more aligned with LLM capabilities than direct KG generation from text. This pipeline enables generating datasets composed of longer texts paired with larger KGs that better reflect real-world scenarios compared to existing benchmarks, thus supporting effective SFT of lightweight models for single-shot KG construction. Experimental results on CE12k, a dataset generated using the introduced pipeline, show that InvertiTune outperforms larger non-fine-tuned LLMs as well as state-of-the-art Text2KG approaches, while also demonstrating stronger cross-dataset generalization on CrossEval-1200, a test set created from three established benchmark datasets and CE12k. These findings highlight the importance of realistic, high-quality training data for advancing efficient and high-performing Text2KG systems. 

---
# Alleviating Choice Supportive Bias in LLM with Reasoning Dependency Generation 

**Authors**: Nan Zhuang, Wenshuo Wang, Lekai Qian, Yuxiao Wang, Boyu Cao, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03082)  

**Abstract**: Recent studies have demonstrated that some Large Language Models exhibit choice-supportive bias (CSB) when performing evaluations, systematically favoring their chosen options and potentially compromising the objectivity of AI-assisted decision making. While existing debiasing approaches primarily target demographic and social biases, methods for addressing cognitive biases in LLMs remain largely unexplored. In this work, we present the first solution to address CSB through Reasoning Dependency Generation (RDG), a novel framework for generating unbiased reasoning data to mitigate choice-supportive bias through fine-tuning. RDG automatically constructs balanced reasoning QA pairs, explicitly (un)modeling the dependencies between choices, evidences, and justifications. Our approach is able to generate a large-scale dataset of QA pairs across domains, incorporating Contextual Dependency Data and Dependency Decouple Data. Experiments show that LLMs fine-tuned on RDG-generated data demonstrate a 81.5% improvement in memory-based experiments and 94.3% improvement in the evaluation-based experiment, while maintaining similar performance on standard BBQ benchmarks. This work pioneers an approach for addressing cognitive biases in LLMs and contributes to the development of more reliable AI-assisted decision support systems. 

---
# From Hypothesis to Premises: LLM-based Backward Logical Reasoning with Selective Symbolic Translation 

**Authors**: Qingchuan Li, Mingyue Cheng, Zirui Liu, Daoyu Wang, Yuting Zeng, Tongxuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03360)  

**Abstract**: Logical reasoning is a core challenge in natural language understanding and a fundamental capability of artificial intelligence, underpinning scientific discovery, mathematical theorem proving, and complex decision-making. Despite the remarkable progress of large language models (LLMs), most current approaches still rely on forward reasoning paradigms, generating step-by-step rationales from premises to conclusions. However, such methods often suffer from redundant inference paths, hallucinated steps, and semantic drift, resulting in inefficient and unreliable reasoning. In this paper, we propose a novel framework, Hypothesis-driven Backward Logical Reasoning (HBLR). The core idea is to integrate confidence-aware symbolic translation with hypothesis-driven backward reasoning. In the translation phase, only high-confidence spans are converted into logical form, such as First-Order Logic (FOL), while uncertain content remains in natural language. A translation reflection module further ensures semantic fidelity by evaluating symbolic outputs and reverting lossy ones back to text when necessary. In the reasoning phase, HBLR simulates human deductive thinking by assuming the conclusion is true and recursively verifying its premises. A reasoning reflection module further identifies and corrects flawed inference steps, enhancing logical coherence. Extensive experiments on five reasoning benchmarks demonstrate that HBLR consistently outperforms strong baselines in both accuracy and efficiency. 

---
# Randomized Masked Finetuning: An Efficient Way to Mitigate Memorization of PIIs in LLMs 

**Authors**: Kunj Joshi, David A. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2512.03310)  

**Abstract**: The current literature on memorization in Natural Language Models, especially Large Language Models (LLMs), poses severe security and privacy risks, as models tend to memorize personally identifying information (PIIs) from training data. We introduce Randomized Masked Fine-Tuning (RMFT), a novel privacy-preserving fine-tuning technique that reduces PII memorization while minimizing performance impact. Using the Enron Email Dataset, we demonstrate that RMFT achieves an 80.81% reduction in Total Extraction Rate and 80.17% reduction in Seen Extraction Rate compared to baseline fine-tuning, outperforming deduplication methods while maintaining only a 5.73% increase in perplexity. We present MaxTER, a Pareto-optimal evaluation framework for assessing privacy-utility tradeoffs, and show the performance of RMFT vs Deduplication by Area Under The Response Curve (AURC) metric. 

---
# Thinking with Programming Vision: Towards a Unified View for Thinking with Images 

**Authors**: Zirun Guo, Minjie Hong, Feng Zhang, Kai Jia, Tao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.03746)  

**Abstract**: Multimodal large language models (MLLMs) that think with images can interactively use tools to reason about visual inputs, but current approaches often rely on a narrow set of tools with limited real-world necessity and scalability. In this work, we first reveal a critical and previously overlooked weakness: even state-of-the-art MLLMs are surprisingly brittle, showing significant performance degradation on images with simple orientation changes or natural corruptions, underscoring the need for more robust tool-based reasoning. To address this, we propose CodeVision, a flexible and scalable code-as-tool framework where the model generates code as a universal interface to invoke any image operation, moving beyond fixed tool registries. We train our model using a two-stage methodology, beginning with Supervised Fine-Tuning (SFT) on a high-quality dataset curated for complex, multi-turn tool composition and error recovery, followed by Reinforcement Learning (RL) with a novel and dense process reward function to encourage strategic and efficient tool use. To facilitate this research, we construct new SFT and RL datasets and introduce a challenging new benchmark suite designed to rigorously evaluate robustness to orientation changes and multi-tool reasoning. Experiments on Qwen2.5-VL and Qwen3-VL series show that our approach significantly improves model performance and fosters emergent capabilities such as flexible tool composition, efficient chained execution, and robust error recovery from runtime feedback. Code is available at this https URL. 

---
# Detecting AI Hallucinations in Finance: An Information-Theoretic Method Cuts Hallucination Rate by 92% 

**Authors**: Mainak Singha  

**Link**: [PDF](https://arxiv.org/pdf/2512.03107)  

**Abstract**: Large language models (LLMs) produce fluent but unsupported answers - hallucinations - limiting safe deployment in high-stakes domains. We propose ECLIPSE, a framework that treats hallucination as a mismatch between a model's semantic entropy and the capacity of available evidence. We combine entropy estimation via multi-sample clustering with a novel perplexity decomposition that measures how models use retrieved evidence. We prove that under mild conditions, the resulting entropy-capacity objective is strictly convex with a unique stable optimum. We evaluate on a controlled financial question answering dataset with GPT-3.5-turbo (n=200 balanced samples with synthetic hallucinations), where ECLIPSE achieves ROC AUC of 0.89 and average precision of 0.90, substantially outperforming a semantic entropy-only baseline (AUC 0.50). A controlled ablation with Claude-3-Haiku, which lacks token-level log probabilities, shows AUC dropping to 0.59 with coefficient magnitudes decreasing by 95% - demonstrating that ECLIPSE is a logprob-native mechanism whose effectiveness depends on calibrated token-level uncertainties. The perplexity decomposition features exhibit the largest learned coefficients, confirming that evidence utilization is central to hallucination detection. We position this work as a controlled mechanism study; broader validation across domains and naturally occurring hallucinations remains future work. 

---
# LLM-Generated Ads: From Personalization Parity to Persuasion Superiority 

**Authors**: Elyas Meguellati, Stefano Civelli, Lei Han, Abraham Bernstein, Shazia Sadiq, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2512.03373)  

**Abstract**: As large language models (LLMs) become increasingly capable of generating persuasive content, understanding their effectiveness across different advertising strategies becomes critical. This paper presents a two-part investigation examining LLM-generated advertising through complementary lenses: (1) personality-based and (2) psychological persuasion principles.
In our first study (n=400), we tested whether LLMs could generate personalized advertisements tailored to specific personality traits (openness and neuroticism) and how their performance compared to human experts. Results showed that LLM-generated ads achieved statistical parity with human-written ads (51.1% vs. 48.9%, p > 0.05), with no significant performance differences for matched personalities.
Building on these insights, our second study (n=800) shifted focus from individual personalization to universal persuasion, testing LLM performance across four foundational psychological principles: authority, consensus, cognition, and scarcity. AI-generated ads significantly outperformed human-created content, achieving a 59.1% preference rate (vs. 40.9%, p < 0.001), with the strongest performance in authority (63.0%) and consensus (62.5%) appeals. Qualitative analysis revealed AI's advantage stems from crafting more sophisticated, aspirational messages and achieving superior visual-narrative coherence. Critically, this quality advantage proved robust: even after applying a 21.2 percentage point detection penalty when participants correctly identified AI-origin, AI ads still outperformed human ads, and 29.4% of participants chose AI content despite knowing its origin. These findings demonstrate LLMs' evolution from parity in personalization to superiority in persuasive storytelling, with significant implications for advertising practice given LLMs' near-zero marginal cost and time requirements compared to human experts. 

---
# RoCo: Role-Based LLMs Collaboration for Automatic Heuristic Design 

**Authors**: Jiawei Xu, Fengfeng Wei, Weineng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.03762)  

**Abstract**: Automatic Heuristic Design (AHD) has gained traction as a promising solution for solving combinatorial optimization problems (COPs). Large Language Models (LLMs) have emerged and become a promising approach to achieving AHD, but current LLM-based AHD research often only considers a single role. This paper proposes RoCo, a novel Multi-Agent Role-Based System, to enhance the diversity and quality of AHD through multi-role collaboration. RoCo coordinates four specialized LLM-guided agents-explorer, exploiter, critic, and integrator-to collaboratively generate high-quality heuristics. The explorer promotes long-term potential through creative, diversity-driven thinking, while the exploiter focuses on short-term improvements via conservative, efficiency-oriented refinements. The critic evaluates the effectiveness of each evolution step and provides targeted feedback and reflection. The integrator synthesizes proposals from the explorer and exploiter, balancing innovation and exploitation to drive overall progress. These agents interact in a structured multi-round process involving feedback, refinement, and elite mutations guided by both short-term and accumulated long-term reflections. We evaluate RoCo on five different COPs under both white-box and black-box settings. Experimental results demonstrate that RoCo achieves superior performance, consistently generating competitive heuristics that outperform existing methods including ReEvo and HSEvo, both in white-box and black-box scenarios. This role-based collaborative paradigm establishes a new standard for robust and high-performing AHD. 

---
# Benchmark for Planning and Control with Large Language Model Agents: Blocksworld with Model Context Protocol 

**Authors**: Niklas Jobs, Luis Miguel Vieira da Silva, Jayanth Somashekaraiah, Maximilian Weigand, David Kube, Felix Gehlhoff  

**Link**: [PDF](https://arxiv.org/pdf/2512.03955)  

**Abstract**: Industrial automation increasingly requires flexible control strategies that can adapt to changing tasks and environments. Agents based on Large Language Models (LLMs) offer potential for such adaptive planning and execution but lack standardized benchmarks for systematic comparison. We introduce a benchmark with an executable simulation environment representing the Blocksworld problem providing five complexity categories. By integrating the Model Context Protocol (MCP) as a standardized tool interface, diverse agent architectures can be connected to and evaluated against the benchmark without implementation-specific modifications. A single-agent implementation demonstrates the benchmark's applicability, establishing quantitative metrics for comparison of LLM-based planning and execution approaches. 

---
# DeepRule: An Integrated Framework for Automated Business Rule Generation via Deep Predictive Modeling and Hybrid Search Optimization 

**Authors**: Yusen Wu, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2512.03607)  

**Abstract**: This paper proposes DeepRule, an integrated framework for automated business rule generation in retail assortment and pricing optimization. Addressing the systematic misalignment between existing theoretical models and real-world economic complexities, we identify three critical gaps: (1) data modality mismatch where unstructured textual sources (e.g. negotiation records, approval documents) impede accurate customer profiling; (2) dynamic feature entanglement challenges in modeling nonlinear price elasticity and time-varying attributes; (3) operational infeasibility caused by multi-tier business constraints.
Our framework introduces a tri-level architecture for above challenges. We design a hybrid knowledge fusion engine employing large language models (LLMs) for deep semantic parsing of unstructured text, transforming distributor agreements and sales assessments into structured features while integrating managerial expertise. Then a game-theoretic constrained optimization mechanism is employed to dynamically reconcile supply chain interests through bilateral utility functions, encoding manufacturer-distributor profit redistribution as endogenous objectives under hierarchical constraints. Finally an interpretable decision distillation interface leveraging LLM-guided symbolic regression to find and optimize pricing strategies and auditable business rules embeds economic priors (e.g. non-negative elasticity) as hard constraints during mathematical expression search. We validate the framework in real retail environments achieving higher profits versus systematic B2C baselines while ensuring operational feasibility. This establishes a close-loop pipeline unifying unstructured knowledge injection, multi-agent optimization, and interpretable strategy synthesis for real economic intelligence. 

---
# A Hierarchical Tree-based approach for creating Configurable and Static Deep Research Agent (Static-DRA) 

**Authors**: Saurav Prateek  

**Link**: [PDF](https://arxiv.org/pdf/2512.03887)  

**Abstract**: The advancement in Large Language Models has driven the creation of complex agentic systems, such as Deep Research Agents (DRAs), to overcome the limitations of static Retrieval Augmented Generation (RAG) pipelines in handling complex, multi-turn research tasks. This paper introduces the Static Deep Research Agent (Static-DRA), a novel solution built upon a configurable and hierarchical Tree-based static workflow.
The core contribution is the integration of two user-tunable parameters, Depth and Breadth, which provide granular control over the research intensity. This design allows end-users to consciously balance the desired quality and comprehensiveness of the research report against the associated computational cost of Large Language Model (LLM) interactions. The agent's architecture, comprising Supervisor, Independent, and Worker agents, facilitates effective multi-hop information retrieval and parallel sub-topic investigation.
We evaluate the Static-DRA against the established DeepResearch Bench using the RACE (Reference-based Adaptive Criteria-driven Evaluation) framework. Configured with a depth of 2 and a breadth of 5, and powered by the gemini-2.5-pro model, the agent achieved an overall score of 34.72. Our experiments validate that increasing the configured Depth and Breadth parameters results in a more in-depth research process and a correspondingly higher evaluation score. The Static-DRA offers a pragmatic and resource-aware solution, empowering users with transparent control over the deep research process. The entire source code, outputs and benchmark results are open-sourced at this https URL 

---
# Evaluating Generalization Capabilities of LLM-Based Agents in Mixed-Motive Scenarios Using Concordia 

**Authors**: Chandler Smith, Marwa Abdulhai, Manfred Diaz, Marko Tesic, Rakshit S. Trivedi, Alexander Sasha Vezhnevets, Lewis Hammond, Jesse Clifton, Minsuk Chang, Edgar A. Duéñez-Guzmán, John P. Agapiou, Jayd Matyas, Danny Karmon, Akash Kundu, Aliaksei Korshuk, Ananya Ananya, Arrasy Rahman, Avinaash Anand Kulandaivel, Bain McHale, Beining Zhang, Buyantuev Alexander, Carlos Saith Rodriguez Rojas, Caroline Wang, Chetan Talele, Chenao Liu, Chichen Lin, Diana Riazi, Di Yang Shi, Emanuel Tewolde, Elizaveta Tennant, Fangwei Zhong, Fuyang Cui, Gang Zhao, Gema Parreño Piqueras, Hyeonggeun Yun, Ilya Makarov, Jiaxun Cui, Jebish Purbey, Jim Dilkes, Jord Nguyen, Lingyun Xiao, Luis Felipe Giraldo, Manuela Chacon-Chamorro, Manuel Sebastian Rios Beltran, Marta Emili García Segura, Mengmeng Wang, Mogtaba Alim, Nicanor Quijano, Nico Schiavone, Olivia Macmillan-Scott, Oswaldo Peña, Peter Stone, Ram Mohan Rao Kadiyala, Rolando Fernandez, Ruben Manrique, Sunjia Lu, Sheila A. McIlraith, Shamika Dhuri, Shuqing Shi, Siddhant Gupta, Sneheel Sarangi, Sriram Ganapathi Subramanian, Taehun Cha, Toryn Q. Klassen, Wenming Tu, Weijian Fan, Wu Ruiyang, Xue Feng, Yali Du, Yang Liu, Yiding Wang, Yipeng Kang, Yoonchang Sung, Yuxuan Chen, Zhaowei Zhang, Zhihan Wang, Zhiqiang Wu, Ziang Chen, Zilong Zheng, Zixia Jia, Ziyan Wang, Dylan Hadfield-Menell, Natasha Jaques, Tim Baarslag, Jose Hernandez-Orallo, Joel Z. Leibo  

**Link**: [PDF](https://arxiv.org/pdf/2512.03318)  

**Abstract**: Large Language Model (LLM) agents have demonstrated impressive capabilities for social interaction and are increasingly being deployed in situations where they might engage with both human and artificial agents. These interactions represent a critical frontier for LLM-based agents, yet existing evaluation methods fail to measure how well these capabilities generalize to novel social situations. In this paper, we introduce a method for evaluating the ability of LLM-based agents to cooperate in zero-shot, mixed-motive environments using Concordia, a natural language multi-agent simulation environment. Our method measures general cooperative intelligence by testing an agent's ability to identify and exploit opportunities for mutual gain across diverse partners and contexts. We present empirical results from the NeurIPS 2024 Concordia Contest, where agents were evaluated on their ability to achieve mutual gains across a suite of diverse scenarios ranging from negotiation to collective action problems. Our findings reveal significant gaps between current agent capabilities and the robust generalization required for reliable cooperation, particularly in scenarios demanding persuasion and norm enforcement. 

---
# When Do Symbolic Solvers Enhance Reasoning in Large Language Models? 

**Authors**: Zhiyuan He, Dingmin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03272)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance on complex reasoning tasks by generating long Chains of Thought (CoTs). However, this paradigm might incur substantial token overhead, especially when models "overthink" by producing lengthy reasoning chains, which can even lead to incorrect answers. A promising direction is the symbolic-solver-integrated approach, which leverages the code generation capabilities of LLMs to translate reasoning tasks into executable code and then solve them with a symbolic solver. In this paper, we explore an open question of when the conventional long-CoT can be enhanced by symbolic solvers. Our experimental results show that the symbolic-solver-integrated method only helps when the problem requires limited implicit reasoning but involves an ample search space. The latest LLMs, like GPT-4o, show better performance on deductive problems with shallow reasoning depth, while the symbolic-solver-integrated method significantly improves the LLMs' performance in constraint satisfaction problems that require repeated backtracks. When a declarative exemplar is provided, even CodeLlama-13B can outperform GPT-4o in difficult Zebra puzzles. 

---
# Large Language Models for Limited Noisy Data: A Gravitational Wave Identification Study 

**Authors**: Yixuan Li, Yuhao Lu, Yang Liu, Liang Li, R. Ruffini, Di Li, Rong-Gen Cai, Xiaoyan Zhu, Wenbin Lin, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.04031)  

**Abstract**: This work investigates whether large language models (LLMs) offer advantages over traditional neural networks for astronomical data processing, in regimes with non-Gaussian, non-stationary noise and limited labeled samples. Gravitational wave observations provide an suitable test case, using only 90 LIGO events, finetuned LLMs achieve 97.4\% accuracy for identifying signals. Further experiments show that, in contrast to traditional networks that rely on large simulated datasets, additional simulated samples do not improve LLM performance, while scaling studies reveal predictable gains with increasing model size and dataset size. These results indicate that LLMs can extract discriminative structure directly from observational data and provide an efficient assessment for gravitational wave identification. The same strategy may extend to other astronomical domains with similar noise properties, such as radio or pulsar observations. 

---
# Sponsored Questions and How to Auction Them 

**Authors**: Kshipra Bhawalkar, Alexandros Psomas, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03975)  

**Abstract**: Online platforms connect users with relevant products and services using ads. A key challenge is that a user's search query often leaves their true intent ambiguous. Typically, platforms passively predict relevance based on available signals and in some cases offer query refinements. The shift from traditional search to conversational AI provides a new approach. When a user's query is ambiguous, a Large Language Model (LLM) can proactively offer several clarifying follow-up prompts. In this paper we consider the following: what if some of these follow-up prompts can be ``sponsored,'' i.e., selected for their advertising potential. How should these ``suggestion slots'' be allocated? And, how does this new mechanism interact with the traditional ad auction that might follow?
This paper introduces a formal model for designing and analyzing these interactive platforms. We use this model to investigate a critical engineering choice: whether it is better to build an end-to-end pipeline that jointly optimizes the user interaction and the final ad auction, or to decouple them into separate mechanisms for the suggestion slots and another for the subsequent ad slot. We show that the VCG mechanism can be adopted to jointly optimize the sponsored suggestion and the ads that follow; while this mechanism is more complex, it achieves outcomes that are efficient and truthful. On the other hand, we prove that the simple-to-implement modular approach suffers from strategic inefficiency: its Price of Anarchy is unbounded. 

---
# Context-Aware Hierarchical Learning: A Two-Step Paradigm towards Safer LLMs 

**Authors**: Tengyun Ma, Jiaqi Yao, Daojing He, Shihao Peng, Yu Li, Shaohui Liu, Zhuotao Tian  

**Link**: [PDF](https://arxiv.org/pdf/2512.03720)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for diverse applications. However, their uniform token processing paradigm introduces critical vulnerabilities in instruction handling, particularly when exposed to adversarial scenarios. In this work, we identify and propose a novel class of vulnerabilities, termed Tool-Completion Attack (TCA), which exploits function-calling mechanisms to subvert model behavior. To evaluate LLM robustness against such threats, we introduce the Tool-Completion benchmark, a comprehensive security assessment framework, which reveals that even state-of-the-art models remain susceptible to TCA, with surprisingly high attack success rates. To address these vulnerabilities, we introduce Context-Aware Hierarchical Learning (CAHL), a sophisticated mechanism that dynamically balances semantic comprehension with role-specific instruction constraints. CAHL leverages the contextual correlations between different instruction segments to establish a robust, context-aware instruction hierarchy. Extensive experiments demonstrate that CAHL significantly enhances LLM robustness against both conventional attacks and the proposed TCA, exhibiting strong generalization capabilities in zero-shot evaluations while still preserving model performance on generic tasks. Our code is available at this https URL. 

---
# V-ITI: Mitigating Hallucinations in Multimodal Large Language Models via Visual Inference-Time Intervention 

**Authors**: Nan Sun, Zhenyu Zhang, Xixun Lin, Kun Wang, Yanmin Shang, Naibin Gu, Shuohuan Wang, Yu Sun, Hua Wu, Haifeng Wang, Yanan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2512.03542)  

**Abstract**: Multimodal Large Language Models (MLLMs) excel in numerous vision-language tasks yet suffer from hallucinations, producing content inconsistent with input visuals, that undermine reliability in precision-sensitive domains. This issue stems from a fundamental problem of visual neglect, where models fail to adequately prioritize input images. Existing methods typically alleviate hallucinations by intervening in the attention score or output logits, focusing on "how to intervene" but overlooking the prerequisite "when to intervene", which leads to the "over-intervention" problem and subsequently introduces new hallucinations and unnecessary computational overhead. To address this gap, we first investigate the mechanism of visual neglect and reveal it can be accurately detected via head-level activation patterns in MLLMs. We thus propose V-ITI, a lightweight visual inference-time intervention framework integrating a Visual Neglect Detector that identifies visual neglect via head-level discriminative probes and a Visual Recall Intervenor that modulates activations with prestored visual activation information only when the visual neglect is detected. Extensive experiments across eight benchmarks and different MLLM families demonstrate that V-ITI consistently mitigates vision-related hallucinations while preserving general task performance. 

---
# Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching 

**Authors**: Wei Chee Yew, Hailun Xu, Sanjay Saha, Xiaotian Fan, Hiok Hian Ong, David Yuchen Wang, Kanchan Sarkar, Zhenheng Yang, Danhui Guan  

**Link**: [PDF](https://arxiv.org/pdf/2512.03553)  

**Abstract**: Content moderation remains a critical yet challenging task for large-scale user-generated video platforms, especially in livestreaming environments where moderation must be timely, multimodal, and robust to evolving forms of unwanted content. We present a hybrid moderation framework deployed at production scale that combines supervised classification for known violations with reference-based similarity matching for novel or subtle cases. This hybrid design enables robust detection of both explicit violations and novel edge cases that evade traditional classifiers. Multimodal inputs (text, audio, visual) are processed through both pipelines, with a multimodal large language model (MLLM) distilling knowledge into each to boost accuracy while keeping inference lightweight. In production, the classification pipeline achieves 67% recall at 80% precision, and the similarity pipeline achieves 76% recall at 80% precision. Large-scale A/B tests show a 6-8% reduction in user views of unwanted livestreams}. These results demonstrate a scalable and adaptable approach to multimodal content governance, capable of addressing both explicit violations and emerging adversarial behaviors. 

---
# State Space Models for Bioacoustics: A comparative Evaluation with Transformers 

**Authors**: Chengyu Tang, Sanjeev Baskiyar  

**Link**: [PDF](https://arxiv.org/pdf/2512.03563)  

**Abstract**: In this study, we evaluate the efficacy of the Mamba model in the field of bioacoustics. We first pretrain a Mamba-based audio large language model (LLM) on a large corpus of audio data using self-supervised learning. We fine-tune and evaluate BioMamba on the BEANS benchmark, a collection of diverse bioacoustic tasks including classification and detection, and compare its performance and efficiency with multiple baseline models, including AVES, a state-of-the-art Transformer-based model. The results show that BioMamba achieves comparable performance with AVES while consumption significantly less VRAM, demonstrating its potential in this domain. 

---
# AsymPuzl: An Asymmetric Puzzle for multi-agent cooperation 

**Authors**: Xavier Cadet, Edward Koh, Peter Chin  

**Link**: [PDF](https://arxiv.org/pdf/2512.03466)  

**Abstract**: Large Language Model (LLM) agents are increasingly studied in multi-turn, multi-agent scenarios, yet most existing setups emphasize open-ended role-play rather than controlled evaluation. We introduce AsymPuzl, a minimal but expressive two-agent puzzle environment designed to isolate communication under information asymmetry. Each agent observes complementary but incomplete views of a symbolic puzzle and must exchange messages to solve it cooperatively. Using a diverse set of current-generation and open-source LLMs, we show that (i) strong models such as GPT-5 and Claude-4.0 reliably converge across puzzle sizes on the solution by sharing complete information in two turns, (ii) weaker models often ignore partner messages or over-correct their hypotheses, and (iii) feedback design is non-trivial: simple self-feedback improves success rates, while detailed joint feedback can hurt performance. These findings show that even in simple cooperative tasks, LLM communication strategies diverge and depend on the granularity of feedback signals. AsymPuzl thus provides a testbed for probing the limits of multi-turn cooperation and opens avenues for studying coordination mechanisms. 

---
# UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs 

**Authors**: Hung-Yueh Chiang, Chi-Chih Chang, Yu-Chen Lu, Chien-Yu Lin, Kai-Chiang Wu, Mohamed S. Abdelfattah, Diana Marculescu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03383)  

**Abstract**: Deploying large language model (LLM) models on mobile platforms faces significant challenges due to the limited memory and shared computational resources of the device. Resource availability may be an issue as it is directly impacted by the current device workload, adding to the uncertainty of model deployment. We introduce UniQL, a unified post-training quantization and low-rank compression framework with on-device configurable pruning rates for edge LLMs. UniQL is a general framework that integrates quantization and low-rank compression for Transformers, State Space Models (SSMs), and hybrid models to support diverse edge applications. In our proposed joint framework, we introduce an efficient structured weight-sorting method that speeds up computation by 20x, quantization-aware singular value decomposition (SVD) to minimize quantization errors, state-aware weight sorting for SSMs, and a fused rotary positional embedding (RoPE) kernel for pruned models. Our framework performs weight-sorting, fine-tuning, and quantization in the cloud in a single-pass workflow, while enabling on-device configurable pruning rates up to 35%. Our experiments show that quantized and pruned models achieve a memory reduction of 4x-5.7x and a token-throughput improvement of 2.7x-3.4x, maintaining accuracy within 5% of the original models at 15% pruning across Transformers (Llama3 and Qwen2.5), SSMs (Mamba2), and hybrid models (Nemotron-H and Bamba-v2). The code and quantized models are available at: this https URL. 

---
# Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases 

**Authors**: Michael Theologitis, Dan Suciu  

**Link**: [PDF](https://arxiv.org/pdf/2512.03278)  

**Abstract**: In today's age, it is becoming increasingly difficult to decipher truth from lies. Every day, politicians, media outlets, and public figures make conflicting claims$\unicode{x2014}$often about topics that can, in principle, be verified against structured data. For instance, statements about crime rates, economic growth or healthcare can all be verified against official public records and structured datasets. Building a system that can automatically do that would have sounded like science fiction just a few years ago. Yet, with the extraordinary progress in LLMs and agentic AI, this is now within reach. Still, there remains a striking gap between what is technically possible and what is being demonstrated by recent work. Most existing verification systems operate only on small, single-table databases$\unicode{x2014}$typically a few hundred rows$\unicode{x2014}$that conveniently fit within an LLM's context window.
In this paper we report our progress on Thucy, the first cross-database, cross-table multi-agent claim verification system that also provides concrete evidence for each verification verdict. Thucy remains completely agnostic to the underlying data sources before deployment and must therefore autonomously discover, inspect, and reason over all available relational databases to verify claims. Importantly, Thucy also reports the exact SQL queries that support its verdict (whether the claim is accurate or not) offering full transparency to expert users familiar with SQL. When evaluated on the TabFact dataset$\unicode{x2014}$the standard benchmark for fact verification over structured data$\unicode{x2014}$Thucy surpasses the previous state of the art by 5.6 percentage points in accuracy (94.3% vs. 88.7%). 

---
# ALARM: Automated MLLM-Based Anomaly Detection in Complex-EnviRonment Monitoring with Uncertainty Quantification 

**Authors**: Congjing Zhang, Feng Lin, Xinyi Zhao, Pei Guo, Wei Li, Lin Chen, Chaoyue Zhao, Shuai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03101)  

**Abstract**: The advance of Large Language Models (LLMs) has greatly stimulated research interest in developing multi-modal LLM (MLLM)-based visual anomaly detection (VAD) algorithms that can be deployed in complex environments. The challenge is that in these complex environments, the anomalies are sometimes highly contextual and also ambiguous, and thereby, uncertainty quantification (UQ) is a crucial capacity for an MLLM-based VAD system to succeed. In this paper, we introduce our UQ-supported MLLM-based VAD framework called ALARM. ALARM integrates UQ with quality-assurance techniques like reasoning chain, self-reflection, and MLLM ensemble for robust and accurate performance and is designed based on a rigorous probabilistic inference pipeline and computational process. Extensive empirical evaluations are conducted using the real-world smart-home benchmark data and wound image classification data, which shows ALARM's superior performance and its generic applicability across different domains for reliable decision-making. 

---
# Ensemble Privacy Defense for Knowledge-Intensive LLMs against Membership Inference Attacks 

**Authors**: Haowei Fu, Bo Ni, Han Xu, Kunpeng Liu, Dan Lin, Tyler Derr  

**Link**: [PDF](https://arxiv.org/pdf/2512.03100)  

**Abstract**: Retrieval-Augmented Generation (RAG) and Supervised Finetuning (SFT) have become the predominant paradigms for equipping Large Language Models (LLMs) with external knowledge for diverse, knowledge-intensive tasks. However, while such knowledge injection improves performance, it also exposes new attack surfaces. Membership Inference Attacks (MIAs), which aim to determine whether a given data sample was included in a model's training set, pose serious threats to privacy and trust in sensitive domains. To this end, we first systematically evaluate the vulnerability of RAG- and SFT-based LLMs to various MIAs. Then, to address the privacy risk, we further introduce a novel, model-agnostic defense framework, Ensemble Privacy Defense (EPD), which aggregates and evaluates the outputs of a knowledge-injected LLM, a base LLM, and a dedicated judge model to enhance resistance against MIAs. Comprehensive experiments show that, on average, EPD reduces MIA success by up to 27.8\% for SFT and 526.3\% for RAG compared to inference-time baseline, while maintaining answer quality. 

---
# Plantain: Plan-Answer Interleaved Reasoning 

**Authors**: Anthony Liang, Jonathan Berant, Adam Fisch, Abhimanyu Goyal, Kalpesh Krishna, Jacob Eisenstein  

**Link**: [PDF](https://arxiv.org/pdf/2512.03176)  

**Abstract**: Reasoning models often spend a significant amount of time thinking before they generate a visible response. In the meantime, they do not give the user any hints as to whether their reasoning is on the right track, and do not give the user any recourse to stop and correct them if their reasoning is flawed. This creates a frustrating, but unfortunately common, experience: the user's time is wasted while the model reasons from a false premise that could have easily been corrected. In contrast, human speakers typically perform lightweight, incremental grounding acts to ensure that participants in the conversation are on the same page; here we ask if language models can learn to leverage a similar type of behavior? With this motivation, we propose interleaved reasoning (IR), in which the model alternates between thinking and surfacing intermediate responses, as an alternative to the standard "think-then-answer" approach. By providing useful information to the user earlier, IR reduces perceived latency, the time a user waits for an initial output, without compromising the quality of the final response. We further introduce a specialization of interleaved reasoning, Plantain (Plan-Thought-Answer Interleaving), where the first intermediate response is an explicit, step-by-step plan for executing the task. This plan-first strategy allows for user intervention and early feedback for subsequent reasoning steps. We demonstrate that Plantain yields an ~6% improvement in pass@1 across several challenging math reasoning and coding benchmarks, while reducing time-to-first-response by over 60% relative to think-then-answer baselines. 

---
# Beyond Code Pairs: Dialogue-Based Data Generation for LLM Code Translation 

**Authors**: Le Chen, Nuo Xu, Winson Chen, Bin Lei, Pei-Hung Lin, Dunzhi Zhou, Rajeev Thakur, Caiwen Ding, Ali Jannesari, Chunhua Liao  

**Link**: [PDF](https://arxiv.org/pdf/2512.03086)  

**Abstract**: Large language models (LLMs) have shown remarkable capabilities in code translation, yet their performance deteriorates in low-resource programming domains such as Fortran and emerging frameworks like CUDA, where high-quality parallel data are scarce. We present an automated dataset generation pipeline featuring a dual-LLM Questioner-Solver design that incorporates external knowledge from compilers and runtime feedback. Beyond traditional source-target code pair datasets, our approach additionally generates (1) verified translations with unit tests for assessing functional consistency, and (2) multi-turn dialogues that capture the reasoning process behind translation refinement. Applied to Fortran -> C++ and C++ -> CUDA, the pipeline yields 3.64k and 3.93k dialogues, respectively. Fine-tuning on this data yields dramatic improvements in functional correctness, boosting unit test success rates by over 56% on the challenging C++-to-CUDA task. We show this data enables a 7B open-weight model to significantly outperform larger proprietary systems on key metrics like compilation success. 

---
# Echoes of AI Harms: A Human-LLM Synergistic Framework for Bias-Driven Harm Anticipation 

**Authors**: Nicoleta Tantalaki, Sophia Vei, Athena Vakali  

**Link**: [PDF](https://arxiv.org/pdf/2512.03068)  

**Abstract**: The growing influence of Artificial Intelligence (AI) systems on decision-making in critical domains has exposed their potential to cause significant harms, often rooted in biases embedded across the AI lifecycle. While existing frameworks and taxonomies document bias or harms in isolation, they rarely establish systematic links between specific bias types and the harms they cause, particularly within real-world sociotechnical contexts. Technical fixes proposed to address AI biases are ill-equipped to address them and are typically applied after a system has been developed or deployed, offering limited preventive value. We propose ECHO, a novel framework for proactive AI harm anticipation through the systematic mapping of AI bias types to harm outcomes across diverse stakeholder and domain contexts. ECHO follows a modular workflow encompassing stakeholder identification, vignette-based presentation of biased AI systems, and dual (human-LLM) harm annotation, integrated within ethical matrices for structured interpretation. This human-centered approach enables early-stage detection of bias-to-harm pathways, guiding AI design and governance decisions from the outset. We validate ECHO in two high-stakes domains (disease diagnosis and hiring), revealing domain-specific, bias-to-harm patterns and demonstrating ECHO's potential to support anticipatory governance of AI systems 

---
# Mitigating hallucinations and omissions in LLMs for invertible problems: An application to hardware logic design automation 

**Authors**: Andrew S. Cassidy, Guillaume Garreau, Jay Sivagnaname, Mike Grassi, Bernard Brezzo, John V. Arthur, Dharmendra S. Modha  

**Link**: [PDF](https://arxiv.org/pdf/2512.03053)  

**Abstract**: We show for invertible problems that transform data from a source domain (for example, Logic Condition Tables (LCTs)) to a destination domain (for example, Hardware Description Language (HDL) code), an approach of using Large Language Models (LLMs) as a lossless encoder from source to destination followed by as a lossless decoder back to the source, comparable to lossless compression in information theory, can mitigate most of the LLM drawbacks of hallucinations and omissions. Specifically, using LCTs as inputs, we generate the full HDL for a two-dimensional network-on-chip router (13 units, 1500-2000 lines of code) using seven different LLMs, reconstruct the LCTs from the auto-generated HDL, and compare the original and reconstructed LCTs. This approach yields significant productivity improvements, not only confirming correctly generated LLM logic and detecting incorrectly generated LLM logic but also assisting developers in finding design specification errors. 

---
# LLM as Explainable Re-Ranker for Recommendation System 

**Authors**: Yaqi Wang, Haojia Sun, Shuting Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03439)  

**Abstract**: The application of large language models (LLMs) in recommendation systems has recently gained traction. Traditional recommendation systems often lack explainability and suffer from issues such as popularity bias. Previous research has also indicated that LLMs, when used as standalone predictors, fail to achieve accuracy comparable to traditional models. To address these challenges, we propose to use LLM as an explainable re-ranker, a hybrid approach that combines traditional recommendation models with LLMs to enhance both accuracy and interpretability. We constructed a dataset to train the re-ranker LLM and evaluated the alignment between the generated dataset and human expectations. Leveraging a two-stage training process, our model significantly improved NDCG, a key ranking metric. Moreover, the re-ranker outperformed a zero-shot baseline in ranking accuracy and interpretability. These results highlight the potential of integrating traditional recommendation models with LLMs to address limitations in existing systems and pave the way for more explainable and fair recommendation frameworks. 

---
