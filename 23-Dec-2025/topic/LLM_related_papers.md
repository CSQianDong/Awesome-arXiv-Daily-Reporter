# Activations as Features: Probing LLMs for Generalizable Essay Scoring Representations 

**Authors**: Jinwei Chi, Ke Wang, Yu Chen, Xuanye Lin, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19456)  

**Abstract**: Automated essay scoring (AES) is a challenging task in cross-prompt settings due to the diversity of scoring criteria. While previous studies have focused on the output of large language models (LLMs) to improve scoring accuracy, we believe activations from intermediate layers may also provide valuable information. To explore this possibility, we evaluated the discriminative power of LLMs' activations in cross-prompt essay scoring task. Specifically, we used activations to fit probes and further analyzed the effects of different models and input content of LLMs on this discriminative power. By computing the directions of essays across various trait dimensions under different prompts, we analyzed the variation in evaluation perspectives of large language models concerning essay types and traits. Results show that the activations possess strong discriminative power in evaluating essay quality and that LLMs can adapt their evaluation perspectives to different traits and essay types, effectively handling the diversity of scoring criteria in cross-prompt settings. 

---
# Event Extraction in Large Language Model 

**Authors**: Bobo Li, Xudong Han, Jiang Liu, Yuzhe Ding, Liqiang Jing, Zhaoqi Zhang, Jinheng Li, Xinya Du, Fei Li, Meishan Zhang, Min Zhang, Aixin Sun, Philip S. Yu, Hao Fei  

**Link**: [PDF](https://arxiv.org/pdf/2512.19537)  

**Abstract**: Large language models (LLMs) and multimodal LLMs are changing event extraction (EE): prompting and generation can often produce structured outputs in zero shot or few shot settings. Yet LLM based pipelines face deployment gaps, including hallucinations under weak constraints, fragile temporal and causal linking over long contexts and across documents, and limited long horizon knowledge management within a bounded context window. We argue that EE should be viewed as a system component that provides a cognitive scaffold for LLM centered solutions. Event schemas and slot constraints create interfaces for grounding and verification; event centric structures act as controlled intermediate representations for stepwise reasoning; event links support relation aware retrieval with graph based RAG; and event stores offer updatable episodic and agent memory beyond the context window. This survey covers EE in text and multimodal settings, organizing tasks and taxonomy, tracing method evolution from rule based and neural models to instruction driven and generative frameworks, and summarizing formulations, decoding strategies, architectures, representations, datasets, and evaluation. We also review cross lingual, low resource, and domain specific settings, and highlight open challenges and future directions for reliable event centric systems. Finally, we outline open challenges and future directions that are central to the LLM era, aiming to evolve EE from static extraction into a structurally reliable, agent ready perception and memory layer for open world systems. 

---
# A Large-Language-Model Framework for Automated Humanitarian Situation Reporting 

**Authors**: Ivan Decostanzi, Yelena Mejova, Kyriaki Kalimeri  

**Link**: [PDF](https://arxiv.org/pdf/2512.19475)  

**Abstract**: Timely and accurate situational reports are essential for humanitarian decision-making, yet current workflows remain largely manual, resource intensive, and inconsistent. We present a fully automated framework that uses large language models (LLMs) to transform heterogeneous humanitarian documents into structured and evidence-grounded reports. The system integrates semantic text clustering, automatic question generation, retrieval augmented answer extraction with citations, multi-level summarization, and executive summary generation, supported by internal evaluation metrics that emulate expert reasoning. We evaluated the framework across 13 humanitarian events, including natural disasters and conflicts, using more than 1,100 documents from verified sources such as ReliefWeb. The generated questions achieved 84.7 percent relevance, 84.0 percent importance, and 76.4 percent urgency. The extracted answers reached 86.3 percent relevance, with citation precision and recall both exceeding 76 percent. Agreement between human and LLM based evaluations surpassed an F1 score of 0.80. Comparative analysis shows that the proposed framework produces reports that are more structured, interpretable, and actionable than existing baselines. By combining LLM reasoning with transparent citation linking and multi-level evaluation, this study demonstrates that generative AI can autonomously produce accurate, verifiable, and operationally useful humanitarian situation reports. 

---
# CodeSimpleQA: Scaling Factuality in Code Large Language Models 

**Authors**: Jian Yang, Wei Zhang, Yizhi Li, Shawn Guo, Haowen Wang, Aishan Liu, Ge Zhang, Zili Wang, Zhoujun Li, Xianglong Liu, Weifeng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2512.19424)  

**Abstract**: Large language models (LLMs) have made significant strides in code generation, achieving impressive capabilities in synthesizing code snippets from natural language instructions. However, a critical challenge remains in ensuring LLMs generate factually accurate responses about programming concepts, technical implementations, etc. Most previous code-related benchmarks focus on code execution correctness, overlooking the factual accuracy of programming knowledge. To address this gap, we present CodeSimpleQA, a comprehensive bilingual benchmark designed to evaluate the factual accuracy of code LLMs in answering code-related questions, which contains carefully curated question-answer pairs in both English and Chinese, covering diverse programming languages and major computer science domains. Further, we create CodeSimpleQA-Instruct, a large-scale instruction corpus with 66M samples, and develop a post-training framework combining supervised fine-tuning and reinforcement learning. Our comprehensive evaluation of diverse LLMs reveals that even frontier LLMs struggle with code factuality. Our proposed framework demonstrates substantial improvements over the base model, underscoring the critical importance of factuality-aware alignment in developing reliable code LLMs. 

---
# Exploring the features used for summary evaluation by Human and GPT 

**Authors**: Zahra Sadeghi, Evangelos Milios, Frank Rudzicz  

**Link**: [PDF](https://arxiv.org/pdf/2512.19620)  

**Abstract**: Summary assessment involves evaluating how well a generated summary reflects the key ideas and meaning of the source text, requiring a deep understanding of the content. Large Language Models (LLMs) have been used to automate this process, acting as judges to evaluate summaries with respect to the original text. While previous research investigated the alignment between LLMs and Human responses, it is not yet well understood what properties or features are exploited by them when asked to evaluate based on a particular quality dimension, and there has not been much attention towards mapping between evaluation scores and metrics. In this paper, we address this issue and discover features aligned with Human and Generative Pre-trained Transformers (GPTs) responses by studying statistical and machine learning metrics. Furthermore, we show that instructing GPTs to employ metrics used by Human can improve their judgment and conforming them better with human responses. 

---
# Increasing the Thinking Budget is Not All You Need 

**Authors**: Ignacio Iacobacci, Zhaozhi Qian, Faroq AL-Tam, Muhammad AL-Qurishi, Riad Souissi  

**Link**: [PDF](https://arxiv.org/pdf/2512.19585)  

**Abstract**: Recently, a new wave of thinking-capable Large Language Models has emerged, demonstrating exceptional capabilities across a wide range of reasoning benchmarks. Early studies have begun to explore how the amount of compute in terms of the length of the reasoning process, the so-called thinking budget, impacts model performance. In this work, we propose a systematic investigation of the thinking budget as a key parameter, examining its interaction with various configurations such as self-consistency, reflection, and others. Our goal is to provide an informative, balanced comparison framework that considers both performance outcomes and computational cost. Among our findings, we discovered that simply increasing the thinking budget is not the most effective use of compute. More accurate responses can instead be achieved through alternative configurations, such as self-consistency and self-reflection. 

---
# CienaLLM: Generative Climate-Impact Extraction from News Articles with Autoregressive LLMs 

**Authors**: Javier Vela-Tambo, Jorge Gracia, Fernando Dominguez-Castro  

**Link**: [PDF](https://arxiv.org/pdf/2512.19305)  

**Abstract**: Understanding and monitoring the socio-economic impacts of climate hazards requires extracting structured information from heterogeneous news articles on a large scale. To that end, we have developed CienaLLM, a modular framework based on schema-guided Generative Information Extraction. CienaLLM uses open-weight Large Language Models for zero-shot information extraction from news articles, and supports configurable prompts and output schemas, multi-step pipelines, and cloud or on-premise inference. To systematically assess how the choice of LLM family, size, precision regime, and prompting strategy affect performance, we run a large factorial study in models, precisions, and prompt engineering techniques. An additional response parsing step nearly eliminates format errors while preserving accuracy; larger models deliver the strongest and most stable performance, while quantization offers substantial efficiency gains with modest accuracy trade-offs; and prompt strategies show heterogeneous, model-specific effects. CienaLLM matches or outperforms the supervised baseline in accuracy for extracting drought impacts from Spanish news, although at a higher inference cost. While evaluated in droughts, the schema-driven and model-agnostic design is suitable for adapting to related information extraction tasks (e.g., other hazards, sectors, or languages) by editing prompts and schemas rather than retraining. We release code, configurations, and schemas to support reproducible use. 

---
# Exploring Zero-Shot ACSA with Unified Meaning Representation in Chain-of-Thought Prompting 

**Authors**: Filippos Ventirozos, Peter Appleby, Matthew Shardlow  

**Link**: [PDF](https://arxiv.org/pdf/2512.19651)  

**Abstract**: Aspect-Category Sentiment Analysis (ACSA) provides granular insights by identifying specific themes within reviews and their associated sentiment. While supervised learning approaches dominate this field, the scarcity and high cost of annotated data for new domains present significant barriers. We argue that leveraging large language models (LLMs) in a zero-shot setting is a practical alternative where resources for data annotation are limited. In this work, we propose a novel Chain-of-Thought (CoT) prompting technique that utilises an intermediate Unified Meaning Representation (UMR) to structure the reasoning process for the ACSA task. We evaluate this UMR-based approach against a standard CoT baseline across three models (Qwen3-4B, Qwen3-8B, and Gemini-2.5-Pro) and four diverse datasets. Our findings suggest that UMR effectiveness may be model-dependent. Whilst preliminary results indicate comparable performance for mid-sized models such as Qwen3-8B, these observations warrant further investigation, particularly regarding the potential applicability to smaller model architectures. Further research is required to establish the generalisability of these findings across different model scales. 

---
# Auto-Prompting with Retrieval Guidance for Frame Detection in Logistics 

**Authors**: Do Minh Duc, Quan Xuan Truong, Nguyen Tat Dat, Nguyen Van Vinh  

**Link**: [PDF](https://arxiv.org/pdf/2512.19247)  

**Abstract**: Prompt engineering plays a critical role in adapting large language models (LLMs) to complex reasoning and labeling tasks without the need for extensive fine-tuning. In this paper, we propose a novel prompt optimization pipeline for frame detection in logistics texts, combining retrieval-augmented generation (RAG), few-shot prompting, chain-of-thought (CoT) reasoning, and automatic CoT synthesis (Auto-CoT) to generate highly effective task-specific prompts. Central to our approach is an LLM-based prompt optimizer agent that iteratively refines the prompts using retrieved examples, performance feedback, and internal self-evaluation. Our framework is evaluated on a real-world logistics text annotation task, where reasoning accuracy and labeling efficiency are critical. Experimental results show that the optimized prompts - particularly those enhanced via Auto-CoT and RAG - improve real-world inference accuracy by up to 15% compared to baseline zero-shot or static prompts. The system demonstrates consistent improvements across multiple LLMs, including GPT-4o, Qwen 2.5 (72B), and LLaMA 3.1 (70B), validating its generalizability and practical value. These findings suggest that structured prompt optimization is a viable alternative to full fine-tuning, offering scalable solutions for deploying LLMs in domain-specific NLP applications such as logistics. 

---
# GenEnv: Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators 

**Authors**: Jiacheng Guo, Ling Yang, Peter Chen, Qixin Xiao, Yinjie Wang, Xinzhe Juan, Jiahao Qiu, Ke Shen, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19682)  

**Abstract**: Training capable Large Language Model (LLM) agents is critically bottlenecked by the high cost and static nature of real-world interaction data. We address this by introducing GenEnv, a framework that establishes a difficulty-aligned co-evolutionary game between an agent and a scalable, generative environment simulator. Unlike traditional methods that evolve models on static datasets, GenEnv instantiates a dataevolving: the simulator acts as a dynamic curriculum policy, continuously generating tasks specifically tailored to the agent's ``zone of proximal development''. This process is guided by a simple but effective $\alpha$-Curriculum Reward, which aligns task difficulty with the agent's current capabilities. We evaluate GenEnv on five benchmarks, including API-Bank, ALFWorld, BFCL, Bamboogle, and TravelPlanner. Across these tasks, GenEnv improves agent performance by up to \textbf{+40.3\%} over 7B baselines and matches or exceeds the average performance of larger models. Compared to Gemini 2.5 Pro-based offline data augmentation, GenEnv achieves better performance while using 3.3$\times$ less data. By shifting from static supervision to adaptive simulation, GenEnv provides a data-efficient pathway for scaling agent capabilities. 

---
# AWPO: Enhancing Tool-Use of Large Language Models through Explicit Integration of Reasoning Rewards 

**Authors**: Zihan Lin, Xiaohan Wang, Hexiong Yang, Jiajun Chai, Jie Cao, Guojun Yin, Wei Lin, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2512.19126)  

**Abstract**: While reinforcement learning (RL) shows promise in training tool-use large language models (LLMs) using verifiable outcome rewards, existing methods largely overlook the potential of explicit reasoning rewards to bolster reasoning and tool utilization. Furthermore, natively combining reasoning and outcome rewards may yield suboptimal performance or conflict with the primary optimization objective. To address this, we propose advantage-weighted policy optimization (AWPO) -- a principled RL framework that effectively integrates explicit reasoning rewards to enhance tool-use capability. AWPO incorporates variance-aware gating and difficulty-aware weighting to adaptively modulate advantages from reasoning signals based on group-relative statistics, alongside a tailored clipping mechanism for stable optimization. Extensive experiments demonstrate that AWPO achieves state-of-the-art performance across standard tool-use benchmarks, significantly outperforming strong baselines and leading closed-source models in challenging multi-turn scenarios. Notably, with exceptional parameter efficiency, our 4B model surpasses Grok-4 by 16.0 percent in multi-turn accuracy while preserving generalization capability on the out-of-distribution MMLU-Pro benchmark. 

---
# ChemATP: A Training-Free Chemical Reasoning Framework for Large Language Models 

**Authors**: Mingxu Zhang, Dazhong Shen, Qi Zhang, Ying Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.19240)  

**Abstract**: Large Language Models (LLMs) exhibit strong general reasoning but struggle in molecular science due to the lack of explicit chemical priors in standard string representations. Current solutions face a fundamental dilemma. Training-based methods inject priors into parameters, but this static coupling hinders rapid knowledge updates and often compromises the model's general reasoning capabilities. Conversely, existing training-free methods avoid these issues but rely on surface-level prompting, failing to provide the fine-grained atom-level priors essential for precise chemical reasoning. To address this issue, we introduce ChemATP, a framework that decouples chemical knowledge from the reasoning engine. By constructing the first atom-level textual knowledge base, ChemATP enables frozen LLMs to explicitly retrieve and reason over this information dynamically. This architecture ensures interpretability and adaptability while preserving the LLM's intrinsic general intelligence. Experiments show that ChemATP significantly outperforms training-free baselines and rivals state-of-the-art training-based models, demonstrating that explicit prior injection is a competitive alternative to implicit parameter updates. 

---
# QuCo-RAG: Quantifying Uncertainty from the Pre-training Corpus for Dynamic Retrieval-Augmented Generation 

**Authors**: Dehai Min, Kailin Zhang, Tongtong Wu, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.19134)  

**Abstract**: Dynamic Retrieval-Augmented Generation adaptively determines when to retrieve during generation to mitigate hallucinations in large language models (LLMs). However, existing methods rely on model-internal signals (e.g., logits, entropy), which are fundamentally unreliable because LLMs are typically ill-calibrated and often exhibit high confidence in erroneous outputs. We propose QuCo-RAG, which shifts from subjective confidence to objective statistics computed from pre-training data. Our method quantifies uncertainty through two stages: (1) before generation, we identify low-frequency entities indicating long-tail knowledge gaps; (2) during generation, we verify entity co-occurrence in the pre-training corpus, where zero co-occurrence often signals hallucination risk. Both stages leverage Infini-gram for millisecond-latency queries over 4 trillion tokens, triggering retrieval when uncertainty is high. Experiments on multi-hop QA benchmarks show QuCo-RAG achieves EM gains of 5--12 points over state-of-the-art baselines with OLMo-2 models, and transfers effectively to models with undisclosed pre-training data (Llama, Qwen, GPT), improving EM by up to 14 points. Domain generalization on biomedical QA further validates the robustness of our paradigm. These results establish corpus-grounded verification as a principled, practically model-agnostic paradigm for dynamic RAG. Our code is publicly available at this https URL. 

---
# SiamGPT: Quality-First Fine-Tuning for Stable Thai Text Generation 

**Authors**: Thittipat Pairatsuppawat, Abhibhu Tachaapornchai, Paweekorn Kusolsomboon, Chutikan Chaiwong, Thodsaporn Chay-intr, Kobkrit Viriyayudhakorn, Nongnuch Ketui, Aslan B. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2512.19455)  

**Abstract**: Open-weights large language models remain difficult to deploy for Thai due to unstable generation under complex instructions, despite strong English performance. To mitigate these limitations, We present SiamGPT-32B, an open-weights model based on Qwen3-32B, fine-tuned with a Quality-First strategy emphasizing curated supervision over data scale. The fine-tuning pipeline combines translated high-complexity English instruction data with a Thai-adapted AutoIF framework for instruction and linguistic constraints. Using supervised fine-tuning only, without continual pretraining or corpus expansion, SiamGPT-32B improves instruction adherence, multi-turn robustness, and linguistic stability. Evaluations on the SEA-HELM benchmark show that SiamGPT-32B achieves the strongest overall performance among similar-scale open-weights Thai models, with consistent gains in instruction following, multi-turn dialogue, and natural language understanding. 

---
# FASTRIC: Prompt Specification Language for Verifiable LLM Interactions 

**Authors**: Wen-Long Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.18940)  

**Abstract**: Large Language Models (LLMs) execute complex multi-turn interaction protocols but lack formal specifications to verify execution against designer intent. We introduce FASTRIC, a Prompt Specification Language that makes implicit Finite State Machines (FSMs) explicit in natural language prompts, enabling conformance verification through execution trace analysis. The LLM serves as intelligent execution agent: interpreting designer-encoded FSMs to execute specified behavioral roles. Unlike symbolic specification languages requiring parsers and compilers, FASTRIC leverages LLMs as unified infrastructure-simultaneously parser, interpreter, runtime environment, and development assistant. FASTRIC guides designers to articulate seven FSM elements (Final States, Agents, States, Triggers, Roles, Initial State, Constraints) structuring multi-turn interactions. Specification formality-ranging from implicit descriptions that frontier models infer to explicit step-by-step instructions for weaker models-serves as a design parameter. We introduce procedural conformance as verification metric measuring execution adherence to FSM specifications. Testing a 3-state kindergarten tutoring FSM across four formality levels and three model scales (14.7B, 685B, 1T+ parameters) reveals optimal specification formality is a function of model capacity. DeepSeek-V3.2 (685B) achieves perfect conformance (1.00) at L2-L4; ChatGPT-5 (~1T) peaks at L3 (0.90) before collapsing at L4 (0.39); Phi4 (14.7B) shows no stable optimum with high variance (SD=0.16-0.36). These findings reveal model-specific formality ranges-"Goldilocks zones"-where specifications provide sufficient structure without over-constraint, establishing Prompt Specification Engineering for creating verifiable interaction protocols, transforming multi-turn interaction design from heuristic art to systematic engineering with measurable procedural guarantees. 

---
# Evaluating the Challenges of LLMs in Real-world Medical Follow-up: A Comparative Study and An Optimized Framework 

**Authors**: Jinyan Liu, Zikang Chen, Qinchuan Wang, Tan Xie, Heming Zheng, Xudong Lv  

**Link**: [PDF](https://arxiv.org/pdf/2512.18999)  

**Abstract**: When applied directly in an end-to-end manner to medical follow-up tasks, Large Language Models (LLMs) often suffer from uncontrolled dialog flow and inaccurate information extraction due to the complexity of follow-up forms. To address this limitation, we designed and compared two follow-up chatbot systems: an end-to-end LLM-based system (control group) and a modular pipeline with structured process control (experimental group). Experimental results show that while the end-to-end approach frequently fails on lengthy and complex forms, our modular method-built on task decomposition, semantic clustering, and flow management-substantially improves dialog stability and extraction accuracy. Moreover, it reduces the number of dialogue turns by 46.73% and lowers token consumption by 80% to 87.5%. These findings highlight the necessity of integrating external control mechanisms when deploying LLMs in high-stakes medical follow-up scenarios. 

---
# MDToC: Metacognitive Dynamic Tree of Concepts for Boosting Mathematical Problem-Solving of Large Language Models 

**Authors**: Tung Duong Ta, Tim Oates  

**Link**: [PDF](https://arxiv.org/pdf/2512.18841)  

**Abstract**: Despite advances in mathematical reasoning capabilities, Large Language Models (LLMs) still struggle with calculation verification when using established prompting techniques. We present MDToC (Metacognitive Dynamic Tree of Concepts), a three-phase approach that constructs a concept tree, develops accuracy-verified calculations for each concept, and employs majority voting to evaluate competing solutions. Evaluations across CHAMP, MATH, and Game-of-24 benchmarks demonstrate our MDToC's effectiveness, with GPT-4-Turbo achieving 58.1\% on CHAMP, 86.6\% on MATH, and 85\% on Game-of-24 - outperforming GoT by 5\%, 5.4\%, and 4\% on all these tasks, respectively, without hand-engineered hints. MDToC consistently surpasses existing prompting methods across all backbone models, yielding improvements of up to 7.6\% over ToT and 6.2\% over GoT, establishing metacognitive calculation verification as a promising direction for enhanced mathematical reasoning. 

---
# Can LLMs Estimate Student Struggles? Human-AI Difficulty Alignment with Proficiency Simulation for Item Difficulty Prediction 

**Authors**: Ming Li, Han Chen, Yunze Xiao, Jian Chen, Hong Jiao, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.18880)  

**Abstract**: Accurate estimation of item (question or task) difficulty is critical for educational assessment but suffers from the cold start problem. While Large Language Models demonstrate superhuman problem-solving capabilities, it remains an open question whether they can perceive the cognitive struggles of human learners. In this work, we present a large-scale empirical analysis of Human-AI Difficulty Alignment for over 20 models across diverse domains such as medical knowledge and mathematical reasoning. Our findings reveal a systematic misalignment where scaling up model size is not reliably helpful; instead of aligning with humans, models converge toward a shared machine consensus. We observe that high performance often impedes accurate difficulty estimation, as models struggle to simulate the capability limitations of students even when being explicitly prompted to adopt specific proficiency levels. Furthermore, we identify a critical lack of introspection, as models fail to predict their own limitations. These results suggest that general problem-solving capability does not imply an understanding of human cognitive struggles, highlighting the challenge of using current models for automated difficulty prediction. 

---
# A Large Language Model Based Method for Complex Logical Reasoning over Knowledge Graphs 

**Authors**: Ziyan Zhang, Chao Wang, Zhuo Chen, Lei Chen, Chiyi Li, Kai Song  

**Link**: [PDF](https://arxiv.org/pdf/2512.19092)  

**Abstract**: Reasoning over knowledge graphs (KGs) with first-order logic (FOL) queries is challenging due to the inherent incompleteness of real-world KGs and the compositional complexity of logical query structures. Most existing methods rely on embedding entities and relations into continuous geometric spaces and answer queries via differentiable set operations. While effective for simple query patterns, these approaches often struggle to generalize to complex queries involving multiple operators, deeper reasoning chains, or heterogeneous KG schemas. We propose ROG (Reasoning Over knowledge Graphs with large language models), an ensemble-style framework that combines query-aware KG neighborhood retrieval with large language model (LLM)-based chain-of-thought reasoning. ROG decomposes complex FOL queries into sequences of simpler sub-queries, retrieves compact, query-relevant subgraphs as contextual evidence, and performs step-by-step logical inference using an LLM, avoiding the need for task-specific embedding optimization. Experiments on standard KG reasoning benchmarks demonstrate that ROG consistently outperforms strong embedding-based baselines in terms of mean reciprocal rank (MRR), with particularly notable gains on high-complexity query types. These results suggest that integrating structured KG retrieval with LLM-driven logical reasoning offers a robust and effective alternative for complex KG reasoning tasks. 

---
# Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design 

**Authors**: Yuchen Li, Handing Wang, Bing Xue, Mengjie Zhang, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.18682)  

**Abstract**: In the high-cost simulation-driven design domain, translating ambiguous design requirements into a mathematical optimization formulation is a bottleneck for optimizing product performance. This process is time-consuming and heavily reliant on expert knowledge. While large language models (LLMs) offer potential for automating this task, existing approaches either suffer from poor formalization that fails to accurately align with the design intent or rely on solver feedback for data filtering, which is unavailable due to the high simulation costs. To address this challenge, we propose APF, a framework for solver-independent, automated problem formulation via LLMs designed to automatically convert engineers' natural language requirements into executable optimization models. The core of this framework is an innovative pipeline for automatically generating high-quality data, which overcomes the difficulty of constructing suitable fine-tuning datasets in the absence of high-cost solver feedback with the help of data generation and test instance annotation. The generated high-quality dataset is used to perform supervised fine-tuning on LLMs, significantly enhancing their ability to generate accurate and executable optimization problem formulations. Experimental results on antenna design demonstrate that APF significantly outperforms the existing methods in both the accuracy of requirement formalization and the quality of resulting radiation efficiency curves in meeting the design goals. 

---
# LLM-CAS: Dynamic Neuron Perturbation for Real-Time Hallucination Correction 

**Authors**: Jensen Zhang, Ningyuan Liu, Yijia Fan, Zihao Huang, Qinglin Zeng, Kaitong Cai, Jian Wang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18623)  

**Abstract**: Large language models (LLMs) often generate hallucinated content that lacks factual or contextual grounding, limiting their reliability in critical applications. Existing approaches such as supervised fine-tuning and reinforcement learning from human feedback are data intensive and computationally expensive, while static parameter editing methods struggle with context dependent errors and catastrophic forgetting.
We propose LLM-CAS, a framework that formulates real-time hallucination correction as a hierarchical reinforcement learning problem. LLM-CAS trains an agent to learn a policy that dynamically selects temporary neuron perturbations during inference based on the current context. Unlike prior dynamic approaches that rely on heuristic or predefined adjustments, this policy driven mechanism enables adaptive and fine grained correction without permanent parameter modification.
Experiments across multiple language models demonstrate that LLM-CAS consistently improves factual accuracy, achieving gains of 10.98 percentage points on StoryCloze, 2.71 points on TriviaQA, and 2.06 points on the MC1 score of TruthfulQA. These results outperform both static editing methods such as ITI and CAA and the dynamic SADI framework. Overall, LLM-CAS provides an efficient and context aware solution for improving the reliability of LLMs, with promising potential for future multimodal extensions. 

---
# Remedy-R: Generative Reasoning for Machine Translation Evaluation without Error Annotations 

**Authors**: Shaomu Tan, Ryosuke Mitani, Ritvik Choudhary, Qiyu Wu, Toshiyuki Sekiya, Christof Monz  

**Link**: [PDF](https://arxiv.org/pdf/2512.18906)  

**Abstract**: Over the years, automatic MT metrics have hillclimbed benchmarks and presented strong and sometimes human-level agreement with human ratings. Yet they remain black-box, offering little insight into their decision-making and often failing under real-world out-of-distribution (OOD) inputs. We introduce Remedy-R, a reasoning-driven generative MT metric trained with reinforcement learning from pairwise translation preferences, without requiring error-span annotations or distillation from closed LLMs. Remedy-R produces step-by-step analyses of accuracy, fluency, and completeness, followed by a final score, enabling more interpretable assessments. With only 60K training pairs across two language pairs, Remedy-R remains competitive with top scalar metrics and GPT-4-based judges on WMT22-24 meta-evaluation, generalizes to other languages, and exhibits strong robustness on OOD stress tests. Moreover, Remedy-R models generate self-reflective feedback that can be reused for translation improvement. Building on this finding, we introduce Remedy-R Agent, a simple evaluate-revise pipeline that leverages Remedy-R's evaluation analysis to refine translations. This agent consistently improves translation quality across diverse models, including Qwen2.5, ALMA-R, GPT-4o-mini, and Gemini-2.0-Flash, suggesting that Remedy-R's reasoning captures translation-relevant information and is practically useful. 

---
# From Word to World: Can Large Language Models be Implicit Text-based World Models? 

**Authors**: Yixia Li, Hongru Wang, Jiahao Qiu, Zhenfei Yin, Dongdong Zhang, Cheng Qian, Zeping Li, Pony Ma, Guanhua Chen, Heng Ji, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18832)  

**Abstract**: Agentic reinforcement learning increasingly relies on experience-driven scaling, yet real-world environments remain non-adaptive, limited in coverage, and difficult to scale. World models offer a potential way to improve learning efficiency through simulated experience, but it remains unclear whether large language models can reliably serve this role and under what conditions they meaningfully benefit agents. We study these questions in text-based environments, which provide a controlled setting to reinterpret language modeling as next-state prediction under interaction. We introduce a three-level framework for evaluating LLM-based world models: (i) fidelity and consistency, (ii) scalability and robustness, and (iii) agent utility. Across five representative environments, we find that sufficiently trained world models maintain coherent latent state, scale predictably with data and model size, and improve agent performance via action verification, synthetic trajectory generation, and warm-starting reinforcement learning. Meanwhile, these gains depend critically on behavioral coverage and environment complexity, delineating clear boundry on when world modeling effectively supports agent learning. 

---
# LLMs on Drugs: Language Models Are Few-Shot Consumers 

**Authors**: Alexander Doudkin  

**Link**: [PDF](https://arxiv.org/pdf/2512.18546)  

**Abstract**: Large language models (LLMs) are sensitive to the personas imposed on them at inference time, yet prompt-level "drug" interventions have never been benchmarked rigorously. We present the first controlled study of psychoactive framings on GPT-5-mini using ARC-Challenge. Four single-sentence prompts -- LSD, cocaine, alcohol, and cannabis -- are compared against a sober control across 100 validation items per condition, with deterministic decoding, full logging, Wilson confidence intervals, and Fisher exact tests. Control accuracy is 0.45; alcohol collapses to 0.10 (p = 3.2e-8), cocaine to 0.21 (p = 4.9e-4), LSD to 0.19 (p = 1.3e-4), and cannabis to 0.30 (p = 0.041), largely because persona prompts disrupt the mandated "Answer: <LETTER>" template. Persona text therefore behaves like a "few-shot consumable" that can destroy reliability without touching model weights. All experimental code, raw results, and analysis scripts are available at this https URL. 

---
# SRS-Stories: Vocabulary-constrained multilingual story generation for language learning 

**Authors**: Wiktor Kamzela, Mateusz Lango, Ondrej Dusek  

**Link**: [PDF](https://arxiv.org/pdf/2512.18362)  

**Abstract**: In this paper, we use large language models to generate personalized stories for language learners, using only the vocabulary they know. The generated texts are specifically written to teach the user new vocabulary by simply reading stories where it appears in context, while at the same time seamlessly reviewing recently learned vocabulary. The generated stories are enjoyable to read and the vocabulary reviewing/learning is optimized by a Spaced Repetition System. The experiments are conducted in three languages: English, Chinese and Polish, evaluating three story generation methods and three strategies for enforcing lexical constraints. The results show that the generated stories are more grammatical, coherent, and provide better examples of word usage than texts generated by the standard constrained beam search approach 

---
# LLM-based Few-Shot Early Rumor Detection with Imitation Agent 

**Authors**: Fengzhu Zeng, Qian Shao, Ling Cheng, Wei Gao, Shih-Fen Cheng, Jing Ma, Cheng Niu  

**Link**: [PDF](https://arxiv.org/pdf/2512.18352)  

**Abstract**: Early Rumor Detection (EARD) aims to identify the earliest point at which a claim can be accurately classified based on a sequence of social media posts. This is especially challenging in data-scarce settings. While Large Language Models (LLMs) perform well in few-shot NLP tasks, they are not well-suited for time-series data and are computationally expensive for both training and inference. In this work, we propose a novel EARD framework that combines an autonomous agent and an LLM-based detection model, where the agent acts as a reliable decision-maker for \textit{early time point determination}, while the LLM serves as a powerful \textit{rumor detector}. This approach offers the first solution for few-shot EARD, necessitating only the training of a lightweight agent and allowing the LLM to remain training-free. Extensive experiments on four real-world datasets show our approach boosts performance across LLMs and surpasses existing EARD methods in accuracy and earliness. 

---
# DACE For Railway Acronym Disambiguation 

**Authors**: El Mokhtar Hribach, Oussama Mechhour, Mohammed Elmonstaser, Yassine El Boudouri, Othmane Kabal  

**Link**: [PDF](https://arxiv.org/pdf/2512.18357)  

**Abstract**: Acronym Disambiguation (AD) is a fundamental challenge in technical text processing, particularly in specialized sectors where high ambiguity complicates automated analysis. This paper addresses AD within the context of the TextMine'26 competition on French railway documentation. We present DACE (Dynamic Prompting, Retrieval Augmented Generation, Contextual Selection, and Ensemble Aggregation), a framework that enhances Large Language Models through adaptive in-context learning and external domain knowledge injection. By dynamically tailoring prompts to acronym ambiguity and aggregating ensemble predictions, DACE mitigates hallucination and effectively handles low-resource scenarios. Our approach secured the top rank in the competition with an F1 score of 0.9069. 

---
# Mitigating Spurious Correlations in NLI via LLM-Synthesized Counterfactuals and Dynamic Balanced Sampling 

**Authors**: Christopher Román Jaimes  

**Link**: [PDF](https://arxiv.org/pdf/2512.18462)  

**Abstract**: Natural Language Inference (NLI) models frequently rely on spurious correlations rather than semantic reasoning. Existing mitigation strategies often incur high annotation costs or trigger catastrophic forgetting during fine-tuning. We propose an automated, scalable pipeline to address these limitations. First, we introduce Log-Frequency LMI (LF-LMI) to accurately detect semantic artifacts. Second, we generate a high-quality synthetic contrast set via an LLM-synthesis pipeline with multi-judge verification. Finally, we introduce Dynamic Balanced Sampling, a training strategy that rotates the original data distribution to prevent forgetting. Our method improves consistency on a challenging benchmark from 63.5% to 81.0% while maintaining 88.4% in-domain accuracy, significantly outperforming naive fine-tuning. 

---
# LIR$^3$AG: A Lightweight Rerank Reasoning Strategy Framework for Retrieval-Augmented Generation 

**Authors**: Guo Chen, Junjie Huang, Huaijin Xie, Fei Sun, Tao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2512.18329)  

**Abstract**: Retrieval-Augmented Generation (RAG) effectively enhances Large Language Models (LLMs) by incorporating retrieved external knowledge into the generation process. Reasoning models improve LLM performance in multi-hop QA tasks, which require integrating and reasoning over multiple pieces of evidence across different documents to answer a complex question. However, they often introduce substantial computational costs, including increased token consumption and inference latency. To better understand and mitigate this trade-off, we conduct a comprehensive study of reasoning strategies for reasoning models in RAG multi-hop QA tasks. Our findings reveal that reasoning models adopt structured strategies to integrate retrieved and internal knowledge, primarily following two modes: Context-Grounded Reasoning, which relies directly on retrieved content, and Knowledge-Reconciled Reasoning, which resolves conflicts or gaps using internal knowledge. To this end, we propose a novel Lightweight Rerank Reasoning Strategy Framework for RAG (LiR$^3$AG) to enable non-reasoning models to transfer reasoning strategies by restructuring retrieved evidence into coherent reasoning chains. LiR$^3$AG significantly reduce the average 98% output tokens overhead and 58.6% inferencing time while improving 8B non-reasoning model's F1 performance ranging from 6.2% to 22.5% to surpass the performance of 32B reasoning model in RAG, offering a practical and efficient path forward for RAG systems. 

---
# MemEvolve: Meta-Evolution of Agent Memory Systems 

**Authors**: Guibin Zhang, Haotian Ren, Chong Zhan, Zhenhong Zhou, Junhao Wang, He Zhu, Wangchunshu Zhou, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2512.18746)  

**Abstract**: Self-evolving memory systems are unprecedentedly reshaping the evolutionary paradigm of large language model (LLM)-based agents. Prior work has predominantly relied on manually engineered memory architectures to store trajectories, distill experience, and synthesize reusable tools, enabling agents to evolve on the fly within environment interactions. However, this paradigm is fundamentally constrained by the staticity of the memory system itself: while memory facilitates agent-level evolving, the underlying memory architecture cannot be meta-adapted to diverse task contexts. To address this gap, we propose MemEvolve, a meta-evolutionary framework that jointly evolves agents' experiential knowledge and their memory architecture, allowing agent systems not only to accumulate experience but also to progressively refine how they learn from it. To ground MemEvolve in prior research and foster openness in future self-evolving systems, we introduce EvolveLab, a unified self-evolving memory codebase that distills twelve representative memory systems into a modular design space (encode, store, retrieve, manage), providing both a standardized implementation substrate and a fair experimental arena. Extensive evaluations on four challenging agentic benchmarks demonstrate that MemEvolve achieves (I) substantial performance gains, improving frameworks such as SmolAgent and Flash-Searcher by up to $17.06\%$; and (II) strong cross-task and cross-LLM generalization, designing memory architectures that transfer effectively across diverse benchmarks and backbone models. 

---
# Training LLMs with LogicReward for Faithful and Rigorous Reasoning 

**Authors**: Jundong Xu, Hao Fei, Huichi Zhou, Xin Quan, Qijun Huang, Shengqiong Wu, William Yang Wang, Mong-Li Lee, Wynne Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2512.18196)  

**Abstract**: Although LLMs exhibit strong reasoning capabilities, existing training methods largely depend on outcome-based feedback, which can produce correct answers with flawed reasoning. Prior work introduces supervision on intermediate steps but still lacks guarantees of logical soundness, which is crucial in high-stakes scenarios where logical consistency is paramount. To address this, we propose LogicReward, a novel reward system that guides model training by enforcing step-level logical correctness with a theorem prover. We further introduce Autoformalization with Soft Unification, which reduces natural language ambiguity and improves formalization quality, enabling more effective use of the theorem prover. An 8B model trained on data constructed with LogicReward surpasses GPT-4o and o4-mini by 11.6\% and 2\% on natural language inference and logical reasoning tasks with simple training procedures. Further analysis shows that LogicReward enhances reasoning faithfulness, improves generalizability to unseen tasks such as math and commonsense reasoning, and provides a reliable reward signal even without ground-truth labels. We will release all data and code at this https URL. 

---
# LLM Agents Implement an NLG System from Scratch: Building Interpretable Rule-Based RDF-to-Text Generators 

**Authors**: Mateusz Lango, Ondřej Dušek  

**Link**: [PDF](https://arxiv.org/pdf/2512.18360)  

**Abstract**: We present a novel neurosymbolic framework for RDF-to-text generation, in which the model is "trained" through collaborative interactions among multiple LLM agents rather than traditional backpropagation. The LLM agents produce rule-based Python code for a generator for the given domain, based on RDF triples only, with no in-domain human reference texts. The resulting system is fully interpretable, requires no supervised training data, and generates text nearly instantaneously using only a single CPU. Our experiments on the WebNLG and OpenDialKG data show that outputs produced by our approach reduce hallucination, with only slight fluency penalties compared to finetuned or prompted language models 

---
# Towards Efficient Agents: A Co-Design of Inference Architecture and System 

**Authors**: Weizhe Lin, Hui-Ling Zhen, Shuai Yang, Xian Wang, Renxi Liu, Hanting Chen, Wangze Zhang, Chuansai Zhou, Yiming Li, Chen Chen, Xing Li, Zhiyuan Yang, Xiaosong Li, Xianzhi Yu, Zhenhua Dong, Mingxuan Yuan, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18337)  

**Abstract**: The rapid development of large language model (LLM)-based agents has unlocked new possibilities for autonomous multi-turn reasoning and tool-augmented decision-making. However, their real-world deployment is hindered by severe inefficiencies that arise not from isolated model inference, but from the systemic latency accumulated across reasoning loops, context growth, and heterogeneous tool interactions. This paper presents AgentInfer, a unified framework for end-to-end agent acceleration that bridges inference optimization and architectural design. We decompose the problem into four synergistic components: AgentCollab, a hierarchical dual-model reasoning framework that balances large- and small-model usage through dynamic role assignment; AgentSched, a cache-aware hybrid scheduler that minimizes latency under heterogeneous request patterns; AgentSAM, a suffix-automaton-based speculative decoding method that reuses multi-session semantic memory to achieve low-overhead inference acceleration; and AgentCompress, a semantic compression mechanism that asynchronously distills and reorganizes agent memory without disrupting ongoing reasoning. Together, these modules form a Self-Evolution Engine capable of sustaining efficiency and cognitive stability throughout long-horizon reasoning tasks. Experiments on the BrowseComp-zh and DeepDiver benchmarks demonstrate that through the synergistic collaboration of these methods, AgentInfer reduces ineffective token consumption by over 50%, achieving an overall 1.8-2.5 times speedup with preserved accuracy. These results underscore that optimizing for agentic task completion-rather than merely per-token throughput-is the key to building scalable, efficient, and self-improving intelligent systems. 

---
# Q-KVComm: Efficient Multi-Agent Communication Via Adaptive KV Cache Compression 

**Authors**: Boris Kriuk, Logic Ng  

**Link**: [PDF](https://arxiv.org/pdf/2512.17914)  

**Abstract**: Multi-agent Large Language Model (LLM) systems face a critical bottleneck: redundant transmission of contextual information between agents consumes excessive bandwidth and computational resources. Traditional approaches discard internal semantic representations and transmit raw text, forcing receiving agents to recompute similar representations from scratch. We introduce Q-KVComm, a new protocol that enables direct transmission of compressed key-value (KV) cache representations between LLM agents. Q-KVComm combines three key innovations: (1) adaptive layer-wise quantization that allocates variable bit-widths based on sensitivity profiling, (2) hybrid information extraction that preserves critical facts across content domains, and (3) heterogeneous model calibration establishing cross-architecture communication. Extensive experiments across three diverse question-answering datasets demonstrate that Q-KVComm achieves 5-6x compression ratios while maintaining semantic fidelity, with coherence quality scores above 0.77 across all scenarios. The protocol exhibits robust performance across model sizes (1.1B-1.5B parameters) and adapts to real-world applications including conversational QA and multi-hop reasoning. Our work establishes a new paradigm for LLM agent communication, shifting from text-based to representation-based information exchange. 

---
# Towards Reasoning-Preserving Unlearning in Multimodal Large Language Models 

**Authors**: Hongji Li, Junchi yao, Manjiang Yu, Priyanka Singh, Xue Li, Di Wang, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2512.17911)  

**Abstract**: Machine unlearning aims to erase requested data from trained models without full retraining. For Reasoning Multimodal Large Language Models (RMLLMs), this is uniquely challenging: intermediate chain-of-thought steps can still leak sensitive information even when final answers are forgotten, and overly aggressive interventions easily damage general reasoning ability. Yet no benchmark jointly evaluates how well unlearning methods suppress reasoning-level leakage while preserving reasoning competence. We address this gap with RMLLMU-Bench, the first benchmark for RMLLM unlearning that extends standard forgetting metrics with dedicated measures of reasoning leakage and reasoning retention. A systematic evaluation on RMLLMU-Bench reveals that existing unlearning methods for MLLMs and Large (Language) Reasoning Models (LRMs) either leave substantial leakage in the reasoning process or severely degrade reasoning performance. To address these gaps, we propose R-MUSE (Reasoning-preserving MLLM Unlearning via Subspace guidance and Adaptive Steering), a training-free and inference-time intervention framework that steers internal representations to forget both answers and reasoning traces while explicitly preserving general reasoning. Experiments on RMLLMU-Bench demonstrate that R-MUSE achieves a substantially better balance between effective forgetting and reasoning retention. 

---
# From Retrieval to Reasoning: A Framework for Cyber Threat Intelligence NER with Explicit and Adaptive Instructions 

**Authors**: Jiaren Peng, Hongda Sun, Xuan Tian, Cheng Huang, Zeqing Li, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2512.19414)  

**Abstract**: The automation of Cyber Threat Intelligence (CTI) relies heavily on Named Entity Recognition (NER) to extract critical entities from unstructured text. Currently, Large Language Models (LLMs) primarily address this task through retrieval-based In-Context Learning (ICL). This paper analyzes this mainstream paradigm, revealing a fundamental flaw: its success stems not from global semantic similarity but largely from the incidental overlap of entity types within retrieved examples. This exposes the limitations of relying on unreliable implicit induction. To address this, we propose TTPrompt, a framework shifting from implicit induction to explicit instruction. TTPrompt maps the core concepts of CTI's Tactics, Techniques, and Procedures (TTPs) into an instruction hierarchy: formulating task definitions as Tactics, guiding strategies as Techniques, and annotation guidelines as Procedures. Furthermore, to handle the adaptability challenge of static guidelines, we introduce Feedback-driven Instruction Refinement (FIR). FIR enables LLMs to self-refine guidelines by learning from errors on minimal labeled data, adapting to distinct annotation dialects. Experiments on five CTI NER benchmarks demonstrate that TTPrompt consistently surpasses retrieval-based baselines. Notably, with refinement on just 1% of training data, it rivals models fine-tuned on the full dataset. For instance, on LADDER, its Micro F1 of 71.96% approaches the fine-tuned baseline, and on the complex CTINexus, its Macro F1 exceeds the fine-tuned ACLM model by 10.91%. 

---
# Bottom-up Policy Optimization: Your Language Model Policy Secretly Contains Internal Policies 

**Authors**: Yuqiao Tan, Minzheng Wang, Shizhu He, Huanxuan Liao, Chengfeng Zhao, Qiunan Lu, Tian Liang, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19673)  

**Abstract**: Existing reinforcement learning (RL) approaches treat large language models (LLMs) as a single unified policy, overlooking their internal mechanisms. Understanding how policy evolves across layers and modules is therefore crucial for enabling more targeted optimization and raveling out complex reasoning mechanisms. In this paper, we decompose the language model policy by leveraging the intrinsic split of the Transformer residual stream and the equivalence between the composition of hidden states with the unembedding matrix and the resulting samplable policy. This decomposition reveals Internal Layer Policies, corresponding to contributions from individual layers, and Internal Modular Policies, which align with the self-attention and feed-forward network (FFN) components within each layer. By analyzing the entropy of internal policy, we find that: (a) Early layers keep high entropy for exploration, top layers converge to near-zero entropy for refinement, with convergence patterns varying across model series. (b) LLama's prediction space rapidly converges in the final layer, whereas Qwen-series models, especially Qwen3, exhibit a more human-like, progressively structured reasoning pattern. Motivated by these findings, we propose Bottom-up Policy Optimization (BuPO), a novel RL paradigm that directly optimizes the internal layer policy during early training. By aligning training objective at lower layer, BuPO reconstructs foundational reasoning capabilities and achieves superior performance. Extensive experiments on complex reasoning benchmarks demonstrates the effectiveness of our method. Our code is available at this https URL. 

---
# BanglaForge: LLM Collaboration with Self-Refinement for Bangla Code Generation 

**Authors**: Mahir Labib Dihan, Sadif Ahmed, Md Nafiu Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2512.19122)  

**Abstract**: Bangla is a low-resource language for code generation, lacking large-scale annotated datasets and tools to transform natural language specifications into executable programs. This makes Bangla-to-code generation a challenging task requiring innovative solutions. To address this, we introduce BanglaForge, a novel framework for generating code from Bangla function descriptions. BanglaForge leverages a retrieval-augmented dual-model collaboration paradigm with self-refinement, combining in-context learning, llm-based translation, systematic prompt engineering, and iterative self-refinement based on execution feedback, where a coder generates initial solutions and a reviewer enhances them for robustness. On the BLP-2025 Bangla Code Generation benchmark, BanglaForge achieves a competitive Pass@1 accuracy of 84.00%, demonstrating the effectiveness of retrieval, model collaboration, and self-refinement for low-resource Bangla code generation. 

---
# Toward Training Superintelligent Software Agents through Self-Play SWE-RL 

**Authors**: Yuxiang Wei, Zhiqing Sun, Emily McMilin, Jonas Gehring, David Zhang, Gabriel Synnaeve, Daniel Fried, Lingming Zhang, Sida Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18552)  

**Abstract**: While current software agents powered by large language models (LLMs) and agentic reinforcement learning (RL) can boost programmer productivity, their training data (e.g., GitHub issues and pull requests) and environments (e.g., pass-to-pass and fail-to-pass tests) heavily depend on human knowledge or curation, posing a fundamental barrier to superintelligence. In this paper, we present Self-play SWE-RL (SSR), a first step toward training paradigms for superintelligent software agents. Our approach takes minimal data assumptions, only requiring access to sandboxed repositories with source code and installed dependencies, with no need for human-labeled issues or tests. Grounded in these real-world codebases, a single LLM agent is trained via reinforcement learning in a self-play setting to iteratively inject and repair software bugs of increasing complexity, with each bug formally specified by a test patch rather than a natural language issue description. On the SWE-bench Verified and SWE-Bench Pro benchmarks, SSR achieves notable self-improvement (+10.4 and +7.8 points, respectively) and consistently outperforms the human-data baseline over the entire training trajectory, despite being evaluated on natural language issues absent from self-play. Our results, albeit early, suggest a path where agents autonomously gather extensive learning experiences from real-world software repositories, ultimately enabling superintelligent systems that exceed human capabilities in understanding how systems are constructed, solving novel challenges, and autonomously creating new software from scratch. 

---
# Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models 

**Authors**: Shubham Kumar Nigam, Parjanya Aditya Shukla, Noel Shallum, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2512.18004)  

**Abstract**: Handwritten text recognition (HTR) and machine translation continue to pose significant challenges, particularly for low-resource languages like Marathi, which lack large digitized corpora and exhibit high variability in handwriting styles. The conventional approach to address this involves a two-stage pipeline: an OCR system extracts text from handwritten images, which is then translated into the target language using a machine translation model. In this work, we explore and compare the performance of traditional OCR-MT pipelines with Vision Large Language Models that aim to unify these stages and directly translate handwritten text images in a single, end-to-end step. Our motivation is grounded in the urgent need for scalable, accurate translation systems to digitize legal records such as FIRs, charge sheets, and witness statements in India's district and high courts. We evaluate both approaches on a curated dataset of handwritten Marathi legal documents, with the goal of enabling efficient legal document processing, even in low-resource environments. Our findings offer actionable insights toward building robust, edge-deployable solutions that enhance access to legal information for non-native speakers and legal professionals alike. 

---
# Graph-O1 : Monte Carlo Tree Search with Reinforcement Learning for Text-Attributed Graph Reasoning 

**Authors**: Lihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.17912)  

**Abstract**: ChatGPT said: Text-attributed graphs, where nodes and edges contain rich textual information, are widely used across diverse domains. A central challenge in this setting is question answering, which requires jointly leveraging unstructured text and the structured relational signals within the graph. Although Large Language Models (LLMs) have made significant advances in natural language understanding, their direct use for reasoning over text-attributed graphs remains limited. Retrieval-augmented generation methods that operate purely on text often treat passages as isolated units, ignoring the interconnected structure of the graph. Conversely, graph-based RAG methods that serialize large subgraphs into long textual sequences quickly become infeasible due to LLM context-length constraints, resulting in fragmented reasoning and degraded accuracy. To overcome these limitations, we introduce Graph-O1, an agentic GraphRAG framework that enables LLMs to conduct stepwise, interactive reasoning over graphs. Our approach integrates Monte Carlo Tree Search (MCTS) with end-to-end reinforcement learning, allowing the model to selectively explore and retrieve only the most informative subgraph components. The reasoning procedure is framed as a multi-turn interaction between the agent and the graph environment, and the agent is trained through a unified reward mechanism. Extensive experiments across multiple LLM backbones demonstrate that Graph-O1 consistently surpasses state-of-the-art baselines, producing answers that are more accurate, reliable, and interpretable. 

---
# VIGOR+: Iterative Confounder Generation and Validation via LLM-CEVAE Feedback Loop 

**Authors**: JiaWei Zhu, ZiHeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19349)  

**Abstract**: Hidden confounding remains a fundamental challenge in causal inference from observational data. Recent advances leverage Large Language Models (LLMs) to generate plausible hidden confounders based on domain knowledge, yet a critical gap exists: LLM-generated confounders often exhibit semantic plausibility without statistical utility. We propose VIGOR+ (Variational Information Gain for iterative cOnfounder Refinement), a novel framework that closes the loop between LLM-based confounder generation and CEVAE-based statistical validation. Unlike prior approaches that treat generation and validation as separate stages, VIGOR+ establishes an iterative feedback mechanism: validation signals from CEVAE (including information gain, latent consistency metrics, and diagnostic messages) are transformed into natural language feedback that guides subsequent LLM generation rounds. This iterative refinement continues until convergence criteria are met. We formalize the feedback mechanism, prove convergence properties under mild assumptions, and provide a complete algorithmic framework. 

---
# An Agentic Framework for Autonomous Materials Computation 

**Authors**: Zeyu Xia, Jinzhe Ma, Congjie Zheng, Shufei Zhang, Yuqiang Li, Hang Su, P. Hu, Changshui Zhang, Xingao Gong, Wanli Ouyang, Lei Bai, Dongzhan Zhou, Mao Su  

**Link**: [PDF](https://arxiv.org/pdf/2512.19458)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for accelerating scientific discovery, yet their static knowledge and hallucination issues hinder autonomous research applications. Recent advances integrate LLMs into agentic frameworks, enabling retrieval, reasoning, and tool use for complex scientific workflows. Here, we present a domain-specialized agent designed for reliable automation of first-principles materials computations. By embedding domain expertise, the agent ensures physically coherent multi-step workflows and consistently selects convergent, well-posed parameters, thereby enabling reliable end-to-end computational execution. A new benchmark of diverse computational tasks demonstrates that our system significantly outperforms standalone LLMs in both accuracy and robustness. This work establishes a verifiable foundation for autonomous computational experimentation and represents a key step toward fully automated scientific discovery. 

---
# Scalably Enhancing the Clinical Validity of a Task Benchmark with Physician Oversight 

**Authors**: Junze Ye, Daniel Tawfik, Alex J. Goodell, Nikhil V. Kotha, Mark K. Buyyounouski, Mohsen Bayati  

**Link**: [PDF](https://arxiv.org/pdf/2512.19691)  

**Abstract**: Automating the calculation of clinical risk scores offers a significant opportunity to reduce physician administrative burden and enhance patient care. The current standard for evaluating this capability is MedCalc-Bench, a large-scale dataset constructed using LLM-based feature extraction and rule-based aggregation. However, treating such model-generated benchmarks as static oracles risks enshrining historical model errors as evaluation gold standards, a problem dangerously amplified when these datasets serve as reward signals for Reinforcement Learning (RL). In this work, we propose viewing benchmarks for complex tasks such as clinical score computation as ''in-progress living documents'' that should be periodically re-evaluated as the processes for creating them improve. We introduce a systematic, physician-in-the-loop pipeline that leverages advanced agentic verifiers to audit and relabel MedCalc-Bench, utilizing automated triage to reserve scarce clinician attention for the most contentious instances. Our audit reveals that a notable fraction of original labels diverge from medical ground truth due to extraction errors, calculator logic mismatches, and clinical ambiguity. To study whether this label noise meaningfully impacts downstream RL training, we fine-tune a Qwen3-8B model via Group Relative Policy Optimization (GRPO) and demonstrate that training on corrected labels yields an 8.7% absolute improvement in accuracy over the original baseline -- validating that label noise materially affects model evaluation. These findings underscore that in safety-critical domains, rigorous benchmark maintenance is a prerequisite for genuine model alignment. 

---
# Helios: A Foundational Language Model for Smart Energy Knowledge Reasoning and Application 

**Authors**: Haoyu Jiang, Fanjie Zeng, Boan Qu, Xiaojie Lin, Wei Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2512.19299)  

**Abstract**: In the global drive toward carbon neutrality, deeply coordinated smart energy systems underpin industrial transformation. However, the interdisciplinary, fragmented, and fast-evolving expertise in this domain prevents general-purpose LLMs, which lack domain knowledge and physical-constraint awareness, from delivering precise engineering-aligned inference and generation. To address these challenges, we introduce Helios, a large language model tailored to the smart energy domain, together with a comprehensive suite of resources to advance LLM research in this field. Specifically, we develop Enersys, a multi-agent collaborative framework for end-to-end dataset construction, through which we produce: (1) a smart energy knowledge base, EnerBase, to enrich the model's foundational expertise; (2) an instruction fine-tuning dataset, EnerInstruct, to strengthen performance on domain-specific downstream tasks; and (3) an RLHF dataset, EnerReinforce, to align the model with human preferences and industry standards. Leveraging these resources, Helios undergoes large-scale pretraining, SFT, and RLHF. We also release EnerBench, a benchmark for evaluating LLMs in smart energy scenarios, and demonstrate that our approach significantly enhances domain knowledge mastery, task execution accuracy, and alignment with human preferences. 

---
# Generation of Programmatic Rules for Document Forgery Detection Using Large Language Models 

**Authors**: Valentin Schmidberger, Manuel Eberhardinger, Setareh Maghsudi, Johannes Maucher  

**Link**: [PDF](https://arxiv.org/pdf/2512.19228)  

**Abstract**: Document forgery poses a growing threat to legal, economic, and governmental processes, requiring increasingly sophisticated verification mechanisms. One approach involves the use of plausibility checks, rule-based procedures that assess the correctness and internal consistency of data, to detect anomalies or signs of manipulation. Although these verification procedures are essential for ensuring data integrity, existing plausibility checks are manually implemented by software engineers, which is time-consuming. Recent advances in code generation with large language models (LLMs) offer new potential for automating and scaling the generation of these checks. However, adapting LLMs to the specific requirements of an unknown domain remains a significant challenge. This work investigates the extent to which LLMs, adapted on domain-specific code and data through different fine-tuning strategies, can generate rule-based plausibility checks for forgery detection on constrained hardware resources. We fine-tune open-source LLMs, Llama 3.1 8B and OpenCoder 8B, on structured datasets derived from real-world application scenarios and evaluate the generated plausibility checks on previously unseen forgery patterns. The results demonstrate that the models are capable of generating executable and effective verification procedures. This also highlights the potential of LLMs as scalable tools to support human decision-making in security-sensitive contexts where comprehensibility is required. 

---
# Observer, Not Player: Simulating Theory of Mind in LLMs through Game Observation 

**Authors**: Jerry Wang, Ting Yiu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19210)  

**Abstract**: We present an interactive framework for evaluating whether large language models (LLMs) exhibit genuine "understanding" in a simple yet strategic environment. As a running example, we focus on Rock-Paper-Scissors (RPS), which, despite its apparent simplicity, requires sequential reasoning, adaptation, and strategy recognition. Our system positions the LLM as an Observer whose task is to identify which strategies are being played and to articulate the reasoning behind this judgment. The purpose is not to test knowledge of Rock-Paper-Scissors itself, but to probe whether the model can exhibit mind-like reasoning about sequential behavior. To support systematic evaluation, we provide a benchmark consisting of both static strategies and lightweight dynamic strategies specified by well-prompted rules. We quantify alignment between the Observer's predictions and the ground-truth distributions induced by actual strategy pairs using three complementary signals: Cross-Entropy, Brier score, and Expected Value (EV) discrepancy. These metrics are further integrated into a unified score, the Union Loss, which balances calibration, sensitivity, and payoff alignment. Together with a Strategy Identification Rate (SIR) metric, our framework captures not only predictive accuracy but also whether the model can stably identify the latent strategies in play. The demo emphasizes interactivity, transparency, and reproducibility. Users can adjust LLM distributions in real time, visualize losses as they evolve, and directly inspect reasoning snippets to identify where and why failures occur. In doing so, our system provides a practical and interpretable proxy for mind-like inference in sequential games, offering insights into both the strengths and limitations of current LLM reasoning. 

---
# Understanding Chain-of-Thought in Large Language Models via Topological Data Analysis 

**Authors**: Chenghao Li, Chaoning Zhang, Yi Lu, Shuxu Chen, Xudong Wang, Jiaquan Zhang, Zhicheng Wang, Zhengxun Jin, Kuien Liu, Sung-Ho Bae, Guoqing Wang, Yang Yang, Hen Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2512.19135)  

**Abstract**: With the development of large language models (LLMs), particularly with the introduction of the long reasoning chain technique, the reasoning ability of LLMs in complex problem-solving has been significantly enhanced. While acknowledging the power of long reasoning chains, we cannot help but wonder: Why do different reasoning chains perform differently in reasoning? What components of the reasoning chains play a key role? Existing studies mainly focus on evaluating reasoning chains from a functional perspective, with little attention paid to their structural mechanisms. To address this gap, this work is the first to analyze and evaluate the quality of the reasoning chain from a structural perspective. We apply persistent homology from Topological Data Analysis (TDA) to map reasoning steps into semantic space, extract topological features, and analyze structural changes. These changes reveal semantic coherence, logical redundancy, and identify logical breaks and gaps. By calculating homology groups, we assess connectivity and redundancy at various scales, using barcode and persistence diagrams to quantify stability and consistency. Our results show that the topological structural complexity of reasoning chains correlates positively with accuracy. More complex chains identify correct answers sooner, while successful reasoning exhibits simpler topologies, reducing redundancy and cycles, enhancing efficiency and interpretability. This work provides a new perspective on reasoning chain quality assessment and offers guidance for future optimization. 

---
# FC-MIR: A Mobile Screen Awareness Framework for Intent-Aware Recommendation based on Frame-Compressed Multimodal Trajectory Reasoning 

**Authors**: Zhe Yang, Xiaoshuang Sheng, Zhengnan Zhang, Jidong Wu, Zexing Wang, Xin He, Shenghua Xu, Guanjing Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2512.19107)  

**Abstract**: Identifying user intent from mobile UI operation trajectories is critical for advancing UI understanding and enabling task automation agents. While Multimodal Large Language Models (MLLMs) excel at video understanding tasks, their real-time mobile deployment is constrained by heavy computational costs and inefficient redundant frame processing. To address these issues, we propose the FC-MIR framework: leveraging keyframe sampling and adaptive concatenation, it cuts visual redundancy to boost inference efficiency, while integrating state-of-the-art closed-source MLLMs or fine-tuned models (e.g., Qwen3-VL) for trajectory summarization and intent prediction. We further expand task scope to explore generating post-prediction operations and search suggestions, and introduce a fine-grained metric to evaluate the practical utility of summaries, predictions, and suggestions. For rigorous assessment, we construct a UI trajectory dataset covering scenarios from UI-Agents (Agent-I) and real user interactions (Person-I). Experimental results show our compression method retains performance at 50%-60% compression rates; both closed-source and fine-tuned MLLMs demonstrate strong intent summarization, supporting potential lightweight on-device deployment. However, MLLMs still struggle with useful and "surprising" suggestions, leaving room for improvement. Finally, we deploy the framework in a real-world setting, integrating UI perception and UI-Agent proxies to lay a foundation for future progress in this field. 

---
# Population-Evolve: a Parallel Sampling and Evolutionary Method for LLM Math Reasoning 

**Authors**: Yanzhi Zhang, Yitong Duan, Zhaoxi Zhang, Jiyan He, Shuxin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.19081)  

**Abstract**: Test-time scaling has emerged as a promising direction for enhancing the reasoning capabilities of Large Language Models in last few years. In this work, we propose Population-Evolve, a training-free method inspired by Genetic Algorithms to optimize LLM reasoning. Our approach maintains a dynamic population of candidate solutions for each problem via parallel reasoning. By incorporating an evolve prompt, the LLM self-evolves its population in all iterations. Upon convergence, the final answer is derived via majority voting. Furthermore, we establish a unification framework that interprets existing test-time scaling strategies through the lens of genetic algorithms. Empirical results demonstrate that Population-Evolve achieves superior accuracy with low performance variance and computational efficiency. Our findings highlight the potential of evolutionary strategies to unlock the reasoning power of LLMs during inference. 

---
# Tool-Augmented Hybrid Ensemble Reasoning with Distillation for Bilingual Mathematical Problem Solving 

**Authors**: Peiqing Lu, Yuan Zhang, Haoyun Zhang, Jiasen Zheng, Kejian Tong, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2512.19093)  

**Abstract**: Bilingual mathematical problem solving needs a clear link between language reasoning and symbolic calculation. Large language models often handle language well but are weak in accurate computation. This paper presents HERALD (Hybrid Ensemble Reasoning with Adaptive Learning and Distillation), a framework that joins reasoning and calculation using NuminaMath-7B-TIR, GPT-4o, and Mistral-7B. HERALD uses adaptive routing, tool-based reinforcement learning, and knowledge distillation to connect different reasoning paths. Confidence calibration keeps weighting stable, and dual-path checking keeps results correct. Reinforcement learning controls tool use to cut redundancy, and distillation lowers delay without hurting accuracy. The system shows that combining symbolic checking, adaptive ensembles, and bilingual fine-tuning helps achieve both fluent reasoning and precise calculation. HERALD offers a practical solution for multilingual mathematical reasoning with better accuracy, stability, and clarity. 

---
# Training Multimodal Large Reasoning Models Needs Better Thoughts: A Three-Stage Framework for Long Chain-of-Thought Synthesis and Selection 

**Authors**: Yizhi Wang, Linan Yue, Min-Ling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18956)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable performance on complex reasoning tasks through long Chain-of-Thought (CoT) reasoning. Extending these successes to multimodal reasoning remains challenging due to the increased complexity of integrating diverse input modalities and the scarcity of high-quality long CoT training data. Existing multimodal datasets and CoT synthesis methods still suffer from limited reasoning depth, modality conversion errors, and rigid generation pipelines, hindering model performance and stability. To this end, in this paper, we propose SynSelect, a novel three-stage Synthesis-Selection framework for generating high-quality long CoT data tailored to multimodal reasoning tasks. Specifically, SynSelect first leverages multiple heterogeneous multimodal LRMs to produce diverse candidate CoTs, and then applies both instance and batch level selection to filter high-quality CoTs that can effectively enhance the model's reasoning capabilities. Extensive experiments on multiple multimodal benchmarks demonstrate that models supervised fine-tuned on SynSelect-generated data significantly outperform baselines and achieve further improvements after reinforcement learning post-training. Our results validate SynSelect as an effective approach for advancing multimodal LRMs reasoning capabilities. 

---
# Can abstract concepts from LLM improve SLM performance? 

**Authors**: Siddharth Tandon  

**Link**: [PDF](https://arxiv.org/pdf/2512.19069)  

**Abstract**: Large language models (LLMs) excel at diverse tasks, but their deployment on resource-constrained devices remains challenging. Existing methods like quantization, pruning, and distillation can reduce memory footprint but often demand extensive experimentation and careful infrastructure design. Leveraging existing techniques for extracting high-level concepts (represented as steering vectors) from larger models, we investigate their transferability to smaller language models (SLM) during inference. We demonstrate through extensive experimentation that these concepts can be effectively transferred to smaller models, irrespective of their family (e.g., Phi, Llama, Qwen), leading to performance improvements across a wide range of tasks. Furthermore, we introduce inference-time scaling to enhance performance by dynamically adjusting the steering intensity which has resulted in a 7-15\% of accuracy improvement for Qwen3-0.6B. 

---
# IntelliCode: A Multi-Agent LLM Tutoring System with Centralized Learner Modeling 

**Authors**: Jones David, Shreya Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2512.18669)  

**Abstract**: LLM-based tutors are typically single-turn assistants that lack persistent representations of learner knowledge, making it difficult to provide principled, transparent, and long-term pedagogical support. We introduce IntelliCode, a multi-agent LLM tutoring system built around a centralized, versioned learner state that integrates mastery estimates, misconceptions, review schedules, and engagement signals. A StateGraph Orchestrator coordinates six specialized agents: skill assessment, learner profiling, graduated hinting, curriculum selection, spaced repetition, and engagement monitoring, each operating as a pure transformation over the shared state under a single-writer policy. This architecture enables auditable mastery updates, proficiency-aware hints, dependency-aware curriculum adaptation, and safety-aligned prompting.
The demo showcases an end-to-end tutoring workflow: a learner attempts a DSA problem, receives a conceptual hint when stuck, submits a corrected solution, and immediately sees mastery updates and a personalized review interval. We report validation results with simulated learners, showing stable state updates, improved task success with graduated hints, and diverse curriculum coverage. IntelliCode demonstrates how persistent learner modeling, orchestrated multi-agent reasoning, and principled instructional design can be combined to produce transparent and reliable LLM-driven tutoring. 

---
# Vox Deorum: A Hybrid LLM Architecture for 4X / Grand Strategy Game AI -- Lessons from Civilization V 

**Authors**: John Chen, Sihan Cheng, Can Gurkan, Ryan Lay, Moez Salahuddin  

**Link**: [PDF](https://arxiv.org/pdf/2512.18564)  

**Abstract**: Large Language Models' capacity to reason in natural language makes them uniquely promising for 4X and grand strategy games, enabling more natural human-AI gameplay interactions such as collaboration and negotiation. However, these games present unique challenges due to their complexity and long-horizon nature, while latency and cost factors may hinder LLMs' real-world deployment. Working on a classic 4X strategy game, Sid Meier's Civilization V with the Vox Populi mod, we introduce Vox Deorum, a hybrid LLM+X architecture. Our layered technical design empowers LLMs to handle macro-strategic reasoning, delegating tactical execution to subsystems (e.g., algorithmic AI or reinforcement learning AI in the future). We validate our approach through 2,327 complete games, comparing two open-source LLMs with a simple prompt against Vox Populi's enhanced AI. Results show that LLMs achieve competitive end-to-end gameplay while exhibiting play styles that diverge substantially from algorithmic AI and from each other. Our work establishes a viable architecture for integrating LLMs in commercial 4X games, opening new opportunities for game design and agentic AI research. 

---
# Large Language Models as Discounted Bayesian Filters 

**Authors**: Jensen Zhang, Jing Yang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18489)  

**Abstract**: Large Language Models (LLMs) demonstrate strong few-shot generalization through in-context learning, yet their reasoning in dynamic and stochastic environments remains opaque. Prior studies mainly focus on static tasks and overlook the online adaptation required when beliefs must be continuously updated, which is a key capability for LLMs acting as world models or agents. We introduce a Bayesian filtering framework to evaluate online inference in LLMs. Our probabilistic probe suite spans both multivariate discrete distributions, such as dice rolls, and continuous distributions, such as Gaussian processes, where ground-truth parameters shift over time. We find that while LLM belief updates resemble Bayesian posteriors, they are more accurately characterized by an exponential forgetting filter with a model-specific discount factor smaller than one. This reveals systematic discounting of older evidence that varies significantly across model architectures. Although inherent priors are often miscalibrated, the updating mechanism itself remains structured and principled. We further validate these findings in a simulated agent task and propose prompting strategies that effectively recalibrate priors with minimal computational cost. 

---
# Reflective Confidence: Correcting Reasoning Flaws via Online Self-Correction 

**Authors**: Qinglin Zeng, Jing Yang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18605)  

**Abstract**: Large language models (LLMs) have achieved strong performance on complex reasoning tasks using techniques such as chain-of-thought and self-consistency. However, ensemble-based approaches, especially self-consistency which relies on multiple reasoning trajectories, often incur substantial computational overhead. To improve efficiency, prior work has leveraged internal confidence signals, where early stopping strategies such as DeepConf reduce cost by terminating low-confidence trajectories. However, this strategy discards incomplete reasoning paths and wastes partial computation.
We propose reflective confidence, a novel reasoning framework that transforms low-confidence signals from termination indicators into reflection triggers. When confidence falls below a threshold, instead of stopping generation, the model produces a reflection prompt to analyze the current reasoning state, identify potential errors, and continue generation along a corrected trajectory. Experiments on mathematical reasoning benchmarks, including AIME 2025, demonstrate significant accuracy improvements over advanced early-stopping baselines at comparable computational cost, validating the effectiveness of proactive self-correction over passive discarding. 

---
# HARBOR: Holistic Adaptive Risk assessment model for BehaviORal healthcare 

**Authors**: Aditya Siddhant  

**Link**: [PDF](https://arxiv.org/pdf/2512.18829)  

**Abstract**: Behavioral healthcare risk assessment remains a challenging problem due to the highly multimodal nature of patient data and the temporal dynamics of mood and affective disorders. While large language models (LLMs) have demonstrated strong reasoning capabilities, their effectiveness in structured clinical risk scoring remains unclear. In this work, we introduce HARBOR, a behavioral health aware language model designed to predict a discrete mood and risk score, termed the Harbor Risk Score (HRS), on an integer scale from -3 (severe depression) to +3 (mania). We also release PEARL, a longitudinal behavioral healthcare dataset spanning four years of monthly observations from three patients, containing physiological, behavioral, and self reported mental health signals. We benchmark traditional machine learning models, proprietary LLMs, and HARBOR across multiple evaluation settings and ablations. Our results show that HARBOR outperforms classical baselines and off the shelf LLMs, achieving 69 percent accuracy compared to 54 percent for logistic regression and 29 percent for the strongest proprietary LLM baseline. 

---
# CORE: Concept-Oriented Reinforcement for Bridging the Definition-Application Gap in Mathematical Reasoning 

**Authors**: Zijun Gao, Zhikun Xu, Xiao Ye, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2512.18857)  

**Abstract**: Large language models (LLMs) often solve challenging math exercises yet fail to apply the concept right when the problem requires genuine understanding. Popular Reinforcement Learning with Verifiable Rewards (RLVR) pipelines reinforce final answers but provide little fine-grained conceptual signal, so models improve at pattern reuse rather than conceptual applications. We introduce CORE (Concept-Oriented REinforcement), an RL training framework that turns explicit concepts into a controllable supervision signal. Starting from a high-quality, low-contamination textbook resource that links verifiable exercises to concise concept descriptions, we run a sanity probe showing LLMs can restate definitions but fail concept-linked quizzes, quantifying the conceptual reasoning gap. CORE then (i) synthesizes concept-aligned quizzes, (ii) injects brief concept snippets during rollouts to elicit concept-primed trajectories, and (iii) reinforces conceptual reasoning via trajectory replacement after group failures, a lightweight forward-KL constraint that aligns unguided with concept-primed policies, or standard GRPO directly on concept-aligned quizzes. Across several models, CORE delivers consistent gains over vanilla and SFT baselines on both in-domain concept-exercise suites and diverse out-of-domain math benchmarks. CORE unifies direct training on concept-aligned quizzes and concept-injected rollouts under outcome regularization. It provides fine-grained conceptual supervision that bridges problem-solving competence and genuine conceptual reasoning, while remaining algorithm- and verifier-agnostic. 

---
# MSC-180: A Benchmark for Automated Formal Theorem Proving from Mathematical Subject Classification 

**Authors**: Sirui Li, Wangyue Lu, Xiaorui Shi, Ke Weng, Haozhe Sun, Minghe Yu, Tiancheng Zhang, Ge Yu, Hengyu Liu, Lun Du  

**Link**: [PDF](https://arxiv.org/pdf/2512.18256)  

**Abstract**: Automated Theorem Proving (ATP) represents a core research direction in artificial intelligence for achieving formal reasoning and verification, playing a significant role in advancing machine intelligence. However, current large language model (LLM)-based theorem provers suffer from limitations such as restricted domain coverage and weak generalization in mathematical reasoning. To address these issues, we propose MSC-180, a benchmark for evaluation based on the MSC2020 mathematical subject classification. It comprises 180 formal verification problems, 3 advanced problems from each of 60 mathematical branches, spanning from undergraduate to graduate levels. Each problem has undergone multiple rounds of verification and refinement by domain experts to ensure formal accuracy. Evaluations of state-of-the-art LLM-based theorem provers under the pass@32 setting reveal that the best model achieves only an 18.89% overall pass rate, with prominent issues including significant domain bias (maximum domain coverage 41.7%) and a difficulty gap (significantly lower pass rates on graduate-level problems). To further quantify performance variability across mathematical domains, we introduce the coefficient of variation (CV) as an evaluation metric. The observed CV values are 4-6 times higher than the statistical high-variability threshold, indicating that the models still rely on pattern matching from training corpora rather than possessing transferable reasoning mechanisms and systematic generalization capabilities. MSC-180, together with its multi-dimensional evaluation framework, provides a discriminative and systematic benchmark for driving the development of next-generation AI systems with genuine mathematical reasoning abilities. 

---
# NL2CA: Auto-formalizing Cognitive Decision-Making from Natural Language Using an Unsupervised CriticNL2LTL Framework 

**Authors**: Zihao Deng, Yijia Li, Renrui Zhang, Peijun Ye  

**Link**: [PDF](https://arxiv.org/pdf/2512.18189)  

**Abstract**: Cognitive computing models offer a formal and interpretable way to characterize human's deliberation and decision-making, yet their development remains labor-intensive. In this paper, we propose NL2CA, a novel method for auto-formalizing cognitive decision-making rules from natural language descriptions of human experience. Different from most related work that exploits either pure manual or human guided interactive modeling, our method is fully automated without any human intervention. The approach first translates text into Linear Temporal Logic (LTL) using a fine-tuned large language model (LLM), then refines the logic via an unsupervised Critic Tree, and finally transforms the output into executable production rules compatible with symbolic cognitive frameworks. Based on the resulted rules, a cognitive agent is further constructed and optimized through cognitive reinforcement learning according to the real-world behavioral data. Our method is validated in two domains: (1) NL-to-LTL translation, where our CriticNL2LTL module achieves consistent performance across both expert and large-scale benchmarks without human-in-the-loop feed-backs, and (2) cognitive driving simulation, where agents automatically constructed from human interviews have successfully learned the diverse decision patterns of about 70 trials in different critical scenarios. Experimental results demonstrate that NL2CA enables scalable, interpretable, and human-aligned cognitive modeling from unstructured textual data, offering a novel paradigm to automatically design symbolic cognitive agents. 

---
# Intelligent Human-Machine Partnership for Manufacturing: Enhancing Warehouse Planning through Simulation-Driven Knowledge Graphs and LLM Collaboration 

**Authors**: Himabindu Thogaru, Saisubramaniam Gopalakrishnan, Zishan Ahmad, Anirudh Deodhar  

**Link**: [PDF](https://arxiv.org/pdf/2512.18265)  

**Abstract**: Manufacturing planners face complex operational challenges that require seamless collaboration between human expertise and intelligent systems to achieve optimal performance in modern production environments. Traditional approaches to analyzing simulation-based manufacturing data often create barriers between human decision-makers and critical operational insights, limiting effective partnership in manufacturing planning. Our framework establishes a collaborative intelligence system integrating Knowledge Graphs and Large Language Model-based agents to bridge this gap, empowering manufacturing professionals through natural language interfaces for complex operational analysis. The system transforms simulation data into semantically rich representations, enabling planners to interact naturally with operational insights without specialized expertise. A collaborative LLM agent works alongside human decision-makers, employing iterative reasoning that mirrors human analytical thinking while generating precise queries for knowledge extraction and providing transparent validation. This partnership approach to manufacturing bottleneck identification, validated through operational scenarios, demonstrates enhanced performance while maintaining human oversight and decision authority. For operational inquiries, the system achieves near-perfect accuracy through natural language interaction. For investigative scenarios requiring collaborative analysis, we demonstrate the framework's effectiveness in supporting human experts to uncover interconnected operational issues that enhance understanding and decision-making. This work advances collaborative manufacturing by creating intuitive methods for actionable insights, reducing cognitive load while amplifying human analytical capabilities in evolving manufacturing ecosystems. 

---
# ESearch-R1: Learning Cost-Aware MLLM Agents for Interactive Embodied Search via Reinforcement Learning 

**Authors**: Weijie Zhou, Xuangtang Xiong, Ye Tian, Lijun Yue, Xinyu Wu, Wei Li, Chaoyang Zhao, Honghui Dong, Ming Tang, Jinqiao Wang, Zhengyou Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18571)  

**Abstract**: Multimodal Large Language Models (MLLMs) have empowered embodied agents with remarkable capabilities in planning and reasoning. However, when facing ambiguous natural language instructions (e.g., "fetch the tool" in a cluttered room), current agents often fail to balance the high cost of physical exploration against the cognitive cost of human interaction. They typically treat disambiguation as a passive perception problem, lacking the strategic reasoning to minimize total task execution costs. To bridge this gap, we propose ESearch-R1, a cost-aware embodied reasoning framework that unifies interactive dialogue (Ask), episodic memory retrieval (GetMemory), and physical navigation (Navigate) into a single decision process. We introduce HC-GRPO (Heterogeneous Cost-Aware Group Relative Policy Optimization). Unlike traditional PPO which relies on a separate value critic, HC-GRPO optimizes the MLLM by sampling groups of reasoning trajectories and reinforcing those that achieve the optimal trade-off between information gain and heterogeneous costs (e.g., navigate time, and human attention). Extensive experiments in AI2-THOR demonstrate that ESearch-R1 significantly outperforms standard ReAct-based agents. It improves task success rates while reducing total operational costs by approximately 50\%, validating the effectiveness of GRPO in aligning MLLM agents with physical world constraints. 

---
# Sophia: A Persistent Agent Framework of Artificial Life 

**Authors**: Mingyang Sun, Feng Hong, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.18202)  

**Abstract**: The development of LLMs has elevated AI agents from task-specific tools to long-lived, decision-making entities. Yet, most architectures remain static and reactive, tethered to manually defined, narrow scenarios. These systems excel at perception (System 1) and deliberation (System 2) but lack a persistent meta-layer to maintain identity, verify reasoning, and align short-term actions with long-term survival. We first propose a third stratum, System 3, that presides over the agent's narrative identity and long-horizon adaptation. The framework maps selected psychological constructs to concrete computational modules, thereby translating abstract notions of artificial life into implementable design requirements. The ideas coalesce in Sophia, a "Persistent Agent" wrapper that grafts a continuous self-improvement loop onto any LLM-centric System 1/2 stack. Sophia is driven by four synergistic mechanisms: process-supervised thought search, narrative memory, user and self modeling, and a hybrid reward system. Together, they transform repetitive reasoning into a self-driven, autobiographical process, enabling identity continuity and transparent behavioral explanations. Although the paper is primarily conceptual, we provide a compact engineering prototype to anchor the discussion. Quantitatively, Sophia independently initiates and executes various intrinsic tasks while achieving an 80% reduction in reasoning steps for recurring operations. Notably, meta-cognitive persistence yielded a 40% gain in success for high-complexity tasks, effectively bridging the performance gap between simple and sophisticated goals. Qualitatively, System 3 exhibited a coherent narrative identity and an innate capacity for task organization. By fusing psychological insight with a lightweight reinforcement-learning core, the persistent agent architecture advances a possible practical pathway toward artificial life. 

---
# Rethinking Multi-Agent Intelligence Through the Lens of Small-World Networks 

**Authors**: Boxuan Wang, Zhuoyun Li, Xiaowei Huang, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2512.18094)  

**Abstract**: Large language models (LLMs) have enabled multi-agent systems (MAS) in which multiple agents argue, critique, and coordinate to solve complex tasks, making communication topology a first-class design choice. Yet most existing LLM-based MAS either adopt fully connected graphs, simple sparse rings, or ad-hoc dynamic selection, with little structural guidance. In this work, we revisit classic theory on small-world (SW) networks and ask: what changes if we treat SW connectivity as a design prior for MAS? We first bridge insights from neuroscience and complex networks to MAS, highlighting how SW structures balance local clustering and long-range integration. Using multi-agent debate (MAD) as a controlled testbed, experiment results show that SW connectivity yields nearly the same accuracy and token cost, while substantially stabilizing consensus trajectories. Building on this, we introduce an uncertainty-guided rewiring scheme for scaling MAS, where long-range shortcuts are added between epistemically divergent agents using LLM-oriented uncertainty signals (e.g., semantic entropy). This yields controllable SW structures that adapt to task difficulty and agent heterogeneity. Finally, we discuss broader implications of SW priors for MAS design, framing them as stabilizers of reasoning, enhancers of robustness, scalable coordinators, and inductive biases for emergent cognitive roles. 

---
# MapTrace: Scalable Data Generation for Route Tracing on Maps 

**Authors**: Artemis Panagopoulou, Aveek Purohit, Achin Kulshrestha, Soroosh Yazdani, Mohit Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2512.19609)  

**Abstract**: While Multimodal Large Language Models have achieved human-like performance on many visual and textual reasoning tasks, their proficiency in fine-grained spatial understanding, such as route tracing on maps remains limited. Unlike humans, who can quickly learn to parse and navigate maps, current models often fail to respect fundamental path constraints, in part due to the prohibitive cost and difficulty of collecting large-scale, pixel-accurate path annotations. To address this, we introduce a scalable synthetic data generation pipeline that leverages synthetic map images and pixel-level parsing to automatically produce precise annotations for this challenging task. Using this pipeline, we construct a fine-tuning dataset of 23k path samples across 4k maps, enabling models to acquire more human-like spatial capabilities. Using this dataset, we fine-tune both open-source and proprietary MLLMs. Results on MapBench show that finetuning substantially improves robustness, raising success rates by up to 6.4 points, while also reducing path-tracing error (NDTW). These gains highlight that fine-grained spatial reasoning, absent in pretrained models, can be explicitly taught with synthetic supervision. 

---
# Anatomy-R1: Enhancing Anatomy Reasoning in Multimodal Large Language Models via Anatomical Similarity Curriculum and Group Diversity Augmentation 

**Authors**: Ziyang Song, Zelin Zang, Zuyao Chen, Xusheng Liang, Dong Yi, Jinlin Wu, Hongbin Liu, Jiebo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2512.19512)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved impressive progress in natural image reasoning, yet their potential in medical imaging remains underexplored, especially in clinical anatomical surgical images. Anatomy understanding tasks demand precise understanding and clinically coherent answers, which are difficult to achieve due to the complexity of medical data and the scarcity of high-quality expert annotations. These challenges limit the effectiveness of conventional Supervised Fine-Tuning (SFT) strategies. While recent work has demonstrated that Group Relative Policy Optimization (GRPO) can enhance reasoning in MLLMs without relying on large amounts of data, we find two weaknesses that hinder GRPO's reasoning performance in anatomy recognition: 1) knowledge cannot be effectively shared between different anatomical structures, resulting in uneven information gain and preventing the model from converging, and 2) the model quickly converges to a single reasoning path, suppressing the exploration of diverse strategies. To overcome these challenges, we propose two novel methods. First, we implement a progressive learning strategy called Anatomical Similarity Curriculum Learning by controlling question difficulty via the similarity of answer choices, enabling the model to master complex problems incrementally. Second, we utilize question augmentation referred to as Group Diversity Question Augmentation to expand the model's search space for difficult queries, mitigating the tendency to produce uniform responses. Comprehensive experiments on the SGG-VQA and OmniMedVQA benchmarks show our method achieves a significant improvement across the two benchmarks, demonstrating its effectiveness in enhancing the medical reasoning capabilities of MLLMs. The code can be found in this https URL 

---
# NEURO-GUARD: Neuro-Symbolic Generalization and Unbiased Adaptive Routing for Diagnostics -- Explainable Medical AI 

**Authors**: Midhat Urooj, Ayan Banerjee, Sandeep Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2512.18177)  

**Abstract**: Accurate yet interpretable image-based diagnosis remains a central challenge in medical AI, particularly in settings characterized by limited data, subtle visual cues, and high-stakes clinical decision-making. Most existing vision models rely on purely data-driven learning and produce black-box predictions with limited interpretability and poor cross-domain generalization, hindering their real-world clinical adoption. We present NEURO-GUARD, a novel knowledge-guided vision framework that integrates Vision Transformers (ViTs) with language-driven reasoning to improve performance, transparency, and domain robustness. NEURO-GUARD employs a retrieval-augmented generation (RAG) mechanism for self-verification, in which a large language model (LLM) iteratively generates, evaluates, and refines feature-extraction code for medical images. By grounding this process in clinical guidelines and expert knowledge, the framework progressively enhances feature detection and classification beyond purely data-driven baselines. Extensive experiments on diabetic retinopathy classification across four benchmark datasets APTOS, EyePACS, Messidor-1, and Messidor-2 demonstrate that NEURO-GUARD improves accuracy by 6.2% over a ViT-only baseline (84.69% vs. 78.4%) and achieves a 5% gain in domain generalization. Additional evaluations on MRI-based seizure detection further confirm its cross-domain robustness, consistently outperforming existing methods.
Overall, NEURO-GUARD bridges symbolic medical reasoning with subsymbolic visual learning, enabling interpretable, knowledge-aware, and generalizable medical image diagnosis while achieving state-of-the-art performance across multiple datasets. 

---
# OmniMER: Indonesian Multimodal Emotion Recognition via Auxiliary-Enhanced LLM Adaptation 

**Authors**: Xueming Yan, Boyan Xu, Yaochu Jin, Lixian Xiao, Wenlong Ye, Runyang Cai, Zeqi Zheng, Jingfa Liu, Aimin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.19379)  

**Abstract**: Indonesian, spoken by over 200 million people, remains underserved in multimodal emotion recognition research despite its dominant presence on Southeast Asian social media platforms. We introduce IndoMER, the first multimodal emotion recognition benchmark for Indonesian, comprising 1,944 video segments from 203 speakers with temporally aligned text, audio, and visual annotations across seven emotion categories. The dataset exhibits realistic challenges including cross-modal inconsistency and long-tailed class distributions shaped by Indonesian cultural communication norms. To address these challenges, we propose OmniMER, a multimodal adaptation framework built upon Qwen2.5-Omni that enhances emotion recognition through three auxiliary modality-specific perception tasks: emotion keyword extraction for text, facial expression analysis for video, and prosody analysis for audio. These auxiliary tasks help the model identify emotion-relevant cues in each modality before fusion, reducing reliance on spurious correlations in low-resource settings. Experiments on IndoMER show that OmniMER achieves 0.582 Macro-F1 on sentiment classification and 0.454 on emotion recognition, outperforming the base model by 7.6 and 22.1 absolute points respectively. Cross-lingual evaluation on the Chinese CH-SIMS dataset further demonstrates the generalizability of the proposed framework. The dataset and code are publicly available. this https URL 

---
# Identifying Features Associated with Bias Against 93 Stigmatized Groups in Language Models and Guardrail Model Safety Mitigation 

**Authors**: Anna-Maria Gueorguieva, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2512.19238)  

**Abstract**: Large language models (LLMs) have been shown to exhibit social bias, however, bias towards non-protected stigmatized identities remain understudied. Furthermore, what social features of stigmas are associated with bias in LLM outputs is unknown. From psychology literature, it has been shown that stigmas contain six shared social features: aesthetics, concealability, course, disruptiveness, origin, and peril. In this study, we investigate if human and LLM ratings of the features of stigmas, along with prompt style and type of stigma, have effect on bias towards stigmatized groups in LLM outputs. We measure bias against 93 stigmatized groups across three widely used LLMs (Granite 3.0-8B, Llama-3.1-8B, Mistral-7B) using SocialStigmaQA, a benchmark that includes 37 social scenarios about stigmatized identities; for example deciding wether to recommend them for an internship. We find that stigmas rated by humans to be highly perilous (e.g., being a gang member or having HIV) have the most biased outputs from SocialStigmaQA prompts (60% of outputs from all models) while sociodemographic stigmas (e.g. Asian-American or old age) have the least amount of biased outputs (11%). We test if the amount of biased outputs could be decreased by using guardrail models, models meant to identify harmful input, using each LLM's respective guardrail model (Granite Guardian 3.0, Llama Guard 3.0, Mistral Moderation API). We find that bias decreases significantly by 10.4%, 1.4%, and 7.8%, respectively. However, we show that features with significant effect on bias remain unchanged post-mitigation and that guardrail models often fail to recognize the intent of bias in prompts. This work has implications for using LLMs in scenarios involving stigmatized groups and we suggest future work towards improving guardrail models for bias mitigation. 

---
# A Dataset and Preliminary Study of Using GPT-5 for Code-change Impact Analysis 

**Authors**: Katharina Stengg, Christian Macho, Martin Pinzger  

**Link**: [PDF](https://arxiv.org/pdf/2512.19481)  

**Abstract**: Understanding source code changes and their impact on other code entities is a crucial skill in software development. However, the analysis of code changes and their impact is often performed manually and therefore is time-consuming. Recent advancements in AI, and in particular large language models (LLMs) show promises to help developers in various code analysis tasks. However, the extent to which this potential can be utilized for understanding code changes and their impact is underexplored. To address this gap, we study the capabilities of GPT-5 and GPT-5-mini to predict the code entities impacted by given source code changes. We construct a dataset containing information about seed-changes, change pairs, and change types for each commit. Existing datasets lack crucial information about seed changes and impacted code entities. Our experiments evaluate the LLMs in two configurations: (1) seed-change information and the parent commit tree and (2) seed-change information, the parent commit tree, and the diff hunk of each seed change. We found that both LLMs perform poorly in the two experiments, whereas GPT-5 outperforms GPT-5-mini. Furthermore, the provision of the diff hunks helps both models to slightly improve their performance. 

---
# MixKVQ: Query-Aware Mixed-Precision KV Cache Quantization for Long-Context Reasoning 

**Authors**: Tao Zhang, Ziqian Zeng, Hao Peng, Huiping Zhuang, Cen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2512.19206)  

**Abstract**: Long Chain-of-Thought (CoT) reasoning has significantly advanced the capabilities of Large Language Models (LLMs), but this progress is accompanied by substantial memory and latency overhead from the extensive Key-Value (KV) cache. Although KV cache quantization is a promising compression technique, existing low-bit quantization methods often exhibit severe performance degradation on complex reasoning tasks. Fixed-precision quantization struggles to handle outlier channels in the key cache, while current mixed-precision strategies fail to accurately identify components requiring high-precision representation. We find that an effective low-bit KV cache quantization strategy must consider two factors: a key channel's intrinsic quantization difficulty and its relevance to the query. Based on this insight, we propose MixKVQ, a novel plug-and-play method that introduces a lightweight, query-aware algorithm to identify and preserve critical key channels that need higher precision, while applying per-token quantization for value cache. Experiments on complex reasoning datasets demonstrate that our approach significantly outperforms existing low-bit methods, achieving performance comparable to a full-precision baseline at a substantially reduced memory footprint. 

---
# The Erasure Illusion: Stress-Testing the Generalization of LLM Forgetting Evaluation 

**Authors**: Hengrui Jia, Taoran Li, Jonas Guan, Varun Chandrasekaran  

**Link**: [PDF](https://arxiv.org/pdf/2512.19025)  

**Abstract**: Machine unlearning aims to remove specific data influences from trained models, a capability essential for adhering to copyright laws and ensuring AI safety. Current unlearning metrics typically measure success by monitoring the model's performance degradation on the specific unlearning dataset ($D_u$). We argue that for Large Language Models (LLMs), this evaluation paradigm is insufficient and potentially misleading. Many real-world uses of unlearning--motivated by copyright or safety--implicitly target not only verbatim content in $D_u$, but also behaviors influenced by the broader generalizations the model derived from it. We demonstrate that LLMs can pass standard unlearning evaluation and appear to have ``forgotten'' the target knowledge, while simultaneously retaining strong capabilities on content that is semantically adjacent to $D_u$. This phenomenon indicates that erasing exact sentences does not necessarily equate to removing the underlying knowledge. To address this gap, we propose \name, an automated stress-testing framework that generates a surrogate dataset, $\tilde{D}_u$. This surrogate set is constructed to be semantically derived from $D_u$ yet sufficiently distinct in embedding space. By comparing unlearning metric scores between $D_u$ and $\tilde{D}_u$, we can stress-test the reliability of the metric itself. Our extensive evaluation across three LLM families (Llama-3-8B, Qwen2.5-7B, and Zephyr-7B-$\beta$), three distinct datasets, and seven standard metrics reveals widespread inconsistencies. We find that current metrics frequently overestimate unlearning success, failing to detect retained knowledge exposed by our stress-test datasets. 

---
# When Less is More: 8-bit Quantization Improves Continual Learning in Large Language Models 

**Authors**: Michael S. Zhang, Rishi A. Ruia, Arnav Kewalram, Saathvik Dharmapuram, Utkarsh Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.18934)  

**Abstract**: Catastrophic forgetting poses a fundamental challenge in continual learning, particularly when models are quantized for deployment efficiency. We systematically investigate the interplay between quantization precision (FP16, INT8, INT4) and replay buffer strategies in large language models, revealing unexpected dynamics. While FP16 achieves superior initial task performance (74.44% on NLU), we observe a striking inversion on subsequent tasks: quantized models outperform FP16 by 8-15% on final task forward accuracy, with INT4 achieving nearly double FP16's performance on Code generation (40% vs 20%). Critically, even minimal replay buffers (0.1%) dramatically improve retention - increasing NLU retention after Math training from 45% to 65% across all precision levels - with INT8 consistently achieving the optimal balance between learning plasticity and knowledge retention. We hypothesize that quantization-induced noise acts as implicit regularization, preventing the overfitting to new task gradients that plagues high-precision models. These findings challenge the conventional wisdom that higher precision is always preferable, suggesting instead that INT8 quantization offers both computational efficiency and superior continual learning dynamics. Our results provide practical guidelines for deploying compressed models in continual learning scenarios: small replay buffers (1-2%) suffice for NLU tasks, while Math and Code benefit from moderate buffers (5-10%), with quantized models requiring less replay than FP16 to achieve comparable retention. Code is available at this https URL. 

---
# CrashChat: A Multimodal Large Language Model for Multitask Traffic Crash Video Analysis 

**Authors**: Kaidi Liang, Ke Li, Xianbiao Hu, Ruwen Qin  

**Link**: [PDF](https://arxiv.org/pdf/2512.18878)  

**Abstract**: Automating crash video analysis is essential to leverage the growing availability of driving video data for traffic safety research and accountability attribution in autonomous driving. Crash video analysis is a challenging multitask problem due to the complex spatiotemporal dynamics of crash events in video data and the diverse analytical requirements involved. It requires capabilities spanning crash recognition, temporal grounding, and high-level video understanding. Existing models, however, cannot perform all these tasks within a unified framework, and effective training strategies for such models remain underexplored. To fill these gaps, this paper proposes CrashChat, a multimodal large language model (MLLM) for multitask traffic crash analysis, built upon VideoLLaMA3. CrashChat acquires domain-specific knowledge through instruction fine-tuning and employs a novel multitask learning strategy based on task decoupling and grouping, which maximizes the benefit of joint learning within and across task groups while mitigating negative transfer. Numerical experiments on consolidated public datasets demonstrate that CrashChat consistently outperforms existing MLLMs across model scales and traditional vision-based methods, achieving state-of-the-art performance. It reaches near-perfect accuracy in crash recognition, a 176\% improvement in crash localization, and a 40\% improvement in the more challenging pre-crash localization. Compared to general MLLMs, it substantially enhances textual accuracy and content coverage in crash description and reasoning tasks, with 0.18-0.41 increases in BLEU scores and 0.18-0.42 increases in ROUGE scores. Beyond its strong performance, CrashChat is a convenient, end-to-end analytical tool ready for practical implementation. The dataset and implementation code for CrashChat are available at this https URL. 

---
# Explainable and Fine-Grained Safeguarding of LLM Multi-Agent Systems via Bi-Level Graph Anomaly Detection 

**Authors**: Junjun Pan, Yixin Liu, Rui Miao, Kaize Ding, Yu Zheng, Quoc Viet Hung Nguyen, Alan Wee-Chung Liew, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2512.18733)  

**Abstract**: Large language model (LLM)-based multi-agent systems (MAS) have shown strong capabilities in solving complex tasks. As MAS become increasingly autonomous in various safety-critical tasks, detecting malicious agents has become a critical security concern. Although existing graph anomaly detection (GAD)-based defenses can identify anomalous agents, they mainly rely on coarse sentence-level information and overlook fine-grained lexical cues, leading to suboptimal performance. Moreover, the lack of interpretability in these methods limits their reliability and real-world applicability. To address these limitations, we propose XG-Guard, an explainable and fine-grained safeguarding framework for detecting malicious agents in MAS. To incorporate both coarse and fine-grained textual information for anomalous agent identification, we utilize a bi-level agent encoder to jointly model the sentence- and token-level representations of each agent. A theme-based anomaly detector further captures the evolving discussion focus in MAS dialogues, while a bi-level score fusion mechanism quantifies token-level contributions for explanation. Extensive experiments across diverse MAS topologies and attack scenarios demonstrate robust detection performance and strong interpretability of XG-Guard. 

---
# A Multi-agent Text2SQL Framework using Small Language Models and Execution Feedback 

**Authors**: Thanh Dat Hoang, Thanh Trung Huynh, Matthias Weidlich, Thanh Tam Nguyen, Tong Chen, Hongzhi Yin, Quoc Viet Hung Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2512.18622)  

**Abstract**: Text2SQL, the task of generating SQL queries from natural language text, is a critical challenge in data engineering. Recently, Large Language Models (LLMs) have demonstrated superior performance for this task due to their advanced comprehension and generation capabilities. However, privacy and cost considerations prevent companies from using Text2SQL solutions based on external LLMs offered as a service. Rather, small LLMs (SLMs) that are openly available and can hosted in-house are adopted. These SLMs, in turn, lack the generalization capabilities of larger LLMs, which impairs their effectiveness for complex tasks such as Text2SQL. To address these limitations, we propose MATS, a novel Text2SQL framework designed specifically for SLMs. MATS uses a multi-agent mechanism that assigns specialized roles to auxiliary agents, reducing individual workloads and fostering interaction. A training scheme based on reinforcement learning aligns these agents using feedback obtained during execution, thereby maintaining competitive performance despite a limited LLM size. Evaluation results using on benchmark datasets show that MATS, deployed on a single- GPU server, yields accuracy that are on-par with large-scale LLMs when using significantly fewer parameters. Our source code and data are available at this https URL. 

---
# VeruSAGE: A Study of Agent-Based Verification for Rust Systems 

**Authors**: Chenyuan Yang, Natalie Neamtu, Chris Hawblitzel, Jacob R. Lorch, Shan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2512.18436)  

**Abstract**: Large language models (LLMs) have shown impressive capability to understand and develop code. However, their capability to rigorously reason about and prove code correctness remains in question. This paper offers a comprehensive study of LLMs' capability to develop correctness proofs for system software written in Rust. We curate a new system-verification benchmark suite, VeruSAGE-Bench, which consists of 849 proof tasks extracted from eight open-source Verus-verified Rust systems. Furthermore, we design different agent systems to match the strengths and weaknesses of different LLMs (o4-mini, GPT-5, Sonnet 4, and Sonnet 4.5). Our study shows that different tools and agent settings are needed to stimulate the system-verification capability of different types of LLMs. The best LLM-agent combination in our study completes over 80% of system-verification tasks in VeruSAGE-Bench. It also completes over 90% of a set of system proof tasks not part of VeruSAGE-Bench because they had not yet been finished by human experts. This result shows the great potential for LLM-assisted development of verified system software. 

---
# Secret mixtures of experts inside your LLM 

**Authors**: Enric Boix-Adsera  

**Link**: [PDF](https://arxiv.org/pdf/2512.18452)  

**Abstract**: Despite being one of the earliest neural network layers, the Multilayer Perceptron (MLP) is arguably one of the least understood parts of the transformer architecture due to its dense computation and lack of easy visualization. This paper seeks to understand the MLP layers in dense LLM models by hypothesizing that these layers secretly approximately perform a sparse computation -- namely, that they can be well approximated by sparsely-activating Mixture of Experts (MoE) layers.
Our hypothesis is based on a novel theoretical connection between MoE models and Sparse Autoencoder (SAE) structure in activation space. We empirically validate the hypothesis on pretrained LLMs, and demonstrate that the activation distribution matters -- these results do not hold for Gaussian data, but rather rely crucially on structure in the distribution of neural network activations.
Our results shine light on a general principle at play in MLP layers inside LLMs, and give an explanation for the effectiveness of modern MoE-based transformers. Additionally, our experimental explorations suggest new directions for more efficient MoE architecture design based on low-rank routers. 

---
# LLaViDA: A Large Language Vision Driving Assistant for Explicit Reasoning and Enhanced Trajectory Planning 

**Authors**: Yudong Liu, Spencer Hallyburton, Jiwoo Kim, Yueqian Lin, Yiming Li, Qinsi Wang, Hui Ye, Jingwei Sun, Miroslav Pajic, Yiran Chen, Hai Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.18211)  

**Abstract**: Trajectory planning is a fundamental yet challenging component of autonomous driving. End-to-end planners frequently falter under adverse weather, unpredictable human behavior, or complex road layouts, primarily because they lack strong generalization or few-shot capabilities beyond their training data. We propose LLaViDA, a Large Language Vision Driving Assistant that leverages a Vision-Language Model (VLM) for object motion prediction, semantic grounding, and chain-of-thought reasoning for trajectory planning in autonomous driving. A two-stage training pipeline--supervised fine-tuning followed by Trajectory Preference Optimization (TPO)--enhances scene understanding and trajectory planning by injecting regression-based supervision, produces a powerful "VLM Trajectory Planner for Autonomous Driving." On the NuScenes benchmark, LLaViDA surpasses state-of-the-art end-to-end and other recent VLM/LLM-based baselines in open-loop trajectory planning task, achieving an average L2 trajectory error of 0.31 m and a collision rate of 0.10% on the NuScenes test set. The code for this paper is available at GitHub. 

---
# Holistic Evaluation of State-of-the-Art LLMs for Code Generation 

**Authors**: Le Zhang, Suresh Kothari  

**Link**: [PDF](https://arxiv.org/pdf/2512.18131)  

**Abstract**: This study presents a comprehensive empirical evaluation of six state-of-the-art large language models (LLMs) for code generation, including both general-purpose and code-specialized models. Using a dataset of 944 real-world LeetCode problems across five programming languages, we assess model performance using rigorous metrics: compile-time errors, runtime errors, functional failures, and algorithmic suboptimalities. The results reveal significant performance variations, with DeepSeek-R1 and GPT-4.1 consistently outperform others in terms of correctness, efficiency, and robustness. Through detailed case studies, we identify common failure scenarios such as syntax errors, logical flaws, and suboptimal algorithms, highlighting the critical role of prompt engineering and human oversight in improving results. Based on these findings, we provide actionable recommendations for developers and practitioners, emphasizing that successful LLM deployment depends on careful model selection, effective prompt design, and context-aware usage to ensure reliable code generation in real-world software development tasks. 

---
# Breaking Minds, Breaking Systems: Jailbreaking Large Language Models via Human-like Psychological Manipulation 

**Authors**: Zehao Liu, Xi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2512.18244)  

**Abstract**: Large Language Models (LLMs) have gained considerable popularity and protected by increasingly sophisticated safety mechanisms. However, jailbreak attacks continue to pose a critical security threat by inducing models to generate policy-violating behaviors. Current paradigms focus on input-level anomalies, overlooking that the model's internal psychometric state can be systematically manipulated. To address this, we introduce Psychological Jailbreak, a new jailbreak attack paradigm that exposes a stateful psychological attack surface in LLMs, where attackers exploit the manipulation of a model's psychological state across interactions. Building on this insight, we propose Human-like Psychological Manipulation (HPM), a black-box jailbreak method that dynamically profiles a target model's latent psychological vulnerabilities and synthesizes tailored multi-turn attack strategies. By leveraging the model's optimization for anthropomorphic consistency, HPM creates a psychological pressure where social compliance overrides safety constraints. To systematically measure psychological safety, we construct an evaluation framework incorporating psychometric datasets and the Policy Corruption Score (PCS). Benchmarking against various models (e.g., GPT-4o, DeepSeek-V3, Gemini-2-Flash), HPM achieves a mean Attack Success Rate (ASR) of 88.1%, outperforming state-of-the-art attack baselines. Our experiments demonstrate robust penetration against advanced defenses, including adversarial prompt optimization (e.g., RPO) and cognitive interventions (e.g., Self-Reminder). Ultimately, PCS analysis confirms HPM induces safety breakdown to satisfy manipulated contexts. Our work advocates for a fundamental paradigm shift from static content filtering to psychological safety, prioritizing the development of psychological defense mechanisms against deep cognitive manipulation. 

---
# Specification and Detection of LLM Code Smells 

**Authors**: Brahim Mahmoudi, Zacharie Chenail-Larcher, Naouel Moha, Quentin Stievenert, Florent Avellaneda  

**Link**: [PDF](https://arxiv.org/pdf/2512.18020)  

**Abstract**: Large Language Models (LLMs) have gained massive popularity in recent years and are increasingly integrated into software systems for diverse purposes. However, poorly integrating them in source code may undermine software system quality. Yet, to our knowledge, there is no formal catalog of code smells specific to coding practices for LLM inference. In this paper, we introduce the concept of LLM code smells and formalize five recurrent problematic coding practices related to LLM inference in software systems, based on relevant literature. We extend the detection tool SpecDetect4AI to cover the newly defined LLM code smells and use it to validate their prevalence in a dataset of 200 open-source LLM systems. Our results show that LLM code smells affect 60.50% of the analyzed systems, with a detection precision of 86.06%. 

---
# Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition 

**Authors**: Ellie Zhou, Jihoon Chung, Olga Russakovsky  

**Link**: [PDF](https://arxiv.org/pdf/2512.17953)  

**Abstract**: Human action recognition models often rely on background cues rather than human movement and pose to make predictions, a behavior known as background bias. We present a systematic analysis of background bias across classification models, contrastive text-image pretrained models, and Video Large Language Models (VLLM) and find that all exhibit a strong tendency to default to background reasoning. Next, we propose mitigation strategies for classification models and show that incorporating segmented human input effectively decreases background bias by 3.78%. Finally, we explore manual and automated prompt tuning for VLLMs, demonstrating that prompt design can steer predictions towards human-focused reasoning by 9.85%. 

---
# Inferring Latent Market Forces: Evaluating LLM Detection of Gamma Exposure Patterns via Obfuscation Testing 

**Authors**: Christopher Regan, Ying Xie  

**Link**: [PDF](https://arxiv.org/pdf/2512.17923)  

**Abstract**: We introduce obfuscation testing, a novel methodology for validating whether large language models detect structural market patterns through causal reasoning rather than temporal association. Testing three dealer hedging constraint patterns (gamma positioning, stock pinning, 0DTE hedging) on 242 trading days (95.6% coverage) of S&P 500 options data, we find LLMs achieve 71.5% detection rate using unbiased prompts that provide only raw gamma exposure values without regime labels or temporal context. The WHO-WHOM-WHAT causal framework forces models to identify the economic actors (dealers), affected parties (directional traders), and structural mechanisms (forced hedging) underlying observed market dynamics. Critically, detection accuracy (91.2%) remains stable even as economic profitability varies quarterly, demonstrating that models identify structural constraints rather than profitable patterns. When prompted with regime labels, detection increases to 100%, but the 71.5% unbiased rate validates genuine pattern recognition. Our findings suggest LLMs possess emergent capabilities for detecting complex financial mechanisms through pure structural reasoning, with implications for systematic strategy development, risk management, and our understanding of how transformer architectures process financial market dynamics. 

---
# Efficient Multi-Adapter LLM Serving via Cross-Model KV-Cache Reuse with Activated LoRA 

**Authors**: Allison Li, Kristjan Greenewald, Thomas Parnell, Navid Azizan  

**Link**: [PDF](https://arxiv.org/pdf/2512.17910)  

**Abstract**: Modern large language model (LLM) systems increasingly rely on multi-turn pipelines that are composed of multiple task-specific adapters, yet existing serving frameworks remain inefficient, incurring substantial recomputation overhead when switching between adapters. We present the first LLM serving engine that supports cross-model prefix cache reuse between base and adapted models via Activated LoRA (aLoRA), enabling efficient and fine-grained adapter switching during inference. Our design extends the vLLM framework by introducing base-aligned block hashing and activation-aware masking within the model execution path, permitting cache reuse across models while preserving compatibility with existing serving engine optimizations. Integrated into a production-grade inference stack, this approach supports dynamic adapter activation without excessive key-value tensor recomputation. Evaluation across representative multi-turn, multi-adapter pipelines demonstrates up to 58x end-to-end latency reduction and over 100x time-to-first-token improvement relative to standard LoRA baselines, with benefits that scale with model size and sequence length and manifest across all stages of the request lifecycle. This work bridges parameter-efficient model adaptation with high-performance serving, providing the first complete realization of cross-model KV-cache reuse in modern LLM inference engines. 

---
