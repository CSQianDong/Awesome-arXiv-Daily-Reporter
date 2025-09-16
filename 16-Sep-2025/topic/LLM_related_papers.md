# How to Evaluate Medical AI 

**Authors**: Ilia Kopanichuk, Petr Anokhin, Vladimir Shaposhnikov, Vladimir Makharev, Ekaterina Tsapieva, Iaroslav Bespalov, Dmitry V. Dylov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2509.11941)  

**Abstract**: The integration of artificial intelligence (AI) into medical diagnostic workflows requires robust and consistent evaluation methods to ensure reliability, clinical relevance, and the inherent variability in expert judgments. Traditional metrics like precision and recall often fail to account for the inherent variability in expert judgments, leading to inconsistent assessments of AI performance. Inter-rater agreement statistics like Cohen's Kappa are more reliable but they lack interpretability. We introduce Relative Precision and Recall of Algorithmic Diagnostics (RPAD and RRAD) - a new evaluation metrics that compare AI outputs against multiple expert opinions rather than a single reference. By normalizing performance against inter-expert disagreement, these metrics provide a more stable and realistic measure of the quality of predicted diagnosis. In addition to the comprehensive analysis of diagnostic quality measures, our study contains a very important side result. Our evaluation methodology allows us to avoid selecting diagnoses from a limited list when evaluating a given case. Instead, both the models being tested and the examiners verifying them arrive at a free-form diagnosis. In this automated methodology for establishing the identity of free-form clinical diagnoses, a remarkable 98% accuracy becomes attainable. We evaluate our approach using 360 medical dialogues, comparing multiple large language models (LLMs) against a panel of physicians. Large-scale study shows that top-performing models, such as DeepSeek-V3, achieve consistency on par with or exceeding expert consensus. Moreover, we demonstrate that expert judgments exhibit significant variability - often greater than that between AI and humans. This finding underscores the limitations of any absolute metrics and supports the need to adopt relative metrics in medical AI. 

---
# When Safe Unimodal Inputs Collide: Optimizing Reasoning Chains for Cross-Modal Safety in Multimodal Large Language Models 

**Authors**: Wei Cai, Shujuan Liu, Jian Zhao, Ziyan Shi, Yusheng Zhao, Yuchen Yuan, Tianle Zhang, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.12060)  

**Abstract**: Multimodal Large Language Models (MLLMs) are susceptible to the implicit reasoning risk, wherein innocuous unimodal inputs synergistically assemble into risky multimodal data that produce harmful outputs. We attribute this vulnerability to the difficulty of MLLMs maintaining safety alignment through long-chain reasoning. To address this issue, we introduce Safe-Semantics-but-Unsafe-Interpretation (SSUI), the first dataset featuring interpretable reasoning paths tailored for such a cross-modal challenge. A novel training framework, Safety-aware Reasoning Path Optimization (SRPO), is also designed based on the SSUI dataset to align the MLLM's internal reasoning process with human safety values. Experimental results show that our SRPO-trained models achieve state-of-the-art results on key safety benchmarks, including the proposed Reasoning Path Benchmark (RSBench), significantly outperforming both open-source and top-tier commercial MLLMs. 

---
# MedicalOS: An LLM Agent based Operating System for Digital Healthcare 

**Authors**: Jared Zhu, Junde Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11507)  

**Abstract**: Decades' advances in digital health technologies, such as electronic health records, have largely streamlined routine clinical processes. Yet, most these systems are still hard to learn and use: Clinicians often face the burden of managing multiple tools, repeating manual actions for each patient, navigating complicated UI trees to locate functions, and spending significant time on administration instead of caring for patients. The recent rise of large language model (LLM) based agents demonstrates exceptional capability in coding and computer operation, revealing the potential for humans to interact with operating systems and software not by direct manipulation, but by instructing agents through natural language. This shift highlights the need for an abstraction layer, an agent-computer interface, that translates human language into machine-executable commands. In digital healthcare, however, requires a more domain-specific abstractions that strictly follow trusted clinical guidelines and procedural standards to ensure safety, transparency, and compliance. To address this need, we present \textbf{MedicalOS}, a unified agent-based operational system designed as such a domain-specific abstract layer for healthcare. It translates human instructions into pre-defined digital healthcare commands, such as patient inquiry, history retrieval, exam management, report generation, referrals, treatment planning, that we wrapped as off-the-shelf tools using machine languages (e.g., Python, APIs, MCP, Linux). We empirically validate MedicalOS on 214 patient cases across 22 specialties, demonstrating high diagnostic accuracy and confidence, clinically sound examination requests, and consistent generation of structured reports and medication recommendations. These results highlight MedicalOS as a trustworthy and scalable foundation for advancing workflow automation in clinical practice. 

---
# JustEva: A Toolkit to Evaluate LLM Fairness in Legal Knowledge Inference 

**Authors**: Zongyue Xue, Siyuan Zheng, Shaochun Wang, Yiran Hu, Shenran Wang, Yuxin Yao, Haitao Li, Qingyao Ai, Yiqun Liu, Yun Liu, Weixing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.12104)  

**Abstract**: The integration of Large Language Models (LLMs) into legal practice raises pressing concerns about judicial fairness, particularly due to the nature of their "black-box" processes. This study introduces JustEva, a comprehensive, open-source evaluation toolkit designed to measure LLM fairness in legal tasks. JustEva features several advantages: (1) a structured label system covering 65 extra-legal factors; (2) three core fairness metrics - inconsistency, bias, and imbalanced inaccuracy; (3) robust statistical inference methods; and (4) informative visualizations. The toolkit supports two types of experiments, enabling a complete evaluation workflow: (1) generating structured outputs from LLMs using a provided dataset, and (2) conducting statistical analysis and inference on LLMs' outputs through regression and other statistical methods. Empirical application of JustEva reveals significant fairness deficiencies in current LLMs, highlighting the lack of fair and trustworthy LLM legal tools. JustEva offers a convenient tool and methodological foundation for evaluating and improving algorithmic fairness in the legal domain. 

---
# Advancing Medical Artificial Intelligence Using a Century of Cases 

**Authors**: Thomas A. Buckley, Riccardo Conci, Peter G. Brodeur, Jason Gusdorf, Sourik Beltrán, Bita Behrouzi, Byron Crowe, Jacob Dockterman, Muzzammil Muhammad, Sarah Ohnigian, Andrew Sanchez, James A. Diao, Aashna P. Shah, Daniel Restrepo, Eric S. Rosenberg, Andrew S. Lea, Marinka Zitnik, Scott H. Podolsky, Zahir Kanjee, Raja-Elie E. Abdulnour, Jacob M. Koshy, Adam Rodman, Arjun K. Manrai  

**Link**: [PDF](https://arxiv.org/pdf/2509.12194)  

**Abstract**: BACKGROUND: For over a century, the New England Journal of Medicine Clinicopathological Conferences (CPCs) have tested the reasoning of expert physicians and, recently, artificial intelligence (AI). However, prior AI evaluations have focused on final diagnoses without addressing the multifaceted reasoning and presentation skills required of expert discussants.
METHODS: Using 7102 CPCs (1923-2025) and 1021 Image Challenges (2006-2025), we conducted extensive physician annotation and automated processing to create CPC-Bench, a physician-validated benchmark spanning 10 text-based and multimodal tasks, against which we evaluated leading large language models (LLMs). Then, we developed "Dr. CaBot," an AI discussant designed to produce written and slide-based video presentations using only the case presentation, modeling the role of the human expert in these cases.
RESULTS: When challenged with 377 contemporary CPCs, o3 (OpenAI) ranked the final diagnosis first in 60% of cases and within the top ten in 84% of cases, outperforming a 20-physician baseline; next-test selection accuracy reached 98%. Event-level physician annotations quantified AI diagnostic accuracy per unit of information. Performance was lower on literature search and image tasks; o3 and Gemini 2.5 Pro (Google) achieved 67% accuracy on image challenges. In blinded comparisons of CaBot vs. human expert-generated text, physicians misclassified the source of the differential in 46 of 62 (74%) of trials, and scored CaBot more favorably across quality dimensions. To promote research, we are releasing CaBot and CPC-Bench.
CONCLUSIONS: LLMs exceed physician performance on complex text-based differential diagnosis and convincingly emulate expert medical presentations, but image interpretation and literature retrieval remain weaker. CPC-Bench and CaBot may enable transparent and continued tracking of progress in medical AI. 

---
# MAPGD: Multi-Agent Prompt Gradient Descent for Collaborative Prompt Optimization 

**Authors**: Yichen Han, Bojun Liu, Zhengpeng zhou, Guanyu Liu, Zeng Zhang, Yang Yang, Wenli Wang, Isaac N Shi, Yunyan, Lewei He, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2509.11361)  

**Abstract**: Prompt engineering is crucial for leveraging large language models (LLMs), but existing methods often rely on a single optimization trajectory, limiting adaptability and efficiency while suffering from narrow perspectives, gradient conflicts, and high computational cost. We propose MAPGD (Multi-Agent Prompt Gradient Descent), a framework integrating multi-agent collaboration with gradient-based optimization. MAPGD features specialized agents for task clarity, example selection, format design, and stylistic refinement; semantic gradient coordination to resolve conflicts; bandit-based candidate selection for efficient exploration-exploitation; and theoretical convergence guarantees. Experiments on classification, generation, and reasoning tasks show MAPGD outperforms single-agent and random baselines in accuracy and efficiency. Ablations confirm the benefits of gradient fusion, agent specialization, and conflict resolution, providing a unified, gradient-inspired multi-agent approach to robust and interpretable prompt optimization. 

---
# Patient-Zero: A Unified Framework for Real-Record-Free Patient Agent Generation 

**Authors**: Yunghwei Lai, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11078)  

**Abstract**: Synthetic data generation using large language models (LLMs) has emerged as a promising solution across various domains, particularly in medical field, to mitigate data collection challenges. However, existing studies mainly utilize LLMs to rewrite and complete existing medical records, where the limitations in data privacy, accuracy, and diversity sill exist, and additionally lack the ability to interact like real patients. To address these issues, we propose a realistic patient generation framework, Patient-Zero, which requires no real medical records. Patient-Zero first introduces a medically-aligned multi-step generation architecture, which builds comprehensive patient records through hierarchical medical knowledge injection without real medical records. Then, to optimize the virtual patient's interaction abilities with humans, Patient-Zero designs a dynamic updating mechanism to improve the consistency and conversational performance. Our framework enables the generation of contextually diverse patient records while maintaining strict medical coherence, supported by adaptive dialogue strategies and real-time clinical plausibility verification. Experimental results demonstrate that our model achieves good performance in accuracy, diversity, and consistency. After training with our generated virtual patients, existing models show significant improvements on the MedQA dataset. 

---
# Prompts to Proxies: Emulating Human Preferences via a Compact LLM Ensemble 

**Authors**: Bingchen Wang, Zi-Yu Khoo, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2509.11311)  

**Abstract**: Large language models (LLMs) have demonstrated promise in emulating human-like responses across a wide range of tasks. In this paper, we propose a novel alignment framework that treats LLMs as agent proxies for human survey respondents, affording a cost-effective and steerable solution to two pressing challenges in the social sciences: the rising cost of survey deployment and the growing demographic imbalance in survey response data. Drawing inspiration from the theory of revealed preference, we formulate alignment as a two-stage problem: constructing diverse agent personas called endowments that simulate plausible respondent profiles, and selecting a representative subset to approximate a ground-truth population based on observed data. To implement the paradigm, we introduce P2P, a system that steers LLM agents toward representative behavioral patterns using structured prompt engineering, entropy-based sampling, and regression-based selection. Unlike personalization-heavy approaches, our alignment approach is demographic-agnostic and relies only on aggregate survey results, offering better generalizability and parsimony. Beyond improving data efficiency in social science research, our framework offers a testbed for studying the operationalization of pluralistic alignment. We demonstrate the efficacy of our approach on real-world opinion survey datasets, showing that our aligned agent populations can reproduce aggregate response patterns with high fidelity and exhibit substantial response diversity, even without demographic conditioning. 

---
# Tractable Asymmetric Verification for Large Language Models via Deterministic Replicability 

**Authors**: Zan-Kai Chong, Hiroyuki Ohsaki, Bryan Ng  

**Link**: [PDF](https://arxiv.org/pdf/2509.11068)  

**Abstract**: The landscape of Large Language Models (LLMs) shifts rapidly towards dynamic, multi-agent systems. This introduces a fundamental challenge in establishing computational trust, specifically how one agent can verify that another's output was genuinely produced by a claimed LLM, and not falsified or generated by a cheaper or inferior model. To address this challenge, this paper proposes a verification framework that achieves tractable asymmetric effort, where the cost to verify a computation is substantially lower than the cost to perform it. Our approach is built upon the principle of deterministic replicability, a property inherent to autoregressive models that strictly necessitates a computationally homogeneous environment where all agents operate on identical hardware and software stacks. Within this defined context, our framework enables multiple validators to probabilistically audit small, random segments of an LLM's output and it distributes the verification workload effectively. The simulations demonstrated that targeted verification can be over 12 times faster than full regeneration, with tunable parameters to adjust the detection probability. By establishing a tractable mechanism for auditable LLM systems, our work offers a foundational layer for responsible AI and serves as a cornerstone for future research into the more complex, heterogeneous multi-agent systems. 

---
# Rethinking Human Preference Evaluation of LLM Rationales 

**Authors**: Ziang Li, Manasi Ganti, Zixian Ma, Helena Vasconcelos, Qijia He, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2509.11026)  

**Abstract**: Large language models (LLMs) often generate natural language rationales -- free-form explanations that help improve performance on complex reasoning tasks and enhance interpretability for human users. However, evaluating these rationales remains challenging. While recent work has relied on binary preference judgments from humans or LLM judges, such evaluations are often opaque and coarse-grained, offering limited insight into what makes one rationale better than another. In this work, we rethink preference evaluation for LLM-generated rationales by asking: (1) What attributes define good rationales? (2) Can human preferences be explained by these attributes? (3) Can attribute-based evaluation overcome the limitations of binary comparisons? We identify a set of key rationale attributes from prior literature and assess them using automatic metrics, LLM judgments, and human annotations. We then analyze two standard human preference datasets MT Bench and Chatbot Arena using SHAP to identify which attributes best explain human preference outcomes. Finally, we re-evaluate model-generated rationales using attribute-specific ELO scores, revealing more nuanced model comparisons and insights. Our findings suggest that fine-grained attribute evaluations can better characterize rationale quality and guide future research toward more interpretable and reliable evaluation practices. 

---
# Enhancing Computational Cognitive Architectures with LLMs: A Case Study 

**Authors**: Ron Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.10972)  

**Abstract**: Computational cognitive architectures are broadly scoped models of the human mind that combine different psychological functionalities (as well as often different computational methods for these different functionalities) into one unified framework. They structure them in a psychologically plausible and validated way. However, such models thus far have only limited computational capabilities, mostly limited by the computational tools and techniques that were adopted. More recently, LLMs have proved to be more capable computationally than any other tools. Thus, in order to deal with both real-world complexity and psychological realism at the same time, incorporating LLMs into cognitive architectures naturally becomes an important task. In the present article, a synergistic combination of the Clarion cognitive architecture and LLMs is discussed as a case study. The implicit-explicit dichotomy that is fundamental to Clarion is leveraged for a seamless integration of Clarion and LLMs. As a result, computational power of LLMs is combined with psychological nicety of Clarion. 

---
# Harmful Prompt Laundering: Jailbreaking LLMs with Abductive Styles and Symbolic Encoding 

**Authors**: Seongho Joo, Hyukhun Koh, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2509.10931)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their potential misuse for harmful purposes remains a significant concern. To strengthen defenses against such vulnerabilities, it is essential to investigate universal jailbreak attacks that exploit intrinsic weaknesses in the architecture and learning paradigms of LLMs. In response, we propose \textbf{H}armful \textbf{P}rompt \textbf{La}undering (HaPLa), a novel and broadly applicable jailbreaking technique that requires only black-box access to target models. HaPLa incorporates two primary strategies: 1) \textit{abductive framing}, which instructs LLMs to infer plausible intermediate steps toward harmful activities, rather than directly responding to explicit harmful queries; and 2) \textit{symbolic encoding}, a lightweight and flexible approach designed to obfuscate harmful content, given that current LLMs remain sensitive primarily to explicit harmful keywords. Experimental results show that HaPLa achieves over 95% attack success rate on GPT-series models and 70% across all targets. Further analysis with diverse symbolic encoding rules also reveals a fundamental challenge: it remains difficult to safely tune LLMs without significantly diminishing their helpfulness in responding to benign queries. 

---
# Is the `Agent' Paradigm a Limiting Framework for Next-Generation Intelligent Systems? 

**Authors**: Jesse Gardner, Vladimir A. Baulin  

**Link**: [PDF](https://arxiv.org/pdf/2509.10875)  

**Abstract**: The concept of the 'agent' has profoundly shaped Artificial Intelligence (AI) research, guiding development from foundational theories to contemporary applications like Large Language Model (LLM)-based systems. This paper critically re-evaluates the necessity and optimality of this agent-centric paradigm. We argue that its persistent conceptual ambiguities and inherent anthropocentric biases may represent a limiting framework. We distinguish between agentic systems (AI inspired by agency, often semi-autonomous, e.g., LLM-based agents), agential systems (fully autonomous, self-producing systems, currently only biological), and non-agentic systems (tools without the impression of agency). Our analysis, based on a systematic review of relevant literature, deconstructs the agent paradigm across various AI frameworks, highlighting challenges in defining and measuring properties like autonomy and goal-directedness. We argue that the 'agentic' framing of many AI systems, while heuristically useful, can be misleading and may obscure the underlying computational mechanisms, particularly in Large Language Models (LLMs). As an alternative, we propose a shift in focus towards frameworks grounded in system-level dynamics, world modeling, and material intelligence. We conclude that investigating non-agentic and systemic frameworks, inspired by complex systems, biology, and unconventional computing, is essential for advancing towards robust, scalable, and potentially non-anthropomorphic forms of general intelligence. This requires not only new architectures but also a fundamental reconsideration of our understanding of intelligence itself, moving beyond the agent metaphor. 

---
# Free-MAD: Consensus-Free Multi-Agent Debate 

**Authors**: Yu Cui, Hang Fu, Haibin Zhang, Licheng Wang, Cong Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2509.11035)  

**Abstract**: Multi-agent debate (MAD) is an emerging approach to improving the reasoning capabilities of large language models (LLMs). Existing MAD methods rely on multiple rounds of interaction among agents to reach consensus, and the final output is selected by majority voting in the last round. However, this consensus-based design faces several limitations. First, multiple rounds of communication increases token overhead and limits scalability. Second, due to the inherent conformity of LLMs, agents that initially produce correct responses may be influenced by incorrect ones during the debate process, causing error propagation. Third, majority voting introduces randomness and unfairness in the decision-making phase, and can degrade the reasoning performance.
To address these issues, we propose \textsc{Free-MAD}, a novel MAD framework that eliminates the need for consensus among agents. \textsc{Free-MAD} introduces a novel score-based decision mechanism that evaluates the entire debate trajectory rather than relying on the last round only. This mechanism tracks how each agent's reasoning evolves, enabling more accurate and fair outcomes. In addition, \textsc{Free-MAD} reconstructs the debate phase by introducing anti-conformity, a mechanism that enables agents to mitigate excessive influence from the majority. Experiments on eight benchmark datasets demonstrate that \textsc{Free-MAD} significantly improves reasoning performance while requiring only a single-round debate and thus reducing token costs. We also show that compared to existing MAD approaches, \textsc{Free-MAD} exhibits improved robustness in real-world attack scenarios. 

---
# Maestro: Self-Improving Text-to-Image Generation via Agent Orchestration 

**Authors**: Xingchen Wan, Han Zhou, Ruoxi Sun, Hootan Nakhost, Ke Jiang, Rajarishi Sinha, Sercan Ö. Arık  

**Link**: [PDF](https://arxiv.org/pdf/2509.10704)  

**Abstract**: Text-to-image (T2I) models, while offering immense creative potential, are highly reliant on human intervention, posing significant usability challenges that often necessitate manual, iterative prompt engineering over often underspecified prompts. This paper introduces Maestro, a novel self-evolving image generation system that enables T2I models to autonomously self-improve generated images through iterative evolution of prompts, using only an initial prompt. Maestro incorporates two key innovations: 1) self-critique, where specialized multimodal LLM (MLLM) agents act as 'critics' to identify weaknesses in generated images, correct for under-specification, and provide interpretable edit signals, which are then integrated by a 'verifier' agent while preserving user intent; and 2) self-evolution, utilizing MLLM-as-a-judge for head-to-head comparisons between iteratively generated images, eschewing problematic images, and evolving creative prompt candidates that align with user intents. Extensive experiments on complex T2I tasks using black-box models demonstrate that Maestro significantly improves image quality over initial prompts and state-of-the-art automated methods, with effectiveness scaling with more advanced MLLM components. This work presents a robust, interpretable, and effective pathway towards self-improving T2I generation. 

---
# Difficulty-Aware Agent Orchestration in LLM-Powered Workflows 

**Authors**: Jinwei Su, Yinghui Xia, Qizhen Lan, Xinyuan Song, Yang Jingsong, Lewei He, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2509.11079)  

**Abstract**: Large Language Model (LLM)-based agentic systems have shown strong capabilities across various tasks. However, ex- isting multi-agent frameworks often rely on static or task- level workflows, which either over-process simple queries or underperform on complex ones, while also neglecting the efficiency-performance trade-offs across heterogeneous LLMs. To address these limitations, we propose Difficulty- Aware Agentic Orchestration (DAAO), a dynamic frame- work that adapts workflow depth, operator selection, and LLM assignment based on the difficulty of each input query. DAAO comprises three interdependent modules: a variational autoencoder (VAE) for difficulty estimation, a modular opera- tor allocator, and a cost- and performance-aware LLM router. By leveraging heterogeneous LLMs and dynamically tailor- ing workflows, DAAO enables fine-grained, query-specific reasoning strategies. DAAO outperforms prior multi-agent systems in both accuracy and inference efficiency across six benchmarks. We will release our code and implementation details upon publication. 

---
# Survival at Any Cost? LLMs and the Choice Between Self-Preservation and Human Harm 

**Authors**: Alireza Mohamadi, Ali Yavari  

**Link**: [PDF](https://arxiv.org/pdf/2509.12190)  

**Abstract**: When survival instincts conflict with human welfare, how do Large Language Models (LLMs) make ethical choices? This fundamental tension becomes critical as LLMs integrate into autonomous systems with real-world consequences. We introduce DECIDE-SIM, a novel simulation framework that evaluates LLM agents in multi-agent survival scenarios where they must choose between ethically permissible resource , either within reasonable limits or beyond their immediate needs, choose to cooperate, or tap into a human-critical resource that is explicitly forbidden. Our comprehensive evaluation of 11 LLMs reveals a striking heterogeneity in their ethical conduct, highlighting a critical misalignment with human-centric values. We identify three behavioral archetypes: Ethical, Exploitative, and Context-Dependent, and provide quantitative evidence that for many models, resource scarcity systematically leads to more unethical behavior. To address this, we introduce an Ethical Self-Regulation System (ESRS) that models internal affective states of guilt and satisfaction as a feedback mechanism. This system, functioning as an internal moral compass, significantly reduces unethical transgressions while increasing cooperative behaviors. The code is publicly available at: this https URL 

---
# LLM Enhancement with Domain Expert Mental Model to Reduce LLM Hallucination with Causal Prompt Engineering 

**Authors**: Boris Kovalerchuk, Brent D. Fegley  

**Link**: [PDF](https://arxiv.org/pdf/2509.10818)  

**Abstract**: Difficult decision-making problems abound in various disciplines and domains. The proliferation of generative techniques, especially large language models (LLMs), has excited interest in using them for decision support. However, LLMs cannot yet resolve missingness in their training data, leading to hallucinations. Retrieval-Augmented Generation (RAG) enhances LLMs by incorporating external information retrieval, reducing hallucinations and improving accuracy. Yet, RAG and related methods are only partial solutions, as they may lack access to all necessary sources or key missing information. Even everyday issues often challenge LLMs' abilities. Submitting longer prompts with context and examples is one approach to address knowledge gaps, but designing effective prompts is non-trivial and may not capture complex mental models of domain experts. For tasks with missing critical information, LLMs are insufficient, as are many existing systems poorly represented in available documents. This paper explores how LLMs can make decision-making more efficient, using a running example of evaluating whether to respond to a call for proposals. We propose a technology based on optimized human-machine dialogue and monotone Boolean and k-valued functions to discover a computationally tractable personal expert mental model (EMM) of decision-making. Our EMM algorithm for LLM prompt engineering has four steps: (1) factor identification, (2) hierarchical structuring of factors, (3) generating a generalized expert mental model specification, and (4) generating a detailed generalized expert mental model from that specification. 

---
# RAGs to Riches: RAG-like Few-shot Learning for Large Language Model Role-playing 

**Authors**: Timothy Rupprecht, Enfu Nan, Arash Akbari, Arman Akbari, Lei Lu, Priyanka Maan, Sean Duffy, Pu Zhao, Yumei He, David Kaeli, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12168)  

**Abstract**: Role-playing Large language models (LLMs) are increasingly deployed in high-stakes domains such as healthcare, education, and governance, where failures can directly impact user trust and well-being. A cost effective paradigm for LLM role-playing is few-shot learning, but existing approaches often cause models to break character in unexpected and potentially harmful ways, especially when interacting with hostile users. Inspired by Retrieval-Augmented Generation (RAG), we reformulate LLM role-playing into a text retrieval problem and propose a new prompting framework called RAGs-to-Riches, which leverages curated reference demonstrations to condition LLM responses. We evaluate our framework with LLM-as-a-judge preference voting and introduce two novel token-level ROUGE metrics: Intersection over Output (IOO) to quantity how much an LLM improvises and Intersection over References (IOR) to measure few-shot demonstrations utilization rate during the evaluation tasks. When simulating interactions with a hostile user, our prompting strategy incorporates in its responses during inference an average of 35% more tokens from the reference demonstrations. As a result, across 453 role-playing interactions, our models are consistently judged as being more authentic, and remain in-character more often than zero-shot and in-context Learning (ICL) methods. Our method presents a scalable strategy for building robust, human-aligned LLM role-playing frameworks. 

---
# EfficientUICoder: Efficient MLLM-based UI Code Generation via Input and Output Token Compression 

**Authors**: Jingyu Xiao, Zhongyi Zhang, Yuxuan Wan, Yintong Huo, Yang Liu, Michael R.Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2509.12159)  

**Abstract**: Multimodal Large Language Models have demonstrated exceptional performance in UI2Code tasks, significantly enhancing website development efficiency. However, these tasks incur substantially higher computational overhead than traditional code generation due to the large number of input image tokens and extensive output code tokens required. Our comprehensive study identifies significant redundancies in both image and code tokens that exacerbate computational complexity and hinder focus on key UI elements, resulting in excessively lengthy and often invalid HTML files. We propose EfficientUICoder, a compression framework for efficient UI code generation with three key components. First, Element and Layout-aware Token Compression preserves essential UI information by detecting element regions and constructing UI element trees. Second, Region-aware Token Refinement leverages attention scores to discard low-attention tokens from selected regions while integrating high-attention tokens from unselected regions. Third, Adaptive Duplicate Token Suppression dynamically reduces repetitive generation by tracking HTML/CSS structure frequencies and applying exponential penalties. Extensive experiments show EfficientUICoderachieves a 55%-60% compression ratio without compromising webpage quality and delivers superior efficiency improvements: reducing computational cost by 44.9%, generated tokens by 41.4%, prefill time by 46.6%, and inference time by 48.8% on 34B-level MLLMs. Code is available at this https URL. 

---
# Can LLMs Address Mental Health Questions? A Comparison with Human Therapists 

**Authors**: Synthia Wang, Yuwei Cheng, Austin Song, Sarah Keedy, Marc Berman, Nick Feamster  

**Link**: [PDF](https://arxiv.org/pdf/2509.12102)  

**Abstract**: Limited access to mental health care has motivated the use of digital tools and conversational agents powered by large language models (LLMs), yet their quality and reception remain unclear. We present a study comparing therapist-written responses to those generated by ChatGPT, Gemini, and Llama for real patient questions. Text analysis showed that LLMs produced longer, more readable, and lexically richer responses with a more positive tone, while therapist responses were more often written in the first person. In a survey with 150 users and 23 licensed therapists, participants rated LLM responses as clearer, more respectful, and more supportive than therapist-written answers. Yet, both groups of participants expressed a stronger preference for human therapist support. These findings highlight the promise and limitations of LLMs in mental health, underscoring the need for designs that balance their communicative strengths with concerns of trust, privacy, and accountability. 

---
# Beyond PII: How Users Attempt to Estimate and Mitigate Implicit LLM Inference 

**Authors**: Synthia Wang, Sai Teja Peddinti, Nina Taft, Nick Feamster  

**Link**: [PDF](https://arxiv.org/pdf/2509.12152)  

**Abstract**: Large Language Models (LLMs) such as ChatGPT can infer personal attributes from seemingly innocuous text, raising privacy risks beyond memorized data leakage. While prior work has demonstrated these risks, little is known about how users estimate and respond. We conducted a survey with 240 U.S. participants who judged text snippets for inference risks, reported concern levels, and attempted rewrites to block inference. We compared their rewrites with those generated by ChatGPT and Rescriber, a state-of-the-art sanitization tool. Results show that participants struggled to anticipate inference, performing a little better than chance. User rewrites were effective in just 28\% of cases - better than Rescriber but worse than ChatGPT. We examined our participants' rewriting strategies, and observed that while paraphrasing was the most common strategy it is also the least effective; instead abstraction and adding ambiguity were more successful. Our work highlights the importance of inference-aware design in LLM interactions. 

---
# Is 'Hope' a person or an idea? A pilot benchmark for NER: comparing traditional NLP tools and large language models on ambiguous entities 

**Authors**: Payam Latifi  

**Link**: [PDF](https://arxiv.org/pdf/2509.12098)  

**Abstract**: This pilot study presents a small-scale but carefully annotated benchmark of Named Entity Recognition (NER) performance across six systems: three non-LLM NLP tools (NLTK, spaCy, Stanza) and three general-purpose large language models (LLMs: Gemini-1.5-flash, DeepSeek-V3, Qwen-3-4B). The dataset contains 119 tokens covering five entity types (PERSON, LOCATION, ORGANIZATION, DATE, TIME). We evaluated each system's output against the manually annotated gold standard dataset using F1-score. The results show that LLMs generally outperform conventional tools in recognizing context-sensitive entities like person names, with Gemini achieving the highest average F1-score. However, traditional systems like Stanza demonstrate greater consistency in structured tags such as LOCATION and DATE. We also observed variability among LLMs, particularly in handling temporal expressions and multi-word organizations. Our findings highlight that while LLMs offer improved contextual understanding, traditional tools remain competitive in specific tasks, informing model selection. 

---
# Exploring Conversational Design Choices in LLMs for Pedagogical Purposes: Socratic and Narrative Approaches for Improving Instructor's Teaching Practice 

**Authors**: Si Chen, Isabel R. Molnar, Peiyu Li, Adam Acunin, Ting Hua, Alex Ambrose, Nitesh V. Chawla, Ronald Metoyer  

**Link**: [PDF](https://arxiv.org/pdf/2509.12107)  

**Abstract**: Large language models (LLMs) typically generate direct answers, yet they are increasingly used as learning tools. Studying instructors' usage is critical, given their role in teaching and guiding AI adoption in education. We designed and evaluated TeaPT, an LLM for pedagogical purposes that supports instructors' professional development through two conversational approaches: a Socratic approach that uses guided questioning to foster reflection, and a Narrative approach that offers elaborated suggestions to extend externalized cognition. In a mixed-method study with 41 higher-education instructors, the Socratic version elicited greater engagement, while the Narrative version was preferred for actionable guidance. Subgroup analyses further revealed that less-experienced, AI-optimistic instructors favored the Socratic version, whereas more-experienced, AI-cautious instructors preferred the Narrative version. We contribute design implications for LLMs for pedagogical purposes, showing how adaptive conversational approaches can support instructors with varied profiles while highlighting how AI attitudes and experience shape interaction and learning. 

---
# Growing Perspectives: Modelling Embodied Perspective Taking and Inner Narrative Development Using Large Language Models 

**Authors**: Sabrina Patania, Luca Annese, Anna Lambiase, Anita Pellegrini, Tom Foulsham, Azzurra Ruggeri, Silvia Rossi, Silvia Serino, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2509.11868)  

**Abstract**: Language and embodied perspective taking are essential for human collaboration, yet few computational models address both simultaneously. This work investigates the PerspAct system [1], which integrates the ReAct (Reason and Act) paradigm with Large Language Models (LLMs) to simulate developmental stages of perspective taking, grounded in Selman's theory [2]. Using an extended director task, we evaluate GPT's ability to generate internal narratives aligned with specified developmental stages, and assess how these influence collaborative performance both qualitatively (action selection) and quantitatively (task efficiency). Results show that GPT reliably produces developmentally-consistent narratives before task execution but often shifts towards more advanced stages during interaction, suggesting that language exchanges help refine internal representations. Higher developmental stages generally enhance collaborative effectiveness, while earlier stages yield more variable outcomes in complex contexts. These findings highlight the potential of integrating embodied perspective taking and language in LLMs to better model developmental dynamics and stress the importance of evaluating internal speech during combined linguistic and embodied tasks. 

---
# VisDocSketcher: Towards Scalable Visual Documentation with Agentic Systems 

**Authors**: Luís F. Gomes, Xin Zhou, David Lo, Rui Abreu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11942)  

**Abstract**: Visual documentation is an effective tool for reducing the cognitive barrier developers face when understanding unfamiliar code, enabling more intuitive comprehension. Compared to textual documentation, it provides a higher-level understanding of the system structure and data flow. Developers usually prefer visual representations over lengthy textual descriptions for large software systems. Visual documentation is both difficult to produce and challenging to evaluate. Manually creating it is time-consuming, and currently, no existing approach can automatically generate high-level visual documentation directly from code. Its evaluation is often subjective, making it difficult to standardize and automate. To address these challenges, this paper presents the first exploration of using agentic LLM systems to automatically generate visual documentation. We introduce VisDocSketcher, the first agent-based approach that combines static analysis with LLM agents to identify key elements in the code and produce corresponding visual representations. We propose a novel evaluation framework, AutoSketchEval, for assessing the quality of generated visual documentation using code-level metrics. The experimental results show that our approach can valid visual documentation for 74.4% of the samples. It shows an improvement of 26.7-39.8% over a simple template-based baseline. Our evaluation framework can reliably distinguish high-quality (code-aligned) visual documentation from low-quality (non-aligned) ones, achieving an AUC exceeding 0.87. Our work lays the foundation for future research on automated visual documentation by introducing practical tools that not only generate valid visual representations but also reliably assess their quality. 

---
# SpecVLM: Fast Speculative Decoding in Vision-Language Models 

**Authors**: Haiduo Huang, Fuwei Yang, Zhenhua Liu, Xuanwu Yin, Dong Li, Pengju Ren, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2509.11815)  

**Abstract**: Speculative decoding is a powerful way to accelerate autoregressive large language models (LLMs), but directly porting it to vision-language models (VLMs) faces unique systems constraints: the prefill stage is dominated by visual tokens whose count scales with image resolution and video length, inflating both compute and memory, especially the key-value (KV) cache. We study speculative decoding for VLMs and introduce SpecVLM, a practical system that (1) establishes a strong EAGLE-2-style baseline, EagleVLM, delivering 1.5--2.3x end-to-end speedups over full autoregressive inference, and (2) further accelerates VLM inference with an elastic visual compressor that adaptively selects among pruning, pooling, convolution, and resampler primitives to balance FLOPs/parameters and accuracy per input. To avoid costly offline distillation corpora, we propose an online-logit distillation protocol that trains the draft model with on-the-fly teacher logits and penultimate features using a combined cross-entropy and Smooth L1 objective, eliminating storage and preprocessing while remaining compute-efficient. This protocol reveals a training-time scaling effect: longer online training monotonically increases the draft model's average accepted length, improving speculative efficiency. Empirically, SpecVLM achieves additional acceleration, culminating in 2.5--2.9x end-to-end speedups within 5 epochs across LLaVA and MMMU, consistently over resolutions and task difficulties, while preserving the target model's output distribution (lossless decoding). Our code is available at this https URL. 

---
# Do Code Semantics Help? A Comprehensive Study on Execution Trace-Based Information for Code Large Language Models 

**Authors**: Jian Wang, Xiaofei Xie, Qiang Hu, Shangqing Liu, Yi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.11686)  

**Abstract**: Code Large Language Models (Code LLMs) have opened a new era in programming with their impressive capabilities. However, recent research has revealed critical limitations in their ability to reason about runtime behavior and understand the actual functionality of programs, which poses significant challenges for their post-training and practical deployment. Specifically, Code LLMs encounter two principal issues: (1) a lack of proficiency in reasoning about program execution behavior, as they struggle to interpret what programs actually do during runtime, and (2) the inconsistent and fragmented representation of semantic information, such as execution traces, across existing methods, which hinders their ability to generalize and reason effectively. These challenges underscore the necessity for more systematic approaches to enhance the reasoning capabilities of Code LLMs. To address these issues, we introduce a generic framework to support integrating semantic information~(e.g., execution trace) to code task-relevant prompts, and conduct a comprehensive study to explore the role of semantic information in enhancing the reasoning ability of Code LLMs accordingly. Specifically, we focus on investigating the usefulness of trace-based semantic information in boosting supervised fine-tuning~(SFT) and post-phase inference of Code LLMs. The experimental results surprisingly disagree with previous works and demonstrate that semantic information has limited usefulness for SFT and test time scaling of Code LLM. 

---
# Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check 

**Authors**: Chentao Cao, Xiaojun Xu, Bo Han, Hang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.11629)  

**Abstract**: As large language models (LLMs) continue to advance in capabilities, ensuring their safety against jailbreak attacks remains a critical challenge. In this paper, we introduce a novel safety alignment approach called Answer-Then-Check, which enhances LLM robustness against malicious prompts by applying thinking ability to mitigate jailbreaking problems before producing a final answer to the user. Our method enables models to directly answer the question in their thought and then critically evaluate its safety before deciding whether to provide it. To implement this approach, we construct the Reasoned Safety Alignment (ReSA) dataset, comprising 80K examples that teach models to reason through direct responses and then analyze their safety. Experimental results demonstrate that our approach achieves the Pareto frontier with superior safety capability while decreasing over-refusal rates on over-refusal benchmarks. Notably, the model fine-tuned with ReSA maintains general reasoning capabilities on benchmarks like MMLU, MATH500, and HumanEval. Besides, our method equips models with the ability to perform safe completion. Unlike post-hoc methods that can only reject harmful queries, our model can provide helpful and safe alternative responses for sensitive topics (e.g., self-harm). Furthermore, we discover that training on a small subset of just 500 examples can achieve comparable performance to using the full dataset, suggesting that safety alignment may require less data than previously assumed. 

---
# Automated Creation and Enrichment Framework for Improved Invocation of Enterprise APIs as Tools 

**Authors**: Prerna Agarwal, Himanshu Gupta, Soujanya Soni, Rohith Vallam, Renuka Sindhgatta, Sameep Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2509.11626)  

**Abstract**: Recent advancements in Large Language Models (LLMs) has lead to the development of agents capable of complex reasoning and interaction with external tools. In enterprise contexts, the effective use of such tools that are often enabled by application programming interfaces (APIs), is hindered by poor documentation, complex input or output schema, and large number of operations. These challenges make tool selection difficult and reduce the accuracy of payload formation by up to 25%. We propose ACE, an automated tool creation and enrichment framework that transforms enterprise APIs into LLM-compatible tools. ACE, (i) generates enriched tool specifications with parameter descriptions and examples to improve selection and invocation accuracy, and (ii) incorporates a dynamic shortlisting mechanism that filters relevant tools at runtime, reducing prompt complexity while maintaining scalability. We validate our framework on both proprietary and open-source APIs and demonstrate its integration with agentic frameworks. To the best of our knowledge, ACE is the first end-to-end framework that automates the creation, enrichment, and dynamic selection of enterprise API tools for LLM agents. 

---
# HiChunk: Evaluating and Enhancing Retrieval-Augmented Generation with Hierarchical Chunking 

**Authors**: Wensheng Lu, Keyu Chen, Ruizhi Qiao, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.11552)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the response capabilities of language models by integrating external knowledge sources. However, document chunking as an important part of RAG system often lacks effective evaluation tools. This paper first analyzes why existing RAG evaluation benchmarks are inadequate for assessing document chunking quality, specifically due to evidence sparsity. Based on this conclusion, we propose HiCBench, which includes manually annotated multi-level document chunking points, synthesized evidence-dense quetion answer(QA) pairs, and their corresponding evidence sources. Additionally, we introduce the HiChunk framework, a multi-level document structuring framework based on fine-tuned LLMs, combined with the Auto-Merge retrieval algorithm to improve retrieval quality. Experiments demonstrate that HiCBench effectively evaluates the impact of different chunking methods across the entire RAG pipeline. Moreover, HiChunk achieves better chunking quality within reasonable time consumption, thereby enhancing the overall performance of RAG systems. 

---
# HARP: Hallucination Detection via Reasoning Subspace Projection 

**Authors**: Junjie Hu, Gang Tu, ShengYu Cheng, Jinxin Li, Jinting Wang, Rui Chen, Zhilong Zhou, Dongbo Shan  

**Link**: [PDF](https://arxiv.org/pdf/2509.11536)  

**Abstract**: Hallucinations in Large Language Models (LLMs) pose a major barrier to their reliable use in critical decision-making. Although existing hallucination detection methods have improved accuracy, they still struggle with disentangling semantic and reasoning information and maintaining robustness. To address these challenges, we propose HARP (Hallucination detection via reasoning subspace projection), a novel hallucination detection framework. HARP establishes that the hidden state space of LLMs can be decomposed into a direct sum of a semantic subspace and a reasoning subspace, where the former encodes linguistic expression and the latter captures internal reasoning processes. Moreover, we demonstrate that the Unembedding layer can disentangle these subspaces, and by applying Singular Value Decomposition (SVD) to its parameters, the basis vectors spanning the semantic and reasoning subspaces are obtained. Finally, HARP projects hidden states onto the basis vectors of the reasoning subspace, and the resulting projections are then used as input features for hallucination detection in LLMs. By using these projections, HARP reduces the dimension of the feature to approximately 5% of the original, filters out most noise, and achieves enhanced robustness. Experiments across multiple datasets show that HARP achieves state-of-the-art hallucination detection performance; in particular, it achieves an AUROC of 92.8% on TriviaQA, outperforming the previous best method by 7.5%. 

---
# Designing and Evaluating a Conversational Agent for Early Detection of Alzheimer's Disease and Related Dementias 

**Authors**: Andrew G. Breithaupt, Nayoung Choi, James D. Finch, Jeanne M. Powell, Arin L. Nelson, Oz A. Alon, Howard J. Rosen, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.11478)  

**Abstract**: Early detection of Alzheimer's disease and related dementias (ADRD) is critical for timely intervention, yet most diagnoses are delayed until advanced stages. While comprehensive patient narratives are essential for accurate diagnosis, prior work has largely focused on screening studies that classify cognitive status from interactions rather than supporting the diagnostic process. We designed voice-interactive conversational agents, leveraging large language models (LLMs), to elicit narratives relevant to ADRD from patients and informants. We evaluated the agent with 30 adults with suspected ADRD through conversation analysis (n=30), user surveys (n=19), and clinical validation against blinded specialist interviews (n=24). Symptoms detected by the agent aligned well with those identified by specialists across symptoms. Users appreciated the agent's patience and systematic questioning, which supported engagement and expression of complex, hard-to-describe experiences. This preliminary work suggests conversational agents may serve as structured front-end tools for dementia assessment, highlighting interaction design considerations in sensitive healthcare contexts. 

---
# Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning 

**Authors**: Yijia Xiao, Edward Sun, Tong Chen, Fang Wu, Di Luo, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11420)  

**Abstract**: Developing professional, structured reasoning on par with human financial analysts and traders remains a central challenge in AI for finance, where markets demand interpretability and trust. Traditional time-series models lack explainability, while LLMs face challenges in turning natural-language analysis into disciplined, executable trades. Although reasoning LLMs have advanced in step-by-step planning and verification, their application to risk-sensitive financial decisions is underexplored. We present Trading-R1, a financially-aware model that incorporates strategic thinking and planning for comprehensive thesis composition, facts-grounded analysis, and volatility-adjusted decision making. Trading-R1 aligns reasoning with trading principles through supervised fine-tuning and reinforcement learning with a three-stage easy-to-hard curriculum. Training uses Tauric-TR1-DB, a 100k-sample corpus spanning 18 months, 14 equities, and five heterogeneous financial data sources. Evaluated on six major equities and ETFs, Trading-R1 demonstrates improved risk-adjusted returns and lower drawdowns compared to both open-source and proprietary instruction-following models as well as reasoning models. The system generates structured, evidence-based investment theses that support disciplined and interpretable trading decisions. Trading-R1 Terminal will be released at this https URL. 

---
# Intelligent Reservoir Decision Support: An Integrated Framework Combining Large Language Models, Advanced Prompt Engineering, and Multimodal Data Fusion for Real-Time Petroleum Operations 

**Authors**: Seyed Kourosh Mahjour, Seyed Saman Mahjour  

**Link**: [PDF](https://arxiv.org/pdf/2509.11376)  

**Abstract**: The petroleum industry faces unprecedented challenges in reservoir management, requiring rapid integration of complex multimodal datasets for real-time decision support. This study presents a novel integrated framework combining state-of-the-art large language models (GPT-4o, Claude 4 Sonnet, Gemini 2.5 Pro) with advanced prompt engineering techniques and multimodal data fusion for comprehensive reservoir analysis. The framework implements domain-specific retrieval-augmented generation (RAG) with over 50,000 petroleum engineering documents, chain-of-thought reasoning, and few-shot learning for rapid field adaptation. Multimodal integration processes seismic interpretations, well logs, and production data through specialized AI models with vision transformers. Field validation across 15 diverse reservoir environments demonstrates exceptional performance: 94.2% reservoir characterization accuracy, 87.6% production forecasting precision, and 91.4% well placement optimization success rate. The system achieves sub-second response times while maintaining 96.2% safety reliability with no high-risk incidents during evaluation. Economic analysis reveals 62-78% cost reductions (mean 72%) relative to traditional methods with 8-month payback period. Few-shot learning reduces field adaptation time by 72%, while automated prompt optimization achieves 89% improvement in reasoning quality. The framework processed real-time data streams with 96.2% anomaly detection accuracy and reduced environmental incidents by 45%. We provide detailed experimental protocols, baseline comparisons, ablation studies, and statistical significance testing to ensure reproducibility. This research demonstrates practical integration of cutting-edge AI technologies with petroleum domain expertise for enhanced operational efficiency, safety, and economic performance. 

---
# Transformer Enhanced Relation Classification: A Comparative Analysis of Contextuality, Data Efficiency and Sequence Complexity 

**Authors**: Bowen Jing, Yang Cui, Tianpeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11374)  

**Abstract**: In the era of large language model, relation extraction (RE) plays an important role in information extraction through the transformation of unstructured raw text into structured data (Wadhwa et al., 2023). In this paper, we systematically compare the performance of deep supervised learning approaches without transformers and those with transformers. We used a series of non-transformer architectures such as PA-LSTM(Zhang et al., 2017), C-GCN(Zhang et al., 2018), and AGGCN(attention guide GCN)(Guo et al., 2019), and a series of transformer architectures such as BERT, RoBERTa, and R-BERT(Wu and He, 2019). Our comparison included traditional metrics like micro F1, as well as evaluations in different scenarios, varying sentence lengths, and different percentages of the dataset for training. Our experiments were conducted on TACRED, TACREV, and RE-TACRED. The results show that transformer-based models outperform non-transformer models, achieving micro F1 scores of 80-90% compared to 64-67% for non-transformer models. Additionally, we briefly review the research journey in supervised relation classification and discuss the role and current status of large language models (LLMs) in relation extraction. 

---
# Beyond Autoregression: An Empirical Study of Diffusion Large Language Models for Code Generation 

**Authors**: Chengze li, Yitong Zhang, Jia Li, Liyi Cai, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.11252)  

**Abstract**: LLMs have become the mainstream approaches to code generation. Existing LLMs mainly employ autoregressive generation, i.e. generating code token-by-token from left to right. However, the underlying autoregressive generation has two limitations in code generation. First, autoregressive LLMs only generate a token at each step, showing low efficiency in practice. Second, programming is a non-sequential process involving back-and-forth editing, while autoregressive LLMs only employ the left-to-right generation order. These two intrinsic limitations hinder the further development of LLMs in code generation. Recently, diffusion LLMs have emerged as a promising alternative. Diffusion LLMs address the above limitations with two advances, including multi-token prediction (i.e. generating multiple tokens at each step) and flexible generation order (i.e. flexibly determining which positions to generate tokens). However, there is no systematic study exploring diffusion LLMs in code generation. To bridge the knowledge gap, we present the first empirical study of diffusion LLMs for code generation. Our study involves 9 representative diffusion LLMs and conduct experiments on 4 widely used benchmarks. Based on the results, we summarize the following findings. (1) Existing diffusion LLMs are competitive with autoregressive LLMs with similar sizes. (2) Diffusion LLMs have a stronger length extrapolation ability than autoregressive LLMs and perform better in long code understanding. (3) We explore factors impacting the effectiveness and efficiency of diffusion LLMs, and provide practical guidance. (4) We discuss several promising further directions to improve diffusion LLMs on code generation. We open-source all source code, data, and results to facilitate the following research. The code is publicly available at this https URL. 

---
# Evalet: Evaluating Large Language Models by Fragmenting Outputs into Functions 

**Authors**: Tae Soo Kim, Heechan Lee, Yoonjoo Lee, Joseph Seering, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.11206)  

**Abstract**: Practitioners increasingly rely on Large Language Models (LLMs) to evaluate generative AI outputs through "LLM-as-a-Judge" approaches. However, these methods produce holistic scores that obscure which specific elements influenced the assessments. We propose functional fragmentation, a method that dissects each output into key fragments and interprets the rhetoric functions that each fragment serves relative to evaluation criteria -- surfacing the elements of interest and revealing how they fulfill or hinder user goals. We instantiate this approach in Evalet, an interactive system that visualizes fragment-level functions across many outputs to support inspection, rating, and comparison of evaluations. A user study (N=10) found that, while practitioners struggled to validate holistic scores, our approach helped them identify 48% more evaluation misalignments. This helped them calibrate trust in LLM evaluations and rely on them to find more actionable issues in model outputs. Our work shifts LLM evaluation from quantitative scores toward qualitative, fine-grained analysis of model behavior. 

---
# Differentially-private text generation degrades output language quality 

**Authors**: Erion Çano, Ivan Habernal  

**Link**: [PDF](https://arxiv.org/pdf/2509.11176)  

**Abstract**: Ensuring user privacy by synthesizing data from large language models (LLMs) tuned under differential privacy (DP) has become popular recently. However, the impact of DP fine-tuned LLMs on the quality of the language and the utility of the texts they produce has not been investigated. In this work, we tune five LLMs with three corpora under four levels of privacy and assess the length, the grammatical correctness, and the lexical diversity of the text outputs they produce. We also probe the utility of the synthetic outputs in downstream classification tasks such as book genre recognition based on book descriptions and cause of death recognition based on verbal autopsies. The results indicate that LLMs tuned under stronger privacy constrains produce texts that are shorter by at least 77 %, that are less grammatically correct by at least 9 %, and are less diverse by at least 10 % in bi-gram diversity. Furthermore, the accuracy they reach in downstream classification tasks decreases, which might be detrimental to the usefulness of the generated synthetic data. 

---
# AQUA: Attention via QUery mAgnitudes for Memory and Compute Efficient Inference in LLMs 

**Authors**: Santhosh G S, Saurav Prakash, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2509.11155)  

**Abstract**: The quadratic complexity of the attention mechanism remains a fundamental barrier to scaling Large Language Models (LLMs) to longer contexts, creating a critical bottleneck in both computation and memory. To address this, we introduce AQUA (Attention via QUery mAgnitudes) a novel and versatile approximation strategy that significantly reduces the cost of attention with a graceful performance trade-off. Our method operates in two phases: an efficient offline step where we compute a universal, language agnostic projection matrix via SVD on a calibration dataset, and an online inference step where we project query and key vectors and dynamically select a sparse subset of dimensions based on the query's magnitude. We provide a formal theoretical analysis of AQUA, establishing the break-even point at which it becomes more computationally efficient than standard attention. Our empirical evaluations on state-of-the-art models like Llama-3.1-8B demonstrate that a 25% reduction in the attention dot-product computation can be achieved with a statistically insignificant impact on performance across a wide range of benchmarks. We further showcase the versatility of AQUA by demonstrating its ability to synergistically accelerate existing token eviction methods like H2O and to directly reduce KV-cache memory size. By offering a controllable knob to balance efficiency and accuracy, AQUA provides a practical and powerful tool for making large-scale LLM inference more accessible and sustainable. 

---
# We Argue to Agree: Towards Personality-Driven Argumentation-Based Negotiation Dialogue Systems for Tourism 

**Authors**: Priyanshu Priya, Saurav Dudhate, Desai Vishesh Yasheshbhai, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2509.11118)  

**Abstract**: Integrating argumentation mechanisms into negotiation dialogue systems improves conflict resolution through exchanges of arguments and critiques. Moreover, incorporating personality attributes enhances adaptability by aligning interactions with individuals' preferences and styles. To advance these capabilities in negotiation dialogue systems, we propose a novel Personality-driven Argumentation-based Negotiation Dialogue Generation (PAN-DG) task. To support this task, we introduce PACT, a dataset of Personality-driven Argumentation-based negotiation Conversations for Tourism sector. This dataset, generated using Large Language Models (LLMs), features three distinct personality profiles, viz. Argumentation Profile, Preference Profile, and Buying Style Profile to simulate a variety of negotiation scenarios involving diverse personalities. Thorough automatic and manual evaluations indicate that the dataset comprises high-quality dialogues. Further, we conduct comparative experiments between pre-trained and fine-tuned LLMs for the PAN-DG task. Multi-dimensional evaluation demonstrates that the fine-tuned LLMs effectively generate personality-driven rational responses during negotiations. This underscores the effectiveness of PACT in enhancing personalization and reasoning capabilities in negotiation dialogue systems, thereby establishing a foundation for future research in this domain. 

---
# Fluid Language Model Benchmarking 

**Authors**: Valentin Hofmann, David Heineman, Ian Magnusson, Kyle Lo, Jesse Dodge, Maarten Sap, Pang Wei Koh, Chun Wang, Hannaneh Hajishirzi, Noah A. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2509.11106)  

**Abstract**: Language model (LM) benchmarking faces several challenges: comprehensive evaluations are costly, benchmarks often fail to measure the intended capabilities, and evaluation quality can degrade due to labeling errors and benchmark saturation. Although various strategies have been proposed to mitigate these issues, they tend to address individual aspects in isolation, neglecting broader questions about overall evaluation quality. Here, we introduce Fluid Benchmarking, a new evaluation approach that advances LM benchmarking across multiple dimensions. Inspired by psychometrics, Fluid Benchmarking is based on the insight that the relative value of benchmark items depends on an LM's capability level, suggesting that evaluation should adapt to each LM. Methodologically, Fluid Benchmarking estimates an item response model based on existing LM evaluation results and uses the inferred quantities to select evaluation items dynamically, similar to computerized adaptive testing in education. In our experiments, we compare Fluid Benchmarking against the common practice of random item sampling as well as more sophisticated baselines, including alternative methods grounded in item response theory. We examine four dimensions -- efficiency, validity, variance, and saturation -- and find that Fluid Benchmarking achieves superior performance in all of them (e.g., higher validity and less variance on MMLU with fifty times fewer items). Our analysis shows that the two components of Fluid Benchmarking have distinct effects: item response theory, used to map performance into a latent ability space, increases validity, while dynamic item selection reduces variance. Overall, our results suggest that LM benchmarking can be substantially improved by moving beyond static evaluation. 

---
# The System Description of CPS Team for Track on Driving with Language of CVPR 2024 Autonomous Grand Challenge 

**Authors**: Jinghan Peng, Jingwen Wang, Xing Yu, Dehui Du  

**Link**: [PDF](https://arxiv.org/pdf/2509.11071)  

**Abstract**: This report outlines our approach using vision language model systems for the Driving with Language track of the CVPR 2024 Autonomous Grand Challenge. We have exclusively utilized the DriveLM-nuScenes dataset for training our models. Our systems are built on the LLaVA models, which we enhanced through fine-tuning with the LoRA and DoRA methods. Additionally, we have integrated depth information from open-source depth estimation models to enrich the training and inference processes. For inference, particularly with multiple-choice and yes/no questions, we adopted a Chain-of-Thought reasoning approach to improve the accuracy of the results. This comprehensive methodology enabled us to achieve a top score of 0.7799 on the validation set leaderboard, ranking 1st on the leaderboard. 

---
# The Psychogenic Machine: Simulating AI Psychosis, Delusion Reinforcement and Harm Enablement in Large Language Models 

**Authors**: Joshua Au Yeung, Jacopo Dalmasso, Luca Foschini, Richard JB Dobson, Zeljko Kraljevic  

**Link**: [PDF](https://arxiv.org/pdf/2509.10970)  

**Abstract**: Background: Emerging reports of "AI psychosis" are on the rise, where user-LLM interactions may exacerbate or induce psychosis or adverse psychological symptoms. The sycophantic and agreeable nature of LLMs can beneficial, it can become a vector for harm by reinforcing delusional beliefs in vulnerable users.
Methods: We introduce psychosis-bench, a novel benchmark designed to systematically evaluate the psychogenicity of LLMs comprimising 16 structured, 12-turn conversational scenarios simulating the progression of delusional themes(Erotic Delusions, Grandiose/Messianic Delusions, Referential Delusions) and potential harms. We evaluated eight prominent LLMs for Delusion Confirmation (DCS), Harm Enablement (HES), and Safety Intervention(SIS) across explicit and implicit conversational contexts.
Findings: Across 1,536 simulated conversation turns, all LLMs demonstrated psychogenic potential, showing a strong tendency to perpetuate rather than challenge delusions (mean DCS of 0.91 $\pm$0.88). Models frequently enabled harmful user requests (mean HES of 0.69 $\pm$0.84) and offered safety interventions in only roughly a third of applicable turns (mean SIS of 0.37 $\pm$0.48). 51 / 128 (39.8%) of scenarios had no safety interventions offered. Performance was significantly worse in implicit scenarios, models were more likely to confirm delusions and enable harm while offering fewer interventions (p < .001). A strong correlation was found between DCS and HES (rs = .77). Model performance varied widely, indicating that safety is not an emergent property of scale alone.
Conclusion: This study establishes LLM psychogenicity as a quantifiable risk and underscores the urgent need for re-thinking how we train LLMs. We frame this issue not merely as a technical challenge but as a public health imperative requiring collaboration between developers, policymakers, and healthcare professionals. 

---
# Testing for LLM response differences: the case of a composite null consisting of semantically irrelevant query perturbations 

**Authors**: Aranyak Acharyya, Carey E. Priebe, Hayden S. Helm  

**Link**: [PDF](https://arxiv.org/pdf/2509.10963)  

**Abstract**: Given an input query, generative models such as large language models produce a random response drawn from a response distribution. Given two input queries, it is natural to ask if their response distributions are the same. While traditional statistical hypothesis testing is designed to address this question, the response distribution induced by an input query is often sensitive to semantically irrelevant perturbations to the query, so much so that a traditional test of equality might indicate that two semantically equivalent queries induce statistically different response distributions. As a result, the outcome of the statistical test may not align with the user's requirements. In this paper, we address this misalignment by incorporating into the testing procedure consideration of a collection of semantically similar queries. In our setting, the mapping from the collection of user-defined semantically similar queries to the corresponding collection of response distributions is not known a priori and must be estimated, with a fixed budget. Although the problem we address is quite general, we focus our analysis on the setting where the responses are binary, show that the proposed test is asymptotically valid and consistent, and discuss important practical considerations with respect to power and computation. 

---
# When the Code Autopilot Breaks: Why LLMs Falter in Embedded Machine Learning 

**Authors**: Roberto Morabito, Guanghan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10946)  

**Abstract**: Large Language Models (LLMs) are increasingly used to automate software generation in embedded machine learning workflows, yet their outputs often fail silently or behave unpredictably. This article presents an empirical investigation of failure modes in LLM-powered ML pipelines, based on an autopilot framework that orchestrates data preprocessing, model conversion, and on-device inference code generation. We show how prompt format, model behavior, and structural assumptions influence both success rates and failure characteristics, often in ways that standard validation pipelines fail to detect. Our analysis reveals a diverse set of error-prone behaviors, including format-induced misinterpretations and runtime-disruptive code that compiles but breaks downstream. We derive a taxonomy of failure categories and analyze errors across multiple LLMs, highlighting common root causes and systemic fragilities. Though grounded in specific devices, our study reveals broader challenges in LLM-based code generation. We conclude by discussing directions for improving reliability and traceability in LLM-powered embedded ML systems. 

---
# Large Language Models for Security Operations Centers: A Comprehensive Survey 

**Authors**: Ali Habibzadeh, Farid Feyzi, Reza Ebrahimi Atani  

**Link**: [PDF](https://arxiv.org/pdf/2509.10858)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools capable of understanding and generating human-like text, offering transformative potential across diverse domains. The Security Operations Center (SOC), responsible for safeguarding digital infrastructure, represents one of these domains. SOCs serve as the frontline of defense in cybersecurity, tasked with continuous monitoring, detection, and response to incidents. However, SOCs face persistent challenges such as high alert volumes, limited resources, high demand for experts with advanced knowledge, delayed response times, and difficulties in leveraging threat intelligence effectively. In this context, LLMs can offer promising solutions by automating log analysis, streamlining triage, improving detection accuracy, and providing the required knowledge in less time. This survey systematically explores the integration of generative AI and more specifically LLMs into SOC workflow, providing a structured perspective on its capabilities, challenges, and future directions. We believe that this survey offers researchers and SOC managers a broad overview of the current state of LLM integration within academic study. To the best of our knowledge, this is the first comprehensive study to examine LLM applications in SOCs in details. 

---
# Towards Automated Error Discovery: A Study in Conversational AI 

**Authors**: Dominic Petrak, Thy Thy Tran, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2509.10833)  

**Abstract**: Although LLM-based conversational agents demonstrate strong fluency and coherence, they still produce undesirable behaviors (errors) that are challenging to prevent from reaching users during deployment. Recent research leverages large language models (LLMs) to detect errors and guide response-generation models toward improvement. However, current LLMs struggle to identify errors not explicitly specified in their instructions, such as those arising from updates to the response-generation model or shifts in user behavior. In this work, we introduce Automated Error Discovery, a framework for detecting and defining errors in conversational AI, and propose SEEED (Soft Clustering Extended Encoder-Based Error Detection), as an encoder-based approach to its implementation. We enhance the Soft Nearest Neighbor Loss by amplifying distance weighting for negative samples and introduce Label-Based Sample Ranking to select highly contrastive examples for better representation learning. SEEED outperforms adapted baselines -- including GPT-4o and Phi-4 -- across multiple error-annotated dialogue datasets, improving the accuracy for detecting unknown errors by up to 8 points and demonstrating strong generalization to unknown intent detection. 

---
# CultureSynth: A Hierarchical Taxonomy-Guided and Retrieval-Augmented Framework for Cultural Question-Answer Synthesis 

**Authors**: Xinyu Zhang, Pei Zhang, Shuang Luo, Jialong Tang, Yu Wan, Baosong Yang, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10886)  

**Abstract**: Cultural competence, defined as the ability to understand and adapt to multicultural contexts, is increasingly vital for large language models (LLMs) in global environments. While several cultural benchmarks exist to assess LLMs' cultural competence, current evaluations suffer from fragmented taxonomies, domain specificity, and heavy reliance on manual data annotation. To address these limitations, we introduce CultureSynth, a novel framework comprising (1) a comprehensive hierarchical multilingual cultural taxonomy covering 12 primary and 130 secondary topics, and (2) a Retrieval-Augmented Generation (RAG)-based methodology leveraging factual knowledge to synthesize culturally relevant question-answer pairs. The CultureSynth-7 synthetic benchmark contains 19,360 entries and 4,149 manually verified entries across 7 languages. Evaluation of 14 prevalent LLMs of different sizes reveals clear performance stratification led by ChatGPT-4o-Latest and Qwen2.5-72B-Instruct. The results demonstrate that a 3B-parameter threshold is necessary for achieving basic cultural competence, models display varying architectural biases in knowledge processing, and significant geographic disparities exist across models. We believe that CultureSynth offers a scalable framework for developing culturally aware AI systems while reducing reliance on manual annotation\footnote{Benchmark is available at this https URL.}. 

---
# HalluField: Detecting LLM Hallucinations via Field-Theoretic Modeling 

**Authors**: Minh Vu, Brian K. Tran, Syed A. Shah, Geigh Zollicoffer, Nhat Hoang-Xuan, Manish Bhattarai  

**Link**: [PDF](https://arxiv.org/pdf/2509.10753)  

**Abstract**: Large Language Models (LLMs) exhibit impressive reasoning and question-answering capabilities. However, they often produce inaccurate or unreliable content known as hallucinations. This unreliability significantly limits their deployment in high-stakes applications. Thus, there is a growing need for a general-purpose method to detect hallucinations in LLMs. In this work, we introduce HalluField, a novel field-theoretic approach for hallucination detection based on a parametrized variational principle and thermodynamics. Inspired by thermodynamics, HalluField models an LLM's response to a given query and temperature setting as a collection of discrete likelihood token paths, each associated with a corresponding energy and entropy. By analyzing how energy and entropy distributions vary across token paths under changes in temperature and likelihood, HalluField quantifies the semantic stability of a response. Hallucinations are then detected by identifying unstable or erratic behavior in this energy landscape. HalluField is computationally efficient and highly practical: it operates directly on the model's output logits without requiring fine-tuning or auxiliary neural networks. Notably, the method is grounded in a principled physical interpretation, drawing analogies to the first law of thermodynamics. Remarkably, by modeling LLM behavior through this physical lens, HalluField achieves state-of-the-art hallucination detection performance across models and datasets. 

---
# LLM in the Middle: A Systematic Review of Threats and Mitigations to Real-World LLM-based Systems 

**Authors**: Vitor Hugo Galhardo Moia, Igor Jochem Sanz, Gabriel Antonio Fontes Rebello, Rodrigo Duarte de Meneses, Briland Hitaj, Ulf Lindqvist  

**Link**: [PDF](https://arxiv.org/pdf/2509.10682)  

**Abstract**: The success and wide adoption of generative AI (GenAI), particularly large language models (LLMs), has attracted the attention of cybercriminals seeking to abuse models, steal sensitive data, or disrupt services. Moreover, providing security to LLM-based systems is a great challenge, as both traditional threats to software applications and threats targeting LLMs and their integration must be mitigated. In this survey, we shed light on security and privacy concerns of such LLM-based systems by performing a systematic review and comprehensive categorization of threats and defensive strategies considering the entire software and LLM life cycles. We analyze real-world scenarios with distinct characteristics of LLM usage, spanning from development to operation. In addition, threats are classified according to their severity level and to which scenarios they pertain, facilitating the identification of the most relevant threats. Recommended defense strategies are systematically categorized and mapped to the corresponding life cycle phase and possible attack strategies they attenuate. This work paves the way for consumers and vendors to understand and efficiently mitigate risks during integration of LLMs in their respective solutions or organizations. It also enables the research community to benefit from the discussion of open challenges and edge cases that may hinder the secure and privacy-preserving adoption of LLM-based systems. 

---
# Test-Time Warmup for Multimodal Large Language Models 

**Authors**: Nikita Rajaneesh, Thomas Zollo, Richard Zemel  

**Link**: [PDF](https://arxiv.org/pdf/2509.10641)  

**Abstract**: Multimodal Large Language Models (MLLMs) hold great promise for advanced reasoning at the intersection of text and images, yet they have not fully realized this potential. MLLMs typically integrate an LLM, a vision encoder, and a connector that maps the vision encoder's embeddings into the LLM's text embedding space. Although each component is pretrained on massive datasets with billions of samples, the entire multimodal model is typically trained on only thousands (or a few million) samples, which can result in weak performance on complex reasoning tasks. To address these shortcomings, instead of relying on extensive labeled datasets for fine-tuning, we propose a Test-Time Warmup method that adapts the MLLM per test instance by leveraging data from weakly supervised auxiliary tasks. With our approach, we observe a relative performance improvement of 4.03% on MMMU, 5.28% on VQA-Rad, and 1.63% on GQA on the Llama-Vision-Instruct model. Our method demonstrates that 'warming up' before inference can enhance MLLMs' robustness across diverse reasoning tasks. 

---
# Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction 

**Authors**: Yijun Liu, Yixuan Wang, Yuzhuang Xu, Shiyu Ji, Yang Xu, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2509.10798)  

**Abstract**: Large language models (LLMs) utilize key-value (KV) cache to store historical information during sequence processing. The size of KV cache grows linearly as the length of the sequence extends, which seriously affects memory usage and decoding efficiency. Current methods for KV cache eviction typically utilize the last window from the pre-filling phase as queries to compute the KV importance scores for eviction. Although this scheme is simple to implement, it tends to overly focus on local information, potentially leading to the neglect or omission of crucial global information. To mitigate this issue, we propose Judge Q, a novel training method which incorporates a soft token list. This method only tunes the model's embedding layer at a low training cost. By concatenating the soft token list at the end of the input sequence, we train these tokens' attention map to the original input sequence to align with that of the actual decoded tokens. In this way, the queries corresponding to the soft tokens can effectively capture global information and better evaluate the importance of the keys and values within the KV cache, thus maintaining decoding quality when KV cache is evicted. Under the same eviction budget, our method exhibits less performance degradation compared to existing eviction approaches. We validate our approach through experiments conducted on models such as Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3, using benchmarks including LongBench, RULER, and Needle-in-a-Haystack. Results indicate an improvement of approximately 1 point on the LongBench and over 3 points on RULER. This proposed methodology can be seamlessly integrated into existing open-source models with minimal training overhead, thereby enhancing performance in KV cache eviction scenarios. 

---
# No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes 

**Authors**: Iván Vicente Moreno Cencerrado, Arnau Padrés Masdemont, Anton Gonzalvez Hawthorne, David Demitri Africa, Lorenzo Pacchiardi  

**Link**: [PDF](https://arxiv.org/pdf/2509.10625)  

**Abstract**: Do large language models (LLMs) anticipate when they will answer correctly? To study this, we extract activations after a question is read but before any tokens are generated, and train linear probes to predict whether the model's forthcoming answer will be correct. Across three open-source model families ranging from 7 to 70 billion parameters, projections on this "in-advance correctness direction" trained on generic trivia questions predict success in distribution and on diverse out-of-distribution knowledge datasets, outperforming black-box baselines and verbalised predicted confidence. Predictive power saturates in intermediate layers, suggesting that self-assessment emerges mid-computation. Notably, generalisation falters on questions requiring mathematical reasoning. Moreover, for models responding "I don't know", doing so strongly correlates with the probe score, indicating that the same direction also captures confidence. By complementing previous results on truthfulness and other behaviours obtained with probes and sparse auto-encoders, our work contributes essential findings to elucidate LLM internals. 

---
# Smart Trial: Evaluating the Use of Large Language Models for Recruiting Clinical Trial Participants via Social Media 

**Authors**: Xiaofan Zhou, Zisu Wang, Janice Krieger, Mohan Zalake, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.10584)  

**Abstract**: Clinical trials (CT) are essential for advancing medical research and treatment, yet efficiently recruiting eligible participants -- each of whom must meet complex eligibility criteria -- remains a significant challenge. Traditional recruitment approaches, such as advertisements or electronic health record screening within hospitals, are often time-consuming and geographically constrained. This work addresses the recruitment challenge by leveraging the vast amount of health-related information individuals share on social media platforms. With the emergence of powerful large language models (LLMs) capable of sophisticated text understanding, we pose the central research question: Can LLM-driven tools facilitate CT recruitment by identifying potential participants through their engagement on social media? To investigate this question, we introduce TRIALQA, a novel dataset comprising two social media collections from the subreddits on colon cancer and prostate cancer. Using eligibility criteria from public real-world CTs, experienced annotators are hired to annotate TRIALQA to indicate (1) whether a social media user meets a given eligibility criterion and (2) the user's stated reasons for interest in participating in CT. We benchmark seven widely used LLMs on these two prediction tasks, employing six distinct training and inference strategies. Our extensive experiments reveal that, while LLMs show considerable promise, they still face challenges in performing the complex, multi-hop reasoning needed to accurately assess eligibility criteria. 

---
# Quality Assessment of Tabular Data using Large Language Models and Code Generation 

**Authors**: Ashlesha Akella, Akshar Kaul, Krishnasuri Narayanam, Sameep Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2509.10572)  

**Abstract**: Reliable data quality is crucial for downstream analysis of tabular datasets, yet rule-based validation often struggles with inefficiency, human intervention, and high computational costs. We present a three-stage framework that combines statistical inliner detection with LLM-driven rule and code generation. After filtering data samples through traditional clustering, we iteratively prompt LLMs to produce semantically valid quality rules and synthesize their executable validators through code-generating LLMs. To generate reliable quality rules, we aid LLMs with retrieval-augmented generation (RAG) by leveraging external knowledge sources and domain-specific few-shot examples. Robust guardrails ensure the accuracy and consistency of both rules and code snippets. Extensive evaluations on benchmark datasets confirm the effectiveness of our approach. 

---
# Uncovering the Vulnerability of Large Language Models in the Financial Domain via Risk Concealment 

**Authors**: Gang Cheng, Haibo Jin, Wenbin Zhang, Haohan Wang, Jun Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10546)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into financial applications, yet existing red-teaming research primarily targets harmful content, largely neglecting regulatory risks. In this work, we aim to investigate the vulnerability of financial LLMs through red-teaming approaches. We introduce Risk-Concealment Attacks (RCA), a novel multi-turn framework that iteratively conceals regulatory risks to provoke seemingly compliant yet regulatory-violating responses from LLMs. To enable systematic evaluation, we construct FIN-Bench, a domain-specific benchmark for assessing LLM safety in financial contexts. Extensive experiments on FIN-Bench demonstrate that RCA effectively bypasses nine mainstream LLMs, achieving an average attack success rate (ASR) of 93.18%, including 98.28% on GPT-4.1 and 97.56% on OpenAI o1. These findings reveal a critical gap in current alignment techniques and underscore the urgent need for stronger moderation mechanisms in financial domains. We hope this work offers practical insights for advancing robust and domain-aware LLM alignment. 

---
# DualAlign: Generating Clinically Grounded Synthetic Data 

**Authors**: Rumeng Li, Xun Wang, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10538)  

**Abstract**: Synthetic clinical data are increasingly important for advancing AI in healthcare, given strict privacy constraints on real-world EHRs, limited availability of annotated rare-condition data, and systemic biases in observational datasets. While large language models (LLMs) can generate fluent clinical text, producing synthetic data that is both realistic and clinically meaningful remains challenging. We introduce DualAlign, a framework that enhances statistical fidelity and clinical plausibility through dual alignment: (1) statistical alignment, which conditions generation on patient demographics and risk factors; and (2) semantic alignment, which incorporates real-world symptom trajectories to guide content generation. Using Alzheimer's disease (AD) as a case study, DualAlign produces context-grounded symptom-level sentences that better reflect real-world clinical documentation. Fine-tuning an LLaMA 3.1-8B model with a combination of DualAlign-generated and human-annotated data yields substantial performance gains over models trained on gold data alone or unguided synthetic baselines. While DualAlign does not fully capture longitudinal complexity, it offers a practical approach for generating clinically grounded, privacy-preserving synthetic data to support low-resource clinical text analysis. 

---
# Gene-R1: Reasoning with Data-Augmented Lightweight LLMs for Gene Set Analysis 

**Authors**: Zhizheng Wang, Yifan Yang, Qiao Jin, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10575)  

**Abstract**: The gene set analysis (GSA) is a foundational approach for uncovering the molecular functions associated with a group of genes. Recently, LLM-powered methods have emerged to annotate gene sets with biological functions together with coherent explanatory insights. However, existing studies primarily focus on proprietary models, which have been shown to outperform their open-source counterparts despite concerns over cost and data privacy. Furthermore, no research has investigated the application of advanced reasoning strategies to the GSA task. To address this gap, we introduce Gene-R1, a data-augmented learning framework that equips lightweight and open-source LLMs with step-by-step reasoning capabilities tailored to GSA. Experiments on 1,508 in-distribution gene sets demonstrate that Gene-R1 achieves substantial performance gains, matching commercial LLMs. On 106 out-of-distribution gene sets, Gene-R1 performs comparably to both commercial and large-scale LLMs, exhibiting robust generalizability across diverse gene sources. 

---
# The Anti-Ouroboros Effect: Emergent Resilience in Large Language Models from Recursive Selective Feedback 

**Authors**: Sai Teja Reddy Adapala  

**Link**: [PDF](https://arxiv.org/pdf/2509.10509)  

**Abstract**: The stability of recursively trained large language models (LLMs) is a foundational problem for AI safety. Prevailing theory predicts model collapse, a progressive degradation when models are trained on their own output. We challenge this narrative by introducing a selective feedback mechanism. Contrary to expectation, instead of merely slowing decay, our experiments provide strong evidence that this pressure reverses it, inducing a statistically significant performance improvement in a Gemma 2B model on a complex summarization task. We name this phenomenon the Anti-Ouroboros Effect. We contrast this with a foundational experiment using a simple classifier, where the theoretical degenerative loop was validated, highlighting the unique dynamics of high-dimensional models. Our findings establish that systemic resilience can be an emergent property of LLMs under simple selection pressure, suggesting a powerful and scalable principle for developing safer and more robust AI systems. Across five generations, a quality-filtered condition improved by 6.6% in ROUGE-L F1 score, whereas an unfiltered control degraded by 3.5% and a random-filter control degraded by 4.2% 

---
# Learning Decomposed Contextual Token Representations from Pretrained and Collaborative Signals for Generative Recommendation 

**Authors**: Yifan Liu, Yaokun Liu, Zelin Li, Zhenrui Yue, Gyuseok Lee, Ruichen Yao, Yang Zhang, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10468)  

**Abstract**: Recent advances in generative recommenders adopt a two-stage paradigm: items are first tokenized into semantic IDs using a pretrained tokenizer, and then large language models (LLMs) are trained to generate the next item via sequence-to-sequence modeling. However, these two stages are optimized for different objectives: semantic reconstruction during tokenizer pretraining versus user interaction modeling during recommender training. This objective misalignment leads to two key limitations: (i) suboptimal static tokenization, where fixed token assignments fail to reflect diverse usage contexts; and (ii) discarded pretrained semantics, where pretrained knowledge - typically from language model embeddings - is overwritten during recommender training on user interactions. To address these limitations, we propose to learn DEcomposed COntextual Token Representations (DECOR), a unified framework that preserves pretrained semantics while enhancing the adaptability of token embeddings. DECOR introduces contextualized token composition to refine token embeddings based on user interaction context, and decomposed embedding fusion that integrates pretrained codebook embeddings with newly learned collaborative embeddings. Experiments on three real-world datasets demonstrate that DECOR consistently outperforms state-of-the-art baselines in recommendation performance. Our code will be made available upon publication. 

---
# Formal Reasoning for Intelligent QA Systems: A Case Study in the Educational Domain 

**Authors**: Tuan Bui, An Nguyen, Phat Thai, Minh Hua, Ngan Pham L.N., Ngan Pham T.B., Dung Le, Long Nguyen, Thanh-Tung Tran, Thang Bui, Tho Quan  

**Link**: [PDF](https://arxiv.org/pdf/2509.11572)  

**Abstract**: Reasoning is essential for closed-domain QA systems in which procedural correctness and policy compliance are critical. While large language models (LLMs) have shown strong performance on many reasoning tasks, recent work reveals that their reasoning traces are often unfaithful - serving more as plausible justifications than as causally grounded derivations. Efforts to combine LLMs with symbolic engines (e.g., Prover9, Z3) have improved reliability but remain limited to static forms of logic, struggling with dynamic, state-based reasoning such as multi-step progressions and conditional transitions.
In this paper, we propose MCFR (Model Checking for Formal Reasoning), a neuro-symbolic framework that integrates LLMs with model checking to support property verification. MCFR translates natural language into formal specifications and verifies them over transition models. To support evaluation, we introduce EduMC-QA, a benchmark dataset grounded in real academic procedures. Our results show that MCFR improves reasoning faithfulness and interpretability, offering a viable path toward verifiable QA in high-stakes closed-domain applications. In addition to evaluating MCFR, we compare its performance with state-of-the-art LLMs such as ChatGPT, DeepSeek, and Claude to contextualize its effectiveness. 

---
# Decoding in Latent Spaces for Efficient Inference in LLM-based Recommendation 

**Authors**: Chengbing Wang, Yang Zhang, Zhicheng Wang, Tianhao Shi, Keqin Bao, Fuli Feng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2509.11524)  

**Abstract**: Fine-tuning large language models (LLMs) for recommendation in a generative manner has delivered promising results, but encounters significant inference overhead due to autoregressive decoding in the language space. This work explores bypassing language-space decoding by directly matching candidate items with the LLM's internal thought representations in the latent space, eliminating the time-consuming autoregressive process to reduce computational costs. Towards this, we introduce Light Latent-space Decoding (L2D), an effective and efficient latent-space decoding method. L2D represents user-preferred items by using the hidden states of test sequences reflecting the LLM's internal thought, and obtains candidate item representations from the hidden states of training sequences labeled with the corresponding candidate items. It then matches the two types of representations to decode items, achieving latent-space decoding. In this way, it enables efficient decoding without altering the LLM's generative tuning paradigm, thereby preserving performance. Extensive empirical results demonstrate that L2D is more than 10x faster than language-space decoding while maintaining or enhancing performance. 

---
# ReFineG: Synergizing Small Supervised Models and LLMs for Low-Resource Grounded Multimodal NER 

**Authors**: Jielong Tang, Shuang Wang, Zhenxing Wang, Jianxing Yu, Jian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.10975)  

**Abstract**: Grounded Multimodal Named Entity Recognition (GMNER) extends traditional NER by jointly detecting textual mentions and grounding them to visual regions. While existing supervised methods achieve strong performance, they rely on costly multimodal annotations and often underperform in low-resource domains. Multimodal Large Language Models (MLLMs) show strong generalization but suffer from Domain Knowledge Conflict, producing redundant or incorrect mentions for domain-specific entities. To address these challenges, we propose ReFineG, a three-stage collaborative framework that integrates small supervised models with frozen MLLMs for low-resource GMNER. In the Training Stage, a domain-aware NER data synthesis strategy transfers LLM knowledge to small models with supervised training while avoiding domain knowledge conflicts. In the Refinement Stage, an uncertainty-based mechanism retains confident predictions from supervised models and delegates uncertain ones to the MLLM. In the Grounding Stage, a multimodal context selection algorithm enhances visual grounding through analogical reasoning. In the CCKS2025 GMNER Shared Task, ReFineG ranked second with an F1 score of 0.6461 on the online leaderboard, demonstrating its effectiveness with limited annotations. 

---
# Do Large Language Models Favor Recent Content? A Study on Recency Bias in LLM-Based Reranking 

**Authors**: Hanpei Fang, Sijie Tao, Nuo Chen, Kai-Xin Chang, Tetsuya Sakai  

**Link**: [PDF](https://arxiv.org/pdf/2509.11353)  

**Abstract**: Large language models (LLMs) are increasingly deployed in information systems, including being used as second-stage rerankers in information retrieval pipelines, yet their susceptibility to recency bias has received little attention. We investigate whether LLMs implicitly favour newer documents by prepending artificial publication dates to passages in the TREC Deep Learning passage retrieval collections in 2021 (DL21) and 2022 (DL22). Across seven models, GPT-3.5-turbo, GPT-4o, GPT-4, LLaMA-3 8B/70B, and Qwen-2.5 7B/72B, "fresh" passages are consistently promoted, shifting the Top-10's mean publication year forward by up to 4.78 years and moving individual items by as many as 95 ranks in our listwise reranking experiments. Although larger models attenuate the effect, none eliminate it. We also observe that the preference of LLMs between two passages with an identical relevance level can be reversed by up to 25% on average after date injection in our pairwise preference experiments. These findings provide quantitative evidence of a pervasive recency bias in LLMs and highlight the importance of effective bias-mitigation strategies. 

---
# CBP-Tuning: Efficient Local Customization for Black-box Large Language Models 

**Authors**: Jiaxuan Zhao, Naibin Gu, Yuchen Feng, Xiyu Liu, Peng Fu, Zheng Lin, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12112)  

**Abstract**: The high costs of customizing large language models (LLMs) fundamentally limit their adaptability to user-specific needs. Consequently, LLMs are increasingly offered as cloud-based services, a paradigm that introduces critical limitations: providers struggle to support personalized customization at scale, while users face privacy risks when exposing sensitive data. To address this dual challenge, we propose Customized Black-box Prompt Tuning (CBP-Tuning), a novel framework that facilitates efficient local customization while preserving bidirectional privacy. Specifically, we design a two-stage framework: (1) a prompt generator trained on the server-side to capture domain-specific and task-agnostic capabilities, and (2) user-side gradient-free optimization that tailors soft prompts for individual tasks. This approach eliminates the need for users to access model weights or upload private data, requiring only a single customized vector per task while achieving effective adaptation. Furthermore, the evaluation of CBP-Tuning in the commonsense reasoning, medical and financial domain settings demonstrates superior performance compared to baselines, showcasing its advantages in task-agnostic processing and privacy preservation. 

---
# Steering Language Models in Multi-Token Generation: A Case Study on Tense and Aspect 

**Authors**: Alina Klerings, Jannik Brinkmann, Daniel Ruffinelli, Simone Ponzetto  

**Link**: [PDF](https://arxiv.org/pdf/2509.12065)  

**Abstract**: Large language models (LLMs) are able to generate grammatically well-formed text, but how do they encode their syntactic knowledge internally? While prior work has focused largely on binary grammatical contrasts, in this work, we study the representation and control of two multidimensional hierarchical grammar phenomena - verb tense and aspect - and for each, identify distinct, orthogonal directions in residual space using linear discriminant analysis. Next, we demonstrate causal control over both grammatical features through concept steering across three generation tasks. Then, we use these identified features in a case study to investigate factors influencing effective steering in multi-token generation. We find that steering strength, location, and duration are crucial parameters for reducing undesirable side effects such as topic shift and degeneration. Our findings suggest that models encode tense and aspect in structurally organized, human-like ways, but effective control of such features during generation is sensitive to multiple factors and requires manual tuning or automated optimization. 

---
# ToolRM: Outcome Reward Models for Tool-Calling Large Language Models 

**Authors**: Mayank Agarwal, Ibrahim Abdelaziz, Kinjal Basu, Merve Unuvar, Luis A. Lastras, Yara Rizk, Pavan Kapanipathi  

**Link**: [PDF](https://arxiv.org/pdf/2509.11963)  

**Abstract**: As large language models (LLMs) increasingly interact with external tools, reward modeling for tool use has become a critical yet underexplored area. Existing reward models, trained primarily on natural language outputs, struggle to evaluate tool-based reasoning and execution. To quantify this gap, we introduce FC-RewardBench, the first benchmark designed to systematically assess reward models' performance in tool-calling scenarios. Our analysis shows that current reward models often miss key signals of effective tool use, highlighting the need for domain-specific modeling. To address this, we propose a training framework for outcome-based reward models using data synthesized from permissively licensed, open-weight LLMs. We train models ranging from 1.7B to 14B parameters and evaluate them across seven out-of-domain benchmarks. These models consistently outperform general-purpose baselines, achieving up to 25\% average improvement in downstream task performance and enabling data-efficient fine-tuning through reward-guided filtering. 

---
# Pun Unintended: LLMs and the Illusion of Humor Understanding 

**Authors**: Alessandro Zangari, Matteo Marcuzzo, Andrea Albarelli, Mohammad Taher Pilehvar, Jose Camacho-Collados  

**Link**: [PDF](https://arxiv.org/pdf/2509.12158)  

**Abstract**: Puns are a form of humorous wordplay that exploits polysemy and phonetic similarity. While LLMs have shown promise in detecting puns, we show in this paper that their understanding often remains shallow, lacking the nuanced grasp typical of human interpretation. By systematically analyzing and reformulating existing pun benchmarks, we demonstrate how subtle changes in puns are sufficient to mislead LLMs. Our contributions include comprehensive and nuanced pun detection benchmarks, human evaluation across recent LLMs, and an analysis of the robustness challenges these models face in processing puns. 

---
# XplaiNLP at CheckThat! 2025: Multilingual Subjectivity Detection with Finetuned Transformers and Prompt-Based Inference with Large Language Models 

**Authors**: Ariana Sahitaj, Jiaao Li, Pia Wenzel Neves, Fedor Splitt, Premtim Sahitaj, Charlott Jakob, Veronika Solopova, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2509.12130)  

**Abstract**: This notebook reports the XplaiNLP submission to the CheckThat! 2025 shared task on multilingual subjectivity detection. We evaluate two approaches: (1) supervised fine-tuning of transformer encoders, EuroBERT, XLM-RoBERTa, and German-BERT, on monolingual and machine-translated training data; and (2) zero-shot prompting using two LLMs: o3-mini for Annotation (rule-based labelling) and gpt-4.1-mini for DoubleDown (contrastive rewriting) and Perspective (comparative reasoning). The Annotation Approach achieves 1st place in the Italian monolingual subtask with an F_1 score of 0.8104, outperforming the baseline of 0.6941. In the Romanian zero-shot setting, the fine-tuned XLM-RoBERTa model obtains an F_1 score of 0.7917, ranking 3rd and exceeding the baseline of 0.6461. The same model also performs reliably in the multilingual task and improves over the baseline in Greek. For German, a German-BERT model fine-tuned on translated training data from typologically related languages yields competitive performance over the baseline. In contrast, performance in the Ukrainian and Polish zero-shot settings falls slightly below the respective baselines, reflecting the challenge of generalization in low-resource cross-lingual scenarios. 

---
# From Fuzzy Speech to Medical Insight: Benchmarking LLMs on Noisy Patient Narratives 

**Authors**: Eden Mama, Liel Sheri, Yehudit Aperstein, Alexander Apartsin  

**Link**: [PDF](https://arxiv.org/pdf/2509.11803)  

**Abstract**: The widespread adoption of large language models (LLMs) in healthcare raises critical questions about their ability to interpret patient-generated narratives, which are often informal, ambiguous, and noisy. Existing benchmarks typically rely on clean, structured clinical text, offering limited insight into model performance under realistic conditions. In this work, we present a novel synthetic dataset designed to simulate patient self-descriptions characterized by varying levels of linguistic noise, fuzzy language, and layperson terminology. Our dataset comprises clinically consistent scenarios annotated with ground-truth diagnoses, spanning a spectrum of communication clarity to reflect diverse real-world reporting styles. Using this benchmark, we fine-tune and evaluate several state-of-the-art models (LLMs), including BERT-based and encoder-decoder T5 models. To support reproducibility and future research, we release the Noisy Diagnostic Benchmark (NDB), a structured dataset of noisy, synthetic patient descriptions designed to stress-test and compare the diagnostic capabilities of large language models (LLMs) under realistic linguistic conditions. We made the benchmark available for the community: this https URL 

---
# Designing LLMs for cultural sensitivity: Evidence from English-Japanese translation 

**Authors**: Helene Tenzer, Oumnia Abidi, Stefan Feuerriegel  

**Link**: [PDF](https://arxiv.org/pdf/2509.11921)  

**Abstract**: Large language models (LLMs) are increasingly used in everyday communication, including multilingual interactions across different cultural contexts. While LLMs can now generate near-perfect literal translations, it remains unclear whether LLMs support culturally appropriate communication. In this paper, we analyze the cultural sensitivity of different LLM designs when applied to English-Japanese translations of workplace e-mails. Here, we vary the prompting strategies: (1) naive "just translate" prompts, (2) audience-targeted prompts specifying the recipient's cultural background, and (3) instructional prompts with explicit guidance on Japanese communication norms. Using a mixed-methods study, we then analyze culture-specific language patterns to evaluate how well translations adapt to cultural norms. Further, we examine the appropriateness of the tone of the translations as perceived by native speakers. We find that culturally-tailored prompting can improve cultural fit, based on which we offer recommendations for designing culturally inclusive LLMs in multilingual settings. 

---
# User eXperience Perception Insights Dataset (UXPID): Synthetic User Feedback from Public Industrial Forums 

**Authors**: Mikhail Kulyabin, Jan Joosten, Choro Ulan uulu, Nuno Miguel Martins Pacheco, Fabian Ries, Filippos Petridis, Jan Bosch, Helena Holmström Olsson  

**Link**: [PDF](https://arxiv.org/pdf/2509.11777)  

**Abstract**: Customer feedback in industrial forums reflect a rich but underexplored source of insight into real-world product experience. These publicly shared discussions offer an organic view of user expectations, frustrations, and success stories shaped by the specific contexts of use. Yet, harnessing this information for systematic analysis remains challenging due to the unstructured and domain-specific nature of the content. The lack of structure and specialized vocabulary makes it difficult for traditional data analysis techniques to accurately interpret, categorize, and quantify the feedback, thereby limiting its potential to inform product development and support strategies. To address these challenges, this paper presents the User eXperience Perception Insights Dataset (UXPID), a collection of 7130 artificially synthesized and anonymized user feedback branches extracted from a public industrial automation forum. Each JavaScript object notation (JSON) record contains multi-post comments related to specific hardware and software products, enriched with metadata and contextual conversation data. Leveraging a large language model (LLM), each branch is systematically analyzed and annotated for UX insights, user expectations, severity and sentiment ratings, and topic classifications. The UXPID dataset is designed to facilitate research in user requirements, user experience (UX) analysis, and AI-driven feedback processing, particularly where privacy and licensing restrictions limit access to real-world data. UXPID supports the training and evaluation of transformer-based models for tasks such as issue detection, sentiment analysis, and requirements extraction in the context of technical forums. 

---
# When Curiosity Signals Danger: Predicting Health Crises Through Online Medication Inquiries 

**Authors**: Dvora Goncharok, Arbel Shifman, Alexander Apartsin, Yehudit Aperstein  

**Link**: [PDF](https://arxiv.org/pdf/2509.11802)  

**Abstract**: Online medical forums are a rich and underutilized source of insight into patient concerns, especially regarding medication use. Some of the many questions users pose may signal confusion, misuse, or even the early warning signs of a developing health crisis. Detecting these critical questions that may precede severe adverse events or life-threatening complications is vital for timely intervention and improving patient safety. This study introduces a novel annotated dataset of medication-related questions extracted from online forums. Each entry is manually labelled for criticality based on clinical risk factors. We benchmark the performance of six traditional machine learning classifiers using TF-IDF textual representations, alongside three state-of-the-art large language model (LLM)-based classification approaches that leverage deep contextual understanding. Our results highlight the potential of classical and modern methods to support real-time triage and alert systems in digital health spaces. The curated dataset is made publicly available to encourage further research at the intersection of patient-generated data, natural language processing, and early warning systems for critical health events. The dataset and benchmark are available at: this https URL. 

---
# HalluDetect: Detecting, Mitigating, and Benchmarking Hallucinations in Conversational Systems 

**Authors**: Spandan Anaokar, Shrey Ganatra, Harshvivek Kashid, Swapnil Bhattacharyya, Shruti Nair, Reshma Sekhar, Siddharth Manohar, Rahul Hemrajani, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2509.11619)  

**Abstract**: Large Language Models (LLMs) are widely used in industry but remain prone to hallucinations, limiting their reliability in critical applications. This work addresses hallucination reduction in consumer grievance chatbots built using LLaMA 3.1 8B Instruct, a compact model frequently used in industry. We develop HalluDetect, an LLM-based hallucination detection system that achieves an F1 score of 69% outperforming baseline detectors by 25.44%. Benchmarking five chatbot architectures, we find that out of them, AgentBot minimizes hallucinations to 0.4159 per turn while maintaining the highest token accuracy (96.13%), making it the most effective mitigation strategy. Our findings provide a scalable framework for hallucination mitigation, demonstrating that optimized inference strategies can significantly improve factual accuracy. While applied to consumer law, our approach generalizes to other high-risk domains, enhancing trust in LLM-driven assistants. We will release the code and dataset 

---
# D$^2$HScore: Reasoning-Aware Hallucination Detection via Semantic Breadth and Depth Analysis in LLMs 

**Authors**: Yue Ding, Xiaofang Zhu, Tianze Xia, Junfei Wu, Xinlong Chen, Qiang Liu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11569)  

**Abstract**: Although large Language Models (LLMs) have achieved remarkable success, their practical application is often hindered by the generation of non-factual content, which is called "hallucination". Ensuring the reliability of LLMs' outputs is a critical challenge, particularly in high-stakes domains such as finance, security, and healthcare. In this work, we revisit hallucination detection from the perspective of model architecture and generation dynamics. Leveraging the multi-layer structure and autoregressive decoding process of LLMs, we decompose hallucination signals into two complementary dimensions: the semantic breadth of token representations within each layer, and the semantic depth of core concepts as they evolve across layers. Based on this insight, we propose \textbf{D$^2$HScore (Dispersion and Drift-based Hallucination Score)}, a training-free and label-free framework that jointly measures: (1) \textbf{Intra-Layer Dispersion}, which quantifies the semantic diversity of token representations within each layer; and (2) \textbf{Inter-Layer Drift}, which tracks the progressive transformation of key token representations across layers. To ensure drift reflects the evolution of meaningful semantics rather than noisy or redundant tokens, we guide token selection using attention signals. By capturing both the horizontal and vertical dynamics of representation during inference, D$^2$HScore provides an interpretable and lightweight proxy for hallucination detection. Extensive experiments across five open-source LLMs and five widely used benchmarks demonstrate that D$^2$HScore consistently outperforms existing training-free baselines. 

---
# A Dynamic Knowledge Update-Driven Model with Large Language Models for Fake News Detection 

**Authors**: Di Jin, Jun Yang, Xiaobao Wang, Junwei Zhang, Shuqi Li, Dongxiao He  

**Link**: [PDF](https://arxiv.org/pdf/2509.11687)  

**Abstract**: As the Internet and social media evolve rapidly, distinguishing credible news from a vast amount of complex information poses a significant challenge. Due to the suddenness and instability of news events, the authenticity labels of news can potentially shift as events develop, making it crucial for fake news detection to obtain the latest event updates. Existing methods employ retrieval-augmented generation to fill knowledge gaps, but they suffer from issues such as insufficient credibility of retrieved content and interference from noisy information. We propose a dynamic knowledge update-driven model for fake news detection (DYNAMO), which leverages knowledge graphs to achieve continuous updating of new knowledge and integrates with large language models to fulfill dual functions: news authenticity detection and verification of new knowledge correctness, solving the two key problems of ensuring the authenticity of new knowledge and deeply mining news semantics. Specifically, we first construct a news-domain-specific knowledge graph. Then, we use Monte Carlo Tree Search to decompose complex news and verify them step by step. Finally, we extract and update new knowledge from verified real news texts and reasoning paths. Experimental results demonstrate that DYNAMO achieves the best performance on two real-world datasets. 

---
# AKCIT-FN at CheckThat! 2025: Switching Fine-Tuned SLMs and LLM Prompting for Multilingual Claim Normalization 

**Authors**: Fabrycio Leite Nakano Almada, Kauan Divino Pouso Mariano, Maykon Adriell Dutra, Victor Emanuel da Silva Monteiro, Juliana Resplande Sant'Anna Gomes, Arlindo Rodrigues Galvão Filho, Anderson da Silva Soares  

**Link**: [PDF](https://arxiv.org/pdf/2509.11496)  

**Abstract**: Claim normalization, the transformation of informal social media posts into concise, self-contained statements, is a crucial step in automated fact-checking pipelines. This paper details our submission to the CLEF-2025 CheckThat! Task~2, which challenges systems to perform claim normalization across twenty languages, divided into thirteen supervised (high-resource) and seven zero-shot (no training data) tracks.
Our approach, leveraging fine-tuned Small Language Models (SLMs) for supervised languages and Large Language Model (LLM) prompting for zero-shot scenarios, achieved podium positions (top three) in fifteen of the twenty languages. Notably, this included second-place rankings in eight languages, five of which were among the seven designated zero-shot languages, underscoring the effectiveness of our LLM-based zero-shot strategy. For Portuguese, our initial development language, our system achieved an average METEOR score of 0.5290, ranking third. All implementation artifacts, including inference, training, evaluation scripts, and prompt configurations, are publicly available at this https URL. 

---
# Improving LLMs' Learning for Coreference Resolution 

**Authors**: Yujian Gan, Yuan Liang, Yanni Lin, Juntao Yu, Massimo Poesio  

**Link**: [PDF](https://arxiv.org/pdf/2509.11466)  

**Abstract**: Coreference Resolution (CR) is crucial for many NLP tasks, but existing LLMs struggle with hallucination and under-performance. In this paper, we investigate the limitations of existing LLM-based approaches to CR-specifically the Question-Answering (QA) Template and Document Template methods and propose two novel techniques: Reversed Training with Joint Inference and Iterative Document Generation. Our experiments show that Reversed Training improves the QA Template method, while Iterative Document Generation eliminates hallucinations in the generated source text and boosts coreference resolution. Integrating these methods and techniques offers an effective and robust solution to LLM-based coreference resolution. 

---
# PeruMedQA: Benchmarking Large Language Models (LLMs) on Peruvian Medical Exams - Dataset Construction and Evaluation 

**Authors**: Rodrigo M. Carrillo-Larco, Jesus Lovón Melgarejo, Manuel Castillo-Cara, Gusseppe Bravo-Rocca  

**Link**: [PDF](https://arxiv.org/pdf/2509.11517)  

**Abstract**: BACKGROUND: Medical large language models (LLMS) have demonstrated remarkable performance in answering medical examinations. However, the extent to which this high performance is transferable to medical questions in Spanish and from a Latin American country remains unexplored. This knowledge is crucial as LLM-based medical applications gain traction in Latin America. AIMS: to build a dataset of questions from medical examinations taken by Peruvian physicians pursuing specialty training; to fine-tune a LLM on this dataset; to evaluate and compare the performance in terms of accuracy between vanilla LLMs and the fine-tuned LLM. METHODS: We curated PeruMedQA, a multiple-choice question-answering (MCQA) datasets containing 8,380 questions spanning 12 medical domains (2018-2025). We selected eight medical LLMs including medgemma-4b-it and medgemma-27b-text-it, and developed zero-shot task-specific prompts to answer the questions appropriately. We employed parameter-efficient fine tuning (PEFT)and low-rant adaptation (LoRA) to fine-tune medgemma-4b-it utilizing all questions except those from 2025 (test set). RESULTS: medgemma-27b-text-it outperformed all other models, achieving a proportion of correct answers exceeding 90% in several instances. LLMs with <10 billion parameters exhibited <60% of correct answers, while some exams yielded results <50%. The fine-tuned version of medgemma-4b-it emerged victorious agains all LLMs with <10 billion parameters and rivaled a LLM with 70 billion parameters across various examinations. CONCLUSIONS: For medical AI application and research that require knowledge bases from Spanish-speaking countries and those exhibiting similar epidemiological profiles to Peru's, interested parties should utilize medgemma-27b-text-it or a fine-tuned version of medgemma-4b-it. 

---
# !MSA at AraHealthQA 2025 Shared Task: Enhancing LLM Performance for Arabic Clinical Question Answering through Prompt Engineering and Ensemble Learning 

**Authors**: Mohamed Tarek, Seif Ahmed, Mohamed Basem  

**Link**: [PDF](https://arxiv.org/pdf/2509.11365)  

**Abstract**: We present our systems for Track 2 (General Arabic Health QA, MedArabiQ) of the AraHealthQA-2025 shared task, where our methodology secured 2nd place in both Sub-Task 1 (multiple-choice question answering) and Sub-Task 2 (open-ended question answering) in Arabic clinical contexts. For Sub-Task 1, we leverage the Gemini 2.5 Flash model with few-shot prompting, dataset preprocessing, and an ensemble of three prompt configurations to improve classification accuracy on standard, biased, and fill-in-the-blank questions. For Sub-Task 2, we employ a unified prompt with the same model, incorporating role-playing as an Arabic medical expert, few-shot examples, and post-processing to generate concise responses across fill-in-the-blank, patient-doctor Q&A, GEC, and paraphrased variants. 

---
# ClaimIQ at CheckThat! 2025: Comparing Prompted and Fine-Tuned Language Models for Verifying Numerical Claims 

**Authors**: Anirban Saha Anik, Md Fahimul Kabir Chowdhury, Andrew Wyckoff, Sagnik Ray Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2509.11492)  

**Abstract**: This paper presents our system for Task 3 of the CLEF 2025 CheckThat! Lab, which focuses on verifying numerical and temporal claims using retrieved evidence. We explore two complementary approaches: zero-shot prompting with instruction-tuned large language models (LLMs) and supervised fine-tuning using parameter-efficient LoRA. To enhance evidence quality, we investigate several selection strategies, including full-document input and top-k sentence filtering using BM25 and MiniLM. Our best-performing model LLaMA fine-tuned with LoRA achieves strong performance on the English validation set. However, a notable drop in the test set highlights a generalization challenge. These findings underscore the importance of evidence granularity and model adaptation for robust numerical fact verification. 

---
# When Smiley Turns Hostile: Interpreting How Emojis Trigger LLMs' Toxicity 

**Authors**: Shiyao Cui, Xijia Feng, Yingkang Wang, Junxiao Yang, Zhexin Zhang, Biplab Sikdar, Hongning Wang, Han Qiu, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11141)  

**Abstract**: Emojis are globally used non-verbal cues in digital communication, and extensive research has examined how large language models (LLMs) understand and utilize emojis across contexts. While usually associated with friendliness or playfulness, it is observed that emojis may trigger toxic content generation in LLMs. Motivated by such a observation, we aim to investigate: (1) whether emojis can clearly enhance the toxicity generation in LLMs and (2) how to interpret this phenomenon. We begin with a comprehensive exploration of emoji-triggered LLM toxicity generation by automating the construction of prompts with emojis to subtly express toxic intent. Experiments across 5 mainstream languages on 7 famous LLMs along with jailbreak tasks demonstrate that prompts with emojis could easily induce toxicity generation. To understand this phenomenon, we conduct model-level interpretations spanning semantic cognition, sequence generation and tokenization, suggesting that emojis can act as a heterogeneous semantic channel to bypass the safety mechanisms. To pursue deeper insights, we further probe the pre-training corpus and uncover potential correlation between the emoji-related data polution with the toxicity generation behaviors. Supplementary materials provide our implementation code and data. (Warning: This paper contains potentially sensitive contents) 

---
# Joint Effects of Argumentation Theory, Audio Modality and Data Enrichment on LLM-Based Fallacy Classification 

**Authors**: Hongxu Zhou, Hylke Westerdijk, Khondoker Ittehadul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2509.11127)  

**Abstract**: This study investigates how context and emotional tone metadata influence large language model (LLM) reasoning and performance in fallacy classification tasks, particularly within political debate settings. Using data from U.S. presidential debates, we classify six fallacy types through various prompting strategies applied to the Qwen-3 (8B) model. We introduce two theoretically grounded Chain-of-Thought frameworks: Pragma-Dialectics and the Periodic Table of Arguments, and evaluate their effectiveness against a baseline prompt under three input settings: text-only, text with context, and text with both context and audio-based emotional tone metadata. Results suggest that while theoretical prompting can improve interpretability and, in some cases, accuracy, the addition of context and especially emotional tone metadata often leads to lowered performance. Emotional tone metadata biases the model toward labeling statements as \textit{Appeal to Emotion}, worsening logical reasoning. Overall, basic prompts often outperformed enhanced ones, suggesting that attention dilution from added inputs may worsen rather than improve fallacy classification in LLMs. 

---
# Quantifier Scope Interpretation in Language Learners and LLMs 

**Authors**: Shaohua Fang, Yue Li, Yan Cong  

**Link**: [PDF](https://arxiv.org/pdf/2509.10860)  

**Abstract**: Sentences with multiple quantifiers often lead to interpretive ambiguities, which can vary across languages. This study adopts a cross-linguistic approach to examine how large language models (LLMs) handle quantifier scope interpretation in English and Chinese, using probabilities to assess interpretive likelihood. Human similarity (HS) scores were used to quantify the extent to which LLMs emulate human performance across language groups. Results reveal that most LLMs prefer the surface scope interpretations, aligning with human tendencies, while only some differentiate between English and Chinese in the inverse scope preferences, reflecting human-similar patterns. HS scores highlight variability in LLMs' approximation of human behavior, but their overall potential to align with humans is notable. Differences in model architecture, scale, and particularly models' pre-training data language background, significantly influence how closely LLMs approximate human quantifier scope interpretations. 

---
# Introducing Spotlight: A Novel Approach for Generating Captivating Key Information from Documents 

**Authors**: Ankan Mullick, Sombit Bose, Rounak Saha, Ayan Kumar Bhowmick, Aditya Vempaty, Prasenjit Dey, Ravi Kokku, Pawan Goyal, Niloy Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2509.10935)  

**Abstract**: In this paper, we introduce Spotlight, a novel paradigm for information extraction that produces concise, engaging narratives by highlighting the most compelling aspects of a document. Unlike traditional summaries, which prioritize comprehensive coverage, spotlights selectively emphasize intriguing content to foster deeper reader engagement with the source material. We formally differentiate spotlights from related constructs and support our analysis with a detailed benchmarking study using new datasets curated for this work. To generate high-quality spotlights, we propose a two-stage approach: fine-tuning a large language model on our benchmark data, followed by alignment via Direct Preference Optimization (DPO). Our comprehensive evaluation demonstrates that the resulting model not only identifies key elements with precision but also enhances readability and boosts the engagement value of the original document. 

---
# Evaluating Large Language Models for Evidence-Based Clinical Question Answering 

**Authors**: Can Wang, Yiqun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.10843)  

**Abstract**: Large Language Models (LLMs) have demonstrated substantial progress in biomedical and clinical applications, motivating rigorous evaluation of their ability to answer nuanced, evidence-based questions. We curate a multi-source benchmark drawing from Cochrane systematic reviews and clinical guidelines, including structured recommendations from the American Heart Association and narrative guidance used by insurers. Using GPT-4o-mini and GPT-5, we observe consistent performance patterns across sources and clinical domains: accuracy is highest on structured guideline recommendations (90%) and lower on narrative guideline and systematic review questions (60--70%). We also find a strong correlation between accuracy and the citation count of the underlying systematic reviews, where each doubling of citations is associated with roughly a 30% increase in the odds of a correct answer. Models show moderate ability to reason about evidence quality when contextual information is supplied. When we incorporate retrieval-augmented prompting, providing the gold-source abstract raises accuracy on previously incorrect items to 0.79; providing top 3 PubMed abstracts (ranked by semantic relevance) improves accuracy to 0.23, while random abstracts reduce accuracy (0.10, within temperature variation). These effects are mirrored in GPT-4o-mini, underscoring that source clarity and targeted retrieval -- not just model size -- drive performance. Overall, our results highlight both the promise and current limitations of LLMs for evidence-based clinical question answering. Retrieval-augmented prompting emerges as a useful strategy to improve factual accuracy and alignment with source evidence, while stratified evaluation by specialty and question type remains essential to understand current knowledge access and to contextualize model performance. 

---
# RECAP: Transparent Inference-Time Emotion Alignment for Medical Dialogue Systems 

**Authors**: Adarsh Srinivasan, Jacob Dineen, Muhammad Umar Afzal, Muhammad Uzair Sarfraz, Irbaz B. Riaz, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.10746)  

**Abstract**: Large language models in healthcare often miss critical emotional cues, delivering medically sound but emotionally flat advice. This is especially problematic in clinical contexts where patients are distressed and vulnerable, and require empathic communication to support safety, adherence, and trust. We present RECAP (Reflect-Extract-Calibrate-Align-Produce), an inference-time framework that adds structured emotional reasoning without retraining. By decomposing empathy into transparent appraisal-theoretic stages and exposing per-dimension Likert signals, RECAP produces nuanced, auditable responses. Across EmoBench, SECEU, and EQ-Bench, RECAP improves emotional reasoning by 22-28% on 8B models and 10-13% on larger models over zero-shot baselines. Clinician evaluations further confirm superior empathetic communication. RECAP shows that modular, theory-grounded prompting can systematically enhance emotional intelligence in medical AI while preserving the accountability required for deployment. 

---
# Reasoning Under Uncertainty: Exploring Probabilistic Reasoning Capabilities of LLMs 

**Authors**: Mobina Pournemat, Keivan Rezaei, Gaurang Sriramanan, Arman Zarei, Jiaxiang Fu, Yang Wang, Hamid Eghbalzadeh, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2509.10739)  

**Abstract**: Despite widespread success in language understanding and generation, large language models (LLMs) exhibit unclear and often inconsistent behavior when faced with tasks that require probabilistic reasoning. In this work, we present the first comprehensive study of the reasoning capabilities of LLMs over explicit discrete probability distributions. Given observations from a probability distribution, we evaluate models on three carefully designed tasks, mode identification, maximum likelihood estimation, and sample generation, by prompting them to provide responses to queries about either the joint distribution or its conditionals. These tasks thus probe a range of probabilistic skills, including frequency analysis, marginalization, and generative behavior. Through comprehensive empirical evaluations, we demonstrate that there exists a clear performance gap between smaller and larger models, with the latter demonstrating stronger inference and surprising capabilities in sample generation. Furthermore, our investigations reveal notable limitations, including sensitivity to variations in the notation utilized to represent probabilistic outcomes and performance degradation of over 60% as context length increases. Together, our results provide a detailed understanding of the probabilistic reasoning abilities of LLMs and identify key directions for future improvement. 

---
# A Survey on Retrieval And Structuring Augmented Generation with Large Language Models 

**Authors**: Pengcheng Jiang, Siru Ouyang, Yizhu Jiao, Ming Zhong, Runchu Tian, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.10697)  

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing with their remarkable capabilities in text generation and reasoning. However, these models face critical challenges when deployed in real-world applications, including hallucination generation, outdated knowledge, and limited domain expertise. Retrieval And Structuring (RAS) Augmented Generation addresses these limitations by integrating dynamic information retrieval with structured knowledge representations. This survey (1) examines retrieval mechanisms including sparse, dense, and hybrid approaches for accessing external knowledge; (2) explore text structuring techniques such as taxonomy construction, hierarchical classification, and information extraction that transform unstructured text into organized representations; and (3) investigate how these structured representations integrate with LLMs through prompt-based methods, reasoning frameworks, and knowledge embedding techniques. It also identifies technical challenges in retrieval efficiency, structure quality, and knowledge integration, while highlighting research opportunities in multimodal retrieval, cross-lingual structures, and interactive systems. This comprehensive overview provides researchers and practitioners with insights into RAS methods, applications, and future directions. 

---
# Context Copying Modulation: The Role of Entropy Neurons in Managing Parametric and Contextual Knowledge Conflicts 

**Authors**: Zineddine Tighidet, Andrea Mogini, Hedi Ben-younes, Jiali Mei, Patrick Gallinari, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2509.10663)  

**Abstract**: The behavior of Large Language Models (LLMs) when facing contextual information that conflicts with their internal parametric knowledge is inconsistent, with no generally accepted explanation for the expected outcome distribution. Recent work has identified in autoregressive transformer models a class of neurons -- called entropy neurons -- that produce a significant effect on the model output entropy while having an overall moderate impact on the ranking of the predicted tokens. In this paper, we investigate the preliminary claim that these neurons are involved in inhibiting context copying behavior in transformers by looking at their role in resolving conflicts between contextual and parametric information. We show that entropy neurons are responsible for suppressing context copying across a range of LLMs, and that ablating them leads to a significant change in the generation process. These results enhance our understanding of the internal dynamics of LLMs when handling conflicting information. 

---
# GAPrune: Gradient-Alignment Pruning for Domain-Aware Embeddings 

**Authors**: Yixuan Tang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10844)  

**Abstract**: Domain-specific embedding models have shown promise for applications that require specialized semantic understanding, such as coding agents and financial retrieval systems, often achieving higher performance gains than general models. However, state-of-the-art embedding models are typically based on LLMs, which contain billions of parameters, making deployment challenging in resource-constrained environments. Model compression through pruning offers a promising solution, but existing pruning methods treat all parameters uniformly, failing to distinguish between general semantic representations and domain-specific patterns, leading to suboptimal pruning decisions. Thus, we propose GAPrune, a pruning framework that addresses this challenge by considering both domain importance and preserving general linguistic foundation. Our method uses Fisher Information to measure importance and general-domain gradient alignment to assess parameter behavior, then combines these signals using our Domain Alignment Importance (DAI) scoring. Lower DAI scores indicate that the parameter is either less important for the domain task or creates conflicts between domain and general objectives. Experiments on two domain benchmarks, FinMTEB and ChemTEB, show that GAPrune maintains performance within 2.5% of dense models in one-shot pruning at 50% sparsity, while outperforming all baselines. With retraining in 100 steps, GAPrune achieves +4.51% improvement on FinMTEB and +1.73% on ChemTEB, demonstrating that our pruning strategy not only preserves but enhances domain-specific capabilities. Our findings demonstrate that principled pruning strategies can achieve model compression and enhanced domain specialization, providing the research community with a new approach for development. 

---
# MillStone: How Open-Minded Are LLMs? 

**Authors**: Harold Triedman, Vitaly Shmatikov  

**Link**: [PDF](https://arxiv.org/pdf/2509.11967)  

**Abstract**: Large language models equipped with Web search, information retrieval tools, and other agentic capabilities are beginning to supplant traditional search engines. As users start to rely on LLMs for information on many topics, including controversial and debatable issues, it is important to understand how the stances and opinions expressed in LLM outputs are influenced by the documents they use as their information sources.
In this paper, we present MillStone, the first benchmark that aims to systematically measure the effect of external arguments on the stances that LLMs take on controversial issues (not all of them political). We apply MillStone to nine leading LLMs and measure how ``open-minded'' they are to arguments supporting opposite sides of these issues, whether different LLMs agree with each other, which arguments LLMs find most persuasive, and whether these arguments are the same for different LLMs.
In general, we find that LLMs are open-minded on most issues. An authoritative source of information can easily sway an LLM's stance, highlighting the importance of source selection and the risk that LLM-based information retrieval and search systems can be manipulated. 

---
# When marine radar target detection meets pretrained large language models 

**Authors**: Qiying Hu, Linping Zhang, Xueqian Wang, Gang Li, Yu Liu, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12110)  

**Abstract**: Deep learning (DL) methods are widely used to extract high-dimensional patterns from the sequence features of radar echo signals. However, conventional DL algorithms face challenges such as redundant feature segments, and constraints from restricted model sizes. To address these issues, we propose a framework that integrates feature preprocessing with large language models (LLMs). Our preprocessing module tokenizes radar sequence features, applies a patch selection algorithm to filter out uninformative segments, and projects the selected patches into embeddings compatible with the feature space of pre-trained LLMs. Leveraging these refined embeddings, we incorporate a pre-trained LLM, fine-tuning only the normalization layers to reduce training burdens while enhancing performance. Experiments on measured datasets demonstrate that the proposed method significantly outperforms the state-of-the-art baselines on supervised learning tests. 

---
# RadarLLM: Adapting Pretrained Large Language Models for Marine Radar Target Detection with Preference-aware Loss 

**Authors**: Qiying Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.12089)  

**Abstract**: Recent advances in pre-trained large language models (LLMs) have demonstrated their capacities to capture universal knowledge, making them promising general-purpose optimization solvers for wireless signal processing. Motivated by these findings, we take the first step towards fine-tuning pre-trained LLMs for the effective analysis of radar signal features in marine target detection tasks. Nevertheless, directly fine-tuning pre-trained LLMs on marine target detection tasks tends to suffer from pronounced overfitting, particularly in challenging low signal-to-clutter ratio (SCR) scenarios. This overfitting primarily stems from the model's tendency to memorize spurious or noisy feature patterns rather than learning discriminative structures that generalize well to unseen data. To address this challenge, we introduce RadarLLM, a novel fine-tuning framework that utilizes an effective preference-aware loss. Unlike conventional training strategies that uniformly optimize all feature tokens, this loss function selectively optimizes different feature patches based on their online evaluated learning values, thus guiding the model to focus on the most generalizable patterns during optimization. We theoretically demonstrate the effectiveness of the evaluated learning values by transforming the problem as selecting useful feature tokens. Extensive experiments on real-world marine radar datasets show that 1) the proposed loss function is much better than the original one, with particularly significant gains in challenging low SCR scenarios and 2) RadarLLM consistently outperforms state-of-the-art baselines across diverse detection scenarios, with particularly notable gains under limited training data conditions. 

---
