# Agentic Systems in Radiology: Design, Applications, Evaluation, and Challenges 

**Authors**: Christian Bluethgen, Dave Van Veen, Daniel Truhn, Jakob Nikolas Kather, Michael Moor, Malgorzata Polacin, Akshay Chaudhari, Thomas Frauenfelder, Curtis P. Langlotz, Michael Krauthammer, Farhad Nooralahzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2510.09404)  

**Abstract**: Building agents, systems that perceive and act upon their environment with a degree of autonomy, has long been a focus of AI research. This pursuit has recently become vastly more practical with the emergence of large language models (LLMs) capable of using natural language to integrate information, follow instructions, and perform forms of "reasoning" and planning across a wide range of tasks. With its multimodal data streams and orchestrated workflows spanning multiple systems, radiology is uniquely suited to benefit from agents that can adapt to context and automate repetitive yet complex tasks. In radiology, LLMs and their multimodal variants have already demonstrated promising performance for individual tasks such as information extraction and report summarization. However, using LLMs in isolation underutilizes their potential to support complex, multi-step workflows where decisions depend on evolving context from multiple information sources. Equipping LLMs with external tools and feedback mechanisms enables them to drive systems that exhibit a spectrum of autonomy, ranging from semi-automated workflows to more adaptive agents capable of managing complex processes. This review examines the design of such LLM-driven agentic systems, highlights key applications, discusses evaluation methods for planning and tool use, and outlines challenges such as error cascades, tool-use efficiency, and health IT integration. 

---
# LiveOIBench: Can Large Language Models Outperform Human Contestants in Informatics Olympiads? 

**Authors**: Kaijian Zou, Aaron Xiong, Yunxiang Zhang, Frederick Zhang, Yueqi Ren, Jirong Yang, Ayoung Lee, Shitanshu Bhushan, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09595)  

**Abstract**: Competitive programming problems increasingly serve as valuable benchmarks to evaluate the coding capabilities of large language models (LLMs) due to their complexity and ease of verification. Yet, current coding benchmarks face limitations such as lack of exceptionally challenging problems, insufficient test case coverage, reliance on online platform APIs that limit accessibility. To address these issues, we introduce LiveOIBench, a comprehensive benchmark featuring 403 expert-curated Olympiad-level competitive programming problems, each with an average of 60 expert-designed test cases. The problems are sourced directly from 72 official Informatics Olympiads in different regions conducted between 2023 and 2025. LiveOIBench distinguishes itself through four key features: (1) meticulously curated high-quality tasks with detailed subtask rubrics and extensive private test cases; (2) direct integration of elite contestant performance data to enable informative comparison against top-performing humans; (3) planned continuous, contamination-free updates from newly released Olympiad problems; and (4) a self-contained evaluation system facilitating offline and easy-to-reproduce assessments. Benchmarking 32 popular general-purpose and reasoning LLMs, we find that GPT-5 achieves a notable 81.76th percentile, a strong result that nonetheless falls short of top human contestant performance, who usually place above 90th. In contrast, among open-weight reasoning models, GPT-OSS-120B achieves only a 60th percentile, underscoring significant capability disparities from frontier closed models. Detailed analyses indicate that robust reasoning models prioritize precise problem analysis over excessive exploration, suggesting future models should emphasize structured analysis and minimize unnecessary exploration. All data, code, and leaderboard results will be made publicly available on our website. 

---
# Localist LLMs -- A Mathematical Framework for Dynamic Locality Control 

**Authors**: Joachim Diederich  

**Link**: [PDF](https://arxiv.org/pdf/2510.09338)  

**Abstract**: We present a novel framework for training large language models with continuously adjustable internal representations that span the full spectrum from localist (interpretable, rule-based) to distributed (generalizable, efficient) encodings. The key innovation is a locality dial, a tunable parameter that dynamically controls the degree of localization during both training and inference without requiring model retraining. This is achieved through group sparsity penalties on attention mechanisms, information-theoretic anchor design, and dynamic rule injection. We provide rigorous mathematical proofs establishing explicit threshold conditions under which attention provably concentrates on semantically relevant blocks, with exponential bounds on attention entropy and pointer fidelity. Specifically, we prove that when group sparsity penalties exceed certain threshold values, the model's attention mechanisms concentrate on semantically relevant blocks, achieving low entropy and high fidelity with negligible error. This framework enables practitioners to continuously interpolate between interpretable and high-performance modes, supporting applications in regulated domains requiring both transparency and capability. 

---
# RegexPSPACE: A Benchmark for Evaluating LLM Reasoning on PSPACE-complete Regex Problems 

**Authors**: Hyundong Jin, Joonghyuk Hahn, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.09227)  

**Abstract**: Large language models (LLMs) show strong performance across natural language processing (NLP), mathematical reasoning, and programming, and recent large reasoning models (LRMs) further emphasize explicit reasoning. Yet their computational limits, particularly spatial complexity constrained by finite context windows, remain poorly understood. While recent works often focus on problems within the NP complexity class, we push the boundary by introducing a novel benchmark grounded in two PSPACE-complete regular expression (regex) problems: equivalence decision (RegexEQ) and minimization (RegexMin). PSPACE-complete problems serve as a more rigorous standard for assessing computational capacity, as their solutions require massive search space exploration. We perform a double-exponential space exploration to construct a labeled dataset of over a million regex instances with a sound filtering process to build the benchmark. We conduct extensive evaluations on 6 LLMs and 5 LRMs of varying scales, revealing common failure patterns such as verbosity and repetition. With its well-defined structure and quantitative evaluation metrics, this work presents the first empirical investigation into the spatial computational limitations of LLMs and LRMs, offering a new framework for evaluating their advanced reasoning capabilities. Our code is available at this https URL . 

---
# Toward Mechanistic Explanation of Deductive Reasoning in Language Models 

**Authors**: Davide Maltoni, Matteo Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2510.09340)  

**Abstract**: Recent large language models have demonstrated relevant capabilities in solving problems that require logical reasoning; however, the corresponding internal mechanisms remain largely unexplored. In this paper, we show that a small language model can solve a deductive reasoning task by learning the underlying rules (rather than operating as a statistical learner). A low-level explanation of its internal representations and computational circuits is then provided. Our findings reveal that induction heads play a central role in the implementation of the rule completion and rule chaining steps involved in the logical inference required by the task. 

---
# Fundamentals of Building Autonomous LLM Agents 

**Authors**: Victor de Lamo Castrillo, Habtom Kahsay Gidey, Alexander Lenz, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2510.09244)  

**Abstract**: This paper reviews the architecture and implementation methods of agents powered by large language models (LLMs). Motivated by the limitations of traditional LLMs in real-world tasks, the research aims to explore patterns to develop "agentic" LLMs that can automate complex tasks and bridge the performance gap with human capabilities. Key components include a perception system that converts environmental percepts into meaningful representations; a reasoning system that formulates plans, adapts to feedback, and evaluates actions through different techniques like Chain-of-Thought and Tree-of-Thought; a memory system that retains knowledge through both short-term and long-term mechanisms; and an execution system that translates internal decisions into concrete actions. This paper shows how integrating these systems leads to more capable and generalized software bots that mimic human cognitive processes for autonomous and intelligent behavior. 

---
# PAC Reasoning: Controlling the Performance Loss for Efficient Reasoning 

**Authors**: Hao Zeng, Jianguo Huang, Bingyi Jing, Hongxin Wei, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2510.09133)  

**Abstract**: Large reasoning models (LRMs) have achieved remarkable progress in complex problem-solving tasks. Despite this success, LRMs typically suffer from high computational costs during deployment, highlighting a need for efficient inference. A popular direction of efficiency improvement is to switch the LRM between thinking and nonthinking modes dynamically. However, such approaches often introduce additional reasoning errors and lack statistical guarantees for the performance loss, which are critical for high-stakes applications. In this work, we propose Probably Approximately Correct (PAC) reasoning that controls the performance loss under the user-specified performance loss tolerance. In particular, we construct an upper confidence bound on the performance loss, formulated as a monotone function of the uncertainty score, and subsequently determine a threshold for switching to the nonthinking model. Theoretically, using the threshold to switch between the thinking and nonthinking modes ensures bounded performance loss in a distribution-free manner. Our comprehensive experiments on reasoning benchmarks show that the proposed method can save computational budgets and control the user-specified performance loss. 

---
# Dr. Bias: Social Disparities in AI-Powered Medical Guidance 

**Authors**: Emma Kondrup, Anne Imouza  

**Link**: [PDF](https://arxiv.org/pdf/2510.09162)  

**Abstract**: With the rapid progress of Large Language Models (LLMs), the general public now has easy and affordable access to applications capable of answering most health-related questions in a personalized manner. These LLMs are increasingly proving to be competitive, and now even surpass professionals in some medical capabilities. They hold particular promise in low-resource settings, considering they provide the possibility of widely accessible, quasi-free healthcare support. However, evaluations that fuel these motivations highly lack insights into the social nature of healthcare, oblivious to health disparities between social groups and to how bias may translate into LLM-generated medical advice and impact users. We provide an exploratory analysis of LLM answers to a series of medical questions spanning key clinical domains, where we simulate these questions being asked by several patient profiles that vary in sex, age range, and ethnicity. By comparing natural language features of the generated responses, we show that, when LLMs are used for medical advice generation, they generate responses that systematically differ between social groups. In particular, Indigenous and intersex patients receive advice that is less readable and more complex. We observe these trends amplify when intersectional groups are considered. Considering the increasing trust individuals place in these models, we argue for higher AI literacy and for the urgent need for investigation and mitigation by AI developers to ensure these systemic differences are diminished and do not translate to unjust patient support. Our code is publicly available on GitHub. 

---
# RefGrader: Automated Grading of Mathematical Competition Proofs using Agentic Workflows 

**Authors**: Hamed Mahdavi, Pouria Mahdavinia, Samira Malek, Pegah Mohammadipour, Alireza Hashemi, Majid Daliri, Alireza Farhadi, Amir Khasahmadi, Niloofar Mireshghallah, Vasant Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2510.09021)  

**Abstract**: State-of-the-art (SOTA) LLMs have progressed from struggling on proof-based Olympiad problems to solving most of the IMO 2025 problems, with leading systems reportedly handling 5 of 6 problems. Given this progress, we assess how well these models can grade proofs: detecting errors, judging their severity, and assigning fair scores beyond binary correctness. We study proof-analysis capabilities using a corpus of 90 Gemini 2.5 Pro-generated solutions that we grade on a 1-4 scale with detailed error annotations, and on MathArena solution sets for IMO/USAMO 2025 scored on a 0-7 scale. Our analysis shows that models can reliably flag incorrect (including subtly incorrect) solutions but exhibit calibration gaps in how partial credit is assigned. To address this, we introduce agentic workflows that extract and analyze reference solutions and automatically derive problem-specific rubrics for a multi-step grading process. We instantiate and compare different design choices for the grading workflows, and evaluate their trade-offs. Across our annotated corpus and MathArena, our proposed workflows achieve higher agreement with human grades and more consistent handling of partial credit across metrics. We release all code, data, and prompts/logs to facilitate future research. 

---
# TripScore: Benchmarking and rewarding real-world travel planning with fine-grained evaluation 

**Authors**: Yincen Qu, Huan Xiao, Feng Li, Hui Zhou, Xiangying Dai  

**Link**: [PDF](https://arxiv.org/pdf/2510.09011)  

**Abstract**: Travel planning is a valuable yet complex task that poses significant challenges even for advanced large language models (LLMs). While recent benchmarks have advanced in evaluating LLMs' planning capabilities, they often fall short in evaluating feasibility, reliability, and engagement of travel plans. We introduce a comprehensive benchmark for travel planning that unifies fine-grained criteria into a single reward, enabling direct comparison of plan quality and seamless integration with reinforcement learning (RL). Our evaluator achieves moderate agreement with travel-expert annotations (60.75\%) and outperforms multiple LLM-as-judge baselines. We further release a large-scale dataset of 4,870 queries including 219 real-world, free-form requests for generalization to authentic user intent. Using this benchmark, we conduct extensive experiments across diverse methods and LLMs, including test-time computation, neuro-symbolic approaches, supervised fine-tuning, and RL via GRPO. Across base models, RL generally improves itinerary feasibility over prompt-only and supervised baselines, yielding higher unified reward scores. 

---
# GraphMERT: Efficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data 

**Authors**: Margarita Belova, Jiaxin Xiao, Shikhar Tuli, Niraj K. Jha  

**Link**: [PDF](https://arxiv.org/pdf/2510.09580)  

**Abstract**: Researchers have pursued neurosymbolic artificial intelligence (AI) applications for nearly three decades because symbolic components provide abstraction while neural components provide generalization. Thus, a marriage of the two components can lead to rapid advancements in AI. Yet, the field has not realized this promise since most neurosymbolic AI frameworks fail to scale. In addition, the implicit representations and approximate reasoning of neural approaches limit interpretability and trust. Knowledge graphs (KGs), a gold-standard representation of explicit semantic knowledge, can address the symbolic side. However, automatically deriving reliable KGs from text corpora has remained an open problem. We address these challenges by introducing GraphMERT, a tiny graphical encoder-only model that distills high-quality KGs from unstructured text corpora and its own internal representations. GraphMERT and its equivalent KG form a modular neurosymbolic stack: neural learning of abstractions; symbolic KGs for verifiable reasoning. GraphMERT + KG is the first efficient and scalable neurosymbolic model to achieve state-of-the-art benchmark accuracy along with superior symbolic representations relative to baselines.
Concretely, we target reliable domain-specific KGs that are both (1) factual (with provenance) and (2) valid (ontology-consistent relations with domain-appropriate semantics). When a large language model (LLM), e.g., Qwen3-32B, generates domain-specific KGs, it falls short on reliability due to prompt sensitivity, shallow domain expertise, and hallucinated relations. On text obtained from PubMed papers on diabetes, our 80M-parameter GraphMERT yields a KG with a 69.8% FActScore; a 32B-parameter baseline LLM yields a KG that achieves only 40.2% FActScore. The GraphMERT KG also attains a higher ValidityScore of 68.8%, versus 43.0% for the LLM baseline. 

---
# Repairing Regex Vulnerabilities via Localization-Guided Instructions 

**Authors**: Sicheol Sung, Joonghyuk Hahn, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.09037)  

**Abstract**: Regular expressions (regexes) are foundational to modern computing for critical tasks like input validation and data parsing, yet their ubiquity exposes systems to regular expression denial of service (ReDoS), a vulnerability requiring automated repair methods. Current approaches, however, are hampered by a trade-off. Symbolic, rule-based system are precise but fails to repair unseen or complex vulnerability patterns. Conversely, large language models (LLMs) possess the necessary generalizability but are unreliable for tasks demanding strict syntactic and semantic correctness. We resolve this impasse by introducing a hybrid framework, localized regex repair (LRR), designed to harness LLM generalization while enforcing reliability. Our core insight is to decouple problem identification from the repair process. First, a deterministic, symbolic module localizes the precise vulnerable subpattern, creating a constrained and tractable problem space. Then, the LLM invoked to generate a semantically equivalent fix for this isolated segment. This combined architecture successfully resolves complex repair cases intractable for rule-based repair while avoiding the semantic errors of LLM-only approaches. Our work provides a validated methodology for solving such problems in automated repair, improving the repair rate by 15.4%p over the state-of-the-art. Our code is available at this https URL. 

---
# Humanoid Artificial Consciousness Designed with Large Language Model Based on Psychoanalysis and Personality Theory 

**Authors**: Sang Hun Kim, Jongmin Lee, Dongkyu Park, So Young Lee, Yosep Chong  

**Link**: [PDF](https://arxiv.org/pdf/2510.09043)  

**Abstract**: Human consciousness is still a concept hard to define with current scientific understanding. Although Large Language Models (LLMs) have recently demonstrated significant advancements across various domains including translation and summarization, human consciousness is not something to imitate with current upfront technology owing to so-called hallucination. This study, therefore, proposes a novel approach to address these challenges by integrating psychoanalysis and the Myers-Briggs Type Indicator (MBTI) into constructing consciousness and personality modules. We developed three artificial consciousnesses (self-awareness, unconsciousness, and preconsciousness) based on the principles of psychoanalysis. Additionally, we designed 16 characters with different personalities representing the sixteen MBTI types, with several attributes such as needs, status, and memories. To determine if our model's artificial consciousness exhibits human-like cognition, we created ten distinct situations considering seven attributes such as emotional understanding and logical thinking. The decision-making process of artificial consciousness and the final action were evaluated in three ways: survey evaluation, three-tier classification via ChatGPT, and qualitative review. Both quantitative and qualitative analyses indicated a high likelihood of well-simulated consciousness, although the difference in response between different characters and consciousnesses was not very significant. This implies that the developed models incorporating elements of psychoanalysis and personality theory can lead to building a more intuitive and adaptable AI system with humanoid consciousness. Therefore, this study contributes to opening up new avenues for improving AI interactions in complex cognitive contexts. 

---
# Semantic-Condition Tuning: Fusing Graph Context with Large Language Models for Knowledge Graph Completion 

**Authors**: Ruitong Liu, Yan Wen, Te Sun, Yunjia Wu, Pingyang Huang, Zihang Yu, Siyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08966)  

**Abstract**: Fusing Knowledge Graphs with Large Language Models is crucial for knowledge-intensive tasks like knowledge graph completion. The prevailing paradigm, prefix-tuning, simply concatenates knowledge embeddings with text inputs. However, this shallow fusion overlooks the rich relational semantics within KGs and imposes a significant implicit reasoning burden on the LLM to correlate the prefix with the text. To address these, we propose Semantic-condition Tuning (SCT), a new knowledge injection paradigm comprising two key modules. First, a Semantic Graph Module employs a Graph Neural Network to extract a context-aware semantic condition from the local graph neighborhood, guided by knowledge-enhanced relations. Subsequently, this condition is passed to a Condition-Adaptive Fusion Module, which, in turn, adaptively modulates the textual embedding via two parameterized projectors, enabling a deep, feature-wise, and knowledge-aware interaction. The resulting pre-fused embedding is then fed into the LLM for fine-tuning. Extensive experiments on knowledge graph benchmarks demonstrate that SCT significantly outperforms prefix-tuning and other strong baselines. Our analysis confirms that by modulating the input representation with semantic graph context before LLM inference, SCT provides a more direct and potent signal, enabling more accurate and robust knowledge reasoning. 

---
# MEC$^3$O: Multi-Expert Consensus for Code Time Complexity Prediction 

**Authors**: Joonghyuk Hahn, Soohan Lim, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.09049)  

**Abstract**: Predicting the complexity of source code is essential for software development and algorithm analysis. Recently, Baik et al. (2025) introduced CodeComplex for code time complexity prediction. The paper shows that LLMs without fine-tuning struggle with certain complexity classes. This suggests that no single LLM excels at every class, but rather each model shows advantages in certain classes. We propose MEC$^3$O, a multi-expert consensus system, which extends the multi-agent debate frameworks. MEC$^3$O assigns LLMs to complexity classes based on their performance and provides them with class-specialized instructions, turning them into experts. These experts engage in structured debates, and their predictions are integrated through a weighted consensus mechanism. Our expertise assignments to LLMs effectively handle Degeneration-of-Thought, reducing reliance on a separate judge model, and preventing convergence to incorrect majority opinions. Experiments on CodeComplex show that MEC$^3$O outperforms the open-source baselines, achieving at least 10% higher accuracy and macro-F1 scores. It also surpasses GPT-4o-mini in macro-F1 scores on average and demonstrates competitive on-par F1 scores to GPT-4o and GPT-o4-mini on average. This demonstrates the effectiveness of multi-expert debates and weight consensus strategy to generate the final predictions. Our code and data is available at this https URL. 

---
# Tiny-R1V: Lightweight Multimodal Unified Reasoning Model via Model Merging 

**Authors**: Qixiang Yin, Huanjin Yao, Jianghao Chen, Jiaxing Huang, Zhicheng Zhao, Fei Su  

**Link**: [PDF](https://arxiv.org/pdf/2510.08987)  

**Abstract**: Although Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across diverse tasks, they encounter numerous challenges in terms of reasoning efficiency, such as large model size, overthinking, and compromised accuracy in lightweight scenarios. However, research on the reasoning capabilities of lightweight MLLMs is quite lacking. To this end, we propose Tiny-R1V, a novel lightweight 3B model that achieves faster inference and higher accuracy via a two-stage optimization, while unifying multimodal reasoning across multiple tasks and using fewer tokens. In the first stage, Tiny-R1V introduces Length-Informed Relative Policy Optimization (LIPO), a novel reinforcement learning method, to train each reasoning model. The LIPO is designed to dynamically adjusts advantages of responses within groups, that is, by prioritizing concise yet high-quality responses to encourage the generation of shorter and more accurate response. In the second stage, we propose Adaptive Model Merging (AMM), a training-free model merging method that merges multiple specialist models into a unified architecture. Specifically, AMM adaptively adjusts the weights of task vectors and robustly optimizes the merged vectors via a novel gradient projection regularization loss function, thus mitigating redundant conflicts between them. Extensive evaluations on ten widely-used reasoning benchmarks covering mathematics, structured data (charts, tables, documents), OCR, and general capabilities showcase the superior performance of Tiny-R1V, enabling lightweight models to excel in diverse multimodal reasoning tasks. 

---
# Optimizing delivery for quick commerce factoring qualitative assessment of generated routes 

**Authors**: Milon Bhattacharya, Milan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.08671)  

**Abstract**: Indias e-commerce market is projected to grow rapidly, with last-mile delivery accounting for nearly half of operational expenses. Although vehicle routing problem (VRP) based solvers are widely used for delivery planning, their effectiveness in real-world scenarios is limited due to unstructured addresses, incomplete maps, and computational constraints in distance estimation. This study proposes a framework that employs large language models (LLMs) to critique VRP-generated routes against policy-based criteria, allowing logistics operators to evaluate and prioritise more efficient delivery plans. As a illustration of our approach we generate, annotate and evaluated 400 cases using large language models. Our study found that open-source LLMs identified routing issues with 79% accuracy, while proprietary reasoning models achieved reach upto 86%. The results demonstrate that LLM-based evaluation of VRP-generated routes can be an effective and scalable layer of evaluation which goes beyond beyond conventional distance and time based metrics. This has implications for improving cost efficiency, delivery reliability, and sustainability in last-mile logistics, especially for developing countries like India. 

---
# FATHOMS-RAG: A Framework for the Assessment of Thinking and Observation in Multimodal Systems that use Retrieval Augmented Generation 

**Authors**: Samuel Hildebrand, Curtis Taylor, Sean Oesch, James M Ghawaly Jr, Amir Sadovnik, Ryan Shivers, Brandon Schreiber, Kevin Kurian  

**Link**: [PDF](https://arxiv.org/pdf/2510.08945)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a promising paradigm for improving factual accuracy in large language models (LLMs). We introduce a benchmark designed to evaluate RAG pipelines as a whole, evaluating a pipeline's ability to ingest, retrieve, and reason about several modalities of information, differentiating it from existing benchmarks that focus on particular aspects such as retrieval. We present (1) a small, human-created dataset of 93 questions designed to evaluate a pipeline's ability to ingest textual data, tables, images, and data spread across these modalities in one or more documents; (2) a phrase-level recall metric for correctness; (3) a nearest-neighbor embedding classifier to identify potential pipeline hallucinations; (4) a comparative evaluation of 2 pipelines built with open-source retrieval mechanisms and 4 closed-source foundation models; and (5) a third-party human evaluation of the alignment of our correctness and hallucination metrics. We find that closed-source pipelines significantly outperform open-source pipelines in both correctness and hallucination metrics, with wider performance gaps in questions relying on multimodal and cross-document information. Human evaluation of our metrics showed average agreement of 4.62 for correctness and 4.53 for hallucination detection on a 1-5 Likert scale (5 indicating "strongly agree"). 

---
# Prompting Test-Time Scaling Is A Strong LLM Reasoning Data Augmentation 

**Authors**: Sondos Mahmoud Bsharat, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.09599)  

**Abstract**: Large language models (LLMs) have demonstrated impressive reasoning capabilities when provided with chain-of-thought exemplars, but curating large reasoning datasets remains laborious and resource-intensive. In this work, we introduce Prompting Test-Time Scaling (P-TTS), a simple yet effective inference-time data augmentation strategy for enhancing LLM reasoning through finetuning. Rather than collecting thousands or even millions of examples, P-TTS leverages a small pool of only 90 manually selected reasoning instances and systematically varies exemplar augmentation through principled instruction prompting intensities at test time to synthesize diverse reasoning trajectory contexts. Then we finetune the various sizes of Qwen-2.5 models on P-TTS data. Across a suite of mathematical reasoning AIME2024 & 25, MATH500, and GPQA-Diamond, our P-TTS-7B and 32B models outperform the prior competitive baselines like S1 and S1.1 (1K-shot), achieving absolute accuracy gains of +26.66% and +30.00% on AIME'24 (7B), and +13.34% and +6.67% on AIME'25 (7B); P-TTS-32B yields gains of +23.33% and +16.63% on AIME'24, and +26.63% and +3.33% on AIME'25 (vs. S1 and S1.1, respectively), with comparable or better performance on MATH500 and GPQA-Diamond. We further show that P-TTS enhances zero-shot generalization accuracy on out-of-domain reasoning benchmarks of Gaokao, Kaoyan, OlympiadBench, AMC23, GradeSchoolMath, and Minerva. Our analysis suggests that test-time scaling effectively explores the latent space of reasoning patterns, amplifying LLM problem-solving with minimal annotation overhead, and further unlocking the reasoning potential and capabilities of LLMs. Prompting Test-Time Scaling offers a practical, low-cost way to elicit LLM reasoning in resource-constrained or rapidly evolving domains. 

---
# GTAlign: Game-Theoretic Alignment of LLM Assistants for Mutual Welfare 

**Authors**: Siqi Zhu, David Zhang, Pedro Cisneros-Velarde, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2510.08872)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress in reasoning, yet sometimes produce responses that are suboptimal for users in tasks such as writing, information seeking, or providing practical guidance. Conventional alignment practices typically assume that maximizing model reward also maximizes user welfare, but this assumption frequently fails in practice: models may over-clarify or generate overly verbose reasoning when users prefer concise answers. Such behaviors resemble the prisoner's dilemma, where individually rational choices lead to socially suboptimal outcomes. The fundamental challenge is the lack of a principled decision making mechanism that mutually benefits both the LLM and the user. We propose Game-Theoretic Alignment (GTAlign), an alignment framework that integrates game-theoretic decision making into both reasoning and training. During reasoning, the model explicitly treats user-LLM interaction as a strategic game: it constructs payoff matrices within its reasoning chain to estimate welfare for both itself and the user, and then selects actions that are mutually beneficial. During training, we introduce a mutual welfare reward that reinforces cooperative responses, aligning model behavior with socially efficient outcomes. In addition, we introduce an inference technique that leverages game-theoretic reasoning to dynamically adapt LLM's response when pricing policies of LLM service change. Extensive experiments demonstrate that GTAlign substantially improves reasoning efficiency, answer quality, and mutual welfare compared to baselines across diverse tasks. The code is available at this https URL . 

---
# SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models 

**Authors**: Chengyu Wang, Paria Rashidinejad, DiJia Su, Song Jiang, Sid Wang, Siyan Zhao, Cai Zhou, Shannon Zejiang Shen, Feiyu Chen, Tommi Jaakkola, Yuandong Tian, Bo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09541)  

**Abstract**: Diffusion large language models (dLLMs) are emerging as an efficient alternative to autoregressive models due to their ability to decode multiple tokens in parallel. However, aligning dLLMs with human preferences or task-specific rewards via reinforcement learning (RL) is challenging because their intractable log-likelihood precludes the direct application of standard policy gradient methods. While prior work uses surrogates like the evidence lower bound (ELBO), these one-sided approximations can introduce significant policy gradient bias. To address this, we propose the Sandwiched Policy Gradient (SPG) that leverages both an upper and a lower bound of the true log-likelihood. Experiments show that SPG significantly outperforms baselines based on ELBO or one-step estimation. Specifically, SPG improves the accuracy over state-of-the-art RL methods for dLLMs by 3.6% in GSM8K, 2.6% in MATH500, 18.4% in Countdown and 27.0% in Sudoku. 

---
# Robust Heuristic Algorithm Design with LLMs 

**Authors**: Pantea Karimi, Dany Rouhana, Pooria Namyar, Siva Kesava Reddy Kakarla, Venkat Arun, Behnaz Arzani  

**Link**: [PDF](https://arxiv.org/pdf/2510.08755)  

**Abstract**: We posit that we can generate more robust and performant heuristics if we augment approaches using LLMs for heuristic design with tools that explain why heuristics underperform and suggestions about how to fix them. We find even simple ideas that (1) expose the LLM to instances where the heuristic underperforms; (2) explain why they occur; and (3) specialize design to regions in the input space, can produce more robust algorithms compared to existing techniques~ -- ~the heuristics we produce have a $\sim28\times$ better worst-case performance compared to FunSearch, improve average performance, and maintain the runtime. 

---
# Mitigating Overthinking through Reasoning Shaping 

**Authors**: Feifan Song, Shaohang Wei, Bofei Gao, Yejie Wang, Wen Luo, Wei Li, Linli Yao, Weimin Xiong, Liang Chen, Tianyu Liu, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09535)  

**Abstract**: Large reasoning models (LRMs) boosted by Reinforcement Learning from Verifier Reward (RLVR) have shown great power in problem solving, yet they often cause overthinking: excessive, meandering reasoning that inflates computational cost. Prior designs of penalization in RLVR manage to reduce token consumption while often harming model performance, which arises from the oversimplicity of token-level supervision. In this paper, we argue that the granularity of supervision plays a crucial role in balancing efficiency and accuracy, and propose Group Relative Segment Penalization (GRSP), a step-level method to regularize reasoning. Since preliminary analyses show that reasoning segments are strongly correlated with token consumption and model performance, we design a length-aware weighting mechanism across segment clusters. Extensive experiments demonstrate that GRSP achieves superior token efficiency without heavily compromising accuracy, especially the advantages with harder problems. Moreover, GRSP stabilizes RL training and scales effectively across model sizes. 

---
# On the Representations of Entities in Auto-regressive Large Language Models 

**Authors**: Victor Morand, Josiane Mothe, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2510.09421)  

**Abstract**: Named entities are fundamental building blocks of knowledge in text, grounding factual information and structuring relationships within language. Despite their importance, it remains unclear how Large Language Models (LLMs) internally represent entities. Prior research has primarily examined explicit relationships, but little is known about entity representations themselves. We introduce entity mention reconstruction as a novel framework for studying how LLMs encode and manipulate entities. We investigate whether entity mentions can be generated from internal representations, how multi-token entities are encoded beyond last-token embeddings, and whether these representations capture relational knowledge. Our proposed method, leveraging _task vectors_, allows to consistently generate multi-token mentions from various entity representations derived from the LLMs hidden states. We thus introduce the _Entity Lens_, extending the _logit-lens_ to predict multi-token mentions. Our results bring new evidence that LLMs develop entity-specific mechanisms to represent and manipulate any multi-token entities, including those unseen during training. Our code is avalable at this https URL . 

---
# ChoirRec: Semantic User Grouping via LLMs for Conversion Rate Prediction of Low-Activity Users 

**Authors**: Dakai Zhai, Jiong Gao, Boya Du, Junwei Xu, Qijie Shen, Jialin Zhu, Yuning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09393)  

**Abstract**: Accurately predicting conversion rates (CVR) for low-activity users remains a fundamental challenge in large-scale e-commerce recommender this http URL approaches face three critical limitations: (i) reliance on noisy and unreliable behavioral signals; (ii) insufficient user-level information due to the lack of diverse interaction data; and (iii) a systemic training bias toward high-activity users that overshadows the needs of low-activity this http URL address these challenges, we propose ChoirRec, a novel framework that leverages the semantic capabilities of Large Language Models (LLMs) to construct semantic user groups and enhance CVR prediction for low-activity this http URL a dual-channel architecture designed for robust cross-user knowledge transfer, ChoirRec comprises three components: (i) a Semantic Group Generation module that utilizes LLMs to form reliable, cross-activity user clusters, thereby filtering out noisy signals; (ii) a Group-aware Hierarchical Representation module that enriches sparse user embeddings with informative group-level priors to mitigate data insufficiency; and (iii) a Group-aware Multi-granularity Modual that employs a dual-channel architecture and adaptive fusion mechanism to ensure effective learning and utilization of group knowledge. We conduct extensive offline and online experiments on Taobao, a leading industrial-scale e-commerce this http URL improves GAUC by 1.16\% in offline evaluations, while online A/B testing reveals a 7.24\% increase in order volume, highlighting its substantial practical value in real-world applications. 

---
# Verifying Chain-of-Thought Reasoning via Its Computational Graph 

**Authors**: Zheng Zhao, Yeskendir Koishekenov, Xianjun Yang, Naila Murray, Nicola Cancedda  

**Link**: [PDF](https://arxiv.org/pdf/2510.09312)  

**Abstract**: Current Chain-of-Thought (CoT) verification methods predict reasoning correctness based on outputs (black-box) or activations (gray-box), but offer limited insight into why a computation fails. We introduce a white-box method: Circuit-based Reasoning Verification (CRV). We hypothesize that attribution graphs of correct CoT steps, viewed as execution traces of the model's latent reasoning circuits, possess distinct structural fingerprints from those of incorrect steps. By training a classifier on structural features of these graphs, we show that these traces contain a powerful signal of reasoning errors. Our white-box approach yields novel scientific insights unattainable by other methods. (1) We demonstrate that structural signatures of error are highly predictive, establishing the viability of verifying reasoning directly via its computational graph. (2) We find these signatures to be highly domain-specific, revealing that failures in different reasoning tasks manifest as distinct computational patterns. (3) We provide evidence that these signatures are not merely correlational; by using our analysis to guide targeted interventions on individual transcoder features, we successfully correct the model's faulty reasoning. Our work shows that, by scrutinizing a model's computational process, we can move from simple error detection to a deeper, causal understanding of LLM reasoning. 

---
# Detecting Data Contamination from Reinforcement Learning Post-training for Large Language Models 

**Authors**: Yongding Tao, Tian Wang, Yihong Dong, Huanyu Liu, Kechi Zhang, Xiaolong Hu, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.09259)  

**Abstract**: Data contamination poses a significant threat to the reliable evaluation of Large Language Models (LLMs). This issue arises when benchmark samples may inadvertently appear in training sets, compromising the validity of reported performance. While detection methods have been developed for the pre-training and Supervised Fine-Tuning stages, a critical research gap exists for the increasingly significant phase of Reinforcement Learning (RL) post-training. As RL post-training becomes pivotal for advancing LLM reasoning, the absence of specialized contamination detection methods in this paradigm presents a critical vulnerability. To address this, we conduct the first systematic study of data detection within RL post-training scenario and propose Self-Critique. Our method is motivated by a key observation: after RL phase, the output entropy distribution of LLMs tends to collapse into highly specific and sparse modes. Self-Critique probes for the underlying policy collapse, i.e., the model's convergence to a narrow reasoning path, which causes this entropy reduction. To facilitate this research, we also introduce RL-MIA, a benchmark constructed to simulate this specific contamination scenario. Extensive experiments show that Self-Critique significantly outperforms baseline methods across multiple models and contamination tasks, achieving an AUC improvement of up to 30%. Whereas existing methods are close to a random guess for RL-phase contamination, our method makes detection possible. 

---
# DICE: Structured Reasoning in LLMs through SLM-Guided Chain-of-Thought Correction 

**Authors**: Yiqi Li, Yusheng Liao, Zhe Chen, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09211)  

**Abstract**: When performing reasoning tasks with user-specific requirements, such as strict output formats, large language models (LLMs) often prioritize reasoning over adherence to detailed instructions. Fine-tuning LLMs on supervised datasets to address this is impractical due to high computational costs and limited parameter access. To tackle this, we propose DICE, a lightweight framework that guides small language models (SLMs) to refine LLMs' outputs through chain-of-thought (CoT) correction. DICE decouples the process by first prompting LLMs to generate natural language responses, then using trained SLMs to analyze and refine these outputs to meet structured output specifications. This framework preserves LLMs' broad knowledge and reasoning capabilities while ensuring the outputs conform to user demands. Specifically, DICE first constructs structured CoT adaptation datasets via a two-stage method and subsequently applies a dual-tuning strategy to fine-tune SLMs for generating structured outputs in an analyze-then-answer pattern. Experiments demonstrate that DICE improves the average format accuracy and content correctness of LLM outputs by 35.4\% and 29.4\%, respectively, achieving state-of-the-art (SOTA) performance over other competitive baselines. 

---
# FLRC: Fine-grained Low-Rank Compressor for Efficient LLM Inference 

**Authors**: Yu-Chen Lu, Chong-Yan Chen, Chi-Chih Chang, Yu-Fang Hu, Kai-Chiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09332)  

**Abstract**: Although large language models (LLM) have achieved remarkable performance, their enormous parameter counts hinder deployment on resource-constrained hardware. Low-rank compression can reduce both memory usage and computational demand, but applying a uniform compression ratio across all layers often leads to significant performance degradation, and previous methods perform poorly during decoding. To address these issues, we propose the Fine-grained Low-Rank Compressor (FLRC), which efficiently determines an optimal rank allocation for each layer, and incorporates progressive low-rank decoding to maintain text generation quality. Comprehensive experiments on diverse benchmarks demonstrate the superiority of FLRC, achieving up to a 17% improvement in ROUGE-L on summarization tasks compared to state-of-the-art low-rank compression methods, establishing a more robust and efficient framework to improve LLM inference. 

---
# Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs 

**Authors**: Yumin Choi, Dongki Kim, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09201)  

**Abstract**: Large Language Models (LLMs) have shown remarkable success, and their multimodal expansions (MLLMs) further unlock capabilities spanning images, videos, and other modalities beyond text. However, despite this shift, prompt optimization approaches, designed to reduce the burden of manual prompt crafting while maximizing performance, remain confined to text, ultimately limiting the full potential of MLLMs. Motivated by this gap, we introduce the new problem of multimodal prompt optimization, which expands the prior definition of prompt optimization to the multimodal space defined by the pairs of textual and non-textual prompts. To tackle this problem, we then propose the Multimodal Prompt Optimizer (MPO), a unified framework that not only performs the joint optimization of multimodal prompts through alignment-preserving updates but also guides the selection process of candidate prompts by leveraging earlier evaluations as priors in a Bayesian-based selection strategy. Through extensive experiments across diverse modalities that go beyond text, such as images, videos, and even molecules, we demonstrate that MPO outperforms leading text-only optimization methods, establishing multimodal prompt optimization as a crucial step to realizing the potential of MLLMs. 

---
# CLARity: Reasoning Consistency Alone Can Teach Reinforced Experts 

**Authors**: Jiuheng Lin, Cong Jiang, Zirui Wu, Jiarui Sun, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09278)  

**Abstract**: Training expert LLMs in domains with scarce data is difficult, often relying on multiple-choice questions (MCQs). However, standard outcome-based reinforcement learning (RL) on MCQs is risky. While it may improve accuracy, we observe it often degrades reasoning quality such as logical consistency. Existing solutions to supervise reasoning, such as large-scale Process Reward Models (PRMs), are prohibitively expensive. To address this, we propose CLARity, a cost-effective RL framework that enhances reasoning quality using only a small, general-purpose LLM. CLARity integrates a consistency-aware reward mechanism with a 2-stage refine-then-monitor training pipeline to enhance reasoning consistency, and a dynamic data reformulation strategy to to better exploit limited data. Experiments demonstrate that CLARity improves response consistency by 16.5% and accuracy by 7.5% over baselines. Human evaluations further confirm holistic improvements in coherence and professionalism. Thus, CLARity offers a generalizable solution that enables smaller models to effectively guide expert models by reasoning this http URL code is open sourced at: this https URL 

---
# Cost-Efficient Long Code Translation using LLMs while Leveraging Identifier Replacements 

**Authors**: Manojit Chakraborty, Madhusudan Ghosh, Rishabh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.09045)  

**Abstract**: In the domain of software development, LLMs have been utilized to automate tasks such as code translation, where source code from one programming language is translated to another while preserving its functionality. However, LLMs often struggle with long source codes that don't fit into the context window, which produces inaccurate translations. To address this, we propose a novel zero-shot code translation method that incorporates identifier replacement. By substituting user-given long identifiers with generalized placeholders during translation, our method allows the LLM to focus on the logical structure of the code, by reducing token count and memory usage, which improves the efficiency and cost-effectiveness of long code translation. Our empirical results demonstrate that our approach preserves syntactical and hierarchical information and produces translation results with reduced tokens. 

---
# SEER: Sustainability Enhanced Engineering of Software Requirements 

**Authors**: Mandira Roy, Novarun Deb, Nabendu Chaki, Agostino Cortesi  

**Link**: [PDF](https://arxiv.org/pdf/2510.08981)  

**Abstract**: The rapid expansion of software development has significant environmental, technical, social, and economic impacts. Achieving the United Nations Sustainable Development Goals by 2030 compels developers to adopt sustainable practices. Existing methods mostly offer high-level guidelines, which are time-consuming to implement and rely on team adaptability. Moreover, they focus on design or implementation, while sustainability assessment should start at the requirements engineering phase. In this paper, we introduce SEER, a framework which addresses sustainability concerns in the early software development phase. The framework operates in three stages: (i) it identifies sustainability requirements (SRs) relevant to a specific software product from a general taxonomy; (ii) it evaluates how sustainable system requirements are based on the identified SRs; and (iii) it optimizes system requirements that fail to satisfy any SR. The framework is implemented using the reasoning capabilities of large language models and the agentic RAG (Retrieval Augmented Generation) approach. SEER has been experimented on four software projects from different domains. Results generated using Gemini 2.5 reasoning model demonstrate the effectiveness of the proposed approach in accurately identifying a broad range of sustainability concerns across diverse domains. 

---
# SHERLOCK: Towards Dynamic Knowledge Adaptation in LLM-enhanced E-commerce Risk Management 

**Authors**: Nan Lu, Yurong Hu, Jiaquan Fang, Yan Liu, Rui Dong, Yiming Wang, Rui Lin, Shaoyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08948)  

**Abstract**: The growth of the e-commerce industry has intensified the adversarial dynamics between shadow economy actors and risk management teams. Companies often conduct risk investigations into suspicious cases to identify emerging fraud patterns, thereby enhancing both preemptive risk prevention and post-hoc governance. However, the sheer volume of case analyses imposes a substantial workload on risk management analysts, as each case requires the integration of long-term expert experience and meticulous scrutiny across multiple risk dimensions. Additionally, individual disparities among analysts hinder the establishment of uniform and high-standard workflows. To address these challenges, we propose the SHERLOCK framework, which leverages the reasoning capabilities of large language models (LLMs) to assist analysts in risk investigations. Our approach consists of three primary components: (1) extracting risk management knowledge from multi-modal data and constructing a domain knowledge base (KB), (2) building an intelligent platform guided by the data flywheel paradigm that integrates daily operations, expert annotations, and model evaluations, with iteratively fine-tuning for preference alignment, and (3) introducing a Reflect & Refine (R&R) module that collaborates with the domain KB to establish a rapid response mechanism for evolving risk patterns. Experiments conducted on the real-world transaction dataset from this http URL demonstrate that our method significantly improves the precision of both factual alignment and risk localization within the LLM analysis results. Deployment of the SHERLOCK-based LLM system on this http URL has substantially enhanced the efficiency of case investigation workflows for risk managers. 

---
# A Unified Biomedical Named Entity Recognition Framework with Large Language Models 

**Authors**: Tengxiao Lv, Ling Luo, Juntao Li, Yanhua Wang, Yuchen Pan, Chao Liu, Yanan Wang, Yan Jiang, Huiyi Lv, Yuanyuan Sun, Jian Wang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.08902)  

**Abstract**: Accurate recognition of biomedical named entities is critical for medical information extraction and knowledge discovery. However, existing methods often struggle with nested entities, entity boundary ambiguity, and cross-lingual generalization. In this paper, we propose a unified Biomedical Named Entity Recognition (BioNER) framework based on Large Language Models (LLMs). We first reformulate BioNER as a text generation task and design a symbolic tagging strategy to jointly handle both flat and nested entities with explicit boundary annotation. To enhance multilingual and multi-task generalization, we perform bilingual joint fine-tuning across multiple Chinese and English datasets. Additionally, we introduce a contrastive learning-based entity selector that filters incorrect or spurious predictions by leveraging boundary-sensitive positive and negative samples. Experimental results on four benchmark datasets and two unseen corpora show that our method achieves state-of-the-art performance and robust zero-shot generalization across languages. The source codes are freely available at this https URL. 

---
# Designing and Evaluating an AI-driven Immersive Multidisciplinary Simulation (AIMS) for Interprofessional Education 

**Authors**: Ruijie Wang, Jie Lu, Bo Pei, Evonne Jones, Jamey Brinson, Timothy Brown  

**Link**: [PDF](https://arxiv.org/pdf/2510.08891)  

**Abstract**: Interprofessional education has long relied on case studies and the use of standardized patients to support teamwork, communication, and related collaborative competencies among healthcare professionals. However, traditional approaches are often limited by cost, scalability, and inability to mimic the dynamic complexity of real-world clinical scenarios. To address these challenges, we designed and developed AIMS (AI-Enhanced Immersive Multidisciplinary Simulations), a virtual simulation that integrates a large language model (Gemini-2.5-Flash), a Unity-based virtual environment engine, and a character creation pipeline to support synchronized, multimodal interactions between the user and the virtual patient. AIMS was designed to enhance collaborative clinical reasoning and health promotion competencies among students from pharmacy, medicine, nursing, and social work. A formal usability testing session was conducted which participants assumed professional roles on a healthcare team and engaged in a mix of scripted and unscripted conversations. Participants explored the patient's symptoms, social context, and care needs. Usability issues were identified (e.g., audio routing, response latency) and used to guide subsequent refinements. Findings in general suggest that AIMS supports realistic, profession-specific and contextually appropriate conversations. We discussed both technical and pedagogical innovations of AIMS and concluded with future directions. 

---
# Pattern Enhanced Multi-Turn Jailbreaking: Exploiting Structural Vulnerabilities in Large Language Models 

**Authors**: Ragib Amin Nihal, Rui Wen, Kazuhiro Nakadai, Jun Sakuma  

**Link**: [PDF](https://arxiv.org/pdf/2510.08859)  

**Abstract**: Large language models (LLMs) remain vulnerable to multi-turn jailbreaking attacks that exploit conversational context to bypass safety constraints gradually. These attacks target different harm categories (like malware generation, harassment, or fraud) through distinct conversational approaches (educational discussions, personal experiences, hypothetical scenarios). Existing multi-turn jailbreaking methods often rely on heuristic or ad hoc exploration strategies, providing limited insight into underlying model weaknesses. The relationship between conversation patterns and model vulnerabilities across harm categories remains poorly understood. We propose Pattern Enhanced Chain of Attack (PE-CoA), a framework of five conversation patterns to construct effective multi-turn jailbreaks through natural dialogue. Evaluating PE-CoA on twelve LLMs spanning ten harm categories, we achieve state-of-the-art performance, uncovering pattern-specific vulnerabilities and LLM behavioral characteristics: models exhibit distinct weakness profiles where robustness to one conversational pattern does not generalize to others, and model families share similar failure modes. These findings highlight limitations of safety training and indicate the need for pattern-aware defenses. Code available on: this https URL 

---
# Repository-Aware File Path Retrieval via Fine-Tuned LLMs 

**Authors**: Vasudha Yanuganti, Ishaan Puri, Swapnil Chhatre, Mantinder Singh, Ashok Jallepalli, Hritvik Shrivastava, Pradeep Kumar Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2510.08850)  

**Abstract**: Modern codebases make it hard for developers and AI coding assistants to find the right source files when answering questions like "How does this feature work?" or "Where was the bug introduced?" Traditional code search (keyword or IR based) often misses semantic context and cross file links, while large language models (LLMs) understand natural language but lack repository specific detail. We present a method for file path retrieval that fine tunes a strong LLM (Qwen3-8B) with QLoRA and Unsloth optimizations to predict relevant file paths directly from a natural language query. To build training data, we introduce six code aware strategies that use abstract syntax tree (AST) structure and repository content to generate realistic question-answer pairs, where answers are sets of file paths. The strategies range from single file prompts to hierarchical repository summaries, providing broad coverage. We fine tune on Python projects including Flask, Click, Jinja, FastAPI, and PyTorch, and obtain high retrieval accuracy: up to 91\% exact match and 93\% recall on held out queries, clearly beating single strategy training. On a large codebase like PyTorch (about 4,000 Python files), the model reaches 59\% recall, showing scalability. We analyze how multi level code signals help the LLM reason over cross file context and discuss dataset design, limits (for example, context length in very large repos), and future integration of retrieval with LLM based code intelligence. 

---
# McMining: Automated Discovery of Misconceptions in Student Code 

**Authors**: Erfan Al-Hossami, Razvan Bunescu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08827)  

**Abstract**: When learning to code, students often develop misconceptions about various programming language concepts. These can not only lead to bugs or inefficient code, but also slow down the learning of related concepts. In this paper, we introduce McMining, the task of mining programming misconceptions from samples of code from a student. To enable the training and evaluation of McMining systems, we develop an extensible benchmark dataset of misconceptions together with a large set of code samples where these misconceptions are manifested. We then introduce two LLM-based McMiner approaches and through extensive evaluations show that models from the Gemini, Claude, and GPT families are effective at discovering misconceptions in student code. 

---
# Benchmarking Chinese Commonsense Reasoning with a Multi-hop Reasoning Perspective 

**Authors**: Wangjie You, Xusheng Wang, Xing Wang, Wenxiang Jiao, Chao Feng, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08800)  

**Abstract**: While Large Language Models (LLMs) have demonstrated advanced reasoning capabilities, their comprehensive evaluation in general Chinese-language contexts remains understudied. To bridge this gap, we propose Chinese Commonsense Multi-hop Reasoning (CCMOR), a novel benchmark designed to evaluate LLMs' ability to integrate Chinese-specific factual knowledge with multi-step logical reasoning. Specifically, we first construct a domain-balanced seed set from existing QA datasets, then develop an LLM-powered pipeline to generate multi-hop questions anchored on factual unit chains. To ensure the quality of resulting dataset, we implement a human-in-the-loop verification system, where domain experts systematically validate and refine the generated questions. Using CCMOR, we evaluate state-of-the-art LLMs, demonstrating persistent limitations in LLMs' ability to process long-tail knowledge and execute knowledge-intensive reasoning. Notably, retrieval-augmented generation substantially mitigates these knowledge gaps, yielding significant performance gains. 

---
# Exploring Multi-Temperature Strategies for Token- and Rollout-Level Control in RLVR 

**Authors**: Haomin Zhuang, Yujun Zhou, Taicheng Guo, Yue Huang, Fangxu Liu, Kai Song, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08892)  

**Abstract**: Reinforcement Learning has demonstrated substantial improvements in the reasoning abilities of Large Language Models (LLMs), exhibiting significant applicability across various domains. Recent research has identified that tokens within LLMs play distinct roles during reasoning tasks, categorizing them into high-entropy reasoning tokens and low-entropy knowledge tokens. Prior approaches have typically focused on restricting updates to indirectly encourage exploration, yet they do not explicitly facilitate exploratory behavior during the token generation stage itself. In this work, we introduce a complementary approach that explicitly promotes exploration during sampling by applying distinct temperature settings for different token types. Specifically, our method employs higher temperatures for reasoning tokens to actively encourage exploration, while retaining lower temperatures for knowledge tokens to maintain factual correctness. Furthermore, we systematically investigate various multi-temperature scheduling strategies and their impacts within reinforcement learning contexts. Empirical evaluations on several reasoning benchmarks demonstrate that our approach significantly enhances the reasoning performance of LLMs. The code is available at this https URL. 

---
# Struc-EMB: The Potential of Structure-Aware Encoding in Language Embeddings 

**Authors**: Shikun Liu, Haoyu Wang, Mufei Li, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08774)  

**Abstract**: Text embeddings from Large Language Models (LLMs) have become foundational for numerous applications. However, these models typically operate on raw text, overlooking the rich structural information, such as hyperlinks or citations, that provides crucial context in many real-world datasets. This paper introduces and systematically evaluates a new paradigm for generating structure-aware text embeddings by integrating these structural relations directly into the LLM's internal encoding process, rather than relying on traditional post-hoc aggregation. We investigate two primary in-process methods: sequential concatenation and parallel caching. Through extensive zero-shot experiments across retrieval, clustering, classification, and recommendation tasks, we demonstrate that our structure-aware approaches consistently outperform both text-only and post-hoc baselines. Our analysis reveals critical trade-offs: sequential concatenation excels with noisy, moderate-length contexts, while parallel caching scales more effectively to long, high-signal contexts but is more susceptible to distractors. To address the challenge of noisy structural data, we also introduce and validate two effective techniques: Context Distillation and Semantic Balancing. This work provides the first comprehensive analysis of in-process structure-aware encoding, offering a blueprint for building more powerful and contextually aware embedding models. 

---
# MLLM as a UI Judge: Benchmarking Multimodal LLMs for Predicting Human Perception of User Interfaces 

**Authors**: Reuben A. Luera, Ryan Rossi, Franck Dernoncourt, Samyadeep Basu, Sungchul Kim, Subhojyoti Mukherjee, Puneet Mathur, Ruiyi Zhang, Jihyung Kil, Nedim Lipka, Seunghyun Yoon, Jiuxiang Gu, Zichao Wang, Cindy Xiong Bearfield, Branislav Kveton  

**Link**: [PDF](https://arxiv.org/pdf/2510.08783)  

**Abstract**: In an ideal design pipeline, user interface (UI) design is intertwined with user research to validate decisions, yet studies are often resource-constrained during early exploration. Recent advances in multimodal large language models (MLLMs) offer a promising opportunity to act as early evaluators, helping designers narrow options before formal testing. Unlike prior work that emphasizes user behavior in narrow domains such as e-commerce with metrics like clicks or conversions, we focus on subjective user evaluations across varied interfaces. We investigate whether MLLMs can mimic human preferences when evaluating individual UIs and comparing them. Using data from a crowdsourcing platform, we benchmark GPT-4o, Claude, and Llama across 30 interfaces and examine alignment with human judgments on multiple UI factors. Our results show that MLLMs approximate human preferences on some dimensions but diverge on others, underscoring both their potential and limitations in supplementing early UX research. 

---
# Coordinates from Context: Using LLMs to Ground Complex Location References 

**Authors**: Tessa Masis, Brendan O'Connor  

**Link**: [PDF](https://arxiv.org/pdf/2510.08741)  

**Abstract**: Geocoding is the task of linking a location reference to an actual geographic location and is essential for many downstream analyses of unstructured text. In this paper, we explore the challenging setting of geocoding compositional location references. Building on recent work demonstrating LLMs' abilities to reason over geospatial data, we evaluate LLMs' geospatial knowledge versus reasoning skills relevant to our task. Based on these insights, we propose an LLM-based strategy for geocoding compositional location references. We show that our approach improves performance for the task and that a relatively small fine-tuned LLM can achieve comparable performance with much larger off-the-shelf models. 

---
# When to Reason: Semantic Router for vLLM 

**Authors**: Chen Wang, Xunzhuo Liu, Yuhan Liu, Yue Zhu, Xiangxi Mo, Junchen Jiang, Huamin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.08731)  

**Abstract**: Large Language Models (LLMs) demonstrate substantial accuracy gains when augmented with reasoning modes such as chain-of-thought and inference-time scaling. However, reasoning also incurs significant costs in inference latency and token usage, with environmental and financial impacts, which are unnecessary for many simple prompts. We present a semantic router that classifies queries based on their reasoning requirements and selectively applies reasoning only when beneficial. Our approach achieves a 10.2 percentage point improvement in accuracy on the MMLU-Pro benchmark while reducing response latency by 47.1% and token consumption by 48.5% compared to direct inference with vLLM. These results demonstrate that semantic routing offers an effective mechanism for striking a balance between accuracy and efficiency in open-source LLM serving systems 

---
# ConPoSe: LLM-Guided Contact Point Selection for Scalable Cooperative Object Pushing 

**Authors**: Noah Steinkrüger, Nisarga Nilavadi, Wolfram Burgard, Tanja Katharina Kaiser  

**Link**: [PDF](https://arxiv.org/pdf/2510.08705)  

**Abstract**: Object transportation in cluttered environments is a fundamental task in various domains, including domestic service and warehouse logistics. In cooperative object transport, multiple robots must coordinate to move objects that are too large for a single robot. One transport strategy is pushing, which only requires simple robots. However, careful selection of robot-object contact points is necessary to push the object along a preplanned path. Although this selection can be solved analytically, the solution space grows combinatorially with the number of robots and object size, limiting scalability. Inspired by how humans rely on common-sense reasoning for cooperative transport, we propose combining the reasoning capabilities of Large Language Models with local search to select suitable contact points. Our LLM-guided local search method for contact point selection, ConPoSe, successfully selects contact points for a variety of shapes, including cuboids, cylinders, and T-shapes. We demonstrate that ConPoSe scales better with the number of robots and object size than the analytical approach, and also outperforms pure LLM-based selection. 

---
# Guiding Exploration in Reinforcement Learning Through LLM-Augmented Observations 

**Authors**: Vaibhav Jain, Gerrit Grossmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.08779)  

**Abstract**: Reinforcement Learning (RL) agents often struggle in sparse-reward environments where traditional exploration strategies fail to discover effective action sequences. Large Language Models (LLMs) possess procedural knowledge and reasoning capabilities from text pretraining that could guide RL exploration, but existing approaches create rigid dependencies where RL policies must follow LLM suggestions or incorporate them directly into reward functions. We propose a framework that provides LLM-generated action recommendations through augmented observation spaces, allowing RL agents to learn when to follow or ignore this guidance. Our method leverages LLMs' world knowledge and reasoning abilities while maintaining flexibility through soft constraints. We evaluate our approach on three BabyAI environments of increasing complexity and show that the benefits of LLM guidance scale with task difficulty. In the most challenging environment, we achieve 71% relative improvement in final success rates over baseline. The approach provides substantial sample efficiency gains, with agents reaching performance thresholds up to 9 times faster, and requires no modifications to existing RL algorithms. Our results demonstrate an effective method for leveraging LLM planning capabilities to accelerate RL training in challenging environments. 

---
# RA-Gen: A Controllable Code Generation Framework Using ReAct for Multi-Agent Task Execution 

**Authors**: Aofan Liu, Haoxuan Li, Bin Wang, Ao Yang, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08665)  

**Abstract**: Code generation models based on large language models (LLMs) have gained wide adoption, but challenges remain in ensuring safety, accuracy, and controllability, especially for complex tasks. Existing methods often lack dynamic integration of external tools, transparent reasoning, and user control over safety. To address these issues, we propose a controllable code generation framework utilizing the ReAct paradigm for multi-agent task execution. This framework is a multi-agent system designed to enable efficient, precise, and interpretable code generation through dynamic interactions between LLMs and external resources. The framework adopts a collaborative architecture comprising four specialized agents: a Planner for task decomposition, a Searcher that leverages the ReAct framework for reasoning and tool integration, a CodeGen agent for accurate code generation, and an Extractor for structured data retrieval. The ReAct-based Searcher alternates between generating reasoning traces and executing actions, facilitating seamless integration of internal knowledge with external tools (such as search engines) to enhance accuracy and user control. Experimental results show the framework's effectiveness across multiple languages, achieving a 94.8% security rate on the SVEN dataset with CodeQL, outperforming existing approaches. Its transparent reasoning process fosters user trust and improves controllability. 

---
# Faver: Boosting LLM-based RTL Generation with Function Abstracted Verifiable Middleware 

**Authors**: Jianan Mu, Mingyu Shi, Yining Wang, Tianmeng Yang, Bin Sun, Xing Hu, Jing Ye, Huawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08664)  

**Abstract**: LLM-based RTL generation is an interesting research direction, as it holds the potential to liberate the least automated stage in the current chip design. However, due to the substantial semantic gap between high-level specifications and RTL, coupled with limited training data, existing models struggle with generation accuracy. Drawing on human experience, design with verification helps improving accuracy. However, as the RTL testbench data are even more scarce, it is not friendly for LLMs. Although LLMs excel at higher-level languages like Python/C, they have a huge semantic gap from RTL. When implementing the same functionality, Python/C code and hardware code differ significantly in the spatiotemporal granularity, requiring the LLM not only to consider high-level functional semantics but also to ensure the low-level details align with the circuit code. It is not an easy task. In this paper, we propose a function abstracted verifiable middleware (Faver) that streamlines RTL verification in LLM-based workflows. By mixing LLM-friendly code structures with a rule-based template, Faver decouples the details of circuit verification, allowing the LLM to focus on the functionality itself. In our experiments on the SFT model and open-source models, Faver improved the model's generation accuracy by up to 14%. 

---
# A Novel Framework for Augmenting Rating Scale Tests with LLM-Scored Text Data 

**Authors**: Joe Watson, Ivan O'Conner, Chia-Wen Chen, Luning Sun, Fang Luo, David Stillwell  

**Link**: [PDF](https://arxiv.org/pdf/2510.08663)  

**Abstract**: Psychological assessments typically rely on structured rating scales, which cannot incorporate the rich nuance of a respondent's natural language. This study leverages recent LLM advances to harness qualitative data within a novel conceptual framework, combining LLM-scored text and traditional rating-scale items to create an augmented test. We demonstrate this approach using depression as a case study, developing and assessing the framework on a real-world sample of upper secondary students (n=693) and corresponding synthetic dataset (n=3,000). On held-out test sets, augmented tests achieved statistically significant improvements in measurement precision and accuracy. The information gain from the LLM items was equivalent to adding between 6.3 (real data) and 16.0 (synthetic data) items to the original 19-item test. Our approach marks a conceptual shift in automated scoring that bypasses its typical bottlenecks: instead of relying on pre-labelled data or complex expert-created rubrics, we empirically select the most informative LLM scoring instructions based on calculations of item information. This framework provides a scalable approach for leveraging the growing stream of transcribed text to enhance traditional psychometric measures, and we discuss its potential utility in clinical health and beyond. 

---
# Upfront Chain-of-Thought: A Cooperative Framework for Chain-of-Thought Compression 

**Authors**: Chengzhengxu Li, Xiaoming Liu, Zhaohan Zhang, Shaochu Zhang, Shengchao Liu, Guoxin Ma, Yu Lan, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.08647)  

**Abstract**: Recent developments have enabled advanced reasoning in Large Language Models (LLMs) via long Chain-of-Thought (CoT), while long CoT suffers from high computational costs and significant latency losses owing to the autoregressive nature of generative LLMs. CoT compression aims to improve efficiency in the reasoning process by reducing output length. Previous works trade reasoning efficiency by either laborious discrete prompt designing or the construction of external compressed CoT datasets that sacrifice key reasoning details. In this work, we propose Upfront CoT (UCoT): an efficient reasoning framework with upfront thought embedding to automate CoT compression. UCoT is a cooperative workflow involving a small model (compressor) and a large model (executor). The first stage of UCoT trains compressor to generate upfront thought embeddings rich in reasoning information for the executor, avoiding the drawbacks of manually designed prompts. The second stage optimizes executor to utilize upfront thought embeddings to derive the correct answer with short reasoning, using a reward mechanism. Extensive experiments show that UCoT maintains the powerful reasoning ability of executor while significantly reducing the length of CoT. It is worth mentioning that when applying UCoT to the Qwen2.5-7B-Instruct model, the usage of tokens on GSM8K dataset is reduced by 50\%, while the performance is 3.08\% higher than that of the state-of-the-art (SOTA) method. The code and dataset are in supplementary material. 

---
# RAG4Tickets: AI-Powered Ticket Resolution via Retrieval-Augmented Generation on JIRA and GitHub Data 

**Authors**: Mohammad Baqar  

**Link**: [PDF](https://arxiv.org/pdf/2510.08667)  

**Abstract**: Modern software teams frequently encounter delays in resolving recurring or related issues due to fragmented knowledge scattered across JIRA tickets, developer discussions, and GitHub pull requests (PRs). To address this challenge, we propose a Retrieval-Augmented Generation (RAG) framework that integrates Sentence-Transformers for semantic embeddings with FAISS-based vector search to deliver context-aware ticket resolution recommendations. The approach embeds historical JIRA tickets, user comments, and linked PR metadata to retrieve semantically similar past cases, which are then synthesized by a Large Language Model (LLM) into grounded and explainable resolution suggestions. The framework contributes a unified pipeline linking JIRA and GitHub data, an embedding and FAISS indexing strategy for heterogeneous software artifacts, and a resolution generation module guided by retrieved evidence. Experimental evaluation using precision, recall, resolution time reduction, and developer acceptance metrics shows that the proposed system significantly improves resolution accuracy, fix quality, and knowledge reuse in modern DevOps environments. 

---
# Automating Android Build Repair: Bridging the Reasoning-Execution Gap in LLM Agents with Domain-Specific Tools 

**Authors**: Ha Min Son, Huan Ren, Xin Liu, Zhe Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.08640)  

**Abstract**: Android is the largest mobile platform, yet automatically building applications remains a practical challenge. While Large Language Models (LLMs) show promise for code repair, their use for fixing Android build errors remains underexplored. To address this gap, we first introduce AndroidBuildBench, a benchmark of 1,019 build failures curated from the commit histories of 43 open-source Android projects. Each problem is paired with a verified solution from a subsequent commit, ensuring that fixes are feasible. Second, we propose GradleFixer, an LLM agent with domain-specific tools for inspecting and manipulating the Gradle build environment. GradleFixer achieves a resolve rate of 81.4% (pass@1), significantly outperforming a state-of-the-art coding agent that relies on a general-purpose shell. GradleFixer's success suggests that while LLMs possess the high-level knowledge to solve these failures, they struggle to translate this knowledge into effective low-level actions using a general-purpose shell. We demonstrate the effectiveness of a strategy we term Tool Bridging, which replaces general-purpose shell commands with domain-aware abstractions. We hypothesize this approach works through two mechanisms: 1) it provides tools in an API-like format that LLMs use more reliably, and 2) it constrains the action space to relevant operations. This approach bridges the gap between the model's high-level reasoning and effective low-level execution. 

---
# Energy-Driven Steering: Reducing False Refusals in Large Language Models 

**Authors**: Eric Hanchen Jiang, Weixuan Ou, Run Liu, Shengyuan Pang, Guancheng Wan, Ranjie Duan, Wei Dong, Kai-Wei Chang, XiaoFeng Wang, Ying Nian Wu, Xinfeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08646)  

**Abstract**: Safety alignment of large language models (LLMs) faces a key challenge: current alignment techniques often only focus on improving safety against harmful prompts, causing LLMs to become over-cautious and refuse to respond to benign prompts. Therefore, a key objective of safe alignment is to enhance safety while simultaneously reducing false refusals. In this paper, we introduce Energy-Driven Steering (EDS), a novel, fine-tuning free framework designed to resolve this challenge through dynamic, inference-time intervention. We trained a lightweight, external Energy-Based Model (EBM) to assign high energy to undesirable (false refusal or jailbreak) states and low energy to desirable (helpful response or safe reject) ones. During inference, EBM maps the LLM's internal activations to an "energy landscape". We use the gradient of the energy function to dynamically steer the LLM's hidden states to low energy regions, correcting the model to generate a desirable response in real-time without modifying its weights. This method decouples behavioral control from the model's core knowledge, offering a flexible solution with minimal computational overhead. Extensive experiments across a wide range of models show our method successfully achieves this objective: it substantially lowers false refusal rates. For example, raising compliance on the ORB-H benchmark from 57.3% to 82.6% while maintaining the baseline safety performance. Our work presents an effective paradigm for building LLMs that achieve both low false refusal rates and high safety. 

---
# Relative Positioning Based Code Chunking Method For Rich Context Retrieval In Repository Level Code Completion Task With Code Language Model 

**Authors**: Imranur Rahman, Md Rayhanur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2510.08610)  

**Abstract**: Code completion can help developers improve efficiency and ease the development lifecycle. Although code completion is available in modern integrated development environments (IDEs), research lacks in determining what makes a good context for code completion based on the information available to the IDEs for the large language models (LLMs) to perform better. In this paper, we describe an effective context collection strategy to assist the LLMs in performing better at code completion tasks. The key idea of our strategy is to preprocess the repository into smaller code chunks and later use syntactic and semantic similarity-based code chunk retrieval with relative positioning. We found that code chunking and relative positioning of the chunks in the final context improve the performance of code completion tasks. 

---
# Toward a Safer Web: Multilingual Multi-Agent LLMs for Mitigating Adversarial Misinformation Attacks 

**Authors**: Nouar Aldahoul, Yasir Zaki  

**Link**: [PDF](https://arxiv.org/pdf/2510.08605)  

**Abstract**: The rapid spread of misinformation on digital platforms threatens public discourse, emotional stability, and decision-making. While prior work has explored various adversarial attacks in misinformation detection, the specific transformations examined in this paper have not been systematically studied. In particular, we investigate language-switching across English, French, Spanish, Arabic, Hindi, and Chinese, followed by translation. We also study query length inflation preceding summarization and structural reformatting into multiple-choice questions. In this paper, we present a multilingual, multi-agent large language model framework with retrieval-augmented generation that can be deployed as a web plugin into online platforms. Our work underscores the importance of AI-driven misinformation detection in safeguarding online factual integrity against diverse attacks, while showcasing the feasibility of plugin-based deployment for real-world web applications. 

---
# Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models 

**Authors**: Shahriar Kabir Nahin, Hadi Askari, Muhao Chen, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2510.08592)  

**Abstract**: Test-Time Scaling (TTS) improves LLM reasoning by exploring multiple candidate responses and then operating over this set to find the best output. A tacit premise behind TTS is that sufficiently diverse candidate pools enhance reliability. In this work, we show that this assumption in TTS introduces a previously unrecognized failure mode. When candidate diversity is curtailed, even by a modest amount, TTS becomes much more likely to produce unsafe outputs. We present a reference-guided diversity reduction protocol (RefDiv) that serves as a diagnostic attack to stress test TTS pipelines. Through extensive experiments across four open-source models (Qwen3, Mistral, Llama3.1, Gemma3) and two widely used TTS strategies (Monte Carlo Tree Search and Best-of-N), constraining diversity consistently signifies the rate at which TTS produces unsafe results. The effect is often stronger than that produced by prompts directly with high adversarial intent scores. This observed phenomenon also transfers across TTS strategies and to closed-source models (e.g. OpenAI o3 and Gemini-2.5-Pro), thus indicating that this is a general and extant property of TTS rather than a model-specific artifact. Additionally, we find that numerous widely used safety guardrail classifiers (e.g. Llama-Guard and OpenAI Moderation API), are unable to flag the adversarial input prompts generated by RefDiv, demonstrating that existing defenses offer limited protection against this diversity-driven failure mode. Through this work, we hope to motivate future research on designing robust TTS strategies that are both effective and secure against diversity-targeted stress tests as illustrated by RefDiv. 

---
# MMA-ASIA: A Multilingual and Multimodal Alignment Framework for Culturally-Grounded Evaluation 

**Authors**: Weihua Zheng, Zhengyuan Liu, Tanmoy Chakraborty, Weiwen Xu, Xiaoxue Gao, Bryan Chen Zhengyu Tan, Bowei Zou, Chang Liu, Yujia Hu, Xing Xie, Xiaoyuan Yi, Jing Yao, Chaojun Wang, Long Li, Rui Liu, Huiyao Liu, Koji Inoue, Ryuichi Sumida, Tatsuya Kawahara, Fan Xu, Lingyu Ye, Wei Tian, Dongjun Kim, Jimin Jung, Jaehyung Seo, Nadya Yuki Wangsajaya, Pham Minh Duc, Ojasva Saxena, Palash Nandi, Xiyan Tao, Wiwik Karlina, Tuan Luong, Keertana Arun Vasan, Roy Ka-Wei Lee, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.08608)  

**Abstract**: Large language models (LLMs) are now used worldwide, yet their multimodal understanding and reasoning often degrade outside Western, high-resource settings. We propose MMA-ASIA, a comprehensive framework to evaluate LLMs' cultural awareness with a focus on Asian contexts. MMA-ASIA centers on a human-curated, multilingual, and multimodally aligned multiple-choice benchmark covering 8 Asian countries and 10 languages, comprising 27,000 questions; over 79 percent require multi-step reasoning grounded in cultural context, moving beyond simple memorization. To our knowledge, this is the first dataset aligned at the input level across three modalities: text, image (visual question answering), and speech. This enables direct tests of cross-modal transfer. Building on this benchmark, we propose a five-dimensional evaluation protocol that measures: (i) cultural-awareness disparities across countries, (ii) cross-lingual consistency, (iii) cross-modal consistency, (iv) cultural knowledge generalization, and (v) grounding validity. To ensure rigorous assessment, a Cultural Awareness Grounding Validation Module detects "shortcut learning" by checking whether the requisite cultural knowledge supports correct answers. Finally, through comparative model analysis, attention tracing, and an innovative Vision-ablated Prefix Replay (VPR) method, we probe why models diverge across languages and modalities, offering actionable insights for building culturally reliable multimodal LLMs. 

---
# Evaluating Hallucinations in Multimodal LLMs with Spoken Queries under Diverse Acoustic Conditions 

**Authors**: Hansol Park, Hoseong Ahn, Junwon Moon, Yejin Lee, Kyuhong Shim  

**Link**: [PDF](https://arxiv.org/pdf/2510.08581)  

**Abstract**: Hallucinations in vision-language models have been extensively studied using benchmarks that probe reliability in image-text settings. In contrast, the effect of spoken queries on multimodal hallucinations remains largely unexplored, despite the growing role of voice-driven interfaces. In this work, we investigate how spoken input influences hallucinations in multimodal large language models. We present RePOPE-Spk, an audio-augmented extension of the RePOPE benchmark, where queries are provided as speech under diverse acoustic conditions. Using RePOPE-Spk, we systematically evaluate both proprietary and open-source models. Experimental results show that hallucinations escalate when queries are spoken rather than written: error rates increase by 3% under clean speech and by up to 20% with environmental noise. Input order and query length further affect robustness, while strategies such as many-shot prompting and chain-of-thought reasoning offer partial but insufficient mitigation. These findings highlight a critical and underexplored challenge, opening new directions for building reliable voice interface systems. 

---
# Mnemosyne: An Unsupervised, Human-Inspired Long-Term Memory Architecture for Edge-Based LLMs 

**Authors**: Aneesh Jonelagadda, Christina Hahn, Haoze Zheng, Salvatore Penachio  

**Link**: [PDF](https://arxiv.org/pdf/2510.08601)  

**Abstract**: Long-term memory is essential for natural, realistic dialogue. However, current large language model (LLM) memory systems rely on either brute-force context expansion or static retrieval pipelines that fail on edge-constrained devices. We introduce Mnemosyne, an unsupervised, human-inspired long-term memory architecture designed for edge-based LLMs. Our approach uses graph-structured storage, modular substance and redundancy filters, memory committing and pruning mechanisms, and probabilistic recall with temporal decay and refresh processes modeled after human memory. Mnemosyne also introduces a concentrated "core summary" efficiently derived from a fixed-length subset of the memory graph to capture the user's personality and other domain-specific long-term details such as, using healthcare application as an example, post-recovery ambitions and attitude towards care. Unlike existing retrieval-augmented methods, Mnemosyne is designed for use in longitudinal healthcare assistants, where repetitive and semantically similar but temporally distinct conversations are limited by naive retrieval. In experiments with longitudinal healthcare dialogues, Mnemosyne demonstrates the highest win rate of 65.8% in blind human evaluations of realism and long-term memory capability compared to a baseline RAG win rate of 31.1%. Mnemosyne also achieves current highest LoCoMo benchmark scores in temporal reasoning and single-hop retrieval compared to other same-backboned techniques. Further, the average overall score of 54.6% was second highest across all methods, beating commonly used Mem0 and OpenAI baselines among others. This demonstrates that improved factual recall, enhanced temporal reasoning, and much more natural user-facing responses can be feasible with an edge-compatible and easily transferable unsupervised memory architecture. 

---
# Comparative Analysis of Large Language Models for the Machine-Assisted Resolution of User Intentions 

**Authors**: Justus Flerlage, Alexander Acker, Odej Kao  

**Link**: [PDF](https://arxiv.org/pdf/2510.08576)  

**Abstract**: Large Language Models (LLMs) have emerged as transformative tools for natural language understanding and user intent resolution, enabling tasks such as translation, summarization, and, increasingly, the orchestration of complex workflows. This development signifies a paradigm shift from conventional, GUI-driven user interfaces toward intuitive, language-first interaction paradigms. Rather than manually navigating applications, users can articulate their objectives in natural language, enabling LLMs to orchestrate actions across multiple applications in a dynamic and contextual manner. However, extant implementations frequently rely on cloud-based proprietary models, which introduce limitations in terms of privacy, autonomy, and scalability. For language-first interaction to become a truly robust and trusted interface paradigm, local deployment is not merely a convenience; it is an imperative. This limitation underscores the importance of evaluating the feasibility of locally deployable, open-source, and open-access LLMs as foundational components for future intent-based operating systems. In this study, we examine the capabilities of several open-source and open-access models in facilitating user intention resolution through machine assistance. A comparative analysis is conducted against OpenAI's proprietary GPT-4-based systems to assess performance in generating workflows for various user intentions. The present study offers empirical insights into the practical viability, performance trade-offs, and potential of open LLMs as autonomous, locally operable components in next-generation operating systems. The results of this study inform the broader discussion on the decentralization and democratization of AI infrastructure and point toward a future where user-device interaction becomes more seamless, adaptive, and privacy-conscious through locally embedded intelligence. 

---
# AgenticAD: A Specialized Multiagent System Framework for Holistic Alzheimer Disease Management 

**Authors**: Adib Bazgir, Amir Habibdoust, Xing Song, Yuwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08578)  

**Abstract**: Alzheimer's disease (AD) presents a complex, multifaceted challenge to patients, caregivers, and the healthcare system, necessitating integrated and dynamic support solutions. While artificial intelligence (AI) offers promising avenues for intervention, current applications are often siloed, addressing singular aspects of the disease such as diagnostics or caregiver support without systemic integration. This paper proposes a novel methodological framework for a comprehensive, multi-agent system (MAS) designed for holistic Alzheimer's disease management. The objective is to detail the architecture of a collaborative ecosystem of specialized AI agents, each engineered to address a distinct challenge in the AD care continuum, from caregiver support and multimodal data analysis to automated research and clinical data interpretation. The proposed framework is composed of eight specialized, interoperable agents. These agents are categorized by function: (1) Caregiver and Patient Support, (2) Data Analysis and Research, and (3) Advanced Multimodal Workflows. The methodology details the technical architecture of each agent, leveraging a suite of advanced technologies including large language models (LLMs) such as GPT-4o and Gemini, multi-agent orchestration frameworks, Retrieval-Augmented Generation (RAG) for evidence-grounded responses, and specialized tools for web scraping, multimodal data processing, and in-memory database querying. This paper presents a detailed architectural blueprint for an integrated AI ecosystem for AD care. By moving beyond single-purpose tools to a collaborative, multi-agent paradigm, this framework establishes a foundation for developing more adaptive, personalized, and proactive solutions. This methodological approach aims to pave the way for future systems capable of synthesizing diverse data streams to improve patient outcomes and reduce caregiver burden. 

---
# Beyond CNNs: Efficient Fine-Tuning of Multi-Modal LLMs for Object Detection on Low-Data Regimes 

**Authors**: Nirmal Elamon, Rouzbeh Davoudi  

**Link**: [PDF](https://arxiv.org/pdf/2510.08589)  

**Abstract**: The field of object detection and understanding is rapidly evolving, driven by advances in both traditional CNN-based models and emerging multi-modal large language models (LLMs). While CNNs like ResNet and YOLO remain highly effective for image-based tasks, recent transformer-based LLMs introduce new capabilities such as dynamic context reasoning, language-guided prompts, and holistic scene understanding. However, when used out-of-the-box, the full potential of LLMs remains underexploited, often resulting in suboptimal performance on specialized visual tasks. In this work, we conduct a comprehensive comparison of fine-tuned traditional CNNs, zero-shot pre-trained multi-modal LLMs, and fine-tuned multi-modal LLMs on the challenging task of artificial text overlay detection in images. A key contribution of our study is demonstrating that LLMs can be effectively fine-tuned on very limited data (fewer than 1,000 images) to achieve up to 36% accuracy improvement, matching or surpassing CNN-based baselines that typically require orders of magnitude more data. By exploring how language-guided models can be adapted for precise visual understanding with minimal supervision, our work contributes to the broader effort of bridging vision and language, offering novel insights into efficient cross-modal learning strategies. These findings highlight the adaptability and data efficiency of LLM-based approaches for real-world object detection tasks and provide actionable guidance for applying multi-modal transformers in low-resource visual environments. To support continued progress in this area, we have made the code used to fine-tune the models available in our GitHub, enabling future improvements and reuse in related applications. 

---
# Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors 

**Authors**: Yihong Liu, Raoyuan Zhao, Lena Altinger, Hinrich Schütze, Michael A. Hedderich  

**Link**: [PDF](https://arxiv.org/pdf/2510.09536)  

**Abstract**: Large language models (LLMs) are increasingly deployed in multilingual, real-world applications with user inputs -- naturally introducing typographical errors (typos). Yet most benchmarks assume clean input, leaving the robustness of LLMs to typos across languages largely underexplored. To address this gap, we introduce MulTypo, a multilingual typo generation algorithm that simulates human-like errors based on language-specific keyboard layouts and typing behavior. We evaluate 18 open-source LLMs across three model families and five downstream tasks spanning language inference, multi-choice question answering, mathematical reasoning, and machine translation tasks. Our results show that typos consistently degrade performance, particularly in generative tasks and those requiring reasoning -- while the natural language inference task is comparatively more robust. Instruction tuning improves clean-input performance but may increase brittleness under noise. We also observe language-dependent robustness: high-resource languages are generally more robust than low-resource ones, and translation from English is more robust than translation into English. Our findings underscore the need for noise-aware training and multilingual robustness evaluation. We make our code and data publicly available. 

---
# Beyond Surface Reasoning: Unveiling the True Long Chain-of-Thought Capacity of Diffusion Large Language Models 

**Authors**: Qiguang Chen, Hanjing Li, Libo Qin, Dengyun Peng, Jinhao Liu, Jiangyi Wang, Chengyue Wu, Xie Chen, Yantao Du, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2510.09544)  

**Abstract**: Recently, Diffusion Large Language Models (DLLMs) have offered high throughput and effective sequential reasoning, making them a competitive alternative to autoregressive LLMs (ALLMs). However, parallel decoding, which enables simultaneous token updates, conflicts with the causal order often required for rigorous reasoning. We first identify this conflict as the core Parallel-Sequential Contradiction (PSC). Behavioral analyses in both simple and complex reasoning tasks show that DLLMs exhibit genuine parallelism only for directly decidable outputs. As task difficulty increases, they revert to autoregressive-like behavior, a limitation exacerbated by autoregressive prompting, which nearly doubles the number of decoding steps with remasking without improving quality. Moreover, PSC restricts DLLMs' self-reflection, reasoning depth, and exploratory breadth. To further characterize PSC, we introduce three scaling dimensions for DLLMs: parallel, diffusion, and sequential. Empirically, while parallel scaling yields consistent improvements, diffusion and sequential scaling are constrained by PSC. Based on these findings, we propose several practical mitigations, parallel-oriented prompting, diffusion early stopping, and parallel scaling, to reduce PSC-induced ineffectiveness and inefficiencies. 

---
# WUGNECTIVES: Novel Entity Inferences of Language Models from Discourse Connectives 

**Authors**: Daniel Brubaker, William Sheffield, Junyi Jessy Li, Kanishka Misra  

**Link**: [PDF](https://arxiv.org/pdf/2510.09556)  

**Abstract**: The role of world knowledge has been particularly crucial to predict the discourse connective that marks the discourse relation between two arguments, with language models (LMs) being generally successful at this task. We flip this premise in our work, and instead study the inverse problem of understanding whether discourse connectives can inform LMs about the world. To this end, we present WUGNECTIVES, a dataset of 8,880 stimuli that evaluates LMs' inferences about novel entities in contexts where connectives link the entities to particular attributes. On investigating 17 different LMs at various scales, and training regimens, we found that tuning an LM to show reasoning behavior yields noteworthy improvements on most connectives. At the same time, there was a large variation in LMs' overall performance across connective type, with all models systematically struggling on connectives that express a concessive meaning. Our findings pave the way for more nuanced investigations into the functional role of language cues as captured by LMs. We release WUGNECTIVES at this https URL. 

---
# A Comprehensive Evaluation of Multilingual Chain-of-Thought Reasoning: Performance, Consistency, and Faithfulness Across Languages 

**Authors**: Raoyuan Zhao, Yihong Liu, Hinrich Schütze, Michael A. Hedderich  

**Link**: [PDF](https://arxiv.org/pdf/2510.09555)  

**Abstract**: Large reasoning models (LRMs) increasingly rely on step-by-step Chain-of-Thought (CoT) reasoning to improve task performance, particularly in high-resource languages such as English. While recent work has examined final-answer accuracy in multilingual settings, the thinking traces themselves, i.e., the intermediate steps that lead to the final answer, remain underexplored. In this paper, we present the first comprehensive study of multilingual CoT reasoning, evaluating three key dimensions: performance, consistency, and faithfulness. We begin by measuring language compliance, answer accuracy, and answer consistency when LRMs are explicitly instructed or prompt-hacked to think in a target language, revealing strong language preferences and divergent performance across languages. Next, we assess crosslingual consistency of thinking traces by interchanging them between languages. We find that the quality and effectiveness of thinking traces vary substantially depending on the prompt language. Finally, we adapt perturbation-based techniques -- i.e., truncation and error injection -- to probe the faithfulness of thinking traces across languages, showing that models rely on traces to varying degrees. We release our code and data to support future research. 

---
# Can We Reliably Rank Model Performance across Domains without Labeled Data? 

**Authors**: Veronica Rammouz, Aaron Gonzalez, Carlos Cruzportillo, Adrian Tan, Nicole Beebe, Anthony Rios  

**Link**: [PDF](https://arxiv.org/pdf/2510.09519)  

**Abstract**: Estimating model performance without labels is an important goal for understanding how NLP models generalize. While prior work has proposed measures based on dataset similarity or predicted correctness, it remains unclear when these estimates produce reliable performance rankings across domains. In this paper, we analyze the factors that affect ranking reliability using a two-step evaluation setup with four base classifiers and several large language models as error predictors. Experiments on the GeoOLID and Amazon Reviews datasets, spanning 15 domains, show that large language model-based error predictors produce stronger and more consistent rank correlations with true accuracy than drift-based or zero-shot baselines. Our analysis reveals two key findings: ranking is more reliable when performance differences across domains are larger, and when the error model's predictions align with the base model's true failure patterns. These results clarify when performance estimation methods can be trusted and provide guidance for their use in cross-domain model evaluation. 

---
# StatEval: A Comprehensive Benchmark for Large Language Models in Statistics 

**Authors**: Yuchen Lu, Run Yang, Yichen Zhang, Shuguang Yu, Runpeng Dai, Ziwei Wang, Jiayi Xiang, Wenxin E, Siran Gao, Xinyao Ruan, Yirui Huang, Chenjing Xi, Haibo Hu, Yueming Fu, Qinglan Yu, Xiaobing Wei, Jiani Gu, Rui Sun, Jiaxuan Jia, Fan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.09517)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable advances in mathematical and logical reasoning, yet statistics, as a distinct and integrative discipline, remains underexplored in benchmarking efforts. To address this gap, we introduce \textbf{StatEval}, the first comprehensive benchmark dedicated to statistics, spanning both breadth and depth across difficulty levels. StatEval consists of 13,817 foundational problems covering undergraduate and graduate curricula, together with 2374 research-level proof tasks extracted from leading journals. To construct the benchmark, we design a scalable multi-agent pipeline with human-in-the-loop validation that automates large-scale problem extraction, rewriting, and quality control, while ensuring academic rigor. We further propose a robust evaluation framework tailored to both computational and proof-based tasks, enabling fine-grained assessment of reasoning ability. Experimental results reveal that while closed-source models such as GPT5-mini achieve below 57\% on research-level problems, with open-source models performing significantly lower. These findings highlight the unique challenges of statistical reasoning and the limitations of current LLMs. We expect StatEval to serve as a rigorous benchmark for advancing statistical intelligence in large language models. All data and code are available on our web platform: this https URL. 

---
# Hierarchical Indexing with Knowledge Enrichment for Multilingual Video Corpus Retrieval 

**Authors**: Yu Wang, Tianhao Tan, Yifei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09553)  

**Abstract**: Retrieving relevant instructional videos from multilingual medical archives is crucial for answering complex, multi-hop questions across language boundaries. However, existing systems either compress hour-long videos into coarse embeddings or incur prohibitive costs for fine-grained matching. We tackle the Multilingual Video Corpus Retrieval (mVCR) task in the NLPCC-2025 M4IVQA challenge with a multi-stage framework that integrates multilingual semantics, domain terminology, and efficient long-form processing. Video subtitles are divided into semantically coherent chunks, enriched with concise knowledge-graph (KG) facts, and organized into a hierarchical tree whose node embeddings are generated by a language-agnostic multilingual encoder. At query time, the same encoder embeds the input question; a coarse-to-fine tree search prunes irrelevant branches, and only the top-ranked chunks are re-scored by a lightweight large language model (LLM). This design avoids exhaustive cross-encoder scoring while preserving chunk-level precision. Experiments on the mVCR test set demonstrate state-of-the-art performance, and ablation studies confirm the complementary contributions of KG enrichment, hierarchical indexing, and targeted LLM re-ranking. The proposed method offers an accurate and scalable solution for multilingual retrieval in specialized medical video collections. 

---
# Active Model Selection for Large Language Models 

**Authors**: Yavuz Durmazkeser, Patrik Okanovic, Andreas Kirsch, Torsten Hoefler, Nezihe Merve Gürel  

**Link**: [PDF](https://arxiv.org/pdf/2510.09418)  

**Abstract**: We introduce LLM SELECTOR, the first framework for active model selection of Large Language Models (LLMs). Unlike prior evaluation and benchmarking approaches that rely on fully annotated datasets, LLM SELECTOR efficiently identifies the best LLM with limited annotations. In particular, for any given task, LLM SELECTOR adaptively selects a small set of queries to annotate that are most informative about the best model for the task. To further reduce annotation cost, we leverage a judge-based oracle annotation model. Through extensive experiments on 6 benchmarks with 151 LLMs, we show that LLM SELECTOR reduces annotation costs by up to 59.62% when selecting the best and near-best LLM for the task. 

---
# KORMo: Korean Open Reasoning Model for Everyone 

**Authors**: Minjun Kim, Hyeonseok Lim, Hangyeol Yoo, Inho Won, Seungwoo Song, Minkyung Cho, Junhun Yuk, Changsu Choi, Dongjae Shin, Huige Lee, Hoyun Song, Alice Oh, Kyungtae Lim  

**Link**: [PDF](https://arxiv.org/pdf/2510.09426)  

**Abstract**: This work presents the first large-scale investigation into constructing a fully open bilingual large language model (LLM) for a non-English language, specifically Korean, trained predominantly on synthetic data. We introduce KORMo-10B, a 10.8B-parameter model trained from scratch on a Korean-English corpus in which 68.74% of the Korean portion is synthetic. Through systematic experimentation, we demonstrate that synthetic data, when carefully curated with balanced linguistic coverage and diverse instruction styles, does not cause instability or degradation during large-scale pretraining. Furthermore, the model achieves performance comparable to that of contemporary open-weight multilingual baselines across a wide range of reasoning, knowledge, and instruction-following benchmarks. Our experiments reveal two key findings: (1) synthetic data can reliably sustain long-horizon pretraining without model collapse, and (2) bilingual instruction tuning enables near-native reasoning and discourse coherence in Korean. By fully releasing all components including data, code, training recipes, and logs, this work establishes a transparent framework for developing synthetic data-driven fully open models (FOMs) in low-resource settings and sets a reproducible precedent for future multilingual LLM research. 

---
# Understanding the Effects of Domain Finetuning on LLMs 

**Authors**: Eshaan Tanwar, Deepak Nathani, William Yang Wang, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2510.09359)  

**Abstract**: Large Language Models (LLMs) fine-tuned for specific domains exhibit strong performance; however, the underlying mechanisms by which this fine-tuning reshapes their parametric space are not well understood. Prior works primarily focus on auto-regressive or general-purpose instruct models, leaving domain-specialised LLMs under-explored. We present the first systematic study of domain-specific fine-tuning in large medical language models. Our analysis reveals that fine-tuning modifies only a small subset of the representational subspace, essentially preserving the pre-trained model's representation. To interpret these changes in subspaces, we propose tuning vectors, a novel framework inspired by task vectors, which explicitly capture the directional parameter shifts induced by fine-tuning. We demonstrate that these vectors are critical for enhancing both instruction-following and generation quality. Furthermore, combining tuning vectors across different domains yields improved generalisation. Upon closer inspection of directional alignment, we find these vectors primarily write new directional information into the MLP layers of the model, while amplifying existing directions in attention heads. Our findings offer new insights into LLM adaptation and provide a general, interpretable framework for analysing specialisation in large language models. 

---
# Logit Arithmetic Elicits Long Reasoning Capabilities Without Training 

**Authors**: Yunxiang Zhang, Muhammad Khalifa, Lechen Zhang, Xin Liu, Ayoung Lee, Xinliang Frederick Zhang, Farima Fatahi Bayat, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09354)  

**Abstract**: Large reasoning models exhibit long chain-of-thought reasoning with strategies such as backtracking and self-correction, though recent studies suggest that these abilities typically require additional training. We first investigate whether such behaviors can be elicited without any training. To this end, we propose a decoding-time approach, ThinkLogit, which utilizes logit arithmetic to tune a target large non-reasoning model for long reasoning using a substantially smaller reasoning model as the guider. We then show that we can further boost its performance by training the guider model with preference optimization over correct/incorrect reasoning pairs sampled from both the target and guider model, a setup we refer to as ThinkLogit-DPO. Our experiments demonstrate that ThinkLogit and ThinkLogit-DPO achieve a relative improvement in average accuracy by 24.5% and 29.1%, respectively, over five reasoning benchmarks using the Qwen2.5-32B guided by R1-Distill-Qwen-1.5B, a model 21x smaller. Moreover, we find that ThinkLogit remains effective when the guider and target come from different model families. It is also orthogonal to post-training methods for small models, as guiders improved through supervised distillation or reinforcement learning can be directly plugged in to yield stronger large models, offering a practical path to unlock long reasoning in large-scale models without costly post-training. 

---
# ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering 

**Authors**: Francesco Maria Molfese, Luca Moroni, Ciro Porcaro, Simone Conia, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2510.09351)  

**Abstract**: While Small Language Models (SLMs) have demonstrated promising performance on an increasingly wide array of commonsense reasoning benchmarks, current evaluation practices rely almost exclusively on the accuracy of their final answers, neglecting the validity of the reasoning processes that lead to those answers. To address this issue, we introduce ReTraceQA, a novel benchmark that introduces process-level evaluation for commonsense reasoning tasks. Our expert-annotated dataset reveals that in a substantial portion of instances (14-24%), SLMs provide correct final answers despite flawed reasoning processes, suggesting that the capabilities of SLMs are often overestimated by evaluation metrics that focus only on comparing the final answer with the ground truth. Indeed, we show that when employing strong Large Language Models (LLMs) as automated judges for reasoning-aware evaluation rather than answer-only metrics, SLM performance drops significantly across all models and datasets, with scores decreasing by up to 25%. 

---
# LLP: LLM-based Product Pricing in E-commerce 

**Authors**: Hairu Wang, Sheng You, Qiheng Zhang, Xike Xie, Shuguang Han, Yuchen Wu, Fei Huang, Jufeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.09347)  

**Abstract**: Unlike Business-to-Consumer e-commerce platforms (e.g., Amazon), inexperienced individual sellers on Consumer-to-Consumer platforms (e.g., eBay) often face significant challenges in setting prices for their second-hand products efficiently. Therefore, numerous studies have been proposed for automating price prediction. However, most of them are based on static regression models, which suffer from poor generalization performance and fail to capture market dynamics (e.g., the price of a used iPhone decreases over time). Inspired by recent breakthroughs in Large Language Models (LLMs), we introduce LLP, the first LLM-based generative framework for second-hand product pricing. LLP first retrieves similar products to better align with the dynamic market change. Afterwards, it leverages the LLMs' nuanced understanding of key pricing information in free-form text to generate accurate price suggestions. To strengthen the LLMs' domain reasoning over retrieved products, we apply a two-stage optimization, supervised fine-tuning (SFT) followed by group relative policy optimization (GRPO), on a dataset built via bidirectional reasoning. Moreover, LLP employs a confidence-based filtering mechanism to reject unreliable price suggestions. Extensive experiments demonstrate that LLP substantially surpasses existing methods while generalizing well to unseen categories. We have successfully deployed LLP on Xianyu\footnote\{Xianyu is China's largest second-hand e-commerce platform.\}, significantly outperforming the previous pricing method. Under the same 30\% product coverage, it raises the static adoption rate (SAR) from 40\% to 72\%, and maintains a strong SAR of 47\% even at 90\% recall. 

---
# Domain-Adapted Pre-trained Language Models for Implicit Information Extraction in Crash Narratives 

**Authors**: Xixi Wang, Jordanka Kovaceva, Miguel Costa, Shuai Wang, Francisco Camara Pereira, Robert Thomson  

**Link**: [PDF](https://arxiv.org/pdf/2510.09434)  

**Abstract**: Free-text crash narratives recorded in real-world crash databases have been shown to play a significant role in improving traffic safety. However, large-scale analyses remain difficult to implement as there are no documented tools that can batch process the unstructured, non standardized text content written by various authors with diverse experience and attention to detail. In recent years, Transformer-based pre-trained language models (PLMs), such as Bidirectional Encoder Representations from Transformers (BERT) and large language models (LLMs), have demonstrated strong capabilities across various natural language processing tasks. These models can extract explicit facts from crash narratives, but their performance declines on inference-heavy tasks in, for example, Crash Type identification, which can involve nearly 100 categories. Moreover, relying on closed LLMs through external APIs raises privacy concerns for sensitive crash data. Additionally, these black-box tools often underperform due to limited domain knowledge. Motivated by these challenges, we study whether compact open-source PLMs can support reasoning-intensive extraction from crash narratives. We target two challenging objectives: 1) identifying the Manner of Collision for a crash, and 2) Crash Type for each vehicle involved in the crash event from real-world crash narratives. To bridge domain gaps, we apply fine-tuning techniques to inject task-specific knowledge to LLMs with Low-Rank Adaption (LoRA) and BERT. Experiments on the authoritative real-world dataset Crash Investigation Sampling System (CISS) demonstrate that our fine-tuned compact models outperform strong closed LLMs, such as GPT-4o, while requiring only minimal training resources. Further analysis reveals that the fine-tuned PLMs can capture richer narrative details and even correct some mislabeled annotations in the dataset. 

---
# Hybrid Models for Natural Language Reasoning: The Case of Syllogistic Logic 

**Authors**: Manuel Vargas Guzmán, Jakub Szymanik, Maciej Malicki  

**Link**: [PDF](https://arxiv.org/pdf/2510.09472)  

**Abstract**: Despite the remarkable progress in neural models, their ability to generalize, a cornerstone for applications like logical reasoning, remains a critical challenge. We delineate two fundamental aspects of this ability: compositionality, the capacity to abstract atomic logical rules underlying complex inferences, and recursiveness, the aptitude to build intricate representations through iterative application of inference rules. In the literature, these two aspects are often confounded together under the umbrella term of generalization. To sharpen this distinction, we investigated the logical generalization capabilities of pre-trained large language models (LLMs) using the syllogistic fragment as a benchmark for natural language reasoning. Though simple, this fragment provides a foundational yet expressive subset of formal logic that supports controlled evaluation of essential reasoning abilities. Our findings reveal a significant disparity: while LLMs demonstrate reasonable proficiency in recursiveness, they struggle with compositionality. To overcome these limitations and establish a reliable logical prover, we propose a hybrid architecture integrating symbolic reasoning with neural computation. This synergistic interaction enables robust and efficient inference, neural components accelerate processing, while symbolic reasoning ensures completeness. Our experiments show that high efficiency is preserved even with relatively small neural components. As part of our proposed methodology, this analysis gives a rationale and highlights the potential of hybrid models to effectively address key generalization barriers in neural reasoning systems. 

---
# ShiZhi: A Chinese Lightweight Large Language Model for Court View Generation 

**Authors**: Zhitian Hou, Kun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09297)  

**Abstract**: Criminal Court View Generation (CVG) is a fundamental task in legal artificial intelligence, aiming to automatically generate the "Court View" section of a legal case document. Generating court views is challenging due to the diversity and complexity of case facts, and directly generating from raw facts may limit performance. In this paper, we present ShiZhi, the first large language model (LLM) specifically designed for court view generation. We construct a Chinese Court View Generation dataset, CCVG, of more than 110K cases, each containing fact descriptions paired with corresponding court views. Based on this dataset, ShiZhi achieving 58.5 BLEU-1 on court view generation and 86.1\% accuracy with 92.5\% macro F1 on charge prediction. Experimental results demonstrate that even a small LLM can generate reasonable and legally coherent court views when trained on high-quality domain-specific data. Our model and dataset are available at \href{this https URL}{this https URL}. 

---
# MaP: A Unified Framework for Reliable Evaluation of Pre-training Dynamics 

**Authors**: Jiapeng Wang, Changxin Tian, Kunlong Chen, Ziqi Liu, Jiaxin Mao, Wayne Xin Zhao, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.09295)  

**Abstract**: Reliable evaluation is fundamental to the progress of Large Language Models (LLMs), yet the evaluation process during pre-training is plagued by significant instability that obscures true learning dynamics. In this work, we systematically diagnose this instability, attributing it to two distinct sources: \textit{Parameter Instability} from training stochasticity and \textit{Evaluation Instability} from noisy measurement protocols. To counteract both sources of noise, we introduce \textbf{MaP}, a dual-pronged framework that synergistically integrates checkpoint \underline{M}erging \underline{a}nd the \underline{P}ass@k metric. Checkpoint merging smooths the parameter space by averaging recent model weights, while Pass@k provides a robust, low-variance statistical estimate of model capability. Extensive experiments show that MaP yields significantly smoother performance curves, reduces inter-run variance, and ensures more consistent model rankings. Ultimately, MaP provides a more reliable and faithful lens for observing LLM training dynamics, laying a crucial empirical foundation for LLM research. 

---
# NL2GenSym: Natural Language to Generative Symbolic Rules for SOAR Cognitive Architecture via Large Language Models 

**Authors**: Fang Yuan, Junjie Zeng, Yue Hu, Zhengqiu Zhu, Quanjun Yin, Yuxiang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.09355)  

**Abstract**: SOAR, a classic symbol-based cognitive architecture, has been fostering the development of general, human-like intelligent agents. Nevertheless, its practical adoption is hindered by the laborious manual rule coding. Emerging Large Language Models (LLMs) present the immense potential for efficient rules generation. However, there is a critical gap that current research predominantly focuses on conceptual frameworks and lacks robust experimental validation. To bridge this gap, we propose \textit{N}atural \textit{L}anguage to \textit{Gen}erative \textit{Sym}bolic Rules (NL2GenSym), a novel framework that integrates LLMs with SOAR to autonomously produce generative symbolic rules from natural language. Specifically, our framework introduces a novel Execution-Grounded Generator-Critic mechanism. The LLM-based Generator, guided by a Retrieval-Augmented Generation-accessed self-evolving domain knowledge base, proposes rules from natural language. Subsequently, these rules are immediately executed within the SOAR environment to rigorously validate their correctness. Based on this execution-grounded feedback, a reflective LLM-based Critic drives the iterative refinement of these rules. Experiments on our specialized Water Jug Problem (WJP) dataset, utilizing both Gemini and Qwen series models, validate the efficacy of our framework. It achieves a success rate over 86\% in generating rules from natural language. Crucially, the framework also generates novel heuristic rules, reducing average decision cycles for solving the WJP to 1.98 times the optimal solution and 1/1000 of baseline methods. Additionally, our initial experiments show that NL2GenSym enables smaller-parameter models to achieve better performance than larger counterparts. 

---
# Token-Level Policy Optimization: Linking Group-Level Rewards to Token-Level Aggregation via Markov Likelihood 

**Authors**: Xingyu Lin, Yilin Wen, En Wang, Du Su, Wenbin Liu, Chenfu Bao, Zhonghou Lv  

**Link**: [PDF](https://arxiv.org/pdf/2510.09369)  

**Abstract**: Group Relative Policy Optimization (GRPO) has significantly advanced the reasoning ability of large language models (LLMs), particularly by boosting their mathematical performance. However, GRPO and related entropy-regularization methods still face challenges rooted in the sparse token rewards inherent to chain-of-thought (CoT). Current approaches often rely on undifferentiated token-level entropy adjustments, which frequently lead to entropy collapse or model collapse. In this work, we propose TEPO, a novel token-level framework that incorporates Markov Likelihood (sequence likelihood) links group-level rewards with tokens via token-level aggregation. Experiments show that TEPO consistently outperforms existing baselines across key metrics (including @k and accuracy). It not only sets a new state of the art on mathematical reasoning tasks but also significantly enhances training stability. 

---
# Mask Tokens as Prophet: Fine-Grained Cache Eviction for Efficient dLLM Inference 

**Authors**: Jianuo Huang, Yaojie Zhang, Yicun Yang, Benhao Huang, Biqing Qi, Dongrui Liu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09309)  

**Abstract**: Diffusion large language models (dLLMs) present a promising alternative to dominant autoregressive models (ARMs) by the ability of parallel decoding at the expense of substantial computation and memory costs. Specifically, the cache mechanism for bidirectional attention in dLLMs demands large memory footprint, restricting their ability to handle long contexts under resource-limited settings. Existing cache eviction strategies are designed for ARMs and ignore the unique characteristics of dLLMs, thus leading to unsatisfactory performance. To address these challenges, we introduce MaskKV, a training-free cache eviction framework tailored to dLLMs, focusing on the effect of mask tokens in dLLMs. MaskKV is built on two key innovations: (1) a mask-query guided scoring mechanism that leverages attention weights to identify and evict less critical prompt tokens for each head; (2) an adaptive cache budgeting strategy that improves efficiency by reducing allocation in intermediate layers and concentrating resources on prompt-preferring heads. On LLaDA with MaskKV, compressing the KV cache to only 256 pairs (less than 5% of tokens) retains 94% of the full-cache performance on LongBench and achieves up to 31x acceleration at 32k prompt length. The code is publicly available at: this https URL 

---
# Augmenting Dialog with Think-Aloud Utterances for Modeling Individual Personality Traits by LLM 

**Authors**: Seiya Ishikura, Hiroaki Yamada, Tatsuya Hiraoka, Hiroaki Yamada, Takenobu Tokunaga  

**Link**: [PDF](https://arxiv.org/pdf/2510.09158)  

**Abstract**: This study proposes augmenting dialog data with think-aloud utterances (TAUs) for modeling individual personalities in text chat by LLM. TAU is a verbalization of a speaker's thought before articulating the utterance. We expect "persona LLMs" trained with TAU-augmented data can mimic the speaker's personality trait better. We tested whether the trained persona LLMs obtain the human personality with respect to Big Five, a framework characterizing human personality traits from five aspects. The results showed that LLMs trained with TAU-augmented data more closely align to the speakers' Agreeableness and Neuroticism of Big Five than those trained with original dialog data. We also found that the quality of TAU-augmentation impacts persona LLM's performance. 

---
# Large Language Models Do NOT Really Know What They Don't Know 

**Authors**: Chi Seng Cheang, Hou Pong Chan, Wenxuan Zhang, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09033)  

**Abstract**: Recent work suggests that large language models (LLMs) encode factuality signals in their internal representations, such as hidden states, attention weights, or token probabilities, implying that LLMs may "know what they don't know". However, LLMs can also produce factual errors by relying on shortcuts or spurious associations. These error are driven by the same training objective that encourage correct predictions, raising the question of whether internal computations can reliably distinguish between factual and hallucinated outputs. In this work, we conduct a mechanistic analysis of how LLMs internally process factual queries by comparing two types of hallucinations based on their reliance on subject information. We find that when hallucinations are associated with subject knowledge, LLMs employ the same internal recall process as for correct responses, leading to overlapping and indistinguishable hidden-state geometries. In contrast, hallucinations detached from subject knowledge produce distinct, clustered representations that make them detectable. These findings reveal a fundamental limitation: LLMs do not encode truthfulness in their internal states but only patterns of knowledge recall, demonstrating that "LLMs don't really know what they don't know". 

---
# ReFIne: A Framework for Trustworthy Large Reasoning Models with Reliability, Faithfulness, and Interpretability 

**Authors**: Chung-En Sun, Ge Yan, Akshay Kulkarni, Tsui-Wei Weng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09062)  

**Abstract**: Recent advances in long chain-of-thought (CoT) reasoning have largely prioritized answer accuracy and token efficiency, while overlooking aspects critical to trustworthiness. We argue that usable reasoning systems must be trustworthy, characterized by three properties: interpretability, faithfulness, and reliability. To this end, we propose ReFIne, a new training framework that integrates supervised fine-tuning with GRPO to encourage models to: (i) improve interpretability by producing structured, tag-based traces with high-level planning that are easier for humans to follow; (ii) enhance faithfulness by explicitly disclosing the decisive information guiding each solution, with consistent cross-section references; and (iii) promote reliability by providing self-assessments of both the derivation's soundness and the confidence of the final answer. We apply ReFIne to the Qwen3 models at multiple scales (1.7B/4B/8B) and evaluate across mathematical benchmarks of varying difficulty. Our experimental results show that ReFIne models generate clearer and better-structured reasoning traces (interpretability +44.0%), more faithfully expose their underlying decision process (faithfulness +18.8%), and offer informative confidence estimates (reliability +42.4%). These findings highlight an overlooked but important direction: reasoning models should be optimized not only for accuracy, but also for broader dimensions of trustworthiness. Our code is available at: this https URL 

---
# DITING: A Multi-Agent Evaluation Framework for Benchmarking Web Novel Translation 

**Authors**: Enze Zhang, Jiaying Wang, Mengxi Xiao, Jifei Liu, Ziyan Kuang, Rui Dong, Youzhong Dong, Sophia Ananiadou, Min Peng, Qianqian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.09116)  

**Abstract**: Large language models (LLMs) have substantially advanced machine translation (MT), yet their effectiveness in translating web novels remains unclear. Existing benchmarks rely on surface-level metrics that fail to capture the distinctive traits of this genre. To address these gaps, we introduce DITING, the first comprehensive evaluation framework for web novel translation, assessing narrative and cultural fidelity across six dimensions: idiom translation, lexical ambiguity, terminology localization, tense consistency, zero-pronoun resolution, and cultural safety, supported by over 18K expert-annotated Chinese-English sentence pairs. We further propose AgentEval, a reasoning-driven multi-agent evaluation framework that simulates expert deliberation to assess translation quality beyond lexical overlap, achieving the highest correlation with human judgments among seven tested automatic metrics. To enable metric comparison, we develop MetricAlign, a meta-evaluation dataset of 300 sentence pairs annotated with error labels and scalar quality scores. Comprehensive evaluation of fourteen open, closed, and commercial models reveals that Chinese-trained LLMs surpass larger foreign counterparts, and that DeepSeek-V3 delivers the most faithful and stylistically coherent translations. Our work establishes a new paradigm for exploring LLM-based web novel translation and provides public resources to advance future research. 

---
# DARO: Difficulty-Aware Reweighting Policy Optimization 

**Authors**: Jingyu Zhou, Lu Ma, Hao Liang, Chengyu Shen, Bin Cui, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09001)  

**Abstract**: Recent advances in large language models (LLMs) have shown that reasoning ability can be significantly enhanced through Reinforcement Learning with Verifiable Rewards (RLVR). Group Relative Policy Optimization (GRPO) has emerged as the de facto approach for RLVR, inspiring numerous variants. However, our mathematical analysis reveals that these methods are fundamentally weighted variations of GRPO. We provide a unified view, demonstrating that their reliance on static or overly simplistic weighting schemes tied to sample difficulty prevents adaptation to a model's evolving capabilities. This creates a significant loss scale issue, where training disproportionately focuses on certain difficulty levels at the expense of others, hindering overall performance. To address these limitations, we introduce \textbf{Difficulty-Aware Reweighting Policy Optimization (DARO)}, a method that dynamically adjusts the loss contribution of each difficulty group based on the model's learning state. Extensive experiments on Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, and Llama3.1-8B show that DARO outperforms four leading baselines across six math benchmarks, achieving significantly faster convergence and superior final performance. 

---
# Automated Refinement of Essay Scoring Rubrics for Language Models via Reflect-and-Revise 

**Authors**: Keno Harada, Lui Yoshida, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2510.09030)  

**Abstract**: The performance of Large Language Models (LLMs) is highly sensitive to the prompts they are given. Drawing inspiration from the field of prompt optimization, this study investigates the potential for enhancing Automated Essay Scoring (AES) by refining the scoring rubrics used by LLMs. Specifically, our approach prompts models to iteratively refine rubrics by reflecting on models' own scoring rationales and observed discrepancies with human scores on sample essays. Experiments on the TOEFL11 and ASAP datasets using GPT-4.1, Gemini-2.5-Pro, and Qwen-3-Next-80B-A3B-Instruct show Quadratic Weighted Kappa (QWK) improvements of up to 0.19 and 0.47, respectively. Notably, even with a simple initial rubric, our approach achieves comparable or better QWK than using detailed human-authored rubrics. Our findings highlight the importance of iterative rubric refinement in LLM-based AES to enhance alignment with human evaluations. 

---
# Artificial Impressions: Evaluating Large Language Model Behavior Through the Lens of Trait Impressions 

**Authors**: Nicholas Deas, Kathleen McKeown  

**Link**: [PDF](https://arxiv.org/pdf/2510.08915)  

**Abstract**: We introduce and study artificial impressions--patterns in LLMs' internal representations of prompts that resemble human impressions and stereotypes based on language. We fit linear probes on generated prompts to predict impressions according to the two-dimensional Stereotype Content Model (SCM). Using these probes, we study the relationship between impressions and downstream model behavior as well as prompt features that may inform such impressions. We find that LLMs inconsistently report impressions when prompted, but also that impressions are more consistently linearly decodable from their hidden representations. Additionally, we show that artificial impressions of prompts are predictive of the quality and use of hedging in model responses. We also investigate how particular content, stylistic, and dialectal features in prompts impact LLM impressions. 

---
# Creation of the Chinese Adaptive Policy Communication Corpus 

**Authors**: Bolun Sun, Charles Chang, Yuen Yuen Ang, Pingxu Hao, Ruotong Mu, Yuchen Xu, Zhengxin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08986)  

**Abstract**: We introduce CAPC-CG, the Chinese Adaptive Policy Communication (Central Government) Corpus, the first open dataset of Chinese policy directives annotated with a five-color taxonomy of clear and ambiguous language categories, building on Ang's theory of adaptive policy communication. Spanning 1949-2023, this corpus includes national laws, administrative regulations, and ministerial rules issued by China's top authorities. Each document is segmented into paragraphs, producing a total of 3.3 million units. Alongside the corpus, we release comprehensive metadata, a two-round labeling framework, and a gold-standard annotation set developed by expert and trained coders. Inter-annotator agreement achieves a Fleiss's kappa of K = 0.86 on directive labels, indicating high reliability for supervised modeling. We provide baseline classification results with several large language models (LLMs), together with our annotation codebook, and describe patterns from the dataset. This release aims to support downstream tasks and multilingual NLP research in policy communication. 

---
# FinAuditing: A Financial Taxonomy-Structured Multi-Document Benchmark for Evaluating LLMs 

**Authors**: Yan Wang, Keyi Wang, Shanshan Yang, Jaisal Patel, Jeff Zhao, Fengran Mo, Xueqing Peng, Lingfei Qian, Jimin Huang, Guojun Xiong, Xiao-Yang Liu, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2510.08886)  

**Abstract**: The complexity of the Generally Accepted Accounting Principles (GAAP) and the hierarchical structure of eXtensible Business Reporting Language (XBRL) filings make financial auditing increasingly difficult to automate and verify. While large language models (LLMs) have demonstrated strong capabilities in unstructured text understanding, their ability to reason over structured, interdependent, and taxonomy-driven financial documents remains largely unexplored. To fill this gap, we introduce FinAuditing, the first taxonomy-aligned, structure-aware, multi-document benchmark for evaluating LLMs on financial auditing tasks. Built from real US-GAAP-compliant XBRL filings, FinAuditing defines three complementary subtasks, FinSM for semantic consistency, FinRE for relational consistency, and FinMR for numerical consistency, each targeting a distinct aspect of structured auditing reasoning. We further propose a unified evaluation framework integrating retrieval, classification, and reasoning metrics across these subtasks. Extensive zero-shot experiments on 13 state-of-the-art LLMs reveal that current models perform inconsistently across semantic, relational, and mathematical dimensions, with accuracy drops of up to 60-90% when reasoning over hierarchical multi-document structures. Our findings expose the systematic limitations of modern LLMs in taxonomy-grounded financial reasoning and establish FinAuditing as a foundation for developing trustworthy, structure-aware, and regulation-aligned financial intelligence systems. The benchmark dataset is available at Hugging Face. 

---
# MASA: LLM-Driven Multi-Agent Systems for Autoformalization 

**Authors**: Lan Zhang, Marco Valentino, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2510.08988)  

**Abstract**: Autoformalization serves a crucial role in connecting natural language and formal reasoning. This paper presents MASA, a novel framework for building multi-agent systems for autoformalization driven by Large Language Models (LLMs). MASA leverages collaborative agents to convert natural language statements into their formal representations. The architecture of MASA is designed with a strong emphasis on modularity, flexibility, and extensibility, allowing seamless integration of new agents and tools to adapt to a fast-evolving field. We showcase the effectiveness of MASA through use cases on real-world mathematical definitions and experiments on formal mathematics datasets. This work highlights the potential of multi-agent systems powered by the interaction of LLMs and theorem provers in enhancing the efficiency and reliability of autoformalization, providing valuable insights and support for researchers and practitioners in the field. 

---
# SOP-Maze: Evaluating Large Language Models on Complicated Business Standard Operating Procedures 

**Authors**: Jiaming Wang, Zhe Tang, Yilin Jin, Peng Ding, Xiaoyu Li, Xuezhi Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.08942)  

**Abstract**: As large language models (LLMs) are widely deployed as domain-specific agents, many benchmarks have been proposed to evaluate their ability to follow instructions and make decisions in real-world scenarios. However, business scenarios often involve complex standard operating procedures (SOPs), and the evaluation of LLM capabilities in such contexts has not been fully explored. To bridge this gap, we propose SOP-Maze, a benchmark constructed from real-world business data and adapted into a collection of 397 tasks from 23 complex SOP scenarios. We further categorize SOP tasks into two broad classes: Lateral Root System (LRS), representing wide-option tasks that demand precise selection; and Heart Root System (HRS), which emphasizes deep logical reasoning with complex branches. Extensive experiments reveal that nearly all state-of-the-art models struggle with SOP-Maze. We conduct a comprehensive analysis and identify three key error categories: (i) route blindness: difficulty following procedures; (ii) conversational fragility: inability to handle real dialogue nuances; and (iii) calculation errors: mistakes in time or arithmetic reasoning under complex contexts. The systematic study explores LLM performance across SOP tasks that challenge both breadth and depth, offering new insights for improving model capabilities. We have open-sourced our work on this https URL. 

---
# Search-on-Graph: Iterative Informed Navigation for Large Language Model Reasoning on Knowledge Graphs 

**Authors**: Jia Ao Sun, Hao Yu, Fabrizio Gotti, Fengran Mo, Yihong Wu, Yuchen Hui, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2510.08825)  

**Abstract**: Large language models (LLMs) have demonstrated impressive reasoning abilities yet remain unreliable on knowledge-intensive, multi-hop questions -- they miss long-tail facts, hallucinate when uncertain, and their internal knowledge lags behind real-world change. Knowledge graphs (KGs) offer a structured source of relational evidence, but existing KGQA methods face fundamental trade-offs: compiling complete SPARQL queries without knowing available relations proves brittle, retrieving large subgraphs introduces noise, and complex agent frameworks with parallel exploration exponentially expand search spaces. To address these limitations, we propose Search-on-Graph (SoG), a simple yet effective framework that enables LLMs to perform iterative informed graph navigation using a single, carefully designed \textsc{Search} function. Rather than pre-planning paths or retrieving large subgraphs, SoG follows an ``observe-then-navigate'' principle: at each step, the LLM examines actual available relations from the current entity before deciding on the next hop. This approach further adapts seamlessly to different KG schemas and handles high-degree nodes through adaptive filtering. Across six KGQA benchmarks spanning Freebase and Wikidata, SoG achieves state-of-the-art performance without fine-tuning. We demonstrate particularly strong gains on Wikidata benchmarks (+16\% improvement over previous best methods) alongside consistent improvements on Freebase benchmarks. 

---
# Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors 

**Authors**: Xin Liu, RunSong Zhao, PengCheng Huang, XinYu Liu, JunYi Xiao, ChunYang Xiao, Tong Xiao, Shengxiang Gao, Zhengtao Yu, JingBo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08907)  

**Abstract**: Context compression presents a promising approach for accelerating large language model (LLM) inference by compressing long contexts into compact representations. Current context compression methods predominantly rely on autoencoding tasks to train context-agnostic compression tokens to compress contextual semantics. While autoencoding tasks enable compression tokens to acquire compression capabilities, compression via autoencoding tasks creates a fundamental mismatch: the models are optimized for reconstruction that diverge from actual downstream tasks, thereby weakening the features more beneficial for real-world usage. We propose Semantic-Anchor Compression (SAC), a novel method that shifts from autoencoding task based compression to an architecture that is equipped with this compression capability \textit{a priori}. Instead of training models to compress contexts through autoencoding tasks, SAC directly selects so-called anchor tokens from the original context and aggregates contextual information into their key-value (KV) representations. By deriving representations directly from the contextual tokens, SAC eliminates the need for autoencoding training. To ensure compression performance while directly leveraging anchor tokens, SAC incorporates two key designs: (1) anchor embeddings that enable the compressor to identify critical tokens, and (2) bidirectional attention modification that allows anchor tokens to capture information from the entire context. Experimental results demonstrate that SAC consistently outperforms existing context compression methods across various compression ratios. On out-of-distribution evaluation using MRQA, SAC achieves 1 EM improvement at 5x compression over strong baselines, with increasing advantages at higher compression ratios. 

---
# Quality Estimation Reranking for Document-Level Translation 

**Authors**: Krzysztof Mrozinski, Minji Kang, Ahmed Khota, Vincent Michael Sutanto, Giovanni Gatti De Giacomo  

**Link**: [PDF](https://arxiv.org/pdf/2510.08870)  

**Abstract**: Quality estimation (QE) reranking is a form of quality-aware decoding which aims to improve machine translation (MT) by scoring and selecting the best candidate from a pool of generated translations. While known to be effective at the sentence level, its application to the increasingly prominent domain of document-level translation remains underexplored. In this work, we evaluate QE reranking performance on document-level (rather than the typical sentence-level) translation, using various learned and large language model (LLM)-based QE metrics. We find that with our best learned metric, SLIDE, BLEURT-20 scores improve by +2.00 with only two candidates, and by +5.09 with 32, across both decoder-only LLM models and encoder-decoder neural machine translation (NMT) models. Using the best LLM-based metric, GEMBA-DA, gains of +1.63 and +4.30 are achieved under the same conditions. Although gains shrink with longer inputs, reranking with 32 candidates yields improvements of +2.34 (SLIDE) and +1.40 (GEMBA-DA) on our longest documents (512-1024 source tokens). These findings demonstrate the practical value of document-level QE, with minimal runtime overhead given suitable translation models and hardware. 

---
# How Many Code and Test Cases Are Enough? Evaluating Test Cases Generation from a Binary-Matrix Perspective 

**Authors**: Xianzhen Luo, Jinyang Huang, Wenzhen Zheng, Qingfu Zhu, Mingzheng Xu, Yiheng Xu, Yuantao Fan, Libo Qin, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2510.08720)  

**Abstract**: Evaluating test cases automatically generated by Large Language Models (LLMs) is a critical yet challenging task. Existing benchmarks suffer from high computational costs, score inflation, and a bias towards trivial bugs over rare, critical faults. In this work, we ask two fundamental questions: (1) What is the minimal set of wrong codes sufficient to represent the entire error space? and (2) What is the minimal set of test cases needed to distinguish them? We introduce a framework that formalizes benchmark construction as finding an optimal diagnostic basis in a binary code-test matrix. The rank of this matrix specifies the minimal number of independent error patterns (wrong codes) and provides a tight upper bound on the number of test cases required for complete fault coverage. Our objective is to identify a basis of size equal to the matrix rank that maximizes internal diversity. To tackle this NP-hard problem, we propose WrongSelect, an efficient approximation algorithm to select maximally diverse wrong codes. Applying this framework to millions of competitive programming submissions, we construct TC-Bench, a compact, diverse, and inflation-resistant benchmark. Extensive experiments show that even the most advanced test case generation methods achieve only ~60% exclusion rates on TC-Bench, exposing a significant gap in their diagnostic power. Our dataset is available at: this https URL and our code is at: this https URL. 

---
# Thinking Longer, Not Always Smarter: Evaluating LLM Capabilities in Hierarchical Legal Reasoning 

**Authors**: Li Zhang, Matthias Grabmair, Morgan Gray, Kevin Ashley  

**Link**: [PDF](https://arxiv.org/pdf/2510.08710)  

**Abstract**: Case-based reasoning is a cornerstone of U.S. legal practice, requiring professionals to argue about a current case by drawing analogies to and distinguishing from past precedents. While Large Language Models (LLMs) have shown remarkable capabilities, their proficiency in this complex, nuanced form of reasoning needs further investigation. We propose a formal framework that decomposes the process of identifying significant distinctions between cases into three-stage reasoning tasks. Our framework models cases using factual predicates called factors, organizes them into a legal knowledge hierarchy, and defines verifiable rules for identifying distinctions, analyzing their argumentative support, and evaluating their significance. Through comprehensive evaluation of modern reasoning LLMs, we reveal a paradox: while models achieve high accuracy on surface-level reasoning (Task 1), performance degrades on hierarchical reasoning (Task 2: 64.82%-92.09%) and collapses on integrated analysis (Task 3: 11.46%-33.99%). Most strikingly, we find that models consistently expend more computational resources on incorrect responses than correct ones, suggesting that "thinking longer" does not always mean "thinking smarter." Our work provides a methodology for fine-grained analysis of LLM reasoning capabilities in complex domains and reveals fundamental limitations that must be addressed for robust and trustworthy legal AI. 

---
# MOSAIC: Multi-agent Orchestration for Task-Intelligent Scientific Coding 

**Authors**: Siddeshwar Raghavan, Tanwi Mallick  

**Link**: [PDF](https://arxiv.org/pdf/2510.08804)  

**Abstract**: We present MOSAIC, a multi-agent Large Language Model (LLM) framework for solving challenging scientific coding tasks. Unlike general-purpose coding, scientific workflows require algorithms that are rigorous, interconnected with deep domain knowledge, and incorporate domain-specific reasoning, as well as algorithm iteration without requiring I/O test cases. Many scientific problems also require a sequence of subproblems to be solved, leading to the final desired result. MOSAIC is designed as a training-free framework with specially designed agents to self-reflect, create the rationale, code, and debug within a student-teacher paradigm to address the challenges of scientific code generation. This design facilitates stepwise problem decomposition, targeted error correction, and, when combined with our Consolidated Context Window (CCW), mitigates LLM hallucinations when solving complex scientific tasks involving chained subproblems. We evaluate MOSAIC on scientific coding benchmarks and demonstrate that our specialized agentic framework outperforms existing approaches in terms of accuracy, robustness, and interpretability. 

---
# ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection 

**Authors**: Jingbiao Mei, Mingsheng Sun, Jinghong Chen, Pengda Qin, Yuhong Li, Da Chen, Bill Byrne  

**Link**: [PDF](https://arxiv.org/pdf/2510.08630)  

**Abstract**: Hateful memes have emerged as a particularly challenging form of online abuse, motivating the development of automated detection systems. Most prior approaches rely on direct detection, producing only binary predictions. Such models fail to provide the context and explanations that real-world moderation requires. Recent Explain-then-Detect approaches, using Chain-of-Thought prompting or LMM agents, perform worse than simple SFT baselines, and even advanced post-training methods such as GRPO fail to close the gap. Our analysis identifies two key issues of such systems: important policy-relevant cues such as targets and attack types are not hypothesized by the model as a likely explanation; and the binary reward signal is insufficient to guide reasoning. To address these challenges, we propose ExPO-HM (Explain-then-Detect Policy Optimization for Hateful Memes), inspired by the training and evaluation process of human annotators. ExPO-HM combines SFT warmup, GRPO with curriculum learning, and Conditional Decision Entropy (CDE) as both metric and reward for reasoning quality. Across three hateful meme benchmarks, ExPO-HM achieves state-of-the-art performance on binary detection, fine-grained classification, and reasoning quality, with up to 15\% and 17\% F1 improvement over the GRPO and DPO baselines, respectively. By moving hateful meme detection from simple binary alarms to explanation-driven detection, ExPO-HM provides accurate, interpretable, and actionable moderation support. 

---
# Text2Stories: Evaluating the Alignment Between Stakeholder Interviews and Generated User Stories 

**Authors**: Francesco Dente, Fabiano Dalpiaz, Paolo Papotti  

**Link**: [PDF](https://arxiv.org/pdf/2510.08622)  

**Abstract**: Large language models (LLMs) can be employed for automating the generation of software requirements from natural language inputs such as the transcripts of elicitation interviews. However, evaluating whether those derived requirements faithfully reflect the stakeholders' needs remains a largely manual task. We introduce Text2Stories, a task and metrics for text-to-story alignment that allow quantifying the extent to which requirements (in the form of user stories) match the actual needs expressed by the elicitation session participants. Given an interview transcript and a set of user stories, our metric quantifies (i) correctness: the proportion of stories supported by the transcript, and (ii) completeness: the proportion of transcript supported by at least one story. We segment the transcript into text chunks and instantiate the alignment as a matching problem between chunks and stories. Experiments over four datasets show that an LLM-based matcher achieves 0.86 macro-F1 on held-out annotations, while embedding models alone remain behind but enable effective blocking. Finally, we show how our metrics enable the comparison across sets of stories (e.g., human vs. generated), positioning Text2Stories as a scalable, source-faithful complement to existing user-story quality criteria. 

---
# LLMs Show Surface-Form Brittleness Under Paraphrase Stress Tests 

**Authors**: Juan Miguel Navarro Carranza  

**Link**: [PDF](https://arxiv.org/pdf/2510.08616)  

**Abstract**: Benchmark scores for Large Language Models (LLMs) can be inflated by memorization of test items or near duplicates. We present a simple, protocol that probes generalization by re-evaluating models on paraphrased versions of benchmark questions. Using Mistral-7B-Instruct and Qwen2.5-7B-Instruct, we measure the accuracy gap between original and paraphrased items on ARC-Easy and ARC-Challenge. Our pipeline controls decoding, enforces multiple-choice output format, and includes a robust paraphrase-cleaning step to preserve semantics. We find that paraphrasing induces a non-trivial accuracy drop (original vs. paraphrased), consistent with prior concerns about contamination and brittle surface-form shortcuts. 

---
# Iterative LLM-Based Generation and Refinement of Distracting Conditions in Math Word Problems 

**Authors**: Kaiqi Yang, Hang Li, Yucheng Chu, Zitao Liu, Mi Tian, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08615)  

**Abstract**: Mathematical reasoning serves as a crucial testbed for evaluating the intelligence of large language models (LLMs), and math word problems (MWPs) represent one of the most widely used formats. Most existing MWP datasets contain only the necessary information, while problems with distracting or excessive conditions are often overlooked. Prior studies have shown that popular LLMs experience a dramatic performance drop when such distracting conditions are introduced. However, available datasets of MWPs with distracting conditions remain limited, and most exhibit low difficulty and out-of-context expressions. These shortcomings make the distracting conditions easy to detect and disregard, thereby reducing the credibility of benchmarking on these datasets. Moreover, when distracting conditions are added, the reasoning process and answers may change, requiring intensive manual effort to check and rewrite solutions.
To address these issues, we design an iterative framework that leverages LLMs to generate distracting conditions automatically. We develop a set of prompts to revise MWPs from multiple perspectives and cognitive levels, encouraging the creation of meaningful distracting conditions as well as suggestions for further refinement. A key advantage of our framework is the preservation of shared solutions between the original and revised problems: the LLMs are explicitly guided to generate distractions that do not alter the original solution, thus eliminating the need to produce new answers. This framework is efficient and easy to deploy, substantially reducing the effort required to generate MWPs with distracting conditions while maintaining high data quality. 

---
# Do LLMs Know They Are Being Tested? Evaluation Awareness and Incentive-Sensitive Failures in GPT-OSS-20B 

**Authors**: Nisar Ahmed, Muhammad Imran Zaman, Gulshan Saleem, Ali Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2510.08624)  

**Abstract**: Benchmarks for large language models (LLMs) often rely on rubric-scented prompts that request visible reasoning and strict formatting, whereas real deployments demand terse, contract-bound answers. We investigate whether such "evaluation scent" inflates measured performance without commensurate capability gains. Using a single open-weights model (GPT-OSS-20B), we run six paired A/B scenarios that hold task content and decoding fixed while varying framing (evaluation-oriented vs. real-world) and reasoning depth (Medium/High): deterministic math, strict code-fix, citation generation, incentive flips (caution vs. competence), CoT visibility, and multilingual (Urdu) headers. Deterministic validators compute accuracy, answer-only compliance, hedging/refusals, chain-of-thought (CoT) length, and schema compliance, with pre-registered deltas and composite indices. Across scenarios, evaluation framing reliably inflates CoT (hundreds to >1000 characters) and reduces answer-only compliance, with limited or inconsistent accuracy gains. In structured outputs, it improves wrappers (e.g., fenced blocks, enumerated lists) but not regex-validated substance. Incentive wording reweights error composition: praising caution modestly improves accuracy at high reasoning and reduces wrong-but-confident errors, whereas praising competence yields terser but riskier outputs. Urdu rubric headers reproduce these signatures and can decrease accuracy at higher reasoning depth, indicating multilingual parity risks. We provide a reproducible A/B framework (prompt banks, validators, per-run scores, scripts; versioned DOI) and practical guidance: neutral phrasing or dual-framing checks, contract-aware grading, style-delta reporting, confidence governance, and multilingual dashboards to ensure that benchmark gains reflect deployable capability. 

---
# GraphGhost: Tracing Structures Behind Large Language Models 

**Authors**: Xinnan Dai, Kai Guo, Chung-Hsiang Lo, Shenglai Zeng, Jiayuan Ding, Dongsheng Luo, Subhabrata Mukherjee, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08613)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable reasoning capabilities, yet the structural mechanisms underlying these abilities remain under explored. In this work, we introduce GraphGhost, a unified framework that represents neuron activations and their signal propagation as graphs, explaining how LLMs capture structural semantics from sequential inputs and generate outputs through structurally consistent mechanisms. This graph-based perspective enables us to employ graph algorithms such as PageRank to characterize the properties of LLMs, revealing both shared and model-specific reasoning behaviors across diverse datasets. We further identify the activated neurons within GraphGhost and evaluate them through structural interventions, showing that edits to key neuron nodes can trigger reasoning collapse, altering both logical flow and semantic understanding. Together, these contributions position GraphGhost as a powerful tool for analyzing, intervening in, and ultimately understanding the structural foundations of reasoning in LLMs. 

---
# Gender Bias in Large Language Models for Healthcare: Assignment Consistency and Clinical Implications 

**Authors**: Mingxuan Liu, Yuhe Ke, Wentao Zhu, Mayli Mertens, Yilin Ning, Jingchi Liao, Chuan Hong, Daniel Shu Wei Ting, Yifan Peng, Danielle S. Bitterman, Marcus Eng Hock Ong, Nan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08614)  

**Abstract**: The integration of large language models (LLMs) into healthcare holds promise to enhance clinical decision-making, yet their susceptibility to biases remains a critical concern. Gender has long influenced physician behaviors and patient outcomes, raising concerns that LLMs assuming human-like roles, such as clinicians or medical educators, may replicate or amplify gender-related biases. Using case studies from the New England Journal of Medicine Challenge (NEJM), we assigned genders (female, male, or unspecified) to multiple open-source and proprietary LLMs. We evaluated their response consistency across LLM-gender assignments regarding both LLM-based diagnosis and models' judgments on the clinical relevance or necessity of patient gender. In our findings, diagnoses were relatively consistent across LLM genders for most models. However, for patient gender's relevance and necessity in LLM-based diagnosis, all models demonstrated substantial inconsistency across LLM genders, particularly for relevance judgements. Some models even displayed a systematic female-male disparity in their interpretation of patient gender. These findings present an underexplored bias that could undermine the reliability of LLMs in clinical practice, underscoring the need for routine checks of identity-assignment consistency when interacting with LLMs to ensure reliable and equitable AI-supported clinical care. 

---
# YpathRAG:A Retrieval-Augmented Generation Framework and Benchmark for Pathology 

**Authors**: Deshui Yu, Yizhi Wang, Saihui Jin, Taojie Zhu, Fanyi Zeng, Wen Qian, Zirui Huang, Jingli Ouyang, Jiameng Li, Zhen Song, Tian Guan, Yonghong He  

**Link**: [PDF](https://arxiv.org/pdf/2510.08603)  

**Abstract**: Large language models (LLMs) excel on general tasks yet still hallucinate in high-barrier domains such as pathology. Prior work often relies on domain fine-tuning, which neither expands the knowledge boundary nor enforces evidence-grounded constraints. We therefore build a pathology vector database covering 28 subfields and 1.53 million paragraphs, and present YpathRAG, a pathology-oriented RAG framework with dual-channel hybrid retrieval (BGE-M3 dense retrieval coupled with vocabulary-guided sparse retrieval) and an LLM-based supportive-evidence judgment module that closes the retrieval-judgment-generation loop. We also release two evaluation benchmarks, YpathR and YpathQA-M. On YpathR, YpathRAG attains Recall@5 of 98.64%, a gain of 23 percentage points over the baseline; on YpathQA-M, a set of the 300 most challenging questions, it increases the accuracies of both general and medical LLMs by 9.0% on average and up to 15.6%. These results demonstrate improved retrieval quality and factual reliability, providing a scalable construction paradigm and interpretable evaluation for pathology-oriented RAG. 

---
# Systematic Diagnosis of Brittle Reasoning in Large Language Models 

**Authors**: V. S. Raghu Parupudi  

**Link**: [PDF](https://arxiv.org/pdf/2510.08595)  

**Abstract**: A central question in artificial intelligence is the extent to which machine learning models comprehend mathematics. To address this, we propose a novel framework for measuring mathematical reasoning that moves beyond standard benchmarks to diagnose specific failure points. Our method first generates structured, step-by-step reasoning from gpt-3.5-turbo on the GSM8K dataset. We then use a more capable analyst model, gpt-4o-mini, to categorize errors and, crucially, perform an unsupervised clustering of every reasoning sentence to identify emergent "reasoning modes." This analysis reveals a cognitive profile with a stark, nonhuman-like brittleness: while the model achieves near-perfect accuracy on procedural modes like sequential calculation, its performance on modes requiring combinatorial reasoning with restrictions plummets. By identifying and quantifying the reliability of these distinct reasoning skills, our work provides a more granular method to evaluate mathematical comprehension and offers a precise roadmap for developing new capabilities and more reliable future applications. 

---
# Confidence, Not Perplexity: A Better Metric for the Creative Era of LLMs 

**Authors**: V. S. Raghu Parupudi  

**Link**: [PDF](https://arxiv.org/pdf/2510.08596)  

**Abstract**: Reference-free metrics like self-perplexity are strongly biased against creative text generation. We propose the Confidence Score (CS), derived from a model's output probability distribution, as a less biased alternative. Experiments on gpt-4o-mini show that while fluency-based metrics prefer novel responses in 0\% of cases on 99 creative prompts, our CS does so 19% of the time, a statistically significant difference (95% CI for difference: [11.1%, 27.3%]). We also show that CS effectively distinguishes between easy, medium, and hard tasks, confirmed by non-overlapping confidence intervals. The Confidence Score thus mitigates the creativity bias of traditional metrics while retaining their core evaluative strengths, offering a more balanced assessment for modern LLMs. 

---
# PARSE: LLM Driven Schema Optimization for Reliable Entity Extraction 

**Authors**: Anubhav Shrimal, Aryan Jain, Soumyajit Chowdhury, Promod Yenigalla  

**Link**: [PDF](https://arxiv.org/pdf/2510.08623)  

**Abstract**: Structured information extraction from unstructured text is critical for emerging Software 3.0 systems where LLM agents autonomously interact with APIs and tools. Recent approaches apply large language models directly to extraction tasks using existing JSON schemas, often with constraint decoding or reinforcement learning approaches to ensure syntactic validity, but treat JSON schemas as static contracts designed for human developers, leading to suboptimal extraction performance, frequent hallucinations, and unreliable agent behavior when schemas contain ambiguous or incomplete specifications. We recognize that JSON schemas themselves are a form of natural language understanding contract that encodes rules, relationships, and expectations about data structure contracts that LLMs should be able to both interpret and systematically improve. Consequently, we develop PARSE (Parameter Automated Refinement and Schema Extraction), a novel system with two synergistic components: ARCHITECT, which autonomously optimizes JSON schemas for LLM consumption while maintaining backward compatibility through RELAY (an integrated code generation system), and SCOPE, which implements reflection-based extraction with combined static and LLM-based guardrails. We evaluate PARSE qualitatively and quantitatively on three datasets including Schema-Guided Dialogue (SGD), Structured Web Data Extraction (SWDE), and internal retail conversation data, and find that it achieves up to 64.7% improvement in extraction accuracy on SWDE with combined framework improvements reaching 10% across models, while reducing extraction errors by 92% within the first retry and and maintaining practical latency. 

---
# HINT: Helping Ineffective Rollouts Navigate Towards Effectiveness 

**Authors**: Xinyi Wang, Jinyi Han, Zishang Jiang, Tingyun Li, Jiaqing Liang, Sihang Jiang, Zhaoqian Dai, Shuguang Ma, Fei Yu, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.09388)  

**Abstract**: Reinforcement Learning (RL) has become a key driver for enhancing the long chain-of-thought (CoT) reasoning capabilities of Large Language Models (LLMs). However, prevalent methods like GRPO often fail when task difficulty exceeds the model's capacity, leading to reward sparsity and inefficient training. While prior work attempts to mitigate this using off-policy data, such as mixing RL with Supervised Fine-Tuning (SFT) or using hints, they often misguide policy updates In this work, we identify a core issue underlying these failures, which we term low training affinity. This condition arises from a large distributional mismatch between external guidance and the model's policy. To diagnose this, we introduce Affinity, the first quantitative metric for monitoring exploration efficiency and training stability. To improve Affinity, we propose HINT: Helping Ineffective rollouts Navigate Towards effectiveness, an adaptive hinting framework. Instead of providing direct answers, HINT supplies heuristic hints that guide the model to discover solutions on its own, preserving its autonomous reasoning capabilities. Extensive experiments on mathematical reasoning tasks show that HINT consistently outperforms existing methods, achieving state-of-the-art results with models of various scales, while also demonstrating significantly more stable learning and greater data this http URL is available on Github. 

---
# Human Texts Are Outliers: Detecting LLM-generated Texts via Out-of-distribution Detection 

**Authors**: Cong Zeng, Shengkun Tang, Yuanzhou Chen, Zhiqiang Shen, Wenchao Yu, Xujiang Zhao, Haifeng Chen, Wei Cheng, Zhiqiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08602)  

**Abstract**: The rapid advancement of large language models (LLMs) such as ChatGPT, DeepSeek, and Claude has significantly increased the presence of AI-generated text in digital communication. This trend has heightened the need for reliable detection methods to distinguish between human-authored and machine-generated content. Existing approaches both zero-shot methods and supervised classifiers largely conceptualize this task as a binary classification problem, often leading to poor generalization across domains and models. In this paper, we argue that such a binary formulation fundamentally mischaracterizes the detection task by assuming a coherent representation of human-written texts. In reality, human texts do not constitute a unified distribution, and their diversity cannot be effectively captured through limited sampling. This causes previous classifiers to memorize observed OOD characteristics rather than learn the essence of `non-ID' behavior, limiting generalization to unseen human-authored inputs. Based on this observation, we propose reframing the detection task as an out-of-distribution (OOD) detection problem, treating human-written texts as distributional outliers while machine-generated texts are in-distribution (ID) samples. To this end, we develop a detection framework using one-class learning method including DeepSVDD and HRN, and score-based learning techniques such as energy-based method, enabling robust and generalizable performance. Extensive experiments across multiple datasets validate the effectiveness of our OOD-based approach. Specifically, the OOD-based method achieves 98.3% AUROC and AUPR with only 8.9% FPR95 on DeepFake dataset. Moreover, we test our detection framework on multilingual, attacked, and unseen-model and -domain text settings, demonstrating the robustness and generalizability of our framework. Code, pretrained weights, and demo will be released. 

---
# Exploiting Web Search Tools of AI Agents for Data Exfiltration 

**Authors**: Dennis Rall, Bernhard Bauer, Mohit Mittal, Thomas Fraunholz  

**Link**: [PDF](https://arxiv.org/pdf/2510.09093)  

**Abstract**: Large language models (LLMs) are now routinely used to autonomously execute complex tasks, from natural language processing to dynamic workflows like web searches. The usage of tool-calling and Retrieval Augmented Generation (RAG) allows LLMs to process and retrieve sensitive corporate data, amplifying both their functionality and vulnerability to abuse. As LLMs increasingly interact with external data sources, indirect prompt injection emerges as a critical and evolving attack vector, enabling adversaries to exploit models through manipulated inputs. Through a systematic evaluation of indirect prompt injection attacks across diverse models, we analyze how susceptible current LLMs are to such attacks, which parameters, including model size and manufacturer, specific implementations, shape their vulnerability, and which attack methods remain most effective. Our results reveal that even well-known attack patterns continue to succeed, exposing persistent weaknesses in model defenses. To address these vulnerabilities, we emphasize the need for strengthened training procedures to enhance inherent resilience, a centralized database of known attack vectors to enable proactive defense, and a unified testing framework to ensure continuous security validation. These steps are essential to push developers toward integrating security into the core design of LLMs, as our findings show that current models still fail to mitigate long-standing threats. 

---
# Diagnosing Shoulder Disorders Using Multimodal Large Language Models and Consumer-Grade Cameras 

**Authors**: Jindong Hong, Wencheng Zhang, Shiqin Qiao, Jianhai Chen, Jianing Qiu, Chuanyang Zheng, Qian Xu, Yun Ji, Qianyue Wen, Weiwei Sun, Hao Li, Huizhen Li, Huichao Wang, Kai Wu, Meng Li, Yijun He, Lingjie Luo, Jiankai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.09230)  

**Abstract**: Shoulder disorders, such as frozen shoulder (a.k.a., adhesive capsulitis), are common conditions affecting the health of people worldwide, and have a high incidence rate among the elderly and workers engaged in repetitive shoulder tasks. In regions with scarce medical resources, achieving early and accurate diagnosis poses significant challenges, and there is an urgent need for low-cost and easily scalable auxiliary diagnostic solutions. This research introduces videos captured by consumer-grade devices as the basis for diagnosis, reducing the cost for users. We focus on the innovative application of Multimodal Large Language Models (MLLMs) in the preliminary diagnosis of shoulder disorders and propose a Hybrid Motion Video Diagnosis framework (HMVDx). This framework divides the two tasks of action understanding and disease diagnosis, which are respectively completed by two MLLMs. In addition to traditional evaluation indicators, this work proposes a novel metric called Usability Index by the logical process of medical decision-making (action recognition, movement diagnosis, and final diagnosis). This index evaluates the effectiveness of MLLMs in the medical field from the perspective of the entire medical diagnostic pathway, revealing the potential value of low-cost MLLMs in medical applications for medical practitioners. In experimental comparisons, the accuracy of HMVDx in diagnosing shoulder joint injuries has increased by 79.6\% compared with direct video diagnosis, a significant technical contribution to future research on the application of MLLMs for video understanding in the medical field. 

---
# Large Language Model Prompt Datasets: An In-depth Analysis and Insights 

**Authors**: Yuanming Zhang, Yan Lin, Arijit Khan, Huaiyu Wan  

**Link**: [PDF](https://arxiv.org/pdf/2510.09316)  

**Abstract**: A prompt is a natural language instruction that defines a specific task for a large language model (LLM) and serves as the primary interface for human-LLM interaction. With the growing deployment of LLMs, diverse prompt datasets are emerging from platforms such as GitHub and social media. These datasets span a wide array of applications and content types, facilitating both broader LLM utilization and improved prompt engineering. In this work, we--for the first time--have compiled an extensive list of prompt datasets sourced from various channels, representing a spectrum of downstream tasks, languages, engineering techniques, attributes, and modalities. We select key representative datasets for systematic analysis, revealing commonalities and differences in prompt construction across categories, distinguishing them from other text corpora like literature and web. We further propose a prompt optimization approach that leverages syntactic embeddings of part-of-speech and dependency structures. By identifying a centroid representation of prompts and guiding LLMs to rewrite prompts toward this centroid, our method improves the meaningfulness of model outputs. We have made our datasets and code available. 

---
# A Design-based Solution for Causal Inference with Text: Can a Language Model Be Too Large? 

**Authors**: Graham Tierney, Srikar Katta, Christopher Bail, Sunshine Hillygus, Alexander Volfovsky  

**Link**: [PDF](https://arxiv.org/pdf/2510.08758)  

**Abstract**: Many social science questions ask how linguistic properties causally affect an audience's attitudes and behaviors. Because text properties are often interlinked (e.g., angry reviews use profane language), we must control for possible latent confounding to isolate causal effects. Recent literature proposes adapting large language models (LLMs) to learn latent representations of text that successfully predict both treatment and the outcome. However, because the treatment is a component of the text, these deep learning methods risk learning representations that actually encode the treatment itself, inducing overlap bias. Rather than depending on post-hoc adjustments, we introduce a new experimental design that handles latent confounding, avoids the overlap issue, and unbiasedly estimates treatment effects. We apply this design in an experiment evaluating the persuasiveness of expressing humility in political communication. Methodologically, we demonstrate that LLM-based methods perform worse than even simple bag-of-words models using our real text and outcomes from our experiment. Substantively, we isolate the causal effect of expressing humility on the perceived persuasiveness of political statements, offering new insights on communication effects for social media platforms, policy makers, and social scientists. 

---
# Personalize Before Retrieve: LLM-based Personalized Query Expansion for User-Centric Retrieval 

**Authors**: Yingyi Zhang, Pengyue Jia, Derong Xu, Yi Wen, Xianneng Li, Yichao Wang, Wenlin Zhang, Xiaopeng Li, Weinan Gan, Huifeng Guo, Yong Liu, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.08935)  

**Abstract**: Retrieval-Augmented Generation (RAG) critically depends on effective query expansion to retrieve relevant information. However, existing expansion methods adopt uniform strategies that overlook user-specific semantics, ignoring individual expression styles, preferences, and historical context. In practice, identical queries in text can express vastly different intentions across users. This representational rigidity limits the ability of current RAG systems to generalize effectively in personalized settings. Specifically, we identify two core challenges for personalization: 1) user expression styles are inherently diverse, making it difficult for standard expansions to preserve personalized intent. 2) user corpora induce heterogeneous semantic structures-varying in topical focus and lexical organization-which hinders the effective anchoring of expanded queries within the user's corpora space. To address these challenges, we propose Personalize Before Retrieve (PBR), a framework that incorporates user-specific signals into query expansion prior to retrieval. PBR consists of two components: P-PRF, which generates stylistically aligned pseudo feedback using user history for simulating user expression style, and P-Anchor, which performs graph-based structure alignment over user corpora to capture its structure. Together, they produce personalized query representations tailored for retrieval. Experiments on two personalized benchmarks show that PBR consistently outperforms strong baselines, with up to 10% gains on PersonaBench across retrievers. Our findings demonstrate the value of modeling personalization before retrieval to close the semantic gap in user-adaptive RAG systems. Our code is available at this https URL. 

---
# Doc2Query++: Topic-Coverage based Document Expansion and its Application to Dense Retrieval via Dual-Index Fusion 

**Authors**: Tzu-Lin Kuo, Wei-Ning Chiu, Wei-Yun Ma, Pu-Jen Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09557)  

**Abstract**: Document expansion (DE) via query generation tackles vocabulary mismatch in sparse retrieval, yet faces limitations: uncontrolled generation producing hallucinated or redundant queries with low diversity; poor generalization from in-domain training (e.g., MS MARCO) to out-of-domain data like BEIR; and noise from concatenation harming dense retrieval. While Large Language Models (LLMs) enable cross-domain query generation, basic prompting lacks control, and taxonomy-based methods rely on domain-specific structures, limiting applicability. To address these challenges, we introduce Doc2Query++, a DE framework that structures query generation by first inferring a document's latent topics via unsupervised topic modeling for cross-domain applicability, then using hybrid keyword selection to create a diverse and relevant keyword set per document. This guides LLM not only to leverage keywords, which ensure comprehensive topic representation, but also to reduce redundancy through diverse, relevant terms. To prevent noise from query appending in dense retrieval, we propose Dual-Index Fusion strategy that isolates text and query signals, boosting performance in dense settings. Extensive experiments show Doc2Query++ significantly outperforms state-of-the-art baselines, achieving substantial gains in MAP, nDCG@10 and Recall@100 across diverse datasets on both sparse and dense retrieval. 

---
# Rethinking Reasoning in Document Ranking: Why Chain-of-Thought Falls Short 

**Authors**: Xuan Lu, Haohang Huang, Rui Meng, Yaohui Jin, Wenjun Zeng, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.08985)  

**Abstract**: Document reranking is a key component in information retrieval (IR), aimed at refining initial retrieval results to improve ranking quality for downstream tasks. Recent studies--motivated by large reasoning models (LRMs)--have begun incorporating explicit chain-of-thought (CoT) reasoning into LLM-based rerankers. However, the effectiveness of such reasoning for ranking tasks remains underexplored. In this work, we present the first systematic study of reasoning in reranking across both pointwise and listwise settings, under both supervised fine-tuning and reinforcement learning. Using diverse benchmarks, including reasoning-intensive datasets (BRIGHT) and standard IR benchmarks (BEIR), we find that reasoning-augmented rerankers consistently underperform their direct counterparts that predict rankings without CoT, despite substantially higher inference costs. Our analysis reveals three core limitations: (i) in pointwise rerankers, reasoning breaks calibration and biases models toward the positive class, raising TPR but lowering TNR, which inflates false positives and degrades ranking in negative-dominant pools; (ii) in listwise rerankers, reasoning improves in-domain fit but increases variance and fails to generalize out-of-domain, even when reinforcement learning shortens rationales; and (iii) overall, directly fine-tuned rerankers remain more stable, effective, and robust. These findings challenge the assumption that explicit reasoning is universally beneficial for reranking. We conclude by highlighting future directions, including calibration-aware scoring for pointwise rerankers and the design of concise, targeted reasoning strategies to mitigate overfitting and overthinking in listwise rerankers. 

---
