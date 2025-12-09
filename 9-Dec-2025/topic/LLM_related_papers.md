# Metric-Fair Prompting: Treating Similar Samples Similarly 

**Authors**: Jing Wang, Jie Shen, Xing Niu, Tong Zhang, Jeremy Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2512.07608)  

**Abstract**: We introduce \emph{Metric-Fair Prompting}, a fairness-aware prompting framework that guides large language models (LLMs) to make decisions under metric-fairness constraints. In the application of multiple-choice medical question answering, each {(question, option)} pair is treated as a binary instance with label $+1$ (correct) or $-1$ (incorrect). To promote {individual fairness}~--~treating similar instances similarly~--~we compute question similarity using NLP embeddings and solve items in \emph{joint pairs of similar questions} rather than in isolation. The prompt enforces a global decision protocol: extract decisive clinical features, map each \((\text{question}, \text{option})\) to a score $f(x)$ that acts as confidence, and impose a Lipschitz-style constraint so that similar inputs receive similar scores and, hence, consistent outputs. Evaluated on the {MedQA (US)} benchmark, Metric-Fair Prompting is shown to improve performance over standard single-item prompting, demonstrating that fairness-guided, confidence-oriented reasoning can enhance LLM accuracy on high-stakes clinical multiple-choice questions. 

---
# Do Generalisation Results Generalise? 

**Authors**: Matteo Boglioni, Andrea Sgobbi, Gabriel Tavernini, Francesco Rita, Marius Mosbach, Tiago Pimentel  

**Link**: [PDF](https://arxiv.org/pdf/2512.07832)  

**Abstract**: A large language model's (LLM's) out-of-distribution (OOD) generalisation ability is crucial to its deployment. Previous work assessing LLMs' generalisation performance, however, typically focuses on a single out-of-distribution dataset. This approach may fail to precisely evaluate the capabilities of the model, as the data shifts encountered once a model is deployed are much more diverse. In this work, we investigate whether OOD generalisation results generalise. More specifically, we evaluate a model's performance across multiple OOD testsets throughout a finetuning run; we then evaluate the partial correlation of performances across these testsets, regressing out in-domain performance. This allows us to assess how correlated are generalisation performances once in-domain performance is controlled for. Analysing OLMo2 and OPT, we observe no overarching trend in generalisation results: the existence of a positive or negative correlation between any two OOD testsets depends strongly on the specific choice of model analysed. 

---
# HalluShift++: Bridging Language and Vision through Internal Representation Shifts for Hierarchical Hallucinations in MLLMs 

**Authors**: Sujoy Nath, Arkaprabha Basu, Sharanya Dasgupta, Swagatam Das  

**Link**: [PDF](https://arxiv.org/pdf/2512.07687)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in vision-language understanding tasks. While these models often produce linguistically coherent output, they often suffer from hallucinations, generating descriptions that are factually inconsistent with the visual content, potentially leading to adverse consequences. Therefore, the assessment of hallucinations in MLLM has become increasingly crucial in the model development process. Contemporary methodologies predominantly depend on external LLM evaluators, which are themselves susceptible to hallucinations and may present challenges in terms of domain adaptation. In this study, we propose the hypothesis that hallucination manifests as measurable irregularities within the internal layer dynamics of MLLMs, not merely due to distributional shifts but also in the context of layer-wise analysis of specific assumptions. By incorporating such modifications, \textsc{\textsc{HalluShift++}} broadens the efficacy of hallucination detection from text-based large language models (LLMs) to encompass multimodal scenarios. Our codebase is available at this https URL. 

---
# Complementary Learning Approach for Text Classification using Large Language Models 

**Authors**: Navid Asgari, Benjamin M. Cole  

**Link**: [PDF](https://arxiv.org/pdf/2512.07583)  

**Abstract**: In this study, we propose a structured methodology that utilizes large language models (LLMs) in a cost-efficient and parsimonious manner, integrating the strengths of scholars and machines while offsetting their respective weaknesses. Our methodology, facilitated through a chain of thought and few-shot learning prompting from computer science, extends best practices for co-author teams in qualitative research to human-machine teams in quantitative research. This allows humans to utilize abductive reasoning and natural language to interrogate not just what the machine has done but also what the human has done. Our method highlights how scholars can manage inherent weaknesses OF LLMs using careful, low-cost techniques. We demonstrate how to use the methodology to interrogate human-machine rating discrepancies for a sample of 1,934 press releases announcing pharmaceutical alliances (1990-2017). 

---
# MoCoRP: Modeling Consistent Relations between Persona and Response for Persona-based Dialogue 

**Authors**: Kyungro Lee, Dongha Choi, Hyunju Lee  

**Link**: [PDF](https://arxiv.org/pdf/2512.07544)  

**Abstract**: As dialogue systems become increasingly important across various domains, a key challenge in persona-based dialogue is generating engaging and context-specific interactions while ensuring the model acts with a coherent personality. However, existing persona-based dialogue datasets lack explicit relations between persona sentences and responses, which makes it difficult for models to effectively capture persona information. To address these issues, we propose MoCoRP (Modeling Consistent Relations between Persona and Response), a framework that incorporates explicit relations into language models. MoCoRP leverages an NLI expert to explicitly extract the NLI relations between persona sentences and responses, enabling the model to effectively incorporate appropriate persona information from the context into its responses. We applied this framework to pre-trained models like BART and further extended it to modern large language models (LLMs) through alignment tuning. Experimental results on the public datasets ConvAI2 and MPChat demonstrate that MoCoRP outperforms existing baselines, achieving superior persona consistency and engaging, context-aware dialogue generation. Furthermore, our model not only excels in quantitative metrics but also shows significant improvements in qualitative aspects. These results highlight the effectiveness of explicitly modeling persona-response relations in persona-based dialogue. The source codes of MoCoRP are available at this https URL. 

---
# On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models 

**Authors**: Charlie Zhang, Graham Neubig, Xiang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2512.07783)  

**Abstract**: Recent reinforcement learning (RL) techniques have yielded impressive reasoning improvements in language models, yet it remains unclear whether post-training truly extends a model's reasoning ability beyond what it acquires during pre-training. A central challenge is the lack of control in modern training pipelines: large-scale pre-training corpora are opaque, mid-training is often underexamined, and RL objectives interact with unknown prior knowledge in complex ways. To resolve this ambiguity, we develop a fully controlled experimental framework that isolates the causal contributions of pre-training, mid-training, and RL-based post-training. Our approach employs synthetic reasoning tasks with explicit atomic operations, parseable step-by-step reasoning traces, and systematic manipulation of training distributions. We evaluate models along two axes: extrapolative generalization to more complex compositions and contextual generalization across surface contexts. Using this framework, we reconcile competing views on RL's effectiveness. We show that: 1) RL produces true capability gains (pass@128) only when pre-training leaves sufficient headroom and when RL data target the model's edge of competence, tasks at the boundary that are difficult but not yet out of reach. 2) Contextual generalization requires minimal yet sufficient pre-training exposure, after which RL can reliably transfer. 3) Mid-training significantly enhances performance under fixed compute compared with RL only, demonstrating its central but underexplored role in training pipelines. 4) Process-level rewards reduce reward hacking and improve reasoning fidelity. Together, these results clarify the interplay between pre-training, mid-training, and RL, offering a foundation for understanding and improving reasoning LM training strategies. 

---
# Bridging Code Graphs and Large Language Models for Better Code Understanding 

**Authors**: Zeqi Chen, Zhaoyang Chu, Yi Gui, Feng Guo, Yao Wan, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2512.07666)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance in code intelligence tasks such as code generation, summarization, and translation. However, their reliance on linearized token sequences limits their ability to understand the structural semantics of programs. While prior studies have explored graphaugmented prompting and structure-aware pretraining, they either suffer from prompt length constraints or require task-specific architectural changes that are incompatible with large-scale instructionfollowing LLMs. To address these limitations, this paper proposes CGBridge, a novel plug-and-play method that enhances LLMs with Code Graph information through an external, trainable Bridge module. CGBridge first pre-trains a code graph encoder via selfsupervised learning on a large-scale dataset of 270K code graphs to learn structural code semantics. It then trains an external module to bridge the modality gap among code, graph, and text by aligning their semantics through cross-modal attention mechanisms. Finally, the bridge module generates structure-informed prompts, which are injected into a frozen LLM, and is fine-tuned for downstream code intelligence tasks. Experiments show that CGBridge achieves notable improvements over both the original model and the graphaugmented prompting method. Specifically, it yields a 16.19% and 9.12% relative gain in LLM-as-a-Judge on code summarization, and a 9.84% and 38.87% relative gain in Execution Accuracy on code translation. Moreover, CGBridge achieves over 4x faster inference than LoRA-tuned models, demonstrating both effectiveness and efficiency in structure-aware code understanding. 

---
# Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning 

**Authors**: Tong Wu, Yang Liu, Jun Bai, Zixia Jia, Shuyi Zhang, Ziyong Lin, Yanting Wang, Song-Chun Zhu, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.07461)  

**Abstract**: We introduce Native Parallel Reasoner (NPR), a teacher-free framework that enables Large Language Models (LLMs) to self-evolve genuine parallel reasoning capabilities. NPR transforms the model from sequential emulation to native parallel cognition through three key innovations: 1) a self-distilled progressive training paradigm that transitions from ``cold-start'' format discovery to strict topological constraints without external supervision; 2) a novel Parallel-Aware Policy Optimization (PAPO) algorithm that optimizes branching policies directly within the execution graph, allowing the model to learn adaptive decomposition via trial and error; and 3) a robust NPR Engine that refactors memory management and flow control of SGLang to enable stable, large-scale parallel RL training. Across eight reasoning benchmarks, NPR trained on Qwen3-4B achieves performance gains of up to 24.5% and inference speedups up to 4.6x. Unlike prior baselines that often fall back to autoregressive decoding, NPR demonstrates 100% genuine parallel execution, establishing a new standard for self-evolving, efficient, and scalable agentic reasoning. 

---
# Investigating Training and Generalization in Faithful Self-Explanations of Large Language Models 

**Authors**: Tomoki Doi, Masaru Isonuma, Hitomi Yanaka  

**Link**: [PDF](https://arxiv.org/pdf/2512.07288)  

**Abstract**: Large language models have the potential to generate explanations for their own predictions in a variety of styles based on user instructions. Recent research has examined whether these self-explanations faithfully reflect the models' actual behavior and has found that they often lack faithfulness. However, the question of how to improve faithfulness remains underexplored. Moreover, because different explanation styles have superficially distinct characteristics, it is unclear whether improvements observed in one style also arise when using other styles. This study analyzes the effects of training for faithful self-explanations and the extent to which these effects generalize, using three classification tasks and three explanation styles. We construct one-word constrained explanations that are likely to be faithful using a feature attribution method, and use these pseudo-faithful self-explanations for continual learning on instruction-tuned models. Our experiments demonstrate that training can improve self-explanation faithfulness across all classification tasks and explanation styles, and that these improvements also show signs of generalization to the multi-word settings and to unseen tasks. Furthermore, we find consistent cross-style generalization among three styles, suggesting that training may contribute to a broader improvement in faithful self-explanation ability. 

---
# Ensembling LLM-Induced Decision Trees for Explainable and Robust Error Detection 

**Authors**: Mengqi Wang, Jianwei Wang, Qing Liu, Xiwei Xu, Zhenchang Xing, Liming Zhu, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.07246)  

**Abstract**: Error detection (ED), which aims to identify incorrect or inconsistent cell values in tabular data, is important for ensuring data quality. Recent state-of-the-art ED methods leverage the pre-trained knowledge and semantic capability embedded in large language models (LLMs) to directly label whether a cell is erroneous. However, this LLM-as-a-labeler pipeline (1) relies on the black box, implicit decision process, thus failing to provide explainability for the detection results, and (2) is highly sensitive to prompts, yielding inconsistent outputs due to inherent model stochasticity, therefore lacking robustness. To address these limitations, we propose an LLM-as-an-inducer framework that adopts LLM to induce the decision tree for ED (termed TreeED) and further ensembles multiple such trees for consensus detection (termed ForestED), thereby improving explainability and robustness. Specifically, based on prompts derived from data context, decision tree specifications and output requirements, TreeED queries the LLM to induce the decision tree skeleton, whose root-to-leaf decision paths specify the stepwise procedure for evaluating a given sample. Each tree contains three types of nodes: (1) rule nodes that perform simple validation checks (e.g., format or range), (2) Graph Neural Network (GNN) nodes that capture complex patterns (e.g., functional dependencies), and (3) leaf nodes that output the final decision types (error or clean). Furthermore, ForestED employs uncertainty-based sampling to obtain multiple row subsets, constructing a decision tree for each subset using TreeED. It then leverages an Expectation-Maximization-based algorithm that jointly estimates tree reliability and optimizes the consensus ED prediction. Extensive xperiments demonstrate that our methods are accurate, explainable and robust, achieving an average F1-score improvement of 16.1% over the best baseline. 

---
# NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models 

**Authors**: Feng Liang, Weixin Zeng, Runhao Zhao, Xiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.07218)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing tasks. However, temporal reasoning, particularly under complex temporal constraints, remains a major challenge. To this end, existing approaches have explored symbolic methods, which encode temporal structure explicitly, and reflective mechanisms, which revise reasoning errors through multi-step inference. Nonetheless, symbolic approaches often underutilize the reasoning capabilities of LLMs, while reflective methods typically lack structured temporal representations, which can result in inconsistent or hallucinated reasoning. As a result, even when the correct temporal context is available, LLMs may still misinterpret or misapply time-related information, leading to incomplete or inaccurate answers. To address these limitations, in this work, we propose Neuro-Symbolic Temporal Reasoning (NeSTR), a novel framework that integrates structured symbolic representations with hybrid reflective reasoning to enhance the temporal sensitivity of LLM inference. NeSTR preserves explicit temporal relations through symbolic encoding, enforces logical consistency via verification, and corrects flawed inferences using abductive reflection. Extensive experiments on diverse temporal question answering benchmarks demonstrate that NeSTR achieves superior zero-shot performance and consistently improves temporal reasoning without any fine-tuning, showcasing the advantage of neuro-symbolic integration in enhancing temporal understanding in large language models. 

---
# Enhancing Agentic RL with Progressive Reward Shaping and Value-based Sampling Policy Optimization 

**Authors**: Zhuoran Zhuang, Ye Chen, Jianghao Su, Chao Luo, Luhui Liu, Xia Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2512.07478)  

**Abstract**: Large Language Models (LLMs) empowered with Tool-Integrated Reasoning (TIR) can iteratively plan, call external tools, and integrate returned information to solve complex, long-horizon reasoning tasks. Agentic Reinforcement Learning (Agentic RL) optimizes such models over full tool-interaction trajectories, but two key challenges hinder effectiveness: (1) Sparse, non-instructive rewards, such as binary 0-1 verifiable signals, provide limited guidance for intermediate steps and slow convergence; (2) Gradient degradation in Group Relative Policy Optimization (GRPO), where identical rewards within a rollout group yield zero advantage, reducing sample efficiency and destabilizing training. To address these challenges, we propose two complementary techniques: Progressive Reward Shaping (PRS) and Value-based Sampling Policy Optimization (VSPO). PRS is a curriculum-inspired reward design that introduces dense, stage-wise feedback - encouraging models to first master parseable and properly formatted tool calls, then optimize for factual correctness and answer quality. We instantiate PRS for short-form QA (with a length-aware BLEU to fairly score concise answers) and long-form QA (with LLM-as-a-Judge scoring to prevent reward hacking). VSPO is an enhanced GRPO variant that replaces low-value samples with prompts selected by a task-value metric balancing difficulty and uncertainty, and applies value-smoothing clipping to stabilize gradient updates. Experiments on multiple short-form and long-form QA benchmarks show that PRS consistently outperforms traditional binary rewards, and VSPO achieves superior stability, faster convergence, and higher final performance compared to PPO, GRPO, CISPO, and SFT-only baselines. Together, PRS and VSPO yield LLM-based TIR agents that generalize better across domains. 

---
# Persian-Phi: Efficient Cross-Lingual Adaptation of Compact LLMs via Curriculum Learning 

**Authors**: Amir Mohammad Akhlaghi, Amirhossein Shabani, Mostafa Abdolmaleki, Saeed Reza Kheradpisheh  

**Link**: [PDF](https://arxiv.org/pdf/2512.07454)  

**Abstract**: The democratization of AI is currently hindered by the immense computational costs required to train Large Language Models (LLMs) for low-resource languages. This paper presents Persian-Phi, a 3.8B parameter model that challenges the assumption that robust multilingual capabilities require massive model sizes or multilingual baselines. We demonstrate how Microsoft Phi-3 Mini -- originally a monolingual English model -- can be effectively adapted to Persian through a novel, resource-efficient curriculum learning pipeline. Our approach employs a unique "warm-up" stage using bilingual narratives (Tiny Stories) to align embeddings prior to heavy training, followed by continual pretraining and instruction tuning via Parameter-Efficient Fine-Tuning (PEFT). Despite its compact size, Persian-Phi achieves competitive results on Open Persian LLM Leaderboard in HuggingFace. Our findings provide a validated, scalable framework for extending the reach of state-of-the-art LLMs to underrepresented languages with minimal hardware resources. The Persian-Phi model is publicly available at this https URL. 

---
# FVA-RAG: Falsification-Verification Alignment for Mitigating Sycophantic Hallucinations 

**Authors**: Mayank Ravishankara  

**Link**: [PDF](https://arxiv.org/pdf/2512.07015)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have significantly reduced hallucinations in Large Language Models (LLMs) by grounding responses in external context. However, standard RAG architectures suffer from a critical vulnerability: Retrieval Sycophancy. When presented with a query based on a false premise or a common misconception, vector-based retrievers tend to fetch documents that align with the user's bias rather than objective truth, leading the model to "hallucinate with citations."
In this work, we introduce Falsification-Verification Alignment RAG (FVA-RAG), a framework that shifts the retrieval paradigm from Inductive Verification (seeking support) to Deductive Falsification (seeking disproof). Unlike existing "Self-Correction" methods that rely on internal consistency, FVA-RAG deploys a distinct Adversarial Retrieval Policy that actively generates "Kill Queries"-targeted search terms designed to surface contradictory evidence. We introduce a dual-verification mechanism that explicitly weighs the draft answer against this "Anti-Context." Preliminary experiments on a dataset of common misconceptions demonstrate that FVA-RAG significantly improves robustness against sycophantic hallucinations compared to standard RAG baselines, effectively acting as an inference-time "Red Team" for factual generation. 

---
# Do Large Language Models Truly Understand Cross-cultural Differences? 

**Authors**: Shiwei Guo, Sihang Jiang, Qianxi He, Yanghua Xiao, Jiaqing Liang, Bi Yude, Minggui He, Shimin Tao, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.07075)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated strong performance on multilingual tasks. Given its wide range of applications, cross-cultural understanding capability is a crucial competency. However, existing benchmarks for evaluating whether LLMs genuinely possess this capability suffer from three key limitations: a lack of contextual scenarios, insufficient cross-cultural concept mapping, and limited deep cultural reasoning capabilities. To address these gaps, we propose SAGE, a scenario-based benchmark built via cross-cultural core concept alignment and generative task design, to evaluate LLMs' cross-cultural understanding and reasoning. Grounded in cultural theory, we categorize cross-cultural capabilities into nine dimensions. Using this framework, we curated 210 core concepts and constructed 4530 test items across 15 specific real-world scenarios, organized under four broader categories of cross-cultural situations, following established item design principles. The SAGE dataset supports continuous expansion, and experiments confirm its transferability to other languages. It reveals model weaknesses across both dimensions and scenarios, exposing systematic limitations in cross-cultural reasoning. While progress has been made, LLMs are still some distance away from reaching a truly nuanced cross-cultural understanding. In compliance with the anonymity policy, we include data and code in the supplement materials. In future versions, we will make them publicly available online. 

---
# Large Language Models and Forensic Linguistics: Navigating Opportunities and Threats in the Age of Generative AI 

**Authors**: George Mikros  

**Link**: [PDF](https://arxiv.org/pdf/2512.06922)  

**Abstract**: Large language models (LLMs) present a dual challenge for forensic linguistics. They serve as powerful analytical tools enabling scalable corpus analysis and embedding-based authorship attribution, while simultaneously destabilising foundational assumptions about idiolect through style mimicry, authorship obfuscation, and the proliferation of synthetic texts. Recent stylometric research indicates that LLMs can approximate surface stylistic features yet exhibit detectable differences from human writers, a tension with significant forensic implications. However, current AI-text detection techniques, whether classifier-based, stylometric, or watermarking approaches, face substantial limitations: high false positive rates for non-native English writers and vulnerability to adversarial strategies such as homoglyph substitution. These uncertainties raise concerns under legal admissibility standards, particularly the Daubert and Kumho Tire frameworks. The article concludes that forensic linguistics requires methodological reconfiguration to remain scientifically credible and legally admissible. Proposed adaptations include hybrid human-AI workflows, explainable detection paradigms beyond binary classification, and validation regimes measuring error and bias across diverse populations. The discipline's core insight, i.e., that language reveals information about its producer, remains valid but must accommodate increasingly complex chains of human and machine authorship. 

---
# Prompting-in-a-Series: Psychology-Informed Contents and Embeddings for Personality Recognition With Decoder-Only Models 

**Authors**: Jing Jie Tan, Ban-Hoe Kwan, Danny Wee-Kiat Ng, Yan-Chai Hum, Anissa Mokraoui, Shih-Yu Lo  

**Link**: [PDF](https://arxiv.org/pdf/2512.06991)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various natural language processing tasks. This research introduces a novel "Prompting-in-a-Series" algorithm, termed PICEPR (Psychology-Informed Contents Embeddings for Personality Recognition), featuring two pipelines: (a) Contents and (b) Embeddings. The approach demonstrates how a modularised decoder-only LLM can summarize or generate content, which can aid in classifying or enhancing personality recognition functions as a personality feature extractor and a generator for personality-rich content. We conducted various experiments to provide evidence to justify the rationale behind the PICEPR algorithm. Meanwhile, we also explored closed-source models such as \textit{gpt4o} from OpenAI and \textit{gemini} from Google, along with open-source models like \textit{mistral} from Mistral AI, to compare the quality of the generated content. The PICEPR algorithm has achieved a new state-of-the-art performance for personality recognition by 5-15\% improvement. The work repository and models' weight can be found at this https URL. 

---
# Rhea: Role-aware Heuristic Episodic Attention for Conversational LLMs 

**Authors**: Wanyang Hong, Zhaoning Zhang, Yi Chen, Libo Zhang, Baihui Liu, Linbo Qiao, Zhiliang Tian, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.06869)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable performance on single-turn tasks, yet their effectiveness deteriorates in multi-turn conversations. We define this phenomenon as cumulative contextual decay - a progressive degradation of contextual integrity caused by attention pollution, dilution, and drift. To address this challenge, we propose Rhea (Role-aware Heuristic Episodic Attention), a novel framework that decouples conversation history into two functionally independent memory modules: (1) an Instructional Memory (IM) that persistently stores high-fidelity global constraints via a structural priority mechanism, and (2) an Episodic Memory (EM) that dynamically manages user-model interactions via asymmetric noise control and heuristic context retrieval. During inference, Rhea constructs a high signal-to-noise context by applying its priority attention: selectively integrating relevant episodic information while always prioritizing global instructions. To validate this approach, experiments on multiple multi-turn conversation benchmarks - including MT-Eval and Long-MT-Bench+ - show that Rhea mitigates performance decay and improves overall accuracy by 1.04 points on a 10-point scale (a 16% relative gain over strong baselines). Moreover, Rhea maintains near-perfect instruction fidelity (IAR > 8.1) across long-horizon interactions. These results demonstrate that Rhea provides a principled and effective framework for building more precise, instruction-consistent conversational LLMs. 

---
# An Analysis of Large Language Models for Simulating User Responses in Surveys 

**Authors**: Ziyun Yu, Yiru Zhou, Chen Zhao, Hongyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2512.06874)  

**Abstract**: Using Large Language Models (LLMs) to simulate user opinions has received growing attention. Yet LLMs, especially trained with reinforcement learning from human feedback (RLHF), are known to exhibit biases toward dominant viewpoints, raising concerns about their ability to represent users from diverse demographic and cultural backgrounds. In this work, we examine the extent to which LLMs can simulate human responses to cross-domain survey questions through direct prompting and chain-of-thought prompting. We further propose a claim diversification method CLAIMSIM, which elicits viewpoints from LLM parametric knowledge as contextual input. Experiments on the survey question answering task indicate that, while CLAIMSIM produces more diverse responses, both approaches struggle to accurately simulate users. Further analysis reveals two key limitations: (1) LLMs tend to maintain fixed viewpoints across varying demographic features, and generate single-perspective claims; and (2) when presented with conflicting claims, LLMs struggle to reason over nuanced differences among demographic features, limiting their ability to adapt responses to specific user profiles. 

---
# Training Language Models to Use Prolog as a Tool 

**Authors**: Niklas Mellgren, Peter Schneider-Kamp, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2512.07407)  

**Abstract**: Ensuring reliable tool use is critical for safe agentic AI systems. Language models frequently produce unreliable reasoning with plausible but incorrect solutions that are difficult to verify. To address this, we investigate fine-tuning models to use Prolog as an external tool for verifiable computation. Using Group Relative Policy Optimization (GRPO), we fine-tune Qwen2.5-3B-Instruct on a cleaned GSM8K-Prolog-Prover dataset while varying (i) prompt structure, (ii) reward composition (execution, syntax, semantics, structure), and (iii) inference protocol: single-shot, best-of-N, and two agentic modes where Prolog is invoked internally or independently. Our reinforcement learning approach outperforms supervised fine-tuning, with our 3B model achieving zero-shot MMLU performance comparable to 7B few-shot results. Our findings reveal that: 1) joint tuning of prompt, reward, and inference shapes program syntax and logic; 2) best-of-N with external Prolog verification maximizes accuracy on GSM8K; 3) agentic inference with internal repair yields superior zero-shot generalization on MMLU-Stem and MMLU-Pro. These results demonstrate that grounding model reasoning in formal verification systems substantially improves reliability and auditability for safety-critical applications. The source code for reproducing our experiments is available under this https URL 

---
# Large Language Model-Based Generation of Discharge Summaries 

**Authors**: Tiago Rodrigues, Carla Teixeira Lopes  

**Link**: [PDF](https://arxiv.org/pdf/2512.06812)  

**Abstract**: Discharge Summaries are documents written by medical professionals that detail a patient's visit to a care facility. They contain a wealth of information crucial for patient care, and automating their generation could significantly reduce the effort required from healthcare professionals, minimize errors, and ensure that critical patient information is easily accessible and actionable. In this work, we explore the use of five Large Language Models on this task, from open-source models (Mistral, Llama 2) to proprietary systems (GPT-3, GPT-4, Gemini 1.5 Pro), leveraging MIMIC-III summaries and notes. We evaluate them using exact-match, soft-overlap, and reference-free metrics. Our results show that proprietary models, particularly Gemini with one-shot prompting, outperformed others, producing summaries with the highest similarity to the gold-standard ones. Open-source models, while promising, especially Mistral after fine-tuning, lagged in performance, often struggling with hallucinations and repeated information. Human evaluation by a clinical expert confirmed the practical utility of the summaries generated by proprietary models. Despite the challenges, such as hallucinations and missing information, the findings suggest that LLMs, especially proprietary models, are promising candidates for automatic discharge summary generation as long as data privacy is ensured. 

---
# LLM4SFC: Sequential Function Chart Generation via Large Language Models 

**Authors**: Ofek Glick, Vladimir Tchuiev, Marah Ghoummaid, Michal Moshkovitz, Dotan Di-Castro  

**Link**: [PDF](https://arxiv.org/pdf/2512.06787)  

**Abstract**: While Large Language Models (LLMs) are increasingly used for synthesizing textual PLC programming languages like Structured Text (ST) code, other IEC 61131-3 standard graphical languages like Sequential Function Charts (SFCs) remain underexplored. Generating SFCs is challenging due to graphical nature and ST actions embedded within, which are not directly compatible with standard generation techniques, often leading to non-executable code that is incompatible with industrial tool-chains In this work, we introduce LLM4SFC, the first framework to receive natural-language descriptions of industrial workflows and provide executable SFCs. LLM4SFC is based on three components: (i) A reduced structured representation that captures essential topology and in-line ST and reduced textual verbosity; (ii) Fine-tuning and few-shot retrieval-augmented generation (RAG) for alignment with SFC programming conventions; and (iii) A structured generation approach that prunes illegal tokens in real-time to ensure compliance with the textual format of SFCs. We evaluate LLM4SFC on a dataset of real-world SFCs from automated manufacturing projects, using both open-source and proprietary LLMs. The results show that LLM4SFC reliably generates syntactically valid SFC programs effectively bridging graphical and textual PLC languages, achieving a generation generation success of 75% - 94%, paving the way for automated industrial programming. 

---
# "The Dentist is an involved parent, the bartender is not": Revealing Implicit Biases in QA with Implicit BBQ 

**Authors**: Aarushi Wagh, Saniya Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2512.06732)  

**Abstract**: Existing benchmarks evaluating biases in large language models (LLMs) primarily rely on explicit cues, declaring protected attributes like religion, race, gender by name. However, real-world interactions often contain implicit biases, inferred subtly through names, cultural cues, or traits. This critical oversight creates a significant blind spot in fairness evaluation. We introduce ImplicitBBQ, a benchmark extending the Bias Benchmark for QA (BBQ) with implicitly cued protected attributes across 6 categories. Our evaluation of GPT-4o on ImplicitBBQ illustrates troubling performance disparity from explicit BBQ prompts, with accuracy declining up to 7% in the "sexual orientation" subcategory and consistent decline located across most other categories. This indicates that current LLMs contain implicit biases undetected by explicit benchmarks. ImplicitBBQ offers a crucial tool for nuanced fairness evaluation in NLP. 

---
# From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion LLMs 

**Authors**: Yuchuan Tian, Yuchen Liang, Jiacheng Sun, Shuo Zhang, Guangwen Yang, Yingte Shu, Sibo Fang, Tianyu Guo, Kai Han, Chao Xu, Hanting Chen, Xinghao Chen, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.06776)  

**Abstract**: Large language models (LLMs) excel at generation but dominant autoregressive (AR) decoding is inherently sequential, creating a throughput bottleneck. Diffusion Language Models (DLMs)--especially block-wise variants--enable parallel generation and intra-block bidirectional reasoning, yet training large DLMs from scratch is costly and wastes the knowledge in mature AR checkpoints. Prior "adaptation" attempts either modify logits or randomly grow attention masks to full-sequence diffusion, or simply transplant AR weights into a block-diffusion recipe, leaving a fundamental mismatch between AR causality and block-wise bidirectionality unaddressed. We reframe adaptation as a intra-paradigm path from AR to Block-Diffusion by viewing AR as Block-Diffusion with blocksize=1. Concretely, we design the pathway of adaptation as follows: we use a context-causal attention mask (causal in context, bidirectional only within the active block), an efficient parallel adaptation procedure, an auxiliary AR loss to maximize data utilization and retain pretrained knowledge, and gradual increment of the generation block size. The recipe integrates cleanly with masked block-diffusion and maintains train-inference consistency. Built on these components, NBDiff-7B (Base and Instruct) could inherit the long-context modeling and reasoning capabilities, and achieve state-of-the-art performance among the 7B-class DLMs, delivering strong gains on general-knowledge, math, and code benchmarks over strong baselines. These results demonstrate that principled AR-to-block-diffusion adaptation is an effective and compute-efficient alternative to training DLMs from scratch. Codes: this https URL. 

---
# Becoming Experienced Judges: Selective Test-Time Learning for Evaluators 

**Authors**: Seungyeon Jwa, Daechul Ahn, Reokyoung Kim, Dongyeop Kang, Jonghyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2512.06751)  

**Abstract**: Automatic evaluation with large language models, commonly known as LLM-as-a-judge, is now standard across reasoning and alignment tasks. Despite evaluating many samples in deployment, these evaluators typically (i) treat each case independently, missing the opportunity to accumulate experience, and (ii) rely on a single fixed prompt for all cases, neglecting the need for sample-specific evaluation criteria. We introduce Learning While Evaluating (LWE), a framework that allows evaluators to improve sequentially at inference time without requiring training or validation sets. LWE maintains an evolving meta-prompt that (i) produces sample-specific evaluation instructions and (ii) refines itself through self-generated feedback. Furthermore, we propose Selective LWE, which updates the meta-prompt only on self-inconsistent cases, focusing computation where it matters most. This selective approach retains the benefits of sequential learning while being far more cost-effective. Across two pairwise comparison benchmarks, Selective LWE outperforms strong baselines, empirically demonstrating that evaluators can improve during sequential testing with a simple selective update, learning most from the cases they struggle with. 

---
# Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation 

**Authors**: Chengbing Wang, Yang Zhang, Wenjie Wang, Xiaoyan Zhao, Fuli Feng, Xiangnan He, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2512.06690)  

**Abstract**: Preference alignment has enabled large language models (LLMs) to better reflect human expectations, but current methods mostly optimize for population-level preferences, overlooking individual users. Personalization is essential, yet early approaches-such as prompt customization or fine-tuning-struggle to reason over implicit preferences, limiting real-world effectiveness. Recent "think-then-generate" methods address this by reasoning before response generation. However, they face challenges in long-form generation: their static one-shot reasoning must capture all relevant information for the full response generation, making learning difficult and limiting adaptability to evolving content. To address this issue, we propose FlyThinker, an efficient "think-while-generating" framework for personalized long-form generation. FlyThinker employs a separate reasoning model that generates latent token-level reasoning in parallel, which is fused into the generation model to dynamically guide response generation. This design enables reasoning and generation to run concurrently, ensuring inference efficiency. In addition, the reasoning model is designed to depend only on previous responses rather than its own prior outputs, which preserves training parallelism across different positions-allowing all reasoning tokens for training data to be produced in a single forward pass like standard LLM training, ensuring training efficiency. Extensive experiments on real-world benchmarks demonstrate that FlyThinker achieves better personalized generation while keeping training and inference efficiency. 

---
# Classifying German Language Proficiency Levels Using Large Language Models 

**Authors**: Elias-Leander Ahlers, Witold Brunsmann, Malte Schilling  

**Link**: [PDF](https://arxiv.org/pdf/2512.06483)  

**Abstract**: Assessing language proficiency is essential for education, as it enables instruction tailored to learners needs. This paper investigates the use of Large Language Models (LLMs) for automatically classifying German texts according to the Common European Framework of Reference for Languages (CEFR) into different proficiency levels. To support robust training and evaluation, we construct a diverse dataset by combining multiple existing CEFR-annotated corpora with synthetic data. We then evaluate prompt-engineering strategies, fine-tuning of a LLaMA-3-8B-Instruct model and a probing-based approach that utilizes the internal neural state of the LLM for classification. Our results show a consistent performance improvement over prior methods, highlighting the potential of LLMs for reliable and scalable CEFR classification. 

---
# Mechanistic Interpretability of GPT-2: Lexical and Contextual Layers in Sentiment Analysis 

**Authors**: Amartya Hatua  

**Link**: [PDF](https://arxiv.org/pdf/2512.06681)  

**Abstract**: We present a mechanistic interpretability study of GPT-2 that causally examines how sentiment information is processed across its transformer layers. Using systematic activation patching across all 12 layers, we test the hypothesized two-stage sentiment architecture comprising early lexical detection and mid-layer contextual integration. Our experiments confirm that early layers (0-3) act as lexical sentiment detectors, encoding stable, position specific polarity signals that are largely independent of context. However, all three contextual integration hypotheses: Middle Layer Concentration, Phenomenon Specificity, and Distributed Processing are falsified. Instead of mid-layer specialization, we find that contextual phenomena such as negation, sarcasm, domain shifts etc. are integrated primarily in late layers (8-11) through a unified, non-modular mechanism. These experimental findings provide causal evidence that GPT-2's sentiment computation differs from the predicted hierarchical pattern, highlighting the need for further empirical characterization of contextual integration in large language models. 

---
# Convergence of Outputs When Two Large Language Models Interact in a Multi-Agentic Setup 

**Authors**: Aniruddha Maiti, Satya Nimmagadda, Kartha Veerya Jammuladinne, Niladri Sengupta, Ananya Jana  

**Link**: [PDF](https://arxiv.org/pdf/2512.06256)  

**Abstract**: In this work, we report what happens when two large language models respond to each other for many turns without any outside input in a multi-agent setup. The setup begins with a short seed sentence. After that, each model reads the other's output and generates a response. This continues for a fixed number of steps. We used Mistral Nemo Base 2407 and Llama 2 13B hf. We observed that most conversations start coherently but later fall into repetition. In many runs, a short phrase appears and repeats across turns. Once repetition begins, both models tend to produce similar output rather than introducing a new direction in the conversation. This leads to a loop where the same or similar text is produced repeatedly. We describe this behavior as a form of convergence. It occurs even though the models are large, trained separately, and not given any prompt instructions. To study this behavior, we apply lexical and embedding-based metrics to measure how far the conversation drifts from the initial seed and how similar the outputs of the two models becomes as the conversation progresses. 

---
# Empathy by Design: Aligning Large Language Models for Healthcare Dialogue 

**Authors**: Emre Umucu, Guillermina Solis, Leon Garza, Emilia Rivas, Beatrice Lee, Anantaa Kotal, Aritran Piplai  

**Link**: [PDF](https://arxiv.org/pdf/2512.06097)  

**Abstract**: General-purpose large language models (LLMs) have demonstrated remarkable generative and reasoning capabilities but remain limited in healthcare and caregiving applications due to two key deficiencies: factual unreliability and a lack of empathetic communication. These shortcomings pose significant risks in sensitive contexts where users, particularly non-professionals and caregivers, seek medically relevant guidance or emotional reassurance. To address these challenges, we introduce a Direct Preference Optimization (DPO)-based alignment framework designed to improve factual correctness, semantic coherence, and human-centric qualities such as empathy, politeness, and simplicity in caregiver-patient dialogues. Our approach fine-tunes domain-adapted LLMs using pairwise preference data, where preferred responses reflect supportive and accessible communication styles while rejected ones represent prescriptive or overly technical tones. This direct optimization method aligns model outputs with human preferences more efficiently than traditional reinforcement-learning-based alignment. Empirical evaluations across multiple open and proprietary LLMs show that our DPO-tuned models achieve higher semantic alignment, improved factual accuracy, and stronger human-centric evaluation scores compared to baseline and commercial alternatives such as Google medical dialogue systems. These improvements demonstrate that preference-based alignment offers a scalable and transparent pathway toward developing trustworthy, empathetic, and clinically informed AI assistants for caregiver and healthcare communication. Our open-source code is available at: this https URL 

---
# Automated Data Enrichment using Confidence-Aware Fine-Grained Debate among Open-Source LLMs for Mental Health and Online Safety 

**Authors**: Junyu Mao, Anthony Hills, Talia Tseriotou, Maria Liakata, Aya Shamir, Dan Sayda, Dana Atzil-Slonim, Natalie Djohari, Arpan Mandal, Silke Roth, Pamela Ugwudike, Mahesan Niranjan, Stuart E. Middleton  

**Link**: [PDF](https://arxiv.org/pdf/2512.06227)  

**Abstract**: Real-world indicators are important for improving natural language processing (NLP) tasks such as life events for mental health analysis and risky behaviour for online safety, yet labelling such information in NLP training datasets is often costly and/or difficult given the dynamic nature of such events. This paper compares several LLM-based data enrichment methods and introduces a novel Confidence-Aware Fine-Grained Debate (CFD) framework in which multiple LLM agents simulate human annotators and exchange fine-grained evidence to reach consensus. We describe two new expert-annotated datasets, a mental health Reddit wellbeing dataset and an online safety Facebook sharenting risk dataset. Our CFD framework achieves the most robust data enrichment performance compared to a range of baselines and we show that this type of data enrichment consistently improves downstream tasks. Enriched features incorporated via debate transcripts yield the largest gains, outperforming the non-enriched baseline by 10.1% for the online safety task. 

---
# Policy-based Sentence Simplification: Replacing Parallel Corpora with LLM-as-a-Judge 

**Authors**: Xuanxin Wu, Yuki Arase, Masaaki Nagata  

**Link**: [PDF](https://arxiv.org/pdf/2512.06228)  

**Abstract**: Sentence simplification aims to modify a sentence to make it easier to read and understand while preserving the meaning. Different applications require distinct simplification policies, such as replacing only complex words at the lexical level or rewriting the entire sentence while trading off details for simplicity. However, achieving such policy-driven control remains an open challenge. In this work, we introduce a simple yet powerful approach that leverages Large Language Model-as-a-Judge (LLM-as-a-Judge) to automatically construct policy-aligned training data, completely removing the need for costly human annotation or parallel corpora. Our method enables building simplification systems that adapt to diverse simplification policies. Remarkably, even small-scale open-source LLMs such as Phi-3-mini-3.8B surpass GPT-4o on lexical-oriented simplification, while achieving comparable performance on overall rewriting, as verified by both automatic metrics and human evaluations. The consistent improvements across model families and sizes demonstrate the robustness of our approach. 

---
# Do You Feel Comfortable? Detecting Hidden Conversational Escalation in AI Chatbots 

**Authors**: Jihyung Park, Saleh Afroogh, Junfeng Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2512.06193)  

**Abstract**: Large Language Models (LLM) are increasingly integrated into everyday interactions, serving not only as information assistants but also as emotional companions. Even in the absence of explicit toxicity, repeated emotional reinforcement or affective drift can gradually escalate distress in a form of \textit{implicit harm} that traditional toxicity filters fail to detect. Existing guardrail mechanisms often rely on external classifiers or clinical rubrics that may lag behind the nuanced, real-time dynamics of a developing conversation. To address this gap, we propose GAUGE (Guarding Affective Utterance Generation Escalation), a lightweight, logit-based framework for the real-time detection of hidden conversational escalation. GAUGE measures how an LLM's output probabilistically shifts the affective state of a dialogue. 

---
# ReasonBENCH: Benchmarking the (In)Stability of LLM Reasoning 

**Authors**: Nearchos Potamitis, Lars Klein, Akhil Arora  

**Link**: [PDF](https://arxiv.org/pdf/2512.07795)  

**Abstract**: Large language models (LLMs) are increasingly deployed in settings where reasoning, such as multi-step problem solving and chain-of-thought, is essential. Yet, current evaluation practices overwhelmingly report single-run accuracy while ignoring the intrinsic uncertainty that naturally arises from stochastic decoding. This omission creates a blind spot because practitioners cannot reliably assess whether a method's reported performance is stable, reproducible, or cost-consistent. We introduce ReasonBENCH, the first benchmark designed to quantify the underlying instability in LLM reasoning. ReasonBENCH provides (i) a modular evaluation library that standardizes reasoning frameworks, models, and tasks, (ii) a multi-run protocol that reports statistically reliable metrics for both quality and cost, and (iii) a public leaderboard to encourage variance-aware reporting. Across tasks from different domains, we find that the vast majority of reasoning strategies and models exhibit high instability. Notably, even strategies with similar average performance can display confidence intervals up to four times wider, and the top-performing methods often incur higher and less stable costs. Such instability compromises reproducibility across runs and, consequently, the reliability of reported performance. To better understand these dynamics, we further analyze the impact of prompts, model families, and scale on the trade-off between solve rate and stability. Our results highlight reproducibility as a critical dimension for reliable LLM reasoning and provide a foundation for future reasoning methods and uncertainty quantification techniques. ReasonBENCH is publicly available at this https URL . 

---
# Block Sparse Flash Attention 

**Authors**: Daniel Ohayon, Itay Lamprecht, Itay Hubara, Israel Cohen, Daniel Soudry, Noam Elata  

**Link**: [PDF](https://arxiv.org/pdf/2512.07011)  

**Abstract**: Modern large language models increasingly require long contexts for reasoning and multi-document tasks, but attention's quadratic complexity creates a severe computational bottleneck. We present Block-Sparse FlashAttention (BSFA), a drop-in replacement that accelerates long-context inference while preserving model quality. Unlike methods that predict importance before computing scores, BSFA computes exact query-key similarities to select the top-k most important value blocks for each query. By comparing per-block maximum scores against calibrated thresholds, we skip approximately 50% of the computation and memory transfers for pruned blocks. Our training-free approach requires only a one-time threshold calibration on a small dataset to learn the per-layer and per-head attention score distributions. We provide a CUDA kernel implementation that can be used as a drop-in replacement for FlashAttention. On Llama-3.1-8B, BSFA achieves up to 1.10x speedup on real-world reasoning benchmarks and up to 1.24x for needle-in-a-haystack retrieval tasks while maintaining above 99% baseline accuracy, with certain configurations even improving accuracy by focusing on the most relevant content, substantially outperforming existing sparse attention methods. The implementation is available at this https URL 

---
# Generating Storytelling Images with Rich Chains-of-Reasoning 

**Authors**: Xiujie Song, Qi Jia, Shota Watanabe, Xiaoyi Pang, Ruijie Chen, Mengyue Wu, Kenny Q. Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2512.07198)  

**Abstract**: An image can convey a compelling story by presenting rich, logically connected visual clues. These connections form Chains-of-Reasoning (CoRs) within the image, enabling viewers to infer events, causal relationships, and other information, thereby understanding the underlying story. In this paper, we focus on these semantically rich images and define them as Storytelling Images. Such images have diverse applications beyond illustration creation and cognitive screening, leveraging their ability to convey multi-layered information visually and inspire active interpretation. However, due to their complex semantic nature, Storytelling Images are inherently challenging to create, and thus remain relatively scarce. To address this challenge, we introduce the Storytelling Image Generation task, which explores how generative AI models can be leveraged to create such images. Specifically, we propose a two-stage pipeline, StorytellingPainter, which combines the creative reasoning abilities of Large Language Models (LLMs) with the visual synthesis capabilities of Text-to-Image (T2I) models to generate Storytelling Images. Alongside this pipeline, we develop a dedicated evaluation framework comprising three main evaluators: a Semantic Complexity Evaluator, a KNN-based Diversity Evaluator and a Story-Image Alignment Evaluator. Given the critical role of story generation in the Storytelling Image Generation task and the performance disparity between open-source and proprietary LLMs, we further explore tailored training strategies to reduce this gap, resulting in a series of lightweight yet effective models named Mini-Storytellers. Experimental results demonstrate the feasibility and effectiveness of our approaches. The code is available at this https URL. 

---
# MMDuet2: Enhancing Proactive Interaction of Video MLLMs with Multi-Turn Reinforcement Learning 

**Authors**: Yueqian Wang, Songxiang Liu, Disong Wang, Nuo Xu, Guanglu Wan, Huishuai Zhang, Dongyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.06810)  

**Abstract**: Recent advances in video multimodal large language models (Video MLLMs) have significantly enhanced video understanding and multi-modal interaction capabilities. While most existing systems operate in a turn-based manner where the model can only reply after user turns, proactively deciding when to reply during video playback presents a promising yet challenging direction for real-time applications. In this work, we propose a novel text-to-text approach to proactive interaction, where the model autonomously determines whether to respond or remain silent at each turn based on dialogue history and visual context up to current frame of an streaming video. To overcome difficulties in previous methods such as manually tuning response decision thresholds and annotating precise reply times, we introduce a multi-turn RL based training method that encourages timely and accurate responses without requiring precise response time annotations. We train our model MMDuet2 on a dataset of 52k videos with two types of dialogues via SFT and RL. Experimental results demonstrate that MMDuet2 outperforms existing proactive Video MLLM baselines in response timing and quality, achieving state-of-the-art performance on the ProactiveVideoQA benchmark. 

---
# Less Is More for Multi-Step Logical Reasoning of LLM Generalisation Under Rule Removal, Paraphrasing, and Compression 

**Authors**: Qiming Bao, Xiaoxuan Fu  

**Link**: [PDF](https://arxiv.org/pdf/2512.06393)  

**Abstract**: Large language models (LLMs) excel across many natural language tasks, yet their generalisation to structural perturbations in logical contexts remains poorly understood. We introduce a controlled evaluation framework that probes reasoning reliability through four targeted stress tests: (1) rule deletion, removing either redundant or essential rules from a multi-step inference chain; (2) contradictory evidence injection; (3) logic-preserving rewrites generated through several families of equivalence laws (contrapositive, double negation, implication, De Morgan, identity, and commutativity); and (4) multi-law equivalence stacking that introduces 2-5 simultaneous logical transformations.
Across three representative model families: BERT, Qwen2, and LLaMA-like models. Our experiments reveal a strikingly consistent pattern: all models achieve perfect accuracy on the base tasks and remain fully generalise to redundant rule deletion and all equivalence-based rewrites (single or multi-law), but fail sharply under essential rule deletion (dropping to 25% accuracy) and collapse completely in the presence of explicit contradictions (0% accuracy). These results demonstrate that LLMs possess stable invariance to semantic-preserving logical transformations, yet remain fundamentally brittle to missing or conflicting evidence. Our framework provides a clean diagnostic tool for isolating such reasoning failure modes and highlights persistent gaps in the logical generalisation abilities of current LLMs. 

---
# ProAgent: Harnessing On-Demand Sensory Contexts for Proactive LLM Agent Systems 

**Authors**: Bufang Yang, Lilin Xu, Liekang Zeng, Yunqi Guo, Siyang Jiang, Wenrui Lu, Kaiwei Liu, Hancheng Xiang, Xiaofan Jiang, Guoliang Xing, Zhenyu Yan  

**Link**: [PDF](https://arxiv.org/pdf/2512.06721)  

**Abstract**: Large Language Model (LLM) agents are emerging to transform daily life. However, existing LLM agents primarily follow a reactive paradigm, relying on explicit user instructions to initiate services, which increases both physical and cognitive workload. In this paper, we propose ProAgent, the first end-to-end proactive agent system that harnesses massive sensory contexts and LLM reasoning to deliver proactive assistance. ProAgent first employs a proactive-oriented context extraction approach with on-demand tiered perception to continuously sense the environment and derive hierarchical contexts that incorporate both sensory and persona cues. ProAgent then adopts a context-aware proactive reasoner to map these contexts to user needs and tool calls, providing proactive assistance. We implement ProAgent on Augmented Reality (AR) glasses with an edge server and extensively evaluate it on a real-world testbed, a public dataset, and through a user study. Results show that ProAgent achieves up to 33.4% higher proactive prediction accuracy, 16.8% higher tool-calling F1 score, and notable improvements in user satisfaction over state-of-the-art baselines, marking a significant step toward proactive assistants. A video demonstration of ProAgent is available at this https URL. 

---
# Less Is More, but Where? Dynamic Token Compression via LLM-Guided Keyframe Prior 

**Authors**: Yulin Li, Haokun Gui, Ziyang Fan, Junjie Wang, Bin Kang, Bin Chen, Zhuotao Tian  

**Link**: [PDF](https://arxiv.org/pdf/2512.06866)  

**Abstract**: Recent advances in Video Large Language Models (VLLMs) have achieved remarkable video understanding capabilities, yet face critical efficiency bottlenecks due to quadratic computational growth with lengthy visual token sequences of long videos. While existing keyframe sampling methods can improve temporal modeling efficiency, additional computational cost is introduced before feature encoding, and the binary frame selection paradigm is found suboptimal. Therefore, in this work, we propose Dynamic Token compression via LLM-guided Keyframe prior (DyToK), a training-free paradigm that enables dynamic token compression by harnessing VLLMs' inherent attention mechanisms. Our analysis reveals that VLLM attention layers naturally encoding query-conditioned keyframe priors, by which DyToK dynamically adjusts per-frame token retention ratios, prioritizing semantically rich frames while suppressing redundancies. Extensive experiments demonstrate that DyToK achieves state-of-the-art efficiency-accuracy tradeoffs. DyToK shows plug-and-play compatibility with existing compression methods, such as VisionZip and FastV, attaining 4.3x faster inference while preserving accuracy across multiple VLLMs, such as LLaVA-OneVision and Qwen2.5-VL. Code is available at this https URL . 

---
# A Fast and Effective Solution to the Problem of Look-ahead Bias in LLMs 

**Authors**: Humzah Merchant, Bradford Levy  

**Link**: [PDF](https://arxiv.org/pdf/2512.06607)  

**Abstract**: Applying LLMs to predictive tasks in finance is challenging due to look-ahead bias resulting from their training on long time-series data. This precludes the backtests typically employed in finance since retraining frontier models from scratch with a specific knowledge cutoff is prohibitive. In this paper, we introduce a fast, effective, and low-cost alternative. Our method guides generation at inference time by adjusting the logits of a large base model using a pair of smaller, specialized models -- one fine-tuned on information to be forgotten and another on information to be retained. We demonstrate that our method effectively removes both verbatim and semantic knowledge, corrects biases, and outperforms prior methods. 

---
# AI-Generated Compromises for Coalition Formation: Modeling, Simulation, and a Textual Case Study 

**Authors**: Eyal Briman, Ehud Shapiro, Nimrod Talmon  

**Link**: [PDF](https://arxiv.org/pdf/2512.05983)  

**Abstract**: The challenge of finding compromises between agent proposals is fundamental to AI sub-fields such as argumentation, mediation, and negotiation. Building on this tradition, Elkind et al. (2021) introduced a process for coalition formation that seeks majority-supported proposals preferable to the status quo, using a metric space where each agent has an ideal point. The crucial step in this iterative process involves identifying compromise proposals around which agent coalitions can unite. How to effectively find such compromise proposals, however, remains an open question. We address this gap by formalizing a holistic model that encompasses agent bounded rationality and uncertainty and developing AI models to generate such compromise proposals. We focus on the domain of collaboratively writing text documents -- e.g., to enable the democratic creation of a community constitution. We apply NLP (Natural Language Processing) techniques and utilize LLMs (Large Language Models) to create a semantic metric space for text and develop algorithms to suggest suitable compromise points. To evaluate the effectiveness of our algorithms, we simulate various coalition formation processes and demonstrate the potential of AI to facilitate large-scale democratic text editing, such as collaboratively drafting a constitution, an area where traditional tools are limited. 

---
# LLM-Upgraded Graph Reinforcement Learning for Carbon-Aware Job Scheduling in Smart Manufacturing 

**Authors**: Zhiying Yang, Fang Liu, Wei Zhang, Xin Lou, Malcolm Yoke Hean Low, Boon Ping Gan  

**Link**: [PDF](https://arxiv.org/pdf/2512.06351)  

**Abstract**: This paper presents \textsc{Luca}, a \underline{l}arge language model (LLM)-\underline{u}pgraded graph reinforcement learning framework for \underline{c}arbon-\underline{a}ware flexible job shop scheduling. \textsc{Luca} addresses the challenges of dynamic and sustainable scheduling in smart manufacturing systems by integrating a graph neural network and an LLM, guided by a carefully designed in-house prompting strategy, to produce a fused embedding that captures both structural characteristics and contextual semantics of the latest scheduling state. This expressive embedding is then processed by a deep reinforcement learning policy network, which generates real-time scheduling decisions optimized for both makespan and carbon emission objectives. To support sustainability goals, \textsc{Luca} incorporates a dual-objective reward function that encourages both energy efficiency and scheduling timeliness. Experimental results on both synthetic and public datasets demonstrate that \textsc{Luca} consistently outperforms comparison algorithms. For instance, on the synthetic dataset, it achieves an average of 4.1\% and up to 12.2\% lower makespan compared to the best-performing comparison algorithm while maintaining the same emission level. On public datasets, additional gains are observed for both makespan and emission. These results demonstrate that \textsc{Luca} is effective and practical for carbon-aware scheduling in smart manufacturing. 

---
# An Index-based Approach for Efficient and Effective Web Content Extraction 

**Authors**: Yihan Chen, Benfeng Xu, Xiaorui Wang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2512.06641)  

**Abstract**: As web agents (e.g., Deep Research) routinely consume massive volumes of web pages to gather and analyze information, LLM context management -- under large token budgets and low signal density -- emerges as a foundational, high-importance, and technically challenging problem for agentic and RAG pipelines. Existing solutions for extracting relevant content are inadequate: generative extraction models suffer from high latency, rule-based heuristics lack adaptability, and chunk-and-rerank methods are blind to webpage structure. To overcome these issues, we introduce Index-based Web Content Extraction to reframe the extraction process from slow, token-by-token generation into a highly efficient, discriminative task of index prediction, achieving both effectiveness and efficiency. We partition HTML into structure-aware, addressable segments, and extract only the positional indices of content relevant to a given query. This method decouples extraction latency from content length, enabling rapid, query-relevant extraction. We first evaluate our method as a post-retrieval processing component within an RAG QA system and find that it improves QA accuracy. Then we directly measure its match rate with the target content in two scenarios: main content extraction (ME) and query-relevant extraction (QE). Experimental results show that our method outperforms existing works in both accuracy and speed, effectively bridging the gap between LLMs and the vast webpages. 

---
# From Show Programmes to Data: Designing a Workflow to Make Performing Arts Ephemera Accessible Through Language Models 

**Authors**: Clarisse Bardiot, Pierre-Carl Langlais, Bernard Jacquemin, Jacob Hart, Antonios Lagarias, Nicolas Foucault, Aurélie Lemaître-Legargeant, Jeanne Fras  

**Link**: [PDF](https://arxiv.org/pdf/2512.07452)  

**Abstract**: Many heritage institutions hold extensive collections of theatre programmes, which remain largely underused due to their complex layouts and lack of structured metadata. In this paper, we present a workflow for transforming such documents into structured data using a combination of multimodal large language models (LLMs), an ontology-based reasoning model, and a custom extension of the Linked Art framework. We show how vision-language models can accurately parse and transcribe born-digital and digitised programmes, achieving over 98% of correct extraction. To overcome the challenges of semantic annotation, we train a reasoning model (POntAvignon) using reinforcement learning with both formal and semantic rewards. This approach enables automated RDF triple generation and supports alignment with existing knowledge graphs. Through a case study based on the Festival d'Avignon corpus, we demonstrate the potential for large-scale, ontology-driven analysis of performing arts data. Our results open new possibilities for interoperable, explainable, and sustainable computational theatre historiography. 

---
# Enhanced Multimodal Video Retrieval System: Integrating Query Expansion and Cross-modal Temporal Event Retrieval 

**Authors**: Van-Thinh Vo, Minh-Khoi Nguyen, Minh-Huy Tran, Anh-Quan Nguyen-Tran, Duy-Tan Nguyen, Khanh-Loi Nguyen, Anh-Minh Phan  

**Link**: [PDF](https://arxiv.org/pdf/2512.06334)  

**Abstract**: Multimedia information retrieval from videos remains a challenging problem. While recent systems have advanced multimodal search through semantic, object, and OCR queries - and can retrieve temporally consecutive scenes - they often rely on a single query modality for an entire sequence, limiting robustness in complex temporal contexts. To overcome this, we propose a cross-modal temporal event retrieval framework that enables different query modalities to describe distinct scenes within a sequence. To determine decision thresholds for scene transition and slide change adaptively, we build Kernel Density Gaussian Mixture Thresholding (KDE-GMM) algorithm, ensuring optimal keyframe selection. These extracted keyframes act as compact, high-quality visual exemplars that retain each segment's semantic essence, improving retrieval precision and efficiency. Additionally, the system incorporates a large language model (LLM) to refine and expand user queries, enhancing overall retrieval performance. The proposed system's effectiveness and robustness were demonstrated through its strong results in the Ho Chi Minh AI Challenge 2025. 

---
# Towards Efficient Hypergraph and Multi-LLM Agent Recommender Systems 

**Authors**: Tendai Mukande, Esraa Ali, Annalina Caputo, Ruihai Dong, Noel OConnor  

**Link**: [PDF](https://arxiv.org/pdf/2512.06590)  

**Abstract**: Recommender Systems (RSs) have become the cornerstone of various applications such as e-commerce and social media platforms. The evolution of RSs is paramount in the digital era, in which personalised user experience is tailored to the user's preferences. Large Language Models (LLMs) have sparked a new paradigm - generative retrieval and recommendation. Despite their potential, generative RS methods face issues such as hallucination, which degrades the recommendation performance, and high computational cost in practical scenarios. To address these issues, we introduce HGLMRec, a novel Multi-LLM agent-based RS that incorporates a hypergraph encoder designed to capture complex, multi-behaviour relationships between users and items. The HGLMRec model retrieves only the relevant tokens during inference, reducing computational overhead while enriching the retrieval context. Experimental results show performance improvement by HGLMRec against state-of-the-art baselines at lower computational cost. 

---
# Reformulate, Retrieve, Localize: Agents for Repository-Level Bug Localization 

**Authors**: Genevieve Caumartin, Glaucia Melo  

**Link**: [PDF](https://arxiv.org/pdf/2512.07022)  

**Abstract**: Bug localization remains a critical yet time-consuming challenge in large-scale software repositories. Traditional information retrieval-based bug localization (IRBL) methods rely on unchanged bug descriptions, which often contain noisy information, leading to poor retrieval accuracy. Recent advances in large language models (LLMs) have improved bug localization through query reformulation, yet the effect on agent performance remains unexplored. In this study, we investigate how an LLM-powered agent can improve file-level bug localization via lightweight query reformulation and summarization. We first employ an open-source, non-fine-tuned LLM to extract key information from bug reports, such as identifiers and code snippets, and reformulate queries pre-retrieval. Our agent then orchestrates BM25 retrieval using these preprocessed queries, automating localization workflow at scale. Using the best-performing query reformulation technique, our agent achieves 35% better ranking in first-file retrieval than our BM25 baseline and up to +22% file retrieval performance over SWE-agent. 

---
# Large Causal Models from Large Language Models 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2512.07796)  

**Abstract**: We introduce a new paradigm for building large causal models (LCMs) that exploits the enormous potential latent in today's large language models (LLMs). We describe our ongoing experiments with an implemented system called DEMOCRITUS (Decentralized Extraction of Manifold Ontologies of Causal Relations Integrating Topos Universal Slices) aimed at building, organizing, and visualizing LCMs that span disparate domains extracted from carefully targeted textual queries to LLMs. DEMOCRITUS is methodologically distinct from traditional narrow domain and hypothesis centered causal inference that builds causal models from experiments that produce numerical data. A high-quality LLM is used to propose topics, generate causal questions, and extract plausible causal statements from a diverse range of domains. The technical challenge is then to take these isolated, fragmented, potentially ambiguous and possibly conflicting causal claims, and weave them into a coherent whole, converting them into relational causal triples and embedding them into a LCM. Addressing this technical challenge required inventing new categorical machine learning methods, which we can only briefly summarize in this paper, as it is focused more on the systems side of building DEMOCRITUS. We describe the implementation pipeline for DEMOCRITUS comprising of six modules, examine its computational cost profile to determine where the current bottlenecks in scaling the system to larger models. We describe the results of using DEMOCRITUS over a wide range of domains, spanning archaeology, biology, climate change, economics, medicine and technology. We discuss the limitations of the current DEMOCRITUS system, and outline directions for extending its capabilities. 

---
# LocalSearchBench: Benchmarking Agentic Search in Real-World Local Life Services 

**Authors**: Hang He, Chuhuai Yue, Chengqi Dong, Mingxue Tian, Zhenfeng Liu, Jiajun Chai, Xiaohan Wang, Yufei Zhang, Qun Liao, Guojun Yin, Wei Lin, Chengcheng Wan, Haiying Sun, Ting Su  

**Link**: [PDF](https://arxiv.org/pdf/2512.07436)  

**Abstract**: Recent advances in large reasoning models (LRMs) have enabled agentic search systems to perform complex multi-step reasoning across multiple sources. However, most studies focus on general information retrieval and rarely explores vertical domains with unique challenges. In this work, we focus on local life services and introduce LocalSearchBench, which encompass diverse and complex business scenarios. Real-world queries in this domain are often ambiguous and require multi-hop reasoning across merchants and products, remaining challenging and not fully addressed. As the first comprehensive benchmark for agentic search in local life services, LocalSearchBench includes over 150,000 high-quality entries from various cities and business types. We construct 300 multi-hop QA tasks based on real user queries, challenging agents to understand questions and retrieve information in multiple steps. We also developed LocalPlayground, a unified environment integrating multiple tools for agent interaction. Experiments show that even state-of-the-art LRMs struggle on LocalSearchBench: the best model (DeepSeek-V3.1) achieves only 34.34% correctness, and most models have issues with completeness (average 77.33%) and faithfulness (average 61.99%). This highlights the need for specialized benchmarks and domain-specific agent training in local life services. Code, Benchmark, and Leaderboard are available at this http URL. 

---
# RL-MTJail: Reinforcement Learning for Automated Black-Box Multi-Turn Jailbreaking of Large Language Models 

**Authors**: Xiqiao Xiong, Ouxiang Li, Zhuo Liu, Moxin Li, Wentao Shi, Fuli Feng, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2512.07761)  

**Abstract**: Large language models are vulnerable to jailbreak attacks, threatening their safe deployment in real-world applications. This paper studies black-box multi-turn jailbreaks, aiming to train attacker LLMs to elicit harmful content from black-box models through a sequence of prompt-output interactions. Existing approaches typically rely on single turn optimization, which is insufficient for learning long-term attack strategies. To bridge this gap, we formulate the problem as a multi-turn reinforcement learning task, directly optimizing the harmfulness of the final-turn output as the outcome reward. To mitigate sparse supervision and promote long-term attack strategies, we propose two heuristic process rewards: (1) controlling the harmfulness of intermediate outputs to prevent triggering the black-box model's rejection mechanisms, and (2) maintaining the semantic relevance of intermediate outputs to avoid drifting into irrelevant content. Experimental results on multiple benchmarks show consistently improved attack success rates across multiple models, highlighting the effectiveness of our approach. The code is available at this https URL. Warning: This paper contains examples of harmful content. 

---
# How Do LLMs Fail In Agentic Scenarios? A Qualitative Analysis of Success and Failure Scenarios of Various LLMs in Agentic Simulations 

**Authors**: JV Roig  

**Link**: [PDF](https://arxiv.org/pdf/2512.07497)  

**Abstract**: We investigate how large language models (LLMs) fail when operating as autonomous agents with tool-use capabilities. Using the Kamiwaza Agentic Merit Index (KAMI) v0.1 benchmark, we analyze 900 execution traces from three representative models - Granite 4 Small, Llama 4 Maverick, and DeepSeek V3.1 - across filesystem, text extraction, CSV analysis, and SQL scenarios. Rather than focusing on aggregate scores, we perform fine-grained, per-trial behavioral analysis to surface the strategies that enable successful multi-step tool execution and the recurrent failure modes that undermine reliability. Our findings show that model scale alone does not predict agentic robustness: Llama 4 Maverick (400B) performs only marginally better than Granite 4 Small (32B) in some uncertainty-driven tasks, while DeepSeek V3.1's superior reliability derives primarily from post-training reinforcement learning rather than architecture or size. Across models, we identify four recurring failure archetypes: premature action without grounding, over-helpfulness that substitutes missing entities, vulnerability to distractor-induced context pollution, and fragile execution under load. These patterns highlight the need for agentic evaluation methods that emphasize interactive grounding, recovery behavior, and environment-aware adaptation, suggesting that reliable enterprise deployment requires not just stronger models but deliberate training and design choices that reinforce verification, constraint discovery, and adherence to source-of-truth data. 

---
# ContextualSHAP : Enhancing SHAP Explanations Through Contextual Language Generation 

**Authors**: Latifa Dwiyanti, Sergio Ryan Wibisono, Hidetaka Nambo  

**Link**: [PDF](https://arxiv.org/pdf/2512.07178)  

**Abstract**: Explainable Artificial Intelligence (XAI) has become an increasingly important area of research, particularly as machine learning models are deployed in high-stakes domains. Among various XAI approaches, SHAP (SHapley Additive exPlanations) has gained prominence due to its ability to provide both global and local explanations across different machine learning models. While SHAP effectively visualizes feature importance, it often lacks contextual explanations that are meaningful for end-users, especially those without technical backgrounds. To address this gap, we propose a Python package that extends SHAP by integrating it with a large language model (LLM), specifically OpenAI's GPT, to generate contextualized textual explanations. This integration is guided by user-defined parameters (such as feature aliases, descriptions, and additional background) to tailor the explanation to both the model context and the user perspective. We hypothesize that this enhancement can improve the perceived understandability of SHAP explanations. To evaluate the effectiveness of the proposed package, we applied it in a healthcare-related case study and conducted user evaluations involving real end-users. The results, based on Likert-scale surveys and follow-up interviews, indicate that the generated explanations were perceived as more understandable and contextually appropriate compared to visual-only outputs. While the findings are preliminary, they suggest that combining visualization with contextualized text may support more user-friendly and trustworthy model explanations. 

---
# Do Persona-Infused LLMs Affect Performance in a Strategic Reasoning Game? 

**Authors**: John Licato, Stephen Steinle, Brayden Hollis  

**Link**: [PDF](https://arxiv.org/pdf/2512.06867)  

**Abstract**: Although persona prompting in large language models appears to trigger different styles of generated text, it is unclear whether these translate into measurable behavioral differences, much less whether they affect decision-making in an adversarial strategic environment that we provide as open-source. We investigate the impact of persona prompting on strategic performance in PERIL, a world-domination board game. Specifically, we compare the effectiveness of persona-derived heuristic strategies to those chosen manually. Our findings reveal that certain personas associated with strategic thinking improve game performance, but only when a mediator is used to translate personas into heuristic values. We introduce this mediator as a structured translation process, inspired by exploratory factor analysis, that maps LLM-generated inventory responses into heuristics. Results indicate our method enhances heuristic reliability and face validity compared to directly inferred heuristics, allowing us to better study the effect of persona types on decision making. These insights advance our understanding of how persona prompting influences LLM-based decision-making and propose a heuristic generation method that applies psychometric principles to LLMs. 

---
# DoVer: Intervention-Driven Auto Debugging for LLM Multi-Agent Systems 

**Authors**: Ming Ma, Jue Zhang, Fangkai Yang, Yu Kang, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.06749)  

**Abstract**: Large language model (LLM)-based multi-agent systems are challenging to debug because failures often arise from long, branching interaction traces. The prevailing practice is to leverage LLMs for log-based failure localization, attributing errors to a specific agent and step. However, this paradigm has two key limitations: (i) log-only debugging lacks validation, producing untested hypotheses, and (ii) single-step or single-agent attribution is often ill-posed, as we find that multiple distinct interventions can independently repair the failed task. To address the first limitation, we introduce DoVer, an intervention-driven debugging framework, which augments hypothesis generation with active verification through targeted interventions (e.g., editing messages, altering plans). For the second limitation, rather than evaluating on attribution accuracy, we focus on measuring whether the system resolves the failure or makes quantifiable progress toward task success, reflecting a more outcome-oriented view of debugging. Within the Magnetic-One agent framework, on the datasets derived from GAIA and AssistantBench, DoVer flips 18-28% of failed trials into successes, achieves up to 16% milestone progress, and validates or refutes 30-60% of failure hypotheses. DoVer also performs effectively on a different dataset (GSMPlus) and agent framework (AG2), where it recovers 49% of failed trials. These results highlight intervention as a practical mechanism for improving reliability in agentic systems and open opportunities for more robust, scalable debugging methods for LLM-based multi-agent systems. Project website and code will be available at this https URL. 

---
# ClinNoteAgents: An LLM Multi-Agent System for Predicting and Interpreting Heart Failure 30-Day Readmission from Clinical Notes 

**Authors**: Rongjia Zhou, Chengzhuo Li, Carl Yang, Jiaying Lu  

**Link**: [PDF](https://arxiv.org/pdf/2512.07081)  

**Abstract**: Heart failure (HF) is one of the leading causes of rehospitalization among older adults in the United States. Although clinical notes contain rich, detailed patient information and make up a large portion of electronic health records (EHRs), they remain underutilized for HF readmission risk analysis. Traditional computational models for HF readmission often rely on expert-crafted rules, medical thesauri, and ontologies to interpret clinical notes, which are typically written under time pressure and may contain misspellings, abbreviations, and domain-specific jargon. We present ClinNoteAgents, an LLM-based multi-agent framework that transforms free-text clinical notes into (1) structured representations of clinical and social risk factors for association analysis and (2) clinician-style abstractions for HF 30-day readmission prediction. We evaluate ClinNoteAgents on 3,544 notes from 2,065 patients (readmission rate=35.16%), demonstrating strong performance in extracting risk factors from free-text, identifying key contributing factors, and predicting readmission risk. By reducing reliance on structured fields and minimizing manual annotation and model training, ClinNoteAgents provides a scalable and interpretable approach to note-based HF readmission risk modeling in data-limited healthcare systems. 

---
# JT-DA: Enhancing Data Analysis with Tool-Integrated Table Reasoning Large Language Models 

**Authors**: Ce Chi, Xing Wang, Zhendong Wang, Xiaofan Liu, Ce Li, Zhiyan Song, Chen Zhao, Kexin Yang, Boshen Shi, Jingjing Yang, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2512.06859)  

**Abstract**: In this work, we present JT-DA-8B (JiuTian Data Analyst 8B), a specialized large language model designed for complex table reasoning tasks across diverse real-world scenarios. To address the lack of high-quality supervision in tabular reasoning scenarios, we construct a comprehensive and diverse training corpus with 34 well-defined table reasoning tasks, by aggregating 29 public table QA datasets and 3 million tables. An automatic pipeline is proposed to generate realistic multi-step analytical tasks involving reasoning patterns. The model is trained upon open-source JT-Coder-8B model, an 8B-parameter decoder-only foundation model trained from scratch. In the training stage, we leverage LLM-based scoring and workflow-aligned filtering to distill high-quality, table-centric data. Both supervised fine-tuning (SFT) and Reinforcement learning (RL) are adopted to optimize our model. Afterwards, a four-stage table reasoning workflow is proposed, including table preprocessing, table sensing, tool-integrated reasoning, and prompt engineering, to improve model interpretability and execution accuracy. Experimental results show that JT-DA-8B achieves strong performance in various table reasoning tasks, demonstrating the effectiveness of data-centric generation and workflow-driven optimization. 

---
# Stochasticity in Agentic Evaluations: Quantifying Inconsistency with Intraclass Correlation 

**Authors**: Zairah Mustahsan, Abel Lim, Megna Anand, Saahil Jain, Bryan McCann  

**Link**: [PDF](https://arxiv.org/pdf/2512.06710)  

**Abstract**: As large language models become components of larger agentic systems, evaluation reliability becomes critical: unreliable sub-agents introduce brittleness into downstream system behavior. Yet current evaluation practice, reporting a single accuracy number from a single run, obscures the variance underlying these results, making it impossible to distinguish genuine capability improvements from lucky sampling. We propose adopting Intraclass Correlation Coefficient (ICC), a metric from measurement science, to characterize this variance. ICC decomposes observed variance into between-query variance (task difficulty) and within-query variance (agent inconsistency), highlighting whether reported results reflect true capability or measurement noise. We evaluated on GAIA (Levels 1-3, measuring agentic capabilities across varying reasoning complexity) and FRAMES (measuring retrieval and factuality across multiple documents). We found that ICC varies dramatically with task structure, with reasoning and retrieval tasks (FRAMES) exhibit ICC=0.4955-0.7118 across models, and agentic tasks (GAIA) exhibiting ICC=0.304-0.774 across models. For sub-agent replacement decisions in agentic systems, accuracy improvements are only trustworthy if ICC also improves. We demonstrate that ICC converges by n=8-16 trials for structured tasks and n>=32 for complex reasoning, enabling practitioners to set evidence-based resampling budgets. We recommend reporting accuracy alongside ICC and within-query variance as standard practice, and propose updated Evaluation Cards capturing these metrics. By making evaluation stability visible, we aim to transform agentic benchmarking from opaque leaderboard competition to trustworthy experimental science. Our code is open-sourced at this https URL. 

---
# DaGRPO: Rectifying Gradient Conflict in Reasoning via Distinctiveness-Aware Group Relative Policy Optimization 

**Authors**: Xuan Xie, Xuan Wang, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.06337)  

**Abstract**: The evolution of Large Language Models (LLMs) has catalyzed a paradigm shift from superficial instruction following to rigorous long-horizon reasoning. While Group Relative Policy Optimization (GRPO) has emerged as a pivotal mechanism for eliciting such post-training reasoning capabilities due to its exceptional performance, it remains plagued by significant training instability and poor sample efficiency. We theoretically identify the root cause of these issues as the lack of distinctiveness within on-policy rollouts: for routine queries, highly homogeneous samples induce destructive gradient conflicts; whereas for hard queries, the scarcity of valid positive samples results in ineffective optimization. To bridge this gap, we propose Distinctiveness-aware Group Relative Policy Optimization (DaGRPO). DaGRPO incorporates two core mechanisms: (1) Sequence-level Gradient Rectification, which utilizes fine-grained scoring to dynamically mask sample pairs with low distinctiveness, thereby eradicating gradient conflicts at the source; and (2) Off-policy Data Augmentation, which introduces high-quality anchors to recover training signals for challenging tasks. Extensive experiments across 9 mathematical reasoning and out-of-distribution (OOD) generalization benchmarks demonstrate that DaGRPO significantly surpasses existing SFT, GRPO, and hybrid baselines, achieving new state-of-the-art performance (e.g., a +4.7% average accuracy gain on math benchmarks). Furthermore, in-depth analysis confirms that DaGRPO effectively mitigates gradient explosion and accelerates the emergence of long-chain reasoning capabilities. 

---
# In-Context and Few-Shots Learning for Forecasting Time Series Data based on Large Language Models 

**Authors**: Saroj Gopali, Bipin Chhetri, Deepika Giri, Sima Siami-Namini, Akbar Siami Namin  

**Link**: [PDF](https://arxiv.org/pdf/2512.07705)  

**Abstract**: Existing data-driven approaches in modeling and predicting time series data include ARIMA (Autoregressive Integrated Moving Average), Transformer-based models, LSTM (Long Short-Term Memory) and TCN (Temporal Convolutional Network). These approaches, and in particular deep learning-based models such as LSTM and TCN, have shown great results in predicting time series data. With the advancement of leveraging pre-trained foundation models such as Large Language Models (LLMs) and more notably Google's recent foundation model for time series data, {\it TimesFM} (Time Series Foundation Model), it is of interest to investigate whether these foundation models have the capability of outperforming existing modeling approaches in analyzing and predicting time series data.
This paper investigates the performance of using LLM models for time series data prediction. We investigate the in-context learning methodology in the training of LLM models that are specific to the underlying application domain. More specifically, the paper explores training LLMs through in-context, zero-shot and few-shot learning and forecasting time series data with OpenAI {\tt o4-mini} and Gemini 2.5 Flash Lite, as well as the recent Google's Transformer-based TimesFM, a time series-specific foundation model, along with two deep learning models, namely TCN and LSTM networks. The findings indicate that TimesFM has the best overall performance with the lowest RMSE value (0.3023) and the competitive inference time (266 seconds). Furthermore, OpenAI's o4-mini also exhibits a good performance based on Zero Shot learning.
These findings highlight pre-trained time series foundation models as a promising direction for real-time forecasting, enabling accurate and scalable deployment with minimal model adaptation. 

---
# When Large Language Models Do Not Work: Online Incivility Prediction through Graph Neural Networks 

**Authors**: Zihan Chen, Lanyu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.07684)  

**Abstract**: Online incivility has emerged as a widespread and persistent problem in digital communities, imposing substantial social and psychological burdens on users. Although many platforms attempt to curb incivility through moderation and automated detection, the performance of existing approaches often remains limited in both accuracy and efficiency. To address this challenge, we propose a Graph Neural Network (GNN) framework for detecting three types of uncivil behavior (i.e., toxicity, aggression, and personal attacks) within the English Wikipedia community. Our model represents each user comment as a node, with textual similarity between comments defining the edges, allowing the network to jointly learn from both linguistic content and relational structures among comments. We also introduce a dynamically adjusted attention mechanism that adaptively balances nodal and topological features during information aggregation. Empirical evaluations demonstrate that our proposed architecture outperforms 12 state-of-the-art Large Language Models (LLMs) across multiple metrics while requiring significantly lower inference cost. These findings highlight the crucial role of structural context in detecting online incivility and address the limitations of text-only LLM paradigms in behavioral prediction. All datasets and comparative outputs will be publicly available in our repository to support further research and reproducibility. 

---
# Going All-In on LLM Accuracy: Fake Prediction Markets, Real Confidence Signals 

**Authors**: Michael Todasco  

**Link**: [PDF](https://arxiv.org/pdf/2512.05998)  

**Abstract**: Large language models are increasingly used to evaluate other models, yet these judgments typically lack any representation of confidence. This pilot study tests whether framing an evaluation task as a betting game (a fictional prediction market with its own LLM currency) improves forecasting accuracy and surfaces calibrated confidence signals. We generated 100 math and logic questions with verifiable answers. Six Baseline models (three current-generation, three prior-generation) answered all items. Three Predictor models then forecasted, for each question-baseline pair, if the baseline would answer correctly. Each predictor completed matched runs in two conditions: Control (simple correct/incorrect predictions) and Incentive (predictions plus wagers of 1-100,000 LLMCoin under even odds, starting from a 1,000,000 LLMCoin bankroll). Across 5,400 predictions per condition, Incentive runs showed modestly higher accuracy (81.5% vs. 79.1%, p = .089, d = 0.86) and significantly faster learning across rounds (12.0 vs. 2.9 percentage-point improvement from Round 1 to Round 4, p = .011). Most notably, stake size tracked confidence. "Whale" bets of 40,000+ coins were correct ~99% of the time, while small bets (<1,000 coins) showed only ~74% accuracy. The key finding is not that fictional money makes models smarter; accuracy gains were modest and did not reach statistical significance (p = .089) in this pilot. Rather, the betting mechanic created a legible confidence signal absent from binary yes/no outputs. This suggests that simple financial framing may help transform LLMs into risk-aware forecasters, making their internal beliefs visible and usable. The protocol offers a foundation for future work for meta-evaluation systems and what may become LLM-to-LLM prediction markets. 

---
# GENIUS: An Agentic AI Framework for Autonomous Design and Execution of Simulation Protocols 

**Authors**: Mohammad Soleymanibrojeni, Roland Aydin, Diego Guedes-Sobrinho, Alexandre C. Dias, Maurício J. Piotrowski, Wolfgang Wenzel, Celso Ricardo Caldeira Rêgo  

**Link**: [PDF](https://arxiv.org/pdf/2512.06404)  

**Abstract**: Predictive atomistic simulations have propelled materials discovery, yet routine setup and debugging still demand computer specialists. This know-how gap limits Integrated Computational Materials Engineering (ICME), where state-of-the-art codes exist but remain cumbersome for non-experts. We address this bottleneck with GENIUS, an AI-agentic workflow that fuses a smart Quantum ESPRESSO knowledge graph with a tiered hierarchy of large language models supervised by a finite-state error-recovery machine. Here we show that GENIUS translates free-form human-generated prompts into validated input files that run to completion on $\approx$80% of 295 diverse benchmarks, where 76% are autonomously repaired, with success decaying exponentially to a 7% baseline. Compared with LLM-only baselines, GENIUS halves inference costs and virtually eliminates hallucinations. The framework democratizes electronic-structure DFT simulations by intelligently automating protocol generation, validation, and repair, opening large-scale screening and accelerating ICME design loops across academia and industry worldwide. 

---
# An AI-Powered Autonomous Underwater System for Sea Exploration and Scientific Research 

**Authors**: Hamad Almazrouei, Mariam Al Nasseri, Maha Alzaabi  

**Link**: [PDF](https://arxiv.org/pdf/2512.07652)  

**Abstract**: Traditional sea exploration faces significant challenges due to extreme conditions, limited visibility, and high costs, resulting in vast unexplored ocean regions. This paper presents an innovative AI-powered Autonomous Underwater Vehicle (AUV) system designed to overcome these limitations by automating underwater object detection, analysis, and reporting. The system integrates YOLOv12 Nano for real-time object detection, a Convolutional Neural Network (CNN) (ResNet50) for feature extraction, Principal Component Analysis (PCA) for dimensionality reduction, and K-Means++ clustering for grouping marine objects based on visual characteristics. Furthermore, a Large Language Model (LLM) (GPT-4o Mini) is employed to generate structured reports and summaries of underwater findings, enhancing data interpretation. The system was trained and evaluated on a combined dataset of over 55,000 images from the DeepFish and OzFish datasets, capturing diverse Australian marine environments. Experimental results demonstrate the system's capability to detect marine objects with a mAP@0.5 of 0.512, a precision of 0.535, and a recall of 0.438. The integration of PCA effectively reduced feature dimensionality while preserving 98% variance, facilitating K-Means clustering which successfully grouped detected objects based on visual similarities. The LLM integration proved effective in generating insightful summaries of detections and clusters, supported by location data. This integrated approach significantly reduces the risks associated with human diving, increases mission efficiency, and enhances the speed and depth of underwater data analysis, paving the way for more effective scientific research and discovery in challenging marine environments. 

---
# VulnLLM-R: Specialized Reasoning LLM with Agent Scaffold for Vulnerability Detection 

**Authors**: Yuzhou Nie, Hongwei Li, Chengquan Guo, Ruizhe Jiang, Zhun Wang, Bo Li, Dawn Song, Wenbo Guo  

**Link**: [PDF](https://arxiv.org/pdf/2512.07533)  

**Abstract**: We propose VulnLLM-R, the~\emph{first specialized reasoning LLM} for vulnerability detection. Our key insight is that LLMs can reason about program states and analyze the potential vulnerabilities, rather than simple pattern matching. This can improve the model's generalizability and prevent learning shortcuts. However, SOTA reasoning LLMs are typically ultra-large, closed-source, or have limited performance in vulnerability detection. To address this, we propose a novel training recipe with specialized data selection, reasoning data generation, reasoning data filtering and correction, and testing-phase optimization. Using our proposed methodology, we train a reasoning model with seven billion parameters. Through extensive experiments on SOTA datasets across Python, C/C++, and Java, we show that VulnLLM-R has superior effectiveness and efficiency than SOTA static analysis tools and both open-source and commercial large reasoning models. We further conduct a detailed ablation study to validate the key designs in our training recipe. Finally, we construct an agent scaffold around our model and show that it outperforms CodeQL and AFL++ in real-world projects. Our agent further discovers a set of zero-day vulnerabilities in actively maintained repositories. This work represents a pioneering effort to enable real-world, project-level vulnerability detection using AI agents powered by specialized reasoning models. The code is available at~\href{this https URL}{github}. 

---
# Understanding LLM Agent Behaviours via Game Theory: Strategy Recognition, Biases and Multi-Agent Dynamics 

**Authors**: Trung-Kiet Huynh, Duy-Minh Dao-Sy, Thanh-Bang Cao, Phong-Hao Le, Hong-Dan Nguyen, Phu-Quy Nguyen-Lam, Minh-Luan Nguyen-Vo, Hong-Phat Pham, Phu-Hoa Pham, Thien-Kim Than, Chi-Nguyen Tran, Huy Tran, Gia-Thoai Tran-Le, Alessio Buscemi, Le Hong Trang, Anh Han  

**Link**: [PDF](https://arxiv.org/pdf/2512.07462)  

**Abstract**: As Large Language Models (LLMs) increasingly operate as autonomous decision-makers in interactive and multi-agent systems and human societies, understanding their strategic behaviour has profound implications for safety, coordination, and the design of AI-driven social and economic infrastructures. Assessing such behaviour requires methods that capture not only what LLMs output, but the underlying intentions that guide their decisions. In this work, we extend the FAIRGAME framework to systematically evaluate LLM behaviour in repeated social dilemmas through two complementary advances: a payoff-scaled Prisoners Dilemma isolating sensitivity to incentive magnitude, and an integrated multi-agent Public Goods Game with dynamic payoffs and multi-agent histories. These environments reveal consistent behavioural signatures across models and languages, including incentive-sensitive cooperation, cross-linguistic divergence and end-game alignment toward defection. To interpret these patterns, we train traditional supervised classification models on canonical repeated-game strategies and apply them to FAIRGAME trajectories, showing that LLMs exhibit systematic, model- and language-dependent behavioural intentions, with linguistic framing at times exerting effects as strong as architectural differences. Together, these findings provide a unified methodological foundation for auditing LLMs as strategic agents and reveal systematic cooperation biases with direct implications for AI governance, collective decision-making, and the design of safe multi-agent systems. 

---
# Do LLMs Trust the Code They Write? 

**Authors**: Francisco Ribeiro, Claudio Spiess, Prem Devanbu, Sarah Nadi  

**Link**: [PDF](https://arxiv.org/pdf/2512.07404)  

**Abstract**: Despite the effectiveness of large language models (LLMs) for code generation, they often output incorrect code. One reason is that model output probabilities are often not well-correlated with correctness, and reflect only the final output of the generation process. Inspired by findings that LLMs internally encode concepts like truthfulness, this paper explores if LLMs similarly represent code correctness. Specifically, we identify a correctness representation inside LLMs by contrasting the hidden states between pairs of correct and incorrect code for the same programming tasks. By experimenting on four LLMs, we show that exploiting this extracted correctness representation outperforms standard log-likelihood ranking, as well as verbalized model confidence. Furthermore, we explore how this internal correctness signal can be used to select higher-quality code samples, without requiring test execution. Ultimately, this work demonstrates how leveraging internal representations can enhance code generation systems and make LLMs more reliable, thus improving confidence in automatically generated code. 

---
# START: Spatial and Textual Learning for Chart Understanding 

**Authors**: Zhuoming Liu, Xiaofeng Gao, Feiyang Niu, Qiaozi Gao, Liu Liu, Robinson Piramuthu  

**Link**: [PDF](https://arxiv.org/pdf/2512.07186)  

**Abstract**: Chart understanding is crucial for deploying multimodal large language models (MLLMs) in real-world scenarios such as analyzing scientific papers and technical reports. Unlike natural images, charts pair a structured visual layout (spatial property) with an underlying data representation (textual property) -- grasping both is essential for precise, fine-grained chart reasoning. Motivated by this observation, we propose START, the Spatial and Textual learning for chART understanding. Specifically, we introduce (i) chart-element grounding and (ii) chart-to-code generation to strengthen an MLLM's understanding of both chart visual layout and data details. To facilitate spatial and textual learning, we propose the START-Dataset generated with a novel data-generation pipeline that first leverages an MLLM to translate real chart images into executable chart code, recovering the underlying data representation while preserving the visual distribution of real-world charts. We then evolve the code with a Large Language Model (LLM) to ascertain the positions of chart elements that capture the chart's visual structure, addressing challenges that existing methods cannot handle. To evaluate a model's ability to understand chart spatial structures, we propose the Chart Spatial understanding Benchmark (CS-Bench), filling a critical gap in comprehensive chart understanding evaluation. Leveraging spatial and textual learning, START delivers consistent gains across model sizes and benchmarks over the base models and surpasses prior state-of-the-art by a clear margin. Code, data and models will be publicly available. 

---
# RisConFix: LLM-based Automated Repair of Risk-Prone Drone Configurations 

**Authors**: Liping Han, Tingting Nie, Le Yu, Mingzhe Hu, Tao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2512.07122)  

**Abstract**: Flight control software is typically designed with numerous configurable parameters governing multiple functionalities, enabling flexible adaptation to mission diversity and environmental uncertainty. Although developers and manufacturers usually provide recommendations for these parameters to ensure safe and stable operations, certain combinations of parameters with recommended values may still lead to unstable flight behaviors, thereby degrading the drone's robustness. To this end, we propose a Large Language Model (LLM) based approach for real-time repair of risk-prone configurations (named RisConFix) that degrade drone robustness. RisConFix continuously monitors the drone's operational state and automatically triggers a repair mechanism once abnormal flight behaviors are detected. The repair mechanism leverages an LLM to analyze relationships between configuration parameters and flight states, and then generates corrective parameter updates to restore flight stability. To ensure the validity of the updated configuration, RisConFix operates as an iterative process; it continuously monitors the drone's flight state and, if an anomaly persists after applying an update, automatically triggers the next repair cycle. We evaluated RisConFix through a case study of ArduPilot (with 1,421 groups of misconfigurations). Experimental results show that RisConFix achieved a best repair success rate of 97% and an optimal average number of repairs of 1.17, demonstrating its capability to effectively and efficiently repair risk-prone configurations in real time. 

---
# Latency-Response Theory Model: Evaluating Large Language Models via Response Accuracy and Chain-of-Thought Length 

**Authors**: Zhiyu Xu, Jia Liu, Yixin Wang, Yuqi Gu  

**Link**: [PDF](https://arxiv.org/pdf/2512.07019)  

**Abstract**: The proliferation of Large Language Models (LLMs) necessitates valid evaluation methods to provide guidance for both downstream applications and actionable future improvements. The Item Response Theory (IRT) model with Computerized Adaptive Testing has recently emerged as a promising framework for evaluating LLMs via their response accuracy. Beyond simple response accuracy, LLMs' chain of thought (CoT) lengths serve as a vital indicator of their reasoning ability. To leverage the CoT length information to assist the evaluation of LLMs, we propose the Latency-Response Theory (LaRT) model, which jointly models both the response accuracy and CoT length by introducing a key correlation parameter between the latent ability and the latent speed. We derive an efficient stochastic approximation Expectation-Maximization algorithm for parameter estimation. We establish rigorous identifiability results for the latent ability and latent speed parameters to ensure the statistical validity of their estimation. Through both theoretical asymptotic analyses and simulation studies, we demonstrate LaRT's advantages over IRT in terms of superior estimation accuracy and shorter confidence intervals for latent trait estimation. To evaluate LaRT in real data, we collect responses from diverse LLMs on popular benchmark datasets. We find that LaRT yields different LLM rankings than IRT and outperforms IRT across multiple key evaluation metrics including predictive power, item efficiency, ranking validity, and LLM evaluation efficiency. Code and data are available at this https URL. 

---
# ThinkTrap: Denial-of-Service Attacks against Black-box LLM Services via Infinite Thinking 

**Authors**: Yunzhe Li, Jianan Wang, Hongzi Zhu, James Lin, Shan Chang, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2512.07086)  

**Abstract**: Large Language Models (LLMs) have become foundational components in a wide range of applications, including natural language understanding and generation, embodied intelligence, and scientific discovery. As their computational requirements continue to grow, these models are increasingly deployed as cloud-based services, allowing users to access powerful LLMs via the Internet. However, this deployment model introduces a new class of threat: denial-of-service (DoS) attacks via unbounded reasoning, where adversaries craft specially designed inputs that cause the model to enter excessively long or infinite generation loops. These attacks can exhaust backend compute resources, degrading or denying service to legitimate users. To mitigate such risks, many LLM providers adopt a closed-source, black-box setting to obscure model internals. In this paper, we propose ThinkTrap, a novel input-space optimization framework for DoS attacks against LLM services even in black-box environments. The core idea of ThinkTrap is to first map discrete tokens into a continuous embedding space, then undertake efficient black-box optimization in a low-dimensional subspace exploiting input sparsity. The goal of this optimization is to identify adversarial prompts that induce extended or non-terminating generation across several state-of-the-art LLMs, achieving DoS with minimal token overhead. We evaluate the proposed attack across multiple commercial, closed-source LLM services. Our results demonstrate that, even far under the restrictive request frequency limits commonly enforced by these platforms, typically capped at ten requests per minute (10 RPM), the attack can degrade service throughput to as low as 1% of its original capacity, and in some cases, induce complete service failure. 

---
# The Geometry of Persona: Disentangling Personality from Reasoning in Large Language Models 

**Authors**: Zhixiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.07092)  

**Abstract**: Background: The deployment of personalized Large Language Models (LLMs) is currently constrained by the stability-plasticity dilemma. Prevailing alignment methods, such as Supervised Fine-Tuning (SFT), rely on stochastic weight updates that often incur an "alignment tax" -- degrading general reasoning capabilities.
Methods: We propose the Soul Engine, a framework based on the Linear Representation Hypothesis, which posits that personality traits exist as orthogonal linear subspaces. We introduce SoulBench, a dataset constructed via dynamic contextual sampling. Using a dual-head architecture on a frozen Qwen-2.5 base, we extract disentangled personality vectors without modifying the backbone weights.
Results: Our experiments demonstrate three breakthroughs. First, High-Precision Profiling: The model achieves a Mean Squared Error (MSE) of 0.011 against psychological ground truth. Second, Geometric Orthogonality: T-SNE visualization confirms that personality manifolds are distinct and continuous, allowing for "Zero-Shot Personality Injection" that maintains original model intelligence. Third, Deterministic Steering: We achieve robust control over behavior via vector arithmetic, validated through extensive ablation studies.
Conclusion: This work challenges the necessity of fine-tuning for personalization. By transitioning from probabilistic prompting to deterministic latent intervention, we provide a mathematically rigorous foundation for safe, controllable AI personalization. 

---
# BabelCoder: Agentic Code Translation with Specification Alignment 

**Authors**: Fazle Rabbi, Soumit Kanti Saha, Tri Minh Triet Pham, Song Wang, Jinqiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.06902)  

**Abstract**: As software systems evolve, developers increasingly work across multiple programming languages and often face the need to migrate code from one language to another. While automatic code translation offers a promising solution, it has long remained a challenging task. Recent advancements in Large Language Models (LLMs) have shown potential for this task, yet existing approaches remain limited in accuracy and fail to effectively leverage contextual and structural cues within the code. Prior work has explored translation and repair mechanisms, but lacks a structured, agentic framework where multiple specialized agents collaboratively improve translation quality. In this work, we introduce BabelCoder, an agentic framework that performs code translation by decomposing the task into specialized agents for translation, testing, and refinement, each responsible for a specific aspect such as generating code, validating correctness, or repairing errors. We evaluate BabelCoder on four benchmark datasets and compare it against four state-of-the-art baselines. BabelCoder outperforms existing methods by 0.5%-13.5% in 94% of cases, achieving an average accuracy of 94.16%. 

---
# Leveraging LLMs to support co-evolution between definitions and instances of textual DSLs 

**Authors**: Weixing Zhang, Regina Hebig, Daniel Strüber  

**Link**: [PDF](https://arxiv.org/pdf/2512.06836)  

**Abstract**: Software languages evolve over time for various reasons, such as the addition of new features. When the language's grammar definition evolves, textual instances that originally conformed to the grammar become outdated. For DSLs in a model-driven engineering context, there exists a plethora of techniques to co-evolve models with the evolving metamodel. However, these techniques are not geared to support DSLs with a textual syntax -- applying them to textual language definitions and instances may lead to the loss of information from the original instances, such as comments and layout information, which are valuable for software comprehension and maintenance. This study explores the potential of Large Language Model (LLM)-based solutions in achieving grammar and instance co-evolution, with attention to their ability to preserve auxiliary information when directly processing textual instances. By applying two advanced language models, Claude-3.5 and GPT-4o, and conducting experiments across seven case languages, we evaluated the feasibility and limitations of this approach. Our results indicate a good ability of the considered LLMs for migrating textual instances in small-scale cases with limited instance size, which are representative of a subset of cases encountered in practice. In addition, we observe significant challenges with the scalability of LLM-based solutions to larger instances, leading to insights that are useful for informing future research. 

---
# Optimal and Diffusion Transports in Machine Learning 

**Authors**: Gabriel Peyré  

**Link**: [PDF](https://arxiv.org/pdf/2512.06797)  

**Abstract**: Several problems in machine learning are naturally expressed as the design and analysis of time-evolving probability distributions. This includes sampling via diffusion methods, optimizing the weights of neural networks, and analyzing the evolution of token distributions across layers of large language models. While the targeted applications differ (samples, weights, tokens), their mathematical descriptions share a common structure. A key idea is to switch from the Eulerian representation of densities to their Lagrangian counterpart through vector fields that advect particles. This dual view introduces challenges, notably the non-uniqueness of Lagrangian vector fields, but also opportunities to craft density evolutions and flows with favorable properties in terms of regularity, stability, and computational tractability. This survey presents an overview of these methods, with emphasis on two complementary approaches: diffusion methods, which rely on stochastic interpolation processes and underpin modern generative AI, and optimal transport, which defines interpolation by minimizing displacement cost. We illustrate how both approaches appear in applications ranging from sampling, neural network optimization, to modeling the dynamics of transformers for large language models. 

---
# Formal that "Floats" High: Formal Verification of Floating Point Arithmetic 

**Authors**: Hansa Mohanty, Vaisakh Naduvodi Viswambharan, Deepak Narayan Gadde  

**Link**: [PDF](https://arxiv.org/pdf/2512.06850)  

**Abstract**: Formal verification of floating-point arithmetic remains challenging due to non-linear arithmetic behavior and the tight coupling between control and datapath logic. Existing approaches often rely on high-level C models for equivalence checking against Register Transfer Level (RTL) designs, but this introduces abstraction gaps, translation overhead, and limits scalability at the RTL level. To address these challenges, this paper presents a scalable methodology for verifying floating-point arithmetic using direct RTL-to-RTL model checking against a golden reference model. The approach adopts a divide-and conquer strategy that decomposes verification into modular stages, each captured by helper assertions and lemmas that collectively prove a main correctness theorem. Counterexample (CEX)-guided refinement is used to iteratively localize and resolve implementation defects, while targeted fault injection validates the robustness of the verification process against precision-critical datapath errors. To assess scalability and practicality, the methodology is extended with agentic AI-based formal property generation, integrating large language model (LLM)-driven automation with Human-in-the-Loop (HITL) refinement. Coverage analysis evaluates the effectiveness of the approach by comparing handwritten and AI-generated properties in both RTL-to-RTL model checking and standalone RTL verification settings. Results show that direct RTL-to-RTL model checking achieves higher coverage efficiency and requires fewer assertions than standalone verification, especially when combined with AI-generated properties refined through HITL guidance. 

---
# From Description to Score: Can LLMs Quantify Vulnerabilities? 

**Authors**: Sima Jafarikhah, Daniel Thompson, Eva Deans, Hossein Siadati, Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.06781)  

**Abstract**: Manual vulnerability scoring, such as assigning Common Vulnerability Scoring System (CVSS) scores, is a resource-intensive process that is often influenced by subjective interpretation. This study investigates the potential of general-purpose large language models (LLMs), namely ChatGPT, Llama, Grok, DeepSeek, and Gemini, to automate this process by analyzing over 31{,}000 recent Common Vulnerabilities and Exposures (CVE) entries. The results show that LLMs substantially outperform the baseline on certain metrics (e.g., \textit{Availability Impact}), while offering more modest gains on others (e.g., \textit{Attack Complexity}). Moreover, model performance varies across both LLM families and individual CVSS metrics, with ChatGPT-5 attaining the highest precision. Our analysis reveals that LLMs tend to misclassify many of the same CVEs, and ensemble-based meta-classifiers only marginally improve performance. Further examination shows that CVE descriptions often lack critical context or contain ambiguous phrasing, which contributes to systematic misclassifications. These findings underscore the importance of enhancing vulnerability descriptions and incorporating richer contextual details to support more reliable automated reasoning and alleviate the growing backlog of CVEs awaiting triage. 

---
# Towards Small Language Models for Security Query Generation in SOC Workflows 

**Authors**: Saleha Muzammil, Rahul Reddy, Vishal Kamalakrishnan, Hadi Ahmadi, Wajih Ul Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2512.06660)  

**Abstract**: Analysts in Security Operations Centers routinely query massive telemetry streams using Kusto Query Language (KQL). Writing correct KQL requires specialized expertise, and this dependency creates a bottleneck as security teams scale. This paper investigates whether Small Language Models (SLMs) can enable accurate, cost-effective natural-language-to-KQL translation for enterprise security. We propose a three-knob framework targeting prompting, fine-tuning, and architecture design. First, we adapt existing NL2KQL framework for SLMs with lightweight retrieval and introduce error-aware prompting that addresses common parser failures without increasing token count. Second, we apply LoRA fine-tuning with rationale distillation, augmenting each NLQ-KQL pair with a brief chain-of-thought explanation to transfer reasoning from a teacher model while keeping the SLM compact. Third, we propose a two-stage architecture that uses an SLM for candidate generation and a low-cost LLM judge for schema-aware refinement and selection. We evaluate nine models (five SLMs and four LLMs) across syntax correctness, semantic accuracy, table selection, and filter precision, alongside latency and token cost. On Microsoft's NL2KQL Defender Evaluation dataset, our two-stage approach achieves 0.987 syntax and 0.906 semantic accuracy. We further demonstrate generalizability on Microsoft Sentinel data, reaching 0.964 syntax and 0.831 semantic accuracy. These results come at up to 10x lower token cost than GPT-5, establishing SLMs as a practical, scalable foundation for natural-language querying in security operations. 

---
# BEACON: A Unified Behavioral-Tactical Framework for Explainable Cybercrime Analysis with Large Language Models 

**Authors**: Arush Sachdeva, Rajendraprasad Saravanan, Gargi Sarkar, Kavita Vemuri, Sandeep Kumar Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2512.06555)  

**Abstract**: Cybercrime increasingly exploits human cognitive biases in addition to technical vulnerabilities, yet most existing analytical frameworks focus primarily on operational aspects and overlook psychological manipulation. This paper proposes BEACON, a unified dual-dimension framework that integrates behavioral psychology with the tactical lifecycle of cybercrime to enable structured, interpretable, and scalable analysis of cybercrime. We formalize six psychologically grounded manipulation categories derived from Prospect Theory and Cialdini's principles of persuasion, alongside a fourteen-stage cybercrime tactical lifecycle spanning reconnaissance to final impact. A single large language model is fine-tuned using parameter-efficient learning to perform joint multi-label classification across both psychological and tactical dimensions while simultaneously generating human-interpretable explanations. Experiments conducted on a curated dataset of real-world and synthetically augmented cybercrime narratives demonstrate a 20 percent improvement in overall classification accuracy over the base model, along with substantial gains in reasoning quality measured using ROUGE and BERTScore. The proposed system enables automated decomposition of unstructured victim narratives into structured behavioral and operational intelligence, supporting improved cybercrime investigation, case linkage, and proactive scam detection. 

---
# A-3PO: Accelerating Asynchronous LLM Training with Staleness-aware Proximal Policy Approximation 

**Authors**: Xiaocan Li, Shiliang Wu, Zheng Shen  

**Link**: [PDF](https://arxiv.org/pdf/2512.06547)  

**Abstract**: Decoupled loss has been a successful reinforcement learning (RL) algorithm to deal with the high data staleness under the asynchronous RL setting. Decoupled loss improves coupled-loss style of algorithms' (e.g., PPO, GRPO) learning stability by introducing a proximal policy to decouple the off-policy corrections (importance weight) from the controlling policy updates (trust region). However, the proximal policy requires an extra forward pass through the network at each training step, creating a computational bottleneck for large language models. We observe that since the proximal policy only serves as a trust region anchor between the behavior and target policies, we can approximate it through simple interpolation without explicit computation. We call this approach A-3PO (APproximated Proximal Policy Optimization). A-3PO eliminates this overhead, reducing training time by 18% while maintaining comparable performance. Code & off-the-shelf example are available at: this https URL 

---
# RLAX: Large-Scale, Distributed Reinforcement Learning for Large Language Models on TPUs 

**Authors**: Runlong Zhou, Lefan Zhang, Shang-Chen Wu, Kelvin Zou, Hanzhi Zhou, Ke Ye, Yihao Feng, Dong Yin, Alex Guillen Garcia, Dmytro Babych, Rohit Chatterjee, Matthew Hopkins, Xiang Kong, Chang Lan, Lezhi Li, Yiping Ma, Daniele Molinari, Senyu Tong, Yanchao Sun, Thomas Voice, Jianyu Wang, Chong Wang, Simon Wang, Floris Weers, Yechen Xu, Guolin Yin, Muyang Yu, Yi Zhang, Zheng Zhou, Danyang Zhuo, Ruoming Pang, Cheng Leong  

**Link**: [PDF](https://arxiv.org/pdf/2512.06392)  

**Abstract**: Reinforcement learning (RL) has emerged as the de-facto paradigm for improving the reasoning capabilities of large language models (LLMs). We have developed RLAX, a scalable RL framework on TPUs. RLAX employs a parameter-server architecture. A master trainer periodically pushes updated model weights to the parameter server while a fleet of inference workers pull the latest weights and generates new rollouts. We introduce a suite of system techniques to enable scalable and preemptible RL for a diverse set of state-of-art RL algorithms. To accelerate convergence and improve model quality, we have devised new dataset curation and alignment techniques. Large-scale evaluations show that RLAX improves QwQ-32B's pass@8 accuracy by 12.8% in just 12 hours 48 minutes on 1024 v5p TPUs, while remaining robust to preemptions during training. 

---
# Beyond Token-level Supervision: Unlocking the Potential of Decoding-based Regression via Reinforcement Learning 

**Authors**: Ming Chen, Sheng Tang, Rong-Xi Tan, Ziniu Li, Jiacheng Chen, Ke Xue, Chao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2512.06533)  

**Abstract**: Decoding-based regression, which reformulates regression as a sequence generation task, has emerged as a promising paradigm of applying large language models for numerical prediction. However, its progress is hindered by the misalignment between discrete token-level objectives (e.g., cross-entropy) and continuous numerical values. Existing approaches relying on token-level constraints often fail to capture the global magnitude of the target value, limiting their precision and generalization. In this paper, we propose to unlock the potential of decoding-based regression via Reinforcement Learning (RL). We formulate the generation process as a Markov Decision Process, utilizing sequence-level rewards to enforce global numerical coherence. Extensive experiments on tabular regression and code metric regression demonstrate that our method (specifically with ReMax and GRPO) consistently outperforms both state-of-the-art token-level baselines and traditional regression heads, showing the superiority of introducing sequence-level signals. Our analysis further reveals that RL significantly enhances sampling efficiency and predictive precision, establishing decoding-based regression as a robust and accurate paradigm for general-purpose numerical prediction. 

---
# Vec-LUT: Vector Table Lookup for Parallel Ultra-Low-Bit LLM Inference on Edge Devices 

**Authors**: Xiangyu Li, Chengyu Yin, Weijun Wang, Jianyu Wei, Ting Cao, Yunxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.06443)  

**Abstract**: Large language models (LLMs) are increasingly deployed on edge devices. To meet strict resource constraints, real-world deployment has pushed LLM quantization from 8-bit to 4-bit, 2-bit, and now 1.58-bit. Combined with lookup table (LUT)-based inference, CPUs run these ultra-low-bit LLMs even faster than NPUs, opening new opportunities for ubiquitous on-device intelligence.
However, this paper identifies that LUT-based inference underutilizes memory bandwidth during parallel inference, which is required for prefilling, test-time scaling, and other multi-token scenarios. The root cause is the scalar LUT paradigm, which performs repetitive and non-contiguous memory accesses for each token.
To solve the issue, we propose vector LUT, a new lookup paradigm that constructs a unified LUT across parallel tokens, and performs a single $1 \rightarrow N$ lookup per index. To realize it efficiently, we further introduce (1) Vector LUT-Centric Tensor Layout, and (2) Cache-Aware Streamed Lookup techniques. Evaluations on 5 edge devices across 3 LLMs show that Vec-LUT outperforms state-of-the-art baselines by up to $4.2\times$. Our implementation is integrated into this http URL. The code is available at this https URL. 

---
# When Distance Distracts: Representation Distance Bias in BT-Loss for Reward Models 

**Authors**: Tong Xie, Andrew Bai, Yuanhao Ban, Yunqi Hong, Haoyu Li, Cho-jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2512.06343)  

**Abstract**: Reward models are central to Large Language Model (LLM) alignment within the framework of RLHF. The standard objective used in reward modeling is the Bradley-Terry (BT) loss, which learns from pairwise data consisting of a pair of chosen and rejected responses. In this work, we analyze the per-sample gradient of BT-loss and show that its norm scales with two distinct components: (1) the difference in predicted rewards between chosen and rejected responses, which reflects the prediction error, and critically, (2) representation distance between the pair measured in the output space of the final layer. While the first term captures the intended training signal, we show that the second term can significantly impact the update magnitude and misalign learning. Specifically, pairs with small representation distance often receive vanishingly weak updates, even when misranked, while pairs with large distance receive disproportionately strong updates. This leads to gradients from large-distance pairs to overshadow those from small-distance pairs, where fine-grained distinctions are especially important. To overcome this limitation, we propose NormBT, an adaptive pair-wise normalization scheme that balances representation-driven effects and focuses learning signals on prediction error. NormBT is a lightweight, drop-in integration to BT loss with negligible overhead. Across various LLM backbones and datasets, NormBT improves reward model performance consistently, with notable gains of over 5% on the Reasoning category of RewardBench, which contains numerous small-distance pairs. This work reveals a key limitation in the widely used BT objective and provides a simple, effective correction. 

---
# RefBench-PRO: Perceptual and Reasoning Oriented Benchmark for Referring Expression Comprehension 

**Authors**: Tianyi Gao, Hao Li, Han Fang, Xin Wei, Xiaodong Dong, Hongbo Sun, Ye Yuan, Zhongjiang He, Jinglin Xu, Jingmin Xin, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2512.06276)  

**Abstract**: Referring Expression Comprehension (REC) is a vision-language task that localizes a specific image region based on a textual description. Existing REC benchmarks primarily evaluate perceptual capabilities and lack interpretable scoring mechanisms, which cannot reveal the grounding capability of Multi-modal Large Language Model (MLLM) across different cognitive abilities. To address this limitation, we introduce RefBench-PRO, a comprehensive REC benchmark, which decomposes referring expressions into two core dimensions, i.e., perception and reasoning, and further subdivides them into six progressively challenging tasks, such as attribute, position, interaction, commonsense, relation and reject. We also develop a fully automated data-generation pipeline that produces diverse referring expressions across these six sub-dimensions. Furthermore, We propose Ref-R1, an RL-based learning scheme, which incorporates Dynamic IoU-based GRPO to improve localization accuracy under increasingly complex reasoning conditions, establishing a stronger baseline for REC. Extensive experiments demonstrate that our RefBench-PRO enables interpretable evaluation of MLLM on referring expression comprehension, presenting greater challenges in both perception and reasoning. 

---
# DUET: Agentic Design Understanding via Experimentation and Testing 

**Authors**: Gus Henry Smith, Sandesh Adhikary, Vineet Thumuluri, Karthik Suresh, Vivek Pandit, Kartik Hegde, Hamid Shojaei, Chandra Bhagavatula  

**Link**: [PDF](https://arxiv.org/pdf/2512.06247)  

**Abstract**: AI agents powered by large language models (LLMs) are being used to solve increasingly complex software engineering challenges, but struggle with hardware design tasks. Register Transfer Level (RTL) code presents a unique challenge for LLMs, as it encodes complex, dynamic, time-evolving behaviors using the low-level language features of SystemVerilog. LLMs struggle to infer these complex behaviors from the syntax of RTL alone, which limits their ability to complete all downstream tasks like code completion, documentation, or verification. In response to this issue, we present DUET: a general methodology for developing Design Understanding via Experimentation and Testing. DUET mimics how hardware design experts develop an understanding of complex designs: not just via a one-off readthrough of the RTL, but via iterative experimentation using a number of tools. DUET iteratively generates hypotheses, tests them with EDA tools (e.g., simulation, waveform inspection, and formal verification), and integrates the results to build a bottom-up understanding of the design. In our evaluations, we show that DUET improves AI agent performance on formal verification, when compared to a baseline flow without experimentation. 

---
# PrefGen: Multimodal Preference Learning for Preference-Conditioned Image Generation 

**Authors**: Wenyi Mo, Tianyu Zhang, Yalong Bai, Ligong Han, Ying Ba, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2512.06020)  

**Abstract**: Preference-conditioned image generation seeks to adapt generative models to individual users, producing outputs that reflect personal aesthetic choices beyond the given textual prompt. Despite recent progress, existing approaches either fail to capture nuanced user preferences or lack effective mechanisms to encode personalized visual signals. In this work, we propose a multimodal framework that leverages multimodal large language models (MLLMs) to extract rich user representations and inject them into diffusion-based image generation. We train the MLLM with a preference-oriented visual question answering task to capture fine-grained semantic cues. To isolate preference-relevant features, we introduce two complementary probing tasks: inter-user discrimination to distinguish between different users, and intra-user discrimination to separate liked from disliked content. To ensure compatibility with diffusion text encoders, we design a maximum mean discrepancy-based alignment loss that bridges the modality gap while preserving multimodal structure. The resulting embeddings are used to condition the generator, enabling faithful adherence to both prompts and user preferences. Extensive experiments demonstrate that our method substantially outperforms strong baselines in both image quality and preference alignment, highlighting the effectiveness of representation extraction and alignment for personalized generation. 

---
# Future You: Designing and Evaluating Multimodal AI-generated Digital Twins for Strengthening Future Self-Continuity 

**Authors**: Constanze Albrecht, Chayapatr Archiwaranguprok, Rachel Poonsiriwong, Awu Chen, Peggy Yin, Monchai Lertsutthiwong, Kavin Winson, Hal Hershfield, Pattie Maes, Pat Pataranutaporn  

**Link**: [PDF](https://arxiv.org/pdf/2512.06106)  

**Abstract**: What if users could meet their future selves today? AI-generated future selves simulate meaningful encounters with a digital twin decades in the future. As AI systems advance, combining cloned voices, age-progressed facial rendering, and autobiographical narratives, a central question emerges: Does the modality of these future selves alter their psychological and affective impact? How might a text-based chatbot, a voice-only system, or a photorealistic avatar shape present-day decisions and our feeling of connection to the future? We report a randomized controlled study (N=92) evaluating three modalities of AI-generated future selves (text, voice, avatar) against a neutral control condition. We also report a systematic model evaluation between Claude 4 and three other Large Language Models (LLMs), assessing Claude 4 across psychological and interaction dimensions and establishing conversational AI quality as a critical determinant of intervention effectiveness. All personalized modalities strengthened Future Self-Continuity (FSC), emotional well-being, and motivation compared to control, with avatar producing the largest vividness gains, yet with no significant differences between formats. Interaction quality metrics, particularly persuasiveness, realism, and user engagement, emerged as robust predictors of psychological and affective outcomes, indicating that how compelling the interaction feels matters more than the form it takes. Content analysis found thematic patterns: text emphasized career planning, while voice and avatar facilitated personal reflection. Claude 4 outperformed ChatGPT 3.5, Llama 4, and Qwen 3 in enhancing psychological, affective, and FSC outcomes. 

---
# Auto-SPT: Automating Semantic Preserving Transformations for Code 

**Authors**: Ashish Hooda, Mihai Christodorescu, Chuangang Ren, Aaron Wilson, Kassem Fawaz, Somesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2512.06042)  

**Abstract**: Machine learning (ML) models for code clone detection determine whether two pieces of code are semantically equivalent, which in turn is a key building block for software-engineering tasks like refactoring and security tasks like vulnerability and malware detection. While these models are predominantly trained on clean, structured code datasets, real-world code often undergoes a variety of semantic-preserving transformations, including refactoring, minification, automated formatting, and compiler optimizations. To address this critical gap between training and test data, we propose Auto-SPT, a novel framework to automatically construct synthetic-data generators for code. Auto-SPT is designed to produce Semantic Preserving Transformations (SPTs) that alter a program's syntactic structure while preserving its functionality and is instantiated on top of Large Language Models (LLMs). In particular, we use LLMs to craft a diverse set of SPTs, generate strong implementations for these SPTs, and compose them to result into strong transformations. Our formal analysis shows that the diversity of SPTs impacts the strength of their composition. We then empirically demonstrate that Auto-SPT generates more diverse SPTs than existing approaches and these SPTs significantly drop the performance of state-of-the-art code clone detectors. Further experiments show Auto-SPT can be used to enhance code datasets for training, to produce code-clone detection models that are robust to real-world, adversarial code transformations. 

---
# FlockVote: LLM-Empowered Agent-Based Modeling for Simulating U.S. Presidential Elections 

**Authors**: Lingfeng Zhou, Yi Xu, Zhenyu Wang, Dequan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.05982)  

**Abstract**: Modeling complex human behavior, such as voter decisions in national elections, is a long-standing challenge for computational social science. Traditional agent-based models (ABMs) are limited by oversimplified rules, while large-scale statistical models often lack interpretability. We introduce FlockVote, a novel framework that uses Large Language Models (LLMs) to build a "computational laboratory" of LLM agents for political simulation. Each agent is instantiated with a high-fidelity demographic profile and dynamic contextual information (e.g. candidate policies), enabling it to perform nuanced, generative reasoning to simulate a voting decision. We deploy this framework as a testbed on the 2024 U.S. Presidential Election, focusing on seven key swing states. Our simulation's macro-level results successfully replicate the real-world outcome, demonstrating the high fidelity of our "virtual society". The primary contribution is not only the prediction, but also the framework's utility as an interpretable research tool. FlockVote moves beyond black-box outputs, allowing researchers to probe agent-level rationale and analyze the stability and sensitivity of LLM-driven social simulations. 

---
