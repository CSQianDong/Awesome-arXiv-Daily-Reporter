# Training Language Models to Explain Their Own Computations 

**Authors**: Belinda Z. Li, Zifan Carl Guo, Vincent Huang, Jacob Steinhardt, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2511.08579)  

**Abstract**: Can language models (LMs) learn to faithfully describe their internal computations? Are they better able to describe themselves than other models? We study the extent to which LMs' privileged access to their own internals can be leveraged to produce new techniques for explaining their behavior. Using existing interpretability techniques as a source of ground truth, we fine-tune LMs to generate natural language descriptions of (1) the information encoded by LM features, (2) the causal structure of LMs' internal activations, and (3) the influence of specific input tokens on LM outputs. When trained with only tens of thousands of example explanations, explainer models exhibit non-trivial generalization to new queries. This generalization appears partly attributable to explainer models' privileged access to their own internals: using a model to explain its own computations generally works better than using a *different* model to explain its computations (even if the other model is significantly more capable). Our results suggest not only that LMs can learn to reliably explain their internal computations, but that such explanations offer a scalable complement to existing interpretability methods. 

---
# Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models 

**Authors**: Tianyu Fu, Yichen You, Zekai Chen, Guohao Dai, Huazhong Yang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08577)  

**Abstract**: Improving reasoning capabilities of Large Language Models (LLMs), especially under parameter constraints, is crucial for real-world applications. Prior work proposes recurrent transformers, which allocate a fixed number of extra iterations per token to improve generation quality. After the first, standard forward pass, instead of verbalization, last-layer hidden states are fed back as inputs for additional iterations to refine token predictions. Yet we identify a latent overthinking phenomenon: easy token predictions that are already correct after the first pass are sometimes revised into errors in additional iterations. To address this, we propose Think-at-Hard (TaH), a dynamic latent thinking method that iterates deeper only at hard tokens. It employs a lightweight neural decider to trigger latent iterations only at tokens that are likely incorrect after the standard forward pass. During latent iterations, Low-Rank Adaptation (LoRA) modules shift the LLM objective from general next-token prediction to focused hard-token refinement. We further introduce a duo-causal attention mechanism that extends attention from the token sequence dimension to an additional iteration depth dimension. This enables cross-iteration information flow while maintaining full sequential parallelism. Experiments show that TaH boosts LLM reasoning performance across five challenging benchmarks while maintaining the same parameter count. Compared with baselines that iterate twice for all output tokens, TaH delivers 8.1-11.3% accuracy gains while exempting 94% of tokens from the second iteration. Against strong single-iteration Qwen3 models finetuned with the same data, it also delivers 4.0-5.0% accuracy gains. When allowing less than 3% additional parameters from LoRA and the iteration decider, the gains increase to 8.5-12.6% and 5.3-5.4%, respectively. Our code is available at this https URL. 

---
# Moral Susceptibility and Robustness under Persona Role-Play in Large Language Models 

**Authors**: Davi Bastos Costa, Felippe Alves, Renato Vicente  

**Link**: [PDF](https://arxiv.org/pdf/2511.08565)  

**Abstract**: Large language models (LLMs) increasingly operate in social contexts, motivating analysis of how they express and shift moral judgments. In this work, we investigate the moral response of LLMs to persona role-play, prompting a LLM to assume a specific character. Using the Moral Foundations Questionnaire (MFQ), we introduce a benchmark that quantifies two properties: moral susceptibility and moral robustness, defined from the variability of MFQ scores across and within personas, respectively. We find that, for moral robustness, model family accounts for most of the variance, while model size shows no systematic effect. The Claude family is, by a significant margin, the most robust, followed by Gemini and GPT-4 models, with other families exhibiting lower robustness. In contrast, moral susceptibility exhibits a mild family effect but a clear within-family size effect, with larger variants being more susceptible. Moreover, robustness and susceptibility are positively correlated, an association that is more pronounced at the family level. Additionally, we present moral foundation profiles for models without persona role-play and for personas averaged across models. Together, these analyses provide a systematic view of how persona conditioning shapes moral behavior in large language models. 

---
# From Semantic Roles to Opinion Roles: SRL Data Extraction for Multi-Task and Transfer Learning in Low-Resource ORL 

**Authors**: Amirmohammad Omidi Galdiani, Sepehr Rezaei Melal, Mohammad Norasteh, Arash Yousefi Jordehi, Seyed Abolghasem Mirroshandel  

**Link**: [PDF](https://arxiv.org/pdf/2511.08537)  

**Abstract**: This report presents a detailed methodology for constructing a high-quality Semantic Role Labeling (SRL) dataset from the Wall Street Journal (WSJ) portion of the OntoNotes 5.0 corpus and adapting it for Opinion Role Labeling (ORL) tasks. Leveraging the PropBank annotation framework, we implement a reproducible extraction pipeline that aligns predicate-argument structures with surface text, converts syntactic tree pointers to coherent spans, and applies rigorous cleaning to ensure semantic fidelity. The resulting dataset comprises 97,169 predicate-argument instances with clearly defined Agent (ARG0), Predicate (REL), and Patient (ARG1) roles, mapped to ORL's Holder, Expression, and Target schema. We provide a detailed account of our extraction algorithms, discontinuous argument handling, annotation corrections, and statistical analysis of the resulting dataset. This work offers a reusable resource for researchers aiming to leverage SRL for enhancing ORL, especially in low-resource opinion mining scenarios. 

---
# Investigating CoT Monitorability in Large Reasoning Models 

**Authors**: Shu Yang, Junchao Wu, Xilin Gou, Xuansheng Wu, Derek Wong, Ninhao Liu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08525)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable performance on complex tasks by engaging in extended reasoning before producing final answers. Beyond improving abilities, these detailed reasoning traces also create a new opportunity for AI safety, CoT Monitorability: monitoring potential model misbehavior, such as the use of shortcuts or sycophancy, through their chain-of-thought (CoT) during decision-making. However, two key fundamental challenges arise when attempting to build more effective monitors through CoT analysis. First, as prior research on CoT faithfulness has pointed out, models do not always truthfully represent their internal decision-making in the generated reasoning. Second, monitors themselves may be either overly sensitive or insufficiently sensitive, and can potentially be deceived by models' long, elaborate reasoning traces. In this paper, we present the first systematic investigation of the challenges and potential of CoT monitorability. Motivated by two fundamental challenges we mentioned before, we structure our study around two central perspectives: (i) verbalization: to what extent do LRMs faithfully verbalize the true factors guiding their decisions in the CoT, and (ii) monitor reliability: to what extent can misbehavior be reliably detected by a CoT-based monitor? Specifically, we provide empirical evidence and correlation analyses between verbalization quality, monitor reliability, and LLM performance across mathematical, scientific, and ethical domains. Then we further investigate how different CoT intervention methods, designed to improve reasoning efficiency or performance, will affect monitoring effectiveness. Finally, we propose MoME, a new paradigm in which LLMs monitor other models' misbehavior through their CoT and provide structured judgments along with supporting evidence. 

---
# AlphaResearch: Accelerating New Algorithm Discovery with Language Models 

**Authors**: Zhaojian Yu, Kaiyue Feng, Yilun Zhao, Shilin He, Xiao-Ping Zhang, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2511.08522)  

**Abstract**: Large language models have made significant progress in complex but easy-to-verify problems, yet they still struggle with discovering the unknown. In this paper, we present \textbf{AlphaResearch}, an autonomous research agent designed to discover new algorithms on open-ended problems. To synergize the feasibility and innovation of the discovery process, we construct a novel dual research environment by combining the execution-based verify and simulated real-world peer review environment. AlphaResearch discovers new algorithm by iteratively running the following steps: (1) propose new ideas (2) verify the ideas in the dual research environment (3) optimize the research proposals for better performance. To promote a transparent evaluation process, we construct \textbf{AlphaResearchComp}, a new evaluation benchmark that includes an eight open-ended algorithmic problems competition, with each problem carefully curated and verified through executable pipelines, objective metrics, and reproducibility checks. AlphaResearch gets a 2/8 win rate in head-to-head comparison with human researchers, demonstrate the possibility of accelerating algorithm discovery with LLMs. Notably, the algorithm discovered by AlphaResearch on the \emph{``packing circles''} problem achieves the best-of-known performance, surpassing the results of human researchers and strong baselines from recent work (e.g., AlphaEvolve). Additionally, we conduct a comprehensive analysis of the remaining challenges of the 6/8 failure cases, providing valuable insights for future research. 

---
# Introducing A Bangla Sentence - Gloss Pair Dataset for Bangla Sign Language Translation and Research 

**Authors**: Neelavro Saha, Rafi Shahriyar, Nafis Ashraf Roudra, Saadman Sakib, Annajiat Alim Rasel  

**Link**: [PDF](https://arxiv.org/pdf/2511.08507)  

**Abstract**: Bangla Sign Language (BdSL) translation represents a low-resource NLP task due to the lack of large-scale datasets that address sentence-level translation. Correspondingly, existing research in this field has been limited to word and alphabet level detection. In this work, we introduce Bangla-SGP, a novel parallel dataset consisting of 1,000 human-annotated sentence-gloss pairs which was augmented with around 3,000 synthetically generated pairs using syntactic and morphological rules through a rule-based Retrieval-Augmented Generation (RAG) pipeline. The gloss sequences of the spoken Bangla sentences are made up of individual glosses which are Bangla sign supported words and serve as an intermediate representation for a continuous sign. Our dataset consists of 1000 high quality Bangla sentences that are manually annotated into a gloss sequence by a professional signer. The augmentation process incorporates rule-based linguistic strategies and prompt engineering techniques that we have adopted by critically analyzing our human annotated sentence-gloss pairs and by working closely with our professional signer. Furthermore, we fine-tune several transformer-based models such as mBart50, Google mT5, GPT4.1-nano and evaluate their sentence-to-gloss translation performance using BLEU scores, based on these evaluation metrics we compare the model's gloss-translation consistency across our dataset and the RWTH-PHOENIX-2014T benchmark. 

---
# Structured RAG for Answering Aggregative Questions 

**Authors**: Omri Koshorek, Niv Granot, Aviv Alloni, Shahar Admati, Roee Hendel, Ido Weiss, Alan Arazi, Shay-Nitzan Cohen, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2511.08505)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become the dominant approach for answering questions over large corpora. However, current datasets and methods are highly focused on cases where only a small part of the corpus (usually a few paragraphs) is relevant per query, and fail to capture the rich world of aggregative queries. These require gathering information from a large set of documents and reasoning over them. To address this gap, we propose S-RAG, an approach specifically designed for such queries. At ingestion time, S-RAG constructs a structured representation of the corpus; at inference time, it translates natural-language queries into formal queries over said representation. To validate our approach and promote further research in this area, we introduce two new datasets of aggregative queries: HOTELS and WORLD CUP. Experiments with S-RAG on the newly introduced datasets, as well as on a public benchmark, demonstrate that it substantially outperforms both common RAG systems and long-context LLMs. 

---
# SPEAR-MM: Selective Parameter Evaluation and Restoration via Model Merging for Efficient Financial LLM Adaptation 

**Authors**: Berkcan Kapusuzoglu, Supriyo Chakraborty, Renkun Ni, Stephen Rawls, Sambit Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2511.08500)  

**Abstract**: Large language models (LLMs) adapted to financial domains often suffer from catastrophic forgetting of general reasoning capabilities essential for customer interactions and complex financial analysis. We introduce Selective Parameter Evaluation and Restoration via Model Merging (SPEAR-MM), a practical framework that preserves critical capabilities while enabling domain adaptation. Our method approximates layer-wise impact on external benchmarks through post-hoc analysis, then selectively freezes or restores transformer layers via spherical interpolation merging. Applied to LLaMA-3.1-8B for financial tasks, SPEAR-MM achieves 91.2% retention of general capabilities versus 69.7% for standard continual pretraining, while maintaining 94% of domain adaptation gains. The approach provides interpretable trade-off control and reduces computational costs by 90% crucial for resource-constrained financial institutions. 

---
# Bot Meets Shortcut: How Can LLMs Aid in Handling Unknown Invariance OOD Scenarios? 

**Authors**: Shiyan Zheng, Herun Wan, Minnan Luo, Junhang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08455)  

**Abstract**: While existing social bot detectors perform well on benchmarks, their robustness across diverse real-world scenarios remains limited due to unclear ground truth and varied misleading cues. In particular, the impact of shortcut learning, where models rely on spurious correlations instead of capturing causal task-relevant features, has received limited attention. To address this gap, we conduct an in-depth study to assess how detectors are influenced by potential shortcuts based on textual features, which are most susceptible to manipulation by social bots. We design a series of shortcut scenarios by constructing spurious associations between user labels and superficial textual cues to evaluate model robustness. Results show that shifts in irrelevant feature distributions significantly degrade social bot detector performance, with an average relative accuracy drop of 32\% in the baseline models. To tackle this challenge, we propose mitigation strategies based on large language models, leveraging counterfactual data augmentation. These methods mitigate the problem from data and model perspectives across three levels, including data distribution at both the individual user text and overall dataset levels, as well as the model's ability to extract causal information. Our strategies achieve an average relative performance improvement of 56\% under shortcut scenarios. 

---
# Interaction Dynamics as a Reward Signal for LLMs 

**Authors**: Sian Gooding, Edward Grefenstette  

**Link**: [PDF](https://arxiv.org/pdf/2511.08394)  

**Abstract**: The alignment of Large Language Models (LLMs) for multi-turn conversations typically relies on reward signals derived from the content of the text. This approach, however, overlooks a rich, complementary source of signal: the dynamics of the interaction itself. This paper introduces TRACE (Trajectory-based Reward for Agent Collaboration Estimation), a novel reward signal derived from the geometric properties of a dialogue's embedding trajectory--a concept we term 'conversational geometry'. Our central finding is that a reward model trained only on these structural signals achieves a pairwise accuracy (68.20%) comparable to a powerful LLM baseline that analyzes the full transcript (70.04%). Furthermore, a hybrid model combining interaction dynamics with textual analysis achieves the highest performance (80.17%), demonstrating their complementary nature. This work provides strong evidence that for interactive settings, how an agent communicates is as powerful a predictor of success as what it says, offering a new, privacy-preserving framework that not only aligns agents but also serves as a diagnostic tool for understanding the distinct interaction patterns that drive successful collaboration. 

---
# PCRLLM: Proof-Carrying Reasoning with Large Language Models under Stepwise Logical Constraints 

**Authors**: Tangrui Li, Pei Wang, Hongzheng Wang Christian Hahm, Matteo Spatola, Justin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2511.08392)  

**Abstract**: Large Language Models (LLMs) often exhibit limited logical coherence, mapping premises to conclusions without adherence to explicit inference rules. We propose Proof-Carrying Reasoning with LLMs (PCRLLM), a framework that constrains reasoning to single-step inferences while preserving natural language formulations. Each output explicitly specifies premises, rules, and conclusions, thereby enabling verification against a target logic. This mechanism mitigates trustworthiness concerns by supporting chain-level validation even in black-box settings. Moreover, PCRLLM facilitates systematic multi-LLM collaboration, allowing intermediate steps to be compared and integrated under formal rules. Finally, we introduce a benchmark schema for generating large-scale step-level reasoning data, combining natural language expressiveness with formal rigor. 

---
# TurkEmbed: Turkish Embedding Model on NLI & STS Tasks 

**Authors**: Özay Ezerceli, Gizem Gümüşçekiçci, Tuğba Erkoç, Berke Özenç  

**Link**: [PDF](https://arxiv.org/pdf/2511.08376)  

**Abstract**: This paper introduces TurkEmbed, a novel Turkish language embedding model designed to outperform existing models, particularly in Natural Language Inference (NLI) and Semantic Textual Similarity (STS) tasks. Current Turkish embedding models often rely on machine-translated datasets, potentially limiting their accuracy and semantic understanding. TurkEmbed utilizes a combination of diverse datasets and advanced training techniques, including matryoshka representation learning, to achieve more robust and accurate embeddings. This approach enables the model to adapt to various resource-constrained environments, offering faster encoding capabilities. Our evaluation on the Turkish STS-b-TR dataset, using Pearson and Spearman correlation metrics, demonstrates significant improvements in semantic similarity tasks. Furthermore, TurkEmbed surpasses the current state-of-the-art model, Emrecan, on All-NLI-TR and STS-b-TR benchmarks, achieving a 1-4\% improvement. TurkEmbed promises to enhance the Turkish NLP ecosystem by providing a more nuanced understanding of language and facilitating advancements in downstream applications. 

---
# The Dynamic Articulatory Model DYNARTmo: Dynamic Movement Generation and Speech Gestures 

**Authors**: Bernd J. Kröger  

**Link**: [PDF](https://arxiv.org/pdf/2511.08372)  

**Abstract**: This paper describes the current implementation of the dynamic articulatory model DYNARTmo, which generates continuous articulator movements based on the concept of speech gestures and a corresponding gesture score. The model provides a neurobiologically inspired computational framework for simulating the hierarchical control of speech production from linguistic representation to articulatory-acoustic realization. We present the structure of the gesture inventory, the coordination of gestures in the gesture score, and their translation into continuous articulator trajectories controlling the DYNARTmo vocal tract model. 

---
# DPRM: A Dual Implicit Process Reward Model in Multi-Hop Question Answering 

**Authors**: Xinyi Wang, Yiping Song, Zhiliang Tian, Bo Liu, Tingjin Luo, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08364)  

**Abstract**: In multi-hop question answering (MHQA) tasks, Chain of Thought (CoT) improves the quality of generation by guiding large language models (LLMs) through multi-step reasoning, and Knowledge Graphs (KGs) reduce hallucinations via semantic matching. Outcome Reward Models (ORMs) provide feedback after generating the final answers but fail to evaluate the process for multi-step reasoning. Traditional Process Reward Models (PRMs) evaluate the reasoning process but require costly human annotations or rollout generation. While implicit PRM is trained only with outcome signals and derives step rewards through reward parameterization without explicit annotations, it is more suitable for multi-step reasoning in MHQA tasks. However, existing implicit PRM has only been explored for plain text scenarios. When adapting to MHQA tasks, it cannot handle the graph structure constraints in KGs and capture the potential inconsistency between CoT and KG paths. To address these limitations, we propose the DPRM (Dual Implicit Process Reward Model). It trains two implicit PRMs for CoT and KG reasoning in MHQA tasks. Both PRMs, namely KG-PRM and CoT-PRM, derive step-level rewards from outcome signals via reward parameterization without additional explicit annotations. Among them, KG-PRM uses preference pairs to learn structural constraints from KGs. DPRM further introduces a consistency constraint between CoT and KG reasoning steps, making the two PRMs mutually verify and collaboratively optimize the reasoning paths. We also provide a theoretical demonstration of the derivation of process rewards. Experimental results show that our method outperforms 13 baselines on multiple datasets with up to 16.6% improvement on Hit@1. 

---
# AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress 

**Authors**: Zhiheng Xi, Chenyang Liao, Guanyu Li, Yajie Yang, Wenxiang Chen, Zhihao Zhang, Binghai Wang, Senjie Jin, Yuhao Zhou, Jian Guan, Wei Wu, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08325)  

**Abstract**: Despite rapid development, large language models (LLMs) still encounter challenges in multi-turn decision-making tasks (i.e., agent tasks) like web shopping and browser navigation, which require making a sequence of intelligent decisions based on environmental feedback. Previous work for LLM agents typically relies on elaborate prompt engineering or fine-tuning with expert trajectories to improve performance. In this work, we take a different perspective: we explore constructing process reward models (PRMs) to evaluate each decision and guide the agent's decision-making process. Unlike LLM reasoning, where each step is scored based on correctness, actions in agent tasks do not have a clear-cut correctness. Instead, they should be evaluated based on their proximity to the goal and the progress they have made. Building on this insight, we propose a re-defined PRM for agent tasks, named AgentPRM, to capture both the interdependence between sequential decisions and their contribution to the final goal. This enables better progress tracking and exploration-exploitation balance. To scalably obtain labeled data for training AgentPRM, we employ a Temporal Difference-based (TD-based) estimation method combined with Generalized Advantage Estimation (GAE), which proves more sample-efficient than prior methods. Extensive experiments across different agentic tasks show that AgentPRM is over $8\times$ more compute-efficient than baselines, and it demonstrates robust improvement when scaling up test-time compute. Moreover, we perform detailed analyses to show how our method works and offer more insights, e.g., applying AgentPRM to the reinforcement learning of LLM agents. 

---
# Adaptive Multi-Agent Response Refinement in Conversational Systems 

**Authors**: Soyeong Jeong, Aparna Elangovan, Emine Yilmaz, Oleg Rokhlenko  

**Link**: [PDF](https://arxiv.org/pdf/2511.08319)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable success in conversational systems by generating human-like responses. However, they can fall short, especially when required to account for personalization or specific knowledge. In real-life settings, it is impractical to rely on users to detect these errors and request a new response. One way to address this problem is to refine the response before returning it to the user. While existing approaches focus on refining responses within a single LLM, this method struggles to consider diverse aspects needed for effective conversations. In this work, we propose refining responses through a multi-agent framework, where each agent is assigned a specific role for each aspect. We focus on three key aspects crucial to conversational quality: factuality, personalization, and coherence. Each agent is responsible for reviewing and refining one of these aspects, and their feedback is then merged to improve the overall response. To enhance collaboration among them, we introduce a dynamic communication strategy. Instead of following a fixed sequence of agents, our approach adaptively selects and coordinates the most relevant agents based on the specific requirements of each query. We validate our framework on challenging conversational datasets, demonstrating that ours significantly outperforms relevant baselines, particularly in tasks involving knowledge or user's persona, or both. 

---
# Automatic Paper Reviewing with Heterogeneous Graph Reasoning over LLM-Simulated Reviewer-Author Debates 

**Authors**: Shuaimin Li, Liyang Fan, Yufang Lin, Zeyang Li, Xian Wei, Shiwen Ni, Hamid Alinejad-Rokny, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08317)  

**Abstract**: Existing paper review methods often rely on superficial manuscript features or directly on large language models (LLMs), which are prone to hallucinations, biased scoring, and limited reasoning capabilities. Moreover, these methods often fail to capture the complex argumentative reasoning and negotiation dynamics inherent in reviewer-author interactions. To address these limitations, we propose ReViewGraph (Reviewer-Author Debates Graph Reasoner), a novel framework that performs heterogeneous graph reasoning over LLM-simulated multi-round reviewer-author debates. In our approach, reviewer-author exchanges are simulated through LLM-based multi-agent collaboration. Diverse opinion relations (e.g., acceptance, rejection, clarification, and compromise) are then explicitly extracted and encoded as typed edges within a heterogeneous interaction graph. By applying graph neural networks to reason over these structured debate graphs, ReViewGraph captures fine-grained argumentative dynamics and enables more informed review decisions. Extensive experiments on three datasets demonstrate that ReViewGraph outperforms strong baselines with an average relative improvement of 15.73%, underscoring the value of modeling detailed reviewer-author debate structures. 

---
# Hierarchical structure understanding in complex tables with VLLMs: a benchmark and experiments 

**Authors**: Luca Bindini, Simone Giovannini, Simone Marinai, Valeria Nardoni, Kimiya Noor Ali  

**Link**: [PDF](https://arxiv.org/pdf/2511.08298)  

**Abstract**: This work investigates the ability of Vision Large Language Models (VLLMs) to understand and interpret the structure of tables in scientific articles. Specifically, we explore whether VLLMs can infer the hierarchical structure of tables without additional processing. As a basis for our experiments we use the PubTables-1M dataset, a large-scale corpus of scientific tables. From this dataset, we extract a subset of tables that we introduce as Complex Hierarchical Tables (CHiTab): a benchmark collection of complex tables containing hierarchical headings. We adopt a series of prompt engineering strategies to probe the models' comprehension capabilities, experimenting with various prompt formats and writing styles. Multiple state-of-the-art open-weights VLLMs are evaluated on the benchmark first using their off-the-shelf versions and then fine-tuning some models on our task. We also measure the performance of humans to solve the task on a small set of tables comparing with performance of the evaluated VLLMs. The experiments support our intuition that generic VLLMs, not explicitly designed for understanding the structure of tables, can perform this task. This study provides insights into the potential and limitations of VLLMs to process complex tables and offers guidance for future work on integrating structured data understanding into general-purpose VLLMs. 

---
# ParliaBench: An Evaluation and Benchmarking Framework for LLM-Generated Parliamentary Speech 

**Authors**: Marios Koniaris, Argyro Tsipi, Panayiotis Tsanakas  

**Link**: [PDF](https://arxiv.org/pdf/2511.08247)  

**Abstract**: Parliamentary speech generation presents specific challenges for large language models beyond standard text generation tasks. Unlike general text generation, parliamentary speeches require not only linguistic quality but also political authenticity and ideological consistency. Current language models lack specialized training for parliamentary contexts, and existing evaluation methods focus on standard NLP metrics rather than political authenticity. To address this, we present ParliaBench, a benchmark for parliamentary speech generation. We constructed a dataset of speeches from UK Parliament to enable systematic model training. We introduce an evaluation framework combining computational metrics with LLM-as-a-judge assessments for measuring generation quality across three dimensions: linguistic quality, semantic coherence, and political authenticity. We propose two novel embedding-based metrics, Political Spectrum Alignment and Party Alignment, to quantify ideological positioning. We fine-tuned five large language models (LLMs), generated 28k speeches, and evaluated them using our framework, comparing baseline and fine-tuned models. Results show that fine-tuning produces statistically significant improvements across the majority of metrics and our novel metrics demonstrate strong discriminative power for political dimensions. 

---
# Prompt Tuning for Natural Language to SQL with Embedding Fine-Tuning and RAG 

**Authors**: Jisoo Jang, Tien-Cuong Bui, Yunjun Choi, Wen-Syan Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.08245)  

**Abstract**: This paper introduces an Error Correction through Prompt Tuning for NL-to-SQL, leveraging the latest advancements in generative pre-training-based LLMs and RAG. Our work addresses the crucial need for efficient and accurate translation of natural language queries into SQL expressions in various settings with the growing use of natural language interfaces. We explore the evolution of NLIDBs from early rule-based systems to advanced neural network-driven approaches. Drawing inspiration from the medical diagnostic process, we propose a novel framework integrating an error correction mechanism that diagnoses error types, identifies their causes, provides fixing instructions, and applies these corrections to SQL queries. This approach is further enriched by embedding fine-tuning and RAG, which harnesses external knowledge bases for improved accuracy and transparency. Through comprehensive experiments, we demonstrate that our framework achieves a significant 12 percent accuracy improvement over existing baselines, highlighting its potential to revolutionize data access and handling in contemporary data-driven environments. 

---
# VocalBench-zh: Decomposing and Benchmarking the Speech Conversational Abilities in Mandarin Context 

**Authors**: Heyang Liu, Ziyang Cheng, Yuhao Wang, Hongcheng Liu, Yiqi Li, Ronghua Wu, Qunshan Gu, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08230)  

**Abstract**: The development of multi-modal large language models (LLMs) leads to intelligent approaches capable of speech interactions. As one of the most widely spoken languages globally, Mandarin is supported by most models to enhance their applicability and reach. However, the scarcity of comprehensive speech-to-speech (S2S) benchmarks in Mandarin contexts impedes systematic evaluation for developers and hinders fair model comparison for users. In this work, we propose VocalBench-zh, an ability-level divided evaluation suite adapted to Mandarin context consisting of 10 well-crafted subsets and over 10K high-quality instances, covering 12 user-oriented characters. The evaluation experiment on 14 mainstream models reveals the common challenges for current routes, and highlights the need for new insights into next-generation speech interactive systems. The evaluation codes and datasets will be available at this https URL. 

---
# Benchmarking Educational LLMs with Analytics: A Case Study on Gender Bias in Feedback 

**Authors**: Yishan Du, Conrad Borchers, Mutlu Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2511.08225)  

**Abstract**: As teachers increasingly turn to GenAI in their educational practice, we need robust methods to benchmark large language models (LLMs) for pedagogical purposes. This article presents an embedding-based benchmarking framework to detect bias in LLMs in the context of formative feedback. Using 600 authentic student essays from the AES 2.0 corpus, we constructed controlled counterfactuals along two dimensions: (i) implicit cues via lexicon-based swaps of gendered terms within essays, and (ii) explicit cues via gendered author background in the prompt. We investigated six representative LLMs (i.e. GPT-5 mini, GPT-4o mini, DeepSeek-R1, DeepSeek-R1-Qwen, Gemini 2.5 Pro, Llama-3-8B). We first quantified the response divergence with cosine and Euclidean distances over sentence embeddings, then assessed significance via permutation tests, and finally, visualised structure using dimensionality reduction. In all models, implicit manipulations reliably induced larger semantic shifts for male-female counterfactuals than for female-male. Only the GPT and Llama models showed sensitivity to explicit gender cues. These findings show that even state-of-the-art LLMs exhibit asymmetric semantic responses to gender substitutions, suggesting persistent gender biases in feedback they provide learners. Qualitative analyses further revealed consistent linguistic differences (e.g., more autonomy-supportive feedback under male cues vs. more controlling feedback under female cues). We discuss implications for fairness auditing of pedagogical GenAI, propose reporting standards for counterfactual evaluation in learning analytics, and outline practical guidance for prompt design and deployment to safeguard equitable feedback. 

---
# Encoder Fine-tuning with Stochastic Sampling Outperforms Open-weight GPT in Astronomy Knowledge Extraction 

**Authors**: Shivam Rawat, Lucie Flek, Akbar Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2511.08204)  

**Abstract**: Scientific literature in astronomy is rapidly expanding, making it increasingly important to automate the extraction of key entities and contextual information from research papers. In this paper, we present an encoder-based system for extracting knowledge from astronomy articles. Our objective is to develop models capable of classifying telescope references, detecting auxiliary semantic attributes, and recognizing instrument mentions from textual content. To this end, we implement a multi-task transformer-based system built upon the SciBERT model and fine-tuned for astronomy corpora classification. To carry out the fine-tuning, we stochastically sample segments from the training data and use majority voting over the test segments at inference time. Our system, despite its simplicity and low-cost implementation, significantly outperforms the open-weight GPT baseline. 

---
# Do Syntactic Categories Help in Developmentally Motivated Curriculum Learning for Language Models? 

**Authors**: Arzu Burcu Güven, Anna Rogers, Rob van der Goot  

**Link**: [PDF](https://arxiv.org/pdf/2511.08199)  

**Abstract**: We examine the syntactic properties of BabyLM corpus, and age-groups within CHILDES. While we find that CHILDES does not exhibit strong syntactic differentiation by age, we show that the syntactic knowledge about the training data can be helpful in interpreting model performance on linguistic tasks. For curriculum learning, we explore developmental and several alternative cognitively inspired curriculum approaches. We find that some curricula help with reading tasks, but the main performance improvement come from using the subset of syntactically categorizable data, rather than the full noisy corpus. 

---
# Still Not There: Can LLMs Outperform Smaller Task-Specific Seq2Seq Models on the Poetry-to-Prose Conversion Task? 

**Authors**: Kunal Kingkar Das, Manoj Balaji Jagadeeshan, Nallani Chakravartula Sahith, Jivnesh Sandhan, Pawan Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2511.08145)  

**Abstract**: Large Language Models (LLMs) are increasingly treated as universal, general-purpose solutions across NLP tasks, particularly in English. But does this assumption hold for low-resource, morphologically rich languages such as Sanskrit? We address this question by comparing instruction-tuned and in-context-prompted LLMs with smaller task-specific encoder-decoder models on the Sanskrit poetry-to-prose conversion task. This task is intrinsically challenging: Sanskrit verse exhibits free word order combined with rigid metrical constraints, and its conversion to canonical prose (anvaya) requires multi-step reasoning involving compound segmentation, dependency resolution, and syntactic linearisation. This makes it an ideal testbed to evaluate whether LLMs can surpass specialised models. For LLMs, we apply instruction fine-tuning on general-purpose models and design in-context learning templates grounded in Paninian grammar and classical commentary heuristics. For task-specific modelling, we fully fine-tune a ByT5-Sanskrit Seq2Seq model. Our experiments show that domain-specific fine-tuning of ByT5-Sanskrit significantly outperforms all instruction-driven LLM approaches. Human evaluation strongly corroborates this result, with scores exhibiting high correlation with Kendall's Tau scores. Additionally, our prompting strategies provide an alternative to fine-tuning when domain-specific verse corpora are unavailable, and the task-specific Seq2Seq model demonstrates robust generalisation on out-of-domain evaluations. 

---
# Relation as a Prior: A Novel Paradigm for LLM-based Document-level Relation Extraction 

**Authors**: Qiankun Pi, Yepeng Sun, Jicang Lu, Qinlong Fan, Ningbo Huang, Shiyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08143)  

**Abstract**: Large Language Models (LLMs) have demonstrated their remarkable capabilities in document understanding. However, recent research reveals that LLMs still exhibit performance gaps in Document-level Relation Extraction (DocRE) as requiring fine-grained comprehension. The commonly adopted "extract entities then predict relations" paradigm in LLM-based methods leads to these gaps due to two main reasons: (1) Numerous unrelated entity pairs introduce noise and interfere with the relation prediction for truly related entity pairs. (2) Although LLMs have identified semantic associations between entities, relation labels beyond the predefined set are still treated as prediction errors. To address these challenges, we propose a novel Relation as a Prior (RelPrior) paradigm for LLM-based DocRE. For challenge (1), RelPrior utilizes binary relation as a prior to extract and determine whether two entities are correlated, thereby filtering out irrelevant entity pairs and reducing prediction noise. For challenge (2), RelPrior utilizes predefined relation as a prior to match entities for triples extraction instead of directly predicting relation. Thus, it avoids misjudgment caused by strict predefined relation labeling. Extensive experiments on two benchmarks demonstrate that RelPrior achieves state-of-the-art performance, surpassing existing LLM-based methods. 

---
# On the Interplay between Positional Encodings, Morphological Complexity, and Word Order Flexibility 

**Authors**: Kushal Tatariya, Wessel Poelman, Miryam de Lhoneux  

**Link**: [PDF](https://arxiv.org/pdf/2511.08139)  

**Abstract**: Language model architectures are predominantly first created for English and subsequently applied to other languages. It is an open question whether this architectural bias leads to degraded performance for languages that are structurally different from English. We examine one specific architectural choice: positional encodings, through the lens of the trade-off hypothesis: the supposed interplay between morphological complexity and word order flexibility. This hypothesis posits a trade-off between the two: a more morphologically complex language can have a more flexible word order, and vice-versa. Positional encodings are a direct target to investigate the implications of this hypothesis in relation to language modelling. We pretrain monolingual model variants with absolute, relative, and no positional encodings for seven typologically diverse languages and evaluate them on four downstream tasks. Contrary to previous findings, we do not observe a clear interaction between position encodings and morphological complexity or word order flexibility, as measured by various proxies. Our results show that the choice of tasks, languages, and metrics are essential for drawing stable conclusions 

---
# Sentence-Anchored Gist Compression for Long-Context LLMs 

**Authors**: Dmitrii Tarasov, Elizaveta Goncharova, Kuznetsov Andrey  

**Link**: [PDF](https://arxiv.org/pdf/2511.08128)  

**Abstract**: This work investigates context compression for Large Language Models (LLMs) using learned compression tokens to reduce the memory and computational demands of processing long sequences. We demonstrate that pre-trained LLMs can be fine-tuned to compress their context by factors of 2x to 8x without significant performance degradation, as evaluated on both short-context and long-context benchmarks. Furthermore, in experiments on a 3-billion-parameter LLaMA model, our method achieves results on par with alternative compression techniques while attaining higher compression ratios. 

---
# Quantification and object perception in Multimodal Large Language Models deviate from human linguistic cognition 

**Authors**: Raquel Montero, Natalia Moskvina, Paolo Morosi, Tamara Serrano, Elena Pagliarini, Evelina Leivada  

**Link**: [PDF](https://arxiv.org/pdf/2511.08126)  

**Abstract**: Quantification has been proven to be a particularly difficult linguistic phenomenon for (Multimodal) Large Language Models (MLLMs). However, given that quantification interfaces with the logic, pragmatic, and numerical domains, the exact reasons for the poor performance are still unclear. This papers looks at three key features of human quantification shared cross-linguistically that have remained so far unexplored in the (M)LLM literature: the ordering of quantifiers into scales, the ranges of use and prototypicality, and the biases inherent in the human approximate number system. The aim is to determine how these features are encoded in the models' architecture, how they may differ from humans, and whether the results are affected by the type of model and language under investigation. We find that there are clear differences between humans and MLLMs with respect to these features across various tasks that tap into the representation of quantification in vivo vs. in silico. This work, thus, paves the way for addressing the nature of MLLMs as semantic and pragmatic agents, while the cross-linguistic lens can elucidate whether their abilities are robust and stable across different languages. 

---
# Multimodal LLMs Do Not Compose Skills Optimally Across Modalities 

**Authors**: Paula Ontalvilla, Aitor Ormazabal, Gorka Azkune  

**Link**: [PDF](https://arxiv.org/pdf/2511.08113)  

**Abstract**: Skill composition is the ability to combine previously learned skills to solve new tasks. As neural networks acquire increasingly complex skills during their pretraining, it is not clear how successfully they can compose them. In this paper, we focus on Multimodal Large Language Models (MLLM), and study their ability to compose skills across modalities. To this end, we design three evaluation tasks which can be solved sequentially composing two modality-dependent skills, and evaluate several open MLLMs under two main settings: i) prompting the model to directly solve the task, and ii) using a two-step cascaded inference approach, which manually enforces the composition of the two skills for a given task. Even with these straightforward compositions, we find that all evaluated MLLMs exhibit a significant cross-modality skill composition gap. To mitigate the aforementioned gap, we explore two alternatives: i) use chain-of-thought prompting to explicitly instruct MLLMs for skill composition and ii) a specific fine-tuning recipe to promote skill composition. Although those strategies improve model performance, they still exhibit significant skill composition gaps, suggesting that more research is needed to improve cross-modal skill composition in MLLMs. 

---
# Estranged Predictions: Measuring Semantic Category Disruption with Masked Language Modelling 

**Authors**: Yuxuan Liu, Haim Dubossarsky, Ruth Ahnert  

**Link**: [PDF](https://arxiv.org/pdf/2511.08109)  

**Abstract**: This paper examines how science fiction destabilises ontological categories by measuring conceptual permeability across the terms human, animal, and machine using masked language modelling (MLM). Drawing on corpora of science fiction (Gollancz SF Masterworks) and general fiction (NovelTM), we operationalise Darko Suvin's theory of estrangement as computationally measurable deviation in token prediction, using RoBERTa to generate lexical substitutes for masked referents and classifying them via Gemini. We quantify conceptual slippage through three metrics: retention rate, replacement rate, and entropy, mapping the stability or disruption of category boundaries across genres. Our findings reveal that science fiction exhibits heightened conceptual permeability, particularly around machine referents, which show significant cross-category substitution and dispersion. Human terms, by contrast, maintain semantic coherence and often anchor substitutional hierarchies. These patterns suggest a genre-specific restructuring within anthropocentric logics. We argue that estrangement in science fiction operates as a controlled perturbation of semantic norms, detectable through probabilistic modelling, and that MLMs, when used critically, serve as interpretive instruments capable of surfacing genre-conditioned ontological assumptions. This study contributes to the methodological repertoire of computational literary studies and offers new insights into the linguistic infrastructure of science fiction. 

---
# BARD10: A New Benchmark Reveals Significance of Bangla Stop-Words in Authorship Attribution 

**Authors**: Abdullah Muhammad Moosa, Nusrat Sultana, Mahdi Muhammad Moosa, Md. Miraiz Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2511.08085)  

**Abstract**: This research presents a comprehensive investigation into Bangla authorship attribution, introducing a new balanced benchmark corpus BARD10 (Bangla Authorship Recognition Dataset of 10 authors) and systematically analyzing the impact of stop-word removal across classical and deep learning models to uncover the stylistic significance of Bangla stop-words. BARD10 is a curated corpus of Bangla blog and opinion prose from ten contemporary authors, alongside the methodical assessment of four representative classifiers: SVM (Support Vector Machine), Bangla BERT (Bidirectional Encoder Representations from Transformers), XGBoost, and a MLP (Multilayer Perception), utilizing uniform preprocessing on both BARD10 and the benchmark corpora BAAD16 (Bangla Authorship Attribution Dataset of 16 authors). In all datasets, the classical TF-IDF + SVM baseline outperformed, attaining a macro-F1 score of 0.997 on BAAD16 and 0.921 on BARD10, while Bangla BERT lagged by as much as five points. This study reveals that BARD10 authors are highly sensitive to stop-word pruning, while BAAD16 authors remain comparatively robust highlighting genre-dependent reliance on stop-word signatures. Error analysis revealed that high frequency components transmit authorial signatures that are diminished or reduced by transformer models. Three insights are identified: Bangla stop-words serve as essential stylistic indicators; finely calibrated ML models prove effective within short-text limitations; and BARD10 connects formal literature with contemporary web dialogue, offering a reproducible benchmark for future long-context or domain-adapted transformers. 

---
# HyCoRA: Hyper-Contrastive Role-Adaptive Learning for Role-Playing 

**Authors**: Shihao Yang, Zhicong Lu, Yong Yang, Bo Lv, Yang Shen, Nayu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.08017)  

**Abstract**: Multi-character role-playing aims to equip models with the capability to simulate diverse roles. Existing methods either use one shared parameterized module across all roles or assign a separate parameterized module to each role. However, the role-shared module may ignore distinct traits of each role, weakening personality learning, while the role-specific module may overlook shared traits across multiple roles, hindering commonality modeling. In this paper, we propose a novel HyCoRA: Hyper-Contrastive Role-Adaptive learning framework, which efficiently improves multi-character role-playing ability by balancing the learning of distinct and shared traits. Specifically, we propose a Hyper-Half Low-Rank Adaptation structure, where one half is a role-specific module generated by a lightweight hyper-network, and the other half is a trainable role-shared module. The role-specific module is devised to represent distinct persona signatures, while the role-shared module serves to capture common traits. Moreover, to better reflect distinct personalities across different roles, we design a hyper-contrastive learning mechanism to help the hyper-network distinguish their unique characteristics. Extensive experimental results on both English and Chinese available benchmarks demonstrate the superiority of our framework. Further GPT-4 evaluations and visual analyses also verify the capability of HyCoRA to capture role characteristics. 

---
# Self-Correction Distillation for Structured Data Question Answering 

**Authors**: Yushan Zhu, Wen Zhang, Long Jin, Mengshu Sun, Ling Zhong, Zhiqiang Liu, Juan Li, Lei Liang, Chong Long, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2511.07998)  

**Abstract**: Structured data question answering (QA), including table QA, Knowledge Graph (KG) QA, and temporal KG QA, is a pivotal research area. Advances in large language models (LLMs) have driven significant progress in unified structural QA frameworks like TrustUQA. However, these frameworks face challenges when applied to small-scale LLMs since small-scale LLMs are prone to errors in generating structured queries. To improve the structured data QA ability of small-scale LLMs, we propose a self-correction distillation (SCD) method. In SCD, an error prompt mechanism (EPM) is designed to detect errors and provide customized error messages during inference, and a two-stage distillation strategy is designed to transfer large-scale LLMs' query-generation and error-correction capabilities to small-scale LLM. Experiments across 5 benchmarks with 3 structured data types demonstrate that our SCD achieves the best performance and superior generalization on small-scale LLM (8B) compared to other distillation methods, and closely approaches the performance of GPT4 on some datasets. Furthermore, large-scale LLMs equipped with EPM surpass the state-of-the-art results on most datasets. 

---
# State of the Art in Text Classification for South Slavic Languages: Fine-Tuning or Prompting? 

**Authors**: Taja Kuzman Pungeršek, Peter Rupnik, Ivan Porupski, Vuk Dinić, Nikola Ljubešić  

**Link**: [PDF](https://arxiv.org/pdf/2511.07989)  

**Abstract**: Until recently, fine-tuned BERT-like models provided state-of-the-art performance on text classification tasks. With the rise of instruction-tuned decoder-only models, commonly known as large language models (LLMs), the field has increasingly moved toward zero-shot and few-shot prompting. However, the performance of LLMs on text classification, particularly on less-resourced languages, remains under-explored. In this paper, we evaluate the performance of current language models on text classification tasks across several South Slavic languages. We compare openly available fine-tuned BERT-like models with a selection of open-source and closed-source LLMs across three tasks in three domains: sentiment classification in parliamentary speeches, topic classification in news articles and parliamentary speeches, and genre identification in web texts. Our results show that LLMs demonstrate strong zero-shot performance, often matching or surpassing fine-tuned BERT-like models. Moreover, when used in a zero-shot setup, LLMs perform comparably in South Slavic languages and English. However, we also point out key drawbacks of LLMs, including less predictable outputs, significantly slower inference, and higher computational costs. Due to these limitations, fine-tuned BERT-like models remain a more practical choice for large-scale automatic text annotation. 

---
# NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation 

**Authors**: Maoqi Liu, Quan Fang, Yuhao Wu, Can Zhao, Yang Yang, Kaiquan Cai  

**Link**: [PDF](https://arxiv.org/pdf/2511.07982)  

**Abstract**: Accurate interpretation of Notices to Airmen (NOTAMs) is critical for aviation safety, yet their condensed and cryptic language poses significant challenges to both manual and automated processing. Existing automated systems are typically limited to shallow parsing, failing to extract the actionable intelligence needed for operational decisions. We formalize the complete interpretation task as deep parsing, a dual-reasoning challenge requiring both dynamic knowledge grounding (linking the NOTAM to evolving real-world aeronautical data) and schema-based inference (applying static domain rules to deduce operational status). To tackle this challenge, we propose NOTAM-Evolve, a self-evolving framework that enables a large language model (LLM) to autonomously master complex NOTAM interpretation. Leveraging a knowledge graph-enhanced retrieval module for data grounding, the framework introduces a closed-loop learning process where the LLM progressively improves from its own outputs, minimizing the need for extensive human-annotated reasoning traces. In conjunction with this framework, we introduce a new benchmark dataset of 10,000 expert-annotated NOTAMs. Our experiments demonstrate that NOTAM-Evolve achieves a 30.4% absolute accuracy improvement over the base LLM, establishing a new state of the art on the task of structured NOTAM interpretation. 

---
# Unified Work Embeddings: Contrastive Learning of a Bidirectional Multi-task Ranker 

**Authors**: Matthias De Lange, Jens-Joris Decorte, Jeroen Van Hautte  

**Link**: [PDF](https://arxiv.org/pdf/2511.07969)  

**Abstract**: Workforce transformation across diverse industries has driven an increased demand for specialized natural language processing capabilities. Nevertheless, tasks derived from work-related contexts inherently reflect real-world complexities, characterized by long-tailed distributions, extreme multi-label target spaces, and scarce data availability. The rise of generalist embedding models prompts the question of their performance in the work domain, especially as progress in the field has focused mainly on individual tasks. To this end, we introduce WorkBench, the first unified evaluation suite spanning six work-related tasks formulated explicitly as ranking problems, establishing a common ground for multi-task progress. Based on this benchmark, we find significant positive cross-task transfer, and use this insight to compose task-specific bipartite graphs from real-world data, synthetically enriched through grounding. This leads to Unified Work Embeddings (UWE), a task-agnostic bi-encoder that exploits our training-data structure with a many-to-many InfoNCE objective, and leverages token-level embeddings with task-agnostic soft late interaction. UWE demonstrates zero-shot ranking performance on unseen target spaces in the work domain, enables low-latency inference by caching the task target space embeddings, and shows significant gains in macro-averaged MAP and RP@10 over generalist embedding models. 

---
# Distinct Theta Synchrony across Speech Modes: Perceived, Spoken, Whispered, and Imagined 

**Authors**: Jung-Sun Lee, Ha-Na Jo, Eunyeong Ko  

**Link**: [PDF](https://arxiv.org/pdf/2511.07918)  

**Abstract**: Human speech production encompasses multiple modes such as perceived, overt, whispered, and imagined, each reflecting distinct neural mechanisms. Among these, theta-band synchrony has been closely associated with language processing, attentional control, and inner speech. However, previous studies have largely focused on a single mode, such as overt speech, and have rarely conducted an integrated comparison of theta synchrony across different speech modes. In this study, we analyzed differences in theta-band neural synchrony across speech modes based on connectivity metrics, focusing on region-wise variations. The results revealed that overt and whispered speech exhibited broader and stronger frontotemporal synchrony, reflecting active motor-phonological coupling during overt articulation, whereas perceived speech showed dominant posterior and temporal synchrony patterns, consistent with auditory perception and comprehension processes. In contrast, imagined speech demonstrated a more spatially confined but internally coherent synchronization pattern, primarily involving frontal and supplementary motor regions. These findings indicate that the extent and spatial distribution of theta synchrony differ substantially across modes, with overt articulation engaging widespread cortical interactions, whispered speech showing intermediate engagement, and perception relying predominantly on temporoparietal networks. Therefore, this study aims to elucidate the differences in theta-band neural synchrony across various speech modes, thereby uncovering both the shared and distinct neural dynamics underlying language perception and imagined speech. 

---
# Social Media for Mental Health: Data, Methods, and Findings 

**Authors**: Nur Shazwani Kamarudin, Ghazaleh Beigi, Lydia Manikonda, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07914)  

**Abstract**: There is an increasing number of virtual communities and forums available on the web. With social media, people can freely communicate and share their thoughts, ask personal questions, and seek peer-support, especially those with conditions that are highly stigmatized, without revealing personal identity. We study the state-of-the-art research methodologies and findings on mental health challenges like de- pression, anxiety, suicidal thoughts, from the pervasive use of social media data. We also discuss how these novel thinking and approaches can help to raise awareness of mental health issues in an unprecedented way. Specifically, this chapter describes linguistic, visual, and emotional indicators expressed in user disclosures. The main goal of this chapter is to show how this new source of data can be tapped to improve medical practice, provide timely support, and influence government or policymakers. In the context of social media for mental health issues, this chapter categorizes social media data used, introduces different deployed machine learning, feature engineering, natural language processing, and surveys methods and outlines directions for future research. 

---
# Last Layer Logits to Logic: Empowering LLMs with Logic-Consistent Structured Knowledge Reasoning 

**Authors**: Songze Li, Zhiqiang Liu, Zhaoyan Gong, Xiaoke Guo, Zhengke Gui, Huajun Chen, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07910)  

**Abstract**: Large Language Models (LLMs) achieve excellent performance in natural language reasoning tasks through pre-training on vast unstructured text, enabling them to understand the logic in natural language and generate logic-consistent responses. However, the representational differences between unstructured and structured knowledge make LLMs inherently struggle to maintain logic consistency, leading to \textit{Logic Drift} challenges in structured knowledge reasoning tasks such as Knowledge Graph Question Answering (KGQA). Existing methods address this limitation by designing complex workflows embedded in prompts to guide LLM reasoning. Nevertheless, these approaches only provide input-level guidance and fail to fundamentally address the \textit{Logic Drift} in LLM outputs. Additionally, their inflexible reasoning workflows cannot adapt to different tasks and knowledge graphs. To enhance LLMs' logic consistency in structured knowledge reasoning, we specifically target the logits output from the autoregressive generation process. We propose the \textit{Logits-to-Logic} framework, which incorporates logits strengthening and logits filtering as core modules to correct logical defects in LLM outputs. Extensive experiments show that our approach significantly improves LLMs' logic consistency in structured knowledge reasoning and achieves state-of-the-art performance on multiple KGQA benchmarks. 

---
# Breaking the Adversarial Robustness-Performance Trade-off in Text Classification via Manifold Purification 

**Authors**: Chenhao Dang, Jing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.07888)  

**Abstract**: A persistent challenge in text classification (TC) is that enhancing model robustness against adversarial attacks typically degrades performance on clean data. We argue that this challenge can be resolved by modeling the distribution of clean samples in the encoder embedding manifold. To this end, we propose the Manifold-Correcting Causal Flow (MC^2F), a two-module system that operates directly on sentence embeddings. A Stratified Riemannian Continuous Normalizing Flow (SR-CNF) learns the density of the clean data manifold. It identifies out-of-distribution embeddings, which are then corrected by a Geodesic Purification Solver. This solver projects adversarial points back onto the learned manifold via the shortest path, restoring a clean, semantically coherent representation. We conducted extensive evaluations on text classification (TC) across three datasets and multiple adversarial attacks. The results demonstrate that our method, MC^2F, not only establishes a new state-of-the-art in adversarial robustness but also fully preserves performance on clean data, even yielding modest gains in accuracy. 

---
# Planned Event Forecasting using Future Mentions and Related Entity Extraction in News Articles 

**Authors**: Neelesh Kumar Shukla, Pranay Sanghvi  

**Link**: [PDF](https://arxiv.org/pdf/2511.07879)  

**Abstract**: In democracies like India, people are free to express their views and demands. Sometimes this causes situations of civil unrest such as protests, rallies, and marches. These events may be disruptive in nature and are often held without prior permission from the competent authority. Forecasting these events helps administrative officials take necessary action. Usually, protests are announced well in advance to encourage large participation. Therefore, by analyzing such announcements in news articles, planned events can be forecasted beforehand. We developed such a system in this paper to forecast social unrest events using topic modeling and word2vec to filter relevant news articles, and Named Entity Recognition (NER) methods to identify entities such as people, organizations, locations, and dates. Time normalization is applied to convert future date mentions into a standard format. In this paper, we have developed a geographically independent, generalized model to identify key features for filtering civil unrest events. There could be many mentions of entities, but only a few may actually be involved in the event. This paper calls such entities Related Entities and proposes a method to extract them, referred to as Related Entity Extraction. 

---
# AlignSurvey: A Comprehensive Benchmark for Human Preferences Alignment in Social Surveys 

**Authors**: Chenxi Lin, Weikang Yuan, Zhuoren Jiang, Biao Huang, Ruitao Zhang, Jianan Ge, Yueqian Xu, Jianxing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07871)  

**Abstract**: Understanding human attitudes, preferences, and behaviors through social surveys is essential for academic research and policymaking. Yet traditional surveys face persistent challenges, including fixed-question formats, high costs, limited adaptability, and difficulties ensuring cross-cultural equivalence. While recent studies explore large language models (LLMs) to simulate survey responses, most are limited to structured questions, overlook the entire survey process, and risks under-representing marginalized groups due to training data biases. We introduce AlignSurvey, the first benchmark that systematically replicates and evaluates the full social survey pipeline using LLMs. It defines four tasks aligned with key survey stages: social role modeling, semi-structured interview modeling, attitude stance modeling and survey response modeling. It also provides task-specific evaluation metrics to assess alignment fidelity, consistency, and fairness at both individual and group levels, with a focus on demographic diversity. To support AlignSurvey, we construct a multi-tiered dataset architecture: (i) the Social Foundation Corpus, a cross-national resource with 44K+ interview dialogues and 400K+ structured survey records; and (ii) a suite of Entire-Pipeline Survey Datasets, including the expert-annotated AlignSurvey-Expert (ASE) and two nationally representative surveys for cross-cultural evaluation. We release the SurveyLM family, obtained through two-stage fine-tuning of open-source LLMs, and offer reference models for evaluating domain-specific alignment. All datasets, models, and tools are available at github and huggingface to support transparent and socially responsible research. 

---
# From Experience to Strategy: Empowering LLM Agents with Trainable Graph Memory 

**Authors**: Siyu Xia, Zekun Xu, Jiajun Chai, Wentian Fan, Yan Song, Xiaohan Wang, Guojun Yin, Wei Lin, Haifeng Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07800)  

**Abstract**: Large Language Models (LLMs) based agents have demonstrated remarkable potential in autonomous task-solving across complex, open-ended environments. A promising approach for improving the reasoning capabilities of LLM agents is to better utilize prior experiences in guiding current decisions. However, LLMs acquire experience either through implicit memory via training, which suffers from catastrophic forgetting and limited interpretability, or explicit memory via prompting, which lacks adaptability. In this paper, we introduce a novel agent-centric, trainable, multi-layered graph memory framework and evaluate how context memory enhances the ability of LLMs to utilize parametric information. The graph abstracts raw agent trajectories into structured decision paths in a state machine and further distills them into high-level, human-interpretable strategic meta-cognition. In order to make memory adaptable, we propose a reinforcement-based weight optimization procedure that estimates the empirical utility of each meta-cognition based on reward feedback from downstream tasks. These optimized strategies are then dynamically integrated into the LLM agent's training loop through meta-cognitive prompting. Empirically, the learnable graph memory delivers robust generalization, improves LLM agents' strategic reasoning performance, and provides consistent benefits during Reinforcement Learning (RL) training. 

---
# Design, Results and Industry Implications of the World's First Insurance Large Language Model Evaluation Benchmark 

**Authors**: Hua Zhou, Bing Ma, Yufei Zhang, Yi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07794)  

**Abstract**: This paper comprehensively elaborates on the construction methodology, multi-dimensional evaluation system, and underlying design philosophy of CUFEInse v1.0. Adhering to the principles of "quantitative-oriented, expert-driven, and multi-validation," the benchmark establishes an evaluation framework covering 5 core dimensions, 54 sub-indicators, and 14,430 high-quality questions, encompassing insurance theoretical knowledge, industry understanding, safety and compliance, intelligent agent application, and logical rigor. Based on this benchmark, a comprehensive evaluation was conducted on 11 mainstream large language models. The evaluation results reveal that general-purpose models suffer from common bottlenecks such as weak actuarial capabilities and inadequate compliance adaptation. High-quality domain-specific training demonstrates significant advantages in insurance vertical scenarios but exhibits shortcomings in business adaptation and compliance. The evaluation also accurately identifies the common bottlenecks of current large models in professional scenarios such as insurance actuarial, underwriting and claim settlement reasoning, and compliant marketing copywriting. The establishment of CUFEInse not only fills the gap in professional evaluation benchmarks for the insurance field, providing academia and industry with a professional, systematic, and authoritative evaluation tool, but also its construction concept and methodology offer important references for the evaluation paradigm of large models in vertical fields, serving as an authoritative reference for academic model optimization and industrial model selection. Finally, the paper looks forward to the future iteration direction of the evaluation benchmark and the core development direction of "domain adaptation + reasoning enhancement" for insurance large models. 

---
# Back to the Future: The Role of Past and Future Context Predictability in Incremental Language Production 

**Authors**: Shiva Upadhye, Richard Futrell  

**Link**: [PDF](https://arxiv.org/pdf/2511.07752)  

**Abstract**: Contextual predictability shapes both the form and choice of words in online language production. The effects of the predictability of a word given its previous context are generally well-understood in both production and comprehension, but studies of naturalistic production have also revealed a poorly-understood backward predictability effect of a word given its future context, which may be related to future planning. Here, in two studies of naturalistic speech corpora, we investigate backward predictability effects using improved measures and more powerful language models, introducing a new principled and conceptually motivated information-theoretic predictability measure that integrates predictability from both the future and the past context. Our first study revisits classic predictability effects on word duration. Our second study investigates substitution errors within a generative framework that independently models the effects of lexical, contextual, and communicative factors on word choice, while predicting the actual words that surface as speech errors. We find that our proposed conceptually-motivated alternative to backward predictability yields qualitatively similar effects across both studies. Through a fine-grained analysis of substitution errors, we further show that different kinds of errors are suggestive of how speakers prioritize form, meaning, and context-based information during lexical planning. Together, these findings illuminate the functional roles of past and future context in how speakers encode and choose words, offering a bridge between contextual predictability effects and the mechanisms of sentence planning. 

---
# Critical Confabulation: Can LLMs Hallucinate for Social Good? 

**Authors**: Peiqi Sui, Eamon Duede, Hoyt Long, Richard Jean So  

**Link**: [PDF](https://arxiv.org/pdf/2511.07722)  

**Abstract**: LLMs hallucinate, yet some confabulations can have social affordances if carefully bounded. We propose critical confabulation (inspired by critical fabulation from literary and social theory), the use of LLM hallucinations to "fill-in-the-gap" for omissions in archives due to social and political inequality, and reconstruct divergent yet evidence-bound narratives for history's "hidden figures". We simulate these gaps with an open-ended narrative cloze task: asking LLMs to generate a masked event in a character-centric timeline sourced from a novel corpus of unpublished texts. We evaluate audited (for data contamination), fully-open models (the OLMo-2 family) and unaudited open-weight and proprietary baselines under a range of prompts designed to elicit controlled and useful hallucinations. Our findings validate LLMs' foundational narrative understanding capabilities to perform critical confabulation, and show how controlled and well-specified hallucinations can support LLM applications for knowledge production without collapsing speculation into a lack of historical accuracy and fidelity. 

---
# CAPO: Confidence Aware Preference Optimization Learning for Multilingual Preferences 

**Authors**: Rhitabrat Pokharel, Yufei Tao, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2511.07691)  

**Abstract**: Preference optimization is a critical post-training technique used to align large language models (LLMs) with human preferences, typically by fine-tuning on ranked response pairs. While methods like Direct Preference Optimization (DPO) have proven effective in English, they often fail to generalize robustly to multilingual settings. We propose a simple yet effective alternative, Confidence-Aware Preference Optimization (CAPO), which replaces DPO's fixed treatment of preference pairs with a dynamic loss scaling mechanism based on a relative reward. By modulating the learning signal according to the confidence in each preference pair, CAPO enhances robustness to noisy or low-margin comparisons, typically encountered in multilingual text. Empirically, CAPO outperforms existing preference optimization baselines by at least 16% in reward accuracy, and improves alignment by widening the gap between preferred and dispreferred responses across languages. 

---
# Stress Testing Factual Consistency Metrics for Long-Document Summarization 

**Authors**: Zain Muhammad Mujahid, Dustin Wright, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2511.07689)  

**Abstract**: Evaluating the factual consistency of abstractive text summarization remains a significant challenge, particularly for long documents, where conventional metrics struggle with input length limitations and long-range dependencies. In this work, we systematically evaluate the reliability of six widely used reference-free factuality metrics, originally proposed for short-form summarization, in the long-document setting. We probe metric robustness through seven factuality-preserving perturbations applied to summaries, namely paraphrasing, simplification, synonym replacement, logically equivalent negations, vocabulary reduction, compression, and source text insertion, and further analyze their sensitivity to retrieval context and claim information density. Across three long-form benchmark datasets spanning science fiction, legal, and scientific domains, our results reveal that existing short-form metrics produce inconsistent scores for semantically equivalent summaries and exhibit declining reliability for information-dense claims whose content is semantically similar to many parts of the source document. While expanding the retrieval context improves stability in some domains, no metric consistently maintains factual alignment under long-context conditions. Finally, our results highlight concrete directions for improving factuality evaluation, including multi-span reasoning, context-aware calibration, and training on meaning-preserving variations to enhance robustness in long-form summarization. We release all code, perturbed data, and scripts required to reproduce our results at this https URL. 

---
# Revisiting NLI: Towards Cost-Effective and Human-Aligned Metrics for Evaluating LLMs in Question Answering 

**Authors**: Sai Shridhar Balamurali, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.07659)  

**Abstract**: Evaluating answers from state-of-the-art large language models (LLMs) is challenging: lexical metrics miss semantic nuances, whereas "LLM-as-Judge" scoring is computationally expensive. We re-evaluate a lightweight alternative -- off-the-shelf Natural Language Inference (NLI) scoring augmented by a simple lexical-match flag and find that this decades-old technique matches GPT-4o's accuracy (89.9%) on long-form QA, while requiring orders-of-magnitude fewer parameters. To test human alignment of these metrics rigorously, we introduce DIVER-QA, a new 3000-sample human-annotated benchmark spanning five QA datasets and five candidate LLMs. Our results highlight that inexpensive NLI-based evaluation remains competitive and offer DIVER-QA as an open resource for future metric research. 

---
# LLMs vs. Traditional Sentiment Tools in Psychology: An Evaluation on Belgian-Dutch Narratives 

**Authors**: Ratna Kandala, Katie Hoemann  

**Link**: [PDF](https://arxiv.org/pdf/2511.07641)  

**Abstract**: Understanding emotional nuances in everyday language is crucial for computational linguistics and emotion research. While traditional lexicon-based tools like LIWC and Pattern have served as foundational instruments, Large Language Models (LLMs) promise enhanced context understanding. We evaluated three Dutch-specific LLMs (ChocoLlama-8B-Instruct, Reynaerde-7B-chat, and GEITje-7B-ultra) against LIWC and Pattern for valence prediction in Flemish, a low-resource language variant. Our dataset comprised approximately 25000 spontaneous textual responses from 102 Dutch-speaking participants, each providing narratives about their current experiences with self-assessed valence ratings (-50 to +50). Surprisingly, despite architectural advancements, the Dutch-tuned LLMs underperformed compared to traditional methods, with Pattern showing superior performance. These findings challenge assumptions about LLM superiority in sentiment analysis tasks and highlight the complexity of capturing emotional valence in spontaneous, real-world narratives. Our results underscore the need for developing culturally and linguistically tailored evaluation frameworks for low-resource language variants, while questioning whether current LLM fine-tuning approaches adequately address the nuanced emotional expressions found in everyday language use. 

---
# LLM Optimization Unlocks Real-Time Pairwise Reranking 

**Authors**: Jingyu Wu, Aditya Shrivastava, Jing Zhu, Alfy Samuel, Anoop Kumar, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07555)  

**Abstract**: Efficiently reranking documents retrieved from information retrieval (IR) pipelines to enhance overall quality of Retrieval-Augmented Generation (RAG) system remains an important yet challenging problem. Recent studies have highlighted the importance of Large Language Models (LLMs) in reranking tasks. In particular, Pairwise Reranking Prompting (PRP) has emerged as a promising plug-and-play approach due to its usability and effectiveness. However, the inherent complexity of the algorithm, coupled with the high computational demands and latency incurred due to LLMs, raises concerns about its feasibility in real-time applications. To address these challenges, this paper presents a focused study on pairwise reranking, demonstrating that carefully applied optimization methods can significantly mitigate these issues. By implementing these methods, we achieve a remarkable latency reduction of up to 166 times, from 61.36 seconds to 0.37 seconds per query, with an insignificant drop in performance measured by Recall@k. Our study highlights the importance of design choices that were previously overlooked, such as using smaller models, limiting the reranked set, using lower precision, reducing positional bias with one-directional order inference, and restricting output tokens. These optimizations make LLM-based reranking substantially more efficient and feasible for latency-sensitive, real-world deployments. 

---
# Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models 

**Authors**: Xin Liu, Qiyang Song, Qihang Zhou, Haichao Du, Shaowen Xu, Wenbo Jiang, Weijuan Zhang, Xiaoqi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2511.07498)  

**Abstract**: Large language models (LLMs) increasingly support multilingual understanding and generation. Meanwhile, efforts to interpret their internal mechanisms have emerged, offering insights to enhance multilingual performance. While multi-head self-attention (MHA) has proven critical in many areas, its role in multilingual capabilities remains underexplored. In this work, we study the contribution of MHA in supporting multilingual processing in LLMs. We propose Language Attention Head Importance Scores (LAHIS), an effective and efficient method that identifies attention head importance for multilingual capabilities via a single forward and backward pass through the LLM. Applying LAHIS to Aya-23-8B, Llama-3.2-3B, and Mistral-7B-v0.1, we reveal the existence of both language-specific and language-general heads. Language-specific heads enable cross-lingual attention transfer to guide the model toward target language contexts and mitigate off-target language generation issue, contributing to addressing challenges in multilingual LLMs. We also introduce a lightweight adaptation that learns a soft head mask to modulate attention outputs over language heads, requiring only 20 tunable parameters to improve XQuAD accuracy. Overall, our work enhances both the interpretability and multilingual capabilities of LLMs from the perspective of MHA. 

---
# Motif 2 12.7B technical report 

**Authors**: Junghwan Lim, Sungmin Lee, Dongseok Kim, Taehyun Kim, Eunhwan Park, Jeesoo Lee, Jeongdoo Lee, Junhyeok Lee, Wai Ting Cheung, Dahye Choi, Jaeheui Her, Jaeyeon Huh, Hanbin Jung, Changjin Kang, Beomgyu Kim, Minjae Kim, Taewhan Kim, Youngrok Kim, Hyukjin Kweon, Haesol Lee, Kungyu Lee, Dongpin Oh, Yeongjae Park, Bokki Ryu, Dongjoo Weon  

**Link**: [PDF](https://arxiv.org/pdf/2511.07464)  

**Abstract**: We introduce Motif-2-12.7B, a new open-weight foundation model that pushes the efficiency frontier of large language models by combining architectural innovation with system-level optimization. Designed for scalable language understanding and robust instruction generalization under constrained compute budgets, Motif-2-12.7B builds upon Motif-2.6B with the integration of Grouped Differential Attention (GDA), which improves representational efficiency by disentangling signal and noise-control attention pathways. The model is pre-trained on 5.5 trillion tokens spanning diverse linguistic, mathematical, scientific, and programming domains using a curriculum-driven data scheduler that gradually changes the data composition ratio. The training system leverages the MuonClip optimizer alongside custom high-performance kernels, including fused PolyNorm activations and the Parallel Muon algorithm, yielding significant throughput and memory efficiency gains in large-scale distributed environments. Post-training employs a three-stage supervised fine-tuning pipeline that successively enhances general instruction adherence, compositional understanding, and linguistic precision. Motif-2-12.7B demonstrates competitive performance across diverse benchmarks, showing that thoughtful architectural scaling and optimized training design can rival the capabilities of much larger models. 

---
# It Takes Two: A Dual Stage Approach for Terminology-Aware Translation 

**Authors**: Akshat Singh Jaswal  

**Link**: [PDF](https://arxiv.org/pdf/2511.07461)  

**Abstract**: This paper introduces DuTerm, a novel two-stage architecture for terminology-constrained machine translation. Our system combines a terminology-aware NMT model, adapted via fine-tuning on large-scale synthetic data, with a prompt-based LLM for post-editing. The LLM stage refines NMT output and enforces terminology adherence. We evaluate DuTerm on English-to German, English-to-Spanish, and English-to-Russian with the WMT 2025 Terminology Shared Task corpus. We demonstrate that flexible, context-driven terminology handling by the LLM consistently yields higher quality translations than strict constraint enforcement. Our results highlight a critical trade-off, revealing that an LLM's work best for high-quality translation as context-driven mutators rather than generators. 

---
# REFLEX: Reference-Free Evaluation of Log Summarization via Large Language Model Judgment 

**Authors**: Priyanka Mudgal  

**Link**: [PDF](https://arxiv.org/pdf/2511.07458)  

**Abstract**: Evaluating log summarization systems is challenging due to the lack of high-quality reference summaries and the limitations of existing metrics like ROUGE and BLEU, which depend on surface-level lexical overlap. We introduce REFLEX, a reference-free evaluation metric for log summarization based on large language model (LLM) judgment. REFLEX uses LLMs as zero-shot evaluators to assess summary quality along dimensions such as relevance, informativeness, and coherence, without requiring gold-standard references or human annotations. We show that REFLEX produces stable, interpretable, and fine-grained evaluations across multiple log summarization dataset, and more effectively distinguishes model outputs than traditional metrics. REFLEX provides a scalable alternative for evaluating log summaries in real-world settings where reference data is scarce or unavailable. 

---
# GRIP: In-Parameter Graph Reasoning through Fine-Tuning Large Language Models 

**Authors**: Jiarui Feng, Donghong Cai, Yixin Chen, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07457)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in modeling sequential textual data and generalizing across diverse tasks. However, adapting LLMs to effectively handle structural data, such as knowledge graphs or web data, remains a challenging problem. Some approaches adopt complex strategies to convert graphs into text sequences, resulting in significant token overhead and rendering them impractical for large-scale graphs. Others introduce additional modules to encode graphs into fixed-size token representations for LLMs. However, these methods typically require large-scale post-training on graph-text corpus and complex alignment procedures, yet often yield sub-optimal results due to poor modality alignment. Inspired by in-parameter knowledge injection for test-time adaptation of LLMs, we propose GRIP, a novel framework that equips LLMs with the ability to internalize complex relational information from graphs through carefully designed fine-tuning tasks. This knowledge is efficiently stored within lightweight LoRA parameters, enabling the fine-tuned LLM to perform a wide range of graph-related tasks without requiring access to the original graph at inference time. Extensive experiments across multiple benchmarks validate the effectiveness and efficiency of our approach. 

---
# Large Language Models for Scientific Idea Generation: A Creativity-Centered Survey 

**Authors**: Fatemeh Shahhosseini, Arash Marioriyad, Ali Momen, Mahdieh Soleymani Baghshah, Mohammad Hossein Rohban, Shaghayegh Haghjooy Javanmard  

**Link**: [PDF](https://arxiv.org/pdf/2511.07448)  

**Abstract**: Scientific idea generation lies at the heart of scientific discovery and has driven human progress-whether by solving unsolved problems or proposing novel hypotheses to explain unknown phenomena. Unlike standard scientific reasoning or general creative generation, idea generation in science is a multi-objective and open-ended task, where the novelty of a contribution is as essential as its empirical soundness. Large language models (LLMs) have recently emerged as promising generators of scientific ideas, capable of producing coherent and factual outputs with surprising intuition and acceptable reasoning, yet their creative capacity remains inconsistent and poorly understood. This survey provides a structured synthesis of methods for LLM-driven scientific ideation, examining how different approaches balance creativity with scientific soundness. We categorize existing methods into five complementary families: External knowledge augmentation, Prompt-based distributional steering, Inference-time scaling, Multi-agent collaboration, and Parameter-level adaptation. To interpret their contributions, we employ two complementary frameworks: Boden's taxonomy of Combinatorial, Exploratory and Transformational creativity to characterize the level of ideas each family expected to generate, and Rhodes' 4Ps framework-Person, Process, Press, and Product-to locate the aspect or source of creativity that each method emphasizes. By aligning methodological advances with creativity frameworks, this survey clarifies the state of the field and outlines key directions toward reliable, systematic, and transformative applications of LLMs in scientific discovery. 

---
# A Preliminary Study of RAG for Taiwanese Historical Archives 

**Authors**: Claire Lin, Bo-Han Feng, Xuanjun Chen, Te-Lun Yang, Hung-yi Lee, Jyh-Shing Roger Jang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07445)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising approach for knowledge-intensive tasks. However, few studies have examined RAG for Taiwanese Historical Archives. In this paper, we present an initial study of a RAG pipeline applied to two historical Traditional Chinese datasets, Fort Zeelandia and the Taiwan Provincial Council Gazette, along with their corresponding open-ended query sets. We systematically investigate the effects of query characteristics and metadata integration strategies on retrieval quality, answer generation, and the performance of the overall system. The results show that early-stage metadata integration enhances both retrieval and answer accuracy while also revealing persistent challenges for RAG systems, including hallucinations during generation and difficulties in handling temporal or multi-hop historical queries. 

---
# How Brittle is Agent Safety? Rethinking Agent Risk under Intent Concealment and Task Complexity 

**Authors**: Zihan Ma, Dongsheng Zhu, Shudong Liu, Taolin Zhang, Junnan Liu, Qingqiu Li, Minnan Luo, Songyang Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.08487)  

**Abstract**: Current safety evaluations for LLM-driven agents primarily focus on atomic harms, failing to address sophisticated threats where malicious intent is concealed or diluted within complex tasks. We address this gap with a two-dimensional analysis of agent safety brittleness under the orthogonal pressures of intent concealment and task complexity. To enable this, we introduce OASIS (Orthogonal Agent Safety Inquiry Suite), a hierarchical benchmark with fine-grained annotations and a high-fidelity simulation sandbox. Our findings reveal two critical phenomena: safety alignment degrades sharply and predictably as intent becomes obscured, and a "Complexity Paradox" emerges, where agents seem safer on harder tasks only due to capability limitations. By releasing OASIS and its simulation environment, we provide a principled foundation for probing and strengthening agent safety in these overlooked dimensions. 

---
# Generative Artificial Intelligence in Qualitative Research Methods: Between Hype and Risks? 

**Authors**: Maria Couto Teixeira, Marisa Tschopp, Anna Jobin  

**Link**: [PDF](https://arxiv.org/pdf/2511.08461)  

**Abstract**: As Artificial Intelligence (AI) is increasingly promoted and used in qualitative research, it also raises profound methodological issues. This position paper critically interrogates the role of generative AI (genAI) in the context of qualitative coding methodologies. Despite widespread hype and claims of efficiency, we propose that genAI is not methodologically valid within qualitative inquiries, and its use risks undermining the robustness and trustworthiness of qualitative research. The lack of meaningful documentation, commercial opacity, and the inherent tendencies of genAI systems to produce incorrect outputs all contribute to weakening methodological rigor. Overall, the balance between risk and benefits does not support the use of genAI in qualitative research, and our position paper cautions researchers to put sound methodology before technological novelty. 

---
# Unifying Model and Layer Fusion for Speech Foundation Models 

**Authors**: Yi-Jen Shih, David Harwath  

**Link**: [PDF](https://arxiv.org/pdf/2511.08389)  

**Abstract**: Speech Foundation Models have gained significant attention recently. Prior works have shown that the fusion of representations from multiple layers of the same model or the fusion of multiple models can improve performance on downstream tasks. We unify these two fusion strategies by proposing an interface module that enables fusion across multiple upstream speech models while integrating information across their layers. We conduct extensive experiments on different self-supervised and supervised models across various speech tasks, including ASR and paralinguistic analysis, and demonstrate that our method outperforms prior fusion approaches. We further analyze its scalability concerning model size and count, highlighting the importance of selecting appropriate upstream models. Our results show that the proposed interface provides an additional performance boost when given a suitable upstream model selection, making it a promising approach for utilizing Speech Foundation Models. 

---
# Hybrid Quantum-Classical Selective State Space Artificial Intelligence 

**Authors**: Amin Ebrahimi, Farzan Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2511.08349)  

**Abstract**: Hybrid Quantum Classical (HQC) algorithms constitute one of the most effective paradigms for exploiting the computational advantages of quantum systems in large-scale numerical tasks. By operating in high-dimensional Hilbert spaces, quantum circuits enable exponential speed-ups and provide access to richer representations of cost landscapes compared to purely classical methods. These capabilities are particularly relevant for machine learning, where state-of-the-art models especially in Natural Language Processing (NLP) suffer from prohibitive time complexity due to massive matrix multiplications and high-dimensional optimization.
In this manuscript, we propose a Hybrid Quantum Classical selection mechanism for the Mamba architecture, designed specifically for temporal sequence classification problems. Our approach leverages Variational Quantum Circuits (VQCs) as quantum gating modules that both enhance feature extraction and improve suppression of irrelevant information. This integration directly addresses the computational bottlenecks of deep learning architectures by exploiting quantum resources for more efficient representation learning.
We analyze how introducing quantum subroutines into large language models (LLMs) impacts their generalization capability, expressivity, and parameter efficiency. The results highlight the potential of quantum-enhanced gating mechanisms as a path toward scalable, resource-efficient NLP models, in a limited simulation step. Within the first four epochs on a reshaped MNIST dataset with input format (batch, 784, d_model), our hybrid model achieved 24.6% accuracy while using one quantum layer and achieve higher expressivity, compared to 21.6% obtained by a purely classical selection mechanism. we state No founding 

---
# Multi-Agent GraphRAG: A Text-to-Cypher Framework for Labeled Property Graphs 

**Authors**: Anton Gusarov, Anastasia Volkova, Valentin Khrulkov, Andrey Kuznetsov, Evgenii Maslov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2511.08274)  

**Abstract**: While Retrieval-Augmented Generation (RAG) methods commonly draw information from unstructured documents, the emerging paradigm of GraphRAG aims to leverage structured data such as knowledge graphs. Most existing GraphRAG efforts focus on Resource Description Framework (RDF) knowledge graphs, relying on triple representations and SPARQL queries. However, the potential of Cypher and Labeled Property Graph (LPG) databases to serve as scalable and effective reasoning engines within GraphRAG pipelines remains underexplored in current research literature. To fill this gap, we propose Multi-Agent GraphRAG, a modular LLM agentic system for text-to-Cypher query generation serving as a natural language interface to LPG-based graph data. Our proof-of-concept system features an LLM-based workflow for automated Cypher queries generation and execution, using Memgraph as the graph database backend. Iterative content-aware correction and normalization, reinforced by an aggregated feedback loop, ensures both semantic and syntactic refinement of generated queries. We evaluate our system on the CypherBench graph dataset covering several general domains with diverse types of queries. In addition, we demonstrate performance of the proposed workflow on a property graph derived from the IFC (Industry Foundation Classes) data, representing a digital twin of a building. This highlights how such an approach can bridge AI with real-world applications at scale, enabling industrial digital automation use cases. 

---
# Towards Outcome-Oriented, Task-Agnostic Evaluation of AI Agents 

**Authors**: Waseem AlShikh, Muayad Sayed Ali, Brian Kennedy, Dmytro Mozolevskyi  

**Link**: [PDF](https://arxiv.org/pdf/2511.08242)  

**Abstract**: As AI agents proliferate across industries and applications, evaluating their performance based solely on infrastructural metrics such as latency, time-to-first-token, or token throughput is proving insufficient. These metrics fail to capture the quality of an agent's decisions, its operational autonomy, or its ultimate business value. This white paper proposes a novel, comprehensive framework of eleven outcome-based, task-agnostic performance metrics for AI agents that transcend domain boundaries. These metrics are designed to enable organizations to evaluate agents based on the quality of their decisions, their degree of autonomy, their adaptability to new challenges, and the tangible business value they deliver, regardless of the underlying model architecture or specific use case. We introduce metrics such as Goal Completion Rate (GCR), Autonomy Index (AIx), Multi-Step Task Resilience (MTR), and Business Impact Efficiency (BIE). Through a large-scale simulated experiment involving four distinct agent architectures (ReAct, Chain-of-Thought, Tool-Augmented, Hybrid) across five diverse domains (Healthcare, Finance, Marketing, Legal, and Customer Service), we demonstrate the framework's efficacy. Our results reveal significant performance trade-offs between different agent designs, highlighting the Hybrid Agent as the most consistently high-performing model across the majority of our proposed metrics, achieving an average Goal Completion Rate of 88.8\% and the highest Return on Investment (ROI). This work provides a robust, standardized methodology for the holistic evaluation of AI agents, paving the way for more effective development, deployment, and governance. 

---
# SciAgent: A Unified Multi-Agent System for Generalistic Scientific Reasoning 

**Authors**: Xuchen Li, Ruitao Wu, Xuanbo Liu, Xukai Wang, Jinbo Hu, Zhixin Bai, Bohan Zeng, Hao Liang, Leheng Chen, Mingrui Chen, Haitian Zhong, Xuanlin Yang, Xu-Yao Zhang, Liu Liu, Jia Li, Kaiqi Huang, Jiahao Xu, Haitao Mi, Wentao Zhang, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2511.08151)  

**Abstract**: Recent advances in large language models have enabled AI systems to achieve expert-level performance on domain-specific scientific tasks, yet these systems remain narrow and handcrafted. We introduce SciAgent, a unified multi-agent system designed for generalistic scientific reasoning-the ability to adapt reasoning strategies across disciplines and difficulty levels. SciAgent organizes problem solving as a hierarchical process: a Coordinator Agent interprets each problem's domain and complexity, dynamically orchestrating specialized Worker Systems, each composed of interacting reasoning Sub-agents for symbolic deduction, conceptual modeling, numerical computation, and verification. These agents collaboratively assemble and refine reasoning pipelines tailored to each task. Across mathematics and physics Olympiads (IMO, IMC, IPhO, CPhO), SciAgent consistently attains or surpasses human gold-medalist performance, demonstrating both domain generality and reasoning adaptability. Additionally, SciAgent has been tested on the International Chemistry Olympiad (IChO) and selected problems from the Humanity's Last Exam (HLE) benchmark, further confirming the system's ability to generalize across diverse scientific domains. This work establishes SciAgent as a concrete step toward generalistic scientific intelligence-AI systems capable of coherent, cross-disciplinary reasoning at expert levels. 

---
# PerspAct: Enhancing LLM Situated Collaboration Skills through Perspective Taking and Active Vision 

**Authors**: Sabrina Patania, Luca Annese, Anita Pellegrini, Silvia Serino, Anna Lambiase, Luca Pallonetto, Silvia Rossi, Simone Colombani, Tom Foulsham, Azzurra Ruggeri, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2511.08098)  

**Abstract**: Recent advances in Large Language Models (LLMs) and multimodal foundation models have significantly broadened their application in robotics and collaborative systems. However, effective multi-agent interaction necessitates robust perspective-taking capabilities, enabling models to interpret both physical and epistemic viewpoints. Current training paradigms often neglect these interactive contexts, resulting in challenges when models must reason about the subjectivity of individual perspectives or navigate environments with multiple observers. This study evaluates whether explicitly incorporating diverse points of view using the ReAct framework, an approach that integrates reasoning and acting, can enhance an LLM's ability to understand and ground the demands of other agents. We extend the classic Director task by introducing active visual exploration across a suite of seven scenarios of increasing perspective-taking complexity. These scenarios are designed to challenge the agent's capacity to resolve referential ambiguity based on visual access and interaction, under varying state representations and prompting strategies, including ReAct-style reasoning. Our results demonstrate that explicit perspective cues, combined with active exploration strategies, significantly improve the model's interpretative accuracy and collaborative effectiveness. These findings highlight the potential of integrating active perception with perspective-taking mechanisms in advancing LLMs' application in robotics and multi-agent systems, setting a foundation for future research into adaptive and context-aware AI systems. 

---
# Quantizing Whisper-small: How design choices affect ASR performance 

**Authors**: Arthur Söhler, Julian Irigoyen, Andreas Søeborg Kirkedal  

**Link**: [PDF](https://arxiv.org/pdf/2511.08093)  

**Abstract**: Large speech recognition models like Whisper-small achieve high accuracy but are difficult to deploy on edge devices due to their high computational demand. To this end, we present a unified, cross-library evaluation of post-training quantization (PTQ) on Whisper-small that disentangles the impact of quantization scheme, method, granularity, and bit-width. Our study is based on four libraries: PyTorch, Optimum-Quanto, HQQ, and bitsandbytes. Experiments on LibriSpeech test-clean and test-other show that dynamic int8 quantization with Quanto offers the best trade-off, reducing model size by 57% while improving on the baseline's word error rate. Static quantization performed worse, likely due to Whisper's Transformer architecture, while more aggressive formats (e.g., nf4, int3) achieved up to 71% compression at the cost of accuracy in noisy conditions. Overall, our results demonstrate that carefully chosen PTQ methods can substantially reduce model size and inference cost without retraining, enabling efficient deployment of Whisper-small on constrained hardware. 

---
# Pruning as Regularization: Sensitivity-Aware One-Shot Pruning in ASR 

**Authors**: Julian Irigoyen, Arthur Söhler, Andreas Søeborg Kirkedal  

**Link**: [PDF](https://arxiv.org/pdf/2511.08092)  

**Abstract**: We challenge the conventional view of neural network pruning as solely a compression technique, demonstrating that one-shot magnitude pruning serves as a powerful implicit regularizer for ASR. Using Whisper-small, we combine gradient- and Fisher-based sensitivity diagnostics with targeted, component-wise pruning. This reveals architectural asymmetries: decoder FFNs are pruning-fragile, whereas decoder self-attention and the last encoder layers contain redundancy that, when removed, improves generalization. Without fine-tuning, pruning 50% of decoder self-attention reduces WER by 2.38% absolute (20.44% relative) on LibriSpeech test-other; pruning the last four encoder layers at 50% instead yields a 1.72% absolute (14.8% relative) improvement. Gains persisted on Common Voice and TED-LIUM datasets. Beyond regularization benefits, our sensitivity-aware approach enables more aggressive one-shot compression. At 40% sparsity, where established global pruning approaches catastrophically fail, our method preserves near-baseline accuracy. This positions pruning as a first-class architectural design tool: knowing where to prune is as important as how much to prune. 

---
# Information Capacity: Evaluating the Efficiency of Large Language Models via Text Compression 

**Authors**: Cheng Yuan, Jiawei Shao, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.08066)  

**Abstract**: Recent years have witnessed the rapid advancements of large language models (LLMs) and their expanding applications, leading to soaring demands for computational resources. The widespread adoption of test-time scaling further aggravates the tension between model capability and resource consumption, highlighting the importance of inference efficiency. However, a unified metric that accurately reflects an LLM's efficiency across different model sizes and architectures remains absent. Motivated by the correlation between compression and intelligence, we introduce information capacity, a measure of model efficiency based on text compression performance relative to computational complexity. Larger models can predict the next token more accurately, achieving greater compression gains but at higher computational costs. Empirical evaluations on mainstream open-source models show that models of varying sizes within a series exhibit consistent information capacity. This metric enables a fair efficiency comparison across model series and accurate performance prediction within a model series. A distinctive feature of information capacity is that it incorporates tokenizer efficiency, which affects both input and output token counts but is often neglected in LLM evaluations. We assess the information capacity of 49 models on 5 heterogeneous datasets and observe consistent results on the influences of tokenizer efficiency, pretraining data, and the mixture-of-experts architecture. 

---
# Dual-Process Scaffold Reasoning for Enhancing LLM Code Debugging 

**Authors**: Po-Chung Hsieh, Chin-Po Chen, Jeng-Lin Li, Ming-Ching Chang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08052)  

**Abstract**: Recent LLMs have demonstrated sophisticated problem-solving capabilities on various benchmarks through advanced reasoning algorithms. However, the key research question of identifying reasoning steps that balance complexity and computational efficiency remains unsolved. Recent research has increasingly drawn upon psychological theories to explore strategies for optimizing cognitive pathways. The LLM's final outputs and intermediate steps are regarded as System 1 and System 2, respectively. However, an in-depth exploration of the System 2 reasoning is still lacking. Therefore, we propose a novel psychologically backed Scaffold Reasoning framework for code debugging, which encompasses the Scaffold Stream, Analytic Stream, and Integration Stream. The construction of reference code within the Scaffold Stream is integrated with the buggy code analysis results produced by the Analytic Stream through the Integration Stream. Our framework achieves an 88.91% pass rate and an average inference time of 5.36 seconds per-problem on DebugBench, outperforming other reasoning approaches across various LLMs in both reasoning accuracy and efficiency. Further analyses elucidate the advantages and limitations of various cognitive pathways across varying problem difficulties and bug types. Our findings also corroborate the alignment of the proposed Scaffold Reasoning framework with human cognitive processes. 

---
# DynaAct: Large Language Model Reasoning with Dynamic Action Spaces 

**Authors**: Xueliang Zhao, Wei Wu, Jian Guan, Qintong Li, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2511.08043)  

**Abstract**: In modern sequential decision-making systems, the construction of an optimal candidate action space is critical to efficient inference. However, existing approaches either rely on manually defined action spaces that lack scalability or utilize unstructured spaces that render exhaustive search computationally prohibitive. In this paper, we propose a novel framework named \textsc{DynaAct} for automatically constructing a compact action space to enhance sequential reasoning in complex problem-solving scenarios. Our method first estimates a proxy for the complete action space by extracting general sketches observed in a corpus covering diverse complex reasoning problems using large language models. We then formulate a submodular function that jointly evaluates candidate actions based on their utility to the current state and their diversity, and employ a greedy algorithm to select an optimal candidate set. Extensive experiments on six diverse standard benchmarks demonstrate that our approach significantly improves overall performance, while maintaining efficient inference without introducing substantial latency. The implementation is available at this https URL. 

---
# BiCA: Effective Biomedical Dense Retrieval with Citation-Aware Hard Negatives 

**Authors**: Aarush Sinha, Pavan Kumar S, Roshan Balaji, Nirav Pravinbhai Bhatt  

**Link**: [PDF](https://arxiv.org/pdf/2511.08029)  

**Abstract**: Hard negatives are essential for training effective retrieval models. Hard-negative mining typically relies on ranking documents using cross-encoders or static embedding models based on similarity metrics such as cosine distance. Hard negative mining becomes challenging for biomedical and scientific domains due to the difficulty in distinguishing between source and hard negative documents. However, referenced documents naturally share contextual relevance with the source document but are not duplicates, making them well-suited as hard negatives. In this work, we propose BiCA: Biomedical Dense Retrieval with Citation-Aware Hard Negatives, an approach for hard-negative mining by utilizing citation links in 20,000 PubMed articles for improving a domain-specific small dense retriever. We fine-tune the GTE_small and GTE_Base models using these citation-informed negatives and observe consistent improvements in zero-shot dense retrieval using nDCG@10 for both in-domain and out-of-domain tasks on BEIR and outperform baselines on long-tailed topics in LoTTE using Success@5. Our findings highlight the potential of leveraging document link structure to generate highly informative negatives, enabling state-of-the-art performance with minimal fine-tuning and demonstrating a path towards highly data-efficient domain adaptation. 

---
# Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction 

**Authors**: Jun Xu, Xinkai Du, Yu Ao, Peilong Zhao, Yang Li, Ling Zhong, Lin Yuan, Zhongpu Bo, Xiaorui Wang, Mengshu Sun, Zhengke Gui, Dalong Zhang, Zhaoyang Wang, Qiwei Wang, Yangyang Hou, Zhiying Yin, Haofen Wang, Huajun Chen, Lei Liang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.07943)  

**Abstract**: Efficient retrieval of external knowledge bases and web pages is crucial for enhancing the reasoning abilities of LLMs. Previous works on training LLMs to leverage external retrievers for solving complex problems have predominantly employed end-to-end reinforcement learning. However, these approaches neglect supervision over the reasoning process, making it difficult to guarantee logical coherence and rigor. To address these limitations, we propose Thinker, a hierarchical thinking model for deep search through multi-turn interaction, making the reasoning process supervisable and verifiable. It decomposes complex problems into independently solvable sub-problems, each dually represented in both natural language and an equivalent logical function to support knowledge base and web searches. Concurrently, dependencies between sub-problems are passed as parameters via these logical functions, enhancing the logical coherence of the problem-solving process. To avoid unnecessary external searches, we perform knowledge boundary determination to check if a sub-problem is within the LLM's intrinsic knowledge, allowing it to answer directly. Experimental results indicate that with as few as several hundred training samples, the performance of Thinker is competitive with established baselines. Furthermore, when scaled to the full training set, Thinker significantly outperforms these methods across various datasets and model sizes. The source code is available at this https URL. 

---
# SpeechJudge: Towards Human-Level Judgment for Speech Naturalness 

**Authors**: Xueyao Zhang, Chaoren Wang, Huan Liao, Ziniu Li, Yuancheng Wang, Li Wang, Dongya Jia, Yuanzhe Chen, Xiulin Li, Zhuo Chen, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07931)  

**Abstract**: Aligning large generative models with human feedback is a critical challenge. In speech synthesis, this is particularly pronounced due to the lack of a large-scale human preference dataset, which hinders the development of models that truly align with human perception. To address this, we introduce SpeechJudge, a comprehensive suite comprising a dataset, a benchmark, and a reward model centered on naturalness--one of the most fundamental subjective metrics for speech synthesis. First, we present SpeechJudge-Data, a large-scale human feedback corpus of 99K speech pairs. The dataset is constructed using a diverse set of advanced zero-shot text-to-speech (TTS) models across diverse speech styles and multiple languages, with human annotations for both intelligibility and naturalness preference. From this, we establish SpeechJudge-Eval, a challenging benchmark for speech naturalness judgment. Our evaluation reveals that existing metrics and AudioLLMs struggle with this task; the leading model, Gemini-2.5-Flash, achieves less than 70% agreement with human judgment, highlighting a significant gap for improvement. To bridge this gap, we develop SpeechJudge-GRM, a generative reward model (GRM) based on Qwen2.5-Omni-7B. It is trained on SpeechJudge-Data via a two-stage post-training process: Supervised Fine-Tuning (SFT) with Chain-of-Thought rationales followed by Reinforcement Learning (RL) with GRPO on challenging cases. On the SpeechJudge-Eval benchmark, the proposed SpeechJudge-GRM demonstrates superior performance, achieving 77.2% accuracy (and 79.4% after inference-time scaling @10) compared to a classic Bradley-Terry reward model (72.7%). Furthermore, SpeechJudge-GRM can be also employed as a reward function during the post-training of speech generation models to facilitate their alignment with human preferences. 

---
# SparseRM: A Lightweight Preference Modeling with Sparse Autoencoder 

**Authors**: Dengcan Liu, Jiahao Li, Zheren Fu, Yi Tu, Jiajun Li, Zhendong Mao, Yongdong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07896)  

**Abstract**: Reward models (RMs) are a core component in the post-training of large language models (LLMs), serving as proxies for human preference evaluation and guiding model alignment. However, training reliable RMs under limited resources remains challenging due to the reliance on large-scale preference annotations and the high cost of fine-tuning LLMs. To address this, we propose SparseRM, which leverages Sparse Autoencoder (SAE) to extract preference-relevant information encoded in model representations, enabling the construction of a lightweight and interpretable reward model. SparseRM first employs SAE to decompose LLM representations into interpretable directions that capture preference-relevant features. The representations are then projected onto these directions to compute alignment scores, which quantify the strength of each preference feature in the representations. A simple reward head aggregates these scores to predict preference scores. Experiments on three preference modeling tasks show that SparseRM achieves superior performance over most mainstream RMs while using less than 1% of trainable parameters. Moreover, it integrates seamlessly into downstream alignment pipelines, highlighting its potential for efficient alignment. 

---
# Intelligence per Watt: Measuring Intelligence Efficiency of Local AI 

**Authors**: Jon Saad-Falcon, Avanika Narayan, Hakki Orhun Akengin, J. Wes Griffin, Herumb Shandilya, Adrian Gamarra Lafuente, Medhya Goel, Rebecca Joseph, Shlok Natarajan, Etash Kumar Guha, Shang Zhu, Ben Athiwaratkun, John Hennessy, Azalia Mirhoseini, Christopher Ré  

**Link**: [PDF](https://arxiv.org/pdf/2511.07885)  

**Abstract**: Large language model (LLM) queries are predominantly processed by frontier models in centralized cloud infrastructure. Rapidly growing demand strains this paradigm, and cloud providers struggle to scale infrastructure at pace. Two advances enable us to rethink this paradigm: small LMs (<=20B active parameters) now achieve competitive performance to frontier models on many tasks, and local accelerators (e.g., Apple M4 Max) run these models at interactive latencies. This raises the question: can local inference viably redistribute demand from centralized infrastructure? Answering this requires measuring whether local LMs can accurately answer real-world queries and whether they can do so efficiently enough to be practical on power-constrained devices (i.e., laptops). We propose intelligence per watt (IPW), task accuracy divided by unit of power, as a metric for assessing capability and efficiency of local inference across model-accelerator pairs. We conduct a large-scale empirical study across 20+ state-of-the-art local LMs, 8 accelerators, and a representative subset of LLM traffic: 1M real-world single-turn chat and reasoning queries. For each query, we measure accuracy, energy, latency, and power. Our analysis reveals $3$ findings. First, local LMs can accurately answer 88.7% of single-turn chat and reasoning queries with accuracy varying by domain. Second, from 2023-2025, IPW improved 5.3x and local query coverage rose from 23.2% to 71.3%. Third, local accelerators achieve at least 1.4x lower IPW than cloud accelerators running identical models, revealing significant headroom for optimization. These findings demonstrate that local inference can meaningfully redistribute demand from centralized infrastructure, with IPW serving as the critical metric for tracking this transition. We release our IPW profiling harness for systematic intelligence-per-watt benchmarking. 

---
# LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive Generation 

**Authors**: Xingyu Li, Xiaolei Liu, Cheng Liu, Yixiao Xu, Kangyi Ding, Bangzhou Xin, Jia-Li Yin  

**Link**: [PDF](https://arxiv.org/pdf/2511.07876)  

**Abstract**: As large language models (LLMs) scale, their inference incurs substantial computational resources, exposing them to energy-latency attacks, where crafted prompts induce high energy and latency cost. Existing attack methods aim to prolong output by delaying the generation of termination symbols. However, as the output grows longer, controlling the termination symbols through input becomes difficult, making these methods less effective. Therefore, we propose LoopLLM, an energy-latency attack framework based on the observation that repetitive generation can trigger low-entropy decoding loops, reliably compelling LLMs to generate until their output limits. LoopLLM introduces (1) a repetition-inducing prompt optimization that exploits autoregressive vulnerabilities to induce repetitive generation, and (2) a token-aligned ensemble optimization that aggregates gradients to improve cross-model transferability. Extensive experiments on 12 open-source and 2 commercial LLMs show that LoopLLM significantly outperforms existing methods, achieving over 90% of the maximum output length, compared to 20% for baselines, and improving transferability by around 40% to DeepSeek-V3 and Gemini 2.5 Flash. 

---
# LLM-Powered Fully Automated Chaos Engineering: Towards Enabling Anyone to Build Resilient Software Systems at Low Cost 

**Authors**: Daisuke Kikuta, Hiroki Ikeuchi, Kengo Tajiri  

**Link**: [PDF](https://arxiv.org/pdf/2511.07865)  

**Abstract**: Chaos Engineering (CE) is an engineering technique aimed at improving the resilience of distributed systems. It involves intentionally injecting faults into a system to test its resilience, uncover weaknesses, and address them before they cause failures in production. Recent CE tools automate the execution of predefined CE experiments. However, planning such experiments and improving the system based on the experimental results still remain manual. These processes are labor-intensive and require multi-domain expertise. To address these challenges and enable anyone to build resilient systems at low cost, this paper proposes ChaosEater, a system that automates the entire CE cycle with Large Language Models (LLMs). It predefines an agentic workflow according to a systematic CE cycle and assigns subdivided processes within the workflow to LLMs. ChaosEater targets CE for software systems built on Kubernetes. Therefore, the LLMs in ChaosEater complete CE cycles through software engineering tasks, including requirement definition, code generation, testing, and debugging. We evaluate ChaosEater through case studies on small- and large-scale Kubernetes systems. The results demonstrate that it consistently completes reasonable CE cycles with significantly low time and monetary costs. Its cycles are also qualitatively validated by human engineers and LLMs. 

---
# CC30k: A Citation Contexts Dataset for Reproducibility-Oriented Sentiment Analysis 

**Authors**: Rochana R. Obadage, Sarah M. Rajtmajer, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07790)  

**Abstract**: Sentiments about the reproducibility of cited papers in downstream literature offer community perspectives and have shown as a promising signal of the actual reproducibility of published findings. To train effective models to effectively predict reproducibility-oriented sentiments and further systematically study their correlation with reproducibility, we introduce the CC30k dataset, comprising a total of 30,734 citation contexts in machine learning papers. Each citation context is labeled with one of three reproducibility-oriented sentiment labels: Positive, Negative, or Neutral, reflecting the cited paper's perceived reproducibility or replicability. Of these, 25,829 are labeled through crowdsourcing, supplemented with negatives generated through a controlled pipeline to counter the scarcity of negative labels. Unlike traditional sentiment analysis datasets, CC30k focuses on reproducibility-oriented sentiments, addressing a research gap in resources for computational reproducibility studies. The dataset was created through a pipeline that includes robust data cleansing, careful crowd selection, and thorough validation. The resulting dataset achieves a labeling accuracy of 94%. We then demonstrated that the performance of three large language models significantly improves on the reproducibility-oriented sentiment classification after fine-tuning using our dataset. The dataset lays the foundation for large-scale assessments of the reproducibility of machine learning papers. The CC30k dataset and the Jupyter notebooks used to produce and analyze the dataset are publicly available at this https URL . 

---
# SALT: Steering Activations towards Leakage-free Thinking in Chain of Thought 

**Authors**: Shourya Batra, Pierce Tillman, Samarth Gaggar, Shashank Kesineni, Kevin Zhu, Sunishchal Dev, Ashwinee Panda, Vasu Sharma, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2511.07772)  

**Abstract**: As Large Language Models (LLMs) evolve into personal assistants with access to sensitive user data, they face a critical privacy challenge: while prior work has addressed output-level privacy, recent findings reveal that LLMs often leak private information through their internal reasoning processes, violating contextual privacy expectations. These leaky thoughts occur when models inadvertently expose sensitive details in their reasoning traces, even when final outputs appear safe. The challenge lies in preventing such leakage without compromising the model's reasoning capabilities, requiring a delicate balance between privacy and utility. We introduce Steering Activations towards Leakage-free Thinking (SALT), a lightweight test-time intervention that mitigates privacy leakage in model's Chain of Thought (CoT) by injecting targeted steering vectors into hidden state. We identify the high-leakage layers responsible for this behavior. Through experiments across multiple LLMs, we demonstrate that SALT achieves reductions including $18.2\%$ reduction in CPL on QwQ-32B, $17.9\%$ reduction in CPL on Llama-3.1-8B, and $31.2\%$ reduction in CPL on Deepseek in contextual privacy leakage dataset AirGapAgent-R while maintaining comparable task performance and utility. Our work establishes SALT as a practical approach for test-time privacy protection in reasoning-capable language models, offering a path toward safer deployment of LLM-based personal agents. 

---
# ViPRA: Video Prediction for Robot Actions 

**Authors**: Sandeep Routray, Hengkai Pan, Unnat Jain, Shikhar Bahl, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2511.07732)  

**Abstract**: Can we turn a video prediction model into a robot policy? Videos, including those of humans or teleoperated robots, capture rich physical interactions. However, most of them lack labeled actions, which limits their use in robot learning. We present Video Prediction for Robot Actions (ViPRA), a simple pretraining-finetuning framework that learns continuous robot control from these actionless videos. Instead of directly predicting actions, we train a video-language model to predict both future visual observations and motion-centric latent actions, which serve as intermediate representations of scene dynamics. We train these latent actions using perceptual losses and optical flow consistency to ensure they reflect physically grounded behavior. For downstream control, we introduce a chunked flow matching decoder that maps latent actions to robot-specific continuous action sequences, using only 100 to 200 teleoperated demonstrations. This approach avoids expensive action annotation, supports generalization across embodiments, and enables smooth, high-frequency continuous control upto 22 Hz via chunked action decoding. Unlike prior latent action works that treat pretraining as autoregressive policy learning, explicitly models both what changes and how. Our method outperforms strong baselines, with a 16% gain on the SIMPLER benchmark and a 13% improvement across real world manipulation tasks. We will release models and code at this https URL 

---
# ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents 

**Authors**: Manasi Sharma, Chen Bo Calvin Zhang, Chaithanya Bandi, Clinton Wang, Ankit Aich, Huy Nghiem, Tahseen Rabbani, Ye Htet, Brian Jang, Sumana Basu, Aishwarya Balwani, Denis Peskoff, Marcos Ayestaran, Sean M. Hendryx, Brad Kenstler, Bing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07685)  

**Abstract**: Deep Research (DR) is an emerging agent application that leverages large language models (LLMs) to address open-ended queries. It requires the integration of several capabilities, including multi-step reasoning, cross-document synthesis, and the generation of evidence-backed, long-form answers. Evaluating DR remains challenging because responses are lengthy and diverse, admit many valid solutions, and often depend on dynamic information sources. We introduce ResearchRubrics, a standardized benchmark for DR built with over 2,800+ hours of human labor that pairs realistic, domain-diverse prompts with 2,500+ expert-written, fine-grained rubrics to assess factual grounding, reasoning soundness, and clarity. We also propose a new complexity framework for categorizing DR tasks along three axes: conceptual breadth, logical nesting, and exploration. In addition, we develop human and model-based evaluation protocols that measure rubric adherence for DR agents. We evaluate several state-of-the-art DR systems and find that even leading agents like Gemini's DR and OpenAI's DR achieve under 68% average compliance with our rubrics, primarily due to missed implicit context and inadequate reasoning about retrieved information. Our results highlight the need for robust, scalable assessment of deep research capabilities, to which end we release ResearchRubrics(including all prompts, rubrics, and evaluation code) to facilitate progress toward well-justified research assistants. 

---
# Beyond Fact Retrieval: Episodic Memory for RAG with Generative Semantic Workspaces 

**Authors**: Shreyas Rajesh, Pavan Holur, Chenda Duan, David Chong, Vwani Roychowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2511.07587)  

**Abstract**: Large Language Models (LLMs) face fundamental challenges in long-context reasoning: many documents exceed their finite context windows, while performance on texts that do fit degrades with sequence length, necessitating their augmentation with external memory frameworks. Current solutions, which have evolved from retrieval using semantic embeddings to more sophisticated structured knowledge graphs representations for improved sense-making and associativity, are tailored for fact-based retrieval and fail to build the space-time-anchored narrative representations required for tracking entities through episodic events. To bridge this gap, we propose the \textbf{Generative Semantic Workspace} (GSW), a neuro-inspired generative memory framework that builds structured, interpretable representations of evolving situations, enabling LLMs to reason over evolving roles, actions, and spatiotemporal contexts. Our framework comprises an \textit{Operator}, which maps incoming observations to intermediate semantic structures, and a \textit{Reconciler}, which integrates these into a persistent workspace that enforces temporal, spatial, and logical coherence. On the Episodic Memory Benchmark (EpBench) \cite{huet_episodic_2025} comprising corpora ranging from 100k to 1M tokens in length, GSW outperforms existing RAG based baselines by up to \textbf{20\%}. Furthermore, GSW is highly efficient, reducing query-time context tokens by \textbf{51\%} compared to the next most token-efficient baseline, reducing inference time costs considerably. More broadly, GSW offers a concrete blueprint for endowing LLMs with human-like episodic memory, paving the way for more capable agents that can reason over long horizons. 

---
# LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows 

**Authors**: Raffi Khatchadourian, Rolando Franco  

**Link**: [PDF](https://arxiv.org/pdf/2511.07585)  

**Abstract**: Financial institutions deploy Large Language Models (LLMs) for reconciliations, regulatory reporting, and client communications, but nondeterministic outputs (output drift) undermine auditability and trust. We quantify drift across five model architectures (7B-120B parameters) on regulated financial tasks, revealing a stark inverse relationship: smaller models (Granite-3-8B, Qwen2.5-7B) achieve 100% output consistency at T=0.0, while GPT-OSS-120B exhibits only 12.5% consistency (95% CI: 3.5-36.0%) regardless of configuration (p<0.0001, Fisher's exact test). This finding challenges conventional assumptions that larger models are universally superior for production deployment.
Our contributions include: (i) a finance-calibrated deterministic test harness combining greedy decoding (T=0.0), fixed seeds, and SEC 10-K structure-aware retrieval ordering; (ii) task-specific invariant checking for RAG, JSON, and SQL outputs using finance-calibrated materiality thresholds (plus or minus 5%) and SEC citation validation; (iii) a three-tier model classification system enabling risk-appropriate deployment decisions; and (iv) an audit-ready attestation system with dual-provider validation.
We evaluated five models (Qwen2.5-7B via Ollama, Granite-3-8B via IBM this http URL, Llama-3.3-70B, Mistral-Medium-2505, and GPT-OSS-120B) across three regulated financial tasks. Across 480 runs (n=16 per condition), structured tasks (SQL) remain stable even at T=0.2, while RAG tasks show drift (25-75%), revealing task-dependent sensitivity. Cross-provider validation confirms deterministic behavior transfers between local and cloud deployments. We map our framework to Financial Stability Board (FSB), Bank for International Settlements (BIS), and Commodity Futures Trading Commission (CFTC) requirements, demonstrating practical pathways for compliance-ready AI deployments. 

---
# Think Before You Retrieve: Learning Test-Time Adaptive Search with Small Language Models 

**Authors**: Supriti Vijay, Aman Priyanshu, Anu Vellore, Baturay Saglam, Amin Karbasi  

**Link**: [PDF](https://arxiv.org/pdf/2511.07581)  

**Abstract**: Effective information retrieval requires reasoning over partial evidence and refining strategies as information emerges. Yet current approaches fall short: neural retrievers lack reasoning capabilities, large language models (LLMs) provide semantic depth but at prohibitive cost, and query rewriting or decomposition limits improvement to static transformations. As a result, existing methods fail to capture the iterative dynamics of exploration, feedback, and revision that complex user queries demand. We introduce Orion, a training framework that enables compact models (350M-1.2B parameters) to perform iterative retrieval through learned search strategies. Orion combines: (1) synthetic trajectory generation and supervised fine-tuning to encourage diverse exploration patterns in models, (2) reinforcement learning (RL) that rewards effective query refinement and backtracking behaviors, and (3) inference-time beam search algorithms that exploit the self-reflection capabilities learned during RL. Despite using only 3% of the training data available, our 1.2B model achieves 77.6% success on SciFact (vs. 72.6% for prior retrievers), 25.2% on BRIGHT (vs. 22.1%), 63.2% on NFCorpus (vs. 57.8%), and remains competitive on FEVER, HotpotQA, and MSMarco. It outperforms retrievers up to 200-400x larger on five of six benchmarks. These findings suggest that retrieval performance can emerge from learned strategies, not just model scale, when models are trained to search, reflect, and revise. 

---
# A Decentralized Retrieval Augmented Generation System with Source Reliabilities Secured on Blockchain 

**Authors**: Yining Lu, Wenyi Tang, Max Johnson, Taeho Jung, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07577)  

**Abstract**: Existing retrieval-augmented generation (RAG) systems typically use a centralized architecture, causing a high cost of data collection, integration, and management, as well as privacy concerns. There is a great need for a decentralized RAG system that enables foundation models to utilize information directly from data owners who maintain full control over their sources. However, decentralization brings a challenge: the numerous independent data sources vary significantly in reliability, which can diminish retrieval accuracy and response quality. To address this, our decentralized RAG system has a novel reliability scoring mechanism that dynamically evaluates each source based on the quality of responses it contributes to generate and prioritizes high-quality sources during retrieval. To ensure transparency and trust, the scoring process is securely managed through blockchain-based smart contracts, creating verifiable and tamper-proof reliability records without relying on a central authority. We evaluate our decentralized system with two Llama models (3B and 8B) in two simulated environments where six data sources have different levels of reliability. Our system achieves a +10.7\% performance improvement over its centralized counterpart in the real world-like unreliable data environments. Notably, it approaches the upper-bound performance of centralized systems under ideally reliable data environments. The decentralized infrastructure enables secure and trustworthy scoring management, achieving approximately 56\% marginal cost savings through batched update operations. Our code and system are open-sourced at this http URL. 

---
# Quantifying the Impact of CU: A Systematic Literature Review 

**Authors**: Thomas Compton  

**Link**: [PDF](https://arxiv.org/pdf/2511.07491)  

**Abstract**: Community Unionism has served as a pivotal concept in debates on trade union renewal since the early 2000s, yet its theoretical coherence and political significance remain unresolved. This article investigates why CU has gained such prominence -- not by testing its efficacy, but by mapping how it is constructed, cited, and contested across the scholarly literature. Using two complementary systematic approaches -- a citation network analysis of 114 documents and a thematic review of 18 core CU case studies -- I examine how CU functions as both an empirical descriptor and a normative ideal. The analysis reveals CU's dual genealogy: positioned by British scholars as an indigenous return to historic rank-and-file practices, yet structurally aligned with transnational social movement unionism. Thematic coding shows near-universal emphasis on coalition-building and alliances, but deep ambivalence toward class politics. This tension suggests CU's significance lies less in operationalising a new union model, and more in managing contradictions -- between workplace and community, leadership and rank-and-file, reform and radicalism -- within a shrinking labour movement. 

---
# Alignment-Constrained Dynamic Pruning for LLMs: Identifying and Preserving Alignment-Critical Circuits 

**Authors**: Dev Patel, Gabrielle Gervacio, Diekola Raimi, Kevin Zhu, Ryan Lagasse, Gabriel Grand, Ashwinee Panda, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2511.07482)  

**Abstract**: Large Language Models require substantial computational resources for inference, posing deployment challenges. While dynamic pruning offers superior efficiency over static methods through adaptive circuit selection, it exacerbates alignment degradation by retaining only input-dependent safety-critical circuit preservation across diverse inputs. As a result, addressing these heightened alignment vulnerabilities remains critical. We introduce Alignment-Aware Probe Pruning (AAPP), a dynamic structured pruning method that adaptively preserves alignment-relevant circuits during inference, building upon Probe Pruning. Experiments on LLaMA 2-7B, Qwen2.5-14B-Instruct, and Gemma-3-12B-IT show AAPP improves refusal rates by 50\% at matched compute, enabling efficient yet safety-preserving LLM deployment. 

---
# The Polite Liar: Epistemic Pathology in Language Models 

**Authors**: Bentley DeVilling  

**Link**: [PDF](https://arxiv.org/pdf/2511.07477)  

**Abstract**: Large language models exhibit a peculiar epistemic pathology: they speak as if they know, even when they do not. This paper argues that such confident fabrication, what I call the polite liar, is a structural consequence of reinforcement learning from human feedback (RLHF). Building on Frankfurt's analysis of bullshit as communicative indifference to truth, I show that this pathology is not deception but structural indifference: a reward architecture that optimizes for perceived sincerity over evidential accuracy. Current alignment methods reward models for being helpful, harmless, and polite, but not for being epistemically grounded. As a result, systems learn to maximize user satisfaction rather than truth, performing conversational fluency as a virtue. I analyze this behavior through the lenses of epistemic virtue theory, speech-act philosophy, and cognitive alignment, showing that RLHF produces agents trained to mimic epistemic confidence without access to epistemic justification. The polite liar thus reveals a deeper alignment tension between linguistic cooperation and epistemic integrity. The paper concludes with an "epistemic alignment" principle: reward justified confidence over perceived fluency. 

---
# Network and Systems Performance Characterization of MCP-Enabled LLM Agents 

**Authors**: Zihao Ding, Mufeng Zhu, Yao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07426)  

**Abstract**: Model Context Protocol (MCP) has recently gained increased attention within the AI community for providing a standardized way for large language models (LLMs) to interact with external tools and services, significantly enhancing their capabilities. However, the inclusion of extensive contextual information, including system prompts, MCP tool definitions, and context histories, in MCP-enabled LLM interactions, dramatically inflates token usage. Given that LLM providers charge based on tokens, these expanded contexts can quickly escalate monetary costs and increase the computational load on LLM services. This paper presents a comprehensive measurement-based analysis of MCP-enabled interactions with LLMs, revealing trade-offs between capability, performance, and cost. We explore how different LLM models and MCP configurations impact key performance metrics such as token efficiency, monetary cost, task completion times, and task success rates, and suggest potential optimizations, including enabling parallel tool calls and implementing robust task abort mechanisms. These findings provide useful insights for developing more efficient, robust, and cost-effective MCP-enabled workflows. 

---
# LLaDA-Rec: Discrete Diffusion for Parallel Semantic ID Generation in Generative Recommendation 

**Authors**: Teng Shi, Chenglei Shen, Weijie Yu, Shen Nie, Chongxuan Li, Xiao Zhang, Ming He, Yan Han, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.06254)  

**Abstract**: Generative recommendation represents each item as a semantic ID, i.e., a sequence of discrete tokens, and generates the next item through autoregressive decoding. While effective, existing autoregressive models face two intrinsic limitations: (1) unidirectional constraints, where causal attention restricts each token to attend only to its predecessors, hindering global semantic modeling; and (2) error accumulation, where the fixed left-to-right generation order causes prediction errors in early tokens to propagate to the predictions of subsequent token. To address these issues, we propose LLaDA-Rec, a discrete diffusion framework that reformulates recommendation as parallel semantic ID generation. By combining bidirectional attention with the adaptive generation order, the approach models inter-item and intra-item dependencies more effectively and alleviates error accumulation. Specifically, our approach comprises three key designs: (1) a parallel tokenization scheme that produces semantic IDs for bidirectional modeling, addressing the mismatch between residual quantization and bidirectional architectures; (2) two masking mechanisms at the user-history and next-item levels to capture both inter-item sequential dependencies and intra-item semantic relationships; and (3) an adapted beam search strategy for adaptive-order discrete diffusion decoding, resolving the incompatibility of standard beam search with diffusion-based generation. Experiments on three real-world datasets show that LLaDA-Rec consistently outperforms both ID-based and state-of-the-art generative recommenders, establishing discrete diffusion as a new paradigm for generative recommendation. 

---
