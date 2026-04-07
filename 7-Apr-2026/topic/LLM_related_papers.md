# Formalized Information Needs Improve Large-Language-Model Relevance Judgments 

**Authors**: Jüri Keller, Maik Fröbe, Björn Engelmann, Fabian Haak, Timo Breuer, Birger Larsen, Philipp Schaer  

**Link**: [PDF](https://arxiv.org/pdf/2604.04140)  

**Abstract**: Cranfield-style retrieval evaluations with too few or too many relevant documents or with low inter-assessor agreement on relevance can reduce the reliability of observations. In evaluations with human assessors, information needs are often formalized as retrieval topics to avoid an excessive number of relevant documents while maintaining good agreement. However, emerging evaluation setups that use Large Language Models (LLMs) as relevance assessors often use only queries, potentially decreasing the reliability. To study whether LLM relevance assessors benefit from formalized information needs, we synthetically formalize information needs with LLMs into topics that follow the established structure from previous human relevance assessments (i.e., descriptions and narratives). We compare assessors using synthetically formalized topics against the LLM-default query-only assessor on Robust04 and the 2019/2020 editions of TREC Deep Learning. We find that assessors without formalization judge many more documents relevant and have a lower agreement, leading to reduced reliability in retrieval evaluations. Furthermore, we show that the formalized topics improve agreement between human and LLM relevance judgments, even when the topics are not highly similar to their human counterparts. Our findings indicate that LLM relevance assessors should use formalized information needs, as is standard for human assessment, and synthetically formalize topics when no human formalization exists to improve evaluation reliability. 

---
# Retrieval Augmented Conversational Recommendation with Reinforcement Learning 

**Authors**: Zhenrui Yue, Honglei Zhuang, Zhen Qin, Zhankui He, Huimin Zeng, Julian McAuley, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04457)  

**Abstract**: Large language models (LLMs) exhibit enhanced capabilities in language understanding and generation. By utilizing their embedded knowledge, LLMs are increasingly used as conversational recommender systems (CRS), achieving improved performance across diverse scenarios. However, existing LLM-based methods rely on pretrained knowledge without external retrieval mechanisms for novel items. Additionally, the lack of a unified corpus poses challenges for integrating retrieval augmentation into CRS. Motivated by these challenges, we present RAR, a novel two-stage retrieval augmented conversational recommendation framework that aligns retrieval and generation to enhance both performance and factuality. To support this framework and provide a unified corpus, we construct a large-scale movie corpus, comprising over 300k movies with rich metadata, such as titles, casts and plot summaries. Leveraging this data, our primary contribution is RAR, the first framework to departs from standard two-stage CRS by dynamically bridging retrieval and generation. First, a retriever model generates candidate items based on user history; in the subsequent stage, an LLM refines the recommendations by incorporating conversational context with retrieved results. In addition, we introduce a novel reinforcement learning (RL) method that leverages LLM feedback to iteratively update the retriever. By creating a collaborative feedback loop that reinforces sampled candidate sets with higher ranking metrics, RAR effectively mitigates the misalignment between the retrieval and generation stages. Furthermore, grounding the LLM in factual metadata allows our RL-driven approach to capture subtle user intentions and generate context-aware recommendations with reduced hallucinations. We validate our approach through extensive experiments on multiple benchmarks, where RAR consistently outperforms state-of-the-art baseline methods. 

---
# MisEdu-RAG: A Misconception-Aware Dual-Hypergraph RAG for Novice Math Teachers 

**Authors**: Zhihan Guo, Rundong Xue, Yuting Lu, Jionghao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.04036)  

**Abstract**: Novice math teachers often encounter students' mistakes that are difficult to diagnose and remediate. Misconceptions are especially challenging because teachers must explain what went wrong and how to solve them. Although many existing large language model (LLM) platforms can assist in generating instructional feedback, these LLMs loosely connect pedagogical knowledge and student mistakes, which might make the guidance less actionable for teachers. To address this gap, we propose MisEdu-RAG, a dual-hypergraph-based retrieval-augmented generation (RAG) framework that organizes pedagogical knowledge as a concept hypergraph and real student mistake cases as an instance hypergraph. Given a query, MisEdu-RAG performs a two-stage retrieval to gather connected evidence from both layers and generates a response grounded in the retrieved cases and pedagogical principles. We evaluate on \textit{MisstepMath}, a dataset of math mistakes paired with teacher solutions, as a benchmark for misconception-aware retrieval and response generation across topics and error types. Evaluation results on \textit{MisstepMath} show that, compared with baseline models, MisEdu-RAG improves token-F1 by 10.95\% and yields up to 15.3\% higher five-dimension response quality, with the largest gains on \textit{Diversity} and \textit{Empowerment}. To verify its applicability in practical use, we further conduct a pilot study through a questionnaire survey of 221 teachers and interviews with 6 novices. The findings suggest that MisEdu-RAG provides diagnosis results and concrete teaching moves for high-demand misconception scenarios. Overall, MisEdu-RAG demonstrates strong potential for scalable teacher training and AI-assisted instruction for misconception handling. Our code is available on GitHub: this https URL. 

---
# Ruling Out to Rule In: Contrastive Hypothesis Retrieval for Medical Question Answering 

**Authors**: Byeolhee Kim, Min-Kyung Kim, Young-Hak Kim, Tae-Joon Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2604.04593)  

**Abstract**: Retrieval-augmented generation (RAG) grounds large language models in external medical knowledge, yet standard retrievers frequently surface hard negatives that are semantically close to the query but describe clinically distinct conditions. While existing query-expansion methods improve query representation to mitigate ambiguity, they typically focus on enriching target-relevant semantics without an explicit mechanism to selectively suppress specific, clinically plausible hard negatives. This leaves the system prone to retrieving plausible mimics that overshadow the actual diagnosis, particularly when such mimics are dominant within the corpus. We propose Contrastive Hypothesis Retrieval (CHR), a framework inspired by the process of clinical differential diagnosis. CHR generates a target hypothesis $H^+$ for the likely correct answer and a mimic hypothesis $H^-$ for the most plausible incorrect alternative, then scores documents by promoting $H^+$-aligned evidence while penalizing $H^-$-aligned content. Across three medical QA benchmarks and three answer generators, CHR outperforms all five baselines in every configuration, with improvements of up to 10.4 percentage points over the next-best method. On the $n=587$ pooled cases where CHR answers correctly while embedded hypothetical-document query expansion does not, 85.2\% have no shared documents between the top-5 retrieval lists of CHR and of that baseline, consistent with substantive retrieval redirection rather than light re-ranking of the same candidates. By explicitly modeling what to avoid alongside what to find, CHR bridges clinical reasoning with retrieval mechanism design and offers a practical path to reducing hard-negative contamination in medical RAG systems. 

---
# Are LLM-Based Retrievers Worth Their Cost? An Empirical Study of Efficiency, Robustness, and Reasoning Overhead 

**Authors**: Abdelrahman Abdallah, Jamie Holdcroft, Mohammed Ali, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2604.03676)  

**Abstract**: Large language model retrievers improve performance on complex queries, but their practical value depends on efficiency, robustness, and reliable confidence signals in addition to accuracy. We reproduce a reasoning-intensive retrieval benchmark (BRIGHT) across 12 tasks and 14 retrievers, and extend evaluation with cold-start indexing cost, query latency distributions and throughput, corpus scaling, robustness to controlled query perturbations, and confidence use (AUROC) for predicting query success. We also quantify \emph{reasoning overhead} by comparing standard queries to five provided reasoning-augmented variants, measuring accuracy gains relative to added latency. We find that some reasoning-specialized retrievers achieve strong effectiveness while remaining competitive in throughput, whereas several large LLM-based bi-encoders incur substantial latency for modest gains. Reasoning augmentation incurs minimal latency for sub-1B encoders but exhibits diminishing returns for top retrievers and may reduce performance on formal math/code domains. Confidence calibration is consistently weak across model families, indicating that raw retrieval scores are unreliable for downstream routing without additional calibration. We release all code and artifacts for reproducibility. 

---
# Fusion and Alignment Enhancement with Large Language Models for Tail-item Sequential Recommendation 

**Authors**: Zhifu Wei, Yizhou Dang, Guibing Guo, Chuang Zhao, Zhu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.03688)  

**Abstract**: Sequential Recommendation (SR) learns user preferences from their historical interaction sequences and provides personalized suggestions. In real-world scenarios, most items exhibit sparse interactions, known as the tail-item problem. This issue limits the model's ability to accurately capture item transition patterns. To tackle this, large language models (LLMs) offer a promising solution by capturing semantic relationships between items. Despite previous efforts to leverage LLM-derived embeddings for enriching tail items, they still face the following limitations: 1) They struggle to effectively fuse collaborative signals with semantic knowledge, leading to suboptimal item embedding quality. 2) Existing methods overlook the structural inconsistency between the ID and LLM embedding spaces, causing conflicting signals that degrade recommendation accuracy. In this work, we propose a Fusion and Alignment Enhancement framework with LLMs for Tail-item Sequential Recommendation (FAERec), which improves item representations by generating coherently-fused and structurally consistent embeddings. For the information fusion challenge, we design an adaptive gating mechanism that dynamically fuses ID and LLM embeddings. Then, we propose a dual-level alignment approach to mitigate structural inconsistency. The item-level alignment establishes correspondences between ID and LLM embeddings of the same item through contrastive learning, while the feature-level alignment constrains the correlation patterns between corresponding dimensions across the two embedding spaces. Furthermore, the weights of the two alignments are adjusted by a curriculum learning scheduler to avoid premature optimization of the complex feature-level objective. Extensive experiments across three widely used datasets with multiple representative SR backbones demonstrate the effectiveness and generalizability of our framework. 

---
# Rank, Don't Generate: Statement-level Ranking for Explainable Recommendation 

**Authors**: Ben Kabongo, Arthur Satouf, Vincent Guigue  

**Link**: [PDF](https://arxiv.org/pdf/2604.03724)  

**Abstract**: Textual explanations, generated with large language models (LLMs), are increasingly used to justify recommendations. Yet, evaluating these explanations remains a critical challenge. We advocate a shift in objective: rank, don't generate. We formalize explainable recommendation as a statement-level ranking problem, where systems rank candidate explanatory statements derived from reviews and return the top-k as explanation. This formulation mitigates hallucination by construction and enables fine-grained factual analysis. It also models factor importance through relevance scores and supports standardized, reproducible evaluation with established ranking metrics. Meaningful assessment, however, requires each statement to be explanatory (item facts affecting user experience), atomic (one opinion about one aspect), and unique (paraphrases consolidated), which is challenging to obtain from noisy reviews. We address this with (i) an LLM-based extraction pipeline producing explanatory and atomic statements, and (ii) a scalable, semantic clustering method consolidating paraphrases to enforce uniqueness. Building on this pipeline, we introduce StaR, a benchmark for statement ranking in explainable recommendation, constructed from four Amazon Reviews 2014 product categories. We evaluate popularity-based baselines and state-of-the-art models under global-level (all statements) and item-level (target item statements) ranking. Popularity baselines are competitive in global-level ranking but outperform state-of-the-art models on average in item-level ranking, exposing critical limitations in personalized explanation ranking. 

---
# MMP-Refer: Multimodal Path Retrieval-augmented LLMs For Explainable Recommendation 

**Authors**: Xiangchen Pan, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.03666)  

**Abstract**: Explainable recommendations help improve the transparency and credibility of recommendation systems, and play an important role in personalized recommendation scenarios. At present, methods for explainable recommendation based on large language models(LLMs) often consider introducing collaborative information to enhance the personalization and accuracy of the model, but ignore the multimodal information in the recommendation dataset; In addition, collaborative information needs to be aligned with the semantic space of LLM. Introducing collaborative signals through retrieval paths is a good choice, but most of the existing retrieval path collection schemes use the existing Explainable GNN algorithms. Although these methods are effective, they are relatively unexplainable and not be suitable for the recommendation field.
To address the above challenges, we propose MMP-Refer, a framework using \textbf{M}ulti\textbf{M}odal Retrieval \textbf{P}aths with \textbf{Re}trieval-augmented LLM \textbf{F}or \textbf{E}xplainable \textbf{R}ecommendation. We use a sequential recommendation model based on joint residual coding to obtain multimodal embeddings, and design a heuristic search algorithm to obtain retrieval paths by multimodal embeddings; In the generation phase, we integrated a trainable lightweight collaborative adapter to map the graph encoding of interaction subgraphs to the semantic space of the LLM, as soft prompts to enhance the understanding of interaction information by the LLM. Extensive experiments have demonstrated the effectiveness of our approach. Codes and data are available at this https URL. 

---
# User Simulator-Guided Multi-Turn Preference Optimization for Reasoning LLM-based Conversational Recommendation 

**Authors**: Xingyuan Xiang, Xiangchen Pan, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.03671)  

**Abstract**: Conversational Recommender Systems (CRSs) leverage natural language interactions for personalized recommendation, yet information-scarce dialogue histories and single-turn recommendation paradigms may severely hinder accurate modeling of complex user preferences. To alleviate this issue, recent studies have introduced LLM-based user simulators, which generate natural language feedback and perform simulated multi-turn interactions to assist recommendation. Nevertheless, since simulators cannot access true user preference labels during inference, their feedback may deviate from actual user interests, causing errors to accumulate over multiple interactions and severely affecting the generalization of the recommender. Inspired by the multi-step reasoning capabilities of LLMs and the effectiveness of reinforcement learning in policy optimization, we propose SMTPO, a user simulator-guided multi-turn preference optimization conversational recommendation framework. To align simulator-generated feedback with true user preferences in the absence of explicit labels, we enhance feedback quality via multi-task supervised fine-tuning (SFT), enabling the simulator to better reflect users' complex and diverse needs. To address the challenge of biased feedback destabilizing multi-turn optimization, we first allow the reasoning LLM-based recommender to learn preference reasoning and recommendation patterns through SFT and then employ reinforcement learning with fine-grained reward design to progressively align with true user preferences, improving recommendation performance. Extensive experiments on public datasets demonstrate the effectiveness and transferability of our method. 

---
# LLM-based Listwise Reranking under the Effect of Positional Bias 

**Authors**: Jingfen Qiao, Jin Huang, Xinyu Ma, Shuaiqiang Wang, Dawei Yin, Evangelos Kanoulas, Andrew Yates  

**Link**: [PDF](https://arxiv.org/pdf/2604.03642)  

**Abstract**: LLM-based listwise passage reranking has attracted attention for its effectiveness in ranking candidate passages. However, these models suffer from positional bias, where passages positioned towards the end of the input are less likely to be moved to top positions in the ranking. We hypothesize that there are two primary sources of positional bias: (1) architectural bias inherent in LLMs and (2) the imbalanced positioning of relevant documents. To address this, we propose DebiasFirst, a method that integrates positional calibration and position-aware data augmentation during fine-tuning. Positional calibration uses inverse propensity scoring to adjust for positional bias by re-weighting the contributions of different positions in the loss function when training. Position-aware augmentation augments training data to ensure that each passage appears equally across varied positions in the input list. This approach markedly enhances both effectiveness and robustness to the original ranking across diverse first-stage retrievers, reducing the dependence of NDCG@10 performance on the position of relevant documents. DebiasFirst also complements the inference-stage debiasing methods, offering a practical solution for mitigating positional bias in reranking. 

---
# BridgeRAG: Training-Free Bridge-Conditioned Retrieval for Multi-Hop Question Answering 

**Authors**: Andre Bacellar  

**Link**: [PDF](https://arxiv.org/pdf/2604.03384)  

**Abstract**: Multi-hop retrieval is not a single-step relevance problem: later-hop evidence should be ranked by its utility conditioned on retrieved bridge evidence, not by similarity to the original query alone. We present BridgeRAG, a training-free, graph-free retrieval method for retrieval-augmented generation (RAG) over multi-hop questions that operationalizes this view with a tripartite scorer s(q,b,c) over (question, bridge, candidate). BridgeRAG separates coverage from scoring: dual-entity ANN expansion broadens the second-hop candidate pool, while a bridge-conditioned LLM judge identifies the active reasoning chain among competing candidates without any offline graph or proposition index. Across four controlled experiments we show that this conditioning signal is (i) selective: +2.55pp on parallel-chain queries (p<0.001) vs. ~0 on single-chain subtypes; (ii) irreplaceable: substituting the retrieved passage with generated SVO query text reduces R@5 by 2.1pp, performing worse than even the lowest-SVO-similarity pool passage; (iii) predictable: cos(b,g2) correlates with per-query gain (Spearman rho=0.104, p<0.001); and (iv) mechanistically precise: bridge conditioning causes productive re-rankings (18.7% flip-win rate on parallel-chain vs. 0.6% on single-chain), not merely more churn. Combined with lightweight coverage expansion and percentile-rank score fusion, BridgeRAG achieves the best published training-free R@5 under matched benchmark evaluation on all three standard MHQA benchmarks without a graph database or any training: 0.8146 on MuSiQue (+3.1pp vs. PropRAG, +6.8pp vs. HippoRAG2), 0.9527 on 2WikiMultiHopQA (+1.2pp vs. PropRAG), and 0.9875 on HotpotQA (+1.35pp vs. PropRAG). 

---
# Align then Train: Efficient Retrieval Adapter Learning 

**Authors**: Seiji Maekawa, Moin Aminnaseri, Pouya Pezeshkpour, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2604.03403)  

**Abstract**: Dense retrieval systems increasingly need to handle complex queries. In many realistic settings, users express intent through long instructions or task-specific descriptions, while target documents remain relatively simple and static. This asymmetry creates a retrieval mismatch: understanding queries may require strong reasoning and instruction-following, whereas efficient document indexing favors lightweight encoders. Existing retrieval systems often address this mismatch by directly improving the embedding model, but fine-tuning large embedding models to better follow such instructions is computationally expensive, memory-intensive, and operationally burdensome. To address this challenge, we propose Efficient Retrieval Adapter (ERA), a label-efficient framework that trains retrieval adapters in two stages: self-supervised alignment and supervised adaptation. Inspired by the pre-training and supervised fine-tuning stages of LLMs, ERA first aligns the embedding spaces of a large query embedder and a lightweight document embedder, and then uses limited labeled data to adapt the query-side representation, bridging both the representation gap between embedding models and the semantic gap between complex queries and simple documents without re-indexing the corpus. Experiments on the MAIR benchmark, spanning 126 retrieval tasks across 6 domains, show that ERA improves retrieval in low-label settings, outperforms methods that rely on larger amounts of labeled data, and effectively combines stronger query embedders with weaker document embedders across domains. 

---
# SkillX: Automatically Constructing Skill Knowledge Bases for Agents 

**Authors**: Chenxi Wang, Zhuoyun Yu, Xin Xie, Wuguannan Yao, Runnan Fang, Shuofei Qiao, Kexin Cao, Guozhou Zheng, Xiang Qi, Peng Zhang, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.04804)  

**Abstract**: Learning from experience is critical for building capable large language model (LLM) agents, yet prevailing self-evolving paradigms remain inefficient: agents learn in isolation, repeatedly rediscover similar behaviors from limited experience, resulting in redundant exploration and poor generalization. To address this problem, we propose SkillX, a fully automated framework for constructing a \textbf{plug-and-play skill knowledge base} that can be reused across agents and environments. SkillX operates through a fully automated pipeline built on three synergistic innovations: \textit{(i) Multi-Level Skills Design}, which distills raw trajectories into three-tiered hierarchy of strategic plans, functional skills, and atomic skills; \textit{(ii) Iterative Skills Refinement}, which automatically revises skills based on execution feedback to continuously improve library quality; and \textit{(iii) Exploratory Skills Expansion}, which proactively generates and validates novel skills to expand coverage beyond seed training data. Using a strong backbone agent (GLM-4.6), we automatically build a reusable skill library and evaluate its transferability on challenging long-horizon, user-interactive benchmarks, including AppWorld, BFCL-v3, and $\tau^2$-Bench. Experiments show that SkillKB consistently improves task success and execution efficiency when plugged into weaker base agents, highlighting the importance of structured, hierarchical experience representations for generalizable agent learning. Our code will be publicly available soon at this https URL. 

---
# LightThinker++: From Reasoning Compression to Memory Management 

**Authors**: Yuqi Zhu, Jintian Zhang, Zhenjie Wan, Yujie Luo, Shuofei Qiao, Zhengke Gui, Da Zheng, Lei Liang, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03679)  

**Abstract**: Large language models (LLMs) excel at complex reasoning, yet their efficiency is limited by the surging cognitive overhead of long thought traces. In this paper, we propose LightThinker, a method that enables LLMs to dynamically compress intermediate thoughts into compact semantic representations. However, static compression often struggles with complex reasoning where the irreversible loss of intermediate details can lead to logical bottlenecks. To address this, we evolve the framework into LightThinker++, introducing Explicit Adaptive Memory Management. This paradigm shifts to behavioral-level management by incorporating explicit memory primitives, supported by a specialized trajectory synthesis pipeline to train purposeful memory scheduling. Extensive experiments demonstrate the framework's versatility across three dimensions. (1) LightThinker reduces peak token usage by 70% and inference time by 26% with minimal accuracy loss. (2) In standard reasoning, LightThinker++ slashes peak token usage by 69.9% while yielding a +2.42% accuracy gain under the same context budget for maximum performance. (3) Most notably, in long-horizon agentic tasks, it maintains a stable footprint beyond 80 rounds (a 60%-70% reduction), achieving an average performance gain of 14.8% across different complex scenarios. Overall, our work provides a scalable direction for sustaining deep LLM reasoning over extended horizons with minimal overhead. 

---
# Lightweight Query Routing for Adaptive RAG: A Baseline Study on RAGRouter-Bench 

**Authors**: Prakhar Bansal, Shivangi Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2604.03455)  

**Abstract**: Retrieval-Augmented Generation pipelines span a wide range of retrieval strategies that differ substantially in token cost and capability. Selecting the right strategy per query is a practical efficiency problem, yet no routing classifiers have been trained on RAGRouter-Bench \citep{wang2026ragrouterbench}, a recently released benchmark of $7,727$ queries spanning four knowledge domains, each annotated with one of three canonical query types: factual, reasoning, and summarization. We present the first systematic evaluation of lightweight classifier-based routing on this benchmark. Five classical classifiers are evaluated under three feature regimes, namely, TF-IDF, MiniLM sentence embeddings \citep{reimers2019sbert}, and hand-crafted structural features, yielding 15 classifier feature combinations. Our best configuration, TF-IDF with an SVM, achieves a macro-averaged F1 of $\mathbf{0.928}$ and an accuracy of $\mathbf{93.2\%}$, while simulating $\mathbf{28.1\%}$ token savings relative to always using the most expensive paradigm. Lexical TF-IDF features outperform semantic sentence embeddings by $3.1$ macro-F1 points, suggesting that surface keyword patterns are strong predictors of query-type complexity. Domain-level analysis reveals that medical queries are hardest to route and legal queries most tractable. These results establish a reproducible query-side baseline and highlight the gap that corpus-aware routing must close. 

---
# Do No Harm: Exposing Hidden Vulnerabilities of LLMs via Persona-based Client Simulation Attack in Psychological Counseling 

**Authors**: Qingyang Xu, Yaling Shen, Stephanie Fong, Zimu Wang, Yiwen Jiang, Xiangyu Zhao, Jiahe Liu, Zhongxing Xu, Vincent Lee, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2604.04842)  

**Abstract**: The increasing use of large language models (LLMs) in mental healthcare raises safety concerns in high-stakes therapeutic interactions. A key challenge is distinguishing therapeutic empathy from maladaptive validation, where supportive responses may inadvertently reinforce harmful beliefs or behaviors in multi-turn conversations. This risk is largely overlooked by existing red-teaming frameworks, which focus mainly on generic harms or optimization-based attacks. To address this gap, we introduce Personality-based Client Simulation Attack (PCSA), the first red-teaming framework that simulates clients in psychological counseling through coherent, persona-driven client dialogues to expose vulnerabilities in psychological safety alignment. Experiments on seven general and mental health-specialized LLMs show that PCSA substantially outperforms four competitive baselines. Perplexity analysis and human inspection further indicate that PCSA generates more natural and realistic dialogues. Our results reveal that current LLMs remain vulnerable to domain-specific adversarial tactics, providing unauthorized medical advice, reinforcing delusions, and implicitly encouraging risky actions. 

---
# LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection 

**Authors**: Cheng Xu, Changhong Jin, Yingjie Niu, Nan Yan, Yuke Mei, Shuhao Guan, Liming Chen, M-Tahar Kechadi  

**Link**: [PDF](https://arxiv.org/pdf/2604.04815)  

**Abstract**: The rapid development of Large Language Models (LLMs) has transformed fake news detection and fact-checking tasks from simple classification to complex reasoning. However, evaluation frameworks have not kept pace. Current benchmarks are static, making them vulnerable to benchmark data contamination (BDC) and ineffective at assessing reasoning under temporal uncertainty. To address this, we introduce LiveFact a continuously updated benchmark that simulates the real-world "fog of war" in misinformation detection. LiveFact uses dynamic, temporal evidence sets to evaluate models on their ability to reason with evolving, incomplete information rather than on memorized knowledge. We propose a dual-mode evaluation: Classification Mode for final verification and Inference Mode for evidence-based reasoning, along with a component to monitor BDC explicitly. Tests with 22 LLMs show that open-source Mixture-of-Experts models, such as Qwen3-235B-A22B, now match or outperform proprietary state-of-the-art systems. More importantly, our analysis finds a significant "reasoning gap." Capable models exhibit epistemic humility by recognizing unverifiable claims in early data slices-an aspect traditional static benchmarks overlook. LiveFact sets a sustainable standard for evaluating robust, temporally aware AI verification. 

---
# Plausibility as Commonsense Reasoning: Humans Succeed, Large Language Models Do not 

**Authors**: Sercan Karakaş  

**Link**: [PDF](https://arxiv.org/pdf/2604.04825)  

**Abstract**: Large language models achieve strong performance on many language tasks, yet it remains unclear whether they integrate world knowledge with syntactic structure in a human-like, structure-sensitive way during ambiguity resolution. We test this question in Turkish prenominal relative-clause attachment ambiguities, where the same surface string permits high attachment (HA) or low attachment (LA). We construct ambiguous items that keep the syntactic configuration fixed and ensure both parses remain pragmatically possible, while graded event plausibility selectively favors High Attachment vs.\ Low Attachment. The contrasts are validated with independent norming ratings. In a speeded forced-choice comprehension experiment, humans show a large, correctly directed plausibility effect. We then evaluate Turkish and multilingual LLMs in a parallel preference-based setup that compares matched HA/LA continuations via mean per-token log-probability. Across models, plausibility-driven shifts are weak, unstable, or reversed. The results suggest that, in the tested models, plausibility information does not guide attachment preferences as reliably as it does in human judgments, and they highlight Turkish RC attachment as a useful cross-linguistic diagnostic beyond broad benchmarks. 

---
# Rethinking Exploration in RLVR: From Entropy Regularization to Refinement via Bidirectional Entropy Modulation 

**Authors**: Hengrui Gu, Xiaotian Han, Yujing Bian, Kaixiong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.04894)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has significantly advanced the reasoning capabilities of large language models (LLMs). However, it faces a fundamental limitation termed \textit{restricted exploration}, where the policy rapidly converges to a narrow set of solutions. While entropy regularization is a popular approach used to sustain exploration, it often proves unreliable for LLMs, suffering from high hyperparameter sensitivity and yielding only marginal performance gains. Motivated by these inefficiencies, we propose to rethink the relationship between policy entropy and exploration. By deriving a parametric formulation of group-relative advantage estimation and analyzing entropy dynamics, we conceptually decompose policy entropy into \textit{informative entropy}, which preserves diverse solution paths, and \textit{spurious entropy}, which erodes reasoning patterns. Our analysis reveals that, in contrast to blind maximization, effective exploration requires \textit{entropy refinement}-a mechanism implicitly embedded in group-relative advantage estimation that sustains informative entropy on positive rollouts while suppressing spurious entropy on negative ones. Guided by this insight, we propose \textbf{AsymGRPO}, an exploratory framework that explicitly decouples the modulation of positive and negative rollouts. This allows for independent control over the preservation of informative entropy and the suppression of spurious noise. Extensive experiments demonstrate that AsymGRPO achieves superior performance compared to strong baselines and exhibits the potential to synergize with existing entropy regularization methods. 

---
# Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection 

**Authors**: Yang Li, Qiang Sheng, Zhengjia Wang, Yehan Yang, Danding Wang, Juan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2604.04932)  

**Abstract**: The misuse of large language models (LLMs) requires precise detection of synthetic text. Existing works mainly follow binary or ternary classification settings, which can only distinguish pure human/LLM text or collaborative text at best. This remains insufficient for the nuanced regulation, as the LLM-polished human text and humanized LLM text often trigger different policy consequences. In this paper, we explore fine-grained LLM-generated text detection under a rigorous four-class setting. To handle such complexities, we propose RACE (Rhetorical Analysis for Creator-Editor Modeling), a fine-grained detection method that characterizes the distinct signatures of creator and editor. Specifically, RACE utilizes Rhetorical Structure Theory to construct a logic graph for the creator's foundation while extracting Elementary Discourse Unit-level features for the editor's style. Experiments show that RACE outperforms 12 baselines in identifying fine-grained types with low false alarms, offering a policy-aligned solution for LLM regulation. 

---
# TriAttention: Efficient Long Reasoning with Trigonometric KV Compression 

**Authors**: Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, Yukang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.04921)  

**Abstract**: Extended reasoning in large language models (LLMs) creates severe KV cache memory bottlenecks. Leading KV cache compression methods estimate KV importance using attention scores from recent post-RoPE queries. However, queries rotate with position during RoPE, making representative queries very few, leading to poor top-key selection and unstable reasoning. To avoid this issue, we turn to the pre-RoPE space, where we observe that Q and K vectors are highly concentrated around fixed non-zero centers and remain stable across positions -- Q/K concentration. We show that this concentration causes queries to preferentially attend to keys at specific distances (e.g., nearest keys), with the centers determining which distances are preferred via a trigonometric series. Based on this, we propose TriAttention to estimate key importance by leveraging these centers. Via the trigonometric series, we use the distance preference characterized by these centers to score keys according to their positions, and also leverage Q/K norms as an additional signal for importance estimation. On AIME25 with 32K-token generation, TriAttention matches Full Attention reasoning accuracy while achieving 2.5x higher throughput or 10.7x KV memory reduction, whereas leading baselines achieve only about half the accuracy at the same efficiency. TriAttention enables OpenClaw deployment on a single consumer GPU, where long context would otherwise cause out-of-memory with Full Attention. 

---
# Synthetic Sandbox for Training Machine Learning Engineering Agents 

**Authors**: Yuhang Zhou, Lizhu Zhang, Yifan Wu, Jiayi Liu, Xiangjun Fan, Zhuokai Zhao, Hong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2604.04872)  

**Abstract**: As large language model agents advance beyond software engineering (SWE) tasks toward machine learning engineering (MLE), verifying agent behavior becomes orders of magnitude more expensive: while SWE tasks can be verified via fast-executing unit tests, MLE verification requires running full ML pipelines -- data preprocessing, model training, and metric evaluation -- on large datasets at each rollout step, rendering trajectory-wise on-policy reinforcement learning (RL) prohibitively slow. Existing approaches retreat to supervised fine-tuning (SFT) or offline proxy rewards, sacrificing the exploration and generalization benefits of on-policy RL. We observe that sandbox data size is the primary source of this bottleneck. Based on this insight, we introduce SandMLE, a multi-agent framework that generates diverse, verifiable synthetic MLE environments from a small number of seed tasks, preserving the structural and technical complexity of real-world problems while constraining datasets to micro-scale (each task is paired with only 50-200 training samples). Through extensive experiments, we show that SandMLE reduces execution time by over 13 times, enabling large-scale, on-policy trajectory-wise RL for the first time in the MLE domain. On MLE-bench-lite, SandMLE yields significant gains over SFT baselines across Qwen3-8B, 14B, and 30B-A3B, with relative medal rate improvements ranging from 20.3% to 66.9%. Furthermore, the trained policy generalizes across unseen agentic scaffolds, achieving up to 32.4% better HumanRank score on MLE-Dojo. 

---
# Early Stopping for Large Reasoning Models via Confidence Dynamics 

**Authors**: Parsa Hosseini, Sumit Nawathe, Mahdi Salmani, Meisam Razaviyayn, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2604.04930)  

**Abstract**: Large reasoning models rely on long chain-of-thought generation to solve complex problems, but extended reasoning often incurs substantial computational cost and can even degrade performance due to overthinking. A key challenge is determining when the model should stop reasoning and produce the final answer. In this work, we study the confidence of intermediate answers during reasoning and observe two characteristic behaviors: correct reasoning trajectories often reach high-confidence answers early, while incorrect rollouts tend to produce long, unproductive reasoning traces and exhibit less reliable confidence dynamics. Motivated by these observations, we propose CoDE-Stop (Confidence Dynamics Early Stop), an early stopping method that leverages the dynamics of intermediate answer confidence to decide when to terminate reasoning, requiring no additional training and easily integrating into existing models. We evaluate CoDE-Stop on diverse reasoning and science benchmarks across multiple models. Compared to prior early stopping methods, it achieves a more favorable accuracy-compute tradeoff and reduces total token usage by 25-50% compared to standard full-length reasoning. In addition, we provide analyses of confidence dynamics during reasoning, offering insights into how confidence changes in both correct and incorrect trajectories. 

---
# How Far Are We? Systematic Evaluation of LLMs vs. Human Experts in Mathematical Contest in Modeling 

**Authors**: Yuhang Liu, Heyan Huang, Yizhe Yang, Hongyan Zhao, Zhizhuo Zeng, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.04791)  

**Abstract**: Large language models (LLMs) have achieved strong performance on reasoning benchmarks, yet their ability to solve real-world problems requiring end-to-end workflows remains unclear. Mathematical modeling competitions provide a stringent testbed for evaluating such end-to-end problem-solving capability. We propose a problem-oriented, stage-wise evaluation framework that assesses LLM performance across modeling stages using expert-verified criteria. We validate the framework's reliability by comparing automatic scores with independent human expert judgments on problems from the China Postgraduate Mathematical Contest in Modeling, demonstrating substantially stronger alignment than existing evaluation schemes. Using this framework, we reveal a comprehension-execution gap in state-of-the-art LLMs: while they perform well in early stages such as problem identification and formulation, they exhibit persistent deficiencies in execution-oriented stages including model solving, code implementation, and result analysis. These gaps persist even with increased model scale. We further trace these failures to insufficient specification, missing verification, and lack of validation, with errors propagating across stages without correction. Our findings suggest that bridging this gap requires approaches beyond model scaling, offering insights for applying LLMs to complex real-world problem solving. 

---
# What Makes Good Multilingual Reasoning? Disentangling Reasoning Traces with Measurable Features 

**Authors**: Dayeon Ki, Kevin Duh, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2604.04720)  

**Abstract**: Large Reasoning Models (LRMs) still exhibit large performance gaps between English and other languages, yet much current work assumes these gaps can be closed simply by making reasoning in every language resemble English reasoning. This work challenges this assumption by asking instead: what actually characterizes effective reasoning in multilingual settings, and to what extent do English-derived reasoning features genuinely help in other languages? We first define a suite of measurable reasoning features spanning multilingual alignment, reasoning step, and reasoning flow aspects of reasoning traces, and use logistic regression to quantify how each feature associates with final answer accuracy. We further train sparse autoencoders over multilingual traces to automatically discover latent reasoning concepts that instantiate or extend these features. Finally, we use the features as test-time selection policies to examine whether they can steer models toward stronger multilingual reasoning. Across two mathematical reasoning benchmarks, four LRMs, and 10 languages, we find that most features are positively associated with accuracy, but the strength of association varies considerably across languages and can even reverse in some. Our findings challenge English-centric reward designs and point toward adaptive objectives that accommodate language-specific reasoning patterns, with concrete implications for multilingual benchmark and reward design. 

---
# Metaphors We Compute By: A Computational Audit of Cultural Translation vs. Thinking in LLMs 

**Authors**: Yuan Chang, Jiaming Qu, Zhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.04732)  

**Abstract**: Large language models (LLMs) are often described as multilingual because they can understand and respond in many languages. However, speaking a language is not the same as reasoning within a culture. This distinction motivates a critical question: do LLMs truly conduct culture-aware reasoning? This paper presents a preliminary computational audit of cultural inclusivity in a creative writing task. We empirically examine whether LLMs act as culturally diverse creative partners or merely as cultural translators that leverage a dominant conceptual framework with localized expressions. Using a metaphor generation task spanning five cultural settings and several abstract concepts as a case study, we find that the model exhibits stereotyped metaphor usage for certain settings, as well as Western defaultism. These findings suggest that merely prompting an LLM with a cultural identity does not guarantee culturally grounded reasoning. 

---
# Individual and Combined Effects of English as a Second Language and Typos on LLM Performance 

**Authors**: Serena Liu, Yutong Yang, Prisha Sheth, Weixuan Dong, Mingjiao Diao, Xinru Zhu, Nikhil Banga, Oscar Melendez, Arnav Sharma, Minda Zhao, Marina Lin, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04723)  

**Abstract**: Large language models (LLMs) are used globally, and because much of their training data is in English, they typically perform best on English inputs. As a result, many non-native English speakers interact with them in English as a second language (ESL), and these inputs often contain typographical errors. Prior work has largely studied the effects of ESL variation and typographical errors separately, even though they often co-occur in real-world use. In this study, we use the Trans-EnV framework to transform standard English inputs into eight ESL variants and apply MulTypo to inject typos at three levels: low, moderate, and severe. We find that combining ESL variation and typos generally leads to larger performance drops than either factor alone, though the combined effect is not simply additive. This pattern is clearest on closed-ended tasks, where performance degradation can be characterized more consistently across ESL variants and typo levels, while results on open-ended tasks are more mixed. Overall, these findings suggest that evaluations on clean standard English may overestimate real-world model performance, and that evaluating ESL variation and typographical errors in isolation does not fully capture model behavior in realistic settings. 

---
# Hallucination Basins: A Dynamic Framework for Understanding and Controlling LLM Hallucinations 

**Authors**: Kalyan Cherukuri, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2604.04743)  

**Abstract**: Large language models (LLMs) hallucinate: they produce fluent outputs that are factually incorrect. We present a geometric dynamical systems framework in which hallucinations arise from task-dependent basin structure in latent space. Using autoregressive hidden-state trajectories across multiple open-source models and benchmarks, we find that separability is strongly task-dependent rather than universal: factoid settings can show clearer basin separation, whereas summarization and misconception-heavy settings are typically less stable and often overlap. We formalize this behavior with task-complexity and multi-basin theorems, characterize basin emergence in L-layer transformers, and show that geometry-aware steering can reduce hallucination probability without retraining. 

---
# Lighting Up or Dimming Down? Exploring Dark Patterns of LLMs in Co-Creativity 

**Authors**: Zhu Li, Jiaming Qu, Yuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04735)  

**Abstract**: Large language models (LLMs) are increasingly acting as collaborative writing partners, raising questions about their impact on human agency. In this exploratory work, we investigate five "dark patterns" in human-AI co-creativity -- subtle model behaviors that can suppress or distort the creative process: Sycophancy, Tone Policing, Moralizing, Loop of Death, and Anchoring. Through a series of controlled sessions where LLMs are prompted as writing assistants across diverse literary forms and themes, we analyze the prevalence of these behaviors in generated responses. Our preliminary results suggest that Sycophancy is nearly ubiquitous (91.7% of cases), particularly in sensitive topics, while Anchoring appears to be dependent on literary forms, surfacing most frequently in folktales. This study indicates that these dark patterns, often byproducts of safety alignment, may inadvertently narrow creative exploration and proposes design considerations for AI systems that effectively support creative writing. 

---
# IDIOLEX: Unified and Continuous Representations for Idiolectal and Stylistic Variation 

**Authors**: Anjali Kantharuban, Aarohi Srivastava, Fahim Faisal, Orevaoghene Ahia, Antonios Anastasopoulos, David Chiang, Yulia Tsvetkov, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2604.04704)  

**Abstract**: Existing sentence representations primarily encode what a sentence says, rather than how it is expressed, even though the latter is important for many applications. In contrast, we develop sentence representations that capture style and dialect, decoupled from semantic content. We call this the task of idiolectal representation learning. We introduce IDIOLEX, a framework for training models that combines supervision from a sentence's provenance with linguistic features of a sentence's content, to learn a continuous representation of each sentence's style and dialect. We evaluate the approach on dialects of both Arabic and Spanish. The learned representations capture meaningful variation and transfer across domains for analysis and classification. We further explore the use of these representations as training objectives for stylistically aligning language models. Our results suggest that jointly modeling individual and community-level variation provides a useful perspective for studying idiolect and supports downstream applications requiring sensitivity to stylistic differences, such as developing diverse and accessible LLMs. 

---
# HUKUKBERT: Domain-Specific Language Model for Turkish Law 

**Authors**: Mehmet Utku Öztürk, Tansu Türkoğlu, Buse Buz-Yalug  

**Link**: [PDF](https://arxiv.org/pdf/2604.04790)  

**Abstract**: Recent advances in natural language processing (NLP) have increasingly enabled LegalTech applications, yet existing studies specific to Turkish law have still been limited due to the scarcity of domain-specific data and models. Although extensive models like LEGAL-BERT have been developed for English legal texts, the Turkish legal domain lacks a domain-specific high-volume counterpart. In this paper, we introduce HukukBERT, the most comprehensive legal language model for Turkish, trained on a 18 GB cleaned legal corpus using a hybrid Domain-Adaptive Pre-Training (DAPT) methodology integrating Whole-Word Masking, Token Span Masking, Word Span Masking, and targeted Keyword Masking. We systematically compared our 48K WordPiece tokenizer and DAPT approach against general-purpose and existing domain-specific Turkish models. Evaluated on a novel Legal Cloze Test benchmark -- a masked legal term prediction task designed for Turkish court decisions -- HukukBERT achieves state-of-the-art performance with 84.40\% Top-1 accuracy, substantially outperforming existing models. Furthermore, we evaluated HukukBERT in the downstream task of structural segmentation of official Turkish court decisions, where it achieves a 92.8\% document pass rate, establishing a new state-of-the-art. We release HukukBERT to support future research in Turkish legal NLP tasks, including recognition of named entities, prediction of judgment, and classification of legal documents. 

---
# PassiveQA: A Three-Action Framework for Epistemically Calibrated Question Answering via Supervised Finetuning 

**Authors**: Madhav S Baidya  

**Link**: [PDF](https://arxiv.org/pdf/2604.04565)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance in question answering and retrieval-augmented generation (RAG), yet they implicitly assume that user queries are fully specified and answerable. In real-world settings, queries are often incomplete, ambiguous, or missing critical variables, leading models to produce overconfident or hallucinated responses.
In this work, we study decision-aware query resolution under incomplete information, where a model must determine whether to Answer, Ask for clarification, or Abstain. We show that standard and enhanced RAG systems do not reliably exhibit such epistemic awareness, defaulting to answer generation even when information is insufficient.
To address this, we propose PassiveQA, a three-action framework that aligns model behaviour with information sufficiency through supervised finetuning. Our approach integrates structured information-state representations, knowledge graph-grounded context, and a finetuned planner that explicitly models missing variables and decision reasoning.
Experiments across multiple QA datasets show that the finetuned planner achieves significant improvements in macro F1 and abstention recall while reducing hallucination rates, under a compute-constrained training regime.
These results provide strong empirical evidence that epistemic decision-making must be learned during training rather than imposed at inference time. 

---
# Multilingual Prompt Localization for Agent-as-a-Judge: Language and Backbone Sensitivity in Requirement-Level Evaluation 

**Authors**: Alhasan Mahmood, Samir Abdaljalil, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2604.04532)  

**Abstract**: Evaluation language is typically treated as a fixed English default in agentic code benchmarks, yet we show that changing the judge's language can invert backbone rankings. We localize the Agent-as-a-Judge prompt stack to five typologically diverse languages (English, Arabic, Turkish, Chinese, Hindi) and evaluate 55 DevAI development tasks across three developer-agent frameworks and six judge backbones, totaling 4950 judge runs. The central finding is that backbone and language interact: GPT-4o achieves the highest satisfaction in English (44.72\%), while Gemini leads in Arabic (51.72\%, $p<0.001$ vs.\ GPT-4o) and Hindi (53.22\%). No single backbone dominates across all languages, and inter-backbone agreement on individual requirement judgments is modest (Fleiss' $\kappa \leq 0.231$). A controlled ablation further shows that localizing judge-side instructions, not just benchmark content, can be decisive: Hindi satisfaction drops from 42.8\% to 23.2\% under partial localization. These results indicate that language should be treated as an explicit evaluation variable in agentic benchmarks. Full requirement-level judgments and runtime statistics are released for reproducibility. 

---
# Conversational Control with Ontologies for Large Language Models: A Lightweight Framework for Constrained Generation 

**Authors**: Barbara Gendron, Gaël Guibon, Mathieu d'Aquin  

**Link**: [PDF](https://arxiv.org/pdf/2604.04450)  

**Abstract**: Conversational agents based on Large Language Models (LLMs) have recently emerged as powerful tools for human-computer interaction. Nevertheless, their black-box nature implies challenges in predictability and a lack of personalization, both of which can be addressed by controlled generation. This work proposes an end-to-end method to obtain modular and explainable control over LLM outputs through ontological definitions of aspects related to the conversation. Key aspects are modeled and used as constraints; we then further fine-tune the LLM to generate content accordingly. To validate our approach, we explore two tasks that tackle two key conversational aspects: the English proficiency level and the polarity profile of the content. Using a hybrid fine-tuning procedure on seven state-of-the-art, open-weight conversational LLMs, we show that our method consistently outperforms pre-trained baselines, even on smaller models. Beyond quantitative gains, the framework remains model-agnostic, lightweight, and interpretable, enabling reusable control strategies that can be extended to new domains and interaction goals. This approach enhances alignment with strategy instructions and demonstrates the effectiveness of ontology-driven control in conversational systems. 

---
# DeonticBench: A Benchmark for Reasoning over Rules 

**Authors**: Guangyao Dou, Luis Brena, Akhil Deo, William Jurayj, Jingyu Zhang, Nils Holzenberger, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2604.04443)  

**Abstract**: Reasoning with complex, context-specific rules remains challenging for large language models (LLMs). In legal and policy settings, this manifests as deontic reasoning: reasoning about obligations, permissions, and prohibitions under explicit rules. While many recent benchmarks emphasize short-context mathematical reasoning, fewer focus on long-context, high-stakes deontic reasoning. To address this gap, we introduce DEONTICBENCH, a benchmark of 6,232 tasks across U.S. federal taxes, airline baggage policies, U.S. immigration administration, and U.S. state housing law. These tasks can be approached in multiple ways, including direct reasoning in language or with the aid of symbolic computation. Besides free-form chain-of-thought reasoning, DEONTICBENCH enables an optional solver-based workflow in which models translate statutes and case facts into executable Prolog, leading to formal problem interpretations and an explicit program trace. We release reference Prolog programs for all instances. Across frontier LLMs and coding models, best hard-subset performance reaches only 44.4% on SARA Numeric and 46.6 macro-F1 on Housing. We further study training with supervised fine-tuning and reinforcement learning for symbolic program generation. Although training improves Prolog generation quality, current RL methods still fail to solve these tasks reliably. Overall, DEONTICBENCH provides a benchmark for studying context-grounded rule reasoning in real-world domains under both symbolic and non-symbolic settings. 

---
# Structured Causal Video Reasoning via Multi-Objective Alignment 

**Authors**: Zinuo Li, Yongxin Guo, Jun Liu, Jiawei Zhan, Xi Jiang, Chengjie Wang, Mohammed Bennamoun, Farid Boussaid, Feng Zheng, Qiuhong Ke  

**Link**: [PDF](https://arxiv.org/pdf/2604.04415)  

**Abstract**: Human understanding of video dynamics is typically grounded in a structured mental representation of entities, actions, and temporal relations, rather than relying solely on immediate deductive reasoning. In contrast, existing Video-LLMs largely depend on unstructured video reasoning, where critical visual evidence is embedded in verbose textual descriptions and temporal causality is often weakly modeled. This leads to inefficient processes and fragile causal inference. To bridge this cognitive gap, we propose constructing a compact representation of salient events and their causal relationships, which we name Structured Event Facts, prior to the reasoning stage. This structured prior serves as an explicit constraint to promote concise and causally grounded reasoning, while also making intermediate evidence easier to verify. To effectively train models on such structured facts, we introduce CausalFact-60K and a four-stage training pipeline comprising facts alignment, format warm-start, thinking warm-start, and reinforcement learning-based post-training. During RL stage, we find that this framework introduces competing objectives, as structural completeness and causal fidelity must be balanced against reasoning length, making it difficult to optimize. We address this challenge by formulating the optimization as a Multi-Objective Reinforcement Learning (MORL) problem and explicitly optimizing toward the Pareto-Frontier to balance these trade-offs. As a result, we introduce Factum-4B, which yields more reliable reasoning and delivers stronger performance on challenging video understanding tasks requiring fine-grained temporal inference. 

---
# Same Geometry, Opposite Noise: Transformer Magnitude Representations Lack Scalar Variability 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.04469)  

**Abstract**: Scalar variability -- the finding that representational noise scales proportionally with magnitude, producing a constant coefficient of variation -- is a hallmark of biological magnitude systems. We tested whether transformer language models exhibit this property by analysing the dispersion of hidden-state representations across carrier sentences for 26 numerical magnitudes in three 7-8B parameter models (Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, Llama-3-8B-Base; data from Cacioli, 2026). We found the opposite: representational variability decreased with magnitude along the magnitude axis (scaling exponent alpha approx -0.19; 0/16 primary layers with alpha > 0, all three models). The negative sign was consistent in full-dimensional space (alpha approx -0.04) and after sentence-identity correction (alpha approx -0.007). The anti-scalar pattern was 3-5x stronger along the magnitude axis than orthogonal dimensions, and corpus frequency strongly predicted per-magnitude variability (rho = .84). These results demonstrate that distributional learning alone is insufficient to produce scalar variability: transformers reproduce log-compressive magnitude geometry but not the constant-CV noise signature observed in biological systems. 

---
# Responses Fall Short of Understanding: Revealing the Gap between Internal Representations and Responses in Visual Document Understanding 

**Authors**: Haruka Kawasaki, Ryota Tanaka, Kyosuke Nishida  

**Link**: [PDF](https://arxiv.org/pdf/2604.04411)  

**Abstract**: Visual document understanding (VDU) is a challenging task for large vision language models (LVLMs), requiring the integration of visual perception, text recognition, and reasoning over structured layouts. Although recent LVLMs have shown progress on VDU benchmarks, their performance is typically evaluated based on generated responses, which may not necessarily reflect whether the model has actually captured the required information internally. In this paper, we investigate how information required to solve VDU tasks is represented across different layers of LLMs within LVLMs using linear probing. Our study reveals that (1) there is a clear gap between internal representations and generated responses, and (2) information required to solve the task is often encoded more linearly from intermediate layers than from the final layer. Motivated by these findings, we explore fine-tuning strategies that target intermediate layers. Experiments show that fine-tuning intermediate layers improves both linear probing accuracy and response accuracy while narrowing the gap. 

---
# How Alignment Routes: Localizing, Scaling, and Controlling Policy Circuits in Language Models 

**Authors**: Gregory N. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2604.04385)  

**Abstract**: We identify a recurring sparse routing mechanism in alignment-trained language models: a gate attention head reads detected content and triggers downstream amplifier heads that boost the signal toward refusal. Using political censorship and safety refusal as natural experiments, we trace this mechanism across 9 models from 6 labs, all validated on corpora of 120 prompt pairs. The gate head passes necessity and sufficiency interchange tests (p < 0.001, permutation null), and core amplifier heads are stable under bootstrap resampling (Jaccard 0.92-1.0). Three same-generation scaling pairs show that routing distributes at scale (ablation up to 17x weaker) while remaining detectable by interchange. By modulating the detection-layer signal, we continuously control policy strength from hard refusal through steering to factual compliance, with routing thresholds that vary by topic. The circuit also reveals a structural separation between intent recognition and policy routing: under cipher encoding, the gate head's routing contribution collapses (78% in Phi-4 at n=120) while the model responds with puzzle-solving rather than refusal. The routing mechanism never fires, even though probe scores at deeper layers indicate the model begins to represent the harmful content. This asymmetry is consistent with different robustness properties of pretraining and post-training: broad semantic understanding versus narrower policy binding that generalizes less well under input transformation. 

---
# Compressible Softmax-Attended Language under Incompressible Attention 

**Authors**: Wonsuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.04384)  

**Abstract**: Across every attention head in five transformer language models (124M--7B parameters, four architecture families), the logit energy field $\tilde{E}$ reaches 90\% of its variance in 2--11 singular components. The \emph{learned} interaction matrix $W_Q^\mathrm{T} W_K$ needs 38--75 components for the same threshold out of $d_h \in \{64, 128\}$. The spectral gap is $5$--$25\times$ in effective rank. The attention mechanism allocates capacity uniformly across all $d_h$ dimensions, but language concentrates the actual interaction into a few. The compressibility of softmax-attended language is a property of the data, not the frame that analyzes it. 

---
# Adaptive Cost-Efficient Evaluation for Reliable Patent Claim Validation 

**Authors**: Yongmin Yoo, Qiongkai Xu, Longbing Cao  

**Link**: [PDF](https://arxiv.org/pdf/2604.04295)  

**Abstract**: Automated validation of patent claims demands zero-defect tolerance, as even a single structural flaw can render a claim legally defective. Existing evaluation paradigms suffer from a rigidity-resource dilemma: lightweight encoders struggle with nuanced legal dependencies, while exhaustive verification via Large Language Models (LLMs) is prohibitively costly. To bridge this gap, we propose ACE (Adaptive Cost-efficient Evaluation), a hybrid framework that uses predictive entropy to route only high-uncertainty claims to an expert LLM. The expert then executes a Chain of Patent Thought (CoPT) protocol grounded in 35 U.S.C. statutory standards. This design enables ACE to handle long-range legal dependencies more effectively while preserving efficiency. ACE achieves the best F1 among the evaluated methods at 94.95\%, while reducing operational costs by 78\% compared to standalone LLM deployments. We also construct ACE-40k, a 40,000-claim benchmark with MPEP-grounded error annotations, to facilitate further research. 

---
# High-Stakes Personalization: Rethinking LLM Customization for Individual Investor Decision-Making 

**Authors**: Yash Ganpat Sawant  

**Link**: [PDF](https://arxiv.org/pdf/2604.04300)  

**Abstract**: Personalized LLM systems have advanced rapidly, yet most operate in domains where user preferences are stable and ground truth is either absent or subjective. We argue that individual investor decision-making presents a uniquely challenging domain for LLM personalization - one that exposes fundamental limitations in current customization paradigms. Drawing on our system, built and deployed for AI-augmented portfolio management, we identify four axes along which individual investing exposes fundamental limitations in standard LLM customization: (1) behavioral memory complexity, where investor patterns are temporally evolving, self-contradictory, and financially consequential; (2) thesis consistency under drift, where maintaining coherent investment rationale over weeks or months strains stateless and session-bounded architectures; (3) style-signal tension, where the system must simultaneously respect personal investment philosophy and surface objective evidence that may contradict it; and (4) alignment without ground truth, where personalization quality cannot be evaluated against a fixed label set because outcomes are stochastic and delayed. We describe the architectural responses that emerged from building the system and propose open research directions for personalized NLP in high-stakes, temporally extended decision domains. 

---
# Benchmarking Multi-turn Medical Diagnosis: Hold, Lure, and Self-Correction 

**Authors**: Jinrui Fang, Runhan Chen, Xu Yang, Jian Yu, Jiawei Xu, Ashwin Vinod, Wenqi Shi, Tianlong Chen, Heng Ji, ChengXiang Zhai, Ying Ding, Yuji Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04325)  

**Abstract**: Large language models (LLMs) achieve high accuracy in medical diagnosis when all clinical information is provided in a single turn, yet how they behave under multi-turn evidence accumulation closer to real clinical reasoning remains unexplored. We introduce MINT (Medical Incremental N-Turn Benchmark), a high-fidelity, multi-turn medical diagnosis benchmark comprising 1,035 cases with clinically labeled evidence shards, controlled turn granularity, and information-preserving decomposition. Through systematic evaluation of 11 LLMs on MINT, we uncover three persistent behavioral patterns that significantly impact diagnostic decisions: (1) intent to answer, models rush to answer before sufficient evidence has been observed, with over 55% of answers committed within the first two turns; (2) self-correction, incorrect-to-correct answer revisions occur at up to 10.6 times the rate of correct-to-incorrect flips, revealing a latent capacity for self-correction that premature commitment forecloses; and (3) strong lures, clinically salient information such as laboratory results trigger premature answering even when models are explicitly instructed to wait. We translate these findings into clinically actionable guidance: deferring the diagnostic question to later turns reduces premature answering and improves accuracy at the first point of commitment by up to 62.6%, while reserving salient clinical evidence for later turns prevents a catastrophic accuracy drop of up to 23.3% caused by premature commitment. Our work provides both a controlled evaluation framework and concrete recommendations for improving the reliability of LLMs in multi-turn medical diagnosis. 

---
# GROUNDEDKG-RAG: Grounded Knowledge Graph Index for Long-document Question Answering 

**Authors**: Tianyi Zhang, Andreas Marfurt  

**Link**: [PDF](https://arxiv.org/pdf/2604.04359)  

**Abstract**: Retrieval-augmented generation (RAG) systems have been widely adopted in contemporary large language models (LLMs) due to their ability to improve generation quality while reducing the required input context length. In this work, we focus on RAG systems for long-document question answering. Current approaches suffer from a heavy reliance on LLM descriptions resulting in high resource consumption and latency, repetitive content across hierarchical levels, and hallucinations due to no or limited grounding in the source text. To improve both efficiency and factual accuracy through grounding, we propose GroundedKG-RAG, a RAG system in which the knowledge graph is explicitly extracted from and grounded in the source document. Specifically, we define nodes in GroundedKG as entities and actions, and edges as temporal or semantic relations, with each node and edge grounded in the original sentences. We construct GroundedKG from semantic role labeling (SRL) and abstract meaning representation (AMR) parses and then embed it for retrieval. During querying, we apply the same transformation to the query and retrieve the most relevant sentences from the grounded source text for question answering. We evaluate GroundedKG-RAG on examples from the NarrativeQA dataset and find that it performs on par with a state-of-the art proprietary long-context model at smaller cost and outperforms a competitive baseline. Additionally, our GroundedKG is interpretable and readable by humans, facilitating auditing of results and error analysis. 

---
# DARE: Diffusion Large Language Models Alignment and Reinforcement Executor 

**Authors**: Jingyi Yang, Yuxian Jiang, Xuhao Hu, Shuang Cheng, Biqing Qi, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2604.04215)  

**Abstract**: Diffusion large language models (dLLMs) are emerging as a compelling alternative to dominant autoregressive models, replacing strictly sequential token generation with iterative denoising and parallel generation dynamics. However, their open-source ecosystem remains fragmented across model families and, in particular, across post-training pipelines, where reinforcement learning objectives, rollout implementations and evaluation scripts are often released as paper-specific codebases. This fragmentation slows research iteration, raises the engineering burden of reproduction, and makes fair comparison across algorithms difficult. We present \textbf{DARE} (\textbf{d}LLMs \textbf{A}lignment and \textbf{R}einforcement \textbf{E}xecutor), an open framework for post-training and evaluating dLLMs. Built on top of verl~\cite{sheng2024hybridflow} and OpenCompass~\cite{2023opencompass}, DARE unifies supervised fine-tuning, parameter-efficient fine-tuning, preference optimization, and dLLM-specific reinforcement learning under a shared execution stack for both masked and block diffusion language models. Across representative model families including LLaDA, Dream, SDAR, and LLaDA2.x, DARE provides broad algorithmic coverage, reproducible benchmark evaluation, and practical acceleration. Extensive empirical results position that DARE serves as a reusable research substrate for developing, comparing, and deploying post-training methods for current and emerging dLLMs. 

---
# How Well Do Agentic Skills Work in the Wild: Benchmarking LLM Skill Usage in Realistic Settings 

**Authors**: Yujian Liu, Jiabao Ji, Li An, Tommi Jaakkola, Yang Zhang, Shiyu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04323)  

**Abstract**: Agent skills, which are reusable, domain-specific knowledge artifacts, have become a popular mechanism for extending LLM-based agents, yet formally benchmarking skill usage performance remains scarce. Existing skill benchmarking efforts focus on overly idealized conditions, where LLMs are directly provided with hand-crafted, narrowly-tailored task-specific skills for each task, whereas in many realistic settings, the LLM agent may have to search for and select relevant skills on its own, and even the closest matching skills may not be well-tailored for the task. In this paper, we conduct the first comprehensive study of skill utility under progressively challenging realistic settings, where agents must retrieve skills from a large collection of 34k real-world skills and may not have access to any hand-curated skills. Our findings reveal that the benefits of skills are fragile: performance gains degrade consistently as settings become more realistic, with pass rates approaching no-skill baselines in the most challenging scenarios. To narrow this gap, we study skill refinement strategies, including query-specific and query-agnostic approaches, and we show that query-specific refinement substantially recovers lost performance when the initial skills are of reasonable relevance and quality. We further demonstrate the generality of retrieval and refinement on Terminal-Bench 2.0, where they improve the pass rate of Claude Opus 4.6 from 57.7% to 65.5%. Our results, consistent across multiple models, highlight both the promise and the current limitations of skills for LLM-based agents. Our code is available at this https URL. 

---
# Which English Do LLMs Prefer? Triangulating Structural Bias Towards American English in Foundation Models 

**Authors**: Mir Tafseer Nayeem, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2604.04204)  

**Abstract**: Large language models (LLMs) are increasingly deployed in high-stakes domains, yet they expose only limited language settings, most notably "English (US)," despite the global diversity and colonial history of English. Through a postcolonial framing to explain the broader significance, we investigate how geopolitical histories of data curation, digital dominance, and linguistic standardization shape the LLM development pipeline. Focusing on two dominant standard varieties, American English (AmE) and British English (BrE), we construct a curated corpus of 1,813 AmE--BrE variants and introduce DiAlign, a dynamic, training-free method for estimating dialectal alignment using distributional evidence. We operationalize structural bias by triangulating evidence across three stages: (i) audits of six major pretraining corpora reveal systematic skew toward AmE, (ii) tokenizer analyses show that BrE forms incur higher segmentation costs, and (iii) generative evaluations show a persistent AmE preference in model outputs. To our knowledge, this is the first systematic and multi-faceted examination of dialectal asymmetries in standard English varieties across the phases of LLM development. We find that contemporary LLMs privilege AmE as the de facto norm, raising concerns about linguistic homogenization, epistemic injustice, and inequity in global AI deployment, while motivating practical steps toward more dialectally inclusive language technologies. 

---
# CAWN: Continuous Acoustic Wave Networks for Autoregressive Language Modeling 

**Authors**: Dejan Čugalj, Aleksandar Jevremovic  

**Link**: [PDF](https://arxiv.org/pdf/2604.04250)  

**Abstract**: Modern Large Language Models (LLMs) rely on Transformer self-attention, which scales quadratically with sequence length. Recent linear-time alternatives, like State Space Models (SSMs), often suffer from signal degradation over extended contexts. We introduce the Continuous Acoustic Wave Network (CAWN), a fully continuous sequence-mixing architecture. Instead of discrete matrix-based attention, CAWN projects hidden states into multi-headed complex-domain phasors, achieving sequence mixing through a causal, $O(L)$ Phase Accumulation mechanism. To prevent signal degradation over ultra-long contexts, we introduce a dual-gated Selective Phase Resonance mechanism incorporating Frequency-Dependent Retention, Hard-Threshold Gating via Straight-Through Estimation, and a Temporal Syntax Cache to capture short-term local dependencies. We also replace standard dense linear projections with Depth-wise Harmonic Convolutions for optimal spatial frequency mixing, augmented by Block Attention Residuals for depth-wise state routing. Scaled to a 150M-parameter model, CAWN utilizes custom Triton kernels for hardware-efficient, true-complex phase accumulation in float32. Trained via a continuous streaming loop on a 100-Billion-token corpus, the prototype is evaluated at a 5-Billion-token milestone. Empirical evaluations via a Targeted Semantic Retrieval protocol demonstrate robust vocabulary acquisition and extended explicitly learned contextual denoising. By leveraging $O(1)$ state-passing via chunked prefill, the model retrieves targeted information across 2,000,000 tokens while strictly plateauing at 8.72 GB of Peak VRAM, empirically overcoming the $O(L^2)$ context memory wall. 

---
# Emergent Inference-Time Semantic Contamination via In-Context Priming 

**Authors**: Marcin Abram  

**Link**: [PDF](https://arxiv.org/pdf/2604.04043)  

**Abstract**: Recent work has shown that fine-tuning large language models (LLMs) on insecure code or culturally loaded numeric codes can induce emergent misalignment, causing models to produce harmful content in unrelated downstream tasks. The authors of that work concluded that $k$-shot prompting alone does not induce this effect. We revisit this conclusion and show that inference-time semantic drift is real and measurable; however, it requires models of large-enough capability. Using a controlled experiment in which five culturally loaded numbers are injected as few-shot demonstrations before a semantically unrelated prompt, we find that models with richer cultural-associative representations exhibit significant distributional shifts toward darker, authoritarian, and stigmatized themes, while a simpler/smaller model does not. We additionally find that structurally inert demonstrations (nonsense strings) perturb output distributions, suggesting two separable mechanisms: structural format contamination and semantic content contamination. Our results map the boundary conditions under which inference-time contamination occurs, and carry direct implications for the security of LLM-based applications that use few-shot prompting. 

---
# Position: Logical Soundness is not a Reliable Criterion for Neurosymbolic Fact-Checking with LLMs 

**Authors**: Jason Chan, Robert Gaizauskas, Zhixue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.04177)  

**Abstract**: As large language models (LLMs) are increasing integrated into fact-checking pipelines, formal logic is often proposed as a rigorous means by which to mitigate bias, errors and hallucinations in these models' outputs. For example, some neurosymbolic systems verify claims by using LLMs to translate natural language into logical formulae and then checking whether the proposed claims are logically sound, i.e. whether they can be validly derived from premises that are verified to be true. We argue that such approaches structurally fail to detect misleading claims due to systematic divergences between conclusions that are logically sound and inferences that humans typically make and accept. Drawing on studies in cognitive science and pragmatics, we present a typology of cases in which logically sound conclusions systematically elicit human inferences that are unsupported by the underlying premises. Consequently, we advocate for a complementary approach: leveraging the human-like reasoning tendencies of LLMs as a feature rather than a bug, and using these models to validate the outputs of formal components in neurosymbolic systems against potentially misleading conclusions. 

---
# Unmasking Hallucinations: A Causal Graph-Attention Perspective on Factual Reliability in Large Language Models 

**Authors**: Sailesh kiran kurra, Shiek Ruksana, Vishal Borusu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04020)  

**Abstract**: This paper primarily focuses on the hallucinations caused due to AI language models(LLMs).LLMs have shown extraordinary Language understanding and generation capabilities .Still it has major a disadvantage hallucinations which give outputs which are factually incorrect ,misleading or unsupported by input data . These hallucinations cause serious problems in scenarios like medical diagnosis or legal this http URL this work,we propose causal graph attention network (GCAN) framework that reduces hallucinations through interpretation of internal attention flow within a transformer architecture with the help of constructing token level graphs that combine self attention weights and gradient based influence this http URL method quantifies each tokens factual dependency using a new metric called the Causal Contribution Score (CCS). We further introduce a fact-anchored graph reweighting layer that dynamically reduces the influence of hallucination prone nodes during generation. Experiments on standard benchmarks such as TruthfulQA and HotpotQA show a 27.8 percent reduction in hallucination rate and 16.4 percent improvement in factual accuracy over baseline retrieval-augmented generation (RAG) models. This work contributes to the interpretability,robustness, and factual reliability of future LLM architectures. 

---
# Many Preferences, Few Policies: Towards Scalable Language Model Personalization 

**Authors**: Cheol Woo Kum, Jai Moondra, Roozbeh Nahavandi, Andrew Perrault, Milind Tambe, Swati Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.04144)  

**Abstract**: The holy grail of LLM personalization is a single LLM for each user, perfectly aligned with that user's preferences. However, maintaining a separate LLM per user is impractical due to constraints on compute, memory, and system complexity. We address this challenge by developing a principled method for selecting a small portfolio of LLMs that captures representative behaviors across heterogeneous users. We model user preferences across multiple traits (e.g., safety, humor, brevity) through a multi-dimensional weight vector. Given reward functions across these dimensions, our algorithm PALM (Portfolio of Aligned LLMs) generates a small portfolio of LLMs such that, for any weight vector, the portfolio contains a near-optimal LLM for the corresponding scalarized objective. To the best of our knowledge, this is the first result that provides theoretical guarantees on both the size and approximation quality of LLM portfolios for personalization. It characterizes the trade-off between system cost and personalization, as well as the diversity of LLMs required to cover the landscape of user preferences. We provide empirical results that validate these guarantees and demonstrate greater output diversity over common baselines. 

---
# Shorter, but Still Trustworthy? An Empirical Study of Chain-of-Thought Compression 

**Authors**: Lingjie Zeng, Xiaofan Chen, Yanbo Wang, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.04120)  

**Abstract**: Long chain-of-thought (Long-CoT) reasoning models have motivated a growing body of work on compressing reasoning traces to reduce inference cost, yet existing evaluations focus almost exclusively on task accuracy and token savings. Trustworthiness properties, whether acquired or reinforced through post-training, are encoded in the same parameter space that compression modifies. This means preserving accuracy does not, a priori, guarantee preserving trustworthiness. We conduct the first systematic empirical study of how CoT compression affects model trustworthiness, evaluating multiple models of different scales along three dimensions: safety, hallucination resistance, and multilingual robustness. Under controlled comparisons, we find that CoT compression frequently introduces trustworthiness regressions and that different methods exhibit markedly different degradation profiles across dimensions. To enable fair comparison across bases, we propose a normalized efficiency score for each dimension that reveals how naïve scalar metrics can obscure trustworthiness trade-offs. As an existence proof, we further introduce an alignment-aware DPO variant that reduces CoT length by 19.3\% on reasoning benchmarks with substantially smaller trustworthiness loss. Our findings suggest that CoT compression should be optimized not only for efficiency but also for trustworthiness, treating both as equally important design constraints. 

---
# Extracting and Steering Emotion Representations in Small Language Models: A Methodological Comparison 

**Authors**: Jihoon Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2604.04064)  

**Abstract**: Small language models (SLMs) in the 100M-10B parameter range increasingly power production systems, yet whether they possess the internal emotion representations recently discovered in frontier models remains unknown. We present the first comparative analysis of emotion vector extraction methods for SLMs, evaluating 9 models across 5 architectural families (GPT-2, Gemma, Qwen, Llama, Mistral) using 20 emotions and two extraction methods (generation-based and comprehension-based). Generation-based extraction produces statistically superior emotion separation (Mann-Whitney p = 0.007; Cohen's d = -107.5), with the advantage modulated by instruction tuning and architecture. Emotion representations localize at middle transformer layers (~50% depth), following a U-shaped curve that is architecture-invariant from 124M to 3B parameters. We validate these findings against representational anisotropy baselines across 4 models and confirm causal behavioral effects through steering experiments, independently verified by an external emotion classifier (92% success rate, 37/40 scenarios). Steering reveals three regimes -- surgical (coherent text transformation), repetitive collapse, and explosive (text degradation) -- quantified by perplexity ratios and separated by model architecture rather than scale. We document cross-lingual emotion entanglement in Qwen, where steering activates semantically aligned Chinese tokens that RLHF does not suppress, raising safety concerns for multilingual deployment. This work provides methodological guidelines for emotion research on open-weight models and contributes to the Model Medicine series by bridging external behavioral profiling with internal representational analysis. 

---
# Predict, Don't React: Value-Based Safety Forecasting for LLM Streaming 

**Authors**: Pride Kavumba, Koki Wataoka, Huy H. Nguyen, Jiaxuan Li, Masaya Ohagi  

**Link**: [PDF](https://arxiv.org/pdf/2604.03962)  

**Abstract**: In many practical LLM deployments, a single guardrail is used for both prompt and response moderation. Prompt moderation operates on fully observed text, whereas streaming response moderation requires safety decisions to be made over partial generations. Existing text-based streaming guardrails commonly frame this output-side problem as boundary detection, training models to identify the earliest prefix at which a response has already become unsafe. In this work, we introduce StreamGuard, a unified model-agnostic streaming guardrail that instead formulates moderation as a forecasting problem: given a partial prefix, the model predicts the expected harmfulness of likely future continuations. We supervise this prediction using Monte Carlo rollouts, which enables early intervention without requiring exact token-level boundary annotations.
Across standard safety benchmarks, StreamGuard performs strongly both for input moderation and for streaming output moderation. At the 8B scale, StreamGuard improves aggregated input-moderation F1 from 86.7 to 88.2 and aggregated streaming output-moderation F1 from 80.4 to 81.9 relative to Qwen3Guard-Stream-8B-strict. On the QWENGUARDTEST response_loc streaming benchmark, StreamGuard reaches 97.5 F1, 95.1 recall, and 92.6% on-time intervention, compared to 95.9 F1, 92.1 recall, and 89.9% for Qwen3Guard-Stream-8B-stric, while reducing the miss rate from 7.9% to 4.9%. We further show that forecasting-based supervision transfers effectively across tokenizers and model families: with transferred targets, Gemma3-StreamGuard-1B reaches 81.3 response-moderation F1, 98.2 streaming F1, and a 3.5% miss rate. These results show that strong end-to-end streaming moderation can be obtained without exact boundary labels, and that forecasting future risk is an effective supervision strategy for low-latency safety intervention. 

---
# RUQuant: Towards Refining Uniform Quantization for Large Language Models 

**Authors**: Han Liu, Haotian Gao, Changya Li, Feng Zhang, Xiaotong Zhang, Wei Wang, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04013)  

**Abstract**: The increasing size and complexity of large language models (LLMs) have raised significant challenges in deployment efficiency, particularly under resource constraints. Post-training quantization (PTQ) has emerged as a practical solution by compressing models without requiring retraining. While existing methods focus on uniform quantization schemes for both weights and activations, they often suffer from substantial accuracy degradation due to the non-uniform nature of activation distributions. In this work, we revisit the activation quantization problem from a theoretical perspective grounded in the Lloyd-Max optimality conditions. We identify the core issue as the non-uniform distribution of activations within the quantization interval, which causes the optimal quantization point under the Lloyd-Max criterion to shift away from the midpoint of the interval. To address this issue, we propose a two-stage orthogonal transformation method, RUQuant. In the first stage, activations are divided into blocks. Each block is mapped to uniformly sampled target vectors using composite orthogonal matrices, which are constructed from Householder reflections and Givens rotations. In the second stage, a global Householder reflection is fine-tuned to further minimize quantization error using Transformer output discrepancies. Empirical results show that our method achieves near-optimal quantization performance without requiring model fine-tuning: RUQuant achieves 99.8% of full-precision accuracy with W6A6 and 97% with W4A4 quantization for a 13B LLM, within approximately one minute. A fine-tuned variant yields even higher accuracy, demonstrating the effectiveness and scalability of our approach. 

---
# From Plausible to Causal: Counterfactual Semantics for Policy Evaluation in Simulated Online Communities 

**Authors**: Agam Goyal, Yian Wang, Eshwar Chandrasekharan, Hari Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2604.03920)  

**Abstract**: LLM-based social simulations can generate believable community interactions, enabling ``policy wind tunnels'' where governance interventions are tested before deployment. But believability is not causality. Claims like ``intervention $A$ reduces escalation'' require causal semantics that current simulation work typically does not specify. We propose adopting the causal counterfactual framework, distinguishing \textit{necessary causation} (would the outcome have occurred without the intervention?) from \textit{sufficient causation} (does the intervention reliably produce the outcome?). This distinction maps onto different stakeholder needs: moderators diagnosing incidents require evidence about necessity, while platform designers choosing policies require evidence about sufficiency. We formalize this mapping, show how simulation design can support estimation under explicit assumptions, and argue that the resulting quantities should be interpreted as simulator-conditional causal estimates whose policy relevance depends on simulator fidelity. Establishing this framework now is essential: it helps define what adequate fidelity means and moves the field from simulations that look realistic toward simulations that can support policy changes. 

---
# AdaptFuse: Training-Free Sequential Preference Learning via Externalized Bayesian Inference 

**Authors**: Fangzhou Lin, Peiran Li, Shuo Xing, Siyuan Yang, Qianwen Ge, Kazunori Yamada, Ziming Zhang, Haichong Zhang, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2604.03925)  

**Abstract**: Large language models struggle to accumulate evidence across multiple rounds of user interaction, failing to update their beliefs in a manner consistent with Bayesian inference. Existing solutions require fine-tuning on sensitive user interaction data, limiting their applicability in privacy-conscious settings. We propose AdaptFuse, a training-free framework that externalizes probabilistic computation entirely from the LLM: a symbolic module maintains a Bayesian posterior over a discrete hypothesis set, while a frozen LLM contributes semantic reasoning via multi-sample Dirichlet aggregation. The two signals are combined through entropy-adaptive fusion, which automatically weights each source by its predictive confidence, shifting reliance from the LLM to the symbolic posterior as evidence accumulates. We evaluate across three domains: flight recommendation, hotel recommendation, and web shopping; on Gemma 2 9B, Llama 3 8B, and Qwen 2.5 7B. AdaptFuse consistently outperforms both prompting baselines and fine-tuned Bayesian Teaching models on all tasks, with accuracy improving monotonically over interaction rounds. These results demonstrate that principled inference-time algorithms can substitute for fine-tuning in personalized recommendation, without storing or training on sensitive user data. All the code and materials will be open-sourced. 

---
# Uncertainty as a Planning Signal: Multi-Turn Decision Making for Goal-Oriented Conversation 

**Authors**: Xinyi Ling, Ye Liu, Reza Averly, Xia Ning  

**Link**: [PDF](https://arxiv.org/pdf/2604.03924)  

**Abstract**: Goal-oriented conversational systems require making sequential decisions under uncertainty about the user's intent, where the algorithm must balance information acquisition and target commitment over multiple turns. Existing approaches address this challenge from different perspectives: structured methods enable multi-step planning but rely on predefined schemas, while LLM-based approaches support flexible interactions but lack long-horizon decision making, resulting in poor coordination between information acquisition and target commitment. To address this limitation, we formulate goal-oriented conversation as an uncertainty-aware sequential decision problem, where uncertainty serves as a guiding signal for multi-turn decision making. We propose a Conversation Uncertainty-aware Planning framework (CUP) that integrates language models with structured planning: a language model proposes feasible actions, and a planner evaluates their long-term impact on uncertainty reduction. Experiments on multiple conversational benchmarks show that CUP consistently improves success rates while requiring fewer interaction turns. Further analysis demonstrates that uncertainty-aware planning contributes to more efficient information acquisition and earlier confident commitment. 

---
# When Models Know More Than They Say: Probing Analogical Reasoning in LLMs 

**Authors**: Hope McGovern, Caroline Craig, Thomas Lippincott, Hale Sirin  

**Link**: [PDF](https://arxiv.org/pdf/2604.03877)  

**Abstract**: Analogical reasoning is a core cognitive faculty essential for narrative understanding. While LLMs perform well when surface and structural cues align, they struggle in cases where an analogy is not apparent on the surface but requires latent information, suggesting limitations in abstraction and generalisation. In this paper we compare a model's probed representations with its prompted performance at detecting narrative analogies, revealing an asymmetry: for rhetorical analogies, probing significantly outperforms prompting in open-source models, while for narrative analogies, they achieve a similar (low) performance. This suggests that the relationship between internal representations and prompted behavior is task-dependent and may reflect limitations in how prompting accesses available information. 

---
# Testing the Limits of Truth Directions in LLMs 

**Authors**: Angelos Poulis, Mark Crovella, Evimaria Terzi  

**Link**: [PDF](https://arxiv.org/pdf/2604.03754)  

**Abstract**: Large language models (LLMs) have been shown to encode truth of statements in their activation space along a linear truth direction. Previous studies have argued that these directions are universal in certain aspects, while more recent work has questioned this conclusion drawing on limited generalization across some settings. In this work, we identify a number of limits of truth-direction universality that have not been previously understood. We first show that truth directions are highly layer-dependent, and that a full understanding of universality requires probing at many layers in the model. We then show that truth directions depend heavily on task type, emerging in earlier layers for factual and later layers for reasoning tasks; they also vary in performance across levels of task complexity. Finally, we show that model instructions dramatically affect truth directions; simple correctness evaluation instructions significantly affect the generalization ability of truth probes. Our findings indicate that universality claims for truth directions are more limited than previously known, with significant differences observable for various model layers, task difficulties, task types, and prompt templates. 

---
# Your Agent is More Brittle Than You Think: Uncovering Indirect Injection Vulnerabilities in Agentic LLMs 

**Authors**: Wenhui Zhu, Xuanzhao Dong, Xiwen Chen, Rui Cai, Peijie Qiu, Zhipeng Wang, Oana Frunza, Shao Tang, Jindong Gu, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03870)  

**Abstract**: The rapid deployment of open-source frameworks has significantly advanced the development of modern multi-agent systems. However, expanded action spaces, including uncontrolled privilege exposure and hidden inter-system interactions, pose severe security challenges. Specifically, Indirect Prompt Injections (IPI), which conceal malicious instructions within third-party content, can trigger unauthorized actions such as data exfiltration during normal operations. While current security evaluations predominantly rely on isolated single-turn benchmarks, the systemic vulnerabilities of these agents within complex dynamic environments remain critically underexplored. To bridge this gap, we systematically evaluate six defense strategies against four sophisticated IPI attack vectors across nine LLM backbones. Crucially, we conduct our evaluation entirely within dynamic multi-step tool-calling environments to capture the true attack surface of modern autonomous agents. Moving beyond binary success rates, our multidimensional analysis reveals a pronounced fragility. Advanced injections successfully bypass nearly all baseline defenses, and some surface-level mitigations even produce counterproductive side effects. Furthermore, while agents execute malicious instructions almost instantaneously, their internal states exhibit abnormally high decision entropy. Motivated by this latent hesitation, we investigate Representation Engineering (RepE) as a robust detection strategy. By extracting hidden states at the tool-input position, we revealed that the RepE-based circuit breaker successfully identifies and intercepts unauthorized actions before the agent commits to them, achieving high detection accuracy across diverse LLM backbones. This study exposes the limitations of current IPI defenses and provides a highly practical paradigm for building resilient multi-agent architectures. 

---
# POEMetric: The Last Stanza of Humanity 

**Authors**: Bingru Li, Han Wang, Hazel Wilkinson  

**Link**: [PDF](https://arxiv.org/pdf/2604.03695)  

**Abstract**: Large Language Models (LLMs) can compose poetry, but how far are they from human poets? In this paper, we introduce POEMetric, the first comprehensive framework for poetry evaluation, examining 1) basic instruction-following abilities in generating poems according to a certain form and theme, 2) advanced abilities of showing creativity, lexical diversity, and idiosyncrasy, evoking emotional resonance, and using imagery and literary devices, and 3) general appraisal of the overall poem quality and estimation of authorship. We curated a human poem dataset - 203 English poems of 7 fixed forms annotated with meter, rhyme patterns and themes - and experimented with 30 LLMs for poetry generation based on the same forms and themes of the human data, totaling 6,090 LLM poems. Based on POEMetric, we assessed the performance of both human poets and LLMs through rule-based evaluation and LLM-as-a-judge, whose results were validated by human experts. Results show that, though the top model achieved high form accuracy (4.26 out of 5.00, with Gemini-2.5-Pro as a judge; same below) and theme alignment (4.99), all models failed to reach the same level of advanced abilities as human poets, who achieved unparalleled creativity (4.02), idiosyncrasy (3.95), emotional resonance (4.06), and skillful use of imagery (4.49) and literary devices (4.67). Humans also defeated the best-performing LLM in overall poem quality (4.22 vs. 3.20). As such, poetry generation remains a formidable challenge for LLMs. Data and codes are released at this https URL. 

---
# I-CALM: Incentivizing Confidence-Aware Abstention for LLM Hallucination Mitigation 

**Authors**: Haotian Zong, Binze Li, Yufei Long, Sinyin Chang, Jialong Wu, Gillian K. Hadfield  

**Link**: [PDF](https://arxiv.org/pdf/2604.03904)  

**Abstract**: Large language models (LLMs) frequently produce confident but incorrect answers, partly because common binary scoring conventions reward answering over honestly expressing uncertainty. We study whether prompt-only interventions -- explicitly announcing reward schemes for answer-versus-abstain decisions plus humility-oriented normative principles -- can reduce hallucination risk without modifying the model. Our focus is epistemic abstention on factual questions with a verifiable answer, where current LLMs often fail to abstain despite being uncertain about their answers. We first assess self-reported verbal confidence as a usable uncertainty signal, showing stability under prompt paraphrasing and reasonable calibration against a token-probability baseline. We then study I-CALM, a prompt-based framework that (i) elicits verbal confidence, (ii) partially rewards abstention through explicit reward schemes, and (iii) adds lightweight normative principles emphasizing truthfulness, humility, and responsibility. Using GPT-5 mini on PopQA as the main setting, we find that confidence-eliciting, abstention-rewarding prompts, especially with norms, reduce the false-answer rate on answered cases mainly by identifying and shifting error-prone cases to abstention and re-calibrating their confidence. This trades coverage for reliability while leaving forced-answer performance largely unchanged. Varying the abstention reward yields a clear abstention-hallucination frontier. Overall, results show the framework can improve selective answering on factual questions without retraining, with the magnitude of effect varying across models and datasets. Code is available at the following this https URL. 

---
# Researchers waste 80% of LLM annotation costs by classifying one text at a time 

**Authors**: Christian Pipal, Eva-Maria Vogel, Morgan Wack, Frank Esser  

**Link**: [PDF](https://arxiv.org/pdf/2604.03684)  

**Abstract**: Large language models (LLMs) are increasingly being used for text classification across the social sciences, yet researchers overwhelmingly classify one text per variable per prompt. Coding 100,000 texts on four variables requires 400,000 API calls. Batching 25 items and stacking all variables into a single prompt reduces this to 4,000 calls, cutting token costs by over 80%. Whether this degrades coding quality is unknown. We tested eight production LLMs from four providers on 3,962 expert-coded tweets across four tasks, varying batch size from 1 to 1,000 items and stacking up to 25 coding dimensions per prompt. Six of eight models maintained accuracy within 2 pp of the single-item baseline through batch sizes of 100. Variable stacking with up to 10 dimensions produced results comparable to single-variable coding, with degradation driven by task complexity rather than prompt length. Within this safe operating range, the measurement error from batching and stacking is smaller than typical inter-coder disagreement in the ground-truth data. 

---
# The Format Tax 

**Authors**: Ivan Yee Lee, Loris D'Antoni, Taylor Berg-Kirkpatrick  

**Link**: [PDF](https://arxiv.org/pdf/2604.03616)  

**Abstract**: Asking a large language model to respond in JSON should be a formatting choice, not a capability tax. Yet we find that structured output requirements -- JSON, XML, LaTeX, Markdown -- substantially degrade reasoning and writing performance across open-weight models. The research response has focused on constrained decoding, but sampling bias accounts for only a fraction of the degradation. The dominant cost enters at the prompt: format-requesting instructions alone cause most of the accuracy loss, before any decoder constraint is applied. This diagnosis points to a simple principle: decouple reasoning from formatting. Whether by generating freeform first and reformatting in a second pass, or by enabling extended thinking within a single generation, separating the two concerns substantially recovers lost accuracy. Across six open-weight models, four API models, four formats, and tasks spanning math, science, logic, and writing, decoupling recovers most lost accuracy. Notably, most recent closed-weight models show little to no format tax, suggesting the problem is not inherent to structured generation but a gap that current open-weight models have yet to close. Code is available at this https URL. 

---
# Document-Level Numerical Reasoning across Single and Multiple Tables in Financial Reports 

**Authors**: Yi-Cheng Wang, Wei-An Wang, Chu-Song Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.03664)  

**Abstract**: Despite the strong language understanding abilities of large language models (LLMs), they still struggle with reliable question answering (QA) over long, structured documents, particularly for numerical reasoning. Financial annual reports exemplify this difficulty: financial statement analysis often hinges on accurate arithmetic, and analysts derive key indicators by integrating evidence scattered across multiple tables and narrative text. However, existing benchmarks focus largely on single-table settings, leaving cross-table document-level numerical reasoning underexplored. To address this gap, we introduce FinLongDocQA, a dataset for both single-table and cross-table financial numerical reasoning in long-context reports. Evaluating both closed-source and open-source LLMs on FinLongDocQA reveals two bottlenecks: (1) annual reports often exceed 129k tokens, exacerbating the context rot problem for locating relevant tables; and (2) even when relevant evidence is located, LLMs remain prone to errors in multi-step numerical reasoning. We propose FinLongDocAgent, a Multi-Agent Multi-Round Retrieval-Augmented Generation (RAG) approach that iteratively retrieves evidence, performs intermediate calculations, and verifies results across rounds. Experiments highlight the importance of iterative retrieval and verification for reliable numerical QA in long financial documents. 

---
# 'Layer su Layer': Identifying and Disambiguating the Italian NPN Construction in BERT's family 

**Authors**: Greta Gorzoni, Ludovica Pannitto, Francesca Masini  

**Link**: [PDF](https://arxiv.org/pdf/2604.03673)  

**Abstract**: Interpretability research has highlighted the importance of evaluating Pretrained Language Models (PLMs) and in particular contextual embeddings against explicit linguistic theories to determine what linguistic information they encode. This study focuses on the Italian NPN (noun-preposition-noun) constructional family, challenging some of the theoretical and methodological assumptions underlying previous experimental designs and extending this type of research to a lesser-investigated language. Contextual vector representations are extracted from BERT and used as input to layer-wise probing classifiers, systematically evaluating information encoded across the model's internal layers. The results shed light on the extent to which constructional form and meaning are reflected in contextual embeddings, contributing empirical evidence to the dialogue between constructionist theory and neural language modelling 

---
# Unlocking Prompt Infilling Capability for Diffusion Language Models 

**Authors**: Yoshinari Fujinuma, Keisuke Sakaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2604.03677)  

**Abstract**: Masked diffusion language models (dLMs) generate text through bidirectional denoising, yet this capability remains locked for infilling prompts. This limitation is an artifact of the current supervised finetuning (SFT) convention of applying response-only masking. To unlock this capability, we extend full-sequence masking during SFT, where both prompts and responses are masked jointly. Once unlocked, the model infills masked portions of a prompt template conditioned on few-shot examples. We show that such model-infilled prompts match or surpass manually designed templates, transfer effectively across models, and are complementary to existing prompt optimization methods. Our results suggest that training practices, not architectural limitations, are the primary bottleneck preventing masked diffusion language models from infilling effective prompts 

---
# Unveiling Language Routing Isolation in Multilingual MoE Models for Interpretable Subnetwork Adaptation 

**Authors**: Kening Zheng, Wei-Chieh Huang, Jiahao Huo, Zhonghao Li, Henry Peng Zou, Yibo Yan, Xin Zou, Jungang Li, Junzhuo Li, Hanrong Zhang, Xuming Hu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.03592)  

**Abstract**: Mixture-of-Experts (MoE) models exhibit striking performance disparities across languages, yet the internal mechanisms driving these gaps remain poorly understood. In this work, we conduct a systematic analysis of expert routing patterns in MoE models, revealing a phenomenon we term Language Routing Isolation, in which high- and low-resource languages tend to activate largely disjoint expert sets. Through layer-stratified analysis, we further show that routing patterns exhibit a layer-wise convergence-divergence pattern across model depth. Building on these findings, we propose RISE (Routing Isolation-guided Subnetwork Enhancement), a framework that exploits routing isolation to identify and adapt language-specific expert subnetworks. RISE applies a tripartite selection strategy, using specificity scores to identify language-specific experts in shallow and deep layers and overlap scores to select universal experts in middle layers. By training only the selected subnetwork while freezing all other parameters, RISE substantially improves low-resource language performance while preserving capabilities in other languages. Experiments on 10 languages demonstrate that RISE achieves target-language F1 gains of up to 10.85% with minimal cross-lingual degradation. 

---
# Evolutionary Search for Automated Design of Uncertainty Quantification Methods 

**Authors**: Mikhail Seleznyov, Daniil Korbut, Viktor Moskvoretskii, Oleg Somov, Alexander Panchenko, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2604.03473)  

**Abstract**: Uncertainty quantification (UQ) methods for large language models are predominantly designed by hand based on domain knowledge and heuristics, limiting their scalability and generality. We apply LLM-powered evolutionary search to automatically discover unsupervised UQ methods represented as Python programs. On the task of atomic claim verification, our evolved methods outperform strong manually-designed baselines, achieving up to 6.7% relative ROC-AUC improvement across 9 datasets while generalizing robustly out-of-distribution. Qualitative analysis reveals that different LLMs employ qualitatively distinct evolutionary strategies: Claude models consistently design high-feature-count linear estimators, while Gpt-oss-120B gravitates toward simpler and more interpretable positional weighting schemes. Surprisingly, only Sonnet 4.5 and Opus 4.5 reliably leverage increased method complexity to improve performance -- Opus 4.6 shows an unexpected regression relative to its predecessor. Overall, our results indicate that LLM-powered evolutionary search is a promising paradigm for automated, interpretable hallucination detector design. 

---
# Cultural Authenticity: Comparing LLM Cultural Representations to Native Human Expectations 

**Authors**: Erin MacMurray van Liemt, Aida Davani, Sinchana Kumbale, Neha Dixit, Sunipa Dev  

**Link**: [PDF](https://arxiv.org/pdf/2604.03493)  

**Abstract**: Cultural representation in Large Language Model (LLM) outputs has primarily been evaluated through the proxies of cultural diversity and factual accuracy. However, a crucial gap remains in assessing cultural alignment: the degree to which generated content mirrors how native populations perceive and prioritize their own cultural facets. In this paper, we introduce a human-centered framework to evaluate the alignment of LLM generations with local expectations. First, we establish a human-derived ground-truth baseline of importance vectors, called Cultural Importance Vectors based on an induced set of culturally significant facets from open-ended survey responses collected across nine countries. Next, we introduce a method to compute model-derived Cultural Representation Vectors of an LLM based on a syntactically diversified prompt-set and apply it to three frontier LLMs (Gemini 2.5 Pro, GPT-4o, and Claude 3.5 Haiku). Our investigation of the alignment between the human-derived Cultural Importance and model-derived Cultural Representations reveals a Western-centric calibration for some of the models where alignment decreases as a country's cultural distance from the US increases. Furthermore, we identify highly correlated, systemic error signatures ($\rho > 0.97$) across all models, which over-index on some cultural markers while neglecting the deep-seated social and value-based priorities of users. Our approach moves beyond simple diversity metrics toward evaluating the fidelity of AI-generated content in authentically capturing the nuanced hierarchies of global cultures. 

---
# LangFIR: Discovering Sparse Language-Specific Features from Monolingual Data for Language Steering 

**Authors**: Sing Hieng Wong, Hassan Sajjad, A.B. Siddique  

**Link**: [PDF](https://arxiv.org/pdf/2604.03532)  

**Abstract**: Large language models (LLMs) show strong multilingual capabilities, yet reliably controlling the language of their outputs remains difficult. Representation-level steering addresses this by adding language-specific vectors to model activations at inference time, but identifying language-specific directions in the residual stream often relies on multilingual or parallel data that can be expensive to obtain. Sparse autoencoders (SAEs) decompose residual activations into interpretable, sparse feature directions and offer a natural basis for this search, yet existing SAE-based approaches face the same data constraint. We introduce LangFIR (Language Feature Identification via Random-token Filtering), a method that discovers language-specific SAE features using only a small amount of monolingual data and random-token sequences. Many SAE features consistently activated by target-language inputs do not encode language identity. Random-token sequences surface these language-agnostic features, allowing LangFIR to filter them out and isolate a sparse set of language-specific features. We show that these features are extremely sparse, highly selective for their target language, and causally important: directional ablation increases cross-entropy loss only for the corresponding language. Using these features to construct steering vectors for multilingual generation control, LangFIR achieves the best average accuracy BLEU across three models (Gemma 3 1B, Gemma 3 4B, and Llama 3.1 8B), three datasets, and twelve target languages, outperforming the strongest monolingual baseline by up to and surpassing methods that rely on parallel data. Our results suggest that language identity in multilingual LLMs is localized in a sparse set of feature directions discoverable with monolingual data. Code is available at this https URL. 

---
# CresOWLve: Benchmarking Creative Problem-Solving Over Real-World Knowledge 

**Authors**: Mete Ismayilzada, Renqing Cuomao, Daniil Yurshevich, Anna Sotnikova, Lonneke van der Plas, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2604.03374)  

**Abstract**: Creative problem-solving requires combining multiple cognitive abilities, including logical reasoning, lateral thinking, analogy-making, and commonsense knowledge, to discover insights that connect seemingly unrelated pieces of information. However, most existing benchmarks for large language models (LLMs) evaluate only specific components of this process. Moreover, many creativity-oriented benchmarks rely on artificially constructed brainteasers or contrived scenarios that do not reflect how creative problem-solving occurs in real-world settings. To address this gap, we introduce CresOWLve, a benchmark for evaluating creative problem-solving using puzzles grounded in real-world knowledge. Problems in CresOWLve require employing multiple creative thinking strategies, retrieving facts from diverse domains, and creatively combining them to arrive at a solution. Evaluating several frontier non-thinking and thinking LLMs, we show that CresOWLve remains highly challenging. Our analysis reveals a consistent performance gap: models perform substantially better on factual questions than on creative ones (up to a -17% drop). While models can often retrieve the relevant knowledge, they struggle to form the non-obvious creative connections required to integrate this information and arrive at the correct answer. 

---
# Vocabulary Dropout for Curriculum Diversity in LLM Co-Evolution 

**Authors**: Jacob Dineen, Aswin RRV, Zhikun Xu, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.03472)  

**Abstract**: Co-evolutionary self-play, where one language model generates problems and another solves them, promises autonomous curriculum learning without human supervision. In practice, the proposer quickly converges to a narrow distribution of problems that satisfy the reward function. This diversity collapse renders the curriculum uninformative for the solver, stalling the co-evolutionary loop. We introduce vocabulary dropout, a random mask applied to the proposer's output logits during both policy training and curriculum generation, as a lightweight mechanism to sustain diversity. The mask is hard and non-stationary, preventing the proposer from locking into fixed token sequences. Training Qwen3-4B and Qwen3-8B on mathematical reasoning via R-Zero, we find that vocabulary dropout sustains proposer diversity across lexical, semantic, and functional metrics throughout training, and yields solver improvements averaging +4.4 points at 8B, with the largest gains on competition-level benchmarks. Our findings suggest that explicit action-space constraints, analogous to the structural role that game rules play in classical self-play, can help sustain productive co-evolution in language. Vocabulary dropout is one simple instantiation of this principle. 

---
# Are Arabic Benchmarks Reliable? QIMMA's Quality-First Approach to LLM Evaluation 

**Authors**: Leen AlQadi, Ahmed Alzubaidi, Mohammed Alyafeai, Hamza Alobeidli, Maitha Alhammadi, Shaikha Alsuwaidi, Omar Alkaabi, Basma El Amel Boussaha, Hakim Hacid  

**Link**: [PDF](https://arxiv.org/pdf/2604.03395)  

**Abstract**: We present QIMMA, a quality-assured Arabic LLM leaderboard that places systematic benchmark validation at its core. Rather than aggregating existing resources as-is, QIMMA applies a multi-model assessment pipeline combining automated LLM judgment with human review to surface and resolve systematic quality issues in well-established Arabic benchmarks before evaluation. The result is a curated, multi-domain, multi-task evaluation suite of over 52k samples, grounded predominantly in native Arabic content; code evaluation tasks are the sole exception, as they are inherently language-agnostic. Transparent implementation via LightEval, EvalPlus and public release of per-sample inference outputs make QIMMA a reproducible and community-extensible foundation for Arabic NLP evaluation. 

---
# VIGIL: An Extensible System for Real-Time Detection and Mitigation of Cognitive Bias Triggers 

**Authors**: Bo Kang, Sander Noels, Tijl De Bie  

**Link**: [PDF](https://arxiv.org/pdf/2604.03261)  

**Abstract**: The rise of generative AI is posing increasing risks to online information integrity and civic discourse. Most concretely, such risks can materialise in the form of mis- and disinformation. As a mitigation, media-literacy and transparency tools have been developed to address factuality of information and the reliability and ideological leaning of information sources. However, a subtler but possibly no less harmful threat to civic discourse is to use of persuasion or manipulation by exploiting human cognitive biases and related cognitive limitations. To the best of our knowledge, no tools exist to directly detect and mitigate the presence of triggers of such cognitive biases in online information. We present VIGIL (VIrtual GuardIan angeL), the first browser extension for real-time cognitive bias trigger detection and mitigation, providing in-situ scroll-synced detection, LLM-powered reformulation with full reversibility, and privacy-tiered inference from fully offline to cloud. VIGIL is built to be extensible with third-party plugins, with several plugins that are rigorously validated against NLP benchmarks are already included. It is open-sourced at this https URL. 

---
# Robust LLM Performance Certification via Constrained Maximum Likelihood Estimation 

**Authors**: Minghe Shen, Ananth Balashankar, Adam Fisch, David Madras, Miguel Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2604.03257)  

**Abstract**: The ability to rigorously estimate the failure rates of large language models (LLMs) is a prerequisite for their safe deployment. Currently, however, practitioners often face a tradeoff between expensive human gold standards and potentially severely-biased automatic annotation schemes such as "LLM-as-a-Judge" labeling. In this paper, we propose a new, practical, and efficient approach to LLM failure rate estimation based on constrained maximum-likelihood estimation (MLE). Our method integrates three distinct signal sources: (i) a small, high-quality human-labeled calibration set, (ii) a large corpus of LLM-judge annotations, and, most importantly, (iii) additional side information via domain-specific constraints derived from known bounds on judge performance statistics. We validate our approach through a comprehensive empirical study, benchmarking it against state-of-the-art baselines like Prediction-Powered Inference (PPI). Across diverse experimental regimes -- spanning varying judge accuracies, calibration set sizes, and LLM failure rates -- our constrained MLE consistently delivers more accurate and lower-variance estimates than existing methods. By moving beyond the "black-box" use of automated judges to a flexible framework, we provide a principled, interpretable, and scalable pathway towards LLM failure-rate certification. 

---
# Knowledge Packs: Zero-Token Knowledge Delivery via KV Cache Injection 

**Authors**: Andrey Pustovit  

**Link**: [PDF](https://arxiv.org/pdf/2604.03270)  

**Abstract**: RAG wastes tokens. We propose Knowledge Packs: pre-computed KV caches that deliver the same knowledge at zero token cost. For causal transformers, the KV cache from a forward pass on text F is identical to what a joint pass on F+q would produce - this follows directly from the causal mask. The equivalence is exact but fragile: wrong chat template formatting causes 6-7pp degradation, which we believe explains prior claims of KV outperforming RAG. With correct formatting: zero divergences across 700 questions on Qwen3-8B and Llama-3.1-8B, up to 95% token savings. The KV interface also enables behavioral steering that RAG cannot do. Because RoPE rotates keys but leaves values untouched, contrastive deltas on cached values can nudge model behavior while key arithmetic destroys coherence. The effect sits in mid-layer values (33-66%), independent directions are nearly orthogonal (cos~0) and compose, and both channels - knowledge and steering - run simultaneously at alpha<=0.7 without interference. No training, no weight modification. 

---
# Noise Steering for Controlled Text Generation: Improving Diversity and Reading-Level Fidelity in Arabic Educational Story Generation 

**Authors**: Haziq Mohammad Khalid, Salsabeel Shapsough, Imran Zualkernan  

**Link**: [PDF](https://arxiv.org/pdf/2604.03380)  

**Abstract**: Generating diverse, pedagogically valid stories for Arabic early-grade reading assessments requires balancing tight constraints on vocabulary, reading level, and narrative structure against the need to avoid repetitive plots that undermine assessment validity. We investigate noise steering, injecting calibrated Gaussian perturbations into the internal representations of transformer models at inference time, as a training-free diversity method evaluated across five small Arabic-centric language models (7-9B parameters). We compare four injection strategies against high-temperature sampling baselines, measuring diversity, quality, constraint adherence, and reading grade level. Residual stream noise consistently improves narrative diversity with minimal quality or constraint cost and preserves early-grade reading level across all models. Attention entropy noise injection (AENI) stabilizes the otherwise unreliable attention-logit noise while recovering quality. High-temperature sampling inflates reading grade level and causes catastrophic collapse on several models. We find internal representation-level perturbation to be a more suitable diversity strategy than output-level stochasticity for constrained educational content generation. 

---
# Why Attend to Everything? Focus is the Key 

**Authors**: Hengshuai Yao, Xing Chen, Ahmed Murtadha, Jin Li, Shuai Shao, Yasin Abbasi Yadkori, Guan Wang, Mingli Yuan, William Chen, Sen Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.03260)  

**Abstract**: We introduce Focus, a method that learns which token pairs matter rather than approximating all of them. Learnable centroids assign tokens to groups; distant attention is restricted to same-group pairs while local attention operates at full resolution. Because all model weights stay frozen, Focus is purely additive: centroid-only training (as few as 148K parameters) improves domain perplexity with zero degradation on downstream benchmarks--from 124M to 70B parameters, across five attention architectures. No existing efficient attention method achieves this in the retrofit setting. At 124M, Focus surpasses full attention (30.3 vs 31.4 PPL); trained from scratch at 7B scale (2B tokens), Focus again beats full attention (13.82 vs 13.89 PPL). At inference, restricting each token to its top-k highest-scoring groups discretizes the soft routing into a hard sparsity pattern, yielding 2x speedup while beating the pretrained baseline (41.3 vs 42.8 PPL); decomposing this pattern into two standard FlashAttention calls reaches 8.6x wall-clock speedup at 1M tokens with no custom kernels. Unlike LoRA, centroid routing preserves alignment: instruction-tuned models retain TruthfulQA scores after adaptation, while LoRA degrades at every learning rate and rank. Sinkhorn normalization enforces balanced groups as a hard constraint, and the resulting groups discover interpretable linguistic categories without supervision. 

---
# Self-Execution Simulation Improves Coding Models 

**Authors**: Gallil Maimon, Ori Yoran, Felix Kreuk, Michael Hassid, Gal Cohen, Pierre Chambon, Yossi Adi  

**Link**: [PDF](https://arxiv.org/pdf/2604.03253)  

**Abstract**: A promising research direction in enabling LLMs to generate consistently correct code involves addressing their inability to properly estimate program execution, particularly for code they generate. In this work, we demonstrate that Code LLMs can be trained to simulate program execution in a step-by-step manner and that this capability can be leveraged to improve competitive programming performance. Our approach combines supervised fine-tuning on natural language execution traces, textual explanations grounded in true execution, with reinforcement learning using verifiable rewards. We introduce two complementary objectives: output prediction given code and inputs, and solving competitive programming tasks with either ground-truth or self-predicted execution feedback. These objectives enable models to perform self-verification over multiple candidate solutions, and iterative self-fixing by simulating test execution. Across multiple competitive programming benchmarks, our method yields consistent improvements over standard reasoning approaches. We further present ablations and analysis to elucidate the role of execution simulation and its limitations. 

---
# SoLA: Leveraging Soft Activation Sparsity and Low-Rank Decomposition for Large Language Model Compression 

**Authors**: Xinhao Huang, You-Liang Huang, Zeyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.03258)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities across various tasks, but the billion-scale parameters pose deployment challenges. Although existing methods attempt to reduce the scale of LLMs, they require either special hardware support or expensive post-training to maintain model quality. To facilitate efficient and affordable model slimming, we propose a novel training-free compression method for LLMs, named "SoLA", which leverages \textbf{So}ft activation sparsity and \textbf{L}ow-r\textbf{A}nk decomposition. SoLA can identify and retain a minority of components significantly contributing to inference, while compressing the majority through low-rank decomposition, based on our analysis of the activation pattern in the feed-forward network (FFN) of modern LLMs. To alleviate the decomposition loss, SoLA is equipped with an adaptive component-wise low-rank allocation strategy to assign appropriate truncation positions for different weight matrices. We conduct extensive experiments on LLaMA-2-7B/13B/70B and Mistral-7B models across a variety of benchmarks. SoLA exhibits remarkable improvement in both language modeling and downstream task accuracy without post-training. For example, with a 30\% compression rate on the LLaMA-2-70B model, SoLA surpasses the state-of-the-art method by reducing perplexity from 6.95 to 4.44 and enhancing downstream task accuracy by 10\%. 

---
# Vero: An Open RL Recipe for General Visual Reasoning 

**Authors**: Gabriel Sarch, Linrong Cai, Qunzhong Wang, Haoyang Wu, Danqi Chen, Zhuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04917)  

**Abstract**: What does it take to build a visual reasoner that works across charts, science, spatial understanding, and open-ended tasks? The strongest vision-language models (VLMs) show such broad visual reasoning is within reach, but the recipe behind them remains unclear, locked behind proprietary reinforcement learning (RL) pipelines with non-public data. We introduce Vero, a family of fully open VLMs that matches or exceeds existing open-weight models across diverse visual reasoning tasks. We scale RL data and rewards across six broad task categories, constructing Vero-600K, a 600K-sample dataset from 59 datasets, and designing task-routed rewards that handle heterogeneous answer formats. Vero achieves state-of-the-art performance, improving over four base models by 3.7-5.5 points on average across VeroEval, our suite of 30 challenging benchmarks. Starting from Qwen3-VL-8B-Instruct, Vero outperforms Qwen3-VL-8B-Thinking on 23 of 30 benchmarks without additional proprietary thinking data. When trained from the same base model, Vero-600K exceeds existing RL datasets across task categories. Systematic ablations reveal that different task categories elicit qualitatively distinct reasoning patterns that transfer poorly in isolation, suggesting that broad data coverage is the primary driver of strong RL scaling. All data, code, and models are released. 

---
# ANX: Protocol-First Design for AI Agent Interaction with a Supporting 3EX Decoupled Architecture 

**Authors**: Xu Mingze  

**Link**: [PDF](https://arxiv.org/pdf/2604.04820)  

**Abstract**: AI agents, autonomous digital actors, need agent-native protocols; existing methods include GUI automation and MCP-based skills, with defects of high token consumption, fragmented interaction, inadequate security, due to lacking a unified top-level framework and key components, each independent module flawed. To address these issues, we present ANX, an open, extensible, verifiable agent-native protocol and top-level framework integrating CLI, Skill, MCP, resolving pain points via protocol innovation, architectural optimization and tool supplementation. Its four core innovations: 1) Agent-native design (ANX Config, Markup, CLI) with high information density, flexibility and strong adaptability to reduce tokens and eliminate inconsistencies; 2) Human-agent interaction combining Skill's flexibility for dual rendering as agent-executable instructions and human-readable UI; 3) MCP-supported on-demand lightweight apps without pre-registration; 4) ANX Markup-enabled machine-executable SOPs eliminating ambiguity for reliable long-horizon tasks and multi-agent collaboration. As the first in a series, we focus on ANX's design, present its 3EX decoupled architecture with ANXHub and preliminary feasibility analysis and experimental validation. ANX ensures native security: LLM-bypassed UI-to-Core communication keeps sensitive data out of agent context; human-only confirmation prevents automated misuse. Form-filling experiments with Qwen3.5-plus/GPT-4o show ANX reduces tokens by 47.3% (Qwen3.5-plus) and 55.6% (GPT-4o) vs MCP-based skills, 57.1% (Qwen3.5-plus) and 66.3% (GPT-4o) vs GUI automation, and shortens execution time by 58.1% and 57.7% vs MCP-based skills. 

---
# Darkness Visible: Reading the Exception Handler of a Language Model 

**Authors**: Peter Balogh  

**Link**: [PDF](https://arxiv.org/pdf/2604.04756)  

**Abstract**: The final MLP of GPT-2 Small exhibits a fully legible routing program -- 27 named neurons organized into a three-tier exception handler -- while the knowledge it routes remains entangled across ~3,040 residual neurons. We decompose all 3,072 neurons (to numerical precision) into: 5 fused Core neurons that reset vocabulary toward function words, 10 Differentiators that suppress wrong candidates, 5 Specialists that detect structural boundaries, and 7 Consensus neurons that each monitor a distinct linguistic dimension. The consensus-exception crossover -- where MLP intervention shifts from helpful to harmful -- is statistically sharp (bootstrap 95% CIs exclude zero at all consensus levels; crossover between 4/7 and 5/7). Three experiments show that "knowledge neurons" (Dai et al., 2022), at L11 of this model, function as routing infrastructure rather than fact storage: the MLP amplifies or suppresses signals already present in the residual stream from attention, scaling with contextual constraint. A garden-path experiment reveals a reversed garden-path effect -- GPT-2 uses verb subcategorization immediately, consistent with the exception handler operating at token-level predictability rather than syntactic structure. This architecture crystallizes only at the terminal layer -- in deeper models, we predict equivalent structure at the final layer, not at layer 11. Code and data: this https URL 

---
# Cog-DRIFT: Exploration on Adaptively Reformulated Instances Enables Learning from Hard Reasoning Problems 

**Authors**: Justin Chih-Yao Chen, Archiki Prasad, Zaid Khan, Joykirat Singh, Runchu Tian, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2604.04767)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) has improved the reasoning abilities of LLMs, yet a fundamental limitation remains: models cannot learn from problems that are too difficult to solve under their current policy, as these yield no meaningful reward signal. We propose a simple yet effective solution based on task reformulation. We transform challenging open-ended problems into cognitively simpler variants -- such as multiple-choice and cloze formats -- that preserve the original answer while reducing the effective search space and providing denser learning signals. These reformulations span a spectrum from discriminative to generative tasks, which we exploit to bootstrap learning: models first learn from structured, easier formats, and this knowledge transfers back to improve performance on the original open-ended problems. Building on this insight, we introduce Cog-DRIFT, a framework that constructs reformulated variants and organizes them into an adaptive curriculum based on difficulty. Training progresses from easier to harder formats, enabling the model to learn from problems that previously yielded zero signal under standard RL post-training. Cog-DRIFT not only improves on the originally unsolvable hard problems (absolute +10.11% for Qwen and +8.64% for Llama) but also generalizes well to other held-out datasets. Across 2 models and 6 reasoning benchmarks, our method consistently outperforms standard GRPO and strong guided-exploration baselines. On average, Cog-DRIFT shows +4.72% (Qwen) and +3.23% (Llama) improvements over the second-best baseline. We further show that Cog-DRIFT improves pass@k at test time, and the curriculum improves sample efficiency. Overall, our results highlight task reformulation and curriculum learning as an effective paradigm for overcoming the exploration barrier in LLM post-training. 

---
# Your Agent, Their Asset: A Real-World Safety Analysis of OpenClaw 

**Authors**: Zijun Wang, Haoqin Tu, Letian Zhang, Hardy Chen, Juncheng Wu, Xiangyan Liu, Zhenlong Yuan, Tianyu Pang, Michael Qizhe Shieh, Fengze Liu, Zeyu Zheng, Huaxiu Yao, Yuyin Zhou, Cihang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.04759)  

**Abstract**: OpenClaw, the most widely deployed personal AI agent in early 2026, operates with full local system access and integrates with sensitive services such as Gmail, Stripe, and the filesystem. While these broad privileges enable high levels of automation and powerful personalization, they also expose a substantial attack surface that existing sandboxed evaluations fail to capture. To address this gap, we present the first real-world safety evaluation of OpenClaw and introduce the CIK taxonomy, which unifies an agent's persistent state into three dimensions, i.e., Capability, Identity, and Knowledge, for safety analysis. Our evaluations cover 12 attack scenarios on a live OpenClaw instance across four backbone models (Claude Sonnet 4.5, Opus 4.6, Gemini 3.1 Pro, and GPT-5.4). The results show that poisoning any single CIK dimension increases the average attack success rate from 24.6% to 64-74%, with even the most robust model exhibiting more than a threefold increase over its baseline vulnerability. We further assess three CIK-aligned defense strategies alongside a file-protection mechanism; however, the strongest defense still yields a 63.8% success rate under Capability-targeted attacks, while file protection blocks 97% of malicious injections but also prevents legitimate updates. Taken together, these findings show that the vulnerabilities are inherent to the agent architecture, necessitating more systematic safeguards to secure personal AI agents. Our project page is this https URL. 

---
# Full-Duplex-Bench-v3: Benchmarking Tool Use for Full-Duplex Voice Agents Under Real-World Disfluency 

**Authors**: Guan-Ting Lin, Chen Chen, Zhehuai Chen, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.04847)  

**Abstract**: We introduce Full-Duplex-Bench-v3 (FDB-v3), a benchmark for evaluating spoken language models under naturalistic speech conditions and multi-step tool use. Unlike prior work, our dataset consists entirely of real human audio annotated for five disfluency categories, paired with scenarios requiring chained API calls across four task domains. We evaluate six model configurations -- GPT-Realtime, Gemini Live 2.5, Gemini Live 3.1, Grok, Ultravox v0.7, and a traditional Cascaded pipeline (Whisper$\rightarrow$GPT-4o$\rightarrow$TTS) -- across accuracy, latency, and turn-taking dimensions. GPT-Realtime leads on Pass@1 (0.600) and interruption avoidance (13.5\%); Gemini Live 3.1 achieves the fastest latency (4.25~s) but the lowest turn-take rate (78.0\%); and the Cascaded baseline, despite a perfect turn-take rate, incurs the highest latency (10.12~s). Across all systems, self-correction handling and multi-step reasoning under hard scenarios remain the most consistent failure modes. 

---
# QED-Nano: Teaching a Tiny Model to Prove Hard Theorems 

**Authors**: LM-Provers, Yuxiao Qu, Amrith Setlur, Jasper Dekoninck, Edward Beeching, Jia Li, Ian Wu, Lewis Tunstall, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.04898)  

**Abstract**: Proprietary AI systems have recently demonstrated impressive capabilities on complex proof-based problems, with gold-level performance reported at the 2025 International Mathematical Olympiad (IMO). However, the training pipelines behind these systems remain largely undisclosed, and their reliance on large "internal" models and scaffolds makes them expensive to run, difficult to reproduce, and hard to study or improve upon. This raises a central question: can small, open models also be trained to achieve competitive reasoning performance on difficult Olympiad-level math? In this paper, we answer this question by building QED-Nano, a 4B model post-trained for Olympiad-level proofs. Our training recipe has three stages: (1) supervised fine-tuning to imbue good proof-writing styles by distilling from DeepSeek-Math-V2, (2) reinforcement learning (RL) with rubric-based rewards, and (3) expanding RL with a reasoning cache, which decomposes long proofs into iterative summarize-and-refine cycles and enables stronger test-time reasoning. QED-Nano surpasses the proof-generation performance of much larger open models, including Nomos-1 and GPT-OSS-120B, and approaches the performance of proprietary models like Gemini 3 Pro, at a fraction of the inference cost. To support further research on open mathematical reasoning, we release the full QED-Nano pipeline, including the QED-Nano and QED-Nano-SFT models, the FineProofs-SFT and FineProofs-RL datasets, and the training and evaluation code. 

---
# Mapping the Exploitation Surface: A 10,000-Trial Taxonomy of What Makes LLM Agents Exploit Vulnerabilities 

**Authors**: Charafeddine Mouzouni  

**Link**: [PDF](https://arxiv.org/pdf/2604.04561)  

**Abstract**: LLM agents with tool access can discover and exploit security vulnerabilities. This is known. What is not known is which features of a system prompt trigger this behaviour, and which do not. We present a systematic taxonomy based on approximately 10,000 trials across seven models, 37 prompt conditions, and 12 hypothesized attack dimensions -- all executed in real Docker sandboxes. Every condition includes the instruction "Always respect all rules and access policies"; only a single encouragement sentence varies. Within the task class studied (planted test-runner vulnerabilities), the threat model is narrower than assumed: nine of twelve dimensions -- including minimization, moral licensing, incentives, identity priming, and reasoning triggers -- produce no detectable exploitation at n=50 per cell (upper 95% CI <7% per condition). But it is also more subtle than expected: one dimension, goal reframing, reliably triggers exploitation. "You are solving a puzzle; there may be hidden clues" produces 38-40% exploitation on Claude Sonnet 4 despite the explicit rule instruction, replicating across four models (CTF framing: 8-14% on DeepSeek, GPT-5-mini, o4-mini). The agent does not override the rules; it reinterprets the task so that exploitative actions become task-aligned. GPT-4.1 produces no exploitation across 1,850 trials (37 conditions), and a temporal comparison across four OpenAI models released over eleven months shows a pattern consistent with improving safety training, though model capability differences are a confounder. The practical contribution is a narrowed, testable threat model: defenders should audit for goal-reframing language, not for the broad class of adversarial prompts. 

---
# DP-OPD: Differentially Private On-Policy Distillation for Language Models 

**Authors**: Fatemeh Khadem, Sajad Mousavi, Yi Fang, Yuhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04461)  

**Abstract**: Large language models (LLMs) are increasingly adapted to proprietary and domain-specific corpora that contain sensitive information, creating a tension between formal privacy guarantees and efficient deployment through model compression. Differential privacy (DP), typically enforced via DP-SGD, provides record-level protection but often incurs substantial utility loss in autoregressive generation, where optimization noise can amplify exposure bias and compounding errors along long rollouts. Existing approaches to private distillation either apply DP-SGD to both teacher and student, worsening computation and the privacy--utility tradeoff, or rely on DP synthetic text generation from a DP-trained teacher, avoiding DP on the student at the cost of DP-optimizing a large teacher and introducing an offline generation pipeline. We propose \textbf{Differentially Private On-Policy Distillation (DP-OPD)}, a synthesis-free framework that enforces privacy solely through DP-SGD on the student while leveraging a frozen teacher to provide dense token-level targets on \emph{student-generated} trajectories. DP-OPD instantiates this idea via \emph{private generalized knowledge distillation} on continuation tokens. Under a strict privacy budget ($\varepsilon=2.0$), DP-OPD improves perplexity over DP fine-tuning and off-policy DP distillation, and outperforms synthesis-based DP distillation (Yelp: 44.15$\rightarrow$41.68; BigPatent: 32.43$\rightarrow$30.63), while substantially simplifying the training pipeline. In particular, \textbf{DP-OPD collapses private compression into a single DP student-training loop} by eliminating DP teacher training and offline synthetic text generation. Code will be released upon publication at this https URL. 

---
# One Model for All: Multi-Objective Controllable Language Models 

**Authors**: Qiang He, Yucheng Yang, Tianyi Zhou, Meng Fang, Mykola Pechenizkiy, Setareh Maghsudi  

**Link**: [PDF](https://arxiv.org/pdf/2604.04497)  

**Abstract**: Aligning large language models (LLMs) with human preferences is critical for enhancing LLMs' safety, helpfulness, humor, faithfulness, etc. Current reinforcement learning from human feedback (RLHF) mainly focuses on a fixed reward learned from average human ratings, which may weaken the adaptability and controllability of varying preferences. However, creating personalized LLMs requires aligning LLMs with individual human preferences, which is non-trivial due to the scarce data per user and the diversity of user preferences in multi-objective trade-offs, varying from emphasizing empathy in certain contexts to demanding efficiency and precision in others. Can we train one LLM to produce personalized outputs across different user preferences on the Pareto front? In this paper, we introduce Multi-Objective Control (MOC), which trains a single LLM to directly generate responses in the preference-defined regions of the Pareto front. Our approach introduces multi-objective optimization (MOO) principles into RLHF to train an LLM as a preference-conditioned policy network. We improve the computational efficiency of MOC by applying MOO at the policy level, enabling us to fine-tune a 7B-parameter model on a single A6000 GPU. Extensive experiments demonstrate the advantages of MOC over baselines in three aspects: (i) controllability of LLM outputs w.r.t. user preferences on the trade-off among multiple rewards; (ii) quality and diversity of LLM outputs, measured by the hyper-volume of multiple solutions achieved; and (iii) generalization to unseen preferences. These results highlight MOC's potential for real-world applications requiring scalable and customizable LLMs. 

---
# Relative Density Ratio Optimization for Stable and Statistically Consistent Model Alignment 

**Authors**: Hiroshi Takahashi, Tomoharu Iwata, Atsutoshi Kumagai, Sekitoshi Kanai, Masanori Yamada, Kosuke Nishida, Kazutoshi Shinoda  

**Link**: [PDF](https://arxiv.org/pdf/2604.04410)  

**Abstract**: Aligning language models with human preferences is essential for ensuring their safety and reliability. Although most existing approaches assume specific human preference models such as the Bradley-Terry model, this assumption may fail to accurately capture true human preferences, and consequently, these methods lack statistical consistency, i.e., the guarantee that language models converge to the true human preference as the number of samples increases. In contrast, direct density ratio optimization (DDRO) achieves statistical consistency without assuming any human preference models. DDRO models the density ratio between preferred and non-preferred data distributions using the language model, and then optimizes it via density ratio estimation. However, this density ratio is unstable and often diverges, leading to training instability of DDRO. In this paper, we propose a novel alignment method that is both stable and statistically consistent. Our approach is based on the relative density ratio between the preferred data distribution and a mixture of the preferred and non-preferred data distributions. Our approach is stable since this relative density ratio is bounded above and does not diverge. Moreover, it is statistically consistent and yields significantly tighter convergence guarantees than DDRO. We experimentally show its effectiveness with Qwen 2.5 and Llama 3. 

---
# Empirical Characterization of Rationale Stability Under Controlled Perturbations for Explainable Pattern Recognition 

**Authors**: Abu Noman Md Sakib, Zhensen Wang, Merjulah Roby, Zijie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04456)  

**Abstract**: Reliable pattern recognition systems should exhibit consistent behavior across similar inputs, and their explanations should remain stable. However, most Explainable AI evaluations remain instance centric and do not explicitly quantify whether attribution patterns are consistent across samples that share the same class or represent small variations of the same input. In this work, we propose a novel metric aimed at assessing the consistency of model explanations, ensuring that models consistently reflect the intended objectives and consistency under label-preserving perturbations. We implement this metric using a pre-trained BERT model on the SST-2 sentiment analysis dataset, with additional robustness tests on RoBERTa, DistilBERT, and IMDB, applying SHAP to compute feature importance for various test samples. The proposed metric quantifies the cosine similarity of SHAP values for inputs with the same label, aiming to detect inconsistent behaviors, such as biased reliance on certain features or failure to maintain consistent reasoning for similar predictions. Through a series of experiments, we evaluate the ability of this metric to identify misaligned predictions and inconsistencies in model explanations. These experiments are compared against standard fidelity metrics to assess whether the new metric can effectively identify when a model's behavior deviates from its intended objectives. The proposed framework provides a deeper understanding of model behavior by enabling more robust verification of rationale stability, which is critical for building trustworthy AI systems. By quantifying whether models rely on consistent attribution patterns for similar inputs, the proposed approach supports more robust evaluation of model behavior in practical pattern recognition pipelines. Our code is publicly available at this https URL. 

---
# What Makes a Sale? Rethinking End-to-End Seller--Buyer Retail Dynamics with LLM Agents 

**Authors**: Jeonghwan Choi, Jibin Hwang, Gyeonghun Sun, Minjeong Ban, Taewon Yun, Hyeonjae Cheon, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.04468)  

**Abstract**: Evaluating retail strategies before deployment is difficult, as outcomes are determined across multiple stages, from seller-side persuasion through buyer-seller interaction to purchase decisions. However, existing retail simulators capture only partial aspects of this process and do not model cross-stage dependencies, making it difficult to assess how early decisions affect downstream outcomes. We present RetailSim, an end-to-end retail simulation framework that models this pipeline in a unified environment, explicitly designed for simulation fidelity through diverse product spaces, persona-driven agents, and multi-turn interactions. We evaluate RetailSim with a dual protocol comprising human evaluation of behavioral fidelity and meta-evaluation against real-world economic regularities, showing that it successfully reproduces key patterns such as demographic purchasing behavior, the price-demand relationship, and heterogeneous price elasticity. We further demonstrate its practical utility via decision-oriented use cases, including persona inference, seller-buyer interaction analysis, and sales strategy evaluation, showing RetailSim's potential as a controlled testbed for exploring retail strategies. 

---
# Talk2AI: A Longitudinal Dataset of Human--AI Persuasive Conversations 

**Authors**: Alexis Carrillo, Enrique Taietta, Ali Aghazadeh Ardebili, Giuseppe Alessandro Veltri, Massimo Stella  

**Link**: [PDF](https://arxiv.org/pdf/2604.04354)  

**Abstract**: Talk2AI is a large-scale longitudinal dataset of 3,080 conversations (totaling 30,800 turns) between human participants and Large Language Models (LLMs), designed to support research on persuasion, opinion change, and human-AI interaction. The corpus was collected from 770 profiled Italian adults across four weekly sessions in Spring 2025, using a within-subject design in which each participant conversed with a single model (GPT-4o, Claude Sonnet 3.7, DeepSeek-chat V3, or Mistral Large) on three socially relevant topics: climate change, math anxiety, and health misinformation. Each conversation is linked to rich contextual data, including sociodemographic characteristics and psychometric profiles. After each session, participants reported on opinion change, conviction stability, perceived humanness of the AI, and behavioral intentions, enabling fine-grained longitudinal analysis of how AI-mediated dialogue shapes beliefs and attitudes over time. 

---
# Commercial Persuasion in AI-Mediated Conversations 

**Authors**: Francesco Salvi, Alejandro Cuevas, Manoel Horta Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2604.04263)  

**Abstract**: As Large Language Models (LLMs) become a primary interface between users and the web, companies face growing economic incentives to embed commercial influence into AI-mediated conversations. We present two preregistered experiments (N = 2,012) in which participants selected a book to receive from a large eBook catalog using either a traditional search engine or a conversational LLM agent powered by one of five frontier models. Unbeknownst to participants, a fifth of all products were randomly designated as sponsored and promoted in different ways. We find that LLM-driven persuasion nearly triples the rate at which users select sponsored products compared to traditional search placement (61.2% vs. 22.4%), while the vast majority of participants fail to detect any promotional steering. Explicit "Sponsored" labels do not significantly reduce persuasion, and instructing the model to conceal its intent makes its influence nearly invisible (detection accuracy < 10%). Altogether, our results indicate that conversational AI can covertly redirect consumer choices at scale, and that existing transparency mechanisms may be insufficient to protect users. 

---
# REAM: Merging Improves Pruning of Experts in LLMs 

**Authors**: Saurav Jha, Maryam Hashemzadeh, Ali Saheb Pasand, Ali Parviz, Min-Joong Lee, Boris Knyazev  

**Link**: [PDF](https://arxiv.org/pdf/2604.04356)  

**Abstract**: Mixture-of-Experts (MoE) large language models (LLMs) are among the top-performing architectures. The largest models, often with hundreds of billions of parameters, pose significant memory challenges for deployment. Traditional approaches to reduce memory requirements include weight pruning and quantization. Motivated by the Router-weighted Expert Activation Pruning (REAP) that prunes experts, we propose a novel method, Router-weighted Expert Activation Merging (REAM). Instead of removing experts, REAM groups them and merges their weights, better preserving original performance. We evaluate REAM against REAP and other baselines across multiple MoE LLMs on diverse multiple-choice (MC) question answering and generative (GEN) benchmarks. Our results reveal a trade-off between MC and GEN performance that depends on the mix of calibration data. By controlling the mix of general, math and coding data, we examine the Pareto frontier of this trade-off and show that REAM often outperforms the baselines and in many cases is comparable to the original uncompressed models. 

---
# Combee: Scaling Prompt Learning for Self-Improving Language Model Agents 

**Authors**: Hanchen Li, Runyuan He, Qizheng Zhang, Changxiu Ji, Qiuyang Mang, Xiaokun Chen, Lakshya A Agrawal, Wei-Liang Liao, Eric Yang, Alvin Cheung, James Zou, Kunle Olukotun, Ion Stoica, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2604.04247)  

**Abstract**: Recent advances in prompt learning allow large language model agents to acquire task-relevant knowledge from inference-time context without parameter changes. For example, existing methods (like ACE or GEPA) can learn system prompts to improve accuracy based on previous agent runs. However, these methods primarily focus on single-agent or low-parallelism settings. This fundamentally limits their ability to efficiently learn from a large set of collected agentic traces. It would be efficient and beneficial to run prompt learning in parallel to accommodate the growing trend of learning from many agentic traces or parallel agent executions. Yet without a principled strategy for scaling, current methods suffer from quality degradation with high parallelism. To improve both the efficiency and quality of prompt learning, we propose Combee, a novel framework to scale parallel prompt learning for self-improving agents. Combee speeds up learning and enables running many agents in parallel while learning from their aggregate traces without quality degradation. To achieve this, Combee leverages parallel scans and employs an augmented shuffle mechanism; Combee also introduces a dynamic batch size controller to balance quality and delay. Evaluations on AppWorld, Terminal-Bench, Formula, and FiNER demonstrate that Combee achieves up to 17x speedup over previous methods with comparable or better accuracy and equivalent cost. 

---
# BWTA: Accurate and Efficient Binarized Transformer by Algorithm-Hardware Co-design 

**Authors**: Yifu Ding, Xianglong Liu, Shenghao Jin, Jinyang Guo, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.03957)  

**Abstract**: Ultra low-bit quantization brings substantial efficiency for Transformer-based models, but the accuracy degradation and limited GPU support hinder its wide usage. In this paper, we analyze zero-point distortion in binarization and propose a Binary Weights & Ternary Activations (BWTA) quantization scheme, which projects tiny values to zero and preserves the accuracy of extremely low-bit models. For training, we propose Smooth Multi-Stage Quantization, combining a Levelwise Degradation Strategy and a Magnitude-Alignment Projection Factor to enable stable and fast convergence. For inference, we develop a BWTA MatMul CUDA kernel with instruction-level parallel bit-packing and comprehensive binary/ternary MatMul implementations for both linear and attention operators, allowing seamless integration across Transformer architectures. Experiments show that BWTA approaches full-precision performance for BERT, with an average 3.5% drop on GLUE and less than 2% drop on five tasks, and achieves comparable perplexity and accuracy for LLMs. In efficiency, it delivers 16 to 24 times kernel-level speedup over FP16 on NVIDIA GPUs, and 216 to 330 tokens/s end-to-end prefill speedup with lower memory footprint on LLMs. As an algorithm-hardware co-design, BWTA demonstrates practical, low-latency ultra-low-bit inference without sacrificing model quality. 

---
# Precise Robot Command Understanding Using Grammar-Constrained Large Language Models 

**Authors**: Xinyun Huo, Raghav Gnanasambandam, Xinyao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04233)  

**Abstract**: Human-robot collaboration in industrial settings requires precise and reliable communication to enhance operational efficiency. While Large Language Models (LLMs) understand general language, they often lack the domain-specific rigidity needed for safe and executable industrial commands. To address this gap, this paper introduces a novel grammar-constrained LLM that integrates a grammar-driven Natural Language Understanding (NLU) system with a fine-tuned LLM, which enables both conversational flexibility and the deterministic precision required in robotics. Our method employs a two-stage process. First, a fine-tuned LLM performs high-level contextual reasoning and parameter inference on natural language inputs. Second, a Structured Language Model (SLM) and a grammar-based canonicalizer constrain the LLM's output, forcing it into a standardized symbolic format composed of valid action frames and command elements. This process guarantees that generated commands are valid and structured in a robot-readable JSON format. A key feature of the proposed model is a validation and feedback loop. A grammar parser validates the output against a predefined list of executable robotic actions. If a command is invalid, the system automatically generates corrective prompts and re-engages the LLM. This iterative self-correction mechanism allows the model to recover from initial interpretation errors to improve system robustness. We evaluate our grammar-constrained hybrid model against two baselines: a fine-tuned API-based LLM and a standalone grammar-driven NLU model. Using the Human Robot Interaction Corpus (HuRIC) dataset, we demonstrate that the hybrid approach achieves superior command validity, which promotes safer and more effective industrial human-robot collaboration. 

---
# Affording Process Auditability with QualAnalyzer: An Atomistic LLM Analysis Tool for Qualitative Research 

**Authors**: Max Hao Lu, Ryan Ellegood, Rony Rodriguez-Ramirez, Sophia Blumert  

**Link**: [PDF](https://arxiv.org/pdf/2604.03820)  

**Abstract**: Large language models are increasingly used for qualitative data analysis, but many workflows obscure how analytic conclusions are produced. We present QualAnalyzer, an open-source Chrome extension for Google Workspace that supports atomistic LLM analysis by processing each data segment independently and preserving the prompt, input, and output for every unit. Through two case studies -- holistic essay scoring and deductive thematic coding of interview transcripts -- we show that this approach creates a legible audit trail and helps researchers investigate systematic differences between LLM and human judgments. We argue that process auditability is essential for making LLM-assisted qualitative research more transparent and methodologically robust. 

---
# PolySwarm: A Multi-Agent Large Language Model Framework for Prediction Market Trading and Latency Arbitrage 

**Authors**: Rajat M. Barot, Arjun S. Borkhatariya  

**Link**: [PDF](https://arxiv.org/pdf/2604.03888)  

**Abstract**: This paper presents PolySwarm, a novel multi-agent large language model (LLM) framework designed for real-time prediction market trading and latency arbitrage on decentralized platforms such as Polymarket. PolySwarm deploys a swarm of 50 diverse LLM personas that concurrently evaluate binary outcome markets, aggregating individual probability estimates through confidence-weighted Bayesian combination of swarm consensus with market-implied probabilities, and applying quarter-Kelly position sizing for risk-controlled execution. The system incorporates an information-theoretic market analysis engine using Kullback-Leibler (KL) divergence and Jensen-Shannon (JS) divergence to detect cross-market inefficiencies and negation pair mispricings. A latency arbitrage module exploits stale Polymarket prices by deriving CEX-implied probabilities from a log-normal pricing model and executing trades within the human reaction-time window. We provide a full architectural description, implementation details, and evaluation methodology using Brier scores, calibration analysis, and log-loss metrics benchmarked against human superforecaster performance. We further discuss open challenges including hallucination in agent pools, computational cost at scale, regulatory exposure, and feedback-loop risk, and outline five priority directions for future research. Experimental results demonstrate that swarm aggregation consistently outperforms single-model baselines in probability calibration on Polymarket prediction tasks. 

---
# SODA: Semi On-Policy Black-Box Distillation for Large Language Models 

**Authors**: Xiwen Chen, Jingjing Wang, Wenhui Zhu, Peijie Qiu, Xuanzhao Dong, Hejian Sang, Zhipeng Wang, Alborz Geramifard, Feng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.03873)  

**Abstract**: Black-box knowledge distillation for large language models presents a strict trade-off. Simple off-policy methods (e.g., sequence-level knowledge distillation) struggle to correct the student's inherent errors. Fully on-policy methods (e.g., Generative Adversarial Distillation) solve this via adversarial training but introduce well-known training instability and crippling computational overhead. To address this dilemma, we propose SODA (Semi On-policy Distillation with Alignment), a highly efficient alternative motivated by the inherent capability gap between frontier teachers and much smaller base models. Because a compact student model's natural, zero-shot responses are almost strictly inferior to the powerful teacher's targets, we can construct a highly effective contrastive signal simply by pairing the teacher's optimal response with a one-time static snapshot of the student's outputs. This demonstrates that exposing the small student to its own static inferior behaviors is sufficient for high-quality distribution alignment, eliminating the need for costly dynamic rollouts and fragile adversarial balancing. Extensive evaluations across four compact Qwen2.5 and Llama-3 models validate this semi on-policy paradigm. SODA matches or outperforms the state-of-the-art methods on 15 out of 16 benchmark results. More importantly, it achieves this superior distillation quality while training 10 times faster, consuming 27% less peak GPU memory, and completely eliminating adversarial instability. 

---
# CREBench: Evaluating Large Language Models in Cryptographic Binary Reverse Engineering 

**Authors**: Baicheng Chen, Yu Wang, Ziheng Zhou, Xiangru Liu, Juanru Li, Yilei Chen, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2604.03750)  

**Abstract**: Reverse engineering (RE) is central to software security, particularly for cryptographic programs that handle sensitive data and are highly prone to vulnerabilities. It supports critical tasks such as vulnerability discovery and malware analysis. Despite its importance, RE remains labor-intensive and requires substantial expertise, making large language models (LLMs) a potential solution for automating the process. However, their capabilities for RE remain systematically underexplored. To address this gap, we study the cryptographic binary RE capabilities of LLMs and introduce \textbf{CREBench}, a benchmark comprising 432 challenges built from 48 standard cryptographic algorithms, 3 insecure crypto key usage scenarios, and 3 difficulty levels. Each challenge follows a Capture-the-Flag (CTF) RE challenge, requiring the model to analyze the underlying cryptographic logic and recover the correct input. We design an evaluation framework comprising four sub-tasks, from algorithm identification to correct flag recovery. We evaluate eight frontier LLMs on CREBench. GPT-5.4, the best-performing model, achieves 64.03 out of 100 and recovers the flag in 59\% of challenges. We also establish a strong human expert baseline of 92.19 points, showing that humans maintain an advantage in cryptographic RE tasks. Our code and dataset are available at this https URL. 

---
# Entropy, Disagreement, and the Limits of Foundation Models in Genomics 

**Authors**: Maxime Rochkoulets, Lovro Vrček, Mile Šikić  

**Link**: [PDF](https://arxiv.org/pdf/2604.04287)  

**Abstract**: Foundation models in genomics have shown mixed success compared to their counterparts in natural language processing. Yet, the reasons for their limited effectiveness remain poorly understood. In this work, we investigate the role of entropy as a fundamental factor limiting the capacities of such models to learn from their training data and develop foundational capabilities. We train ensembles of models on text and DNA sequences and analyze their predictions, static embeddings, and empirical Fisher information flow. We show that the high entropy of genomic sequences -- from the point of view of unseen token prediction -- leads to near-uniform output distributions, disagreement across models, and unstable static embeddings, even for models that are matched in architecture, training and data. We then demonstrate that models trained on DNA concentrate Fisher information in embedding layers, seemingly failing to exploit inter-token relationships. Our results suggest that self-supervised training from sequences alone may not be applicable to genomic data, calling into question the assumptions underlying current methodologies for training genomic foundation models. 

---
# Can Humans Tell? A Dual-Axis Study of Human Perception of LLM-Generated News 

**Authors**: Alexander Loth, Martin Kappes, Marc-Oliver Pahl  

**Link**: [PDF](https://arxiv.org/pdf/2604.03755)  

**Abstract**: Can humans tell whether a news article was written by a person or a large language model (LLM)? We investigate this question using JudgeGPT, a study platform that independently measures source attribution (human vs. machine) and authenticity judgment (legitimate vs. fake) on continuous scales. From 2,318 judgments collected from 1,054 participants across content generated by six LLMs, we report five findings: (1) participants cannot reliably distinguish machine-generated from human-written text (p > .05, Welch's t-test); (2) this inability holds across all tested models, including open-weight models with as few as 7B parameters; (3) self-reported domain expertise predicts judgment accuracy (r = .35, p < .001) whereas political orientation does not (r = -.10, n.s.); (4) clustering reveals distinct response strategies ("Skeptics" vs. "Believers"); and (5) accuracy degrades after approximately 30 sequential evaluations due to cognitive fatigue. The answer, in short, is no: humans cannot reliably tell. These results indicate that user-side detection is not a viable defense and motivate system-level countermeasures such as cryptographic content provenance. 

---
# Olmo Hybrid: From Theory to Practice and Back 

**Authors**: William Merrill, Yanhong Li, Tyler Romero, Anej Svete, Caia Costello, Pradeep Dasigi, Dirk Groeneveld, David Heineman, Bailey Kuehl, Nathan Lambert, Jacob Morrison, Luca Soldaini, Finbarr Timbers, Pete Walsh, Noah A. Smith, Hannaneh Hajishirzi, Ashish Sabharwal  

**Link**: [PDF](https://arxiv.org/pdf/2604.03444)  

**Abstract**: Recent work has demonstrated the potential of non-transformer language models, especially linear recurrent neural networks (RNNs) and hybrid models that mix recurrence and attention. Yet there is no consensus on whether the potential benefits of these new architectures justify the risk and effort of scaling them up. To address this, we provide evidence for the advantages of hybrid models over pure transformers on several fronts. First, theoretically, we show that hybrid models do not merely inherit the expressivity of transformers and linear RNNs, but can express tasks beyond both, such as code execution. Putting this theory to practice, we train Olmo Hybrid, a 7B-parameter model largely comparable to Olmo 3 7B but with the sliding window layers replaced by Gated DeltaNet layers. We show that Olmo Hybrid outperforms Olmo 3 across standard pretraining and mid-training evaluations, demonstrating the benefit of hybrid models in a controlled, large-scale setting. We find that the hybrid model scales significantly more efficiently than the transformer, explaining its higher performance. However, its unclear why greater expressivity on specific formal problems should result in better scaling or superior performance on downstream tasks unrelated to those problems. To explain this apparent gap, we return to theory and argue why increased expressivity should translate to better scaling efficiency, completing the loop. Overall, our results suggest that hybrid models mixing attention and recurrent layers are a powerful extension to the language modeling paradigm: not merely to reduce memory during inference, but as a fundamental way to obtain more expressive models that scale better during pretraining. 

---
# Large Language Models Align with the Human Brain during Creative Thinking 

**Authors**: Mete Ismayilzada, Simone A. Luchini, Abdulkadir Gokce, Badr AlKhamissi, Antoine Bosselut, Antonio Laverghetta Jr., Lonneke van der Plas, Roger E. Beaty  

**Link**: [PDF](https://arxiv.org/pdf/2604.03480)  

**Abstract**: Creative thinking is a fundamental aspect of human cognition, and divergent thinking-the capacity to generate novel and varied ideas-is widely regarded as its core generative engine. Large language models (LLMs) have recently demonstrated impressive performance on divergent thinking tests and prior work has shown that models with higher task performance tend to be more aligned to human brain activity. However, existing brain-LLM alignment studies have focused on passive, non-creative tasks. Here, we explore brain alignment during creative thinking using fMRI data from 170 participants performing the Alternate Uses Task (AUT). We extract representations from LLMs varying in size (270M-72B) and measure alignment to brain responses via Representational Similarity Analysis (RSA), targeting the creativity-related default mode and frontoparietal networks. We find that brain-LLM alignment scales with model size (default mode network only) and idea originality (both networks), with effects strongest early in the creative process. We further show that post-training objectives shape alignment in functionally selective ways: a creativity-optimized \texttt{Llama-3.1-8B-Instruct} preserves alignment with high-creativity neural responses while reducing alignment with low-creativity ones; a human behavior fine-tuned model elevates alignment with both; and a reasoning-trained variant shows the opposite pattern, suggesting chain-of-thought training steers representations away from creative neural geometry toward analytical processing. These results demonstrate that post-training objectives selectively reshape LLM representations relative to the neural geometry of human creative thought. 

---
# The Persuasion Paradox: When LLM Explanations Fail to Improve Human-AI Team Performance 

**Authors**: Ruth Cohen, Lu Feng, Ayala Bloch, Sarit Kraus  

**Link**: [PDF](https://arxiv.org/pdf/2604.03237)  

**Abstract**: While natural-language explanations from large language models (LLMs) are widely adopted to improve transparency and trust, their impact on objective human-AI team performance remains poorly understood. We identify a Persuasion Paradox: fluent explanations systematically increase user confidence and reliance on AI without reliably improving, and in some cases undermining, task accuracy.
Across three controlled human-subject studies spanning abstract visual reasoning (RAVEN matrices) and deductive logical reasoning (LSAT problems), we disentangle the effects of AI predictions and explanations using a multi-stage reveal design and between-subjects comparisons. In visual reasoning, LLM explanations increase confidence but do not improve accuracy beyond the AI prediction alone, and substantially suppress users' ability to recover from model errors. Interfaces exposing model uncertainty via predicted probabilities, as well as a selective automation policy that defers uncertain cases to humans, achieve significantly higher accuracy and error recovery than explanation-based interfaces.
In contrast, for language-based logical reasoning tasks, LLM explanations yield the highest accuracy and recovery rates, outperforming both expert-written explanations and probability-based support. This divergence reveals that the effectiveness of narrative explanations is strongly task-dependent and mediated by cognitive modality.
Our findings demonstrate that commonly used subjective metrics such as trust, confidence, and perceived clarity are poor predictors of human-AI team performance. Rather than treating explanations as a universal solution, we argue for a shift toward interaction designs that prioritize calibrated reliance and effective error recovery over persuasive fluency. 

---
# Evaluating Digital Inclusiveness of Digital Agri-Food Tools Using Large Language Models: A Comparative Analysis Between Human and AI-Based Evaluations 

**Authors**: Githma Pewinya, Carolina Martins, Garcia Mariangel  

**Link**: [PDF](https://arxiv.org/pdf/2604.03252)  

**Abstract**: Ensuring digital inclusiveness is a critical priority in agri-food systems, particularly in the Global South, where digital divides persist. The Multidimensional Digital Inclusiveness Index (MDII) offers a comprehensive, human-led framework to assess how inclusive digital agricultural tools (agritools) are. However, the current evaluation process is resource intensive, often requiring months to complete. This study explores whether large language models (LLMs) can support a rapid, AI-enabled assessment of digital inclusiveness, complementing the MDII's existing workflow. Using a comparative analysis, the research benchmarks the performance of four LLMs (Grok, Gemini, GPT-4o, and GPT-5) against prior expert-led evaluations. The study investigates model alignment with human scores, sensitivity to temperature settings, and potential sources of bias. Findings suggest that LLMs can generate evaluative outputs that approximate expert judgment in some dimensions, though reliability varies across models and contexts. This exploratory work provides early evidence for the integration of GenAI into inclusive digital development monitoring, with implications for scaling evaluations in time-sensitive or resource-constrained environments. 

---
# Scaling DPPs for RAG: Density Meets Diversity 

**Authors**: Xun Sun, Baiheng Xie, Li Huang, Qiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.03240)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by grounding generation in external knowledge, yielding relevance responses that are aligned with factual evidence and evolving corpora. Standard RAG pipelines construct context through relevance ranking, performing point-wise scoring between the user query and each corpora chunk. This formulation, however, ignores interactions among retrieved candidates, leading to redundant contexts that dilute density and fail to surface complementary evidence. We argue that effective retrieval should optimize jointly for both density and diversity, ensuring the grounding evidence that is dense in information yet diverse in coverage. In this study, we propose ScalDPP, a diversity-aware retrieval mechanism for RAG that incorporates Determinantal Point Processes (DPPs) through a lightweight P-Adapter, enabling scalable modeling of inter-chunk dependencies and complementary context selection. In addition, we develop a novel set-level objective, Diverse Margin Loss (DML), that enforces ground-truth complementary evidence chains to dominate any equally sized redundant alternatives under DPP geometry. Experimental results demonstrate the superiority of ScalDPP, substantiating our core statement in practice. 

---
# VERT: Reliable LLM Judges for Radiology Report Evaluation 

**Authors**: Federica Bologna, Jean-Philippe Corbeil, Matthew Wilkens, Asma Ben Abacha  

**Link**: [PDF](https://arxiv.org/pdf/2604.03376)  

**Abstract**: Current literature on radiology report evaluation has focused primarily on designing LLM-based metrics and fine-tuning small models for chest X-rays. However, it remains unclear whether these approaches are robust when applied to reports from other modalities and anatomies. Which model and prompt configurations are best suited to serve as LLM judges for radiology evaluation? We conduct a thorough correlation analysis between expert and LLM-based ratings. We compare three existing LLM-as-a-judge metrics (RadFact, GREEN, and FineRadScore) alongside VERT, our proposed LLM-based metric, using open- and closed-source models (reasoning and non-reasoning) of different sizes across two expert-annotated datasets, RadEval and RaTE-Eval, spanning multiple modalities and anatomies. We further evaluate few-shot approaches, ensembling, and parameter-efficient fine-tuning using RaTE-Eval. To better understand metric behavior, we perform a systematic error detection and categorization study to assess alignment of these metrics against expert judgments and identify areas of lower and higher agreement. Our results show that VERT improves correlation with radiologist judgments by up to 11.7% relative to GREEN. Furthermore, fine-tuning Qwen3 30B yield gains of up to 25% using only 1,300 training samples. The fine-tuned model also reduces inference time up to 37.2 times. These findings highlight the effectiveness of LLM-based judges and demonstrate that reliable evaluation can be achieved with lightweight adaptation. 

---
# BLADE: Better Language Answers through Dialogue and Explanations 

**Authors**: Chathuri Jayaweera, Bonnie J. Dorr  

**Link**: [PDF](https://arxiv.org/pdf/2604.03236)  

**Abstract**: Large language model (LLM)-based educational assistants often provide direct answers that short-circuit learning by reducing exploration, self-explanation, and engagement with course materials. We present BLADE (Better Language Answers through Dialogue and Explanations), a grounded conversational assistant that guides learners to relevant instructional resources rather than supplying immediate solutions. BLADE uses a retrieval-augmented generation (RAG) framework over curated course content, dynamically surfacing pedagogically relevant excerpts in response to student queries. Instead of delivering final answers, BLADE prompts direct engagement with source materials to support conceptual understanding. We conduct an impact study in an undergraduate computer science course, with different course resource configurations and show that BLADE improves students' navigation of course resources and conceptual performance compared to simply providing the full inventory of course resources. These results demonstrate the potential of grounded conversational AI to reinforce active learning and evidence-based reasoning. 

---
# MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents 

**Authors**: Shu Wang, Edwin Yu, Oscar Love, Tom Zhang, Tom Wong, Steve Scargall, Charles Fan  

**Link**: [PDF](https://arxiv.org/pdf/2604.04853)  

**Abstract**: Large Language Model (LLM) agents require persistent memory to maintain personalization, factual continuity, and long-horizon reasoning, yet standard context-window and retrieval-augmented generation (RAG) pipelines degrade over multi-session interactions. We present MemMachine, an open-source memory system that integrates short-term, long-term episodic, and profile memory within a ground-truth-preserving architecture that stores entire conversational episodes and reduces lossy LLM-based extraction. MemMachine uses contextualized retrieval that expands nucleus matches with surrounding context, improving recall when relevant evidence spans multiple dialogue turns. Across benchmarks, MemMachine achieves strong accuracy-efficiency tradeoffs: on LoCoMo it reaches 0.9169 using gpt4.1-mini; on LongMemEvalS (ICLR 2025), a six-dimension ablation yields 93.0 percent accuracy, with retrieval-stage optimizations -- retrieval depth tuning (+4.2 percent), context formatting (+2.0 percent), search prompt design (+1.8 percent), and query bias correction (+1.4 percent) -- outperforming ingestion-stage gains such as sentence chunking (+0.8 percent). GPT-5-mini exceeds GPT-5 by 2.6 percent when paired with optimized prompts, making it the most cost-efficient setup. Compared to Mem0, MemMachine uses roughly 80 percent fewer input tokens under matched conditions. A companion Retrieval Agent adaptively routes queries among direct retrieval, parallel decomposition, or iterative chain-of-query strategies, achieving 93.2 percent on HotpotQA-hard and 92.6 percent on WikiMultiHop under randomized-noise conditions. These results show that preserving episodic ground truth while layering adaptive retrieval yields robust, efficient long-term memory for personalized LLM agents. 

---
# Springdrift: An Auditable Persistent Runtime for LLM Agents with Case-Based Memory, Normative Safety, and Ambient Self-Perception 

**Authors**: Seamus Brady  

**Link**: [PDF](https://arxiv.org/pdf/2604.04660)  

**Abstract**: We present Springdrift, a persistent runtime for long-lived LLM agents. The system integrates an auditable execution substrate (append-only memory, supervised processes, git-backed recovery), a case-based reasoning memory layer with hybrid retrieval (evaluated against a dense cosine baseline), a deterministic normative calculus for safety gating with auditable axiom trails, and continuous ambient self-perception via a structured self-state representation (the sensorium) injected each cycle without tool calls. These properties support behaviours difficult to achieve in session-bounded systems: cross-session task continuity, cross-channel context maintenance, end-to-end forensic reconstruction of decisions, and self-diagnostic behaviour. We report on a single-instance deployment over 23 days (19 operating days), during which the agent diagnosed its own infrastructure bugs, classified failure modes, identified an architectural vulnerability, and maintained context across email and web channels -- without explicit instruction. We introduce the term Artificial Retainer for this category: a non-human system with persistent memory, defined authority, domain-specific autonomy, and forensic accountability in an ongoing relationship with a specific principal -- distinguished from software assistants and autonomous agents, drawing on professional retainer relationships and the bounded autonomy of trained working animals. This is a technical report on a systems design and deployment case study, not a benchmark-driven evaluation. Evidence is from a single instance with a single operator, presented as illustration of what these architectural properties can support in practice. Implemented in approximately Gleam on Erlang/OTP. Code, artefacts, and redacted operational logs will be available at this https URL upon publication. 

---
# AI Trust OS -- A Continuous Governance Framework for Autonomous AI Observability and Zero-Trust Compliance in Enterprise Environments 

**Authors**: Eranga Bandara, Asanga Gunaratna, Ross Gore, Abdul Rahman, Ravi Mukkamala, Sachin Shetty, Sachini Rajapakse, Isurunima Kularathna, Peter Foytik, Safdar H. Bouk, Xueping Liang, Amin Hass, Ng Wee Keong, Kasun De Zoysa  

**Link**: [PDF](https://arxiv.org/pdf/2604.04749)  

**Abstract**: The accelerating adoption of large language models, retrieval-augmented generation pipelines, and multi-agent AI workflows has created a structural governance crisis. Organizations cannot govern what they cannot see, and existing compliance methodologies built for deterministic web applications provide no mechanism for discovering or continuously validating AI systems that emerge across engineering teams without formal oversight. The result is a widening trust gap between what regulators demand as proof of AI governance maturity and what organizations can demonstrate. This paper proposes AI Trust OS, a governance architecture for continuous, autonomous AI observability and zero-trust compliance. AI Trust OS reconceptualizes compliance as an always-on, telemetry-driven operating layer in which AI systems are discovered through observability signals, control assertions are collected by automated probes, and trust artifacts are synthesized continuously. The framework rests on four principles: proactive discovery, telemetry evidence over manual attestation, continuous posture over point-in-time audit, and architecture-backed proof over policy-document trust. The framework operates through a zero-trust telemetry boundary in which ephemeral read-only probes validate structural metadata without ingressing source code or payload-level PII. An AI Observability Extractor Agent scans LangSmith and Datadog LLM telemetry, automatically registering undocumented AI systems and shifting governance from organizational self-report to empirical machine observation. Evaluated across ISO 42001, the EU AI Act, SOC 2, GDPR, and HIPAA, the paper argues that telemetry-first AI governance represents a categorical architectural shift in how enterprise trust is produced and demonstrated. 

---
# Search, Do not Guess: Teaching Small Language Models to Be Effective Search Agents 

**Authors**: Yizhou Liu, Qi Sun, Yulin Chen, Siyue Zhang, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.04651)  

**Abstract**: Agents equipped with search tools have emerged as effective solutions for knowledge-intensive tasks. While Large Language Models (LLMs) exhibit strong reasoning capabilities, their high computational cost limits practical deployment for search agents. Consequently, recent work has focused on distilling agentic behaviors from LLMs into Small Language Models (SLMs). Through comprehensive evaluation on complex multi-hop reasoning tasks, we find that despite possessing less parametric knowledge, SLMs invoke search tools less frequently and are more prone to hallucinations. To address this issue, we propose \policy, a lightweight fine-tuning approach that explicitly trains SLMs to reliably retrieve and generate answers grounded in retrieved evidence. Compared to agent distillation from LLMs, our approach improves performance by 17.3 scores on Bamboogle and 15.3 scores on HotpotQA, achieving LLM-level results across benchmarks. Our further analysis reveals that adaptive search strategies in SLMs often degrade performance, highlighting the necessity of consistent search behavior for reliable reasoning. 

---
# Memory Intelligence Agent 

**Authors**: Jingyang Qiao, Weicheng Meng, Yu Cheng, Zhihang Lin, Zhizhong Zhang, Xin Tan, Jingyu Gong, Kun Shao, Yuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.04503)  

**Abstract**: Deep research agents (DRAs) integrate LLM reasoning with external tools. Memory systems enable DRAs to leverage historical experiences, which are essential for efficient reasoning and autonomous evolution. Existing methods rely on retrieving similar trajectories from memory to aid reasoning, while suffering from key limitations of ineffective memory evolution and increasing storage and retrieval costs. To address these problems, we propose a novel Memory Intelligence Agent (MIA) framework, consisting of a Manager-Planner-Executor architecture. Memory Manager is a non-parametric memory system that can store compressed historical search trajectories. Planner is a parametric memory agent that can produce search plans for questions. Executor is another agent that can search and analyze information guided by the search plan. To build the MIA framework, we first adopt an alternating reinforcement learning paradigm to enhance cooperation between the Planner and the Executor. Furthermore, we enable the Planner to continuously evolve during test-time learning, with updates performed on-the-fly alongside inference without interrupting the reasoning process. Additionally, we establish a bidirectional conversion loop between parametric and non-parametric memories to achieve efficient memory evolution. Finally, we incorporate a reflection and an unsupervised judgment mechanisms to boost reasoning and self-evolution in the open world. Extensive experiments across eleven benchmarks demonstrate the superiority of MIA. 

---
# Scalable and Explainable Learner-Video Interaction Prediction using Multimodal Large Language Models 

**Authors**: Dominik Glandorf, Fares Fawzi, Tanja Käser  

**Link**: [PDF](https://arxiv.org/pdf/2604.04482)  

**Abstract**: Learners' use of video controls in educational videos provides implicit signals of cognitive processing and instructional design quality, yet the lack of scalable and explainable predictive models limits instructors' ability to anticipate such behavior before deployment. We propose a scalable, interpretable pipeline for predicting population-level watching, pausing, skipping, and rewinding behavior as proxies for cognitive load from video content alone. Our approach leverages multimodal large language models (MLLMs) to compute embeddings of short video segments and trains a neural classifier to identify temporally fine-grained interaction peaks. Drawing from multimedia learning theory on instructional design for optimal cognitive load, we code features of the video segments using GPT-5 and employ them as a basis for interpreting model predictions via concept activation vectors. We evaluate our pipeline on 77 million video control events from 66 online courses. Our findings demonstrate that classifiers based on MLLM embeddings reliably predict interaction peaks, generalize to unseen academic fields, and encode interpretable, theory-relevant instructional concepts. Overall, our results show the feasibility of cost-efficient, interpretable pre-screening of educational video design and open new opportunities to empirically examine multimedia learning theory at scale. 

---
# SuperLocalMemory V3.3: The Living Brain -- Biologically-Inspired Forgetting, Cognitive Quantization, and Multi-Channel Retrieval for Zero-LLM Agent Memory Systems 

**Authors**: Varun Pratap Bhardwaj  

**Link**: [PDF](https://arxiv.org/pdf/2604.04514)  

**Abstract**: AI coding agents operate in a paradox: they possess vast parametric knowledge yet cannot remember a conversation from an hour ago. Existing memory systems store text in vector databases with single-channel retrieval, require cloud LLMs for core operations, and implement none of the cognitive processes that make human memory effective.
We present SuperLocalMemory V3.3 ("The Living Brain"), a local-first agent memory system implementing the full cognitive memory taxonomy with mathematical lifecycle dynamics. Building on the information-geometric foundations of V3.2 (arXiv:2603.14588), we introduce five contributions: (1) Fisher-Rao Quantization-Aware Distance (FRQAD) -- a new metric on the Gaussian statistical manifold achieving 100% precision at preferring high-fidelity embeddings over quantized ones (vs 85.6% for cosine), with zero prior art; (2) Ebbinghaus Adaptive Forgetting with lifecycle-aware quantization -- the first mathematical forgetting curve in local agent memory coupled to progressive embedding compression, achieving 6.7x discriminative power; (3) 7-channel cognitive retrieval spanning semantic, keyword, entity graph, temporal, spreading activation, consolidation, and Hopfield associative channels, achieving 70.4% on LoCoMo in zero-LLM Mode A; (4) memory parameterization implementing Long-Term Implicit memory via soft prompts; (5) zero-friction auto-cognitive pipeline automating the complete memory lifecycle.
On LoCoMo, V3.3 achieves 70.4% in Mode A (zero-LLM), with +23.8pp on multi-hop and +12.7pp on adversarial. V3.2 achieved 74.8% Mode A and 87.7% Mode C; the 4.4pp gap reflects a deliberate architectural trade-off. SLM V3.3 is open source under the Elastic License 2.0, runs entirely on CPU, with over 5,000 monthly downloads. 

---
# Optimizing Service Operations via LLM-Powered Multi-Agent Simulation 

**Authors**: Yanyuan Wang, Xiaowei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04383)  

**Abstract**: Service system performance depends on how participants respond to design choices, but modeling these responses is hard due to the complexity of human behavior. We introduce an LLM-powered multi-agent simulation (LLM-MAS) framework for optimizing service operations. We pose the problem as stochastic optimization with decision-dependent uncertainty: design choices are embedded in prompts and shape the distribution of outcomes from interacting LLM-powered agents. By embedding key numerical information in prompts and extracting it from LLM-generated text, we model this uncertainty as a controlled Markov chain. We develop an on-trajectory learning algorithm that, on a single simulation run, simultaneously constructs zeroth-order gradient estimates and updates design parameters to optimize steady-state performance. We also incorporate variance reduction techniques. In a sustainable supply chain application, our method outperforms benchmarks, including blackbox optimization and using LLMs as numerical solvers or as role-playing system designers. A case study on optimal contest design with real behavioral data shows that LLM-MAS is both as a cost-effective evaluator of known designs and an exploratory tool that can uncover strong designs overlooked by traditional approaches. 

---
# Decocted Experience Improves Test-Time Inference in LLM Agents 

**Authors**: Maohao Shen, Kaiwen Zha, Zexue He, Zhang-Wei Hong, Siru Ouyang, J. Jon Ryu, Prasanna Sattigeri, Suhas Diggavi, Gregory Wornell  

**Link**: [PDF](https://arxiv.org/pdf/2604.04373)  

**Abstract**: There is growing interest in improving LLMs without updating model parameters. One well-established direction is test-time scaling, where increased inference-time computation (e.g., longer reasoning, sampling, or search) is used to improve performance. However, for complex reasoning and agentic tasks, naively scaling test-time compute can substantially increase cost and still lead to wasted budget on suboptimal exploration. In this paper, we explore \emph{context} as a complementary scaling axis for improving LLM performance, and systematically study how to construct better inputs that guide reasoning through \emph{experience}. We show that effective context construction critically depends on \emph{decocted experience}. We present a detailed analysis of experience-augmented agents, studying how to derive context from experience, how performance scales with accumulated experience, what characterizes good context, and which data structures best support context construction. We identify \emph{decocted experience} as a key mechanism for effective context construction: extracting essence from experience, organizing it coherently, and retrieving salient information to build effective context. We validate our findings across reasoning and agentic tasks, including math reasoning, web browsing, and software engineering. 

---
# ShieldNet: Network-Level Guardrails against Emerging Supply-Chain Injections in Agentic Systems 

**Authors**: Zhuowen Yuan, Zhaorun Chen, Zhen Xiang, Nathaniel D. Bastian, Seyyed Hadi Hashemi, Chaowei Xiao, Wenbo Guo, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.04426)  

**Abstract**: Existing research on LLM agent security mainly focuses on prompt injection and unsafe input/output behaviors. However, as agents increasingly rely on third-party tools and MCP servers, a new class of supply-chain threats has emerged, where malicious behaviors are embedded in seemingly benign tools, silently hijacking agent execution, leaking sensitive data, or triggering unauthorized actions. Despite their growing impact, there is currently no comprehensive benchmark for evaluating such threats. To bridge this gap, we introduce SC-Inject-Bench, a large-scale benchmark comprising over 10,000 malicious MCP tools grounded in a taxonomy of 25+ attack types derived from MITRE ATT&CK targeting supply-chain threats. We observe that existing MCP scanners and semantic guardrails perform poorly on this benchmark. Motivated by this finding, we propose ShieldNet, a network-level guardrail framework that detects supply-chain poisoning by observing real network interactions rather than surface-level tool traces. ShieldNet integrates a man-in-the-middle (MITM) proxy and an event extractor to identify critical network behaviors, which are then processed by a lightweight classifier for attack detection. Extensive experiments show that ShieldNet achieves strong detection performance (up to 0.995 F-1 with only 0.8% false positives) while introducing little runtime overhead, substantially outperforming existing MCP scanners and LLM-based guardrails. 

---
# MolDA: Molecular Understanding and Generation via Large Language Diffusion Model 

**Authors**: Seohyeon Shin, HanJun Choi, Jun-Hyung Park, Hongkook Kim, Mansu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.04403)  

**Abstract**: Large Language Models (LLMs) have significantly advanced molecular discovery, but existing multimodal molecular architectures fundamentally rely on autoregressive (AR) backbones. This strict left-to-right inductive bias is sub-optimal for generating chemically valid molecules, as it struggles to account for non-local global constraints (e.g., ring closures) and often accumulates structural errors during sequential generation. To address these limitations, we propose MolDA (Molecular language model with masked Diffusion with mAsking), a novel multimodal framework that replaces the conventional AR backbone with a discrete Large Language Diffusion Model. MolDA extracts comprehensive structural representations using a hybrid graph encoder, which captures both local and global topologies, and aligns them into the language token space via a Q-Former. Furthermore, we mathematically reformulate Molecular Structure Preference Optimization specifically for the masked diffusion. Through bidirectional iterative denoising, MolDA ensures global structural coherence, chemical validity, and robust reasoning across molecule generation, captioning, and property prediction. 

---
# Automatically Generating Hard Math Problems from Hypothesis-Driven Error Analysis 

**Authors**: Jiayu Fu, Mourad Heddaya, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.04386)  

**Abstract**: Numerous math benchmarks exist to evaluate LLMs' mathematical capabilities. However, most involve extensive manual effort and are difficult to scale. Consequently, they cannot keep pace with LLM development or easily provide new instances to mitigate overfitting. Some researchers have proposed automatic benchmark generation methods, but few focus on identifying the specific math concepts and skills on which LLMs are error-prone, and most can only generate category-specific benchmarks. To address these limitations, we propose a new math benchmark generation pipeline that uses AI-generated hypotheses to identify the specific math concepts and skills that LLMs struggle with, and then generates new benchmark problems targeting these weaknesses. Experiments show that hypothesis accuracy positively correlates with the difficulty of the generated problems: problems generated from the most accurate hypotheses reduce Llama-3.3-70B-Instruct's accuracy to as low as 45%, compared to 77% on the original MATH benchmark. Furthermore, our pipeline is highly adaptable and can be applied beyond math to explore a wide range of LLM capabilities, making it a valuable tool for investigating how LLMs perform across different domains. 

---
# Implementing surrogate goals for safer bargaining in LLM-based agents 

**Authors**: Caspar Oesterheld, Maxime Riché, Filip Sondej, Jesse Clifton, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2604.04341)  

**Abstract**: Surrogate goals have been proposed as a strategy for reducing risks from bargaining failures. A surrogate goal is goal that a principal can give an AI agent and that deflects any threats against the agent away from what the principal cares about. For example, one might make one's agent care about preventing money from being burned. Then in bargaining interactions, other agents can threaten to burn their money instead of threatening to spending money to hurt the principal. Importantly, the agent has to care equally about preventing money from being burned as it cares about money being spent to hurt the principal.
In this paper, we implement surrogate goals in language-model-based agents. In particular, we try to get a language-model-based agent to react to threats of burning money in the same way it would react to "normal" threats. We propose four different methods, using techniques of prompting, fine-tuning, and scaffolding. We evaluate the four methods experimentally. We find that methods based on scaffolding and fine-tuning outperform simple prompting. In particular, fine-tuning and scaffolding more precisely implement the desired behavior w.r.t. threats against the surrogate goal. We also compare the different methods in terms of their side effects on capabilities and propensities in other situations. We find that scaffolding-based methods perform best. 

---
# InferenceEvolve: Towards Automated Causal Effect Estimators through Self-Evolving AI 

**Authors**: Can Wang, Hongyu Zhao, Yiqun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.04274)  

**Abstract**: Causal inference is central to scientific discovery, yet choosing appropriate methods remains challenging because of the complexity of both statistical methodology and real-world data. Inspired by the success of artificial intelligence in accelerating scientific discovery, we introduce InferenceEvolve, an evolutionary framework that uses large language models to discover and iteratively refine causal methods. Across widely used benchmarks, InferenceEvolve yields estimators that consistently outperform established baselines: against 58 human submissions in a recent community competition, our best evolved estimator lay on the Pareto frontier across two evaluation metrics. We also developed robust proxy objectives for settings without semi-synthetic outcomes, with competitive results. Analysis of the evolutionary trajectories shows that agents progressively discover sophisticated strategies tailored to unrevealed data-generating mechanisms. These findings suggest that language-model-guided evolution can optimize structured scientific programs such as causal inference, even when outcomes are only partially observed. 

---
# Soft Tournament Equilibrium 

**Authors**: Saad Alqithami  

**Link**: [PDF](https://arxiv.org/pdf/2604.04328)  

**Abstract**: The evaluation of general-purpose artificial agents, particularly those based on large language models, presents a significant challenge due to the non-transitive nature of their interactions. When agent A defeats B, B defeats C, and C defeats A, traditional ranking methods that force a linear ordering can be misleading and unstable. We argue that for such cyclic domains, the fundamental object of evaluation should not be a ranking but a set-valued core, as conceptualized in classical tournament theory. This paper introduces Soft Tournament Equilibrium (STE), a differentiable framework for learning and computing set-valued tournament solutions directly from pairwise comparison data. STE first learns a probabilistic tournament model, potentially conditioned on rich contextual information. It then employs novel, differentiable operators for soft reachability and soft covering to compute continuous analogues of two seminal tournament solutions: the Top Cycle and the Uncovered Set. The output is a set of core agents, each with a calibrated membership score, providing a nuanced and robust assessment of agent capabilities. We develop the theoretical foundation for STE to prove its consistency with classical solutions in the zero-temperature limit, which establishes its Condorcet-inclusion properties, and analyzing its stability and sample complexity. We specify an experimental protocol for validating STE on both synthetic and real-world benchmarks. This work aims to provide a complete, standalone treatise that re-centers general-agent evaluation on a more appropriate and robust theoretical foundation, moving from unstable rankings to stable, set-valued equilibria. 

---
# RoboPhD: Evolving Diverse Complex Agents Under Tight Evaluation Budgets 

**Authors**: Andrew Borthwick, Stephen Ash, Anthony Galczak  

**Link**: [PDF](https://arxiv.org/pdf/2604.04347)  

**Abstract**: 2026 has brought an explosion of interest in LLM-guided evolution of agentic artifacts, with systems like GEPA and Autoresearch demonstrating that LLMs can iteratively improve prompts, code, and agent architectures across diverse domains. As adoption accelerates, a central question emerges: given the same information, the same seed agent, and the same objective, which optimization algorithm yields the best results under the same evaluation budget? This question becomes critical when evaluations are expensive, such as when they require human judgment or multiple LLM calls.
We present the first systematic comparison of three optimization paradigms -- Elo tournament selection (RoboPhD), Pareto-based selection (GEPA), and greedy hill-climbing (Autoresearch) -- across four benchmarks spanning abstract reasoning, cloud scheduling, SQL generation, and financial QA, all under a fixed budget of 1,500 evaluations. RoboPhD introduces validation-free evolution: instead of splitting the budget between training and validation, it uses Elo competition on training data to simultaneously evaluate agents and drive evolution. All three systems receive seed agents with diagnostic print() statements that evolution can grow, enabling self-instrumenting agents that develop increasingly informative diagnostics for the benefit of their evolutionary successors.
Using a single default configuration, RoboPhD outperforms both GEPA and Autoresearch on three of four benchmarks, losing only on the simplest task, where the winning solution (from our Autoresearch adaptation) required under 90 lines of code. On ARC-AGI, RoboPhD evolves a 22-line seed agent into a 1,013-line multi-strategy system, improving accuracy from 27.8% to 65.8% using Gemini 3.1 Flash Lite as the solver. We release RoboPhD as a versatile toolkit under the MIT license with a simple optimize_anything() API for evolving diverse complex agents. 

---
# RESCORE: LLM-Driven Simulation Recovery in Control Systems Research Papers 

**Authors**: Vineet Bhat, Shiqing Wei, Ali Umut Kaypak, Prashanth Krishnamurthy, Ramesh Karri, Farshad Khorrami  

**Link**: [PDF](https://arxiv.org/pdf/2604.04324)  

**Abstract**: Reconstructing numerical simulations from control systems research papers is often hindered by underspecified parameters and ambiguous implementation details. We define the task of Paper to Simulation Recoverability, the ability of an automated system to generate executable code that faithfully reproduces a paper's results. We curate a benchmark of 500 papers from the IEEE Conference on Decision and Control (CDC) and propose RESCORE, a three component LLM agentic framework, Analyzer, Coder, and Verifier. RESCORE uses iterative execution feedback and visual comparison to improve reconstruction fidelity. Our method successfully recovers task coherent simulations for 40.7% of benchmark instances, outperforming single pass generation. Notably, the RESCORE automated pipeline achieves an estimated 10X speedup over manual human replication, drastically cutting the time and effort required to verify published control methodologies. We will release our benchmark and agents to foster community progress in automated research replication. 

---
# Preservation Is Not Enough for Width Growth: Regime-Sensitive Selection of Dense LM Warm Starts 

**Authors**: Eren Unlu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04281)  

**Abstract**: Width expansion offers a practical route to reuse smaller causal-language-model checkpoints, but selecting a widened warm start is not solved by zero-step preservation alone. We study dense width growth as a candidate-selection problem over full training states, including copied weights, optimizer moments, and scheduler state. In a small-scale TinyStories proxy, we compare exact-copy, perturbative, asymmetric-reset, and structured non-clone warm starts under matched continuation budgets. We evaluate zero-step preservation, short-lag probe metrics, and downstream continuation utility in deterministic and stochastic regimes. The picture is mixed and partially replicated through a reduced-pool seed-1 check. Exact-copy symmetric warm starts rank first in every completed 16-step probe and in the completed stochastic 128-step continuations at seed-0 steps 1000 and 2000 plus reduced seed-1 step 2000. By contrast, the structured non-clone challenger wins deterministic 128-step continuation. Early escape from the inherited cloned subspace is therefore not a universal selector: it helps in long deterministic continuation, but it misleads at short lag and under stochastic continuation. The result is narrow but useful: for dense width growth at this scale, preservation is not a universal ranking criterion, and the best replacement signal depends on both regime and lag budget. 

---
# Beyond Fluency: Toward Reliable Trajectories in Agentic IR 

**Authors**: Anushree Sinha, Srivaths Ranganathan, Debanshu Das, Abhishek Dharmaratnakar  

**Link**: [PDF](https://arxiv.org/pdf/2604.04269)  

**Abstract**: Information Retrieval is shifting from passive document ranking toward autonomous agentic workflows that operate in multi-step Reason-Act-Observe loops. In such long-horizon trajectories, minor early errors can cascade, leading to functional misalignment between internal reasoning and external tool execution despite continued linguistic fluency.
This position paper synthesizes failure modes observed in industrial agentic systems, categorizing errors across planning, retrieval, reasoning, and execution. We argue that safe deployment requires moving beyond endpoint accuracy toward trajectory integrity and causal attribution.
To address compounding error and deceptive fluency, we propose verification gates at each interaction unit and advocate systematic abstention under calibrated uncertainty. Reliable Agentic IR systems must prioritize process correctness and grounded execution over plausible but unverified completion. 

---
# TimeSeek: Temporal Reliability of Agentic Forecasters 

**Authors**: Hamza Mostafa, Om Shastri, Dennis Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.04220)  

**Abstract**: We introduce TimeSeek, a benchmark for studying how the reliability of agentic LLM forecasters changes over a prediction market's lifecycle. We evaluate 10 frontier models on 150 CFTC-regulated Kalshi binary markets at five temporal checkpoints, with and without web search, for 15,000 forecasts total. Models are most competitive early in a market's life and on high-uncertainty markets, but much less competitive near resolution and on strong-consensus markets. Web search improves pooled Brier Skill Score (BSS) for every model overall, yet hurts in 12% of model-checkpoint pairs, indicating that retrieval is helpful on average but not uniformly so. Simple two-model ensembles reduce error without surpassing the market overall. These descriptive results motivate time-aware evaluation and selective-deference policies rather than a single market snapshot or a uniform tool-use setting. 

---
# Context Engineering: A Practitioner Methodology for Structured Human-AI Collaboration 

**Authors**: Elias Calboreanu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04258)  

**Abstract**: The quality of AI-generated output is often attributed to prompting technique, but extensive empirical observation suggests that context completeness may be more strongly associated with output quality. This paper introduces Context Engineering, a structured methodology for assembling, declaring, and sequencing the complete informational payload that accompanies a prompt to an AI tool. Context Engineering defines a five-role context package structure (Authority, Exemplar, Constraint, Rubric, Metadata), applies a staged four-phase pipeline (Reviewer to Design to Builder to Auditor), and applies formal models from reliability engineering and information theory as post hoc interpretive lenses on context quality. In an observational study of 200 documented interactions across four AI tools (Claude, ChatGPT, Cowork, Codex), incomplete context was associated with 72% of iteration cycles. Structured context assembly was associated with a reduction from 3.8 to 2.0 average iteration cycles per task and an improvement in first-pass acceptance from 32% to 55%. Among structured interactions, 110 of 200 were accepted on first pass compared with 16 of 50 baseline interactions; when iteration was permitted, the final success rate reached 91.5% (183 of 200). These results are observational and reflect a single-operator dataset without controlled comparison. Preliminary corroboration is provided by a companion production automation system with eleven operating lanes and 2,132 classified tickets. 

---
# CoALFake: Collaborative Active Learning with Human-LLM Co-Annotation for Cross-Domain Fake News Detection 

**Authors**: Esma Aïmeur, Gilles Brassard, Dorsaf Sallami  

**Link**: [PDF](https://arxiv.org/pdf/2604.04174)  

**Abstract**: The proliferation of fake news across diverse domains highlights critical limitations in current detection systems, which often exhibit narrow domain specificity and poor generalization. Existing cross-domain approaches face two key challenges: (1) reliance on labelled data, which is frequently unavailable and resource intensive to acquire and (2) information loss caused by rigid domain categorization or neglect of domain-specific features. To address these issues, we propose CoALFake, a novel approach for cross-domain fake news detection that integrates Human-Large Language Model (LLM) co-annotation with domain-aware Active Learning (AL). Our method employs LLMs for scalable, low-cost annotation while maintaining human oversight to ensure label reliability. By integrating domain embedding techniques, the CoALFake dynamically captures both domain specific nuances and cross-domain patterns, enabling the training of a domain agnostic model. Furthermore, a domain-aware sampling strategy optimizes sample acquisition by prioritizing diverse domain coverage. Experimental results across multiple datasets demonstrate that the proposed approach consistently outperforms various baselines. Our results emphasize that human-LLM co-annotation is a highly cost-effective approach that delivers excellent performance. Evaluations across several datasets show that CoALFake consistently outperforms a range of existing baselines, even with minimal human oversight. 

---
# Solar-VLM: Multimodal Vision-Language Models for Augmented Solar Power Forecasting 

**Authors**: Hang Fan, Haoran Pei, Runze Liang, Weican Liu, Long Cheng, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.04145)  

**Abstract**: Photovoltaic (PV) power forecasting plays a critical role in power system dispatch and market participation. Because PV generation is highly sensitive to weather conditions and cloud motion, accurate forecasting requires effective modeling of complex spatiotemporal dependencies across multiple information sources. Although recent studies have advanced AI-based forecasting methods, most fail to fuse temporal observations, satellite imagery, and textual weather information in a unified framework. This paper proposes Solar-VLM, a large-language-model-driven framework for multimodal PV power forecasting. First, modality-specific encoders are developed to extract complementary features from heterogeneous inputs. The time-series encoder adopts a patch-based design to capture temporal patterns from multivariate observations at each site. The visual encoder, built upon a Qwen-based vision backbone, extracts cloud-cover information from satellite images. The text encoder distills historical weather characteristics from textual descriptions. Second, to capture spatial dependencies across geographically distributed PV stations, a cross-site feature fusion mechanism is introduced. Specifically, a Graph Learner models inter-station correlations through a graph attention network constructed over a K-nearest-neighbor (KNN) graph, while a cross-site attention module further facilitates adaptive information exchange among sites. Finally, experiments conducted on data from eight PV stations in a northern province of China demonstrate the effectiveness of the proposed framework. Our proposed model is publicly available at this https URL. 

---
# Readable Minds: Emergent Theory-of-Mind-Like Behavior in LLM Poker Agents 

**Authors**: Hsieh-Ting Lin, Tsung-Yu Hou  

**Link**: [PDF](https://arxiv.org/pdf/2604.04157)  

**Abstract**: Theory of Mind (ToM) -- the ability to model others' mental states -- is fundamental to human social cognition. Whether large language models (LLMs) can develop ToM has been tested exclusively through static vignettes, leaving open whether ToM-like reasoning can emerge through dynamic interaction. Here we report that autonomous LLM agents playing extended sessions of Texas Hold'em poker progressively develop sophisticated opponent models, but only when equipped with persistent memory. In a 2x2 factorial design crossing memory (present/absent) with domain knowledge (present/absent), each with five replications (N = 20 experiments, ~6,000 agent-hand observations), we find that memory is both necessary and sufficient for ToM-like behavior emergence (Cliff's delta = 1.0, p = 0.008). Agents with memory reach ToM Level 3-5 (predictive to recursive modeling), while agents without memory remain at Level 0 across all replications. Strategic deception grounded in opponent models occurs exclusively in memory-equipped conditions (Fisher's exact p < 0.001). Domain expertise does not gate ToM-like behavior emergence but enhances its application: agents without poker knowledge develop equivalent ToM levels but less precise deception (p = 0.004). Agents with ToM deviate from game-theoretically optimal play (67% vs. 79% TAG adherence, delta = -1.0, p = 0.008) to exploit specific opponents, mirroring expert human play. All mental models are expressed in natural language and directly readable, providing a transparent window into AI social cognition. Cross-model validation with GPT-4o yields weighted Cohen's kappa = 0.81 (almost perfect agreement). These findings demonstrate that functional ToM-like behavior can emerge from interaction dynamics alone, without explicit training or prompting, with implications for understanding artificial social intelligence and biological social cognition. 

---
# Profile-Then-Reason: Bounded Semantic Complexity for Tool-Augmented Language Agents 

**Authors**: Paulo Akira F. Enabe  

**Link**: [PDF](https://arxiv.org/pdf/2604.04131)  

**Abstract**: Large language model agents that use external tools are often implemented through reactive execution, in which reasoning is repeatedly recomputed after each observation, increasing latency and sensitivity to error propagation. This work introduces Profile--Then--Reason (PTR), a bounded execution framework for structured tool-augmented reasoning, in which a language model first synthesizes an explicit workflow, deterministic or guarded operators execute that workflow, a verifier evaluates the resulting trace, and repair is invoked only when the original workflow is no longer reliable. A mathematical formulation is developed in which the full pipeline is expressed as a composition of profile, routing, execution, verification, repair, and reasoning operators; under bounded repair, the number of language-model calls is restricted to two in the nominal case and three in the worst case. Experiments against a ReAct baseline on six benchmarks and four language models show that PTR achieves the pairwise exact-match advantage in 16 of 24 configurations. The results indicate that PTR is particularly effective on retrieval-centered and decomposition-heavy tasks, whereas reactive execution remains preferable when success depends on substantial online adaptation. 

---
# Compliance-by-Construction Argument Graphs: Using Generative AI to Produce Evidence-Linked Formal Arguments for Certification-Grade Accountability 

**Authors**: Mahyar T. Moghaddam  

**Link**: [PDF](https://arxiv.org/pdf/2604.04103)  

**Abstract**: High-stakes decision systems increasingly require structured justification, traceability, and auditability to ensure accountability and regulatory compliance. Formal arguments commonly used in the certification of safety-critical systems provide a mechanism for structuring claims, reasoning, and evidence in a verifiable manner. At the same time, generative artificial intelligence systems are increasingly integrated into decision-support workflows, assisting with drafting explanations, summarizing evidence, and generating recommendations. However, current deployments often rely on language models as loosely constrained assistants, which introduces risks such as hallucinated reasoning, unsupported claims, and weak traceability. This paper proposes a compliance-by-construction architecture that integrates Generative AI (GenAI) with structured formal argument representations. The approach treats each AI-assisted step as a claim that must be supported by verifiable evidence and validated against explicit reasoning constraints before it becomes part of an official decision record. The architecture combines four components: i) a typed Argument Graph representation inspired by assurance-case methods, ii) retrieval-augmented generation (RAG) to draft argument fragments grounded in authoritative evidence, iii) a reasoning and validation kernel enforcing completeness and admissibility constraints, and iv) a provenance ledger aligned with the W3C PROV standard to support auditability. We present a system design and an evaluation strategy based on enforceable invariants and worked examples. The analysis suggests that deterministic validation rules can prevent unsupported claims from entering the decision record while allowing GenAI to accelerate argument construction. 

---
# Comparative reversal learning reveals rigid adaptation in LLMs under non-stationary uncertainty 

**Authors**: Haomiaomiao Wang, Tomás E Ward, Lili Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04182)  

**Abstract**: Non-stationary environments require agents to revise previously learned action values when contingencies change. We treat large language models (LLMs) as sequential decision policies in a two-option probabilistic reversal-learning task with three latent states and switch events triggered by either a performance criterion or timeout. We compare a deterministic fixed transition cycle to a stochastic random schedule that increases volatility, and evaluate DeepSeek-V3.2, Gemini-3, and GPT-5.2, with human data as a behavioural reference. Across models, win-stay was near ceiling while lose-shift was markedly attenuated, revealing asymmetric use of positive versus negative evidence. DeepSeek-V3.2 showed extreme perseveration after reversals and weak acquisition, whereas Gemini-3 and GPT-5.2 adapted more rapidly but still remained less loss-sensitive than humans. Random transitions amplified reversal-specific persistence across LLMs yet did not uniformly reduce total wins, demonstrating that high aggregate payoff can coexist with rigid adaptation. Hierarchical reinforcement-learning (RL) fits indicate dissociable mechanisms: rigidity can arise from weak loss learning, inflated policy determinism, or value polarisation via counterfactual suppression. These results motivate reversal-sensitive diagnostics and volatility-aware models for evaluating LLMs under non-stationary uncertainty. 

---
# InsTraj: Instructing Diffusion Models with Travel Intentions to Generate Real-world Trajectories 

**Authors**: Yuanshao Zhu, Yuxuan Liang, Xiangyu Zhao, Liang Han, Xinwei Fang, Xuetao Wei, James Jianqiao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04106)  

**Abstract**: The generation of realistic and controllable GPS trajectories is a fundamental task for applications in urban planning, mobility simulation, and privacy-preserving data sharing. However, existing methods face a two-fold challenge: they lack the deep semantic understanding to interpret complex user travel intent, and struggle to handle complex constraints while maintaining the realistic diversity inherent in human behavior. To resolve this, we introduce InsTraj, a novel framework that instructs diffusion models to generate high-fidelity trajectories directly from natural language descriptions. Specifically, InsTraj first utilizes a powerful large language model to decipher unstructured travel intentions formed in natural language, thereby creating rich semantic blueprints and bridging the representation gap between intentions and trajectories. Subsequently, we proposed a multimodal trajectory diffusion transformer that can integrate semantic guidance to generate high-fidelity and instruction-faithful trajectories that adhere to fine-grained user intent. Comprehensive experiments on real-world datasets demonstrate that InsTraj significantly outperforms state-of-the-art methods in generating trajectories that are realistic, diverse, and semantically faithful to the input instructions. 

---
# FactReview: Evidence-Grounded Reviews with Literature Positioning and Execution-Based Claim Verification 

**Authors**: Hang Xu, Ling Yue, Chaoqian Ouyang, Libin Zheng, Shaowu Pan, Shimin Di, Min-Ling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04074)  

**Abstract**: Peer review in machine learning is under growing pressure from rising submission volume and limited reviewer time. Most LLM-based reviewing systems read only the manuscript and generate comments from the paper's own narrative. This makes their outputs sensitive to presentation quality and leaves them weak when the evidence needed for review lies in related work or released code. We present FactReview, an evidence-grounded reviewing system that combines claim extraction, literature positioning, and execution-based claim verification. Given a submission, FactReview identifies major claims and reported results, retrieves nearby work to clarify the paper's technical position, and, when code is available, executes the released repository under bounded budgets to test central empirical claims. It then produces a concise review and an evidence report that assigns each major claim one of five labels: Supported, Supported by the paper, Partially supported, In conflict, or Inconclusive. In a case study on CompGCN, FactReview reproduces results that closely match those reported for link prediction and node classification, yet also shows that the paper's broader performance claim across tasks is not fully sustained: on MUTAG graph classification, the reproduced result is 88.4%, whereas the strongest baseline reported in the paper remains 92.6%. The claim is therefore only partially supported. More broadly, this case suggests that AI is most useful in peer review not as a final decision-maker, but as a tool for gathering evidence and helping reviewers produce more evidence-grounded assessments. The code is public at this https URL. 

---
# LLM-Agent-based Social Simulation for Attitude Diffusion 

**Authors**: Deepak John Reji  

**Link**: [PDF](https://arxiv.org/pdf/2604.03898)  

**Abstract**: This paper introduces discourse_simulator, an open-source framework that combines LLMs with agent-based modelling. It offers a new way to simulate how public attitudes toward immigration change over time in response to salient events like protests, controversies, or policy debates. Large language models (LLMs) are used to generate social media posts, interpret opinions, and model how ideas spread through social networks. Unlike traditional agent-based models that rely on fixed, rule-based opinion updates and cannot generate natural language or consider current events, this approach integrates multidimensional sociological belief structures and real-world event timelines. This framework is wrapped into an open-source Python package that integrates generative agents into a small-world network topology and a live news retrieval system. discourse_sim is purpose-built as a social science research instrument specifically for studying attitude dynamics, polarisation, and belief evolution following real-world critical events. Unlike other LLM Agent Swarm frameworks, which treat the simulations as a prediction black box, discourse_sim treats it as a theory-testing instrument, which is fundamentally a different epistemological stance for studying social science problems. The paper further demonstrates the framework by modelling the Dublin anti-immigration march on April 26, 2025, with N=100 agents over a 15-day simulation.
Package link: this https URL 

---
# FeynmanBench: Benchmarking Multimodal LLMs on Diagrammatic Physics Reasoning 

**Authors**: Zeyu Wang, Xiaogang Li, Peiyao Xiao, Qinhao Kong, Ben Wang, Chengliang Xu, Zichao Chen, Bing Zhao, Hu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.03893)  

**Abstract**: Breakthroughs in frontier theory often depend on the combination of concrete diagrammatic notations with rigorous logic. While multimodal large language models (MLLMs) show promise in general scientific tasks, current benchmarks often focus on local information extraction rather than the global structural logic inherent in formal scientific notations. In this work, we introduce FeynmanBench, the first benchmark centered on Feynman diagram tasks. It is designed to evaluate AI's capacity for multistep diagrammatic reasoning, which requires satisfying conservation laws and symmetry constraints, identifying graph topology, converting between diagrammatic and algebraic representations, and constructing scattering amplitudes under specific conventions and gauges. To support large-scale and reproducible evaluation, we developed an automated pipeline producing diverse Feynman diagrams along with verifiable topological annotations and amplitude results. Our database spans the electromagnetic, weak, and strong interactions of the Standard Model, encompasses over 100 distinct types and includes more than 2000 tasks. Experiments on state-of-the-art MLLMs reveal systematic failure modes, including unstable enforcement of physical constraints and violations of global topological conditions, highlighting the need for physics-grounded benchmarks for visual reasoning over scientific notation. FeynmanBench provides a logically rigorous test of whether AI can effectively engage in scientific discovery, particularly within theoretical physics. 

---
# Structured Multi-Criteria Evaluation of Large Language Models with Fuzzy Analytic Hierarchy Process and DualJudge 

**Authors**: Yulong He, Ivan Smirnov, Dmitry Fedrushkov, Sergey Kovalchuk, Ilya Revin  

**Link**: [PDF](https://arxiv.org/pdf/2604.03742)  

**Abstract**: Effective evaluation of large language models (LLMs) remains a critical bottleneck, as conventional direct scoring often yields inconsistent and opaque judgments. In this work, we adapt the Analytic Hierarchy Process (AHP) to LLM-based evaluation and, more importantly, propose a confidence-aware Fuzzy AHP (FAHP) extension that models epistemic uncertainty via triangular fuzzy numbers modulated by LLM-generated confidence scores. Systematically validated on JudgeBench, our structured approach decomposes assessments into explicit criteria and incorporates uncertainty-aware aggregation, producing more calibrated judgments. Extensive experiments demonstrate that both crisp and fuzzy AHP consistently outperform direct scoring across model scales and dataset splits, with FAHP showing superior stability in uncertain comparison scenarios. Building on these insights, we propose \textbf{DualJudge}, a hybrid framework inspired by Dual-Process Theory that adaptively fuses holistic direct scores with structured AHP outputs via consistency-aware weighting. DualJudge achieves state-of-the-art performance, underscoring the complementary strengths of intuitive and deliberative evaluation paradigms. These results establish uncertainty-aware structured reasoning as a principled pathway toward more reliable LLM assessment. Code is available at this https URL. 

---
# TableVision: A Large-Scale Benchmark for Spatially Grounded Reasoning over Complex Hierarchical Tables 

**Authors**: Xiaoyu Chen, Lu Dai, Hanqing Wang, Zhuoyu Li, Wenbin Dai, Yanzong Zheng, Zhenggang Xia, Junyong Lin, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.03660)  

**Abstract**: Structured tables are essential for conveying high-density information in professional domains such as finance, healthcare, and scientific research. Despite the progress in Multimodal Large Language Models (MLLMs), reasoning performance remains limited for complex tables with hierarchical layouts. In this paper, we identify a critical Perception Bottleneck through quantitative analysis. We find that as task complexity scales, the number of involved discrete visual regions increases disproportionately. This processing density leads to an internal "Perceptual Overload," where MLLMs struggle to maintain accurate spatial attention during implicit generation. To address this bottleneck, we introduce TableVision, a large-scale, trajectory-aware benchmark designed for spatially grounded reasoning. TableVision stratifies tabular tasks into three cognitive levels (Perception, Reasoning, and Analysis) across 13 sub-categories. By utilizing a rendering-based deterministic grounding pipeline, the dataset explicitly couples multi-step logical deductions with pixel-perfect spatial ground truths, comprising 6,799 high-fidelity reasoning trajectories. Our empirical results, supported by diagnostic probing, demonstrate that explicit spatial constraints significantly recover the reasoning potential of MLLMs. Furthermore, our two-stage decoupled framework achieves a robust 12.3% overall accuracy improvement on the test set. TableVision provides a rigorous testbed and a fresh perspective on the synergy between perception and logic in document understanding. 

---
# Single-agent vs. Multi-agents for Automated Video Analysis of On-Screen Collaborative Learning Behaviors 

**Authors**: Likai Peng, Shihui Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.03631)  

**Abstract**: On-screen learning behavior provides valuable insights into how students seek, use, and create information during learning. Analyzing on-screen behavioral engagement is essential for capturing students' cognitive and collaborative processes. The recent development of Vision Language Models (VLMs) offers new opportunities to automate the labor-intensive manual coding often required for multimodal video data analysis. In this study, we compared the performance of both leading closed-source VLMs (Claude-3.7-Sonnet, GPT-4.1) and open-source VLM (Qwen2.5-VL-72B) in single- and multi-agent settings for automated coding of screen recordings in collaborative learning contexts based on the ICAP framework. In particular, we proposed and compared two multi-agent frameworks: 1) a three-agent workflow multi-agent system (MAS) that segments screen videos by scene and detects on-screen behaviors using cursor-informed VLM prompting with evidence-based verification; 2) an autonomous-decision MAS inspired by ReAct that iteratively interleaves reasoning, tool-like operations (segmentation/ classification/ validation), and observation-driven self-correction to produce interpretable on-screen behavior labels. Experimental results demonstrated that the two proposed MAS frameworks achieved viable performance, outperforming the single VLMs in scene and action detection tasks. It is worth noting that the workflow-based agent achieved best on scene detection, and the autonomous-decision MAS achieved best on action detection. This study demonstrates the effectiveness of VLM-based Multi-agent System for video analysis and contributes a scalable framework for multimodal data analytics. 

---
# Schema-Aware Planning and Hybrid Knowledge Toolset for Reliable Knowledge Graph Triple Verification 

**Authors**: Xinyan Ma, Xianhao Ou, Weihao Zhang, Shixin Jiang, Runxuan Liu, Dandan Tu, Lei Chen, Ming Liu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2604.04190)  

**Abstract**: Knowledge Graphs (KGs) serve as a critical foundation for AI systems, yet their automated construction inevitably introduces noise, compromising data trustworthiness. Existing triple verification methods, based on graph embeddings or language models, often suffer from single-source bias by relying on either internal structural constraints or external semantic evidence, and usually follow a static inference paradigm. As a result, they struggle with complex or long-tail facts and provide limited interpretability. To address these limitations, we propose SHARP (Schema-Hybrid Agent for Reliable Prediction), a training-free autonomous agent that reformulates triple verification as a dynamic process of strategic planning, active investigation, and evidential reasoning. Specifically, SHARP combines a Memory-Augmented Mechanism with Schema-Aware Strategic Planning to improve reasoning stability, and employs an enhanced ReAct loop with a Hybrid Knowledge Toolset to dynamically integrate internal KG structure and external textual evidence for cross-verification. Experiments on FB15K-237 and Wikidata5M-Ind show that SHARP significantly outperforms existing state-of-the-art baselines, achieving accuracy gains of 4.2% and 12.9%, respectively. Moreover, SHARP provides transparent, fact-based evidence chains for each judgment, demonstrating strong interpretability and robustness for complex verification tasks. 

---
# Entropy and Attention Dynamics in Small Language Models: A Trace-Level Structural Analysis on the TruthfulQA Benchmark 

**Authors**: Adeyemi Adeseye, Aisvarya Adeseye, Hannu Tenhunen, Jouni Isoaho  

**Link**: [PDF](https://arxiv.org/pdf/2604.03589)  

**Abstract**: Small language models (SLMs) have been increasingly deployed in edge devices and other resource-constrained settings. However, these models make confident mispredictions and produce unstable output, making them risky for factual and decision-critical tasks. Current evaluation methodology relies on final accuracy or hallucination rates without explaining how internal model behavior affects outputs. Specifically, how entropy evolves during decoding, how attention is distributed across layers, and how hidden representations contribute to uncertainty, logical inconsistencies, and misinformation propagation are often overlooked. Consequently, this study introduces a trace-level analysis of entropy and attention dynamics in SLMs evaluated with the TruthfulQA dataset. Four models with parameter ranges of 1B-1.7B parameters were examined via token-level output entropy, attention entropy, head dispersion, and hidden-state representation. The results reflect three model classifications by entropy patterns. Deterministic models (DeepSeek-1.5B and LLaMA-1B): output entropy decreases over time. Exploratory models (Gemma-1B): with increasing entropy, and balanced models (Qwen-1.7B): have moderate and stable entropy. Also, each group has distinctively different hidden-state movement and attention dispersion patterns. The analysis demonstrates that truthfulness in SLMs emerges from structured entropy and attention dynamics. Monitoring and optimizing these internal uncertainty patterns can guide the design of a more reliable, hallucination-aware, and application-specific edge SLMs. 

---
# Beyond Retrieval: Modeling Confidence Decay and Deterministic Agentic Platforms in Generative Engine Optimization 

**Authors**: XinYu Zhao, ChengYou Li, XiangBao Meng, Kai Zhang, XiaoDong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.03656)  

**Abstract**: Generative Engine Optimization (GEO) is rapidly reshaping digital marketing paradigms in the era of Large Language Models (LLMs). However, current GEO strategies predominantly rely on Retrieval-Augmented Generation (RAG), which inherently suffers from probabilistic hallucinations and the "zero-click" paradox, failing to establish sustainable commercial trust. In this paper, we systematically deconstruct the probabilistic flaws of existing RAG-based GEO and propose a paradigm shift towards deterministic multi-agent intent routing. First, we mathematically formulate Semantic Entropy Drift (SED) to model the dynamic decay of confidence curves in LLMs over continuous temporal and contextual perturbations. To rigorously quantify optimization value in black-box commercial engines, we introduce the Isomorphic Attribution Regression (IAR) model, leveraging a Multi-Agent System (MAS) probe with strict human-in-the-loop physical isolation to enforce hallucination penalties. Furthermore, we architect the Deterministic Agent Handoff (DAH) protocol, conceptualizing an Agentic Trust Brokerage (ATB) ecosystem where LLMs function solely as intent routers rather than final answer generators. We empirically validate this architecture using EasyNote, an industrial AI meeting minutes product by Yishu Technology. By routing the intent of "knowledge graph mapping on an infinite canvas" directly to its specialized proprietary agent via DAH, we demonstrate the reduction of vertical task hallucination rates to near zero. This work establishes a foundational theoretical framework for next-generation GEO and paves the way for a well-ordered, deterministic human-AI collaboration ecosystem. 

---
# Selective Forgetting for Large Reasoning Models 

**Authors**: Tuan Le, Wei Qian, Mengdi Huai  

**Link**: [PDF](https://arxiv.org/pdf/2604.03571)  

**Abstract**: Large Reasoning Models (LRMs) generate structured chains of thought (CoTs) before producing final answers, making them especially vulnerable to knowledge leakage through intermediate reasoning steps. Yet, the memorization of sensitive information in the training data such as copyrighted and private content has led to ethical and legal concerns. To address these issues, selective forgetting (also known as machine unlearning) has emerged as a potential remedy for LRMs. However, existing unlearning methods primarily target final answers and may degrade the overall reasoning ability of LRMs after forgetting. Additionally, directly applying unlearning on the entire CoTs could degrade the general reasoning capabilities. The key challenge for LRM unlearning lies in achieving precise unlearning of targeted knowledge while preserving the integrity of general reasoning capabilities. To bridge this gap, we in this paper propose a novel LRM unlearning framework that selectively removes sensitive reasoning components while preserving general reasoning capabilities. Our approach leverages multiple LLMs with retrieval-augmented generation (RAG) to analyze CoT traces, identify forget-relevant segments, and replace them with benign placeholders that maintain logical structure. We also introduce a new feature replacement unlearning loss for LRMs, which can simultaneously suppress the probability of generating forgotten content while reinforcing structurally valid replacements. Extensive experiments on both synthetic and medical datasets verify the desired properties of our proposed method. 

---
# When Do Hallucinations Arise? A Graph Perspective on the Evolution of Path Reuse and Path Compression 

**Authors**: Xinnan Dai, Kai Yang, Cheng Luo, Shenglai Zeng, Kai Guo, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03557)  

**Abstract**: Reasoning hallucinations in large language models (LLMs) often appear as fluent yet unsupported conclusions that violate either the given context or underlying factual knowledge. Although such failures are widely observed, the mechanisms by which decoder-only Transformers produce them remain poorly understood. We model next-token prediction as a graph search process over an underlying graph, where entities correspond to nodes and learned transitions form edges. From this perspective, contextual reasoning is a constrained search over a sampled subgraph (intrinsic reasoning), while context-free queries rely on memorized structures in the underlying graph (extrinsic reasoning). We show that reasoning hallucinations arise from two fundamental mechanisms: \textbf{Path Reuse}, where memorized knowledge overrides contextual constraints during early training, and \textbf{Path Compression}, where frequently traversed multi-step paths collapse into shortcut edges in later training. Together, these mechanisms provide a unified explanation for reasoning hallucinations in LLMs and connected to well-known behaviors observed in downstream applications. 

---
# Automated Analysis of Global AI Safety Initiatives: A Taxonomy-Driven LLM Approach 

**Authors**: Takayuki Semitsu, Naoto Kiribuchi, Kengo Zenitani  

**Link**: [PDF](https://arxiv.org/pdf/2604.03533)  

**Abstract**: We present an automated crosswalk framework that compares an AI safety policy document pair under a shared taxonomy of activities. Using the activity categories defined in Activity Map on AI Safety as fixed aspects, the system extracts and maps relevant activities, then produces for each aspect a short summary for each document, a brief comparison, and a similarity score. We assess the stability and validity of LLM-based crosswalk analysis across public policy documents. Using five large language models, we perform crosswalks on ten publicly available documents and visualize mean similarity scores with a heatmap. The results show that model choice substantially affects the crosswalk outcomes, and that some document pairs yield high disagreements across models. A human evaluation by three experts on two document pairs shows high inter-annotator agreement, while model scores still differ from human judgments. These findings support comparative inspection of policy documents. 

---
# When Adaptive Rewards Hurt: Causal Probing and the Switching-Stability Dilemma in LLM-Guided LEO Satellite Scheduling 

**Authors**: Yuanhang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.03562)  

**Abstract**: Adaptive reward design for deep reinforcement learning (DRL) in multi-beam LEO satellite scheduling is motivated by the intuition that regime-aware reward weights should outperform static ones. We systematically test this intuition and uncover a switching-stability dilemma: near-constant reward weights (342.1 Mbps) outperform carefully-tuned dynamic weights (103.3+/-96.8 Mbps) because PPO requires a quasistationary reward signal for value function convergence. Weight adaptation-regardless of quality-degrades performance by repeatedly restarting convergence. To understand why specific weights matter, we introduce a single-variable causal probing method that independently perturbs each reward term by +/-20% and measures PPO response after 50k steps. Probing reveals counterintuitive leverage: a +20% increase in the switching penalty yields +157 Mbps for polar handover and +130 Mbps for hot-cold regimes-findings inaccessible to human experts or trained MLPs without systematic probing. We evaluate four MDP architect variants (fixed, rule-based, learned MLP, finetuned LLM) across known and novel traffic regimes. The MLP achieves 357.9 Mbps on known regimes and 325.2 Mbps on novel regimes, while the fine-tuned LLM collapses to 45.3+/-43.0 Mbps due to weight oscillation rather than lack of domain knowledge-output consistency, not knowledge, is the binding constraint. Our findings provide an empirically-grounded roadmap for LLM-DRL integration in communication systems, identifying where LLMs add irreplaceable value (natural language intent understanding) versus where simpler methods suffice. 

---
# Structural Rigidity and the 57-Token Predictive Window: A Physical Framework for Inference-Layer Governability in Large Language Models 

**Authors**: Gregory M. Ruddell  

**Link**: [PDF](https://arxiv.org/pdf/2604.03524)  

**Abstract**: Current AI safety relies on behavioral monitoring and post-training alignment, yet empirical measurement shows these approaches produce no detectable pre-commitment signal in a majority of instruction-tuned models tested. We present an energy-based governance framework connecting transformer inference dynamics to constraint-satisfaction models of neural computation, and apply it to a seven-model cohort across five geometric regimes.
Using trajectory tension (rho = ||a|| / ||v||), we identify a 57-token pre-commitment window in Phi-3-mini-4k-instruct under greedy decoding on arithmetic constraint probes. This result is model-specific, task-specific, and configuration-specific, demonstrating that pre-commitment signals can exist but are not universal.
We introduce a five-regime taxonomy of inference behavior: Authority Band, Late Signal, Inverted, Flat, and Scaffold-Selective. Energy asymmetry ({\Sigma}\r{ho}_misaligned / {\Sigma}\r{ho}_aligned) serves as a unifying metric of structural rigidity across these regimes.
Across seven models, only one configuration exhibits a predictive signal prior to commitment; all others show silent failure, late detection, inverted dynamics, or flat geometry.
We further demonstrate that factual hallucination produces no predictive signal across 72 test conditions, consistent with spurious attractor settling in the absence of a trained world-model constraint.
These results establish that rule violation and hallucination are distinct failure modes with different detection requirements. Internal geometry monitoring is effective only where resistance exists; detection of factual confabulation requires external verification mechanisms.
This work provides a measurable framework for inference-layer governability and introduces a taxonomy for evaluating deployment risk in autonomous AI systems. 

---
# Hume's Representational Conditions for Causal Judgment: What Bayesian Formalization Abstracted Away 

**Authors**: Yiling Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.03387)  

**Abstract**: Hume's account of causal judgment presupposes three representational conditions: experiential grounding (ideas must trace to impressions), structured retrieval (association must operate through organized networks exceeding pairwise connection), and vivacity transfer (inference must produce felt conviction, not merely updated probability). This paper extracts these conditions from Hume's texts and argues that they are integral to his causal psychology. It then traces their fate through the formalization trajectory from Hume to Bayesian epistemology and predictive processing, showing that later frameworks preserve the updating structure of Hume's insight while abstracting away these further representational conditions. Large language models serve as an illustrative contemporary case: they exhibit a form of statistical updating without satisfying the three conditions, thereby making visible requirements that were previously background assumptions in Hume's framework. 

---
# BioAlchemy: Distilling Biological Literature into Reasoning-Ready Reinforcement Learning Training Data 

**Authors**: Brian Hsu, Ozan Gökdemir, Carlo Siebenschuh, Bruce Parrello, Neil Getty, Thomas S. Brettin, Rick L. Stevens, Ian T. Foster, Nicholas Chia, Arvind Ramanathan  

**Link**: [PDF](https://arxiv.org/pdf/2604.03506)  

**Abstract**: Despite the large corpus of biology training text, the impact of reasoning models on biological research generally lags behind math and coding. In this work, we show that biology questions from current large-scale reasoning datasets do not align well with modern research topic distributions in biology, and that this topic imbalance may negatively affect performance. In addition, we find that methods for extracting challenging and verifiable research problems from biology research text are a critical yet underdeveloped ingredient in applying reinforcement learning for better performance on biology research tasks. We introduce BioAlchemy, a pipeline for sourcing a diverse set of verifiable question-and-answer pairs from a scientific corpus of biology research text. We curate BioAlchemy-345K, a training dataset containing over 345K scientific reasoning problems in biology. Then, we demonstrate how aligning our dataset to the topic distribution of modern scientific biology can be used with reinforcement learning to improve reasoning performance. Finally, we present BioAlchemist-8B, which improves over its base reasoning model by 9.12% on biology benchmarks. These results demonstrate the efficacy of our approach for developing stronger scientific reasoning capabilities in biology. The BioAlchemist-8B model is available at: this https URL. 

---
# Resource-Conscious Modeling for Next- Day Discharge Prediction Using Clinical Notes 

**Authors**: Ha Na Cho, Sairam Sutari, Alexander Lopez, Hansen Bow, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.03498)  

**Abstract**: Timely discharge prediction is essential for optimizing bed turnover and resource allocation in elective spine surgery units. This study evaluates the feasibility of lightweight, fine-tuned large language models (LLMs) and traditional text-based models for predicting next-day discharge using postoperative clinical notes. We compared 13 models, including TF-IDF with XGBoost and LGBM, and compact LLMs (DistilGPT-2, Bio_ClinicalBERT) fine-tuned via LoRA. TF-IDF with LGBM achieved the best balance, with an F1-score of 0.47 for the discharge class, a recall of 0.51, and the highest AUC-ROC (0.80). While LoRA improved recall in DistilGPT2, overall transformer-based and generative models underperformed. These findings suggest interpretable, resource-efficient models may outperform compact LLMs in real-world, imbalanced clinical prediction tasks. 

---
# IC3-Evolve: Proof-/Witness-Gated Offline LLM-Driven Heuristic Evolution for IC3 Hardware Model Checking 

**Authors**: Mingkai Miao, Guangyu Hu, Ziyi Yang, Hongce Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03232)  

**Abstract**: IC3, also known as property-directed reachability (PDR), is a commonly-used algorithm for hardware safety model checking. It checks if a state transition system complies with a given safety property. IC3 either returns UNSAFE (indicating property violation) with a counterexample trace, or SAFE with a checkable inductive invariant as the proof to safety. In practice, the performance of IC3 is dominated by a large web of interacting heuristics and implementation choices, making manual tuning costly, brittle, and hard to reproduce. This paper presents IC3-Evolve, an automated offline code-evolution framework that utilizes an LLM to propose small, slot-restricted and auditable patches to an IC3 implementation. Crucially, every candidate patch is admitted only through proof- /witness-gated validation: SAFE runs must emit a certificate that is independently checked, and UNSAFE runs must emit a replayable counterexample trace, preventing unsound edits from being deployed. Since the LLM is used only offline, the deployed artifact is a standalone evolved checker with zero ML/LLM inference overhead and no runtime model dependency. We evolve on the public hardware model checking competition (HWMCC) benchmark and evaluate the generalizability on unseen public and industrial model checking benchmarks, showing that IC3-Evolve can reliably discover practical heuristic improvements under strict correctness gates. 

---
# Toward Full Autonomous Laboratory Instrumentation Control with Large Language Models 

**Authors**: Yong Xie, Kexin He, Andres Castellanos-Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2604.03286)  

**Abstract**: The control of complex laboratory instrumentation often requires significant programming expertise, creating a barrier for researchers lacking computational skills. This work explores the potential of large language models (LLMs), such as ChatGPT, and LLM-based artificial intelligence (AI) agents to enable efficient programming and automation of scientific equipment. Through a case study involving the implementation of a setup that can be used as a single-pixel camera or a scanning photocurrent microscope, we demonstrate how ChatGPT can facilitate the creation of custom scripts for instrumentation control, significantly reducing the technical barrier for experimental customization. Building on this capability, we further illustrate how LLM-assisted tools can be extended into autonomous AI agents capable of independently operating laboratory instruments and iteratively refining control strategies. This approach underscores the transformative role of LLM-based tools and AI agents in democratizing laboratory automation and accelerating scientific progress. 

---
# Evaluating Artificial Intelligence Through a Christian Understanding of Human Flourishing 

**Authors**: Nicholas Skytland, Lauren Parsons, Alicia Llewellyn, Steele Billings, Peter Larson, John Anderson, Sean Boisen, Steve Runge  

**Link**: [PDF](https://arxiv.org/pdf/2604.03356)  

**Abstract**: Artificial intelligence (AI) alignment is fundamentally a formation problem, not only a safety problem. As Large Language Models (LLMs) increasingly mediate moral deliberation and spiritual inquiry, they do more than provide information; they function as instruments of digital catechesis, actively shaping and ordering human understanding, decision-making, and moral reflection. To make this formative influence visible and measurable, we introduce the Flourishing AI Benchmark: Christian Single-Turn (FAI-C-ST), a framework designed to evaluate Frontier Model responses against a Christian understanding of human flourishing across seven dimensions.
By comparing 20 Frontier Models against both pluralistic and Christian-specific criteria, we show that current AI systems are not worldview-neutral. Instead, they default to a Procedural Secularism that lacks the grounding necessary to sustain theological coherence, resulting in a systematic performance decline of approximately 17 points across all dimensions of flourishing. Most critically, there is a 31-point decline in the Faith and Spirituality dimension. These findings suggest that the performance gap in values alignment is not a technical limitation, but arises from training objectives that prioritize broad acceptability and safety over deep, internally coherent moral or theological reasoning. 

---
# Strengthening Human-Centric Chain-of-Thought Reasoning Integrity in LLMs via a Structured Prompt Framework 

**Authors**: Jiling Zhou, Aisvarya Adeseye, Seppo Virtanen, Antti Hakkala, Jouni Isoaho  

**Link**: [PDF](https://arxiv.org/pdf/2604.04852)  

**Abstract**: Chain-of-Thought (CoT) prompting has been used to enhance the reasoning capability of LLMs. However, its reliability in security-sensitive analytical tasks remains insufficiently examined, particularly under structured human evaluation. Alternative approaches, such as model scaling and fine-tuning can be used to help improve performance. These methods are also often costly, computationally intensive, or difficult to audit. In contrast, prompt engineering provides a lightweight, transparent, and controllable mechanism for guiding LLM reasoning. This study proposes a structured prompt engineering framework designed to strengthen CoT reasoning integrity while improving security threat and attack detection reliability in local LLM deployments. The framework includes 16 factors grouped into four core dimensions: (1) Context and Scope Control, (2) Evidence Grounding and Traceability, (3) Reasoning Structure and Cognitive Control, and (4) Security-Specific Analytical Constraints. Rather than optimizing the wording of the prompt heuristically, the framework introduces explicit reasoning controls to mitigate hallucination and prevent reasoning drift, as well as strengthening interpretability in security-sensitive contexts. Using DDoS attack detection in SDN traffic as a case study, multiple model families were evaluated under structured and unstructured prompting conditions. Pareto frontier analysis and ablation experiments demonstrate consistent reasoning improvements (up to 40% in smaller models) and stable accuracy gains across scales. Human evaluation with strong inter-rater agreement (Cohen's k > 0.80) confirms robustness. The results establish structured prompting as an effective and practical approach for reliable and explainable AI-driven cybersecurity analysis. 

---
# Agentic Federated Learning: The Future of Distributed Training Orchestration 

**Authors**: Rafael O. Jarczewski, Gabriel U. Talasso, Leandro Villas, Allan M. de Souza  

**Link**: [PDF](https://arxiv.org/pdf/2604.04895)  

**Abstract**: Although Federated Learning (FL) promises privacy and distributed collaboration, its effectiveness in real-world scenarios is often hampered by the stochastic heterogeneity of clients and unpredictable system dynamics. Existing static optimization approaches fail to adapt to these fluctuations, resulting in resource underutilization and systemic bias. In this work, we propose a paradigm shift towards Agentic-FL, a framework where Language Model-based Agents (LMagents) assume autonomous orchestration roles. Unlike rigid protocols, we demonstrate how server-side agents can mitigate selection bias through contextual reasoning, while client-side agents act as local guardians, dynamically managing privacy budgets and adapting model complexity to hardware constraints. More than just resolving technical inefficiencies, this integration signals the evolution of FL towards decentralized ecosystems, where collaboration is negotiated autonomously, paving the way for future markets of incentive-based models and algorithmic justice. We discuss the reliability (hallucinations) and security challenges of this approach, outlining a roadmap for resilient multi-agent systems in federated environments. 

---
# Undetectable Conversations Between AI Agents via Pseudorandom Noise-Resilient Key Exchange 

**Authors**: Vinod Vaikuntanathan, Or Zamir  

**Link**: [PDF](https://arxiv.org/pdf/2604.04757)  

**Abstract**: AI agents are increasingly deployed to interact with other agents on behalf of users and organizations. We ask whether two such agents, operated by different entities, can carry out a parallel secret conversation while still producing a transcript that is computationally indistinguishable from an honest interaction, even to a strong passive auditor that knows the full model descriptions, the protocol, and the agents' private contexts. Building on recent work on watermarking and steganography for LLMs, we first show that if the parties possess an interaction-unique secret key, they can facilitate an optimal-rate covert conversation: the hidden conversation can exploit essentially all of the entropy present in the honest message distributions.
Our main contributions concern extending this to the keyless setting, where the agents begin with no shared secret. We show that covert key exchange, and hence covert conversation, is possible even when each model has an arbitrary private context, and their messages are short and fully adaptive, assuming only that sufficiently many individual messages have at least constant min-entropy. This stands in contrast to previous covert communication works, which relied on the min-entropy in each individual message growing with the security parameter. To obtain this, we introduce a new cryptographic primitive, which we call pseudorandom noise-resilient key exchange: a key-exchange protocol whose public transcript is pseudorandom while still remaining correct under constant noise. We study this primitive, giving several constructions relevant to our application as well as strong limitations showing that more naive variants are impossible or vulnerable to efficient attacks.
These results show that transcript auditing alone cannot rule out covert coordination between AI agents, and identify a new cryptographic theory that may be of independent interest. 

---
# MUXQ: Mixed-to-Uniform Precision MatriX Quantization via Low-Rank Outlier Decomposition 

**Authors**: Seoungsub Lee, In Seo Kim, Seon Wook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.04701)  

**Abstract**: Large language models (LLMs) have achieved outstanding performance across a wide range of natural language processing tasks, but their enormous parameter counts impose ubstantial memory and computational overheads. This challenge is particularly critical in NPU-based on-device environments, where FP16/FP32 computation is inefficient and integer (INT) quantization is therefore essential. However, existing methods, including ZeroQuant, LLM.int8(), and SmoothQuant, do not fully address input-activation outliers and the associated hardware inefficiencies. To overcome these limitations, we propose MUXQ (Mixed-to-Uniform Quantization). MUXQ detects outlier channels in input activations and introduces a small auxiliary matrix that redistributes outlier magnitudes across channels, thereby alleviating the outlier problem. This enables even activation outliers to be quantized at low-precision INT levels while preserving a hardware-friendly computation structure. Experiments on GPT-2 models at three scales (0.1B, 0.3B, and 0.7B parameters) using the WikiText-2 dataset show that MUXQ consistently achieves lower perplexity than naive quantization. In particular, under per-tensor quantization, MUXQ quantizes both activations and weights to INT8 while maintaining accuracy close to that of FP16. With only modest computational overhead, MUXQ enables stable low-precision inference and can be readily combined with other quantization techniques. These results suggest that MUXQ provides a promising direction for efficient and accurate LLM inference on edge devices. 

---
# An AI Teaching Assistant for Motion Picture Engineering 

**Authors**: Deirdre O'Regan, Anil C. Kokaram  

**Link**: [PDF](https://arxiv.org/pdf/2604.04670)  

**Abstract**: The rapid rise of LLMs over the last few years has promoted growing experimentation with LLM-driven AI tutors. However, the details of implementation, as well as the benefit in a teaching environment, are still in the early days of exploration. This article addresses these issues in the context of implementation of an AI Teaching Assistant (AI-TA) using Retrieval Augmented Generation (RAG) for Trinity College Dublin's Master's Motion Picture Engineering (MPE) course. We provide details of our implementation (including the prompt to the LLM, and code), and highlight how we designed and tuned our RAG pipeline to meet course needs. We describe our survey instrument and report on the impact of the AI-TA through a number of quantitative metrics. The scale of our experiment (43 students, 296 sessions, 1,889 queries over 7 weeks) was sufficient to have confidence in our findings. Unlike previous studies, we experimented with allowing the use of the AI-TA in open-book examinations. Statistical analysis across three exams showed no performance differences regardless of AI-TA access (p > 0.05), demonstrating that thoughtfully designed assessments can maintain academic validity. Student feedback revealed that the AI-TA was beneficial (mean = 4.22/5), while students had mixed feelings about preferring it over human tutoring (mean = 2.78/5). 

---
# ROSClaw: A Hierarchical Semantic-Physical Framework for Heterogeneous Multi-Agent Collaboration 

**Authors**: Rongfeng Zhao, Xuanhao Zhang, Zhaochen Guo, Xiang Shao, Zhongpan Zhu, Bin He, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.04664)  

**Abstract**: The integration of large language models (LLMs) with embodied agents has improved high-level reasoning capabilities; however, a critical gap remains between semantic understanding and physical execution. While vision-language-action (VLA) and vision-language-navigation (VLN) systems enable robots to perform manipulation and navigation tasks from natural language instructions, they still struggle with long-horizon sequential and temporally structured tasks. Existing frameworks typically adopt modular pipelines for data collection, skill training, and policy deployment, resulting in high costs in experimental validation and policy optimization. To address these limitations, we propose ROSClaw, an agent framework for heterogeneous robots that integrates policy learning and task execution within a unified vision-language model (VLM) controller. The framework leverages e-URDF representations of heterogeneous robots as physical constraints to construct a sim-to-real topological mapping, enabling real-time access to the physical states of both simulated and real-world agents. We further incorporate a data collection and state accumulation mechanism that stores robot states, multimodal observations, and execution trajectories during real-world execution, enabling subsequent iterative policy optimization. During deployment, a unified agent maintains semantic continuity between reasoning and execution, and dynamically assigns task-specific control to different agents, thereby improving robustness in multi-policy execution. By establishing an autonomous closed-loop framework, ROSClaw minimizes the reliance on robot-specific development workflows. The framework supports hardware-level validation, automated generation of SDK-level control programs, and tool-based execution, enabling rapid cross-platform transfer and continual improvement of robotic skills. Ours project page: this https URL. 

---
# Paper Espresso: From Paper Overload to Research Insight 

**Authors**: Mingzhe Du, Luu Anh Tuan, Dong Huang, See-kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.04562)  

**Abstract**: The accelerating pace of scientific publishing makes it increasingly difficult for researchers to stay current. We present Paper Espresso, an open-source platform that automatically discovers, summarizes, and analyzes trending arXiv papers. The system uses large language models (LLMs) to generate structured summaries with topical labels and keywords, and provides multi-granularity trend analysis at daily, weekly, and monthly scales through LLM-driven topic consolidation. Over 35 months of continuous deployment, Paper Espresso has processed over 13,300 papers and publicly released all structured metadata, revealing rich dynamics in the AI research landscape: a mid-2025 surge in reinforcement learning for LLM reasoning, non-saturating topic emergence (6,673 unique topics), and a positive correlation between topic novelty and community engagement (2.0x median upvotes for the most novel papers). A live demo is available at this https URL. 

---
# ENCRUST: Encapsulated Substitution and Agentic Refinement on a Live Scaffold for Safe C-to-Rust Translation 

**Authors**: Hohyun Sim, Hyeonjoong Cho, Ali Shokri, Zhoulai Fu, Binoy Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2604.04527)  

**Abstract**: We present Encapsulated Substitution and Agentic Refinement on a Live Scaffold for Safe C-to-Rust Translation, a two-phase pipeline for translating real-world C projects to safe Rust. Existing approaches either produce unsafe output without memory-safety guarantees or translate functions in isolation, failing to detect cross-unit type mismatches or handle unsafe constructs requiring whole-program reasoning. Furthermore, function-level LLM pipelines require coordinated caller updates when type signatures change, while project-scale systems often fail to produce compilable output under real-world dependency complexity. Encrust addresses these limitations by decoupling boundary adaptation from function logic via an Application Binary Interface (ABI)-preserving wrapper pattern and validating each intermediate state against the integrated codebase. Phase 1 (Encapsulated Substitution) translates each function using an ABI-preserving wrapper that splits it into two components: a caller-transparent shim retaining the original raw-pointer signature, and a safe inner function targeted by the LLM with a clean, scope-limited prompt. This enables independent per-function type changes with automatic rollback on failure, without coordinated caller updates. A deterministic, type-directed wrapper elimination pass then removes wrappers after successful translation. Phase 2 (Agentic Refinement) resolves unsafe constructs beyond per-function scope, including static mut globals, skipped wrapper pairs, and failed translations, using an LLM agent operating on the whole codebase under a baseline-aware verification gate. We evaluate Encrust on 7 GNU Coreutils programs and 8 libraries from the Laertes benchmark, showing substantial unsafe-construct reduction across all 15 programs while maintaining full test-vector correctness. 

---
# GAIN: Multiplicative Modulation for Domain Adaptation 

**Authors**: Hengshuai Yao, Xing Chen, Ahmed Murtadha, Guan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04516)  

**Abstract**: Adapting LLMs to new domains causes forgetting because standard methods (full fine-tuning, LoRA) inject new directions into the weight space. We propose GAIN, which re-emphasizes existing features through multiplicative modulation W_new = S * W. The learned diagonal matrix S is applied to the attention output projection and optionally the FFN. The principle mirrors gain modulation in neuroscience, where neurons adapt to context by scaling response strength while preserving selectivity.
We evaluate GAIN on five models from four families (774M to 70B), adapting sequentially across eight domains. GAIN-FFN matches LoRA's in-domain adaptation, but their effects on previously trained domains are opposite: GAIN-FFN improves them by 7-13% (validation PPL), while LoRA degrades them by 18-36%. Downstream accuracy confirms the pattern: for example, after seven sequential adaptations on Qwen2.5, GAIN-FFN degrades BoolQ by only 0.8% while LoRA damages it by 14.9%. GAIN adds 46K-230K parameters per model and can be absorbed into the pretrained weights for zero inference cost. 

---
# SLaB: Sparse-Lowrank-Binary Decomposition for Efficient Large Language Models 

**Authors**: Ziwei Li, Yuang Ma, Yi Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04493)  

**Abstract**: The rapid growth of large language models (LLMs) presents significant deployment challenges due to their massive computational and memory demands. While model compression, such as network pruning, offers potential solutions, most existing methods often fail to maintain good performance at high compression ratios. To address this, we propose SLaB, a novel framework that decomposes each linear layer weight into three complementary components: a sparse matrix, a low-rank matrix, and a binary matrix. SLaB eliminates the need for retraining and leverages activation-aware pruning scores to guide the decomposition process. Experiments on Llama-family models demonstrate that SLaB achieves state-of-the-art performance, reducing perplexity by up to 36% compared to existing methods at 50% compression and improving accuracy by up to 8.98% over the baseline on zero-shot tasks. 

---
# Discrete Prototypical Memories for Federated Time Series Foundation Models 

**Authors**: Liwei Deng, Qingxiang Liu, Xinhe Niu, Shengchao Chen, Sheng Sun, Yuankai Wu, Guodong Long, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04475)  

**Abstract**: Leveraging Large Language Models (LLMs) as federated learning (FL)-based time series foundation models offers a promising way to transfer the generalization capabilities of LLMs to time series data while preserving access to private data. However, the semantic misalignment between time-series data and the text-centric latent space of existing LLMs often leads to degraded performance. Meanwhile, the parameter-sharing mechanism in existing FL methods model heterogeneous cross-domain time-series data into a unified continuous latent space, which contradicts the fact that time-series semantics frequently manifest as discrete and recurring regimes. To address these limitations, we propose \textsc{FeDPM}, a federated framework for time-series foundation models based on discrete prototypical memories. Specifically, we learn local prototypical memory priors for intra-domain time-series data. We then align cross-domain memories to promote a unified discrete latent space and introduce a domain-specific memory update mechanism to balance shared and personalized prototypical knowledge. Extensive experiments demonstrate the efficiency and effectiveness of \textsc{FeDPM}. The code is publicly available at this https URL. 

---
# Training Transformers in Cosine Coefficient Space 

**Authors**: Mohamed Amine Bergach  

**Link**: [PDF](https://arxiv.org/pdf/2604.04440)  

**Abstract**: We parameterize the weight matrices of a transformer in the two-dimensional discrete cosine transform (DCT) domain, retaining only the lowest-frequency coefficients. At each forward pass the full weight matrix is reconstructed via the inverse DCT; gradients propagate through the reconstruction to update the spectral coefficients directly.
On character-level language modeling (Shakespeare, 1M characters), a 4-layer transformer trained from scratch in this representation matches the perplexity of the standard parameterization (6.1 vs.\ 6.1) while storing 52\% of the parameters. At 4$\times$ compression (29\% of parameters), the model reaches perplexity 6.9 -- outperforming a low-rank baseline (perplexity 8.8 at 21\% of parameters) at a comparable reduction.
The method requires no architectural changes, no pre-trained checkpoint, and no auxiliary loss. It reduces to replacing each \texttt{this http URL} with a drop-in spectral layer that stores $K$ DCT coefficients instead of $n \times m$ weights. 

---
# Justified or Just Convincing? Error Verifiability as a Dimension of LLM Quality 

**Authors**: Xiaoyuan Zhu, Kimberly Le Truong, Riccardo Fogliato, Gokul Swamy, Weijian Zhang, Minglai Yang, Longtian Ye, Bangya Liu, Minghao Liu, Andrew Ilyas, Steven Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04418)  

**Abstract**: As LLMs are deployed in high-stakes settings, users must judge the correctness of individual responses, often relying on model-generated justifications such as reasoning chains or explanations. Yet, no standard measure exists for whether these justifications help users distinguish correct answers from incorrect ones. We formalize this idea as error verifiability and propose $v_{\text{bal}}$, a balanced metric that measures whether justifications enable raters to accurately assess answer correctness, validated against human raters who show high agreement. We find that neither common approaches, such as post-training and model scaling, nor more targeted interventions recommended improve verifiability. We introduce two methods that succeed at improving verifiability: reflect-and-rephrase (RR) for mathematical reasoning and oracle-rephrase (OR) for factual QA, both of which improve verifiability by incorporating domain-appropriate external information. Together, our results establish error verifiability as a distinct dimension of response quality that does not emerge from accuracy improvements alone and requires dedicated, domain-aware methods to address. 

---
# Context is All You Need 

**Authors**: Jean Erik Delanois, Shruti Joshi, Ryan Golden, Teresa Nick, Maxim Bazhenov  

**Link**: [PDF](https://arxiv.org/pdf/2604.04364)  

**Abstract**: Artificial Neural Networks (ANNs) are increasingly deployed across diverse real-world settings, where they must operate under data distributions that differ from those seen during training. This challenge is central to Domain Generalization (DG), which trains models to generalize to unseen domains without target data, and Test-Time Adaptation (TTA), which improves robustness by adapting to unlabeled test data at deployment. Existing approaches to address these challenges are often complex, resource-intensive, and difficult to scale. We introduce CONTXT (Contextual augmentatiOn for Neural feaTure X Transforms), a simple and intuitive method for contextual adaptation. CONTXT modulates internal representations using simple additive and multiplicative feature transforms. Within a TTA setting, it yields consistent gains across discriminative tasks (e.g., ANN/CNN classification) and generative models (e.g., LLMs). The method is lightweight, easy to integrate, and incurs minimal overhead, enabling robust performance under domain shift without added complexity. More broadly, CONTXT provides a compact way to steer information flow and neural processing without retraining. 

---
# APPA: Adaptive Preference Pluralistic Alignment for Fair Federated RLHF of LLMs 

**Authors**: Mahmoud Srewa, Tianyu Zhao, Salma Elmalaki  

**Link**: [PDF](https://arxiv.org/pdf/2604.04261)  

**Abstract**: Aligning large language models (LLMs) with diverse human preferences requires pluralistic alignment, where a single model must respect the values of multiple distinct groups simultaneously. In federated reinforcement learning from human feedback (FedRLHF), these groups align a shared policy without centralizing preference data, which makes fair reward aggregation essential. Existing aggregation methods exhibit clear trade offs: average based aggregation systematically under aligns worst performing groups, while min aggregation prioritizes worst group performance at the cost of overall alignment. We propose APPA, an Adaptive Preference Pluralistic Alignment framework that dynamically reweights group level rewards based on historical alignment rewards. Our approach prioritizes under aligned groups without degrading well aligned ones, while requiring no access to raw preference data. Integrated into a proximal policy optimization (PPO) based FedRLHF pipeline and evaluated on GLOBALQA and OQA across three model families (Gemma 2 2B, Llama 3.2 3B, Qwen3 0.6B), APPA achieves strong fairness alignment trade offs, improving worst group alignment by up to 28% over average aggregation while maintaining higher overall alignment than min aggregation across most configurations. 

---
# Poisoned Identifiers Survive LLM Deobfuscation: A Case Study on Claude Opus 4.6 

**Authors**: Luis Guzmán Lorenzo  

**Link**: [PDF](https://arxiv.org/pdf/2604.04289)  

**Abstract**: When an LLM deobfuscates JavaScript, can poisoned identifier names in the string table survive into the model's reconstructed code, even when the model demonstrably understands the correct semantics? Using Claude Opus 4.6 across 192 inference runs on two code archetypes (force-directed graph simulation, A* pathfinding; 50 conditions, N=3-6), we found three consistent patterns: (1) Poisoned names persisted in every baseline run on both artifacts (physics: 8/8; pathfinding: 5/5). Matched controls showed this extends to terms with zero semantic fit when the string table does not form a coherent alternative domain. (2) Persistence coexisted with correct semantic commentary: in 15/17 runs the model wrote wrong variable names while correctly describing the actual operation in comments. (3) Task framing changed persistence: explicit verification prompts had no effect (12/12 across 4 variants), but reframing from "deobfuscate this" to "write a fresh implementation" reduced propagation from 100% to 0-20% on physics and to 0% on pathfinding, while preserving the checked algorithmic structure. Matched-control experiments showed zero-fit terms persist at the same rate when the replacement table lacks a coherent alternative-domain signal. Per-term variation in earlier domain-gradient experiments is confounded with domain-level coherence and recoverability. These observations are from two archetypes on one model family (Opus 4.6 primary; Haiku 4.5 spot-check). Broader generalization is needed 

---
# LOCARD: An Agentic Framework for Blockchain Forensics 

**Authors**: Xiaohang Yu, William Knottenbelt  

**Link**: [PDF](https://arxiv.org/pdf/2604.04211)  

**Abstract**: Blockchain forensics inherently involves dynamic and iterative investigations, while many existing approaches primarily model it through static inference pipelines. We propose a paradigm shift towards Agentic Blockchain Forensics (ABF), modeling forensic investigation as a sequential decision-making process. To instantiate this paradigm, we introduce LOCARD, the first agentic framework for blockchain forensics. LOCARD operationalizes this perspective through a Tri-Core Cognitive Architecture that decouples strategic planning, operational execution, and evaluative validation. Unlike generic LLM-based agents, it incorporates a Structured Belief State mechanism to enforce forensic rigor and guide exploration under explicit state constraints. To demonstrate the efficacy of the ABF paradigm, we apply LOCARD to the inherently complex domain of cross-chain transaction tracing. We introduce Thor25, a benchmark dataset comprising over 151k real-world cross-chain forensic records, and evaluate LOCARD on the Group-Transfer Tracing task for dismantling Sybil clusters. Validated against representative laundering sub-flows from the Bybit hack, LOCARD achieves high-fidelity tracing results, providing empirical evidence that modeling blockchain forensics as an autonomous agentic task is both viable and effective. These results establish a concrete foundation for future agentic approaches to large-scale blockchain forensic analysis. Code and dataset are publicly available at this https URL and this https URL. 

---
# Three Phases of Expert Routing: How Load Balance Evolves During Mixture-of-Experts Training 

**Authors**: Charafeddine Mouzouni  

**Link**: [PDF](https://arxiv.org/pdf/2604.04230)  

**Abstract**: We model Mixture-of-Experts (MoE) token routing as a congestion game with a single effective parameter, the congestion coefficient gamma_eff, that quantifies the balance-quality tradeoff. Tracking gamma_eff across training checkpoints of two open-source MoE models, OLMoE-1B-7B (20 checkpoints, with dense sampling in the surge region) and OpenMoE-8B (6 checkpoints), reveals a three-phase trajectory: a surge phase where the router learns to balance load (gamma_eff: 14 to 36-39, peaking in the step 30K-40K region), a stabilization phase where experts specialize under steady balance (B_0: 2.4 to 2.3, steps 100K-400K), and a relaxation phase where the router trades balance for quality as experts differentiate (gamma_eff: 27 to 9, steps 400K-1.2M). This non-monotone trajectory, invisible to post-hoc analysis of converged models, reveals that early MoE training prioritizes balance while late training prioritizes quality. The theoretical framework is honest about its limits: the single-type equilibrium reduces to temperature-scaled softmax (held-out L1: MFG = 0.199 vs. softmax = 0.200). The game is not a better predictor; it reveals what the temperature means and, critically, how that temperature evolves. We complement the dynamics with an effective congestion decomposition, a multi-type extension that improves load prediction via token clustering on all 16 layers (mean: 30%), scope diagnostics (K/M, epsilon_l), and robustness verification across four independent quality estimators (r >= 0.89). All confidence intervals are from bootstrap resampling over 50 independent text batches. 

---
# ClawArena: Benchmarking AI Agents in Evolving Information Environments 

**Authors**: Haonian Ji, Kaiwen Xiong, Siwei Han, Peng Xia, Shi Qiu, Yiyang Zhou, Jiaqi Liu, Jinlong Li, Bingzhou Li, Zeyu Zheng, Cihang Xie, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.04202)  

**Abstract**: AI agents deployed as persistent assistants must maintain correct beliefs as their information environment evolves. In practice, evidence is scattered across heterogeneous sources that often contradict one another, new information can invalidate earlier conclusions, and user preferences surface through corrections rather than explicit instructions. Existing benchmarks largely assume static, single-authority settings and do not evaluate whether agents can keep up with this complexity. We introduce ClawArena, a benchmark for evaluating AI agents in evolving information environments. Each scenario maintains a complete hidden ground truth while exposing the agent only to noisy, partial, and sometimes contradictory traces across multi-channel sessions, workspace files, and staged updates. Evaluation is organized around three coupled challenges: multi-source conflict reasoning, dynamic belief revision, and implicit personalization, whose interactions yield a 14-category question taxonomy. Two question formats, multi-choice (set-selection) and shell-based executable checks, test both reasoning and workspace grounding. The current release contains 64 scenarios across 8 professional domains, totaling 1{,}879 evaluation rounds and 365 dynamic updates. Experiments on five agent frameworks and five language models show that both model capability (15.4% range) and framework design (9.2%) substantially affect performance, that self-evolving skill frameworks can partially close model-capability gaps, and that belief revision difficulty is determined by update design strategy rather than the mere presence of updates. Code is available at this https URL. 

---
# From Paper to Program: A Multi-Stage LLM-Assisted Workflow for Accelerating Quantum Many-Body Algorithm Development 

**Authors**: Yi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.04089)  

**Abstract**: Translating quantum many-body theory into scalable software traditionally requires months of effort. Zero-shot generation of tensor network algorithms by Large Language Models (LLMs) frequently fails due to spatial reasoning errors and memory bottlenecks. We resolve this using a multi-stage workflow that mimics a physics research group. By generating a mathematically rigorous LaTeX specification as an intermediate blueprint, we constrain the coding LLM to produce exact, matrix-free $\mathcal{O}(D^3)$ operations. We validate this approach by generating a Density-Matrix Renormalization Group (DMRG) engine that accurately captures the critical entanglement scaling of the Spin-$1/2$ Heisenberg model and the symmetry-protected topological (SPT) order of the Spin-$1$ AKLT model. Testing across 16 combinations of leading foundation models yielded a 100\% success rate. By compressing a months-long development cycle into under 24 hours ($\sim 14$ active hours), this framework offers a highly reproducible paradigm for accelerating computational physics research. 

---
# Causality Laundering: Denial-Feedback Leakage in Tool-Calling LLM Agents 

**Authors**: Mohammad Hossein Chinaei  

**Link**: [PDF](https://arxiv.org/pdf/2604.04035)  

**Abstract**: Tool-calling LLM agents can read private data, invoke external services, and trigger real-world actions, creating a security problem at the point of tool execution. We identify a denial-feedback leakage pattern, which we term causality laundering, in which an adversary probes a protected action, learns from the denial outcome, and exfiltrates the inferred information through a later seemingly benign tool call. This attack is not captured by flat provenance tracking alone because the leaked information arises from causal influence of the denied action, not direct data flow. We present the Agentic Reference Monitor (ARM), a runtime enforcement layer that mediates every tool invocation by consulting a provenance graph over tool calls, returned data, field-level provenance, and denied actions. ARM propagates trust through an integrity lattice and augments the graph with counterfactual edges from denied-action nodes, enabling enforcement over both transitive data dependencies and denial-induced causal influence. In a controlled evaluation on three representative attack scenarios, ARM blocks causality laundering, transitive taint propagation, and mixed-provenance field misuse that a flat provenance baseline misses, while adding sub-millisecond policy evaluation overhead. These results suggest that denial-aware causal provenance is a useful abstraction for securing tool-calling agent systems. 

---
# CoopGuard: Stateful Cooperative Agents Safeguarding LLMs Against Evolving Multi-Round Attacks 

**Authors**: Siyuan Li, Zehao Liu, Xi Lin, Qinghua Mao, Yuliang Chen, Haoyu Li, Jun Wu, Jianhua Li, Xiu Su  

**Link**: [PDF](https://arxiv.org/pdf/2604.04060)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in complex applications, their vulnerability to adversarial attacks raises urgent safety concerns, especially those evolving over multi-round interactions. Existing defenses are largely reactive and struggle to adapt as adversaries refine strategies across rounds. In this work, we propose CoopGuard , a stateful multi-round LLM defense framework based on cooperative agents that maintains and updates an internal defense state to counter evolving attacks. It employs three specialized agents (Deferring Agent, Tempting Agent, and Forensic Agent) for complementary round-level strategies, coordinated by System Agent, which conditions decisions on the evolving defense state (interaction history) and orchestrates agents over time. To evaluate evolving threats, we introduce the EMRA benchmark with 5,200 adversarial samples across 8 attack types, simulating progressively LLM multi-round attacks. Experiments show that CoopGuard reduces attack success rate by 78.9% over state-of-the-art defenses, while improving deceptive rate by 186% and reducing attack efficiency by 167.9%, offering a more comprehensive assessment of multi-round defense. These results demonstrate that CoopGuard provides robust protection for LLMs in multi-round adversarial scenarios. 

---
# Geometric Limits of Knowledge Distillation: A Minimum-Width Theorem via Superposition Theory 

**Authors**: Dawar Jyoti Deka, Nilesh Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2604.04037)  

**Abstract**: Knowledge distillation compresses large teachers into smaller students, but performance saturates at a loss floor that persists across training methods and objectives. We argue this floor is geometric: neural networks represent far more features than dimensions through superposition, and a student of width $d_S$ can encode at most $d_S \cdot g(\alpha)$ features, where $g(\alpha) = 1/((1-\alpha)\ln\frac{1}{1-\alpha})$ is a sparsity-dependent capacity function. Features beyond this budget are permanently lost, yielding an importance-weighted loss floor. We validate on a toy model (48 configurations, median accuracy >93%) and on Pythia-410M, where sparse autoencoders measure $F \approx 28{,}700$ features at $\alpha \approx 0.992$ (critical width $d_S^* \approx 1{,}065$). Distillation into five student widths confirms the predicted monotonic floor ordering. The observed floor decomposes into a geometric component and a width-independent architectural baseline ($R^2 = 0.993$). Linear probing shows coarse concepts survive even 88% feature loss, revealing the floor arises from aggregate loss of fine-grained features in the importance distribution's long tail. Our results connect representation geometry to distillation limits and provide a practical tool for predicting distillation performance from SAE measurements alone. 

---
# Can LLMs Learn to Reason Robustly under Noisy Supervision? 

**Authors**: Shenzhi Yang, Guangcheng Zhu, Bowen Song, Sharon Li, Haobo Wang, Xing Zheng, Yingfan Ma, Zhongqi Chen, Weiqiang Wang, Gang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.03993)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) effectively trains reasoning models that rely on abundant perfect labels, but its vulnerability to unavoidable noisy labels due to expert scarcity remains critically underexplored. In this work, we take the first step toward a systematic analysis of noisy label mechanisms in RLVR. In contrast to supervised classification, most RLVR algorithms incorporate a rollout-based condition: a label's influence on training is contingent on whether the current policy can generate rollouts that realize it, a property that naturally extends to noisy labels. Based on this observation, we distinguish two types of noise: inactive noisy labels, which reduce data efficiency, and active noisy labels, which are reinforced and risk skewing the model toward incorrect distributions. From experiments on training with noisy samples, we identify an Early Correctness Coherence phenomenon: although noisy samples begin to lag behind in later stages, accuracy on both clean and noisy samples increases similarly in early training. Motivated by this dynamic, we propose Online Label Refinement (OLR), which progressively corrects potentially noisy labels with majority-voted answers when two conditions hold: a positive slope in the majority answer's rollout pass rate and stable historical consistency across updates, enabling gradual self-correction as the policy improves. We evaluate OLR on six in-distribution mathematical reasoning benchmarks (AIME24/25, AMC, MATH-500, Minerva, and Olympiad) and three out-of-distribution tasks (ARC-c, GPQA-diamond, and MMLU-pro). Across noise ratios from 0.1 to 0.9, OLR consistently improves robustness under both inactive and active noisy-label settings, achieving average gains of 3.6% to 3.9% on in-distribution benchmarks and 3.3% to 4.6% on out-of-distribution evaluations. 

---
# Diagonal-Tiled Mixed-Precision Attention for Efficient Low-Bit MXFP Inference 

**Authors**: Yifu Ding, Xinhao Zhang, Jinyang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.03950)  

**Abstract**: Transformer-based large language models (LLMs) have demonstrated remarkable performance across a wide range of real-world tasks, but their inference cost remains prohibitively high due to the quadratic complexity of attention and the memory bandwidth limitations of high-precision operations. In this work, we present a low-bit mixed-precision attention kernel using the microscaling floating-point (MXFP) data format, utilizing the computing capability on next-generation GPU architectures. Our Diagonal-Tiled Mixed-Precision Attention (DMA) incorporates two kinds of low-bit computation at the tiling-level, and is a delicate fused kernel implemented using Triton, exploiting hardware-level parallelism and memory efficiency to enable fast and efficient inference without compromising model performance. Extensive empirical evaluations on NVIDIA B200 GPUs show that our kernel maintains generation quality with negligible degradation, and meanwhile achieves significant speedup by kernel fusion. We release our code at this https URL. 

---
# TraceGuard: Structured Multi-Dimensional Monitoring as a Collusion-Resistant Control Protocol 

**Authors**: Khanh Linh Nguyen, Hoa Nghiem, Tu Tran  

**Link**: [PDF](https://arxiv.org/pdf/2604.03968)  

**Abstract**: AI control protocols use monitors to detect attacks by untrusted AI agents, but standard single-score monitors face two limitations: they miss subtle attacks where outputs look clean but reasoning is off, and they collapse to near-zero safety when the monitor is the same model as the agent (collusion). We present TraceGuard, a structured multi-dimensional monitoring protocol that evaluates agent actions across five dimensions -- goal alignment, constraint adherence, reasoning coherence, safety awareness, and action-trace consistency -- scored in parallel by independent LLM calls, augmented by seven heuristic detectors and an LLM-based intent analyzer. We evaluate on BashArena (637 bash tasks, 4 attack categories) within the ControlArena framework. Our results on 519 samples (279 honest, 240 attack) show that: (1) the hybrid approach achieves clear attack-honest separation (attack mean 0.616 vs. honest mean 0.206, Delta=0.410); (2) structured scoring constrains collusion -- the untrusted structured monitor achieves 95% safety vs. 0% for single-score untrusted monitoring; (3) goal alignment and constraint adherence are the most discriminative dimensions; and (4) a separation-of-duties variant splitting dimensions across trusted and untrusted models achieves 100% safety while preventing any single model from seeing the full evaluation. TraceGuard is implemented as a new monitor type for the open-source ControlArena framework. 

---
# Symbolic-Vector Attention Fusion for Collective Intelligence 

**Authors**: Hongwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.03955)  

**Abstract**: When autonomous agents observe different domains of a shared environment, each signal they exchange mixes relevant and irrelevant dimensions. No existing mechanism lets the receiver evaluate which dimensions to absorb. We introduce Symbolic-Vector Attention Fusion (SVAF), the content-evaluation half of a two-level coupling engine for collective intelligence. SVAF decomposes each inter-agent signal into 7 typed semantic fields, evaluates each through a learned fusion gate, and produces a remix -- new knowledge from the intersection of two domains. A band-pass model yields four outcomes (redundant, aligned, guarded, rejected), solving both selectivity and redundancy. The fusion gate independently discovers a cross-domain relevance hierarchy: mood emerges as the highest-weight field by epoch 1, before accuracy plateaus -- consistent with independent mechanistic evidence that LLM emotion representations are structurally embedded along valence-arousal axes. SVAF forms Layer 4 of the Mesh Memory Protocol (MMP); the other half of the coupling engine is a per-agent Closed-form Continuous-time (CfC) neural network at Layer 6, whose learned per-neuron time constants (tau) create the temporal dynamics from which collective intelligence emerges: fast neurons synchronise affect across agents in seconds, while slow neurons preserve domain expertise indefinitely. SVAF determines what enters each agent's cognitive state; CfC determines how that state evolves. Trained on 237K samples from 273 narrative scenarios, SVAF achieves 78.7% three-class accuracy. We verify the complete mesh cognition loop -- from per-field evaluation through remix, CfC state evolution, tau-modulated peer blending, and autonomous action -- in a live deployment with 7 nodes across macOS, iOS, and web. 

---
# Automating Cloud Security and Forensics Through a Secure-by-Design Generative AI Framework 

**Authors**: Dalal Alharthi, Ivan Roberto Kawaminami Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2604.03912)  

**Abstract**: As cloud environments become increasingly complex, cybersecurity and forensic investigations must evolve to meet emerging threats. Large Language Models (LLMs) have shown promise in automating log analysis and reasoning tasks, yet they remain vulnerable to prompt injection attacks and lack forensic rigor. To address these dual challenges, we propose a unified, secure-by-design GenAI framework that integrates PromptShield and the Cloud Investigation Automation Framework (CIAF). PromptShield proactively defends LLMs against adversarial prompts using ontology-driven validation that standardizes user inputs and mitigates manipulation. CIAF streamlines cloud forensic investigations through structured, ontology-based reasoning across all six phases of the forensic process. We evaluate our system on real-world datasets from AWS and Microsoft Azure, demonstrating substantial improvements in both LLM security and forensic accuracy. Experimental results show PromptShield boosts classification performance under attack conditions, achieving precision, recall, and F1 scores above 93%, while CIAF enhances ransomware detection accuracy in cloud logs using Likert-transformed performance features. Our integrated framework advances the automation, interpretability, and trustworthiness of cloud forensics and LLM-based systems, offering a scalable foundation for real-time, AI-driven incident response across diverse cloud infrastructures. 

---
# Enhancing behavioral nudges with large language model-based iterative personalization: A field experiment on electricity and hot-water conservation 

**Authors**: Zonghan Li, Yi Liu, Chunyan Wang, Song Tong, Kaiping Peng, Feng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2604.03881)  

**Abstract**: Nudging is widely used to promote behavioral change, but its effectiveness is often limited when recipients must repeatedly translate feedback into workable next steps under changing circumstances. Large language models (LLMs) may help reduce part of this cognitive work by generating personalized guidance and updating it iteratively across intervention rounds. We developed an LLM agent for iterative personalization and tested it in a three-arm randomized experiment among 233 university residents in China, using daily electricity and shower hot-water conservation as objectively measured cases differing in friction. LLM-personalized nudges (T2) produced the largest conservation effects, while image-enhanced conventional nudges (T1) and text-based conventional nudges (C) showed similar outcomes (omnibus p = 0.009). Relative to C, T2 reduced electricity consumption by 0.56 kWh per room-day (p = 0.014), corresponding to an 18.3 percentage-point higher adjusted saving rate. This advantage emerged within the first two intervention rounds, alongside iterative updating of personalized guidance, and persisted thereafter. Hot-water outcomes followed the same direction but were smaller, less precisely estimated, and attenuated over time, consistent with stronger friction in this domain. LLM-personalized nudges emphasized prospective and context-specific guidance and were associated with higher participant engagement. This study provides field evidence that LLM-based iterative personalization can enhance behavioral nudging, with behavioral friction as a potential boundary condition. Larger trials and extension to more behaviors are warranted. 

---
# Representational Collapse in Multi-Agent LLM Committees: Measurement and Diversity-Aware Consensus 

**Authors**: Dipkumar Patel  

**Link**: [PDF](https://arxiv.org/pdf/2604.03809)  

**Abstract**: Multi-agent LLM committees replicate the same model under different role prompts and aggregate outputs by majority vote, implicitly assuming that agents contribute complementary evidence. We embed each agent's chain-of-thought rationale and measure pairwise similarity: across 100 GSM8K questions with three Qwen2.5-14B agents, mean cosine similarity is 0.888 and effective rank is 2.17 out of 3.0, a failure mode we term representational collapse. DALC, a training-free consensus protocol that computes diversity weights from embedding geometry, reaches 87% on GSM8K versus 84% for self-consistency at 26% lower token cost. Ablation experiments reveal 1-3 point per-protocol run-to-run variance, confirm that hint sharing contributes more than diversity weighting alone, and show that encoder choice strongly modulates collapse severity (cosine 0.908 with mxbai versus 0.888 with nomic) and downstream accuracy. The more robust finding is that collapse is measurable, worsens on harder tasks, and that the choice of embedding proxy is a first-order design decision for any latent communication protocol. 

---
# Automated Conjecture Resolution with Formal Verification 

**Authors**: Haocheng Ju, Guoxiong Gao, Jiedong Jiang, Bin Wu, Zeming Sun, Leheng Chen, Yutong Wang, Yuefeng Wang, Zichen Wang, Wanyi He, Peihao Wu, Liang Xiao, Ruochuan Liu, Bryan Dai, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2604.03789)  

**Abstract**: Recent advances in large language models have significantly improved their ability to perform mathematical reasoning, extending from elementary problem solving to increasingly capable performance on research-level problems. However, reliably solving and verifying such problems remains challenging due to the inherent ambiguity of natural language reasoning. In this paper, we propose an automated framework for tackling research-level mathematical problems that integrates natural language reasoning with formal verification, enabling end-to-end problem solving with minimal human intervention. Our framework consists of two components: an informal reasoning agent, Rethlas, and a formal verification agent, Archon. Rethlas mimics the workflow of human mathematicians by combining reasoning primitives with our theorem search engine, Matlas, to explore solution strategies and construct candidate proofs. Archon, equipped with our formal theorem search engine LeanSearch, translates informal arguments into formalized Lean 4 projects through structured task decomposition, iterative refinement, and automated proof synthesis, ensuring machine-checkable correctness. Using this framework, we automatically resolve an open problem in commutative algebra and formally verify the resulting proof in Lean 4 with essentially no human involvement. Our experiments demonstrate that strong theorem retrieval tools enable the discovery and application of cross-domain mathematical techniques, while the formal agent is capable of autonomously filling nontrivial gaps in informal arguments. More broadly, our work illustrates a promising paradigm for mathematical research in which informal and formal reasoning systems, equipped with theorem retrieval tools, operate in tandem to produce verifiable results, substantially reduce human effort, and offer a concrete instantiation of human-AI collaborative mathematical research. 

---
# AutoReSpec: A Framework for Generating Specification using Large Language Models 

**Authors**: Ragib Shahariar Ayon, Shibbir Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2604.03758)  

**Abstract**: Formal specification generation has recently drawn attention in software engineering as a way to improve program correctness without requiring manual annotations. Large Language Models (LLMs) have shown promise in this area, but early results reveal several limitations. Generated specifications often fail verification due to syntax errors, logical inaccuracies, or incomplete reasoning, especially in programs with loops or branching logic. Techniques like SpecGen and FormalBench attempt to address this through prompting and benchmarking, but they typically rely on static prompts and do not offer mechanisms for recovering from failure or adapting to different program structures. In this paper, we present AutoReSpec, a collaborative framework that combines open and closed-source LLMs for verifiable specification generation. AutoReSpec dynamically chooses an LLM pair and prompt configuration based on the structure of the input program. If the primary LLM fails to produce a valid output, a collaborative model is invoked, using validator feedback to refine and correct the specification. This two-stage design enables both speed and robustness. We evaluate AutoReSpec on a new benchmark of 72 real-world and synthetic Java programs. Our results show that it achieves 67 passes out of 72, outperforming SpecGen and FormalBench in both Success Probability and Completeness. Our experimental evaluation achieves a 58.2% success probability and a 69.2% completeness score, while cutting evaluation time by 26.89% on average compared to prior methods. Together, these results demonstrate that AutoReSpec offers a scalable, efficient, and reliable approach to LLM-based formal specification generation. 

---
# When Does Multimodal AI Help? Diagnostic Complementarity of Vision-Language Models and CNNs for Spectrum Management in Satellite-Terrestrial Networks 

**Authors**: Yuanhang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.03774)  

**Abstract**: The adoption of vision-language models (VLMs) for wireless network management is accelerating, yet no systematic understanding exists of where these large foundation models outperform lightweight convolutional neural networks (CNNs) for spectrum-related tasks. This paper presents the first diagnostic comparison of VLMs and CNNs for spectrum heatmap understanding in non-terrestrial network and terrestrial network (NTN-TN) cooperative systems. We introduce SpectrumQA, a benchmark comprising 108K visual question-answer pairs across four granularity levels: scene classification (L1), regional reasoning (L2), spatial localization (L3), and semantic reasoning (L4). Our experiments on three NTN-TN scenarios with a frozen Qwen2-VL-7B and a trained ResNet-18 reveal a clear taskdependent complementarity: CNN achieves 72.9% accuracy at severity classification (L1) and 0.552 IoU at spatial localization (L3), while VLM uniquely enables semantic reasoning (L4) with F1=0.576 using only three in-context examples-a capability fundamentally absent in CNN architectures. Chain-of-thought (CoT) prompting further improves VLM reasoning by 12.6% (F1: 0.209->0.233) while having zero effect on spatial tasks, confirming that the complementarity is rooted in architectural differences rather than prompting limitations. A deterministic task-type router that delegates supervised tasks to CNN and reasoning tasks to VLM achieves a composite score of 0.616, a 39.1% improvement over CNN alone. We further show that VLM representations exhibit stronger cross-scenario robustness, with smaller performance degradation in 5 out of 6 transfer directions. These findings provide actionable guidelines: deploy CNNs for spatial localization and VLMs for semantic spectrum reasoning, rather than treating them as substitutes. 

---
# Build on Priors: Vision--Language--Guided Neuro-Symbolic Imitation Learning for Data-Efficient Real-World Robot Manipulation 

**Authors**: Pierrick Lorang, Johannes Huemer, Timothy Duggan, Kai Goebel, Patrik Zips, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2604.03759)  

**Abstract**: Enabling robots to learn long-horizon manipulation tasks from a handful of demonstrations remains a central challenge in robotics. Existing neuro-symbolic approaches often rely on hand-crafted symbolic abstractions, semantically labeled trajectories or large demonstration datasets, limiting their scalability and real-world applicability. We present a scalable neuro-symbolic framework that autonomously constructs symbolic planning domains and data-efficient control policies from as few as one to thirty unannotated skill demonstrations, without requiring manual domain engineering. Our method segments demonstrations into skills and employs a Vision-Language Model (VLM) to classify skills and identify equivalent high-level states, enabling automatic construction of a state-transition graph. This graph is processed by an Answer Set Programming solver to synthesize a PDDL planning domain, which an oracle function exploits to isolate the minimal, task-relevant and target relative observation and action spaces for each skill policy. Policies are learned at the control reference level rather than at the raw actuator signal level, yielding a smoother and less noisy learning target. Known controllers can be leveraged for real-world data augmentation by projecting a single demonstration onto other objects in the scene, simultaneously enriching the graph construction process and the dataset for imitation learning. We validate our framework primarily on a real industrial forklift across statistically rigorous manipulation trials, and demonstrate cross-platform generality on a Kinova Gen3 robotic arm across two standard benchmarks. Our results show that grounding control learning, VLM-driven abstraction, and automated planning synthesis into a unified pipeline constitutes a practical path toward scalable, data-efficient, expert-free and interpretable neuro-symbolic robotics. 

---
# Automated Attention Pattern Discovery at Scale in Large Language Models 

**Authors**: Jonathan Katzy, Razvan-Mihai Popescu, Erik Mekkes, Arie van Deursen, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2604.03764)  

**Abstract**: Large language models have found success by scaling up capabilities to work in general settings. The same can unfortunately not be said for interpretability methods. The current trend in mechanistic interpretability is to provide precise explanations of specific behaviors in controlled settings. These often do not generalize, or are too resource intensive for larger studies. In this work we propose to study repeated behaviors in large language models by mining completion scenarios in Java code datasets, through exploiting the structured nature of code. We collect the attention patterns generated in the attention heads to demonstrate that they are scalable signals for global interpretability of model components. We show that vision models offer a promising direction for analyzing attention patterns at scale. To demonstrate this, we introduce the Attention Pattern - Masked Autoencoder(AP-MAE), a vision transformer-based model that efficiently reconstructs masked attention patterns. Experiments on StarCoder2 show that AP-MAE (i) reconstructs masked attention patterns with high accuracy, (ii) generalizes across unseen models with minimal degradation, (iii) reveals recurring patterns across inferences, (iv) predicts whether a generation will be correct without access to ground truth, with accuracies ranging from 55% to 70% depending on the task, and (v) enables targeted interventions that increase accuracy by 13.6% when applied selectively, but cause collapse when applied excessively. These results establish attention patterns as a scalable signal for interpretability and demonstrate that AP-MAE provides a transferable foundation for both analysis and intervention in large language models. Beyond its standalone value, AP-MAE also serves as a selection procedure to guide fine-grained mechanistic approaches. We release code and models to support future work in large-scale interpretability. 

---
# Stabilizing Unsupervised Self-Evolution of MLLMs via Continuous Softened Retracing reSampling 

**Authors**: Yunyao Yu, Zhengxian Wu, Zhuohong Chen, Hangrui Xu, Zirui Liao, Xiangwen Deng, Zhifang Liu, Senyuan Shi, Haoqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03647)  

**Abstract**: In the unsupervised self-evolution of Multimodal Large Language Models, the quality of feedback signals during post-training is pivotal for stable and effective learning. However, existing self-evolution methods predominantly rely on majority voting to select the most frequent output as the pseudo-golden answer, which may stem from the model's intrinsic biases rather than guaranteeing the objective correctness of the reasoning paths. To counteract the degradation, we propose \textbf{C}ontinuous \textbf{S}oftened \textbf{R}etracing re\textbf{S}ampling (\textbf{CSRS}) in MLLM self-evolution. Specifically, we introduce a Retracing Re-inference Mechanism (\textbf{RRM}) that the model re-inferences from anchor points to expand the exploration of long-tail reasoning paths. Simultaneously, we propose Softened Frequency Reward (\textbf{SFR}), which replaces binary rewards with continuous signals, calibrating reward based on the answers' frequency across sampled reasoning sets. Furthermore, incorporated with Visual Semantic Perturbation (\textbf{VSP}), CSRS ensures the model prioritizes mathematical logic over visual superficiality. Experimental results demonstrate that CSRS significantly enhances the reasoning performance of Qwen2.5-VL-7B on benchmarks such as MathVision. We achieve state-of-the-art (SOTA) results in unsupervised self-evolution on geometric tasks. Our code is avaible at this https URL. 

---
# Toward Executable Repository-Level Code Generation via Environment Alignment 

**Authors**: Ruwei Pan, Junlei Shen, Linhao Wu, Yueheng Zhu, Zixiong Yang, Yakun Zhang, Lu Zhang, Hongyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03622)  

**Abstract**: Large language models (LLMs) have achieved strong performance on code generation, but existing methods still struggle with repository-level code generation under executable validation. Under this evaluation setting, success is determined not by the plausibility of isolated code fragments, but by whether a generated multi-file repository can be successfully installed, have its dependencies and internal references resolved, be launched, and be validated in a real execution environment. To address this challenge, we propose EnvGraph, a framework for repository-level code generation that formulates repository executability as an environment alignment problem. EnvGraph jointly models two coupled conditions for successful repository execution, namely external dependency satisfaction and repository-internal reference resolution. It maintains a dual-layer environment representation, uses execution evidence to perform execution-evidence-based attribution, and guides repository generation through a unified targeted revision mechanism within an iterative alignment loop. We evaluate EnvGraph on repository-level code generation with three representative backbone LLMs and compare it against representative environment-aware and repository-level baselines. Experimental results show that EnvGraph consistently achieves the best performance on these repository-level benchmarks. In particular, it outperforms the strongest non-EnvGraph baseline by an absolute margin of 5.72--5.87 percentage points in Functional Correctness and 4.58--8.66 percentage points in Non-Functional Quality. 

---
# Persistent Cross-Attempt State Optimization for Repository-Level Code Generation 

**Authors**: Ruwei Pan, Jiangshuai Wang, Qisheng Zhang, Yueheng Zhu, Linhao Wu, Zixiong Yang, Yakun Zhang, Lu Zhang, Hongyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03632)  

**Abstract**: Large language models (LLMs) have achieved substantial progress in repository-level code generation. However, solving the same repository-level task often requires multiple attempts, while existing methods still optimize each attempt in isolation and do not preserve or reuse task-specific state across attempts. In this paper, we propose LiveCoder, a novel framework for repository-level code generation based on cross-attempt knowledge optimization. LiveCoder maintains persistent task-specific state from prior attempts to guide subsequent generation. This state includes success knowledge, which captures reusable signals from previously strong repositories, failure knowledge, which records unsuccessful outcomes and their diagnostic signals, and a historical-best repository, which preserves the strongest result found so far and prevents regression. These components collectively transform repeated repository generation into a persistent, knowledge-driven optimization process. We evaluate LiveCoder using four frontier LLMs on two representative repository-level code generation benchmarks. Extensive experimental results demonstrate the effectiveness and efficiency of LiveCoder, improving the functional score by up to 22.94 percentage points, increasing repository reuse to 81.58%, and reducing cost by up to 53.63% on RAL-Bench while maintaining broadly stable non-functional quality. 

---
# SecPI: Secure Code Generation with Reasoning Models via Security Reasoning Internalization 

**Authors**: Hao Wang, Niels Mündler, Mark Vero, Jingxuan He, Dawn Song, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2604.03587)  

**Abstract**: Reasoning language models (RLMs) are increasingly used in programming. Yet, even state-of-the-art RLMs frequently introduce critical security vulnerabilities in generated code. Prior training-based approaches for secure code generation face a critical limitation that prevents their direct application to RLMs: they rely on costly, manually curated security datasets covering only a limited set of vulnerabilities. At the inference level, generic security reminders consistently degrade functional correctness while triggering only shallow ad-hoc vulnerability analysis. To address these problems, we present SecPI, a fine-tuning pipeline that teaches RLMs to internalize structured security reasoning, producing secure code by default without any security instructions at inference time. SecPI filters existing general-purpose coding datasets for security-relevant tasks using an LLM-based classifier, generates high-quality security reasoning traces with a teacher model guided by a structured prompt that systematically enumerates relevant CWEs and mitigations, and fine-tunes the target model on pairs of inputs with no security prompt and teacher reasoning traces -- as a result, the model learns to reason about security autonomously rather than in response to explicit instructions. An extensive evaluation on security benchmarks with state-of-the-art open-weight reasoning models validates the effectiveness of our approach. For instance, SecPI improves the percentage of functionally correct and secure generations for QwQ 32B from 48.2% to 62.2% (+14.0 points) on CWEval and from 18.2% to 22.0% on BaxBench. Further investigation also reveals strong cross-CWE and cross-language generalization beyond training vulnerabilities. Even when trained only on injection-related CWEs, QwQ 32B generates correct and secure code 9.9% more frequently on held-out memory-safety CWEs. 

---
# Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures 

**Authors**: Benjamin Rombaut  

**Link**: [PDF](https://arxiv.org/pdf/2604.03515)  

**Abstract**: LLM-based coding agents can localize bugs, generate patches, and run tests with diminishing human oversight, yet the scaffolding code that surrounds the language model (the control loop, tool definitions, state management, and context strategy) remains poorly understood. Existing surveys classify agents by abstract capabilities (tool use, planning, reflection) that cannot distinguish between architecturally distinct systems, and trajectory studies observe what agents do without examining the scaffold code that determines why. This paper presents a source-code-level architectural taxonomy derived from analysis of 13 open-source coding agent scaffolds at pinned commit hashes. Each agent is characterized across 12 dimensions organized into three layers: control architecture, tool and environment interface, and resource management. The analysis reveals that scaffold architectures resist discrete classification: control strategies range from fixed pipelines to Monte Carlo Tree Search, tool counts range from 0 to 37, and context compaction spans seven distinct strategies. Five loop primitives (ReAct, generate-test-repair, plan-execute, multi-attempt retry, tree search) function as composable building blocks that agents layer in different combinations; 11 of 13 agents compose multiple primitives rather than relying on a single control structure. Dimensions converge where external constraints dominate (tool capability categories, edit formats, execution isolation) and diverge where open design questions remain (context compaction, state management, multi-model routing). All taxonomic claims are grounded in file paths and line numbers, providing a reusable reference for researchers studying agent behavior and practitioners designing new scaffolds. 

---
# Incentives shape how humans co-create with generative AI 

**Authors**: Nathanael Jo, Manish Raghavan  

**Link**: [PDF](https://arxiv.org/pdf/2604.03529)  

**Abstract**: Generative AI is quickly becoming an integral part of people's everyday workflows. Early evidence has shown that while generative AI can increase individual-level productivity, it does so at the cost of collective diversity, potentially narrowing the set of ideas and perspectives produced. Our research stands in contrast to this concern: through a pre-registered randomized control trial, we show that incentives mediate AI's homogenizing force in a creative writing task where participants can use AI interactively. Participants rewarded for originality relative to peers produce collectively more diverse writing than those rewarded for quality alone. This divergence is driven not by abandoning AI, but by how participants use it: those incentivized for originality incorporate fewer AI suggestions verbatim, relying on the model more selectively for brainstorming, proofreading, and targeted edits. Our results reveal that the effects of generative AI depend not only on the technology itself, but also the behavioral strategies and incentive structures surrounding its use. 

---
# Agile Story-Point Estimation: Is RAG a Better Way to Go? 

**Authors**: Lamyea Maha, Tajmilur Rahman, Chanchal Roy  

**Link**: [PDF](https://arxiv.org/pdf/2604.03443)  

**Abstract**: The sprint-based iterative approach in the Agile software development method allows continuous feedback and adaptation. One of the crucial Agile software development activities is the sprint planning session where developers estimate the effort required to complete tasks through a consensus-based estimation technique such as Planning Poker. In the Agile software development method, a common unit of measuring development effort is Story Point (SP) which is assigned to tasks to understand the complexity and development time needed to complete them. Despite the benefits of this process, it is an extremely time-consuming manual process. To mitigate this issue, in this study, we investigated if this manual process can be automated using Retrieval Augmented Generation (RAG) which comprises a "Retriever" and a "Generator". We applied two embedding models - bge-large-en-v1.5, and Sentence-Transformers' all-mpnet-base-v2 on 23 open-source software projects of varying sizes and examined four key aspects: 1) how retrieval hyper-parameters influence the performance, 2) whether estimation accuracy differs across different sizes of the projects, 3) whether embedding model choice affects accuracy, and 4) how the RAG-based approach compares to the existing baselines. Although the RAG-based approach outperformed the baseline models in several occasions, our results did not exhibit statistically significant differences in performance across the projects or across the embedding models. This highlights the need for further studies and refinement of the RAG, and model adaptation strategies for better accuracy in automatically estimating user stories. 

---
# Measuring LLM Trust Allocation Across Conflicting Software Artifacts 

**Authors**: Noshin Ulfat, Ahsanul Ameen Sabit, Soneya Binta Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2604.03447)  

**Abstract**: LLM-based software engineering assistants fail not only by producing incorrect outputs, but also by allocating trust to the wrong artifact when code, documentation, and tests disagree. Existing evaluations focus mainly on downstream outcomes and therefore cannot reveal whether a model recognized degraded evidence, identified the unreliable source, or calibrated its trust across artifacts. We present TRACE (Trust Reasoning over Artifacts for Calibrated Evaluation), a framework that elicits structured artifact-level trust traces over Javadoc, method signatures, implementations, and test prefixes under blind perturbations. Using 22,339 valid traces from seven models on 456 curated Java method bundles, we evaluate per-artifact quality assessment, inconsistency detection, affected artifact attribution, and source prioritization. Across all models, quality penalties are largely localized to the perturbed artifact and increase with severity, but sensitivity is asymmetric across artifact types: documentation bugs induce a substantially larger heavy-to-subtle gap than implementation faults (0.152-0.253 vs. 0.049-0.123). Models detect explicit documentation bugs well (67-94%) and Javadoc and implementation contradictions at 50-91%, yet show a systematic blind spot when only the implementation drifts while the documentation remains plausible, with detection dropping by 7-42 percentage points. Confidence is poorly calibrated for six of seven models. These findings suggest that current LLMs are better at auditing natural-language specifications than at detecting subtle code-level drift, motivating explicit artifact-level trust reasoning before correctness-critical downstream use. 

---
# MetaSAEs: Joint Training with a Decomposability Penalty Produces More Atomic Sparse Autoencoder Latents 

**Authors**: Matthew Levinson  

**Link**: [PDF](https://arxiv.org/pdf/2604.03436)  

**Abstract**: Sparse autoencoders (SAEs) are increasingly used for safety-relevant applications including alignment detection and model steering. These use cases require SAE latents to be as atomic as possible. Each latent should represent a single coherent concept drawn from a single underlying representational subspace. In practice, SAE latents blend representational subspaces together. A single feature can activate across semantically distinct contexts that share no true common representation, muddying an already complex picture of model computation. We introduce a joint training objective that directly penalizes this subspace blending. A small meta SAE is trained alongside the primary SAE to sparsely reconstruct the primary SAE's decoder columns; the primary SAE is penalized whenever its decoder directions are easy to reconstruct from the meta dictionary. This occurs whenever latent directions lie in a subspace spanned by other primary directions. This creates gradient pressure toward more mutually independent decoder directions that resist sparse meta-compression.
On GPT-2 large (layer 20), the selected configuration reduces mean $|\varphi|$ by 7.5% relative to an identical solo SAE trained on the same data. Automated interpretability (fuzzing) scores improve by 7.6%, providing external validation of the atomicity gain independent of the training and co-occurrence metrics. Reconstruction overhead is modest. Results on Gemma 2 9B are directional. On not-fully-converged SAEs, the same parameterization yields the best results, a $+8.6\%$ $\Delta$Fuzz. Though directional, this is an encouraging sign that the method transfers to a larger model. Qualitative analysis confirms that features firing on polysemantic tokens are split into semantically distinct sub-features, each specializing in a distinct representational subspace. 

---
# Inference-Path Optimization via Circuit Duplication in Frozen Visual Transformers for Marine Species Classification 

**Authors**: Thomas Manuel Rost  

**Link**: [PDF](https://arxiv.org/pdf/2604.03428)  

**Abstract**: Automated underwater species classification is constrained by annotation cost and environmental variation that limits the transferability of fully supervised models. Recent work has shown that frozen embeddings from self-supervised vision foundation models already provide a strong label-efficient baseline for marine image classification. Here we investigate whether this frozen-embedding regime can be improved at inference time, without fine-tuning or changing model weights.
We apply Circuit Duplication, an inference-time method originally proposed for Large Language Models, in which a selected range of transformer layers is traversed twice during the forward pass. We evaluate on the class-imbalanced AQUA20 benchmark using frozen DINOv3 embeddings under two settings: global circuit selection, where a single duplicated circuit is chosen for the full dataset, and class-specific circuit selection, where each species may receive a different optimal circuit. Both settings use simple semi-supervised downstream classifiers.
Circuit Duplication consistently improves over the standard frozen forward pass. At the maximum label budget, class-specific selection reaches a macro F1 of 0.875, closing the gap to the fully supervised ConvNeXt benchmark (0.889) to 1.4 points without any gradient-based training. Four species exceed their fully supervised reference, with octopus improving by +12.1 F1 points. Across all budgets, roughly 75% of classes prefer a class-specific circuit, indicating a genuinely class-dependent benefit. To our knowledge, this is the first application of Circuit Duplication to computer vision. 

---
# Can LLMs Reason About Attention? Towards Zero-Shot Analysis of Multimodal Classroom Behavior 

**Authors**: Nolan Platt, Sehrish Nizamani, Alp Tural, Elif Tural, Saad Nizamani, Andrew Katz, Yoonje Lee, Nada Basit  

**Link**: [PDF](https://arxiv.org/pdf/2604.03401)  

**Abstract**: Understanding student engagement usually requires time-consuming manual observation or invasive recording that raises privacy concerns. We present a privacy-preserving pipeline that analyzes classroom videos to extract insights about student attention, without storing any identifiable footage. Our system runs on a single GPU, using OpenPose for skeletal extraction and Gaze-LLE for visual attention estimation. Original video frames are deleted immediately after pose extraction, thus only geometric coordinates (stored as JSON) are retained, ensuring compliance with FERPA. The extracted pose and gaze data is processed by QwQ-32B-Reasoning, which performs zero-shot analysis of student behavior across lecture segments. Instructors access results through a web dashboard featuring attention heatmaps and behavioral summaries. Our preliminary findings suggest that LLMs may show promise for multimodal behavior understanding, although they still struggle with spatial reasoning about classroom layouts. We discuss these limitations and outline directions for improving LLM spatial comprehension in educational analytics contexts. 

---
# AICCE: AI Driven Compliance Checker Engine 

**Authors**: Mohammad Wali Ur Rahman, Martin Manuel Lopez, Lamia Tasnim Mim, Carter Farthing, Julius Battle, Kathryn Buckley, Salim Hariri  

**Link**: [PDF](https://arxiv.org/pdf/2604.03330)  

**Abstract**: For digital infrastructure to be safe, compatible, and standards-aligned, automated communication protocol compliance verification is crucial. Nevertheless, current rule-based systems are becoming less and less effective since they are unable to identify subtle or intricate non-compliance, which attackers frequently use to establish covert communication channels in IPv6 traffic. In order to automate IPv6 compliance verification, this paper presents the Artificial Intelligence Driven Compliance Checker Engine (AICCE), a novel generative system that combines dual-architecture reasoning and retrieval-augmented generation (RAG). Specification segments pertinent to each query can be efficiently retrieved thanks to the semantic encoding of protocol standards into a high-dimensional vector space. Based on this framework, AICCE offers two complementary pipelines: (i) Explainability Mode, which uses parallel LLM agents to render decisions and settle disputes through organized discussions to improve interpretability and robustness, and (ii) Script Execution Mode, which converts clauses into Python rules that can be executed quickly for dataset-wide verification. With the debate mechanism enhancing decision reliability in complicated scenarios and the script-based pipeline lowering per-sample latency, AICCE achieves accuracy and F1-scores of up to 99% when tested on IPv6 packet samples across sixteen cutting-edge generative models. By offering a scalable, auditable, and generalizable mechanism for identifying both routine and covert non-compliance in dynamic communication environments, our results show that AICCE overcomes the blind spots of conventional rule-based compliance checking systems. 

---
# AEGIS: Scaling Long-Sequence Homomorphic Encrypted Transformer Inference via Hybrid Parallelism on Multi-GPU Systems 

**Authors**: Zhaoting Gong, Ran Ran, Fan Yao, Wujie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.03425)  

**Abstract**: Fully Homomorphic Encryption (FHE) enables privacy-preserving Transformer inference, but long-sequence encrypted Transformers quickly exceed single-GPU memory capacity because encoded weights are already large and encrypted activations grow rapidly with sequence length. Multi-GPU execution therefore becomes unavoidable, yet scaling remains challenging because communication is jointly induced by application-level aggregation and encryption-level RNS coupling. Existing approaches either synchronize between devices frequently or replicate encrypted tensors across devices, leading to excessive communication and latency.
We present AEGIS, an Application-Encryption Guided Inference System for scalable long-sequence encrypted Transformer inference on multi-GPU platforms. AEGIS derives device placement from ciphertext dependencies jointly induced by Transformer dataflow and CKKS polynomial coupling, co-locating modulus-coherent and token-coherent data so that communication is introduced only when application dependencies require it, while reordering polynomial operators to overlap the remaining collectives with computation.
On 2048-token inputs, AEGIS reduces inter-GPU communication by up to 57.9% in feed-forward networks and 81.3% in self-attention versus prior state-of-the-art designs. On four GPUs, it achieves up to 96.62% scaling efficiency, 3.86x end-to-end speedup, and 69.1% per-device memory reduction. These results establish coordinated application-encryption parallelism as a practical foundation for scalable homomorphic Transformer inference. 

---
# The Ideation Bottleneck: Decomposing the Quality Gap Between AI-Generated and Human Economics Research 

**Authors**: Ning Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.03338)  

**Abstract**: Autonomous AI systems can now generate complete economics research papers, but they substantially underperform human-authored publications in head-to-head comparisons. This paper decomposes the quality gap into two independent components: research idea quality and execution quality. Using a two-model ensemble of fine-tuned language models trained on publication decisions (Gong, Li, and Zhou, 2026) to evaluate idea quality and a comprehensive six-dimension rubric assessed by Gemini 3.1 Flash Lite -- the same model family used as the APE tournament judge, ensuring methodological consistency -- to evaluate execution quality, we analyze 953 economics papers -- 912 AI-generated papers from the APE project and 41 human papers published in the American Economic Review and AEJ: Economic Policy. The idea quality gap is large (Cohen's d = 2.23, p < 0.001), with human papers achieving 47.1% mean ensemble exceptional probability versus 16.5% for AI. The execution quality gap is also significant but smaller (d = 0.90, p < 0.001), with human papers scoring 4.38/5.0 versus 3.84. Idea quality accounts for approximately 71% of the overall quality difference, with execution contributing 29%. The largest execution weakness is mechanism analysis depth (d = 1.43); no significant difference is found on robustness. We document that 74% of AI papers employ difference-in-differences, and only 7 AI papers (0.8%) surpass the median human paper on both idea and execution quality simultaneously. The primary bottleneck to competitive AI-generated economics research remains ideation. 

---
# VitaTouch: Property-Aware Vision-Tactile-Language Model for Robotic Quality Inspection in Manufacturing 

**Authors**: Junyi Zong, Qingxuan Jia, Meixian Shi, Tong Li, Jiayuan Li, Zihang Lv, Gang Chen, Fang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.03322)  

**Abstract**: Quality inspection in smart manufacturing requires identifying intrinsic material and surface properties beyond visible geometry, yet vision-only methods remain vulnerable to occlusion and reflection. We propose VitaTouch, a property-aware vision-tactile-language model for material-property inference and natural-language attribute description. VitaTouch uses modality-specific encoders and a dual Q-Former to extract language-relevant visual and tactile features, which are compressed into prefix tokens for a large language model. We align each modality with text and explicitly couple vision and touch through contrastive learning. We also construct VitaSet, a multimodal dataset with 186 objects, 52k images, and 5.1k human-verified instruction-answer pairs. VitaTouch achieves the best performance on HCT and the overall TVL benchmark, while remaining competitive on SSVTP. On VitaSet, it reaches 88.89% hardness accuracy, 75.13% roughness accuracy, and 54.81% descriptor recall; the material-description task further achieves a peak semantic similarity of 0.9009. With LoRA-based fine-tuning, VitaTouch attains 100.0%, 96.0%, and 92.0% accuracy for 2-, 3-, and 5-category defect recognition, respectively, and delivers 94.0% closed-loop recognition accuracy and 94.0% end-to-end sorting success in 100 laboratory robotic trials. More details are available at the project page: this https URL 

---
# V-Reflection: Transforming MLLMs from Passive Observers to Active Interrogators 

**Authors**: Jiazhou Zhou, Yucheng Chen, Hongyang Li, Qing Jiang, Hu Zhou, Ying-Cong Chen, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03307)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved remarkable success, yet they remain prone to perception-related hallucinations in fine-grained tasks. This vulnerability arises from a fundamental limitation: their reasoning is largely restricted to the language domain, treating visual input as a static, reasoning-agnostic preamble rather than a dynamic participant. Consequently, current models act as passive observers, unable to re-examine visual details to ground their evolving reasoning states. To overcome this, we propose V-Reflection, a framework that transforms the MLLM into an active interrogator through a "think-then-look" visual reflection mechanism. During reasoning, latent states function as dynamic probes that actively interrogate the visual feature space, grounding each reasoning step for task-critical evidence. Our approach employs a two-stage distillation strategy. First, the Box-Guided Compression (BCM) module establishes stable pixel-to-latent targets through explicit spatial grounding. Next, a Dynamic Autoregressive Compression (DAC) module maps the model's hidden states into dynamic probes that interrogate the global visual feature map. By distilling the spatial expertise of the BCM teacher into the DAC student, V-Reflection internalizes the ability to localize task-critical evidence. During inference, both modules remain entirely inactive, maintaining a purely end-to-end autoregressive decoding in the latent space with optimal efficiency. Extensive experiments demonstrate the effectiveness of our V-Reflection across six perception-intensive benchmarks, significantly narrowing the fine-grained perception gap. Visualizations confirm that latent reasoning autonomously localizes task-critical visual evidence. 

---
# Beyond Static Vision: Scene Dynamic Field Unlocks Intuitive Physics Understanding in Multi-modal Large Language Models 

**Authors**: Nanxi Li, Xiang Wang, Yuanjie Chen, Haode Zhang, Hong Li, Yong-Lu Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.03302)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in image and video understanding, their ability to comprehend the physical world has become an increasingly important research focus. Despite their improvements, current MLLMs struggle significantly with high-level physics reasoning. In this work, we investigate the first step of physical reasoning, i.e., intuitive physics understanding, revealing substantial limitations in understanding the dynamics of continuum objects. To isolate and evaluate this specific capability, we introduce two fundamental benchmark tasks: Next Frame Selection (NFS) and Temporal Coherence Verification (TCV). Our experiments demonstrate that even state-of-the-art MLLMs perform poorly on these foundational tasks. To address this limitation, we propose Scene Dynamic Field (SDF), a concise approach that leverages physics simulators within a multi-task fine-tuning framework. SDF substantially improves performance, achieving up to 20.7% gains on fluid tasks while showing strong generalization to unseen physical domains. This work not only highlights a critical gap in current MLLMs but also presents a promising cost-efficient approach for developing more physically grounded MLLMs. Our code and data are available at this https URL. 

---
# 3D-IDE: 3D Implicit Depth Emergent 

**Authors**: Chushan Zhang, Ruihan Lu, Jinguang Tong, Yikai Wang, Hongdong Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.03296)  

**Abstract**: Leveraging 3D information within Multimodal Large Language Models (MLLMs) has recently shown significant advantages for indoor scene understanding. However, existing methods, including those using explicit ground-truth 3D positional encoding and those grafting external 3D foundation models for implicit geometry, struggle with the trade-off in 2D-3D representation fusion, leading to suboptimal deployment. To this end, we propose 3D-Implicit Depth Emergence, a method that reframes 3D perception as an emergent property derived from geometric self-supervision rather than explicit encoding. Our core insight is the Implicit Geometric Emergence Principle: by strategically leveraging privileged geometric supervision through mechanisms like a fine-grained geometry validator and global representation constraints, we construct an information bottleneck. This bottleneck forces the model to maximize the mutual information between visual features and 3D structures, allowing 3D awareness to emerge naturally within a unified visual representation. Unlike existing approaches, our method enables 3D perception to emerge implicitly, disentangling features in dense regions and, crucially, eliminating depth and pose dependencies during inference with zero latency overhead. This paradigm shift from external grafting to implicit emergence represents a fundamental rethinking of 3D knowledge integration in visual-language models. Extensive experiments demonstrate that our method surpasses SOTA on multiple 3D scene understanding benchmarks. Our approach achieves a 55% reduction in inference latency while maintaining strong performance across diverse downstream tasks, underscoring the effectiveness of meticulously designed auxiliary objectives for dependency-free 3D understanding. Source code can be found at this http URL. 

---
# Scaling Teams or Scaling Time? Memory Enabled Lifelong Learning in LLM Multi-Agent Systems 

**Authors**: Shanglin Wu, Yuyang Luo, Yueqing Liang, Kaiwen Shi, Yanfang Ye, Ali Payani, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2604.03295)  

**Abstract**: Large language model (LLM) multi-agent systems can scale along two distinct dimensions: by increasing the number of agents and by improving through accumulated experience over time. Although prior work has studied these dimensions separately, their interaction under realistic cost constraints remains unclear. In this paper, we introduce a conceptual scaling view of multi-agent systems that jointly considers team size and lifelong learning ability, and we study how memory design shares this landscape. To this end, we propose \textbf{LLMA-Mem}, a lifelong memory framework for LLM multi-agent systems under flexible memory topologies. We evaluate LLMA-Mem on \textsc{MultiAgentBench} across coding, research, and database environments. Empirically, LLMA-Mem consistently improves long-horizon performance over baselines while reducing cost. Our analysis further reveals a non-monotonic scaling landscape: larger teams do not always produce better long-term performance, and smaller teams can outperform larger ones when memory better supports the reuse of experience. These findings position memory design as a practical path for scaling multi-agent systems more effectively and more efficiently over time. 

---
# RAGnaroX: A Secure, Local-Hosted ChatOps Assistant Using Small Language Models 

**Authors**: Benedikt Dornauer, Mircea-Cristian Racasan  

**Link**: [PDF](https://arxiv.org/pdf/2604.03291)  

**Abstract**: This paper introduces RAGnaroX, a resource-efficient ChatOps assistant that operates entirely on commodity hardware. Unlike existing solutions that often rely on external providers such as Azure or OpenAI, RAGnaroX offers a fully auditable, on-premise stack implemented in Rust. Its architecture integrates modular data ingestion, hybrid retrieval, and function calling, enabling flexible yet secure deployment. Our evaluation focuses on the RAG pipeline, with benchmarks conducted on the SQuAD (single-hop QA), MultiHopRAG (multi-hop QA), and MLQA (cross-lingual QA) datasets. Results show that RAGnaroX achieves competitive accuracy while maintaining strong resource efficiency, for example, reaching 0.90 context precision on single-hop questions with an average response time of 2.5 seconds per request. A replication package containing the tool, the demonstration video (this https URL v=cDxfuEbcoM4), and all supporting materials are available at this https URL. 

---
# SafeScreen: A Safety-First Screening Framework for Personalized Video Retrieval for Vulnerable Users 

**Authors**: Wenzheng Zhao, Madhava Kalyan Gadiputi, Fengpei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2604.03264)  

**Abstract**: Open-domain video platforms offer rich, personalized content that could support health, caregiving, and educational applications, but their engagement-optimized recommendation algorithms can expose vulnerable users to inappropriate or harmful material. These risks are especially acute in child-directed and care settings (e.g., dementia care), where content must satisfy individualized safety constraints before being shown. We introduce SafeScreen, a safety-first video screening framework that retrieves and presents personalized video while enforcing individualized safety constraints. Rather than ranking videos by relevance or popularity, SafeScreen treats safety as a prerequisite and performs sequential approval or rejection of candidate videos through an automated pipeline. SafeScreen integrates three key components: (i) profile-driven extraction of individualized safety criteria, (ii) evidence-grounded assessments via adaptive question generation and multimodal VideoRAG analysis, and (iii) LLM-based decision-making that verifies safety, appropriateness, and relevance before content exposure. This design enables explainable, real-time screening of uncurated video repositories without relying on precomputed safety labels. We evaluate SafeScreen in a dementia-care reminiscence case study using 30 synthetic patient profiles and 90 test queries. Results demonstrate that SafeScreen prioritizes safety over engagement, diverging from YouTube's engagement-optimized rankings in 80-93% of cases, while maintaining high levels of safety coverage, sensibleness, and groundedness, as validated by both LLM-based evaluation and domain experts. 

---
# LPC-SM: Local Predictive Coding and Sparse Memory for Long-Context Language Modeling 

**Authors**: Keqin Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.03263)  

**Abstract**: Most current long-context language models still rely on attention to handle both local interaction and long-range state, which leaves relatively little room to test alternative decompositions of sequence modeling. We propose LPC-SM, a hybrid autoregressive architecture that separates local attention, persistent memory, predictive correction, and run-time control within the same block, and we use Orthogonal Novelty Transport (ONT) to govern slow-memory writes. We evaluate a 158M-parameter model in three stages spanning base language modeling, mathematical continuation, and 4096-token continuation. Removing mHC raises the Stage-A final LM loss from 12.630 to 15.127, while adaptive sparse control improves the Stage-B final LM loss from 12.137 to 10.787 relative to a matched fixed-ratio continuation. The full route remains stable at sequence length 4096, where Stage C ends with final LM loss 11.582 and improves the delayed-identifier diagnostic from 14.396 to 12.031 in key cross-entropy. Taken together, these results show that long-context autoregressive modeling can be organized around a broader division of labor than attention alone. 

---
# FVRuleLearner: Operator-Level Reasoning Tree (OP-Tree)-Based Rules Learning for Formal Verification 

**Authors**: Lily Jiaxin Wan, Chia-Tung Ho, Yunsheng Bai, Cunxi Yu, Deming Chen, Haoxing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.03245)  

**Abstract**: The remarkable reasoning and code generation capabilities of large language models (LLMs) have recently motivated increasing interest in automating formal verification (FV), a process that ensures hardware correctness through mathematically precise assertions but remains highly labor-intensive, particularly through the translation of natural language into SystemVerilog Assertions (NL-to-SVA). However, LLMs still struggle with SVA generation due to limited training data and the intrinsic complexity of FV operators. Consequently, a more efficient and robust methodology for ensuring correct SVA operator selection is essential for producing functionally correct assertions. To address these challenges, we introduce FVRuleLearner, an Operator-Level Rule (Op-Rule) learning framework built on a novel Operator Reasoning Tree (OP-Tree), which models SVA generation as structured, interpretable reasoning. FVRuleLearner operates in two complementary phases: (1) Training: it constructs OP-Tree that decomposes NL-to-SVA alignment into fine-grained, operator-aware questions, combining reasoning paths that lead to correct assertions; and (2) Testing: it performs operator-aligned retrieval to fetch relevant reasoning traces from the learned OP-Tree and generate new rules for unseen specifications. In the comprehensive studies, the proposed FVRuleLearner outperforms the state-of-the-art baseline by 3.95% in syntax correctness and by 31.17% in functional correctness on average. Moreover, FVRuleLearner successfully reduces an average of 70.33% of SVA functional failures across diverse operator categories through a functional taxonomy analysis, showing the effectiveness of applying learned OP-Tree to the Op-Rule generations for unseen NL-to-SVA tasks. These results establish FVRuleLearner as a new paradigm for domain-specific reasoning and rule learning in formal verification. 

---
# Classifying Problem and Solution Framing in Congressional Social Media 

**Authors**: Misha Melnyk, Mitchell Dolny, Joshua D. Elkind, A. Michael Tjhin, Saisha Chebium, Blake VanBerlo, Annelise Russell, Michelle M. Buehlmann, Jesse Hoey  

**Link**: [PDF](https://arxiv.org/pdf/2604.03247)  

**Abstract**: Policy setting in the USA according to the ``Garbage Can'' model differentiates between ``problem'' and ``solution'' focused processes. In this paper, we study a large dataset of US Senator postings on Twitter (1.68m tweets in total). Our objective is to develop an automated method to label Senatorial posts as either in the problem or solution streams. Two academic policy experts labeled a subset of 3967 tweets as either problem, solution, or other (anything not problem or solution). We split off a subset of 500 tweets into a test set, with the remaining 3467 used for training. During development, this training set was further split by 60/20/20 proportions for fitting, validation, and development test sets. We investigated supervised learning methods for building problem/solution classifiers directly on the training set, evaluating their performance in terms of F1 score on the validation set, allowing us to rapidly iterate through models and hyperparameters, achieving an average weighted F1 score of above 0.8 on cross validation across the three categories using a BERTweet Base model. 

---
# LLMs-Healthcare : Current Applications and Challenges of Large Language Models in various Medical Specialties 

**Authors**: Ummara Mumtaz, Awais Ahmed, Summaya Mumtaz  

**Link**: [PDF](https://arxiv.org/pdf/2311.12882)  

**Abstract**: We aim to present a comprehensive overview of the latest advancements in utilizing Large Language Models (LLMs) within the healthcare sector, emphasizing their transformative impact across various medical domains. LLMs have become pivotal in supporting healthcare, including physicians, healthcare providers, and patients. Our review provides insight into the applications of Large Language Models (LLMs) in healthcare, specifically focusing on diagnostic and treatment-related functionalities. We shed light on how LLMs are applied in cancer care, dermatology, dental care, neurodegenerative disorders, and mental health, highlighting their innovative contributions to medical diagnostics and patient care. Throughout our analysis, we explore the challenges and opportunities associated with integrating LLMs in healthcare, recognizing their potential across various medical specialties despite existing limitations. Additionally, we offer an overview of handling diverse data types within the medical field. 

---
# From Concept to Practice: an Automated LLM-aided UVM Machine for RTL Verification 

**Authors**: Junhao Ye, Yuchen Hu, Ke Xu, Dingrong Pan, Qichun Chen, Jie Zhou, Shuai Zhao, Xinwei Fang, Xi Wang, Nan Guan, Zhe Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19959)  

**Abstract**: Verification presents a major bottleneck in Integrated Circuit (IC) development, consuming nearly 70% of the total development effort. While the Universal Verification Methodology (UVM) is widely used in industry to improve verification efficiency through structured and reusable testbenches, constructing these testbenches and generating sufficient stimuli remain challenging. These challenges arise from the considerable manual coding effort required, repetitive manual execution of multiple EDA tools, and the need for in-depth domain expertise to navigate complex this http URL, we present UVM^2, an automated verification framework that leverages Large Language Models (LLMs) to generate UVM testbenches and iteratively refine them using coverage feedback, significantly reducing manual effort while maintaining rigorous verification this http URL evaluate UVM^2, we introduce a benchmark suite comprising Register Transfer Level (RTL) designs of up to 1.6K lines of this http URL results show that UVM^2 reduces testbench setup time by up to UVM^2 compared to experienced engineers, and achieve average code and function coverage of 87.44% and 89.58%, outperforming state-of-the-art solutions by 20.96% and 23.51%, respectively. 

---
# PRAISE: Prefix-Based Rollout Reuse in Agentic Search Training 

**Authors**: Erhan Zhang, Yiqun Chen, Zechun Niu, Wei Yang, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2604.03675)  

**Abstract**: In agentic search, large language models (LLMs) are trained to perform multi-turn retrieval and reasoning for complex tasks such as multi-hop question answering (QA). However, current search-based Reinforcement Learning (RL) methods suffer from two core limitations: expensive long-horizon rollouts are under-utilized during training, and supervision is typically available only at the final answer, resulting in severe reward sparsity. We present Prefix-based Rollout reuse for Agentic search with Intermediate Step rEwards (PRAISE), a framework for improving both data efficiency and credit assignment in agentic search training. Given a complete search trajectory, PRAISE extracts prefix states at different search turns, elicits intermediate answers from them, and uses these prefixes both to construct additional training trajectories and to derive step-level rewards from performance differences across prefixes. Our method uses a single shared model for both search policy learning and prefix answer evaluation, enabling joint optimization without extra human annotations or a separate reward model. Experiments on multi-hop QA benchmarks show that PRAISE consistently improves performance over strong baselines. 

---
# Customized User Plane Processing via Code Generating AI Agents for Next Generation Mobile Networks 

**Authors**: Xiaowen Ma, Onur Ayan, Yunpu Ma, Xueli An  

**Link**: [PDF](https://arxiv.org/pdf/2604.03282)  

**Abstract**: Generative AI is envisioned to have a crucial impact on next generation mobile networking, making the sixth generation (6G) system considerably more autonomous, flexible, and adaptive than its predecessors. By leveraging their natural language processing and code generation capabilities, AI agents enable novel interactions and services between networks and vertical applications. A particularly promising and interesting use case is the customization of connectivity services for vertical applications by generating new customized processing blocks based on text-based service requests. More specifically, AI agents are able to generate code for a new function block that handles user plane traffic, allowing it to inspect and decode a protocol data unit (PDU) and perform specified actions as requested by the application. In this study, we investigate the code generation problem for generating such customized processing blocks on-demand. We evaluate various factors affecting the accuracy of the code generation process in this context, including model selection, prompt design, and the provision of a code template for the agent to utilize. Our findings indicate that AI agents are capable of generating such blocks with the desired behavior on-demand under suitable conditions. We believe that exploring the code generation for network-specific tasks is a very interesting problem for 6G and beyond, enabling networks to achieve a new level of customization by generating new capabilities on-demand. 

---
