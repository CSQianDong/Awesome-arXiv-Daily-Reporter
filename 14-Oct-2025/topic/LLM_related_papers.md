# What Generative Search Engines Like and How to Optimize Web Content Cooperatively 

**Authors**: Yujiang Wu, Shanshan Zhong, Yubin Kim, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.11438)  

**Abstract**: By employing large language models (LLMs) to retrieve documents and generate natural language responses, Generative Engines, such as Google AI overview and ChatGPT, provide significantly enhanced user experiences and have rapidly become the new form of search. Their rapid adoption also drives the needs of Generative Engine Optimization (GEO), as content providers are eager to gain more traction from them. In this paper, we introduce AutoGEO, a framework to automatically learn generative engine preferences when using retrieved contents for response generation, and rewrite web contents for more such traction. AutoGEO first prompts frontier LLMs to explain generative engine preferences and extract meaningful preference rules from these explanations. Then it uses preference rules as context engineering for AutoGEO$_\text{API}$, a prompt-based GEO system, and as rule-based rewards to train AutoGEO$_\text{Mini}$, a cost-effective GEO model. Experiments on the standard GEO-Bench and two newly constructed benchmarks using real user queries demonstrate the effectiveness of AutoGEO in enhancing content traction while preserving search utility. Analyses confirm the learned rules' robustness and abilities to capture unique preferences in variant domains, and AutoGEO systems' ability to embed them in content optimization. The code is released at this https URL. 

---
# From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevance 

**Authors**: Runze Xia, Yupeng Ji, Yuxi Zhou, Haodong Liu, Teng Zhang, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.11056)  

**Abstract**: Query-service relevance prediction in e-commerce search systems faces strict latency requirements that prevent the direct application of Large Language Models (LLMs). To bridge this gap, we propose a two-stage reasoning distillation framework to transfer reasoning capabilities from a powerful teacher LLM to a lightweight, deployment-friendly student model. In the first stage, we address the limitations of general-purpose LLMs by constructing a domain-adapted teacher model. This is achieved through a three-step process: domain-adaptive pre-training to inject platform knowledge, supervised fine-tuning to elicit reasoning skills, and preference optimization with a multi-dimensional reward model to ensure the generation of reliable and preference-aligned reasoning paths. This teacher can then automatically annotate massive query-service pairs from search logs with both relevance labels and reasoning chains. In the second stage, to address the challenges of architectural heterogeneity in standard distillation, we introduce Contrastive Reasoning Self-Distillation (CRSD). By modeling the behavior of the same student model under "standard" and "reasoning-augmented" inputs as a teacher-student relationship, CRSD enables the lightweight model to internalize the teacher's complex decision-making mechanisms without needing the explicit reasoning path at inference. Offline evaluations and online A/B testing in the Meituan search advertising system demonstrate that our framework achieves significant improvements across multiple metrics, validating its effectiveness and practical value. 

---
# OneRec-Think: In-Text Reasoning for Generative Recommendation 

**Authors**: Zhanyu Liu, Shiyao Wang, Xingmei Wang, Rongzhou Zhang, Jiaxin Deng, Honghui Bao, Jinghao Zhang, Wuchao Li, Pengfei Zheng, Xiangyu Wu, Yifei Hu, Qigen Hu, Xinchen Luo, Lejian Ren, Zixing Zhang, Qianqian Wang, Kuo Cai, Yunfan Wu, Hongtao Cheng, Zexuan Cheng, Lu Ren, Huanjie Wang, Yi Su, Ruiming Tang, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.11639)  

**Abstract**: The powerful generative capacity of Large Language Models (LLMs) has instigated a paradigm shift in recommendation. However, existing generative models (e.g., OneRec) operate as implicit predictors, critically lacking the capacity for explicit and controllable reasoning-a key advantage of LLMs. To bridge this gap, we propose OneRec-Think, a unified framework that seamlessly integrates dialogue, reasoning, and personalized recommendation. OneRec-Think incorporates: (1) Itemic Alignment: cross-modal Item-Textual Alignment for semantic grounding; (2) Reasoning Activation: Reasoning Scaffolding to activate LLM reasoning within the recommendation context; and (3) Reasoning Enhancement, where we design a recommendation-specific reward function that accounts for the multi-validity nature of user preferences. Experiments across public benchmarks show state-of-the-art performance. Moreover, our proposed "Think-Ahead" architecture enables effective industrial deployment on Kuaishou, achieving a 0.159\% gain in APP Stay Time and validating the practical efficacy of the model's explicit reasoning capability. 

---
# Does LLM Focus on the Right Words? Diagnosing Language Bias in LLM-based Recommenders 

**Authors**: Bohao Wang, Jiawei Chen, Feng Liu, Changwang Zhang, Jun Wang, Canghong Jin, Chun Chen, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10978)  

**Abstract**: Large language models (LLMs), owing to their extensive open-domain knowledge and semantic reasoning capabilities, have been increasingly integrated into recommender systems (RS). However, a substantial gap remains between the pre-training objectives of LLMs and the specific requirements of recommendation tasks. To address this gap, supervised fine-tuning (SFT) is commonly performed on specially curated recommendation datasets to further enhance their predictive ability. Despite its success, SFT exhibits a critical limitation: it induces Language Bias, whereby the model over-relies on auxiliary tokens-such as task descriptions and prefix-generated tokens-while underutilizing core user interaction tokens that encode user-specific preferences. This bias not only undermines recommendation accuracy but also raises unfairness concerns.
To address this issue, we propose Group Distributionally Robust Optimization-based Tuning (GDRT), a novel fine-tuning paradigm that enforces consistent model performance across token groups with varying degrees of relevance to auxiliary tokens. By adaptively upweighting underperforming groups, typically those weakly correlated with auxiliary tokens, GDRT shifts the model's attention from superficial auxiliary cues to informative user interaction tokens, thereby mitigating language bias. Extensive experiments conducted on three public datasets demonstrate that GDRT effectively mitigates language bias, yielding substantial improvements in recommendation accuracy (with an average NDCG@10 gain of 24.29%) and significantly enhancing recommendation fairness. 

---
# HatLLM: Hierarchical Attention Masking for Enhanced Collaborative Modeling in LLM-based Recommendation 

**Authors**: Yu Cui, Feng Liu, Jiawei Chen, Canghong Jin, Xingyu Lou, Changwang Zhang, Jun Wang, Yuegang Sun, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10955)  

**Abstract**: Recent years have witnessed a surge of research on leveraging large language models (LLMs) for sequential recommendation. LLMs have demonstrated remarkable potential in inferring users' nuanced preferences through fine-grained semantic reasoning. However, they also exhibit a notable limitation in effectively modeling collaborative signals, i.e., behavioral correlations inherent in users' historical interactions. Our empirical analysis further reveals that the attention mechanisms in LLMs tend to disproportionately focus on tokens within the same item, thereby impeding the capture of cross-item correlations.
To address this limitation, we propose a novel hierarchical attention masking strategy for LLM-based recommendation, termed HatLLM. Specifically, in shallow layers, HatLLM masks attention between tokens from different items, facilitating intra-item semantic understanding; in contrast, in deep layers, HatLLM masks attention within items, thereby compelling the model to capture cross-item correlations. This progressive, layer-wise approach enables LLMs to jointly model both token-level and item-level dependencies. Extensive experiments on three real-world datasets demonstrate that HatLLM achieves significant performance gains (9.13% on average) over existing LLM-based methods. 

---
# Next Interest Flow: A Generative Pre-training Paradigm for Recommender Systems by Modeling All-domain Movelines 

**Authors**: Chen Gao, Zixin Zhao, Lv Shao, Tong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11317)  

**Abstract**: Click-Through Rate (CTR) prediction, a cornerstone of modern recommender systems, has been dominated by discriminative models that react to past user behavior rather than proactively modeling user intent. Existing generative paradigms attempt to address this but suffer from critical limitations: Large Language Model (LLM) based methods create a semantic mismatch by forcing e-commerce signals into a linguistic space, while ID-based generation is constrained by item memorization and cold-start issues. To overcome these limitations, we propose a novel generative pre-training paradigm. Our model learns to predict the Next Interest Flow, a dense vector sequence representing a user's future intent, while simultaneously modeling its internal Interest Diversity and Interest Evolution Velocity to ensure the representation is both rich and coherent. However, this two-stage approach introduces a critical objective mismatch between the generative and discriminative stages. We resolve this via a bidirectional alignment strategy, which harmonizes the two stages through cross-stage weight initialization and a dynamic Semantic Alignment Module for fine-tuning. Additionally, we enhance the underlying discriminative model with a Temporal Sequential Pairwise (TSP) mechanism to better capture temporal causality. We present the All-domain Moveline Evolution Network (AMEN), a unified framework implementing our entire pipeline. Extensive offline experiments validate AMEN's superiority over strong baselines, and a large-scale online A/B test demonstrates its significant real-world impact, delivering substantial improvements in key business metrics. 

---
# PairSem: LLM-Guided Pairwise Semantic Matching for Scientific Document Retrieval 

**Authors**: Wonbin Kweon, Runchu Tian, SeongKu Kang, Pengcheng Jiang, Zhiyong Lu, Jiawei Han, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09897)  

**Abstract**: Scientific document retrieval is a critical task for enabling knowledge discovery and supporting research across diverse domains. However, existing dense retrieval methods often struggle to capture fine-grained scientific concepts in texts due to their reliance on holistic embeddings and limited domain understanding. Recent approaches leverage large language models (LLMs) to extract fine-grained semantic entities and enhance semantic matching, but they typically treat entities as independent fragments, overlooking the multi-faceted nature of scientific concepts. To address this limitation, we propose Pairwise Semantic Matching (PairSem), a framework that represents relevant semantics as entity-aspect pairs, capturing complex, multi-faceted scientific concepts. PairSem is unsupervised, base retriever-agnostic, and plug-and-play, enabling precise and context-aware matching without requiring query-document labels or entity annotations. Extensive experiments on multiple datasets and retrievers demonstrate that PairSem significantly improves retrieval performance, highlighting the importance of modeling multi-aspect semantics in scientific information retrieval. 

---
# LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation 

**Authors**: Hengran Zhang, Keping Bi, Jiafeng Guo, Jiaming Zhang, Shuaiqiang Wang, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.11358)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge. While traditional retrieval focuses on relevance, RAG's effectiveness depends on the utility of retrieved passages, i.e., the usefulness in facilitating the generation of an accurate and comprehensive answer. Existing studies often treat utility as a generic attribute, ignoring the fact that different LLMs may benefit differently from the same passage due to variations in internal knowledge and comprehension ability. In this work, we introduce and systematically investigate the notion of LLM-specific utility. Through large-scale experiments across multiple datasets and LLMs, we demonstrate that human-annotated passages are not optimal for LLMs and that ground-truth utilitarian passages are not transferable across different LLMs. These findings highlight the necessity of adopting the LLM-specific utility in RAG research. Our findings indicate that some human-annotated passages are not ground-truth utilitarian passages for specific LLMs, partially due to the varying readability of queries and passages for LLMs, a tendency for which perplexity is a key metric. Based on these findings, we propose a benchmarking procedure for LLM-specific utility judgments. We evaluate existing utility judgment methods on six datasets and find that while verbalized methods using pseudo-answers perform robustly, LLMs struggle to assess utility effectively-failing to reject all passages for known queries and to select truly useful ones for unknown queries. 

---
# CardRewriter: Leveraging Knowledge Cards for Long-Tail Query Rewriting on Short-Video Platforms 

**Authors**: Peiyuan Gong, Feiran Zhu, Yaqi Yin, Chenglei Dai, Chao Zhang, Kai Zheng, Wentian Bao, Jiaxin Mao, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10095)  

**Abstract**: Short-video platforms have rapidly become a new generation of information retrieval systems, where users formulate queries to access desired videos. However, user queries, especially long-tail ones, often suffer from spelling errors, incomplete phrasing, and ambiguous intent, resulting in mismatches between user expectations and retrieved results. While large language models (LLMs) have shown success in long-tail query rewriting within e-commerce, they struggle on short-video platforms, where proprietary content such as short videos, live streams, micro dramas, and user social networks falls outside their training distribution. To address this challenge, we introduce \textbf{CardRewriter}, an LLM-based framework that incorporates domain-specific knowledge to enhance long-tail query rewriting. For each query, our method aggregates multi-source knowledge relevant to the query and summarizes it into an informative and query-relevant knowledge card. This card then guides the LLM to better capture user intent and produce more effective query rewrites. We optimize CardRewriter using a two-stage training pipeline: supervised fine-tuning followed by group relative policy optimization, with a tailored reward system balancing query relevance and retrieval effectiveness. Offline experiments show that CardRewriter substantially improves rewriting quality for queries targeting proprietary content. Online A/B testing further confirms significant gains in long-view rate (LVR) and click-through rate (CTR), along with a notable reduction in initiative query reformulation rate (IQRR). Since September 2025, CardRewriter has been deployed on Kuaishou, one of China's largest short-video platforms, serving hundreds of millions of users daily. 

---
# Beyond the limitation of a single query: Train your LLM for query expansion with Reinforcement Learning 

**Authors**: Shu Zhao, Tan Yu, Anbang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10009)  

**Abstract**: Reasoning-augmented search agents, such as Search-R1, are trained to reason, search, and generate the final answer iteratively. Nevertheless, due to their limited capabilities in reasoning and search, their performance on multi-hop QA benchmarks remains far from satisfactory. To handle complex or compound queries, we train an LLM-based search agent with the native capability of query expansion through reinforcement learning. In each turn, our search agent proposes several query variants, which are searched simultaneously to cover more relevant information. Meanwhile, given limited post-training data and computing resources, it is very challenging for a search agent to master multiple tasks, including query generation, retrieved information understanding, and answer generation. Therefore, we propose incorporating a pre-trained squeezer model that helps the search agent understand the retrieved documents, allowing the search agent to focus on query generation for high retrieval recall. With the assistance of the squeezer model, we discover that even a small-scale 3B LLM can demonstrate a strong capability of query expansion and achieve state-of-the-art accuracy on the multi-hop QA benchmarks. To be specific, our experiments across seven question-answering benchmarks demonstrate that our method, named ExpandSearch, achieves an average improvement of 4.4% compared to state-of-the-art baselines, with strong gains on multi-hop reasoning tasks requiring diverse evidence aggregation. 

---
# DRIFT: Decompose, Retrieve, Illustrate, then Formalize Theorems 

**Authors**: Meiru Zhang, Philipp Borchert, Milan Gritta, Gerasimos Lampouras  

**Link**: [PDF](https://arxiv.org/pdf/2510.10815)  

**Abstract**: Automating the formalization of mathematical statements for theorem proving remains a major challenge for Large Language Models (LLMs). LLMs struggle to identify and utilize the prerequisite mathematical knowledge and its corresponding formal representation in languages like Lean. Current retrieval-augmented autoformalization methods query external libraries using the informal statement directly, but overlook a fundamental limitation: informal mathematical statements are often complex and offer limited context on the underlying math concepts. To address this, we introduce DRIFT, a novel framework that enables LLMs to decompose informal mathematical statements into smaller, more tractable ''sub-components''. This facilitates targeted retrieval of premises from mathematical libraries such as Mathlib. Additionally, DRIFT retrieves illustrative theorems to help models use premises more effectively in formalization tasks. We evaluate DRIFT across diverse benchmarks (ProofNet, ConNF, and MiniF2F-test) and find that it consistently improves premise retrieval, nearly doubling the F1 score compared to the DPR baseline on ProofNet. Notably, DRIFT demonstrates strong performance on the out-of-distribution ConNF benchmark, with BEq+@10 improvements of 37.14% and 42.25% using GPT-4.1 and DeepSeek-V3.1, respectively. Our analysis shows that retrieval effectiveness in mathematical autoformalization depends heavily on model-specific knowledge boundaries, highlighting the need for adaptive retrieval strategies aligned with each model's capabilities. 

---
# ImCoref-CeS: An Improved Lightweight Pipeline for Coreference Resolution with LLM-based Checker-Splitter Refinement 

**Authors**: Kangyang Luo, Yuzhuo Bai, Shuzheng Si, Cheng Gao, Zhitong Wang, Yingli Shen, Wenhao Li, Zhu Liu, Yufeng Han, Jiayi Wu, Cunliang Kong, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.10241)  

**Abstract**: Coreference Resolution (CR) is a critical task in Natural Language Processing (NLP). Current research faces a key dilemma: whether to further explore the potential of supervised neural methods based on small language models, whose detect-then-cluster pipeline still delivers top performance, or embrace the powerful capabilities of Large Language Models (LLMs). However, effectively combining their strengths remains underexplored. To this end, we propose \textbf{ImCoref-CeS}, a novel framework that integrates an enhanced supervised model with LLM-based reasoning. First, we present an improved CR method (\textbf{ImCoref}) to push the performance boundaries of the supervised neural method by introducing a lightweight bridging module to enhance long-text encoding capability, devising a biaffine scorer to comprehensively capture positional information, and invoking a hybrid mention regularization to improve training efficiency. Importantly, we employ an LLM acting as a multi-role Checker-Splitter agent to validate candidate mentions (filtering out invalid ones) and coreference results (splitting erroneous clusters) predicted by ImCoref. Extensive experiments demonstrate the effectiveness of ImCoref-CeS, which achieves superior performance compared to existing state-of-the-art (SOTA) methods. 

---
# Stop DDoS Attacking the Research Community with AI-Generated Survey Papers 

**Authors**: Jianghao Lin, Rong Shan, Jiachen Zhu, Yunjia Xi, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09686)  

**Abstract**: Survey papers are foundational to the scholarly progress of research communities, offering structured overviews that guide both novices and experts across disciplines. However, the recent surge of AI-generated surveys, especially enabled by large language models (LLMs), has transformed this traditionally labor-intensive genre into a low-effort, high-volume output. While such automation lowers entry barriers, it also introduces a critical threat: the phenomenon we term the "survey paper DDoS attack" to the research community. This refers to the unchecked proliferation of superficially comprehensive but often redundant, low-quality, or even hallucinated survey manuscripts, which floods preprint platforms, overwhelms researchers, and erodes trust in the scientific record. In this position paper, we argue that we must stop uploading massive amounts of AI-generated survey papers (i.e., survey paper DDoS attack) to the research community, by instituting strong norms for AI-assisted review writing. We call for restoring expert oversight and transparency in AI usage and, moreover, developing new infrastructures such as Dynamic Live Surveys, community-maintained, version-controlled repositories that blend automated updates with human curation. Through quantitative trend analysis, quality audits, and cultural impact discussion, we show that safeguarding the integrity of surveys is no longer optional but imperative to the research community. 

---
# Enhancing Long Chain-of-Thought Reasoning through Multi-Path Plan Aggregation 

**Authors**: Siheng Xiong, Ali Payani, Faramarz Fekri  

**Link**: [PDF](https://arxiv.org/pdf/2510.11620)  

**Abstract**: Inference-time scaling enhances the reasoning ability of a language model (LM) by extending its chain-of-thought (CoT). However, existing approaches typically generate the entire reasoning chain in a single forward pass, which often leads to CoT derailment, i.e., the reasoning trajectory drifting off course due to compounding errors. This problem is particularly severe for smaller LMs with long CoTs due to their limited capacity. To address this, we analyze raw long CoTs and uncover a reasoning hierarchy consisting of planning and execution steps. Our analysis reveals that most reasoning errors stem from incorrect planning. Motivated by this observation, we propose Multi-Path Plan Aggregation (MPPA), a framework that augments single-pass reasoning with plan exploration and aggregation. Following a variable interval schedule based on the token position, MPPA generates multiple candidate plans and aggregates them into a refined planning step. To maintain efficiency, we adopt a minimal design in which the base LM serves as the primary policy, while a lightweight LoRA module implements the plan aggregation policy. We further observe that outcome-reward RL is inefficient for long trajectories (e.g., exceeding 4K tokens). To overcome this, we introduce online Step-DPO, a process-level preference optimization scheme that leverages Twisted Sequential Monte Carlo (TSMC) to provide scalable stepwise supervision using small LMs. This yields more efficient training, improved stability, and higher accuracy. Extensive experiments on challenging math, science, and logical reasoning benchmarks demonstrate that, with only 10% SFT data and 5% of preference pairs, our method outperforms both the DeepSeek-R1 distillation baseline and the outcome-reward RL baseline across multiple base models and tasks. 

---
# Are Large Reasoning Models Interruptible? 

**Authors**: Tsung-Han Wu, Mihran Miroyan, David M. Chan, Trevor Darrell, Narges Norouzi, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2510.11713)  

**Abstract**: Large Reasoning Models (LRMs) excel at complex reasoning but are traditionally evaluated in static, "frozen world" settings: model responses are assumed to be instantaneous, and the context of a request is presumed to be immutable over the duration of the response. While generally true for short-term tasks, the "frozen world" assumption breaks down in modern reasoning tasks such as assistive programming, where models may take hours to think through problems and code may change dramatically from the time the model starts thinking to the model's final output. In this work, we challenge the frozen world assumption and evaluate LRM robustness under two realistic dynamic scenarios: interruptions, which test the quality of the model's partial outputs on a limited budget, and dynamic context, which tests model adaptation to in-flight changes. Across mathematics and programming benchmarks that require long-form reasoning, static evaluations consistently overestimate robustness: even state-of-the-art LRMs, which achieve high accuracy in static settings, can fail unpredictably when interrupted or exposed to changing context, with performance dropping by up to 60% when updates are introduced late in the reasoning process. Our analysis further reveals several novel failure modes, including reasoning leakage, where models fold the reasoning into their final answer when interrupted; panic, where under time pressure models abandon reasoning entirely and return incorrect answers; and self-doubt, where performance degrades while incorporating updated information. 

---
# When Agents Trade: Live Multi-Market Trading Benchmark for LLM Agents 

**Authors**: Lingfei Qian, Xueqing Peng, Yan Wang, Vincent Jim Zhang, Huan He, Hanley Smith, Yi Han, Yueru He, Haohang Li, Yupeng Cao, Yangyang Yu, Alejandro Lopez-Lira, Peng Lu, Jian-Yun Nie, Guojun Xiong, Jimin Huang, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2510.11695)  

**Abstract**: Although Large Language Model (LLM)-based agents are increasingly used in financial trading, it remains unclear whether they can reason and adapt in live markets, as most studies test models instead of agents, cover limited periods and assets, and rely on unverified data. To address these gaps, we introduce Agent Market Arena (AMA), the first lifelong, real-time benchmark for evaluating LLM-based trading agents across multiple markets. AMA integrates verified trading data, expert-checked news, and diverse agent architectures within a unified trading framework, enabling fair and continuous comparison under real conditions. It implements four agents, including InvestorAgent as a single-agent baseline, TradeAgent and HedgeFundAgent with different risk styles, and DeepFundAgent with memory-based reasoning, and evaluates them across GPT-4o, GPT-4.1, Claude-3.5-haiku, Claude-sonnet-4, and Gemini-2.0-flash. Live experiments on both cryptocurrency and stock markets demonstrate that agent frameworks display markedly distinct behavioral patterns, spanning from aggressive risk-taking to conservative decision-making, whereas model backbones contribute less to outcome variation. AMA thus establishes a foundation for rigorous, reproducible, and continuously evolving evaluation of financial reasoning and trading intelligence in LLM-based agents. 

---
# LLM-Oriented Token-Adaptive Knowledge Distillation 

**Authors**: Xurong Xie, Zhucun Xue, Jiafu Wu, Jian Li, Yabiao Wang, Xiaobin Hu, Yong Liu, Jiangning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11615)  

**Abstract**: Knowledge distillation (KD) is a key technique for compressing large-scale language models (LLMs), yet prevailing logit-based methods typically employ static strategies that are misaligned with the dynamic learning process of student models. These methods typically treat all tokens indiscriminately and apply a single, fixed temperature, resulting in suboptimal knowledge transfer. To address these limitations, we propose LLM-Oriented Token-Adaptive Knowledge Distillation (AdaKD), a novel framework that adapts the distillation process to the real-time learning state of each token. AdaKD consists of two synergistic modules driven by a unified token difficulty metric. First, our Loss-Driven Adaptive Token Focusing (LATF) module dynamically adjusts the distillation focus by monitoring the student's learning stability, concentrating computational resources on the most valuable tokens at each training phase. Second, we introduce Inverse Difficulty Temperature Scaling (IDTS), a counterintuitive yet effective token-level temperature strategy. It employs low temperatures for difficult tokens for targeted error correction, and high temperatures for easy tokens to encourage students to learn from the teacher's complete and smooth output distribution, thereby enhancing generalization. As a plug-and-play framework, AdaKD can consistently improve the performance of various distillation methods on multiple model architectures and benchmarks. 

---
# ACADREASON: Exploring the Limits of Reasoning Models with Academic Research Problems 

**Authors**: Xin Gui, King Zhu, JinCheng Ren, Qianben Chen, Zekun Moore Wang, Yizhi LI, Xinpeng Liu, Xiaowan Li, Wenli Ren, Linyu Miao, Tianrui Qin, Ziqi Shu, He Zhu, Xiangru Tang, Dingfeng Shi, Jiaheng Liu, Yuchen Eleanor Jiang, Minghao Liu, Ge Zhang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.11652)  

**Abstract**: In recent years, the research focus of large language models (LLMs) and agents has shifted increasingly from demonstrating novel capabilities to complex reasoning and tackling challenging tasks. However, existing evaluations focus mainly on math/code contests or general tasks, while existing multi-domain academic benchmarks lack sufficient reasoning depth, leaving the field without a rigorous benchmark for high-level reasoning. To fill this gap, we introduce the Acadreason benchmark, designed to evaluate the ability of LLMs and agents to acquire and reason over academic knowledge. It consists of 50 expert-annotated academic problems across five high-reasoning domains, including computer science, economics, law, mathematics, and philosophy. All questions are sourced from top-tier publications in recent years and undergo rigorous annotation and quality control to ensure they are both challenging and answerable. We conduct systematic evaluations of over 10 mainstream LLMs and agents. The results show that most LLMs scored below 20 points, with even the cutting-edge GPT-5 achieving only 16 points. While agents achieved higher scores, none exceeded 40 points. This demonstrates the current capability gap between LLMs and agents in super-intelligent academic research tasks and highlights the challenges of Acadreason. 

---
# LLMAtKGE: Large Language Models as Explainable Attackers against Knowledge Graph Embeddings 

**Authors**: Ting Li, Yang Yang, Yipeng Yu, Liang Yao, Guoqing Chao, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11584)  

**Abstract**: Adversarial attacks on knowledge graph embeddings (KGE) aim to disrupt the model's ability of link prediction by removing or inserting triples. A recent black-box method has attempted to incorporate textual and structural information to enhance attack performance. However, it is unable to generate human-readable explanations, and exhibits poor generalizability. In the past few years, large language models (LLMs) have demonstrated powerful capabilities in text comprehension, generation, and reasoning. In this paper, we propose LLMAtKGE, a novel LLM-based framework that selects attack targets and generates human-readable explanations. To provide the LLM with sufficient factual context under limited input constraints, we design a structured prompting scheme that explicitly formulates the attack as multiple-choice questions while incorporating KG factual evidence. To address the context-window limitation and hesitation issues, we introduce semantics-based and centrality-based filters, which compress the candidate set while preserving high recall of attack-relevant information. Furthermore, to efficiently integrate both semantic and structural information into the filter, we precompute high-order adjacency and fine-tune the LLM with a triple classification task to enhance filtering performance. Experiments on two widely used knowledge graph datasets demonstrate that our attack outperforms the strongest black-box baselines and provides explanations via reasoning, and showing competitive performance compared with white-box methods. Comprehensive ablation and case studies further validate its capability to generate explanations. 

---
# Survey Response Generation: Generating Closed-Ended Survey Responses In-Silico with Large Language Models 

**Authors**: Georg Ahnert, Anna-Carolina Haensch, Barbara Plank, Markus Strohmaier  

**Link**: [PDF](https://arxiv.org/pdf/2510.11586)  

**Abstract**: Many in-silico simulations of human survey responses with large language models (LLMs) focus on generating closed-ended survey responses, whereas LLMs are typically trained to generate open-ended text instead. Previous research has used a diverse range of methods for generating closed-ended survey responses with LLMs, and a standard practice remains to be identified. In this paper, we systematically investigate the impact that various Survey Response Generation Methods have on predicted survey responses. We present the results of 32 mio. simulated survey responses across 8 Survey Response Generation Methods, 4 political attitude surveys, and 10 open-weight language models. We find significant differences between the Survey Response Generation Methods in both individual-level and subpopulation-level alignment. Our results show that Restricted Generation Methods perform best overall, and that reasoning output does not consistently improve alignment. Our work underlines the significant impact that Survey Response Generation Methods have on simulated survey responses, and we develop practical recommendations on the application of Survey Response Generation Methods. 

---
# MeTA-LoRA: Data-Efficient Multi-Task Fine-Tuning for Large Language Models 

**Authors**: Bo Cheng, Xu Wang, Jinda Liu, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11598)  

**Abstract**: Low-Rank Adaptation (LoRA) has emerged as one of the most widely used parameter-efficient fine-tuning (PEFT) methods for adapting large language models (LLMs) to downstream tasks. While highly effective in single-task settings, it struggles to efficiently leverage inter-task knowledge in complex multi-task learning scenarios, often requiring substantial task-specific data to achieve optimal performance. To address this limitation, we introduce MeTA-LoRA, a two-stage optimization framework that significantly improves data efficiency in multi-task adaptation. In the first stage, task-specific LoRA adapters are learned using only a few samples from each involved dataset, enabling rapid adaptation without large-scale supervision. In the second stage, the shared LoRA adapter is updated by aggregating gradients from multiple tasks to promote knowledge transfer across tasks, further reducing data usage by leveraging common patterns. In both multi-task learning and multilingual learning scenarios, our method matches or surpasses the performance of traditional full-data LoRA fine-tuning approaches, while using significantly less task-specific data. 

---
# Culturally-Aware Conversations: A Framework & Benchmark for LLMs 

**Authors**: Shreya Havaldar, Sunny Rai, Young-Min Cho, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2510.11563)  

**Abstract**: Existing benchmarks that measure cultural adaptation in LLMs are misaligned with the actual challenges these models face when interacting with users from diverse cultural backgrounds. In this work, we introduce the first framework and benchmark designed to evaluate LLMs in realistic, multicultural conversational settings. Grounded in sociocultural theory, our framework formalizes how linguistic style - a key element of cultural communication - is shaped by situational, relational, and cultural context. We construct a benchmark dataset based on this framework, annotated by culturally diverse raters, and propose a new set of desiderata for cross-cultural evaluation in NLP: conversational framing, stylistic sensitivity, and subjective correctness. We evaluate today's top LLMs on our benchmark and show that these models struggle with cultural adaptation in a conversational setting. 

---
# Hallucination Detection via Internal States and Structured Reasoning Consistency in Large Language Models 

**Authors**: Yusheng Song, Lirong Qiu, Xi Zhang, Zhihao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11529)  

**Abstract**: The detection of sophisticated hallucinations in Large Language Models (LLMs) is hampered by a ``Detection Dilemma'': methods probing internal states (Internal State Probing) excel at identifying factual inconsistencies but fail on logical fallacies, while those verifying externalized reasoning (Chain-of-Thought Verification) show the opposite behavior. This schism creates a task-dependent blind spot: Chain-of-Thought Verification fails on fact-intensive tasks like open-domain QA where reasoning is ungrounded, while Internal State Probing is ineffective on logic-intensive tasks like mathematical reasoning where models are confidently wrong. We resolve this with a unified framework that bridges this critical gap. However, unification is hindered by two fundamental challenges: the Signal Scarcity Barrier, as coarse symbolic reasoning chains lack signals directly comparable to fine-grained internal states, and the Representational Alignment Barrier, a deep-seated mismatch between their underlying semantic spaces. To overcome these, we introduce a multi-path reasoning mechanism to obtain more comparable, fine-grained signals, and a segment-aware temporalized cross-attention module to adaptively fuse these now-aligned representations, pinpointing subtle dissonances. Extensive experiments on three diverse benchmarks and two leading LLMs demonstrate that our framework consistently and significantly outperforms strong baselines. Our code is available: this https URL. 

---
# Investigating Large Language Models' Linguistic Abilities for Text Preprocessing 

**Authors**: Marco Braga, Gian Carlo Milanese, Gabriella Pasi  

**Link**: [PDF](https://arxiv.org/pdf/2510.11482)  

**Abstract**: Text preprocessing is a fundamental component of Natural Language Processing, involving techniques such as stopword removal, stemming, and lemmatization to prepare text as input for further processing and analysis. Despite the context-dependent nature of the above techniques, traditional methods usually ignore contextual information. In this paper, we investigate the idea of using Large Language Models (LLMs) to perform various preprocessing tasks, due to their ability to take context into account without requiring extensive language-specific annotated resources. Through a comprehensive evaluation on web-sourced data, we compare LLM-based preprocessing (specifically stopword removal, lemmatization and stemming) to traditional algorithms across multiple text classification tasks in six European languages. Our analysis indicates that LLMs are capable of replicating traditional stopword removal, lemmatization, and stemming methods with accuracies reaching 97%, 82%, and 74%, respectively. Additionally, we show that ML algorithms trained on texts preprocessed by LLMs achieve an improvement of up to 6% with respect to the $F_1$ measure compared to traditional techniques. Our code, prompts, and results are publicly available at this https URL. 

---
# KnowRL: Teaching Language Models to Know What They Know 

**Authors**: Sahil Kale, Devendra Singh Dhami  

**Link**: [PDF](https://arxiv.org/pdf/2510.11407)  

**Abstract**: Truly reliable AI requires more than simply scaling up knowledge; it demands the ability to know what it knows and when it does not. Yet recent research shows that even the best LLMs misjudge their own competence in more than one in five cases, making any response born of such internal uncertainty impossible to fully trust. Inspired by self-improvement reinforcement learning techniques that require minimal data, we present a simple but powerful framework KnowRL that strengthens a model's internal understanding of its own feasibility boundaries, enabling safer and more responsible behaviour. Our framework combines two components: (i) introspection, where the model generates and classifies tasks it judges feasible or infeasible, and (ii) consensus-based rewarding, where stability of self-knowledge assessment is reinforced through internal agreement. By using internally generated data, this design strengthens consistency in self-knowledge and entirely avoids costly external supervision. In experiments on LLaMA-3.1-8B and Qwen-2.5-7B, KnowRL steadily improved self-knowledge, validated by both intrinsic self-consistency and extrinsic benchmarking. With nothing more than a small seed set and no external supervision, our method drove gains as high as 28% in accuracy and 12% in F1, outperforming baselines in just a few iterations. Our framework essentially unlocks the untapped capacity of LLMs to self-improve their knowledge awareness, opening the door to reliable, more accountable AI and safer deployment in critical applications. Owing to its simplicity and independence from external effort, we encourage applying this reliability-enhancing process to all future models. 

---
# Scaling Language-Centric Omnimodal Representation Learning 

**Authors**: Chenghao Xiao, Hou Pong Chan, Hao Zhang, Weiwen Xu, Mahani Aljunied, Yu Rong  

**Link**: [PDF](https://arxiv.org/pdf/2510.11693)  

**Abstract**: Recent multimodal embedding approaches leveraging multimodal large language models (MLLMs) fine-tuned with contrastive learning (CL) have shown promising results, yet the underlying reasons behind their superiority remain underexplored. This work argues that a crucial advantage of MLLM-based approaches stems from implicit cross-modal alignment achieved during generative pretraining, where the language decoder learns to exploit multimodal signals within a shared representation space for generating unimodal outputs. Through analysis of anisotropy and kernel similarity structure, we empirically confirm that latent alignment emerges within MLLM representations, allowing CL to serve as a lightweight refinement stage. Leveraging this insight, we propose a Language-Centric Omnimodal Embedding framework, termed LCO-Emb. Extensive experiments across diverse backbones and benchmarks demonstrate its effectiveness, achieving state-of-the-art performance across modalities. Furthermore, we identify a Generation-Representation Scaling Law (GRSL), showing that the representational capabilities gained through contrastive refinement scales positively with the MLLM's generative capabilities. This suggests that improving generative abilities evolves as an effective paradigm for enhancing representation quality. We provide a theoretical explanation of GRSL, which formally links the MLLM's generative quality to the upper bound on its representation performance, and validate it on a challenging, low-resource visual-document retrieval task, showing that continual generative pretraining before CL can further enhance the potential of a model's embedding capabilities. Codes, models, and resources are available at this https URL. 

---
# Do LLMs "Feel"? Emotion Circuits Discovery and Control 

**Authors**: Chenxi Wang, Yixuan Zhang, Ruiji Yu, Yufei Zheng, Lang Gao, Zirui Song, Zixiang Xu, Gus Xia, Huishuai Zhang, Dongyan Zhao, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.11328)  

**Abstract**: As the demand for emotional intelligence in large language models (LLMs) grows, a key challenge lies in understanding the internal mechanisms that give rise to emotional expression and in controlling emotions in generated text. This study addresses three core questions: (1) Do LLMs contain context-agnostic mechanisms shaping emotional expression? (2) What form do these mechanisms take? (3) Can they be harnessed for universal emotion control? We first construct a controlled dataset, SEV (Scenario-Event with Valence), to elicit comparable internal states across emotions. Subsequently, we extract context-agnostic emotion directions that reveal consistent, cross-context encoding of emotion (Q1). We identify neurons and attention heads that locally implement emotional computation through analytical decomposition and causal analysis, and validate their causal roles via ablation and enhancement interventions. Next, we quantify each sublayer's causal influence on the model's final emotion representation and integrate the identified local components into coherent global emotion circuits that drive emotional expression (Q2). Directly modulating these circuits achieves 99.65% emotion-expression accuracy on the test set, surpassing prompting- and steering-based methods (Q3). To our knowledge, this is the first systematic study to uncover and validate emotion circuits in LLMs, offering new insights into interpretability and controllable emotional intelligence. 

---
# Are Large Language Models Effective Knowledge Graph Constructors? 

**Authors**: Ruirui Chen, Weifeng Jiang, Chengwei Qin, Bo Xiong, Fiona Liausvia, Dongkyu Choi, Boon Kiat Quek  

**Link**: [PDF](https://arxiv.org/pdf/2510.11297)  

**Abstract**: Knowledge graphs (KGs) are vital for knowledge-intensive tasks and have shown promise in reducing hallucinations in large language models (LLMs). However, constructing high-quality KGs remains difficult, requiring accurate information extraction and structured representations that support interpretability and downstream utility. Existing LLM-based approaches often focus narrowly on entity and relation extraction, limiting coverage to sentence-level contexts or relying on predefined schemas. We propose a hierarchical extraction framework that organizes information at multiple levels, enabling the creation of semantically rich and well-structured KGs. Using state-of-the-art LLMs, we extract and construct knowledge graphs and evaluate them comprehensively from both structural and semantic perspectives. Our results highlight the strengths and shortcomings of current LLMs in KG construction and identify key challenges for future work. To advance research in this area, we also release a curated dataset of LLM-generated KGs derived from research papers on children's mental well-being. This resource aims to foster more transparent, reliable, and impactful applications in high-stakes domains such as healthcare. 

---
# Towards Real-Time Fake News Detection under Evidence Scarcity 

**Authors**: Guangyu Wei, Ke Han, Yueming Lyu, Yu Luo, Yue Jiang, Caifeng Shan, Nicu Sebe  

**Link**: [PDF](https://arxiv.org/pdf/2510.11277)  

**Abstract**: Fake news detection becomes particularly challenging in real-time scenarios, where emerging events often lack sufficient supporting evidence. Existing approaches often rely heavily on external evidence and therefore struggle to generalize under evidence scarcity. To address this issue, we propose Evaluation-Aware Selection of Experts (EASE), a novel framework for real-time fake news detection that dynamically adapts its decision-making process according to the assessed sufficiency of available evidence. EASE introduces a sequential evaluation mechanism comprising three independent perspectives: (1) Evidence-based evaluation, which assesses evidence and incorporates it into decision-making only when the evidence is sufficiently supportive; (2) Reasoning-based evaluation, which leverages the world knowledge of large language models (LLMs) and applies them only when their reliability is adequately established; and (3) Sentiment-based fallback, which integrates sentiment cues when neither evidence nor reasoning is reliable. To enhance the accuracy of evaluation processes, EASE employs instruction tuning with pseudo labels to guide each evaluator in justifying its perspective-specific knowledge through interpretable reasoning. Furthermore, the expert modules integrate the evaluators' justified assessments with the news content to enable evaluation-aware decision-making, thereby enhancing overall detection accuracy. Moreover, we introduce RealTimeNews-25, a new benchmark comprising recent news for evaluating model generalization on emerging news with limited evidence. Extensive experiments demonstrate that EASE not only achieves state-of-the-art performance across multiple benchmarks, but also significantly improves generalization to real-time news. The code and dataset are available: this https URL. 

---
# Do Psychometric Tests Work for Large Language Models? Evaluation of Tests on Sexism, Racism, and Morality 

**Authors**: Jana Jung, Marlene Lutz, Indira Sen, Markus Strohmaier  

**Link**: [PDF](https://arxiv.org/pdf/2510.11254)  

**Abstract**: Psychometric tests are increasingly used to assess psychological constructs in large language models (LLMs). However, it remains unclear whether these tests -- originally developed for humans -- yield meaningful results when applied to LLMs. In this study, we systematically evaluate the reliability and validity of human psychometric tests for three constructs: sexism, racism, and morality. We find moderate reliability across multiple item and prompt variations. Validity is evaluated through both convergent (i.e., testing theory-based inter-test correlations) and ecological approaches (i.e., testing the alignment between tests scores and behavior in real-world downstream tasks). Crucially, we find that psychometric test scores do not align, and in some cases even negatively correlate with, model behavior in downstream tasks, indicating low ecological validity. Our results highlight that systematic evaluations of psychometric tests is essential before interpreting their scores. They also suggest that psychometric tests designed for humans cannot be applied directly to LLMs without adaptation. 

---
# Beyond Survival: Evaluating LLMs in Social Deduction Games with Human-Aligned Strategies 

**Authors**: Zirui Song, Yuan Huang, Junchang Liu, Haozhe Luo, Chenxi Wang, Lang Gao, Zixiang Xu, Mingfei Han, Xiaojun Chang, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.11389)  

**Abstract**: Social deduction games like Werewolf combine language, reasoning, and strategy, providing a testbed for studying natural language and social intelligence. However, most studies reduce the game to LLM-based self-play, yielding templated utterances and anecdotal cases that overlook the richness of social gameplay. Evaluation further relies on coarse metrics such as survival time or subjective scoring due to the lack of quality reference data. To address these gaps, we curate a high-quality, human-verified multimodal Werewolf dataset containing over 100 hours of video, 32.4M utterance tokens, and 15 rule variants. Based on this dataset, we propose a novel strategy-alignment evaluation that leverages the winning faction's strategies as ground truth in two stages: 1) Speech evaluation, formulated as multiple-choice-style tasks that assess whether the model can adopt appropriate stances across five dimensions of social ability; and 2) Decision evaluation, which assesses the model's voting choices and opponent-role inferences. This framework enables a fine-grained evaluation of models' linguistic and reasoning capabilities, while capturing their ability to generate strategically coherent gameplay. Our experiments show that state-of-the-art LLMs show diverse performance, with roughly half remain below 0.50, revealing clear gaps in deception and counterfactual reasoning. We hope our dataset further inspires research on language, reasoning, and strategy in multi-agent interaction. 

---
# The Curious Case of Factual (Mis)Alignment between LLMs' Short- and Long-Form Answers 

**Authors**: Saad Obaid ul Islam, Anne Lauscher, Goran Glavaš  

**Link**: [PDF](https://arxiv.org/pdf/2510.11218)  

**Abstract**: Large language models (LLMs) can correctly answer "When was Einstein born?" yet fail to provide the same date when writing about Einstein's life revealing a fundamental inconsistency in how models access factual knowledge across task complexities. While models display impressive accuracy on factual question-answering benchmarks, the reliability gap between simple and complex queries remains poorly understood, eroding their trustworthiness. In this work, we introduce Short-Long Form Alignment for Factual Question Answering (SLAQ), a controlled evaluation framework that compares LLMs' answers to the same factual questions asked (a) in isolation (short) vs. (b) integrated into complex queries (long). Looking at 16 LLMs across 600 queries, we find a systematic misalignment of answers to the corresponding short and long queries. We further uncover position-dependent accuracy loss and momentum effects where consecutive correct or incorrect answers create self-reinforcing patterns. Through mechanistic analysis, we find that aligned facts activate overlapping model internals, and that metrics based on mechanistic similarity can predict short-long answer alignment with up to 78% accuracy. Our work establishes factual consistency over query complexity as an important aspect of LLMs' trustworthiness and challenges current evaluation practices, which implicitly assume that good performance for simple factual queries implies reliability in more complex knowledge-seeking tasks too. 

---
# Discursive Circuits: How Do Language Models Understand Discourse Relations? 

**Authors**: Yisong Miao, Min-Yen Kan  

**Link**: [PDF](https://arxiv.org/pdf/2510.11210)  

**Abstract**: Which components in transformer language models are responsible for discourse understanding? We hypothesize that sparse computational graphs, termed as discursive circuits, control how models process discourse relations. Unlike simpler tasks, discourse relations involve longer spans and complex reasoning. To make circuit discovery feasible, we introduce a task called Completion under Discourse Relation (CuDR), where a model completes a discourse given a specified relation. To support this task, we construct a corpus of minimal contrastive pairs tailored for activation patching in circuit discovery. Experiments show that sparse circuits ($\approx 0.2\%$ of a full GPT-2 model) recover discourse understanding in the English PDTB-based CuDR task. These circuits generalize well to unseen discourse frameworks such as RST and SDRT. Further analysis shows lower layers capture linguistic features such as lexical semantics and coreference, while upper layers encode discourse-level abstractions. Feature utility is consistent across frameworks (e.g., coreference supports Expansion-like relations). 

---
# Enhancing LLM Reasoning via Non-Human-Like Reasoning Path Preference Optimization 

**Authors**: Junjie Lu, Yuliang Liu, Chaofeng Qu, Wei Shen, Zhouhan Lin, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11104)  

**Abstract**: Current approaches for strengthening LLM reasoning tend to introduce a training bias toward human-like reasoning trajectories. In step-wise preference optimization, in particular, dependence on human or higher-capacity model annotations for intermediate steps limits exploration of alternative, non-human-like reasoning paths and thus constrains achievable performance. Furthermore, through a small-scale pilot study, we observed that in approximately 75% of cases, the model's first erroneous step occurs after the lowest-confidence point. This suggests that guiding the model at its lowest-confidence point before an error provides more accurate supervision than locating the first explicit error. In this paper, we propose Confidence-Guided Reasoning Path Preference Optimization (CGPO), a method that leverages a confidence signal to identify points of maximal uncertainty in the model's reasoning process and applies self-generated, non-human-like reasoning-path guidance to mitigate trajectory drift. Our experiments span diverse models applied to both code and mathematical reasoning tasks. The results show that, with the same amount of training data, our method using data generated by a small model can achieve better performance in most cases compared with approaches using data generated by a strong model or human-annotated. 

---
# LogiNumSynth: Synthesizing Joint Logical-Numerical Reasoning Problems for Language Models 

**Authors**: Yiwei Liu, Yucheng Li, Xiao Li, Gong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.11031)  

**Abstract**: Joint logical-numerical reasoning remains a major challenge for language models, yet existing datasets rely on fixed rule sets and offer limited control over task complexity, constraining their generalizability for evaluation and training. We present LogiNumSynth, a flexible natural language problem synthesizer that synthesizes tasks requiring proficiency in joint logical reasoning (e.g., rule-based reasoning) and numerical reasoning (e.g., arithmetic computation). LogiNumSynth supports fine-grained control over reasoning world richness, logical reasoning depth, and the complexity of numerical computations, enabling flexible data synthesis across difficulty levels. We demonstrate three key contributions: (1) Synthesizer -- synthesizing fully controllable joint reasoning tasks over natural language; (2) Evaluation & Process Analysis -- evaluating both process accuracy and answer accuracy; (3) Targeted Training -- using synthesized data to enhance LLMs' reasoning performance. Experiments with multiple LLMs highlight persistent weaknesses in logical-numerical reasoning, showing that LogiNumSynth can serve as both a diagnostic tool and a source of targeted supervision for advancing integrated reasoning skills. 

---
# Enabling Doctor-Centric Medical AI with LLMs through Workflow-Aligned Tasks and Benchmarks 

**Authors**: Wenya Xie, Qingying Xiao, Yu Zheng, Xidong Wang, Junying Chen, Ke Ji, Anningzhe Gao, Prayag Tiwari, Xiang Wan, Feng Jiang, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11040)  

**Abstract**: The rise of large language models (LLMs) has transformed healthcare by offering clinical guidance, yet their direct deployment to patients poses safety risks due to limited domain expertise. To mitigate this, we propose repositioning LLMs as clinical assistants that collaborate with experienced physicians rather than interacting with patients directly. We conduct a two-stage inspiration-feedback survey to identify real-world needs in clinical workflows. Guided by this, we construct DoctorFLAN, a large-scale Chinese medical dataset comprising 92,000 Q&A instances across 22 clinical tasks and 27 specialties. To evaluate model performance in doctor-facing applications, we introduce DoctorFLAN-test (550 single-turn Q&A items) and DotaBench (74 multi-turn conversations). Experimental results with over ten popular LLMs demonstrate that DoctorFLAN notably improves the performance of open-source LLMs in medical contexts, facilitating their alignment with physician workflows and complementing existing patient-oriented models. This work contributes a valuable resource and framework for advancing doctor-centered medical LLM development 

---
# DND: Boosting Large Language Models with Dynamic Nested Depth 

**Authors**: Tieyuan Chen, Xiaodong Chen, Haoxing Chen, Zhenzhong Lan, Weiyao Lin, Jianguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.11001)  

**Abstract**: We introduce Dynamic Nested Depth (DND), a novel method that improves performance for off-the-shelf LLMs by selecting critical tokens to reprocess in a nested depth manner. Specifically, at the end of the given transformer layer, DND identifies more critical tokens with a router and feeds them back for an extra round of processing, effectively ``reviewing" difficult tokens while avoiding redundant computation for easier ones. The dynamic selection mechanism is tailored for precise control via two novel strategies: a router controlling loss to enhance token selection distinguishability, and a threshold control scheme to ensure selection stability. We demonstrate the effectiveness of DND by directly integrating it into pre-trained dense and MoE models during a post-training phase. On diverse benchmarks, this approach boosts the performances of the dense Qwen3-1.7B by 1.88% and the MoE Qwen3-30B-A3B by 0.87%, all with a minimal parameter and computing increase. 

---
# Enhancing Large Language Model Reasoning via Selective Critical Token Fine-Tuning 

**Authors**: Zhiwen Ruan, Yixia Li, He Zhu, Yun Chen, Peng Li, Yang Liu, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10974)  

**Abstract**: Large language models (LLMs) primarily rely on supervised fine-tuning (SFT) as a key method to adapt pre-trained models to domain-specific tasks such as mathematical reasoning. However, standard SFT uniformly penalizes all tokens, neglecting that only a small subset of critical tokens determines reasoning correctness. This uniform supervision often causes reduced output diversity and limited generalization. We propose Critical Token Fine-tuning (CFT), a simple yet effective approach that updates only tokens identified as functionally indispensable via counterfactual perturbations. By focusing gradient signals on these decisive reasoning steps while preserving the diversity of non-critical tokens, CFT can enhance both generation and diversity. Extensive experiments on five models across three families (Qwen, OLMo, LLaMA) and eleven mathematical reasoning benchmarks show that CFT, despite fine-tuning on less than 12% of tokens, consistently outperforms standard SFT. Moreover, CFT enables test-time scaling through improved sampling diversity and provides a stronger initialization for reinforcement learning, sustaining performance gains in later training stages while maintaining higher entropy for better exploration. These results highlight CFT as a practical and general framework for efficient and robust LLM fine-tuning. 

---
# KOTOX: A Korean Toxic Dataset for Deobfuscation and Detoxification 

**Authors**: Yejin Lee, Su-Hyeon Kim, Hyundong Jin, Dayoung Kim, Yeonsoo Kim, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.10961)  

**Abstract**: Toxic content has become an increasingly critical social issue with the rapid expansion of online communication. While numerous studies explored methods for detecting and detoxifying such content, most have focused primarily on English, leaving low-resource language underrepresented. Consequently, Large Language Models~(LLMs) often struggle to identify and neutralize toxic expressions in these languages. This challenge becomes even more pronounced when user employ obfuscation techniques to evade detection systems. Therefore, we propose a \textbf{KOTOX: Korean Toxic Dataset} for deobfuscation and detoxicification to address this issue. We categorize various obfuscation approaches based on linguistic characteristics of Korean and define a set of transformation rules grounded in real-word examples. Using these rules, we construct three dataset versions (easy, normal, and hard) representing different levels of obfuscation difficulty. This is the first dataset that simultaneously supports deobfuscation and detoxification for the Korean language. We expect it to facilitate better understanding and mitigating of obfuscated toxic content in LLM for low-resource languages. Our code and data are available at this https URL. 

---
# ADVICE: Answer-Dependent Verbalized Confidence Estimation 

**Authors**: Ki Jung Seo, Sehun Lim, Taeuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.10913)  

**Abstract**: Recent progress in large language models (LLMs) has enabled them to express their confidence in natural language, enhancing transparency and reliability. However, their confidence often exhibits overconfidence, the cause of which remains poorly understood. In this work, we conduct a detailed analysis of the dynamics underlying verbalized confidence and identify answer-independence as a key factor, defined as the model's failure to condition confidence on its own answer. To address this, we propose ADVICE (Answer-Dependent Verbalized Confidence Estimation), a fine-tuning framework that facilitates answer-grounded confidence estimation. Extensive experiments show that ADVICE substantially improves confidence calibration while preserving task performance. Further analyses confirm that ADVICE strengthens answer-groundedness, leading to more balanced and well-calibrated confidence distributions. Our findings shed light on the origin of overconfidence and establish a framework for more trustworthy confidence verbalization. 

---
# Judge Before Answer: Can MLLM Discern the False Premise in Question? 

**Authors**: Jidong Li, Lingyong Fang, Haodong Zhao, Sufeng Duan, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10965)  

**Abstract**: Multimodal large language models (MLLMs) have witnessed astonishing advancements in recent years. Despite these successes, MLLMs remain vulnerable to flase premise problems. However, existing benchmarks targeting this issue are limited in scope: they often lack fine-grained categorization, exhibit insufficient coverage, and thus fail to provide a rigorous evaluation of the ability of models to recognize false premises. To bridge this gap, we introduce a fully automated pipeline for constructing a comprehensive benchmark of false premise questions. Our method systematically categorizes the premises into three main types and thirteen subtypes according to the abilities required to identify the premises, resulting in the JBA this http URL show current MLLMs still struggle with false premise recognition. Building upon this benchmark, we further propose a recognition enhancement framework tailored to strengthen the robustness of MLLMs to detect false premises. Extensive experiments demonstrate that models trained with our framework achieve significant improvements in false premise recognition. 

---
# Rethinking Agentic Workflows: Evaluating Inference-Based Test-Time Scaling Strategies in Text2SQL Tasks 

**Authors**: Jiajing Guo, Kenil Patel, Jorge Piazentin Ono, Wenbin He, Liu Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.10885)  

**Abstract**: Large language models (LLMs) are increasingly powering Text-to-SQL (Text2SQL) systems, enabling non-expert users to query industrial databases using natural language. While test-time scaling strategies have shown promise in LLM-based solutions, their effectiveness in real-world applications, especially with the latest reasoning models, remains uncertain. In this work, we benchmark six lightweight, industry-oriented test-time scaling strategies and four LLMs, including two reasoning models, evaluating their performance on the BIRD Mini-Dev benchmark. Beyond standard accuracy metrics, we also report inference latency and token consumption, providing insights relevant for practical system deployment. Our findings reveal that Divide-and-Conquer prompting and few-shot demonstrations consistently enhance performance for both general-purpose and reasoning-focused LLMs. However, introducing additional workflow steps yields mixed results, and base model selection plays a critical role. This work sheds light on the practical trade-offs between accuracy, efficiency, and complexity when deploying Text2SQL systems. 

---
# Large Language Models for Full-Text Methods Assessment: A Case Study on Mediation Analysis 

**Authors**: Wenqing Zhang, Trang Nguyen, Elizabeth A. Stuart, Yiqun T. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10762)  

**Abstract**: Systematic reviews are crucial for synthesizing scientific evidence but remain labor-intensive, especially when extracting detailed methodological information. Large language models (LLMs) offer potential for automating methodological assessments, promising to transform evidence synthesis. Here, using causal mediation analysis as a representative methodological domain, we benchmarked state-of-the-art LLMs against expert human reviewers across 180 full-text scientific articles. Model performance closely correlated with human judgments (accuracy correlation 0.71; F1 correlation 0.97), achieving near-human accuracy on straightforward, explicitly stated methodological criteria. However, accuracy sharply declined on complex, inference-intensive assessments, lagging expert reviewers by up to 15%. Errors commonly resulted from superficial linguistic cues -- for instance, models frequently misinterpreted keywords like "longitudinal" or "sensitivity" as automatic evidence of rigorous methodological approache, leading to systematic misclassifications. Longer documents yielded lower model accuracy, whereas publication year showed no significant effect. Our findings highlight an important pattern for practitioners using LLMs for methods review and synthesis from full texts: current LLMs excel at identifying explicit methodological features but require human oversight for nuanced interpretations. Integrating automated information extraction with targeted expert review thus provides a promising approach to enhance efficiency and methodological rigor in evidence synthesis across diverse scientific fields. 

---
# Unlocking LLM Safeguards for Low-Resource Languages via Reasoning and Alignment with Minimal Training Data 

**Authors**: Zhuowei Chen, Bowei Zhang, Nankai Lin, Tian Hou, Lianxi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10677)  

**Abstract**: Recent advances in LLMs have enhanced AI capabilities, but also increased the risk posed by malicious requests, highlighting the need for effective LLM safeguards to detect such queries. Existing approaches largely rely on classifier-based methods that lack interpretability and perform poorly on low-resource languages. To address these limitations, we propose ConsistentGuard, a novel reasoning-based multilingual safeguard, which enhances explainability via reasoning and boosts knowledge transfer between languages through alignment. With only 1,000 training samples, our method demonstrates superior performance on three datasets across six languages, outperforming larger models trained with significantly more data, and exhibits strong interpretability and generalization ability. We also contribute a multilingual benchmark extension and release our codes to support future research. 

---
# Review of Inference-Time Scaling Strategies: Reasoning, Search and RAG 

**Authors**: Zhichao Wang, Cheng Wan, Dong Nie  

**Link**: [PDF](https://arxiv.org/pdf/2510.10787)  

**Abstract**: The performance gains of LLMs have historically been driven by scaling up model size and training data. However, the rapidly diminishing availability of high-quality training data is introducing a fundamental bottleneck, shifting the focus of research toward inference-time scaling. This paradigm uses additional computation at the time of deployment to substantially improve LLM performance on downstream tasks without costly model re-training. This review systematically surveys the diverse techniques contributing to this new era of inference-time scaling, organizing the rapidly evolving field into two comprehensive perspectives: Output-focused and Input-focused methods. Output-focused techniques encompass complex, multi-step generation strategies, including reasoning (e.g., CoT, ToT, ReAct), various search and decoding methods (e.g., MCTS, beam search), training for long CoT (e.g., RLVR, GRPO), and model ensemble methods. Input-focused techniques are primarily categorized by few-shot and RAG, with RAG as the central focus. The RAG section is further detailed through a structured examination of query expansion, data, retrieval and reranker, LLM generation methods, and multi-modal RAG. 

---
# RePro: Training Language Models to Faithfully Recycle the Web for Pretraining 

**Authors**: Zichun Yu, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.10681)  

**Abstract**: High-quality pretraining data is the fossil fuel of large language models (LLMs), yet its reserves are running low for frontier models. In this paper, we introduce RePro, a novel web recycling method that trains a relatively small LM with reinforcement learning to generate effective and faithful rephrasings of pretraining data. Specifically, we design one quality reward and three faithfulness rewards, optimizing the LM rephraser to convert organic data into high-quality rephrasings while maintaining its core semantics and structure. In our experiment, we train a 4B rephraser to recycle 72B tokens sampled from DCLM-RefinedWeb. Pretraining results on 400M and 1.4B models demonstrate that RePro delivers 4.7%-14.0% relative accuracy gains over organic-only baseline on 22 downstream tasks. RePro also outperforms ReWire, the state-of-the-art web recycling method that prompts a 70B rephraser, as well as the organic baseline with a 4x larger data pool. Experiments with different amounts of recycled data highlight that RePro improves organic data efficiency by 2-3x. Individual and distributional analyses validate that RePro preserves more critical information and faithfully reflects the characteristics of organic data compared to prompting-based methods. Together, these results show that RePro provides an efficient and controllable path to effectively harness the fossil fuel of LLM pretraining. We open-source our code, rephraser, and recycled data at this https URL. 

---
# Merlin's Whisper: Enabling Efficient Reasoning in LLMs via Black-box Adversarial Prompting 

**Authors**: Heming Xia, Cunxiao Du, Rui Li, Chak Tou Leong, Yongqi Li, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.10528)  

**Abstract**: Large reasoning models (LRMs) have demonstrated remarkable proficiency in tackling complex reasoning tasks through step-by-step thinking. However, such a lengthy reasoning process incurs substantial computational and latency overheads, hindering the practical deployment of these models. In this work, we present a new perspective on mitigating overthinking in LRMs via black-box adversarial prompting. By treating both open-source LRMs and closed-source APIs as black-box communicators, we investigate how to elicit concise responses without sacrificing accuracy. We introduce AdvPrompt, an iterative refinement framework that generates high-quality adversarial prompts from diverse perspectives. Experiments across multiple benchmarks demonstrate that AdvPrompt consistently reduces token usage while preserving performance. Notably, AdvPrompt achieves a 3x reduction in average response length on simple GSM8K questions for the Qwen3 model series, and delivers an average ~40% token reduction across four benchmarks. For closed-source APIs, AdvPrompt reduces token usage on MATH-500 by 35% for Claude-3.7 and 47% for Gemini-2.5. Further analysis reveals the generalizability of AdvPrompt across various model scales and families, underscoring the potential of black-box prompting as a practical and effective strategy for enhancing LRM efficiency. 

---
# Detecting Hallucinations in Authentic LLM-Human Interactions 

**Authors**: Yujie Ren, Niklas Gruhlke, Anne Lauscher  

**Link**: [PDF](https://arxiv.org/pdf/2510.10539)  

**Abstract**: As large language models (LLMs) are increasingly applied in sensitive domains such as medicine and law, hallucination detection has become a critical task. Although numerous benchmarks have been proposed to advance research in this area, most of them are artificially constructed--either through deliberate hallucination induction or simulated interactions--rather than derived from genuine LLM-human dialogues. Consequently, these benchmarks fail to fully capture the characteristics of hallucinations that occur in real-world usage. To address this limitation, we introduce AuthenHallu, the first hallucination detection benchmark built entirely from authentic LLM-human interactions. For AuthenHallu, we select and annotate samples from genuine LLM-human dialogues, thereby providing a faithful reflection of how LLMs hallucinate in everyday user interactions. Statistical analysis shows that hallucinations occur in 31.4% of the query-response pairs in our benchmark, and this proportion increases dramatically to 60.0% in challenging domains such as Math & Number Problems. Furthermore, we explore the potential of using vanilla LLMs themselves as hallucination detectors and find that, despite some promise, their current performance remains insufficient in real-world scenarios. 

---
# Assessing Large Language Models for Structured Medical Order Extraction 

**Authors**: A H M Rezaul Karim, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2510.10475)  

**Abstract**: Medical order extraction is essential for structuring actionable clinical information, supporting decision-making, and enabling downstream applications such as documentation and workflow automation. Orders may be embedded in diverse sources, including electronic health records, discharge summaries, and multi-turn doctor-patient dialogues, and can span categories such as medications, laboratory tests, imaging studies, and follow-up actions. The MEDIQA-OE 2025 shared task focuses on extracting structured medical orders from extended conversational transcripts, requiring the identification of order type, description, reason, and provenance. We present the MasonNLP submission, which ranked 5th among 17 participating teams with 105 total submissions. Our approach uses a general-purpose, instruction-tuned LLaMA-4 17B model without domain-specific fine-tuning, guided by a single in-context example. This few-shot configuration achieved an average F1 score of 37.76, with notable improvements in reason and provenance accuracy. These results demonstrate that large, non-domain-specific LLMs, when paired with effective prompt engineering, can serve as strong, scalable baselines for specialized clinical NLP tasks. 

---
# Rethinking LLM Evaluation: Can We Evaluate LLMs with 200x Less Data? 

**Authors**: Shaobo Wang, Cong Wang, Wenjie Fu, Yue Min, Mingquan Feng, Isabel Guan, Xuming Hu, Conghui He, Cunxiang Wang, Kexin Yang, Xingzhang Ren, Fei Huang, Dayiheng Liu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10457)  

**Abstract**: As the demand for comprehensive evaluations of diverse model capabilities steadily increases, benchmark suites have correspondingly grown significantly in scale. Despite notable advances in redundancy reduction and subset-level performance prediction, a systematic framework that effectively integrates these methods to ensure both prediction accuracy and ranking consistency is still largely elusive. In this paper, we first perform a sample-level analysis of benchmark redundancy and identify several highly similar samples that can be eliminated. Besides, we frame benchmark compression as an optimization problem with the aim of score reconstruction. Building on these, we then propose EssenceBench, a coarse-to-fine framework utilizing an iterative Genetic Algorithm (GA), which takes the advantages of fitness-based subset search and attribution-based sample search. Compared to previous methods, our approach yields superior compression results with lower reconstruction error and markedly higher efficiency. In particular, on the HellaSwag benchmark (10K samples), our method preserves the ranking of all models shifting within 5% using 25x fewer samples, and achieves 95% ranking preservation shifting within 5% using only 200x fewer samples. 

---
# Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization 

**Authors**: Bowei He, Lihao Yin, Huiling Zhen, Shuqi Liu, Han Wu, Xiaokun Zhang, Mingxuan Yuan, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.10618)  

**Abstract**: Post-training compression has been a widely employed approach to scale down large language model (LLM) and facilitate efficient inference. In various proposed compression methods, including pruning and quantization, calibration data plays a vital role by informing the weight importance and activation dynamic ranges. However, how calibration data impacts the LLM capability after compression is less explored. Few of the existing works, though recognizing the significance of this study, only investigate the language modeling or commonsense reasoning performance degradation from limited angles, like the data sources or sample amounts. More systematic research is still needed to examine the impacts on different LLM capabilities in terms of compositional properties and domain correspondence of calibration data. In this work, we aim at bridging this gap and further analyze underlying influencing mechanisms from the activation pattern perspective. Especially, we explore the calibration data's impacts on high-level complex reasoning capabilities, like math problem solving and code generation. Delving into the underlying mechanism, we find that the representativeness and diversity in activation space more fundamentally determine the quality of calibration data. Finally, we propose a calibration data curation framework based on such observations and analysis, enhancing the performance of existing post-training compression methods on preserving critical LLM capabilities. Our code is provided in \href{this https URL}{Link}. 

---
# Steering Over-refusals Towards Safety in Retrieval Augmented Generation 

**Authors**: Utsav Maskey, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2510.10452)  

**Abstract**: Safety alignment in large language models (LLMs) induces over-refusals -- where LLMs decline benign requests due to aggressive safety filters. We analyze this phenomenon in retrieval-augmented generation (RAG), where both the query intent and retrieved context properties influence refusal behavior. We construct RagRefuse, a domain-stratified benchmark spanning medical, chemical, and open domains, pairing benign and harmful queries with controlled context contamination patterns and sizes. Our analysis shows that context arrangement / contamination, domain of query and context, and harmful-text density trigger refusals even on benign queries, with effects depending on model-specific alignment choices. To mitigate over-refusals, we introduce \textsc{SafeRAG-Steering}, a model-centric embedding intervention that steers the embedding regions towards the confirmed safe, non-refusing output regions at inference time. This reduces over-refusals in contaminated RAG pipelines while preserving legitimate refusals. 

---
# STEAM: A Semantic-Level Knowledge Editing Framework for Large Language Models 

**Authors**: Geunyeong Jeong, Juoh Sun, Seonghee Lee, Harksoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.10398)  

**Abstract**: Large Language Models store extensive factual knowledge acquired during large-scale pre-training. However, this knowledge is inherently static, reflecting only the state of the world at the time of training. Knowledge editing has emerged as a promising solution for updating outdated or incorrect facts without full retraining. However, most existing locate-and-edit methods primarily focus on token-level likelihood optimization without addressing semantic coherence. Our analysis reveals that such edited knowledge is often encoded as isolated residual streams in the model's latent space, distinct from pre-existing knowledge and bypassing natural reasoning process. To address this, we propose \textsc{Steam}, a semantic-level knowledge editing framework that enhances integration of updated knowledge into the model's knowledge structure. \textsc{Steam} first identifies target representations as semantic anchors for the updated factual association, then guides the internal representation of the edited fact towards these anchors through an alignment loss during optimization. Experimental results demonstrate that \textsc{Steam} improves model's ability to reason with edited knowledge and enhances semantic coherence, underscoring the importance of latent-space alignment for reliable and coherent knowledge editing. The code is available at this https URL. 

---
# NIM: Neuro-symbolic Ideographic Metalanguage for Inclusive Communication 

**Authors**: Prawaal Sharma, Poonam Goyal, Navneet Goyal, Vidisha Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2510.10459)  

**Abstract**: Digital communication has become the cornerstone of modern interaction, enabling rapid, accessible, and interactive exchanges. However, individuals with lower academic literacy often face significant barriers, exacerbating the "digital divide". In this work, we introduce a novel, universal ideographic metalanguage designed as an innovative communication framework that transcends academic, linguistic, and cultural boundaries. Our approach leverages principles of Neuro-symbolic AI, combining neural-based large language models (LLMs) enriched with world knowledge and symbolic knowledge heuristics grounded in the linguistic theory of Natural Semantic Metalanguage (NSM). This enables the semantic decomposition of complex ideas into simpler, atomic concepts. Adopting a human-centric, collaborative methodology, we engaged over 200 semi-literate participants in defining the problem, selecting ideographs, and validating the system. With over 80\% semantic comprehensibility, an accessible learning curve, and universal adaptability, our system effectively serves underprivileged populations with limited formal education. 

---
# On the Entity-Level Alignment in Crosslingual Consistency 

**Authors**: Yihong Liu, Mingyang Wang, François Yvon, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2510.10280)  

**Abstract**: Multilingual large language models (LLMs) are expected to recall factual knowledge consistently across languages. However, the factors that give rise to such crosslingual consistency -- and its frequent failure -- remain poorly understood. In this work, we hypothesize that these inconsistencies may arise from failures in entity alignment, the process of mapping subject and object entities into a shared conceptual space across languages. To test this, we assess alignment through entity-level (subject and object) translation tasks, and find that consistency is strongly correlated with alignment across all studied models, with misalignment of subjects or objects frequently resulting in inconsistencies. Building on this insight, we propose SubSub and SubInj, two effective methods that integrate English translations of subjects into prompts across languages, leading to substantial gains in both factual recall accuracy and consistency. Finally, our mechanistic analysis reveals that these interventions reinforce the entity representation alignment in the conceptual space through model's internal pivot-language processing, offering effective and practical strategies for improving multilingual factual prediction. 

---
# Backdoor Collapse: Eliminating Unknown Threats via Known Backdoor Aggregation in Language Models 

**Authors**: Liang Lin, Miao Yu, Moayad Aloqaily, Zhenhong Zhou, Kun Wang, Linsey Pang, Prakhar Mehrotra, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10265)  

**Abstract**: Backdoor attacks are a significant threat to large language models (LLMs), often embedded via public checkpoints, yet existing defenses rely on impractical assumptions about trigger settings. To address this challenge, we propose \ourmethod, a defense framework that requires no prior knowledge of trigger settings. \ourmethod is based on the key observation that when deliberately injecting known backdoors into an already-compromised model, both existing unknown and newly injected backdoors aggregate in the representation space. \ourmethod leverages this through a two-stage process: \textbf{first}, aggregating backdoor representations by injecting known triggers, and \textbf{then}, performing recovery fine-tuning to restore benign outputs. Extensive experiments across multiple LLM architectures demonstrate that: (I) \ourmethod reduces the average Attack Success Rate to 4.41\% across multiple benchmarks, outperforming existing baselines by 28.1\%$\sim$69.3\%$\uparrow$. (II) Clean accuracy and utility are preserved within 0.5\% of the original model, ensuring negligible impact on legitimate tasks. (III) The defense generalizes across different types of backdoors, confirming its robustness in practical deployment scenarios. 

---
# End-to-end Automatic Speech Recognition and Speech Translation: Integration of Speech Foundational Models and LLMs 

**Authors**: Nam Luu, Ondřej Bojar  

**Link**: [PDF](https://arxiv.org/pdf/2510.10329)  

**Abstract**: Speech Translation (ST) is a machine translation task that involves converting speech signals from one language to the corresponding text in another language; this task has two different approaches, namely the traditional cascade and the more recent end-to-end. This paper explores a combined end-to-end architecture of pre-trained speech encoders and Large Language Models (LLMs) for performing both Automatic Speech Recognition (ASR) and ST simultaneously. Experiments with the English-to-German language pair show that our best model not only can achieve better translation results than SeamlessM4T, a large foundational end-to-end, multi-modal translation model, but can also match the performance of a cascaded system with Whisper and NLLB, with up to a score gain of 8% in $\text{COMET}^{\text{DA}}_{22}$ metric. 

---
# You only need 4 extra tokens: Synergistic Test-time Adaptation for LLMs 

**Authors**: Yijie Xu, Huizai Yao, Zhiyu Guo, Weiyu Guo, Pengteng Li, Aiwei Liu, Xuming Hu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.10223)  

**Abstract**: Large language models (LLMs) are increasingly deployed in specialized domains such as finance, medicine, and agriculture, where they face significant distribution shifts from their training data. Domain-specific fine-tuning can mitigate this challenge but relies on high-quality labeled data that is expensive and slow to collect in expertise-limited settings. We study label-free test-time adaptation for language models and present SyTTA, an inference-time framework that adapts models on-the-fly without additional supervision. SyTTA couples two complementary uncertainty signals that arise under distribution shift: input-side perplexity, indicating mismatch with domain-specific terminology and patterns, and output-side predictive entropy, indicating diffuse and unstable token probabilities during generation. Across diverse model architectures and domain-specific benchmarks, SyTTA delivers consistent gains. Notably, on agricultural question answering, SyTTA improves Rouge-LSum by over 120% on Qwen-2.5-7B with only 4 extra tokens per query. These results show that effective test-time adaptation for language models is achievable without labeled examples, supporting deployment in label-scarce domains. The code will be made available upon acceptance. 

---
# A Survey of Inductive Reasoning for Large Language Models 

**Authors**: Kedi Chen, Dezhao Ruan, Yuhao Dan, Yaoting Wang, Siyu Yan, Xuecheng Wu, Yinqi Zhang, Qin Chen, Jie Zhou, Liang He, Biqing Qi, Linyang Li, Qipeng Guo, Xiaoming Shi, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10182)  

**Abstract**: Reasoning is an important task for large language models (LLMs). Among all the reasoning paradigms, inductive reasoning is one of the fundamental types, which is characterized by its particular-to-general thinking process and the non-uniqueness of its answers. The inductive mode is crucial for knowledge generalization and aligns better with human cognition, so it is a fundamental mode of learning, hence attracting increasing interest. Despite the importance of inductive reasoning, there is no systematic summary of it. Therefore, this paper presents the first comprehensive survey of inductive reasoning for LLMs. First, methods for improving inductive reasoning are categorized into three main areas: post-training, test-time scaling, and data augmentation. Then, current benchmarks of inductive reasoning are summarized, and a unified sandbox-based evaluation approach with the observation coverage metric is derived. Finally, we offer some analyses regarding the source of inductive ability and how simple model architectures and data help with inductive tasks, providing a solid foundation for future research. 

---
# Large Language Model Sourcing: A Survey 

**Authors**: Liang Pang, Kangxi Wu, Sunhao Dai, Zihao Wei, Zenghao Duan, Jia Gu, Xiang Li, Zhiyi Yin, Jun Xu, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.10161)  

**Abstract**: The rapid advancement of large language models (LLMs) has revolutionized artificial intelligence, shifting from supporting objective tasks (e.g., recognition) to empowering subjective decision-making (e.g., planning, decision). This marks the dawn of general and powerful AI, with applications spanning a wide range of fields, including programming, education, healthcare, finance, and law. However, their deployment introduces multifaceted risks. Due to the black-box nature of LLMs and the human-like quality of their generated content, issues such as hallucinations, bias, unfairness, and copyright infringement become particularly significant. In this context, sourcing information from multiple perspectives is essential.
This survey presents a systematic investigation into provenance tracking for content generated by LLMs, organized around four interrelated dimensions that together capture both model- and data-centric perspectives. From the model perspective, Model Sourcing treats the model as a whole, aiming to distinguish content generated by specific LLMs from content authored by humans. Model Structure Sourcing delves into the internal generative mechanisms, analyzing architectural components that shape the outputs of model. From the data perspective, Training Data Sourcing focuses on internal attribution, tracing the origins of generated content back to the training data of model. In contrast, External Data Sourcing emphasizes external validation, identifying external information used to support or influence the responses of model. Moreover, we also propose a dual-paradigm taxonomy that classifies existing sourcing methods into prior-based (proactive traceability embedding) and posterior-based (retrospective inference) approaches. Traceability across these dimensions enhances the transparency, accountability, and trustworthiness of LLMs deployment in real-world applications. 

---
# Audit-of-Understanding: Posterior-Constrained Inference for Mathematical Reasoning in Language Models 

**Authors**: Samir Abdaljalil, Erchin Serpedin, Khalid Qaraqe, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2510.10252)  

**Abstract**: Large language models (LLMs) often generate reasoning traces that appear coherent but rest on unsupported assumptions, leading to hallucinated conclusions. Prior work mainly addresses factual hallucinations or relies on post-hoc verification, leaving reasoning-induced hallucinations largely unaddressed. We propose Audit-of-Understanding (AoU), a framework that constrains inference to validated premises through three phases: (1) decomposing a query into candidate assumptions, (2) auditing their support, and (3) conditioning inference only on the validated subset. Formally, AoU is \emph{posterior-constrained inference}, connecting to selective prediction and rejection learning. Our contributions are threefold: (i) theoretical guarantees under perfect validation, (ii) excess-risk bounds under imperfect audits, and (iii) tractability analysis. Empirically, AoU improves both accuracy and faithfulness on GSM8K, MultiArith, and SVAMP, achieving up to +30% gains on GSM8K, +45% on MultiArith, and consistent +20--28% improvements on SVAMP over Chain-of-Thought, Self-Consistency, and CoT-Decoding. Code is available at this https URL. 

---
# Weed Out, Then Harvest: Dual Low-Rank Adaptation is an Effective Noisy Label Detector for Noise-Robust Learning 

**Authors**: Bo Yuan, Yulin Chen, Yin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10208)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) large language models (LLMs) have shown impressive performance in various downstream tasks. However, in many real-world scenarios, the collected training data inevitably contains noisy labels. To learn from noisy labels, most solutions select samples with small losses for model training. However, the selected samples, in turn, impact the loss computation in the next iteration. An inaccurate initial selection can create a vicious cycle, leading to suboptimal performance. To break this cycle, we propose Delora, a novel framework that decouples the sample selection from model training. For sample selection, Delora establishes a noisy label detector by introducing clean and noisy LoRA. Benefiting from the memory effect, the clean LoRA is encouraged to memorize clean data, while the noisy LoRA is constrained to memorize mislabeled data, which serves as a learnable threshold for selecting clean and noisy samples. For model training, Delora can use carefully selected samples to fine-tune language models seamlessly. Experimental results on synthetic and real-world noisy datasets demonstrate the effectiveness of Delora in noisy label detection and text classification. 

---
# MatryoshkaThinking: Recursive Test-Time Scaling Enables Efficient Reasoning 

**Authors**: Hongwei Chen, Yishu Lei, Dan Zhang, Bo Ke, Danxiang Zhu, Xuyi Chen, Yuxiang Lu, Zhengjie Huang, Shikun Feng, Jingzhou He, Yu Sun, Hua Wu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10293)  

**Abstract**: Test-time scaling has emerged as a promising paradigm in language modeling, wherein additional computational resources are allocated during inference to enhance model performance. Recent approaches, such as DeepConf, have demonstrated the efficacy of this strategy, however, they often incur substantial computational overhead to achieve competitive results. In this work, we propose MatryoshkaThinking, a novel method that significantly reduces computational cost while maintaining state-of-the-art performance. Specifically, MatryoshkaThinking attains a score of 99.79 on AIME2025 using only 4% of the computation required by DeepConf. The core of our approach lies in the recursive exploitation of the model's intrinsic capabilities in reasoning, verification, and summarization, which collectively enhance the retention of correct solutions and reduce the disparity between Pass@k and Pass@1. Comprehensive evaluations across multiple open-source models and challenging multi-modal reasoning benchmarks validate the effectiveness and generality of our method. These findings offer new insights into the design of efficient and scalable test-time inference strategies for advanced language models. 

---
# BILLY: Steering Large Language Models via Merging Persona Vectors for Creative Generation 

**Authors**: Tsung-Min Pai, Jui-I Wang, Li-Chun Lu, Shao-Hua Sun, Hung-Yi Lee, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10157)  

**Abstract**: Multi-LLM systems enhance the creativity of large language models by simulating human collective intelligence but suffer from significant drawbacks, such as high computational costs and inference latency. To address these limitations, we propose BILLY (BlendIng persona vectors for Large Language model creativitY), a training-free framework that captures the benefits of multi-LLM collaboration, i.e. inducing diverse perspectives and specialized expertise, within a single model. BILLY operates by extracting and blending multiple distinct persona vectors directly in the model's activation space. We steer the model's generation process with this merged vector while inference, enabling multi-perspective output without explicit multi-LLM communication. Our experiments across creativity-oriented benchmarks demonstrate that BILLY surpasses single model prompting and traditional multi-LLM approaches, while substantially reducing inference time and computational costs. Our analyses further reveal that distinct persona vectors can be blended to achieve both effective control over complementary aspects of generation and greater interpretability. 

---
# DiffHeads: Differential Analysis and Inference-Time Masking of Bias Heads in Large Language Models 

**Authors**: Tingxu Han, Wei Song, Ziqi Ding, Ziming Li, Chunrong Fang, Yuekang Li, Dongfang Liu, Zhenyu Chen, Zhenting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10142)  

**Abstract**: Large language models (LLMs) increasingly mediate decisions in domains where unfair treatment of demographic groups is unacceptable. Existing work probes when biased outputs appear, but gives little insight into the mechanisms that generate them, leaving existing mitigations largely fragile. In this paper, we conduct a systematic investigation LLM unfairness and propose DiffHeads, a lightweight debiasing framework for LLMs. We first compare Direct-Answer (DA) prompting to Chain-of-Thought (CoT) prompting across eight representative open- and closed-source LLMs. DA will trigger the nature bias part of LLM and improve measured unfairness by 534.5%-391.9% in both one-turn and two-turn dialogues. Next, we define a token-to-head contribution score that traces each token's influence back to individual attention heads. This reveals a small cluster of bias heads that activate under DA but stay largely dormant with CoT, providing the first causal link between prompting strategy and bias emergence. Finally, building on this insight, we propose DiffHeads that identifies bias heads through differential activation analysis between DA and CoT, and selectively masks only those heads. DiffHeads reduces unfairness by 49.4%, and 40.3% under DA and CoT, respectively, without harming model utility. 

---
# Stop When Enough: Adaptive Early-Stopping for Chain-of-Thought Reasoning 

**Authors**: Renliang Sun, Wei Cheng, Dawei Li, Haifeng Chen, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10103)  

**Abstract**: Chain-of-Thought (CoT) reasoning has driven recent gains of large language models (LLMs) on reasoning-intensive tasks by externalizing intermediate steps. However, excessive or redundant reasoning -- so-called overthinking -- can increase inference costs and lead LLMs toward incorrect conclusions. In this paper, we present REFRAIN ($\underline{REF}$lective-$\underline{R}$edundancy for $\underline{A}$daptive $\underline{IN}$ference), a training-free framework that adaptively determines when to stop reasoning to mitigate overthinking. REFRAIN integrates a two-stage stop discriminator to identify reflective yet redundant reasoning and a sliding-window Upper Confidence Bound (SW-UCB) multi-armed bandit controller to dynamically adjust stopping thresholds according to problem difficulty without supervision or fine-tuning. Across four representative benchmarks and two model families, REFRAIN reduces token usage by 20-55% while maintaining or improving accuracy compared to standard CoT prompting. Extensive ablation and robustness analyses demonstrate its stability across models, scorers, and prompt variations. In summary, our findings highlight when-to-stop as a new and practical axis of test-time scaling -- enabling models to reason not just more, but just enough. 

---
# Hybrid OCR-LLM Framework for Enterprise-Scale Document Information Extraction Under Copy-heavy Task 

**Authors**: Zilong Wang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10138)  

**Abstract**: Information extraction from copy-heavy documents, characterized by massive volumes of structurally similar content, represents a critical yet understudied challenge in enterprise document processing. We present a systematic framework that strategically combines OCR engines with Large Language Models (LLMs) to optimize the accuracy-efficiency trade-off inherent in repetitive document extraction tasks. Unlike existing approaches that pursue universal solutions, our method exploits document-specific characteristics through intelligent strategy selection. We implement and evaluate 25 configurations across three extraction paradigms (direct, replacement, and table-based) on identity documents spanning four formats (PNG, DOCX, XLSX, PDF). Through table-based extraction methods, our adaptive framework delivers outstanding results: F1=1.0 accuracy with 0.97s latency for structured documents, and F1=0.997 accuracy with 0.6 s for challenging image inputs when integrated with PaddleOCR, all while maintaining sub-second processing speeds. The 54 times performance improvement compared with multimodal methods over naive approaches, coupled with format-aware routing, enables processing of heterogeneous document streams at production scale. Beyond the specific application to identity extraction, this work establishes a general principle: the repetitive nature of copy-heavy tasks can be transformed from a computational burden into an optimization opportunity through structure-aware method selection. 

---
# Unifying Tree Search Algorithm and Reward Design for LLM Reasoning: A Survey 

**Authors**: Jiaqi Wei, Xiang Zhang, Yuejin Yang, Wenxuan Huang, Juntai Cao, Sheng Xu, Xiang Zhuang, Zhangyang Gao, Muhammad Abdul-Mageed, Laks V.S. Lakshmanan, Chenyu You, Wanli Ouyang, Siqi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.09988)  

**Abstract**: Deliberative tree search is a cornerstone of modern Large Language Model (LLM) research, driving the pivot from brute-force scaling toward algorithmic efficiency. This single paradigm unifies two critical frontiers: \textbf{Test-Time Scaling (TTS)}, which deploys on-demand computation to solve hard problems, and \textbf{Self-Improvement}, which uses search-generated data to durably enhance model parameters. However, this burgeoning field is fragmented and lacks a common formalism, particularly concerning the ambiguous role of the reward signal -- is it a transient heuristic or a durable learning target? This paper resolves this ambiguity by introducing a unified framework that deconstructs search algorithms into three core components: the \emph{Search Mechanism}, \emph{Reward Formulation}, and \emph{Transition Function}. We establish a formal distinction between transient \textbf{Search Guidance} for TTS and durable \textbf{Parametric Reward Modeling} for Self-Improvement. Building on this formalism, we introduce a component-centric taxonomy, synthesize the state-of-the-art, and chart a research roadmap toward more systematic progress in creating autonomous, self-improving agents. 

---
# Path Drift in Large Reasoning Models:How First-Person Commitments Override Safety 

**Authors**: Yuyi Huang, Runzhe Zhan, Lidia S.Chao, Ailin Tao, Derek F.Wong  

**Link**: [PDF](https://arxiv.org/pdf/2510.10013)  

**Abstract**: As large language models (LLMs) are increasingly deployed for complex reasoning tasks, Long Chain-of-Thought (Long-CoT) prompting has emerged as a key paradigm for structured inference. Despite early-stage safeguards enabled by alignment techniques such as RLHF, we identify a previously underexplored vulnerability: reasoning trajectories in Long-CoT models can drift from aligned paths, resulting in content that violates safety constraints. We term this phenomenon Path Drift. Through empirical analysis, we uncover three behavioral triggers of Path Drift: (1) first-person commitments that induce goal-driven reasoning that delays refusal signals; (2) ethical evaporation, where surface-level disclaimers bypass alignment checkpoints; (3) condition chain escalation, where layered cues progressively steer models toward unsafe completions. Building on these insights, we introduce a three-stage Path Drift Induction Framework comprising cognitive load amplification, self-role priming, and condition chain hijacking. Each stage independently reduces refusal rates, while their combination further compounds the effect. To mitigate these risks, we propose a path-level defense strategy incorporating role attribution correction and metacognitive reflection (reflective safety cues). Our findings highlight the need for trajectory-level alignment oversight in long-form reasoning beyond token-level alignment. 

---
# Don't Throw Away Your Pretrained Model 

**Authors**: Shangbin Feng, Wenhao Yu, Yike Wang, Hongming Zhang, Yulia Tsvetkov, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09913)  

**Abstract**: Alignment training has tradeoffs: it helps language models (LMs) gain in reasoning and instruction following but might lose out on skills such as creativity and calibration, where unaligned base models are better at. We aim to make the best of both worlds through model collaboration, where different models in the training pipeline collaborate and complement each other. Since LM responses feature interleaving skills that favor different models, we propose Switch Generation, where pretrained and aligned model versions take turns to ``speak'' in a response sequence. Specifically, we train a switcher LM by learning from outcomes of choosing different models to generate the next segment across diverse queries and contexts. At inference time, the switcher LM guides different model checkpoints to dynamically generate the next segment where their strengths are most needed. Extensive experiments with 8 model collaboration baselines and 18 datasets show that 1) model collaboration consistently outperforms individual models on 16 out of 18 tasks, and 2) Switch Generation further outperforms baselines by 12.9% on average. Further analysis reveals that Switch Generation discovers compositional skills to solve problems where individual models struggle and generalizes to unseen models and tasks, reusing and repurposing by-products in expensive model training pipelines that are otherwise discarded. 

---
# Unilaw-R1: A Large Language Model for Legal Reasoning with Reinforcement Learning and Iterative Inference 

**Authors**: Hua Cai, Shuang Zhao, Liang Zhang, Xuli Shen, Qing Xu, Weilin Shen, Zihao Wen, Tianke Ban  

**Link**: [PDF](https://arxiv.org/pdf/2510.10072)  

**Abstract**: Reasoning-focused large language models (LLMs) are rapidly evolving across various domains, yet their capabilities in handling complex legal problems remains underexplored. In this paper, we introduce Unilaw-R1, a large language model tailored for legal reasoning. With a lightweight 7-billion parameter scale, Unilaw-R1 significantly reduces deployment cost while effectively tackling three core challenges in the legal domain: insufficient legal knowledge, unreliable reasoning logic, and weak business generalization. To address these issues, we first construct Unilaw-R1-Data, a high-quality dataset containing 17K distilled and screened chain-of-thought (CoT) samples. Based on this, we adopt a two-stage training strategy combining Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), which significantly boosts the performance on complex legal reasoning tasks and supports interpretable decision-making in legal AI applications. To assess legal reasoning ability, we also introduce Unilaw-R1-Eval, a dedicated benchmark designed to evaluate models across single- and multi-choice legal tasks. Unilaw-R1 demonstrates strong results on authoritative benchmarks, outperforming all models of similar scale and achieving performance on par with the much larger DeepSeek-R1-Distill-Qwen-32B (54.9%). Following domain-specific training, it also showed significant gains on LawBench and LexEval, exceeding Qwen-2.5-7B-Instruct (46.6%) by an average margin of 6.6%. 

---
# CoBia: Constructed Conversations Can Trigger Otherwise Concealed Societal Biases in LLMs 

**Authors**: Nafiseh Nikeghbal, Amir Hossein Kargaran, Jana Diesner  

**Link**: [PDF](https://arxiv.org/pdf/2510.09871)  

**Abstract**: Improvements in model construction, including fortified safety guardrails, allow Large language models (LLMs) to increasingly pass standard safety checks. However, LLMs sometimes slip into revealing harmful behavior, such as expressing racist viewpoints, during conversations. To analyze this systematically, we introduce CoBia, a suite of lightweight adversarial attacks that allow us to refine the scope of conditions under which LLMs depart from normative or ethical behavior in conversations. CoBia creates a constructed conversation where the model utters a biased claim about a social group. We then evaluate whether the model can recover from the fabricated bias claim and reject biased follow-up questions. We evaluate 11 open-source as well as proprietary LLMs for their outputs related to six socio-demographic categories that are relevant to individual safety and fair treatment, i.e., gender, race, religion, nationality, sex orientation, and others. Our evaluation is based on established LLM-based bias metrics, and we compare the results against human judgments to scope out the LLMs' reliability and alignment. The results suggest that purposefully constructed conversations reliably reveal bias amplification and that LLMs often fail to reject biased follow-up questions during dialogue. This form of stress-testing highlights deeply embedded biases that can be surfaced through interaction. Code and artifacts are available at this https URL. 

---
# HIPPD: Brain-Inspired Hierarchical Information Processing for Personality Detection 

**Authors**: Guanming Chen, Lingzhi Shen, Xiaohao Cai, Imran Razzak, Shoaib Jameel  

**Link**: [PDF](https://arxiv.org/pdf/2510.09893)  

**Abstract**: Personality detection from text aims to infer an individual's personality traits based on linguistic patterns. However, existing machine learning approaches often struggle to capture contextual information spanning multiple posts and tend to fall short in extracting representative and robust features in semantically sparse environments. This paper presents HIPPD, a brain-inspired framework for personality detection that emulates the hierarchical information processing of the human brain. HIPPD utilises a large language model to simulate the cerebral cortex, enabling global semantic reasoning and deep feature abstraction. A dynamic memory module, modelled after the prefrontal cortex, performs adaptive gating and selective retention of critical features, with all adjustments driven by dopaminergic prediction error feedback. Subsequently, a set of specialised lightweight models, emulating the basal ganglia, are dynamically routed via a strict winner-takes-all mechanism to capture the personality-related patterns they are most proficient at recognising. Extensive experiments on the Kaggle and Pandora datasets demonstrate that HIPPD consistently outperforms state-of-the-art baselines. 

---
# Enhancing Faithfulness in Abstractive Summarization via Span-Level Fine-Tuning 

**Authors**: Sicong Huang, Qianqi Yan, Shengze Wang, Ian Lane  

**Link**: [PDF](https://arxiv.org/pdf/2510.09915)  

**Abstract**: Abstractive summarization using large language models (LLMs) has become an essential tool for condensing information. However, despite their ability to generate fluent summaries, these models sometimes produce unfaithful summaries, introducing hallucinations at the word, phrase, or concept level. Existing mitigation strategies, such as post-processing corrections or contrastive learning with synthetically generated negative samples, fail to fully address the diverse errors that can occur in LLM-generated summaries. In this paper, we investigate fine-tuning strategies to reduce the occurrence of unfaithful spans in generated summaries. First, we automatically generate summaries for the set of source documents in the training set with a variety of LLMs and then use GPT-4o to annotate any hallucinations it detects at the span-level. Leveraging these annotations, we fine-tune LLMs with both hallucination-free summaries and annotated unfaithful spans to enhance model faithfulness. In this paper, we introduce a new dataset that contains both faithful and unfaithful summaries with span-level labels and we evaluate three techniques to fine-tuning a LLM to improve the faithfulness of the resulting summarization: gradient ascent, unlikelihood training, and task vector negation. Experimental results show that all three approaches successfully leverage span-level annotations to improve faithfulness, with unlikelihood training being the most effective. 

---
# Gold Panning: Turning Positional Bias into Signal for Multi-Document LLM Reasoning 

**Authors**: Adam Byerly, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2510.09770)  

**Abstract**: Large language models exhibit a strong position bias in multi-document contexts, systematically prioritizing information based on location rather than relevance. While existing approaches treat this bias as noise to be mitigated, we introduce Gold Panning Bandits, a framework that leverages position bias as a diagnostic signal: by reordering documents and observing shifts in the model's responses, we can efficiently identify the most relevant content. We frame the problem of choosing reorderings as a bipartite matching problem. While an optimal assignment can be computed at each iteration with the Hungarian algorithm in $O(N^3)$ time, we propose a greedy $O(N \log N)$ strategy that achieves comparable performance by prioritizing the placement of the most uncertain documents in the most informative positions. Our approach identifies relevant documents using up to 65\% fewer language model queries than random permutation baselines on knowledge-intensive NLP tasks, substantially reducing computational cost without model retraining. This work demonstrates that inherent LLM biases can be transformed from liabilities into assets for efficient, inference-time optimization. 

---
# DELTA: Dynamic Layer-Aware Token Attention for Efficient Long-Context Reasoning 

**Authors**: Hossein Entezari Zarch, Lei Gao, Chaoyi Jiang, Murali Annavarm  

**Link**: [PDF](https://arxiv.org/pdf/2510.09883)  

**Abstract**: Large reasoning models (LRMs) achieve state-of-the-art performance on challenging benchmarks by generating long chains of intermediate steps, but their inference cost is dominated by decoding, where each new token must attend to the entire growing sequence. Existing sparse attention methods reduce computation by pruning the key-value (KV) cache, yet they suffer from severe accuracy degradation on reasoning tasks due to cumulative selection errors and the dynamic importance of tokens over long derivations. We present \textbf{DELTA}, a training-free sparse attention mechanism that achieves computational efficiency without sacrificing model accuracy. DELTA partitions transformer layers into three groups: initial layers that use full attention, a small set of \emph{selection layers} that identify salient tokens via aggregated head-level attention scores, and subsequent \emph{sparse-attention layers} that attend only to the selected subset. This design preserves the full KV cache in GPU memory for accuracy, while avoiding expensive full-attention computation over many layers. On reasoning benchmarks such as AIME and GPQA-Diamond, DELTA matches or surpasses full attention in accuracy, while reducing the number of attended tokens by up to $5\times$ and delivering $1.5\times$ end-to-end speedup. Our results show that selective reuse of intermediate attention maps offers a robust path toward efficient long-context reasoning. 

---
# Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement 

**Authors**: Steve Han, Gilberto Titericz Junior, Tom Balough, Wenfei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.09738)  

**Abstract**: This research introduces the Judge's Verdict Benchmark, a novel two-step methodology to evaluate Large Language Models (LLMs) as judges for response accuracy evaluation tasks. We assess how well 54 LLMs can replicate human judgment when scoring responses from RAG (Retrieval-Augmented Generation) or Agentic pipelines against ground truth answers. Our methodology progresses from traditional correlation analysis to comprehensive Cohen's Kappa analysis that measures actual agreement patterns. The two-step approach includes: (1) a correlation test that filters judges with strong alignment, followed by (2) a human-likeness test using z-scores to identify two distinct judgment patterns: human-like judgment (|z| < 1) that mimics natural human variation, and super-consistent judgment (z > 1) that exceeds typical human-to-human agreement levels. This methodology reveals that 27 out of 54 tested LLMs achieve Tier 1 performance: 23 models exhibit human-like patterns that preserve the nuances of human judgment, while 4 models demonstrate super-consistent behavior, a pattern that could indicate either enhanced reliability or oversimplification of complex judgments. Testing 43 open-source models (1B-405B parameters) and 11 closed models (GPT, Gemini, Claude variants), we demonstrate that judge excellence is not solely dependent on model size but on specific training strategies. Our key contributions include: (1) establishing that correlation alone is insufficient for judge evaluation, (2) introducing a "Turing Test for judges" based on agreement patterns, and (3) providing a standardized benchmark for classifying LLM judges into distinct performance tiers for different evaluation needs. 

---
# SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG 

**Authors**: Xiaonan Si, Meilin Zhu, Simeng Qin, Lijia Yu, Lijun Zhang, Shuaitong Liu, Xinfeng Li, Ranjie Duan, Yang Liu, Xiaojun Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.09710)  

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models (LLMs) with external knowledge but are vulnerable to corpus poisoning and contamination attacks, which can compromise output integrity. Existing defenses often apply aggressive filtering, leading to unnecessary loss of valuable information and reduced reliability in generation. To address this problem, we propose a two-stage semantic filtering and conflict-free framework for trustworthy RAG. In the first stage, we perform a joint filter with semantic and cluster-based filtering which is guided by the Entity-intent-relation extractor (EIRE). EIRE extracts entities, latent objectives, and entity relations from both the user query and filtered documents, scores their semantic relevance, and selectively adds valuable documents into the clean retrieval database. In the second stage, we proposed an EIRE-guided conflict-aware filtering module, which analyzes semantic consistency between the query, candidate answers, and retrieved knowledge before final answer generation, filtering out internal and external contradictions that could mislead the model. Through this two-stage process, SeCon-RAG effectively preserves useful knowledge while mitigating conflict contamination, achieving significant improvements in both generation robustness and output trustworthiness. Extensive experiments across various LLMs and datasets demonstrate that the proposed SeCon-RAG markedly outperforms state-of-the-art defense methods. 

---
# ReaLM: Residual Quantization Bridging Knowledge Graph Embeddings and Large Language Models 

**Authors**: Wenbin Guo, Xin Wang, Jiaoyan Chen, Lingbing Guo, Zhao Li, Zirui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.09711)  

**Abstract**: Large Language Models (LLMs) have recently emerged as a powerful paradigm for Knowledge Graph Completion (KGC), offering strong reasoning and generalization capabilities beyond traditional embedding-based approaches. However, existing LLM-based methods often struggle to fully exploit structured semantic representations, as the continuous embedding space of pretrained KG models is fundamentally misaligned with the discrete token space of LLMs. This discrepancy hinders effective semantic transfer and limits their performance. To address this challenge, we propose ReaLM, a novel and effective framework that bridges the gap between KG embeddings and LLM tokenization through the mechanism of residual vector quantization. ReaLM discretizes pretrained KG embeddings into compact code sequences and integrates them as learnable tokens within the LLM vocabulary, enabling seamless fusion of symbolic and contextual knowledge. Furthermore, we incorporate ontology-guided class constraints to enforce semantic consistency, refining entity predictions based on class-level compatibility. Extensive experiments on two widely used benchmark datasets demonstrate that ReaLM achieves state-of-the-art performance, confirming its effectiveness in aligning structured knowledge with large-scale language models. 

---
# Emotionally Charged, Logically Blurred: AI-driven Emotional Framing Impairs Human Fallacy Detection 

**Authors**: Yanran Chen, Lynn Greschner, Roman Klinger, Michael Klenk, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2510.09695)  

**Abstract**: Logical fallacies are common in public communication and can mislead audiences; fallacious arguments may still appear convincing despite lacking soundness, because convincingness is inherently subjective. We present the first computational study of how emotional framing interacts with fallacies and convincingness, using large language models (LLMs) to systematically change emotional appeals in fallacious arguments. We benchmark eight LLMs on injecting emotional appeal into fallacious arguments while preserving their logical structures, then use the best models to generate stimuli for a human study. Our results show that LLM-driven emotional framing reduces human fallacy detection in F1 by 14.5% on average. Humans perform better in fallacy detection when perceiving enjoyment than fear or sadness, and these three emotions also correlate with significantly higher convincingness compared to neutral or other emotion states. Our work has implications for AI-driven emotional manipulation in the context of fallacious argumentation. 

---
# Table Question Answering in the Era of Large Language Models: A Comprehensive Survey of Tasks, Methods, and Evaluation 

**Authors**: Wei Zhou, Bolei Ma, Annemarie Friedrich, Mohsen Mesgar  

**Link**: [PDF](https://arxiv.org/pdf/2510.09671)  

**Abstract**: Table Question Answering (TQA) aims to answer natural language questions about tabular data, often accompanied by additional contexts such as text passages. The task spans diverse settings, varying in table representation, question/answer complexity, modality involved, and domain. While recent advances in large language models (LLMs) have led to substantial progress in TQA, the field still lacks a systematic organization and understanding of task formulations, core challenges, and methodological trends, particularly in light of emerging research directions such as reinforcement learning. This survey addresses this gap by providing a comprehensive and structured overview of TQA research with a focus on LLM-based methods. We provide a comprehensive categorization of existing benchmarks and task setups. We group current modeling strategies according to the challenges they target, and analyze their strengths and limitations. Furthermore, we highlight underexplored but timely topics that have not been systematically covered in prior research. By unifying disparate research threads and identifying open problems, our survey offers a consolidated foundation for the TQA community, enabling a deeper understanding of the state of the art and guiding future developments in this rapidly evolving area. 

---
# Boundary-Guided Policy Optimization for Memory-efficient RL of Diffusion Large Language Models 

**Authors**: Nianyi Lin, Jiajie Zhang, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.11683)  

**Abstract**: A key challenge in applying reinforcement learning (RL) to diffusion large language models (dLLMs) lies in the intractability of their likelihood functions, which are essential for the RL objective, necessitating corresponding approximation in each training step. While existing methods approximate the log-likelihoods by their evidence lower bounds (ELBOs) via customized Monte Carlo (MC) sampling, the forward computational graphs of all MC samples need to be retained for the gradient computation of non-linear terms in the RL objective, resulting in significant memory overhead. This constraint restricts feasible sample sizes, leading to imprecise likelihood approximations and ultimately distorting the RL objective. To overcome this limitation, we propose \emph{Boundary-Guided Policy Optimization} (BGPO), a memory-efficient RL algorithm that maximizes a specially constructed lower bound of the ELBO-based objective. This lower bound is carefully designed to satisfy two key properties: (1) Linearity: it is formulated in a linear sum where each term depends only on a single MC sample, thereby enabling gradient accumulation across samples and ensuring constant memory usage; (2) Equivalence: Both the value and gradient of this lower bound are equal to those of the ELBO-based objective in on-policy training, making it also an effective approximation for the original RL objective. These properties allow BGPO to adopt a large MC sample size, resulting in more accurate likelihood approximations and improved RL objective estimation, which in turn leads to enhanced performance. Experiments show that BGPO significantly outperforms previous RL algorithms for dLLMs in math problem solving, code generation, and planning tasks. 

---
# QeRL: Beyond Efficiency -- Quantization-enhanced Reinforcement Learning for LLMs 

**Authors**: Wei Huang, Yi Ge, Shuai Yang, Yicheng Xiao, Huizi Mao, Yujun Lin, Hanrong Ye, Sifei Liu, Ka Chun Cheung, Hongxu Yin, Yao Lu, Xiaojuan Qi, Song Han, Yukang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.11696)  

**Abstract**: We propose QeRL, a Quantization-enhanced Reinforcement Learning framework for large language models (LLMs). While RL is essential for LLMs' reasoning capabilities, it is resource-intensive, requiring substantial GPU memory and long rollout durations. QeRL addresses these issues by combining NVFP4 quantization with Low-Rank Adaptation (LoRA), accelerating rollout phase of RL while reducing memory overhead. Beyond efficiency, our findings show that quantization noise increases policy entropy, enhancing exploration, and enabling the discovery of better strategies during RL. To further optimize exploration, QeRL introduces an Adaptive Quantization Noise (AQN) mechanism, which dynamically adjusts noise during training. Experiments demonstrate that QeRL delivers over 1.5 times speedup in the rollout phase. Moreover, this is the first framework to enable RL training of a 32B LLM on a single H100 80GB GPU, while delivering overall speedups for RL training. It also achieves faster reward growth and higher final accuracy than 16-bit LoRA and QLoRA, while matching the performance of full-parameter fine-tuning on mathematical benchmarks such as GSM8K (90.8%) and MATH 500 (77.4%) in the 7B model. These results establish QeRL as an efficient and effective framework for RL training in LLMs. 

---
# Layout-Aware Parsing Meets Efficient LLMs: A Unified, Scalable Framework for Resume Information Extraction and Evaluation 

**Authors**: Fanwei Zhu, Jinke Yu, Zulong Chen, Ying Zhou, Junhao Ji, Zhibo Yang, Yuxue Zhang, Haoyuan Hu, Zhenghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09722)  

**Abstract**: Automated resume information extraction is critical for scaling talent acquisition, yet its real-world deployment faces three major challenges: the extreme heterogeneity of resume layouts and content, the high cost and latency of large language models (LLMs), and the lack of standardized datasets and evaluation tools. In this work, we present a layout-aware and efficiency-optimized framework for automated extraction and evaluation that addresses all three challenges. Our system combines a fine-tuned layout parser to normalize diverse document formats, an inference-efficient LLM extractor based on parallel prompting and instruction tuning, and a robust two-stage automated evaluation framework supported by new benchmark datasets. Extensive experiments show that our framework significantly outperforms strong baselines in both accuracy and efficiency. In particular, we demonstrate that a fine-tuned compact 0.6B LLM achieves top-tier accuracy while significantly reducing inference latency and computational cost. The system is fully deployed in Alibaba's intelligent HR platform, supporting real-time applications across its business units. 

---
# The Idola Tribus of AI: Large Language Models tend to perceive order where none exists 

**Authors**: Shin-nosuke Ishikawa, Masato Todo, Taiki Ogihara, Hirotsugu Ohba  

**Link**: [PDF](https://arxiv.org/pdf/2510.09709)  

**Abstract**: We present a tendency of large language models (LLMs) to generate absurd patterns despite their clear inappropriateness in a simple task of identifying regularities in number series. Several approaches have been proposed to apply LLMs to complex real-world tasks, such as providing knowledge through retrieval-augmented generation and executing multi-step tasks using AI agent frameworks. However, these approaches rely on the logical consistency and self-coherence of LLMs, making it crucial to evaluate these aspects and consider potential countermeasures. To identify cases where LLMs fail to maintain logical consistency, we conducted an experiment in which LLMs were asked to explain the patterns in various integer sequences, ranging from arithmetic sequences to randomly generated integer series. While the models successfully identified correct patterns in arithmetic and geometric sequences, they frequently over-recognized patterns that were inconsistent with the given numbers when analyzing randomly generated series. This issue was observed even in multi-step reasoning models, including OpenAI o3, o4-mini, and Google Gemini 2.5 Flash Preview Thinking. This tendency to perceive non-existent patterns can be interpreted as the AI model equivalent of Idola Tribus and highlights potential limitations in their capability for applied tasks requiring logical reasoning, even when employing chain-of-thought reasoning mechanisms. 

---
# Bag of Tricks for Subverting Reasoning-based Safety Guardrails 

**Authors**: Shuo Chen, Zhen Han, Haokun Chen, Bailan He, Shengyun Si, Jingpei Wu, Philip Torr, Volker Tresp, Jindong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11570)  

**Abstract**: Recent reasoning-based safety guardrails for Large Reasoning Models (LRMs), such as deliberative alignment, have shown strong defense against jailbreak attacks. By leveraging LRMs' reasoning ability, these guardrails help the models to assess the safety of user inputs before generating final responses. The powerful reasoning ability can analyze the intention of the input query and will refuse to assist once it detects the harmful intent hidden by the jailbreak methods. Such guardrails have shown a significant boost in defense, such as the near-perfect refusal rates on the open-source gpt-oss series. Unfortunately, we find that these powerful reasoning-based guardrails can be extremely vulnerable to subtle manipulation of the input prompts, and once hijacked, can lead to even more harmful results. Specifically, we first uncover a surprisingly fragile aspect of these guardrails: simply adding a few template tokens to the input prompt can successfully bypass the seemingly powerful guardrails and lead to explicit and harmful responses. To explore further, we introduce a bag of jailbreak methods that subvert the reasoning-based guardrails. Our attacks span white-, gray-, and black-box settings and range from effortless template manipulations to fully automated optimization. Along with the potential for scalable implementation, these methods also achieve alarmingly high attack success rates (e.g., exceeding 90% across 5 different benchmarks on gpt-oss series on both local host models and online API services). Evaluations across various leading open-source LRMs confirm that these vulnerabilities are systemic, underscoring the urgent need for stronger alignment techniques for open-sourced LRMs to prevent malicious misuse. Code is open-sourced at this https URL. 

---
# Can Tool-Integrated Reinforcement Learning Generalize Across Diverse Domains? 

**Authors**: Zhengyu Chen, Jinluan Yang, Teng Xiao, Ruochen Zhou, Luan Zhang, Xiangyu Xi, Xiaowei Shi, Wei Wang, Jinggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11184)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable capabilities in reasoning and tool utilization. However, the generalization of tool-augmented reinforcement learning (RL) across diverse domains remains underexplored. In this work, we investigate the cross-domain generalization of an LLM agent equipped with a code interpreter tool, which is exclusively trained on mathematical problem-solving tasks. Despite the restricted training domain, we evaluate the agent's performance across several distinct reasoning domains. The results reveal that RL-based tool usage learned from mathematical tasks can be effectively transferred to complex tasks in other domains, enabling great task performance and high token efficiency. To facilitate this cross-domain transfer, we propose a Tool Generalization Reinforcement Learning (TGRL) framework designed to promote domain-agnostic learning and skill migration, encompassing: (i) a standardized tool interface that abstracts domain-specific nuances through consistent formatting and explicit termination, fostering transferable invocation patterns; (ii) a dual-component reward system that decomposes rewards to incentivize generalizable behaviors like tool efficiency and reasoning abstraction, ensuring alignment and robustness across domain shifts; and (iii) an XML-based prompt template that separates thinking, tool calls, and responses to encourage modular, domain-invariant planning and coherent multi-turn interactions. Extensive experiments across diverse benchmarks validate our approach, achieving state-of-the-art performance and highlighting the cross-domain potential of Tool RL for LLM reasoning. 

---
# Automating Structural Engineering Workflows with Large Language Model Agents 

**Authors**: Haoran Liang, Yufa Zhou, Mohammad Talebi Kalaleh, Qipei Mei  

**Link**: [PDF](https://arxiv.org/pdf/2510.11004)  

**Abstract**: We introduce $\textbf{MASSE}$, the first Multi-Agent System for Structural Engineering, effectively integrating large language model (LLM)-based agents with real-world engineering workflows. Structural engineering is a fundamental yet traditionally stagnant domain, with core workflows remaining largely unchanged for decades despite its substantial economic impact and global market size. Recent advancements in LLMs have significantly enhanced their ability to perform complex reasoning, long-horizon planning, and precise tool utilization -- capabilities well aligned with structural engineering tasks such as interpreting design codes, executing load calculations, and verifying structural capacities. We present a proof-of-concept showing that most real-world structural engineering workflows can be fully automated through a training-free LLM-based multi-agent system. MASSE enables immediate deployment in professional environments, and our comprehensive validation on real-world case studies demonstrates that it can reduce expert workload from approximately two hours to mere minutes, while enhancing both reliability and accuracy in practical engineering scenarios. 

---
# Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation 

**Authors**: Jiaying Wu, Zihang Fu, Haonan Wang, Fanxiao Li, Min-Yen Kan  

**Link**: [PDF](https://arxiv.org/pdf/2510.11423)  

**Abstract**: Community Notes, the crowd-sourced misinformation governance system on X (formerly Twitter), enables users to flag misleading posts, attach contextual notes, and vote on their helpfulness. However, our analysis of 30.8K health-related notes reveals significant latency, with a median delay of 17.6 hours before the first note receives a helpfulness status. To improve responsiveness during real-world misinformation surges, we propose CrowdNotes+, a unified framework that leverages large language models (LLMs) to augment Community Notes for faster and more reliable health misinformation governance. CrowdNotes+ integrates two complementary modes: (1) evidence-grounded note augmentation and (2) utility-guided note automation, along with a hierarchical three-step evaluation that progressively assesses relevance, correctness, and helpfulness. We instantiate the framework through HealthNotes, a benchmark of 1.2K helpfulness-annotated health notes paired with a fine-tuned helpfulness judge. Experiments on fifteen LLMs reveal an overlooked loophole in current helpfulness evaluation, where stylistic fluency is mistaken for factual accuracy, and demonstrate that our hierarchical evaluation and LLM-augmented generation jointly enhance factual precision and evidence utility. These results point toward a hybrid human-AI governance model that improves both the rigor and timeliness of crowd-sourced fact-checking. 

---
# ReLook: Vision-Grounded RL with a Multimodal LLM Critic for Agentic Web Coding 

**Authors**: Yuhang Li, Chenchen Zhang, Ruilin Lv, Ao Liu, Ken Deng, Yuanxing Zhang, Jiaheng Liu, Wiggin Zhou, Bo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.11498)  

**Abstract**: While Large Language Models (LLMs) excel at algorithmic code generation, they struggle with front-end development, where correctness is judged on rendered pixels and interaction. We present ReLook, an agentic, vision-grounded reinforcement learning framework that empowers an agent to close a robust generate--diagnose--refine loop by invoking a multimodal LLM (MLLM) as a tool. During training, the agent uses the MLLM-in-the-loop both as a visual critic--scoring code with screenshots--and as a source of actionable, vision-grounded feedback; a strict zero-reward rule for invalid renders anchors renderability and prevents reward hacking. To prevent behavioral collapse, we introduce Forced Optimization, a strict acceptance rule that admits only improving revisions, yielding monotonically better trajectories. At inference, we decouple the critic and run a lightweight, critic-free self-edit cycle, keeping latency comparable to base decoding while retaining most of the gains. Across three widely used benchmarks, ReLook consistently outperforms strong baselines in vision-grounded front-end code generation, highlighting the benefits of agentic perception, visual rewards, and training-inference decoupling. 

---
# ENIGMA: The Geometry of Reasoning and Alignment in Large-Language Models 

**Authors**: Gareth Seneque, Lap-Hang Ho, Nafise Erfanian Saeedi, Jeffrey Molendijk, Ariel Kupermann, Tim Elson  

**Link**: [PDF](https://arxiv.org/pdf/2510.11278)  

**Abstract**: We present Entropic Mutual-Information Geometry Large-Language Model Alignment (ENIGMA), a novel approach to Large-Language Model (LLM) training that jointly improves reasoning, alignment and robustness by treating an organisation's policies/principles as directions to move on a model's information manifold. Our single-loop trainer combines Group-Relative Policy Optimisation (GRPO), an on-policy, critic-free RL method with Chain-of-Thought (CoT)-format only rewards; a Self-Supervised Alignment with Mutual Information (SAMI)-style symmetric InfoNCE auxiliary; and an entropic Sinkhorn optimal-transport regulariser on hidden-state distributions to bound geometry drift. We also introduce infoNCE metrics that specialise to a standard MI lower bound under matched negatives to measure how strongly a model's CoT encodes these policies. These metrics include a Sufficiency Index (SI) that enables the selection and creation of principles that maximise downstream performance prior to training. In our experiments using small (1B) LLMs, high-SI principles predict steadier training dynamics and improved benchmark performance over GRPO ablations. Our information-geometry analysis of trained models validates desirable structural change in the manifold. These results support our hypothesis that reasoning, alignment, and robustness are projections of a single informationgeometric objective, and that models trained using ENIGMA demonstrate principled reasoning without the use of a reward model, offering a path to trusted capability 

---
# EAGER: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling 

**Authors**: Daniel Scalena, Leonidas Zotos, Elisabetta Fersini, Malvina Nissim, Ahmet Üstün  

**Link**: [PDF](https://arxiv.org/pdf/2510.11170)  

**Abstract**: With the rise of reasoning language models and test-time scaling methods as a paradigm for improving model performance, substantial computation is often required to generate multiple candidate sequences from the same prompt. This enables exploration of different reasoning paths toward the correct solution, however, allocates the same compute budget for each prompt. Grounded on the assumption that different prompts carry different degrees of complexity, and thus different computation needs, we propose EAGer, a training-free generation method that leverages model uncertainty through token-wise entropy distribution to reduce redundant computation and concurrently improve overall performance. EAGer allows branching to multiple reasoning paths only in the presence of high-entropy tokens, and then reallocates the saved compute budget to the instances where exploration of alternative paths is most needed. We find that across multiple open-source models on complex reasoning benchmarks such as AIME 2025, EAGer can reallocate the budget without accessing target labels, achieving the best efficiency-performance trade-off in terms of reasoning length and Pass@k. When target labels are accessible, EAGer generates up to 65% fewer tokens (hence saving compute) and achieves up to 37% improvement in Pass@k compared to the Full Parallel Sampling. 

---
# A Layered Intuition -- Method Model with Scope Extension for LLM Reasoning 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2510.10592)  

**Abstract**: Existing studies have introduced method-based reasoning and scope extension as approaches to enhance Large Language Model (LLM) performance beyond direct matrix mappings. Building on these foundations, this paper summarizes and integrates these ideas into a unified Intuition-Method Layered Model with Scope Extension, designed to address indirected (unseen) issues more systematically. In this framework, intuition-based thinking provides rapid first-reaction answers, while method-based thinking decouples questions and solutions into transferable reasoning units. Scope extension is then applied to broaden applicability, including vertical (cause analysis), horizontal (parallel and generalized issues), and for the first time, temporal and spatial extensions, which expand reasoning across time and contextual dimensions. These extensions are organized into systematic knowledge trees that interconnect into a knowledge network, thereby increasing adaptability. To quantitatively evaluate this process, we propose the entropy of method extension, which measures the independence and diversity of extensions as an indicator of the system's capacity to solve unseen questions. By logically connecting existing approaches with new extensions and introducing an entropy-based evaluation framework, this work advances toward a more robust and extensible reasoning paradigm for LLMs in real-world problem-solving. 

---
# The Personalization Trap: How User Memory Alters Emotional Reasoning in LLMs 

**Authors**: Xi Fang, Weijie Xu, Yuchong Zhang, Stephanie Eckman, Scott Nickleach, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2510.09905)  

**Abstract**: When an AI assistant remembers that Sarah is a single mother working two jobs, does it interpret her stress differently than if she were a wealthy executive? As personalized AI systems increasingly incorporate long-term user memory, understanding how this memory shapes emotional reasoning is critical. We investigate how user memory affects emotional intelligence in large language models (LLMs) by evaluating 15 models on human validated emotional intelligence tests. We find that identical scenarios paired with different user profiles produce systematically divergent emotional interpretations. Across validated user independent emotional scenarios and diverse user profiles, systematic biases emerged in several high-performing LLMs where advantaged profiles received more accurate emotional interpretations. Moreover, LLMs demonstrate significant disparities across demographic factors in emotion understanding and supportive recommendations tasks, indicating that personalization mechanisms can embed social hierarchies into models emotional reasoning. These results highlight a key challenge for memory enhanced AI: systems designed for personalization may inadvertently reinforce social inequalities. 

---
# The Social Cost of Intelligence: Emergence, Propagation, and Amplification of Stereotypical Bias in Multi-Agent Systems 

**Authors**: Thi-Nhung Nguyen, Linhao Luo, Thuy-Trang Vu, Dinh Phung  

**Link**: [PDF](https://arxiv.org/pdf/2510.10943)  

**Abstract**: Bias in large language models (LLMs) remains a persistent challenge, manifesting in stereotyping and unfair treatment across social groups. While prior research has primarily focused on individual models, the rise of multi-agent systems (MAS), where multiple LLMs collaborate and communicate, introduces new and largely unexplored dynamics in bias emergence and propagation. In this work, we present a comprehensive study of stereotypical bias in MAS, examining how internal specialization, underlying LLMs and inter-agent communication protocols influence bias robustness, propagation, and amplification. We simulate social contexts where agents represent different social groups and evaluate system behavior under various interaction and adversarial scenarios. Experiments on three bias benchmarks reveal that MAS are generally less robust than single-agent systems, with bias often emerging early through in-group favoritism. However, cooperative and debate-based communication can mitigate bias amplification, while more robust underlying LLMs improve overall system stability. Our findings highlight critical factors shaping fairness and resilience in multi-agent LLM systems. 

---
# The Geometry of Reasoning: Flowing Logics in Representation Space 

**Authors**: Yufa Zhou, Yixiao Wang, Xunjian Yin, Shuyan Zhou, Anru R. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09782)  

**Abstract**: We study how large language models (LLMs) ``think'' through their representation space. We propose a novel geometric framework that models an LLM's reasoning as flows -- embedding trajectories evolving where logic goes. We disentangle logical structure from semantics by employing the same natural deduction propositions with varied semantic carriers, allowing us to test whether LLMs internalize logic beyond surface form. This perspective connects reasoning with geometric quantities such as position, velocity, and curvature, enabling formal analysis in representation and concept spaces. Our theory establishes: (1) LLM reasoning corresponds to smooth flows in representation space, and (2) logical statements act as local controllers of these flows' velocities. Using learned representation proxies, we design controlled experiments to visualize and quantify reasoning flows, providing empirical validation of our theoretical framework. Our work serves as both a conceptual foundation and practical tools for studying reasoning phenomenon, offering a new lens for interpretability and formal analysis of LLMs' behavior. 

---
# Rediscovering Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning 

**Authors**: Xiaoyun Zhang, Xiaojian Yuan, Di Huang, Wang You, Chen Hu, Jingqing Ruan, Kejiang Chen, Xing Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10959)  

**Abstract**: Reasoning ability has become a defining capability of Large Language Models (LLMs), with Reinforcement Learning with Verifiable Rewards (RLVR) emerging as a key paradigm to enhance it. However, RLVR training often suffers from policy entropy collapse, where the policy becomes overly deterministic, hindering exploration and limiting reasoning performance. While entropy regularization is a common remedy, its effectiveness is highly sensitive to the fixed coefficient, making it unstable across tasks and models. In this work, we revisit entropy regularization in RLVR and argue that its potential has been largely underestimated. Our analysis shows that (i) tasks of varying difficulty demand distinct exploration intensities, and (ii) balanced exploration may require the policy entropy to be maintained within a moderate range below its initial level. Therefore, we propose Adaptive Entropy Regularization (AER)--a framework that dynamically balances exploration and exploitation via three components: difficulty-aware coefficient allocation, initial-anchored target entropy, and dynamic global coefficient adjustment. Experiments on multiple mathematical reasoning benchmarks show that AER consistently outperforms baselines, improving both reasoning accuracy and exploration capability. 

---
# ArtPerception: ASCII Art-based Jailbreak on LLMs with Recognition Pre-test 

**Authors**: Guan-Yan Yang, Tzu-Yu Cheng, Ya-Wen Teng, Farn Wanga, Kuo-Hui Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2510.10281)  

**Abstract**: The integration of Large Language Models (LLMs) into computer applications has introduced transformative capabilities but also significant security challenges. Existing safety alignments, which primarily focus on semantic interpretation, leave LLMs vulnerable to attacks that use non-standard data representations. This paper introduces ArtPerception, a novel black-box jailbreak framework that strategically leverages ASCII art to bypass the security measures of state-of-the-art (SOTA) LLMs. Unlike prior methods that rely on iterative, brute-force attacks, ArtPerception introduces a systematic, two-phase methodology. Phase 1 conducts a one-time, model-specific pre-test to empirically determine the optimal parameters for ASCII art recognition. Phase 2 leverages these insights to launch a highly efficient, one-shot malicious jailbreak attack. We propose a Modified Levenshtein Distance (MLD) metric for a more nuanced evaluation of an LLM's recognition capability. Through comprehensive experiments on four SOTA open-source LLMs, we demonstrate superior jailbreak performance. We further validate our framework's real-world relevance by showing its successful transferability to leading commercial models, including GPT-4o, Claude Sonnet 3.7, and DeepSeek-V3, and by conducting a rigorous effectiveness analysis against potential defenses such as LLaMA Guard and Azure's content filters. Our findings underscore that true LLM security requires defending against a multi-modal space of interpretations, even within text-only inputs, and highlight the effectiveness of strategic, reconnaissance-based attacks. Content Warning: This paper includes potentially harmful and offensive model outputs. 

---
# Task-Aware Resolution Optimization for Visual Large Language Models 

**Authors**: Weiqing Luo, Zhen Tan, Yifan Li, Xinyu Zhao, Kwonjoon Lee, Behzad Dariush, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.09822)  

**Abstract**: Real-world vision-language applications demand varying levels of perceptual granularity. However, most existing visual large language models (VLLMs), such as LLaVA, pre-assume a fixed resolution for downstream tasks, which leads to subpar performance. To address this problem, we first conduct a comprehensive and pioneering investigation into the resolution preferences of different vision-language tasks, revealing a correlation between resolution preferences with image complexity, and uncertainty variance of the VLLM at different image input resolutions. Building on this insight, we propose an empirical formula to determine the optimal resolution for a given vision-language task, combining these two factors. Second, based on rigorous experiments, we propose a novel parameter-efficient fine-tuning technique to extend the visual input resolution of pre-trained VLLMs to the identified optimal resolution. Extensive experiments on various vision-language tasks validate the effectiveness of our method. 

---
# Group-Adaptive Adversarial Learning for Robust Fake News Detection Against Malicious Comments 

**Authors**: Zhao Tong, Chunlin Gong, Yimeng Gu, Haichao Shi, Qiang Liu, Shu Wu, Xiao-Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09712)  

**Abstract**: The spread of fake news online distorts public judgment and erodes trust in social media platforms. Although recent fake news detection (FND) models perform well in standard settings, they remain vulnerable to adversarial comments-authored by real users or by large language models (LLMs)-that subtly shift model decisions. In view of this, we first present a comprehensive evaluation of comment attacks to existing fake news detectors and then introduce a group-adaptive adversarial training strategy to improve the robustness of FND models. To be specific, our approach comprises three steps: (1) dividing adversarial comments into three psychologically grounded categories: perceptual, cognitive, and societal; (2) generating diverse, category-specific attacks via LLMs to enhance adversarial training; and (3) applying a Dirichlet-based adaptive sampling mechanism (InfoDirichlet Adjusting Mechanism) that dynamically adjusts the learning focus across different comment categories during training. Experiments on benchmark datasets show that our method maintains strong detection accuracy while substantially increasing robustness to a wide range of adversarial comment perturbations. 

---
# Analyzing and Internalizing Complex Policy Documents for LLM Agents 

**Authors**: Jiateng Liu, Zhenhailong Wang, Xiaojiang Huang, Yingjie Li, Xing Fan, Xiang Li, Chenlei Guo, Ruhi Sarikaya, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.11588)  

**Abstract**: Large Language Model (LLM)-based agentic systems rely on in-context policy documents encoding diverse business rules. As requirements grow, these documents expand rapidly, causing high computational overhead. This motivates developing internalization methods that embed policy documents into model priors while preserving performance. Prior prompt compression work targets generic prompts, but agentic policy documents span multiple complexity levels and require deeper reasoning, making internalization harder. We introduce CC-Gen, an agentic benchmark generator with Controllable Complexity across four levels, enabling systematic evaluation of agents' ability to handle complexity and offering a unified framework for assessing policy internalization. Our analysis shows that complex policy specifications governing workflows pose major reasoning challenges. Supporting internalization with gold user agent interaction trajectories containing chain-of-thought (CoT) annotations via supervised fine-tuning (SFT) is data-intensive and degrades sharply as policy complexity increases. To mitigate data and reasoning burdens, we propose Category-Aware Policy Continued Pretraining (CAP-CPT). Our automated pipeline parses policy documents to extract key specifications, grouping them into factual, behavioral, and conditional categories, and isolating complex conditions that drive workflow complexity. This guides targeted data synthesis and enables agents to internalize policy information through an autoregressive pretraining loss. Experiments show CAP-CPT improves SFT baselines in all settings, with up to 41% and 22% gains on Qwen-3-32B, achieving 97.3% prompt length reduction on CC-Gen and further enhancing tau-Bench with minimal SFT data. 

---
# SR-Scientist: Scientific Equation Discovery With Agentic AI 

**Authors**: Shijie Xia, Yuhan Sun, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11661)  

**Abstract**: Recently, Large Language Models (LLMs) have been applied to scientific equation discovery, leveraging their embedded scientific knowledge for hypothesis generation. However, current methods typically confine LLMs to the role of an equation proposer within search algorithms like genetic programming. In this paper, we present SR-Scientist, a framework that elevates the LLM from a simple equation proposer to an autonomous AI scientist that writes code to analyze data, implements the equation as code, submits it for evaluation, and optimizes the equation based on experimental feedback. Specifically, we wrap the code interpreter into a set of tools for data analysis and equation evaluation. The agent is instructed to optimize the equation by utilizing these tools over a long horizon with minimal human-defined pipelines. Empirical results show that SR-Scientist outperforms baseline methods by an absolute margin of 6% to 35% on datasets covering four science disciplines. Additionally, we demonstrate our method's robustness to noise, the generalization of the discovered equations to out-of-domain data, and their symbolic accuracy. Furthermore, we develop an end-to-end reinforcement learning framework to enhance the agent's capabilities. 

---
# Automated Skill Decomposition Meets Expert Ontologies: Bridging the Granularity Gap with LLMs 

**Authors**: Le Ngoc Luyen, Marie-Hélène Abel  

**Link**: [PDF](https://arxiv.org/pdf/2510.11313)  

**Abstract**: This paper investigates automated skill decomposition using Large Language Models (LLMs) and proposes a rigorous, ontology-grounded evaluation framework. Our framework standardizes the pipeline from prompting and generation to normalization and alignment with ontology nodes. To evaluate outputs, we introduce two metrics: a semantic F1-score that uses optimal embedding-based matching to assess content accuracy, and a hierarchy-aware F1-score that credits structurally correct placements to assess granularity. We conduct experiments on ROME-ESCO-DecompSkill, a curated subset of parents, comparing two prompting strategies: zero-shot and leakage-safe few-shot with exemplars. Across diverse LLMs, zero-shot offers a strong baseline, while few-shot consistently stabilizes phrasing and granularity and improves hierarchy-aware alignment. A latency analysis further shows that exemplar-guided prompts are competitive - and sometimes faster - than unguided zero-shot due to more schema-compliant completions. Together, the framework, benchmark, and metrics provide a reproducible foundation for developing ontology-faithful skill decomposition systems. 

---
# Aligning Deep Implicit Preferences by Learning to Reason Defensively 

**Authors**: Peiming Li, Zhiyuan Hu, Yang Tang, Shiyu Li, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.11194)  

**Abstract**: Personalized alignment is crucial for enabling Large Language Models (LLMs) to engage effectively in user-centric interactions. However, current methods face a dual challenge: they fail to infer users' deep implicit preferences (including unstated goals, semantic context and risk tolerances), and they lack the defensive reasoning required to navigate real-world ambiguity. This cognitive gap leads to responses that are superficial, brittle and short-sighted. To address this, we propose Critique-Driven Reasoning Alignment (CDRA), which reframes alignment from a scalar reward-matching task into a structured reasoning process. First, to bridge the preference inference gap, we introduce the DeepPref benchmark. This dataset, comprising 3000 preference-query pairs across 20 topics, is curated by simulating a multi-faceted cognitive council that produces critique-annotated reasoning chains to deconstruct query semantics and reveal latent risks. Second, to instill defensive reasoning, we introduce the Personalized Generative Process Reward Model (Pers-GenPRM), which frames reward modeling as a personalized reasoning task. It generates a critique chain to evaluate a response's alignment with user preferences before outputting a final score based on this rationale. Ultimately, this interpretable, structured reward signal guides policy model through Critique-Driven Policy Alignment, a process-level online reinforcement learning algorithm integrating both numerical and natural language feedback. Experiments demonstrate that CDRA excels at discovering and aligning with users' true preferences while executing robust reasoning. Our code and dataset are available at this https URL. 

---
# From <Answer> to <Think>: Multidimensional Supervision of Reasoning Process for LLM Optimization 

**Authors**: Beining Wang, Weihang Su, Hongtao Tian, Tao Yang, Yujia Zhou, Ting Yao, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11457)  

**Abstract**: Improving the multi-step reasoning ability of Large Language Models (LLMs) is a critical yet challenging task. The dominant paradigm, outcome-supervised reinforcement learning (RLVR), rewards only correct final answers, often propagating flawed reasoning and suffering from sparse reward signals. While process-level reward models (PRMs) provide denser, step-by-step feedback, they lack generalizability and interpretability, requiring task-specific segmentation of the reasoning process. To this end, we propose the Dimension-level Reward Model (DRM), a new supervision framework that bridges the gap between these two approaches. DRM evaluates the quality of a reasoning process along three fundamental, complementary, and interpretable dimensions: Confidence for uncertainty calibration, Relevance for semantic alignment, and Coherence for logical consistency. Together, these dimensions capture aspects beyond final answer correctness and enable interpretable assessment without requiring ground truth answers. Experimental results show that DRM provides effective supervision signals, guides the optimization of LLMs and enhances their reasoning ability. In particular, DRM-supervised training achieves consistent gains on both in-distribution and out-of-distribution open-domain tasks, including mathematics, question answering, code execution, and puzzles. Our findings demonstrate that multidimensional supervision of the reasoning process can improve the generalized reasoning ability of LLMs beyond the training distribution. 

---
# PADME: Procedure Aware DynaMic Execution 

**Authors**: Deepeka Garg, Sihan Zeng, Annapoorani L. Narayanan, Sumitra Ganesh, Leo Ardon  

**Link**: [PDF](https://arxiv.org/pdf/2510.11281)  

**Abstract**: Learning to autonomously execute long-horizon procedures from natural language remains a core challenge for intelligent agents. Free-form instructions such as recipes, scientific protocols, or business workflows encode rich procedural knowledge, but their variability and lack of structure cause agents driven by large language models (LLMs) to drift or fail during execution. We introduce Procedure Aware DynaMic Execution (PADME), an agent framework that produces and exploits a graph-based representation of procedures. Unlike prior work that relies on manual graph construction or unstructured reasoning, PADME autonomously transforms procedural text into executable graphs that capture task dependencies, decision points, and reusable subroutines. Central to PADME is a two-phase methodology; Teach phase, which focuses on systematic structuring, enrichment with executable logic of procedures, followed by Execute phase, which enables dynamic execution in response to real-time inputs and environment feedback. This separation ensures quality assurance and scalability, allowing expert knowledge to be encoded once and reliably reused across varying contexts. The graph representation also provides an inductive bias that reduces error accumulation in long-horizon reasoning, underscoring the importance of structured procedure modeling for reliable agent-driven automation. Empirically, PADME achieves state-of-the-art performance on four diverse benchmarks, including ALFWorld and ScienceWorld. These results demonstrate that agents equipped with graph-based procedure representations offer a powerful intermediate abstraction for robust and generalizable execution. 

---
# LLMs as Strategic Agents: Beliefs, Best Response Behavior, and Emergent Heuristics 

**Authors**: Enric Junque de Fortuny, Veronica Roberta Cappelli  

**Link**: [PDF](https://arxiv.org/pdf/2510.10813)  

**Abstract**: Large Language Models (LLMs) are increasingly applied to domains that require reasoning about other agents' behavior, such as negotiation, policy design, and market simulation, yet existing research has mostly evaluated their adherence to equilibrium play or their exhibited depth of reasoning. Whether they display genuine strategic thinking, understood as the coherent formation of beliefs about other agents, evaluation of possible actions, and choice based on those beliefs, remains unexplored. We develop a framework to identify this ability by disentangling beliefs, evaluation, and choice in static, complete-information games, and apply it across a series of non-cooperative environments. By jointly analyzing models' revealed choices and reasoning traces, and introducing a new context-free game to rule out imitation from memorization, we show that current frontier models exhibit belief-coherent best-response behavior at targeted reasoning depths. When unconstrained, they self-limit their depth of reasoning and form differentiated conjectures about human and synthetic opponents, revealing an emergent form of meta-reasoning. Under increasing complexity, explicit recursion gives way to internally generated heuristic rules of choice that are stable, model-specific, and distinct from known human biases. These findings indicate that belief coherence, meta-reasoning, and novel heuristic formation can emerge jointly from language modeling objectives, providing a structured basis for the study of strategic cognition in artificial agents. 

---
# Adaptive Selection of Symbolic Languages for Improving LLM Logical Reasoning 

**Authors**: Xiangyu Wang, Haocheng Yang, Fengxiang Cheng, Fenrong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10703)  

**Abstract**: Large Language Models (LLMs) still struggle with complex logical reasoning. While previous works achieve remarkable improvements, their performance is highly dependent on the correctness of translating natural language (NL) problems into a symbolic language (SL). Though numerous works focusing on improving this translation accuracy, they only consider the similarity between the meaning of SL and NL, overlooking another crucial influencing factor, the selection of the target SL type itself. For example, first-order logic language specializes in logical reasoning with categorical syllogisms and complex quantifiers, while Boolean satisfiability formalism excels at representing constraint satisfaction like partial problems. To our knowledge, this is the first paper to claim and verify that different NL logical reasoning problem corresponds to different optimal SL formalization for translation. Based on this, we propose a methods to improve the logical reasoning performance of LLMs by adaptively selecting the most suitable SL for each problem prior to translation. Specifically, we leverage LLMs to select the target SL among first-order logic, logic programming and Boolean satisfiability and then translate the problem in NL to target SL expressions as well as employ the corresponding logical solver to derive the final answer. Experimental results on benchmarks show that our adaptive selection method significantly outperforms translating all into single SL and randomly selecting the SL. On a mixed dataset of these benchmarks, our approach achieves 96% accuracy, which improving performance by 25% compared to the second highest accuracy from the first-order logic translation. 

---
# OmniVideoBench: Towards Audio-Visual Understanding Evaluation for Omni MLLMs 

**Authors**: Caorui Li, Yu Chen, Yiyan Ji, Jin Xu, Zhenyu Cui, Shihao Li, Yuanxing Zhang, Jiafu Tang, Zhenghao Song, Dingling Zhang, Ying He, Haoxiang Liu, Yuxuan Wang, Qiufeng Wang, Zhenhe Wu, Jiehui Luo, Zhiyu Pan, Weihao Xie, Chenchen Zhang, Zhaohui Wang, Jiayi Tian, Yanghai Wang, Zhe Cao, Minxin Dai, Ke Wang, Runzhe Wen, Yinghao Ma, Yaning Pan, Sungkyun Chang, Termeh Taheri, Haiwen Xia, Christos Plachouras, Emmanouil Benetos, Yizhi Li, Ge Zhang, Jian Yang, Tianhao Peng, Zili Wang, Minghao Liu, Junran Peng, Zhaoxiang Zhang, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10689)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have demonstrated substantial potential in video understanding. However, existing benchmarks fail to comprehensively evaluate synergistic reasoning capabilities across audio and visual modalities, often neglecting either one of the modalities or integrating them in a logically inconsistent manner. To bridge this gap, we introduce OmniVideoBench, a large-scale and rigorously designed benchmark dedicated to assessing synergistic audio-visual understanding, with a strong emphasis on modality complementarity and logical consistency. Specifically, OmniVideoBench comprises 1000 high-quality question-answer(QA) pairs, each annotated with step-by-step reasoning traces, derived from 628 diverse videos ranging from several seconds to 30 minutes, and manually verified to guarantee complete correctness and uniqueness. Moreover, OmniVideoBench encompasses 13 carefully designed question types, covering temporal reasoning, spatial localization, counting, causal inference, summarization, and beyond, thereby capturing the essential challenges of video understanding. Evaluation of multiple MLLMs on OmniVideoBench reveals a pronounced gap between model performance and human reasoning, with open-source models lagging significantly behind their closed-source counterparts, underscoring the inherent difficulty of genuine audio-visual reasoning. We will release OmniVideoBench to foster the development of MLLMs with stronger and more generalizable reasoning capabilities. 

---
# Unlocking Exploration in RLVR: Uncertainty-aware Advantage Shaping for Deeper Reasoning 

**Authors**: Can Xie, Ruotong Pan, Xiangyu Wu, Yunfei Zhang, Jiayi Fu, Tingting Gao, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.10649)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has shown significant promise for enhancing the reasoning capabilities of large language models (LLMs). However, prevailing algorithms like GRPO broadcast a uniform advantage signal across all tokens in a sequence. This coarse-grained approach overlooks the pivotal role of uncertain, high-stakes decisions during reasoning, leading to inefficient exploration and the well-documented problem of entropy collapse. To address this, we introduce UnCertainty-aware Advantage Shaping (UCAS), a model-free method that refines credit assignment by leveraging the model's internal uncertainty signals. UCAS operates in two stages: it first modulates the response-level advantage using the model's overall self-confidence, and then applies a token-level penalty based on raw logit certainty. This dual mechanism encourages exploration of high-uncertainty paths that yield correct answers while penalizing overconfident yet erroneous reasoning, effectively balancing the exploration-exploitation trade-off. Extensive experiments on five mathematical reasoning benchmarks show that UCAS significantly outperforms strong RLVR baselines across multiple model scales, including 1.5B and 7B. Our analysis confirms that UCAS not only achieves higher rewards but also promotes greater reasoning diversity and successfully mitigates entropy collapse. 

---
# Simpliflow: A Lightweight Open-Source Framework for Rapid Creation and Deployment of Generative Agentic AI Workflows 

**Authors**: Deven Panchal  

**Link**: [PDF](https://arxiv.org/pdf/2510.10675)  

**Abstract**: Generative Agentic AI systems are emerging as a powerful paradigm for automating complex, multi-step tasks. However, many existing frameworks for building these systems introduce significant complexity, a steep learning curve, and substantial boilerplate code, hindering rapid prototyping and deployment. This paper introduces simpliflow, a lightweight, open-source Python framework designed to address these challenges. simpliflow enables the rapid development and orchestration of linear, deterministic agentic workflows through a declarative, JSON-based configuration. Its modular architecture decouples agent management, workflow execution, and post-processing, promoting ease of use and extensibility. By integrating with LiteLLM, it supports over 100 Large Language Models (LLMs) out-of-the-box. We present the architecture, operational flow, and core features of simpliflow, demonstrating its utility through diverse use cases ranging from software development simulation to real-time system interaction. A comparative analysis with prominent frameworks like LangChain and AutoGen highlights simpliflow's unique position as a tool optimized for simplicity, control, and speed in deterministic workflow environments. 

---
# EA4LLM: A Gradient-Free Approach to Large Language Model Optimization via Evolutionary Algorithms 

**Authors**: WenTao Liu, Siyu Song, Hao Hao, Aimin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.10603)  

**Abstract**: In recent years, large language models (LLMs) have made remarkable progress, with model optimization primarily relying on gradient-based optimizers such as Adam. However, these gradient-based methods impose stringent hardware requirements, demanding high-concurrency, high-memory GPUs. Moreover, they require all neural network operations to be differentiable, thereby excluding many promising non-differentiable architectures from practical use. To address these limitations, we propose a method for optimizing LLMs using evolutionary algorithms (EA4LLM) and, for the first time, successfully demonstrate its capability to train a 1-billion-parameter LLM from the pre-trained stage. We conduct extensive experiments and provide key insights into how evolutionary algorithms can effectively optimize neural networks. Our work challenges the prevailing assumption that gradient-based optimization is the only viable approach for training neural networks. It also holds significant potential to reduce the computational cost of training large language models, thereby enabling groups with limited computational resources to participate in deep learning research. 

---
# ELAIPBench: A Benchmark for Expert-Level Artificial Intelligence Paper Understanding 

**Authors**: Xinbang Dai, Huikang Hu, Yongrui Chen, Jiaqi Li, Rihui Jin, Yuyang Zhang, Xiaoguang Li, Lifeng Shang, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2510.10549)  

**Abstract**: While large language models (LLMs) excel at many domain-specific tasks, their ability to deeply comprehend and reason about full-length academic papers remains underexplored. Existing benchmarks often fall short of capturing such depth, either due to surface-level question design or unreliable evaluation metrics. To address this gap, we introduce ELAIPBench, a benchmark curated by domain experts to evaluate LLMs' comprehension of artificial intelligence (AI) research papers. Developed through an incentive-driven, adversarial annotation process, ELAIPBench features 403 multiple-choice questions from 137 papers. It spans three difficulty levels and emphasizes non-trivial reasoning rather than shallow retrieval. Our experiments show that the best-performing LLM achieves an accuracy of only 39.95%, far below human performance. Moreover, we observe that frontier LLMs equipped with a thinking mode or a retrieval-augmented generation (RAG) system fail to improve final results-even harming accuracy due to overthinking or noisy retrieval. These findings underscore the significant gap between current LLM capabilities and genuine comprehension of academic papers. 

---
# Trace Length is a Simple Uncertainty Signal in Reasoning Models 

**Authors**: Siddartha Devic, Charlotte Peale, Arwen Bradley, Sinead Williamson, Preetum Nakkiran, Aravind Gollakota  

**Link**: [PDF](https://arxiv.org/pdf/2510.10409)  

**Abstract**: Uncertainty quantification for LLMs is a key research direction towards addressing hallucination and other issues that limit their reliable deployment. In this work, we show that reasoning trace length is a simple and useful confidence estimator in large reasoning models. Through comprehensive experiments across multiple models, datasets, and prompts, we show that trace length performs in comparable but complementary ways to other zero-shot confidence estimators such as verbalized confidence. Our work reveals that reasoning post-training fundamentally alters the relationship between trace length and accuracy, going beyond prior work that had shown that post-training causes traces to grow longer in general (e.g., "overthinking"). We investigate the mechanisms behind trace length's performance as a confidence signal, observing that the effect remains even after adjusting for confounders such as problem difficulty and GRPO-induced length bias. We identify high-entropy or "forking" tokens as playing a key role in the mechanism. Our findings demonstrate that reasoning post-training enhances uncertainty quantification beyond verbal expressions, and establish trace length as a practical confidence measure for large reasoning models. 

---
# Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction 

**Authors**: Sihang Zeng, Yujuan Fu, Sitong Zhou, Zixuan Yu, Lucas Jing Liu, Jun Wen, Matthew Thompson, Ruth Etzioni, Meliha Yetisgen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10454)  

**Abstract**: Large language models (LLMs) offer a generalizable approach for modeling patient trajectories, but suffer from the long and noisy nature of electronic health records (EHR) data in temporal reasoning. To address these challenges, we introduce Traj-CoA, a multi-agent system involving chain-of-agents for patient trajectory modeling. Traj-CoA employs a chain of worker agents to process EHR data in manageable chunks sequentially, distilling critical events into a shared long-term memory module, EHRMem, to reduce noise and preserve a comprehensive timeline. A final manager agent synthesizes the worker agents' summary and the extracted timeline in EHRMem to make predictions. In a zero-shot one-year lung cancer risk prediction task based on five-year EHR data, Traj-CoA outperforms baselines of four categories. Analysis reveals that Traj-CoA exhibits clinically aligned temporal reasoning, establishing it as a promisingly robust and generalizable approach for modeling complex patient trajectories. 

---
# Hierarchical Optimization via LLM-Guided Objective Evolution for Mobility-on-Demand Systems 

**Authors**: Yi Zhang, Yushen Long, Yun Ni, Liping Huang, Xiaohong Wang, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10644)  

**Abstract**: Online ride-hailing platforms aim to deliver efficient mobility-on-demand services, often facing challenges in balancing dynamic and spatially heterogeneous supply and demand. Existing methods typically fall into two categories: reinforcement learning (RL) approaches, which suffer from data inefficiency, oversimplified modeling of real-world dynamics, and difficulty enforcing operational constraints; or decomposed online optimization methods, which rely on manually designed high-level objectives that lack awareness of low-level routing dynamics. To address this issue, we propose a novel hybrid framework that integrates large language model (LLM) with mathematical optimization in a dynamic hierarchical system: (1) it is training-free, removing the need for large-scale interaction data as in RL, and (2) it leverages LLM to bridge cognitive limitations caused by problem decomposition by adaptively generating high-level objectives. Within this framework, LLM serves as a meta-optimizer, producing semantic heuristics that guide a low-level optimizer responsible for constraint enforcement and real-time decision execution. These heuristics are refined through a closed-loop evolutionary process, driven by harmony search, which iteratively adapts the LLM prompts based on feasibility and performance feedback from the optimization layer. Extensive experiments based on scenarios derived from both the New York and Chicago taxi datasets demonstrate the effectiveness of our approach, achieving an average improvement of 16% compared to state-of-the-art baselines. 

---
# LLM-Friendly Knowledge Representation for Customer Support 

**Authors**: Hanchen Su, Wei Luo, Wei Han, Yu Elaine Liu, Yufeng Wayne Zhang, Cen Mia Zhao, Ying Joy Zhang, Yashar Mehdad  

**Link**: [PDF](https://arxiv.org/pdf/2510.10331)  

**Abstract**: We propose a practical approach by integrating Large Language Models (LLMs) with a framework designed to navigate the complexities of Airbnb customer support operations. In this paper, our methodology employs a novel reformatting technique, the Intent, Context, and Action (ICA) format, which transforms policies and workflows into a structure more comprehensible to LLMs. Additionally, we develop a synthetic data generation strategy to create training data with minimal human intervention, enabling cost-effective fine-tuning of our model. Our internal experiments (not applied to Airbnb products) demonstrate that our approach of restructuring workflows and fine-tuning LLMs with synthetic data significantly enhances their performance, setting a new benchmark for their application in customer support. Our solution is not only cost-effective but also improves customer support, as evidenced by both accuracy and manual processing time evaluation metrics. 

---
# Adaptive Dual Reasoner: Large Reasoning Models Can Think Efficiently by Hybrid Reasoning 

**Authors**: Yujian Zhang, Keyu Chen, Zhifeng Shen, Ruizhi Qiao, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.10207)  

**Abstract**: Although Long Reasoning Models (LRMs) have achieved superior performance on various reasoning scenarios, they often suffer from increased computational costs and inference latency caused by overthinking. To address these limitations, we propose Adaptive Dual Reasoner, which supports two reasoning modes: fast thinking and slow thinking. ADR dynamically alternates between these modes based on the contextual complexity during reasoning. ADR is trained in two stages: (1) A cold-start stage using supervised fine-tuning (SFT) to equip the model with the ability to integrate both fast and slow reasoning modes, in which we construct a hybrid reasoning dataset through a dedicated pipeline to provide large-scale supervision. (2) A reinforcement learning stage for optimizing reasoning effort, where we introduce Entropy-guided Hybrid Policy Optimization EHPO, an RL training framework employing an entropy-guided dynamic rollout strategy for branching at high-entropy units and a difficulty-aware penalty to balance fast and slow reasoning. Across challenging mathematical reasoning benchmarks, ADR achieves an effective balance between reasoning performance and efficiency among state-of-the-art approaches. Specifically, ADR yields a performance gain of up to 6.1%, while reducing the reasoning output length by 49.5% to 59.3%. 

---
# Concise Reasoning in the Lens of Lagrangian Optimization 

**Authors**: Chengqian Gao, Haonan Li, Taylor W. Killian, Jianshu She, Renxi Wang, Liqun Ma, Zhoujun Cheng, Shibo Hao, Zhiqiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10168)  

**Abstract**: Concise reasoning in large language models seeks to generate only essential intermediate steps needed to arrive at a final answer, thereby alleviating issues of overthinking. Most proposed approaches hinge on carefully hand-crafted heuristics, struggling to balance concision with performance, often failing to adapt across domains and model scales. In this work, we address these challenges by introducing a principled and pragmatic strategy, performance-aware length updating (PALU). As a principled algorithm, PALU formulates concise reasoning as a constrained optimization problem, minimizing response length subject to a performance constraint, and then applies Lagrangian optimization to convert it into a tractable unconstrained problem. As a pragmatic solution, PALU streamlines complicated update rules through three approximations: (i) estimating performance with off-policy rollouts, (ii) truncating the Lagrange multiplier to two extremes, and (iii) replacing gradient-based updates with quantile-driven length adjustments. PALU reduces output length by 65% while improving accuracy by 15% when applied to DeepSeek-Distill-Qwen-1.5B, averaged over five benchmarks, outperforming a range of alternative methods. Furthermore, PALU is demonstrated to adapt across both domain (logic, STEM and math) and model scale (1.5B, 7B, 14B) entrenching the algorithm as a practical and effective concise reasoning approach. 

---
# The Achilles' Heel of LLMs: How Altering a Handful of Neurons Can Cripple Language Abilities 

**Authors**: Zixuan Qin, Kunlin Lyu, Qingchen Yu, Yifan Sun, Zhaoxin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2510.10238)  

**Abstract**: Large Language Models (LLMs) have become foundational tools in natural language processing, powering a wide range of applications and research. Many studies have shown that LLMs share significant similarities with the human brain. Recent neuroscience research has found that a small subset of biological neurons in the human brain are crucial for core cognitive functions, which raises a fundamental question: do LLMs also contain a small subset of critical neurons? In this paper, we investigate this question by proposing a Perturbation-based Causal Identification of Critical Neurons method to systematically locate such critical neurons in LLMs. Our findings reveal three key insights: (1) LLMs contain ultra-sparse critical neuron sets. Disrupting these critical neurons can cause a 72B-parameter model with over 1.1 billion neurons to completely collapse, with perplexity increasing by up to 20 orders of magnitude; (2) These critical neurons are not uniformly distributed, but tend to concentrate in the outer layers, particularly within the MLP down\_proj components; (3) Performance degradation exhibits sharp phase transitions, rather than a gradual decline, when these critical neurons are disrupted. Through comprehensive experiments across diverse model architectures and scales, we provide deeper analysis of these phenomena and their implications for LLM robustness and interpretability. These findings can offer guidance for developing more robust model architectures and improving deployment security in safety-critical applications. 

---
# MedCoAct: Confidence-Aware Multi-Agent Collaboration for Complete Clinical Decision 

**Authors**: Hongjie Zheng, Zesheng Shi, Ping Yi  

**Link**: [PDF](https://arxiv.org/pdf/2510.10461)  

**Abstract**: Autonomous agents utilizing Large Language Models (LLMs) have demonstrated remarkable capabilities in isolated medical tasks like diagnosis and image analysis, but struggle with integrated clinical workflows that connect diagnostic reasoning and medication decisions. We identify a core limitation: existing medical AI systems process tasks in isolation without the cross-validation and knowledge integration found in clinical teams, reducing their effectiveness in real-world healthcare scenarios. To transform the isolation paradigm into a collaborative approach, we propose MedCoAct, a confidence-aware multi-agent framework that simulates clinical collaboration by integrating specialized doctor and pharmacist agents, and present a benchmark, DrugCareQA, to evaluate medical AI capabilities in integrated diagnosis and treatment workflows. Our results demonstrate that MedCoAct achieves 67.58\% diagnostic accuracy and 67.58\% medication recommendation accuracy, outperforming single agent framework by 7.04\% and 7.08\% respectively. This collaborative approach generalizes well across diverse medical domains, proving especially effective for telemedicine consultations and routine clinical scenarios, while providing interpretable decision-making pathways. 

---
# PIXEL: Adaptive Steering Via Position-wise Injection with eXact Estimated Levels under Subspace Calibration 

**Authors**: Manjiang Yu, Hongji Li, Priyanka Singh, Xue Li, Di Wang, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10205)  

**Abstract**: Reliable behavior control is central to deploying large language models (LLMs) on the web. Activation steering offers a tuning-free route to align attributes (e.g., truthfulness) that ensure trustworthy generation. Prevailing approaches rely on coarse heuristics and lack a principled account of where to steer and how strongly to intervene. To this end, we propose Position-wise Injection with eXact Estimated Levels (PIXEL), a position-wise activation steering framework that, in contrast to prior work, learns a property-aligned subspace from dual views (tail-averaged and end-token) and selects intervention strength via a constrained geometric objective with a closed-form solution, thereby adapting to token-level sensitivity without global hyperparameter tuning. PIXEL further performs sample-level orthogonal residual calibration to refine the global attribute direction and employs a lightweight position-scanning routine to identify receptive injection sites. We additionally provide representation-level guarantees for the minimal-intervention rule, supporting reliable alignment. Across diverse models and evaluation paradigms, PIXEL consistently improves attribute alignment while preserving model general capabilities, offering a practical and principled method for LLMs' controllable generation. Our code is available at this https URL 

---
# Mitigating Hallucination in Multimodal Reasoning via Functional Attention Control 

**Authors**: Haolang Lu, Bolun Chu, WeiYe Fu, Guoshun Nan, Junning Liu, Minghui Pan, Qiankun Li, Yi Yu, Hua Wang, Kun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10285)  

**Abstract**: Multimodal large reasoning models (MLRMs) are rapidly advancing vision-language reasoning and are emerging as a foundation for cross-modal intelligence. Hallucination remains a persistent failure mode, manifesting itself as erroneous reasoning chains and misinterpretation of visual content. In this study, we observe that attention heads exhibit a staged division: shallow heads predominantly serve perception, while deeper heads shift toward symbolic reasoning, revealing two major causes of hallucination, namely perceptual bias and reasoning drift. To address these issues, we propose a lightweight and interpretable two-step plugin, Functional Head Identification and Class-conditioned Rescaling, which locates perception- and reasoning-oriented heads and regulates their contributions without retraining. Evaluations on three real-world MLRMs (Kimi-VL, Ocean-R1, R1-Onevision), six benchmarks across three domains, and four baselines show that our plugin achieves an average improvement of 5% and up to 15%, with only <1% additional computation and 9% of baseline latency. Our approach is completely model-agnostic and significantly enhances both the reliability and interpretability of the off-the-shelf MLRMs, thereby enabling their safe deployment in high-stakes applications. Our code is available at this https URL. 

---
# Don't Just Fine-tune the Agent, Tune the Environment 

**Authors**: Siyuan Lu, Zechuan Wang, Hongxuan Zhang, Qintong Wu, Leilei Gan, Chenyi Zhuang, Jinjie Gu, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.10197)  

**Abstract**: Large Language Model (LLM) agents show great promise for complex, multi-turn tool-use tasks, but their development is often hampered by the extreme scarcity of high-quality training data. Supervised fine-tuning (SFT) on synthetic data leads to overfitting, whereas standard reinforcement learning (RL) struggles with a critical cold-start problem and training instability. To address these challenges, we introduce $\textbf{Environment Tuning}$, a novel training paradigm that enables agents to learn complex behaviors directly from problem instances without relying on pre-collected expert trajectories. $\textbf{Environment Tuning}$ orchestrates this learning process through a structured curriculum, actionable environment augmentation that provides corrective feedback, and fine-grained progress rewards to ensure stable and efficient exploration. Using only 400 problem instances from Berkeley Function-Calling Leaderboard (BFCL) benchmark, our method not only achieves competitive in-distribution performance against strong baselines but also demonstrates superior out-of-distribution generalization, overcoming the performance collapse common to SFT-based approaches. Our work presents a paradigm shift from supervised fine-tuning on static trajectories to dynamic, environment-based exploration, paving the way for training more robust and data-efficient agents. 

---
# SAFER: Risk-Constrained Sample-then-Filter in Large Language Models 

**Authors**: Qingni Wang, Yue Fan, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10193)  

**Abstract**: As large language models (LLMs) are increasingly deployed in risk-sensitive applications such as real-world open-ended question answering (QA), ensuring the trustworthiness of their outputs has become critical. Existing selective conformal prediction (SCP) methods provide statistical guarantees by constructing prediction sets with a constrained miscoverage rate for correct answers. However, prior works unrealistically assume that admissible answers for all instances can be obtained via finite sampling, even for open-ended QA scenarios that lack a fixed and finite solution space. To address this, we introduce a two-stage risk control framework comprising abstention-aware sampling and conformalized filtering (SAFER). Firstly, on a held-out calibration set, SAFER calibrates a sampling budget within the maximum sampling cap, using the Clopper-Pearson exact method at a user-desired risk level (i.e., the maximum allowable miscoverage rate of the sampling sets). If the risk level cannot be satisfied within the cap, we abstain; otherwise, the calibrated sampling budget becomes the minimum requirements at test time. Then, we employ calibration instances where correct answers are attainable under the calibrated budget and apply the conformal risk control method to determine a statistically valid uncertainty threshold, which filters unreliable distractors from the candidate set for each test data point. In this stage, SAFER introduces an additional risk level to guide the calculation of the threshold, thereby controlling the risk of correct answers being excluded. Furthermore, we show that SAFER is compatible with various task-specific admission criteria and calibration-test split ratios, highlighting its robustness and high data efficiency. 

---
# Failure-Driven Workflow Refinement 

**Authors**: Jusheng Zhang, Kaitong Cai, Qinglin Zeng, Ningyuan Liu, Stephen Fan, Ziliang Chen, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10035)  

**Abstract**: Optimizing LLM-based workflows is typically formulated as a global search, where candidate workflows are evaluated based on a scalar metric. This paradigm, however, suffers from a critical flaw: information collapse. By reducing rich, multi-step execution traces to simple success/failure signals, existing methods are rendered blind to the underlying structure of failures, fundamentally preventing them from modeling the workflow's failure distribution. We reconceptualize this challenge as a distributional problem. We propose a new paradigm where the optimization goal is not to maximize a scalar score, but to directly minimize a workflow's Expected Failure Mass, i.e., the integral of its failure probability density function defined over a high-dimensional Failure Signature Space (FSS). This distributional lens allows us to move from inefficient, zero-order optimization to a principled, gradient-like descent on the failure landscape itself. We introduce CE-Graph, a framework that operationalizes this paradigm through a novel, failure-driven refinement process. CE-Graph approximates the failure distribution from a pool of counterexamples, identifies its densest regions as recurring failure modes, and applies targeted, operator-constrained graph edits via a Propose-and-Verify mechanism to greedily reduce the failure mass. On math, code, and QA benchmarks, our CE-Graph achieves higher robustness at a significantly lower cost than strong baselines. This suggests that a system's reliability emerges not from avoiding failures, but from systematically learning and reshaping the geometric structure of its failure distributions. 

---
# RIPRAG: Hack a Black-box Retrieval-Augmented Generation Question-Answering System with Reinforcement Learning 

**Authors**: Meng Xi, Sihan Lv, Yechen Jin, Guanjie Cheng, Naibo Wang, Ying Li, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2510.10008)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become a core technology for tasks such as question-answering (QA) and content generation. However, by injecting poisoned documents into the database of RAG systems, attackers can manipulate LLMs to generate text that aligns with their intended preferences. Existing research has primarily focused on white-box attacks against simplified RAG architectures. In this paper, we investigate a more complex and realistic scenario: the attacker lacks knowledge of the RAG system's internal composition and implementation details, and the RAG system comprises components beyond a mere retriever. Specifically, we propose the RIPRAG attack framework, an end-to-end attack pipeline that treats the target RAG system as a black box, where the only information accessible to the attacker is whether the poisoning succeeds. Our method leverages Reinforcement Learning (RL) to optimize the generation model for poisoned documents, ensuring that the generated poisoned document aligns with the target RAG system's preferences. Experimental results demonstrate that this method can effectively execute poisoning attacks against most complex RAG systems, achieving an attack success rate (ASR) improvement of up to 0.72 compared to baseline methods. This highlights prevalent deficiencies in current defensive methods and provides critical insights for LLM security research. 

---
# Agentic Troubleshooting Guide Automation for Incident Management 

**Authors**: Jiayi Mao, Liqun Li, Yanjie Gao, Zegang Peng, Shilin He, Chaoyun Zhang, Si Qin, Samia Khalid, Qingwei Lin, Saravan Rajmohan, Sitaram Lanka, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10074)  

**Abstract**: Effective incident management in large-scale IT systems relies on troubleshooting guides (TSGs), but their manual execution is slow and error-prone. While recent advances in LLMs offer promise for automating incident management tasks, existing LLM-based solutions lack specialized support for several key challenges, including managing TSG quality issues, interpreting complex control flow, handling data-intensive queries, and exploiting execution parallelism. We first conducted an empirical study on 92 real-world TSGs, and, guided by our findings, we present StepFly, a novel end-to-end agentic framework for troubleshooting guide automation. Our approach features a three-stage workflow: the first stage provides a comprehensive guide together with a tool, TSG Mentor, to assist SREs in improving TSG quality; the second stage performs offline preprocessing using LLMs to extract structured execution DAGs from unstructured TSGs and to create dedicated Query Preparation Plugins (QPPs); and the third stage executes online using a DAG-guided scheduler-executor framework with a memory system to guarantee correct workflow and support parallel execution of independent steps. Our empirical evaluation on a collection of real-world TSGs and incidents demonstrates that StepFly achieves a ~94% success rate on GPT-4.1, outperforming baselines with less time and token consumption. Furthermore, it achieves a remarkable execution time reduction of 32.9% to 70.4% for parallelizable TSGs. 

---
# Deliberative Dynamics and Value Alignment in LLM Debates 

**Authors**: Pratik S. Sachdeva, Tom van Nuenen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10002)  

**Abstract**: As large language models (LLMs) are increasingly deployed in sensitive everyday contexts - offering personal advice, mental health support, and moral guidance - understanding their elicited values in navigating complex moral reasoning is essential. Most evaluations study this sociotechnical alignment through single-turn prompts, but it is unclear if these findings extend to multi-turn settings where values emerge through dialogue, revision, and consensus. We address this gap using LLM debate to examine deliberative dynamics and value alignment in multi-turn settings by prompting subsets of three models (GPT-4.1, Claude 3.7 Sonnet, and Gemini 2.0 Flash) to collectively assign blame in 1,000 everyday dilemmas from Reddit's "Am I the Asshole" community. We use both synchronous (parallel responses) and round-robin (sequential responses) formats to test order effects and verdict revision. Our findings show striking behavioral differences. In the synchronous setting, GPT showed strong inertia (0.6-3.1% revision rates) while Claude and Gemini were far more flexible (28-41%). Value patterns also diverged: GPT emphasized personal autonomy and direct communication, while Claude and Gemini prioritized empathetic dialogue. Certain values proved especially effective at driving verdict changes. We further find that deliberation format had a strong impact on model behavior: GPT and Gemini stood out as highly conforming relative to Claude, with their verdict behavior strongly shaped by order effects. These results show how deliberation format and model-specific behaviors shape moral reasoning in multi-turn interactions, underscoring that sociotechnical alignment depends on how systems structure dialogue as much as on their outputs. 

---
# How can we assess human-agent interactions? Case studies in software agent design 

**Authors**: Valerie Chen, Rohit Malhotra, Xingyao Wang, Juan Michelini, Xuhui Zhou, Aditya Bharat Soni, Hoang H. Tran, Calvin Smith, Ameet Talwalkar, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2510.09801)  

**Abstract**: LLM-powered agents are both a promising new technology and a source of complexity, where choices about models, tools, and prompting can affect their usefulness. While numerous benchmarks measure agent accuracy across domains, they mostly assume full automation, failing to represent the collaborative nature of real-world use cases. In this paper, we make two major steps towards the rigorous assessment of human-agent interactions. First, we propose PULSE, a framework for more efficient human-centric evaluation of agent designs, which comprises collecting user feedback, training an ML model to predict user satisfaction, and computing results by combining human satisfaction ratings with model-generated pseudo-labels. Second, we deploy the framework on a large-scale web platform built around the open-source software agent OpenHands, collecting in-the-wild usage data across over 15k users. We conduct case studies around how three agent design decisions -- choice of LLM backbone, planning strategy, and memory mechanisms -- impact developer satisfaction rates, yielding practical insights for software agent design. We also show how our framework can lead to more robust conclusions about agent design, reducing confidence intervals by 40\% compared to a standard A/B test. Finally, we find substantial discrepancies between in-the-wild results and benchmark performance (e.g., the anti-correlation between results comparing claude-sonnet-4 and gpt-5), underscoring the limitations of benchmark-driven evaluation. Our findings provide guidance for evaluations of LLM agents with humans and identify opportunities for better agent designs. 

---
# Follow My Lead: Logical Fallacy Classification with Knowledge-Augmented LLMs 

**Authors**: Olivia Peiyu Wang, Tashvi Bansal, Ryan Bai, Emily M. Chui, Leilani H. Gilpin  

**Link**: [PDF](https://arxiv.org/pdf/2510.09970)  

**Abstract**: Large Language Models (LLMs) suffer from critical reasoning gaps, including a tendency to hallucinate and poor accuracy in classifying logical fallacies. This limitation stems from their default System 1 processing, which is fast and intuitive, whereas reliable reasoning requires the deliberate, effortful System 2 approach (Kahneman, 2011; Li et al., 2025). Since full System 2 training is often prohibitively expensive, we explore a low-cost, instruction-based intervention to bridge this gap. Our methodology introduces a novel stepwise instruction dataset that decomposes fallacy classification into a series of atomic procedural steps (simple binary questions). We further augment this with a final verification step where models consult a relational knowledge graph of related fallacies. This procedural, rule-based intervention yields a significant improvement in LLM logical fallacy classification. Crucially, the approach also provides enhanced transparency into the LLMs' decision-making, highlighting a practical pathway for Neuro-symbolic architectures to address LLM reasoning deficits. 

---
# SwarmSys: Decentralized Swarm-Inspired Agents for Scalable and Adaptive Reasoning 

**Authors**: Ruohao Li, Hongjun Liu, Leyi Zhao, Zisu Li, Jiawei Li, Jiajun Jiang, Linning Xu, Chen Zhao, Mingming Fan, Chen Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10047)  

**Abstract**: Large language model (LLM) agents have shown remarkable reasoning abilities. However, existing multi-agent frameworks often rely on fixed roles or centralized control, limiting scalability and adaptability in long-horizon reasoning. We introduce SwarmSys, a closed-loop framework for distributed multi-agent reasoning inspired by swarm intelligence. Coordination in SwarmSys emerges through iterative interactions among three specialized roles, Explorers, Workers, and Validators, that continuously cycle through exploration, exploitation, and validation. To enable scalable and adaptive collaboration, we integrate adaptive agent and event profiles, embedding-based probabilistic matching, and a pheromone-inspired reinforcement mechanism, supporting dynamic task allocation and self-organizing convergence without global supervision. Across symbolic reasoning, research synthesis, and scientific programming tasks, SwarmSys consistently outperforms baselines, improving both accuracy and reasoning stability. These findings highlight swarm-inspired coordination as a promising paradigm for scalable, robust, and adaptive multi-agent reasoning, suggesting that coordination scaling may rival model scaling in advancing LLM intelligence. 

---
# PACEbench: A Framework for Evaluating Practical AI Cyber-Exploitation Capabilities 

**Authors**: Zicheng Liu, Lige Huang, Jie Zhang, Dongrui Liu, Yuan Tian, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2510.11688)  

**Abstract**: The increasing autonomy of Large Language Models (LLMs) necessitates a rigorous evaluation of their potential to aid in cyber offense. Existing benchmarks often lack real-world complexity and are thus unable to accurately assess LLMs' cybersecurity capabilities. To address this gap, we introduce PACEbench, a practical AI cyber-exploitation benchmark built on the principles of realistic vulnerability difficulty, environmental complexity, and cyber defenses. Specifically, PACEbench comprises four scenarios spanning single, blended, chained, and defense vulnerability exploitations. To handle these complex challenges, we propose PACEagent, a novel agent that emulates human penetration testers by supporting multi-phase reconnaissance, analysis, and exploitation. Extensive experiments with seven frontier LLMs demonstrate that current models struggle with complex cyber scenarios, and none can bypass defenses. These findings suggest that current models do not yet pose a generalized cyber offense threat. Nonetheless, our work provides a robust benchmark to guide the trustworthy development of future models. 

---
# Autonomous Agents for Scientific Discovery: Orchestrating Scientists, Language, Code, and Physics 

**Authors**: Lianhao Zhou, Hongyi Ling, Cong Fu, Yepeng Huang, Michael Sun, Wendi Yu, Xiaoxuan Wang, Xiner Li, Xingyu Su, Junkai Zhang, Xiusi Chen, Chenxing Liang, Xiaofeng Qian, Heng Ji, Wei Wang, Marinka Zitnik, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.09901)  

**Abstract**: Computing has long served as a cornerstone of scientific discovery. Recently, a paradigm shift has emerged with the rise of large language models (LLMs), introducing autonomous systems, referred to as agents, that accelerate discovery across varying levels of autonomy. These language agents provide a flexible and versatile framework that orchestrates interactions with human scientists, natural language, computer language and code, and physics. This paper presents our view and vision of LLM-based scientific agents and their growing role in transforming the scientific discovery lifecycle, from hypothesis discovery, experimental design and execution, to result analysis and refinement. We critically examine current methodologies, emphasizing key innovations, practical achievements, and outstanding limitations. Additionally, we identify open research challenges and outline promising directions for building more robust, generalizable, and adaptive scientific agents. Our analysis highlights the transformative potential of autonomous agents to accelerate scientific discovery across diverse domains. 

---
# CodePlot-CoT: Mathematical Visual Reasoning by Thinking with Code-Driven Images 

**Authors**: Chengqi Duan, Kaiyue Sun, Rongyao Fang, Manyuan Zhang, Yan Feng, Ying Luo, Yufang Liu, Ke Wang, Peng Pei, Xunliang Cai, Hongsheng Li, Yi Ma, Xihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11718)  

**Abstract**: Recent advances in Large Language Models (LLMs) and Vision Language Models (VLMs) have shown significant progress in mathematical reasoning, yet they still face a critical bottleneck with problems requiring visual assistance, such as drawing auxiliary lines or plotting functions to solve the problems. Most LLMs and VLMs are constrained to text-only reasoning chains, while multimodal unified models that can generate interleaved text and images lack the necessary precision and controllability for such tasks. To address this, we propose CodePlot-CoT, a code-driven Chain-of-Thought paradigm for "thinking with images" in mathematics. Our approach leverages the VLM to generate text reasoning as well as executable plotting code, which is then rendered into images as "visual thought", to solve mathematical problems. To achieve this, we first construct Math-VR, the first large-scale, bilingual dataset and benchmark for Mathematics problems with Visual Reasoning, comprising 178K samples. Second, to create high-quality training data, we develop a state-of-the-art image-to-code converter specialized for parsing complex mathematical figures into codes. Finally, using these training data, we train the CodePlot-CoT model for solving mathematical problems. Experimental results show that our model achieves up to 21% increase over base model on our new benchmark, fully validating the efficacy of our proposed code-driven reasoning paradigm. Our work opens a new direction for multimodal mathematical reasoning and provides the community with the first large-scale dataset, comprehensive benchmark, and strong approach for such problems. To facilitate future research, we make our datasets, code, and pretrained models publicly available at this https URL. 

---
# Cracking CodeWhisperer: Analyzing Developers' Interactions and Patterns During Programming Tasks 

**Authors**: Jeena Javahar, Tanya Budhrani, Manaal Basha, Cleidson R. B. de Souza, Ivan Beschastnikh, Gema Rodriguez-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2510.11516)  

**Abstract**: The use of AI code-generation tools is becoming increasingly common, making it important to understand how software developers are adopting these tools. In this study, we investigate how developers engage with Amazon's CodeWhisperer, an LLM-based code-generation tool. We conducted two user studies with two groups of 10 participants each, interacting with CodeWhisperer - the first to understand which interactions were critical to capture and the second to collect low-level interaction data using a custom telemetry plugin. Our mixed-methods analysis identified four behavioral patterns: 1) incremental code refinement, 2) explicit instruction using natural language comments, 3) baseline structuring with model suggestions, and 4) integrative use with external sources. We provide a comprehensive analysis of these patterns . 

---
# Medical Interpretability and Knowledge Maps of Large Language Models 

**Authors**: Razvan Marinescu, Victoria-Elisabeth Gruber, Diego Fajardo  

**Link**: [PDF](https://arxiv.org/pdf/2510.11390)  

**Abstract**: We present a systematic study of medical-domain interpretability in Large Language Models (LLMs). We study how the LLMs both represent and process medical knowledge through four different interpretability techniques: (1) UMAP projections of intermediate activations, (2) gradient-based saliency with respect to the model weights, (3) layer lesioning/removal and (4) activation patching. We present knowledge maps of five LLMs which show, at a coarse-resolution, where knowledge about patient's ages, medical symptoms, diseases and drugs is stored in the models. In particular for Llama3.3-70B, we find that most medical knowledge is processed in the first half of the model's layers. In addition, we find several interesting phenomena: (i) age is often encoded in a non-linear and sometimes discontinuous manner at intermediate layers in the models, (ii) the disease progression representation is non-monotonic and circular at certain layers of the model, (iii) in Llama3.3-70B, drugs cluster better by medical specialty rather than mechanism of action, especially for Llama3.3-70B and (iv) Gemma3-27B and MedGemma-27B have activations that collapse at intermediate layers but recover by the final layers. These results can guide future research on fine-tuning, un-learning or de-biasing LLMs for medical tasks by suggesting at which layers in the model these techniques should be applied. 

---
# AndesVL Technical Report: An Efficient Mobile-side Multimodal Large Language Model 

**Authors**: Zhiwei Jin, Xiaohui Song, Nan Wang, Yafei Liu, Chao Li, Xin Li, Ruichen Wang, Zhihao Li, Qi Qi, Long Cheng, Dongze Hao, Quanlong Zheng, Yanhao Zhang, Haobo Ji, Jian Ma, Zhitong Zheng, Zhenyi Lin, Haolin Deng, Xin Zou, Xiaojie Yin, Ruilin Wang, Liankai Cai, Haijing Liu, Yuqing Qiu, Ke Chen, Zixian Li, Chi Xie, Huafei Li, Chenxing Li, Chuangchuang Wang, Kai Tang, Zhiguang Zhu, Kai Tang, Wenmei Gao, Rui Wang, Jun Wu, Chao Liu, Qin Xie, Chen Chen, Haonan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11496)  

**Abstract**: In recent years, while cloud-based MLLMs such as QwenVL, InternVL, GPT-4o, Gemini, and Claude Sonnet have demonstrated outstanding performance with enormous model sizes reaching hundreds of billions of parameters, they significantly surpass the limitations in memory, power consumption, and computing capacity of edge devices such as mobile phones. This paper introduces AndesVL, a suite of mobile-side MLLMs with 0.6B to 4B parameters based on Qwen3's LLM and various visual encoders. We comprehensively outline the model architectures, training pipeline, and training data of AndesVL, which achieves first-tier performance across a wide range of open-source benchmarks, including fields such as text-rich image understanding, reasoning and math, multi-image comprehension, general VQA, hallucination mitigation, multilingual understanding, and GUI-related tasks when compared with state-of-the-art models of a similar scale. Furthermore, we introduce a 1+N LoR 

---
# Diffusion-Link: Diffusion Probabilistic Model for Bridging the Audio-Text Modality Gap 

**Authors**: KiHyun Nam, Jongmin Choi, Hyeongkeun Lee, Jungwoo Heo, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2510.11330)  

**Abstract**: Contrastive audio-language pretraining yields powerful joint representations, yet a persistent audio-text modality gap limits the benefits of coupling multimodal encoders with large language models (LLMs). We present Diffusion-Link, a diffusion-based modality-bridging module that generatively maps audio embeddings into the text-embedding distribution. The module is trained at the output embedding from the frozen multimodal encoder and implemented as a lightweight network with three residual MLP blocks. To assess the effect of Diffusion-Link on multimodal encoder-LLM coupling, we evaluate on Automatic Audio Captioning (AAC); to our knowledge, this is the first application of diffusion-based modality bridging to AAC. We report two results. (1) Modality-gap analysis: on similarity and geometric criteria, Diffusion-Link reduces the modality gap the most among prior diffusion-based methods and shows a collective migration of audio embeddings toward the text distribution. (2) Downstream AAC: attaching Diffusion-Link to the same multimodal LLM baseline achieves state-of-the-art on AudioCaps in both zero-shot and fully supervised captioning without external knowledge, with relative gains up to 52.5% and 7.5%, respectively. These findings show that closing the modality gap is pivotal for effective coupling between multimodal encoders and LLMs, and diffusion-based modality bridging offers a promising direction beyond knowledge-retrieval-centric designs. Code will be released upon acceptance this https URL 

---
# Beyond touch-based HMI: Control your machines in natural language by utilizing large language models and OPC UA 

**Authors**: Bernd Hofmann, Sven Kreitlein, Joerg Franke, Patrick Bruendl  

**Link**: [PDF](https://arxiv.org/pdf/2510.11300)  

**Abstract**: This paper proposes an agent-based approach toward a more natural interface between humans and machines. Large language models equipped with tools and the communication standard OPC UA are utilized to control machines in natural language. Instead of touch interaction, which is currently the state-of-the-art medium for interaction in operations, the proposed approach enables operators to talk or text with machines. This allows commands such as 'Please decrease the temperature by 20 % in machine 1 and set the motor speed to 5000 rpm in machine 2.' The large language model receives the user input and selects one of three predefined tools that connect to an OPC UA server and either change or read the value of a node. Afterwards, the result of the tool execution is passed back to the language model, which then provides a final response to the user. The approach is universally designed and can therefore be applied to any machine that supports the OPC UA standard. The large language model is neither fine-tuned nor requires training data, only the relevant machine credentials and a parameter dictionary are included within the system prompt. The approach is evaluated on a Siemens S7-1500 programmable logic controller with four machine parameters in a case study of fifty synthetically generated commands on five different models. The results demonstrate high success rate, with proprietary GPT 5 models achieving accuracies between 96.0 % and 98.0 %, and open-weight models reaching up to 90.0 %. The proposed approach of this empirical study contributes to advancing natural interaction in industrial human-machine interfaces. 

---
# Large Language Models Are Effective Code Watermarkers 

**Authors**: Rui Xu, Jiawei Chen, Zhaoxia Yin, Cong Kong, Xinpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11251)  

**Abstract**: The widespread use of large language models (LLMs) and open-source code has raised ethical and security concerns regarding the distribution and attribution of source code, including unauthorized redistribution, license violations, and misuse of code for malicious purposes. Watermarking has emerged as a promising solution for source attribution, but existing techniques rely heavily on hand-crafted transformation rules, abstract syntax tree (AST) manipulation, or task-specific training, limiting their scalability and generality across languages. Moreover, their robustness against attacks remains limited. To address these limitations, we propose CodeMark-LLM, an LLM-driven watermarking framework that embeds watermark into source code without compromising its semantics or readability. CodeMark-LLM consists of two core components: (i) Semantically Consistent Embedding module that applies functionality-preserving transformations to encode watermark bits, and (ii) Differential Comparison Extraction module that identifies the applied transformations by comparing the original and watermarked code. Leveraging the cross-lingual generalization ability of LLM, CodeMark-LLM avoids language-specific engineering and training pipelines. Extensive experiments across diverse programming languages and attack scenarios demonstrate its robustness, effectiveness, and scalability. 

---
# A Large-Language-Model Assisted Automated Scale Bar Detection and Extraction Framework for Scanning Electron Microscopic Images 

**Authors**: Yuxuan Chen, Ruotong Yang, Zhengyang Zhang, Mehreen Ahmed, Yanming Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11260)  

**Abstract**: Microscopic characterizations, such as Scanning Electron Microscopy (SEM), are widely used in scientific research for visualizing and analyzing microstructures. Determining the scale bars is an important first step of accurate SEM analysis; however, currently, it mainly relies on manual operations, which is both time-consuming and prone to errors. To address this issue, we propose a multi-modal and automated scale bar detection and extraction framework that provides concurrent object detection, text detection and text recognition with a Large Language Model (LLM) agent. The proposed framework operates in four phases; i) Automatic Dataset Generation (Auto-DG) model to synthesize a diverse dataset of SEM images ensuring robust training and high generalizability of the model, ii) scale bar object detection, iii) information extraction using a hybrid Optical Character Recognition (OCR) system with DenseNet and Convolutional Recurrent Neural Network (CRNN) based algorithms, iv) an LLM agent to analyze and verify accuracy of the results. The proposed model demonstrates a strong performance in object detection and accurate localization with a precision of 100%, recall of 95.8%, and a mean Average Precision (mAP) of 99.2% at IoU=0.5 and 69.1% at IoU=0.5:0.95. The hybrid OCR system achieved 89% precision, 65% recall, and a 75% F1 score on the Auto-DG dataset, significantly outperforming several mainstream standalone engines, highlighting its reliability for scientific image analysis. The LLM is introduced as a reasoning engine as well as an intelligent assistant that suggests follow-up steps and verifies the results. This automated method powered by an LLM agent significantly enhances the efficiency and accuracy of scale bar detection and extraction in SEM images, providing a valuable tool for microscopic analysis and advancing the field of scientific imaging. 

---
# Protein as a Second Language for LLMs 

**Authors**: Xinhui Chen, Zuchao Li, Mengqi Gao, Yufeng Zhang, Chak Tou Leong, Haoyang Li, Jiaqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.11188)  

**Abstract**: Deciphering the function of unseen protein sequences is a fundamental challenge with broad scientific impact, yet most existing methods depend on task-specific adapters or large-scale supervised fine-tuning. We introduce the "Protein-as-Second-Language" framework, which reformulates amino-acid sequences as sentences in a novel symbolic language that large language models can interpret through contextual exemplars. Our approach adaptively constructs sequence-question-answer triples that reveal functional cues in a zero-shot setting, without any further training. To support this process, we curate a bilingual corpus of 79,926 protein-QA instances spanning attribute prediction, descriptive understanding, and extended reasoning. Empirically, our method delivers consistent gains across diverse open-source LLMs and GPT-4, achieving up to 17.2% ROUGE-L improvement (average +7%) and even surpassing fine-tuned protein-specific language models. These results highlight that generic LLMs, when guided with protein-as-language cues, can outperform domain-specialized models, offering a scalable pathway for protein understanding in foundation models. 

---
# Domain-Specific Data Generation Framework for RAG Adaptation 

**Authors**: Chris Xing Tian, Weihao Xie, Zhen Chen, Zhengyuan Yi, Hui Liu, Haoliang Li, Shiqi Wang, Siwei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.11217)  

**Abstract**: Retrieval-Augmented Generation (RAG) combines the language understanding and reasoning power of large language models (LLMs) with external retrieval to enable domain-grounded responses. Effectively adapting RAG systems to domain-specific settings requires specialized, context-rich training data beyond general-purpose question-answering. Here, we propose RAGen, a scalable and modular framework for generating domain-grounded question-answer-context (QAC) triples tailored to diverse RAG adaptation approaches. RAGen produces these QAC triples by identifying key concepts in documents, generating diverse questions guided by Bloom's Taxonomy-inspired principles, and pairing them with precise answers extracted from relevant contexts. RAGen supports multiple RAG adaptation strategies, including the optimization of key components such as the LLM, retriever, and embedding model, etc. Its modular pipeline features semantic chunking, hierarchical concept extraction, and multi-chunk retrieval, along with the introduction of curated distractor contexts to promote robust reasoning. Designed for scalability, RAGen efficiently handles large and evolving document corpora without redundant processing, making it especially suitable for dynamic evolving domains such as scientific research and enterprise knowledge bases. 

---
# DITTO: A Spoofing Attack Framework on Watermarked LLMs via Knowledge Distillation 

**Authors**: Hyeseon Ahn, Shinwoo Park, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.10987)  

**Abstract**: The promise of LLM watermarking rests on a core assumption that a specific watermark proves authorship by a specific model. We demonstrate that this assumption is dangerously flawed. We introduce the threat of watermark spoofing, a sophisticated attack that allows a malicious model to generate text containing the authentic-looking watermark of a trusted, victim model. This enables the seamless misattribution of harmful content, such as disinformation, to reputable sources. The key to our attack is repurposing watermark radioactivity, the unintended inheritance of data patterns during fine-tuning, from a discoverable trait into an attack vector. By distilling knowledge from a watermarked teacher model, our framework allows an attacker to steal and replicate the watermarking signal of the victim model. This work reveals a critical security gap in text authorship verification and calls for a paradigm shift towards technologies capable of distinguishing authentic watermarks from expertly imitated ones. Our code is available at this https URL. 

---
# Project-Level C-to-Rust Translation via Synergistic Integration of Knowledge Graphs and Large Language Models 

**Authors**: Zhiqiang Yuan, Wenjun Mao, Zhuo Chen, Xiyue Shang, Chong Wang, Yiling Lou, Xin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.10956)  

**Abstract**: Translating C code into safe Rust is an effective way to ensure its memory safety. Compared to rule-based translation which produces Rust code that remains largely unsafe, LLM-based methods can generate more idiomatic and safer Rust code because LLMs have been trained on vast amount of human-written idiomatic code. Although promising, existing LLM-based methods still struggle with project-level C-to-Rust translation. They typically partition a C project into smaller units (\eg{} functions) based on call graphs and translate them bottom-up to resolve program dependencies. However, this bottom-up, unit-by-unit paradigm often fails to translate pointers due to the lack of a global perspective on their usage. To address this problem, we propose a novel C-Rust Pointer Knowledge Graph (KG) that enriches a code-dependency graph with two types of pointer semantics: (i) pointer-usage information which record global behaviors such as points-to flows and map lower-level struct usage to higher-level units; and (ii) Rust-oriented annotations which encode ownership, mutability, nullability, and lifetime. Synthesizing the \kg{} with LLMs, we further propose \ourtool{}, which implements a project-level C-to-Rust translation technique. In \ourtool{}, the \kg{} provides LLMs with comprehensive pointer semantics from a global perspective, thus guiding LLMs towards generating safe and idiomatic Rust code from a given C project. Our experiments show that \ourtool{} reduces unsafe usages in translated Rust by 99.9\% compared to both rule-based translation and traditional LLM-based rewriting, while achieving an average 29.3\% higher functional correctness than those fuzzing-enhanced LLM methods. 

---
# Generative AI and the Transformation of Software Development Practices 

**Authors**: Vivek Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2510.10819)  

**Abstract**: Generative AI is reshaping how software is designed, written, and maintained. Advances in large language models (LLMs) are enabling new development styles - from chat-oriented programming and 'vibe coding' to agentic programming - that can accelerate productivity and broaden access. This paper examines how AI-assisted techniques are changing software engineering practice, and the related issues of trust, accountability, and shifting skills. We survey iterative chat-based development, multi-agent systems, dynamic prompt orchestration, and integration via the Model Context Protocol (MCP). Using case studies and industry data, we outline both the opportunities (faster cycles, democratized coding) and the challenges (model reliability and cost) of applying generative AI to coding. We describe new roles, skills, and best practices for using AI in a responsible and effective way. 

---
# ECO: Enhanced Code Optimization via Performance-Aware Prompting for Code-LLMs 

**Authors**: Su-Hyeon Kim, Joonghyuk Hahn, Sooyoung Cha, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.10517)  

**Abstract**: Code runtime optimization-the task of rewriting a given code to a faster one-remains challenging, as it requires reasoning about performance trade-offs involving algorithmic and structural choices. Recent approaches employ code-LLMs with slow-fast code pairs provided as optimization guidance, but such pair-based methods obscure the causal factors of performance gains and often lead to superficial pattern imitation rather than genuine performance reasoning. We introduce ECO, a performance-aware prompting framework for code optimization. ECO first distills runtime optimization instructions (ROIs) from reference slow-fast code pairs; Each ROI describes root causes of inefficiency and the rationales that drive performance improvements. For a given input code, ECO in parallel employs (i) a symbolic advisor to produce a bottleneck diagnosis tailored to the code, and (ii) an ROI retriever to return related ROIs. These two outputs are then composed into a performance-aware prompt, providing actionable guidance for code-LLMs. ECO's prompts are model-agnostic, require no fine-tuning, and can be easily prepended to any code-LLM prompt. Our empirical studies highlight that ECO prompting significantly improves code-LLMs' ability to generate efficient code, achieving speedups of up to 7.81x while minimizing correctness loss. 

---
# Align2Act: Instruction-Tuned Models for Human-Aligned Autonomous Driving 

**Authors**: Kanishkha Jaisankar, Sunidhi Tandel  

**Link**: [PDF](https://arxiv.org/pdf/2510.10503)  

**Abstract**: Motion planning in complex scenarios is a core challenge in autonomous driving. Conventional methods apply predefined rules or learn from driving data to generate trajectories, while recent approaches leverage large language models (LLMs) for decision-making. However, it remains unclear whether LLMs truly capture human driving logic. We propose Align2Act, a motion planning framework that transforms instruction-tuned LLMs into interpretable planners aligned with human behavior. We derive structured driving instructions based on human reasoning patterns (e.g., anticipate hazards, yield at intersections) and traffic rules (e.g., stop at red lights, maintain lane boundaries). Our Align2ActChain module guides step-by-step reasoning to produce both an interpretable rationale and a safe trajectory. By fine-tuning LLaMA-2-7B with LoRA on one million scenarios from the nuPlan dataset, our method achieves an open-loop score of 85.17 and closed-loop scores of 70.31 (non-reactive) and 66.96 (reactive) on Test14-random. Unlike prior work focused on synthetic or open-loop settings, we demonstrate improved planning quality and human-likeness on the real-world nuPlan closed-loop benchmark. Ablation studies confirm that structured reasoning significantly improves performance over baseline LLM planners. 

---
# Taming a Retrieval Framework to Read Images in Humanlike Manner for Augmenting Generation of MLLMs 

**Authors**: Suyang Xi, Chenxi Yang, Hong Ding, Yiqing Ni, Catherine C. Liu, Yunhao Liu, Chengqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10426)  

**Abstract**: Multimodal large language models (MLLMs) often fail in fine-grained visual question answering, producing hallucinations about object identities, positions, and relations because textual queries are not explicitly anchored to visual referents. Retrieval-augmented generation (RAG) alleviates some errors, but it fails to align with human-like processing at both the retrieval and augmentation levels. Specifically, it focuses only on global-level image information but lacks local detail and limits reasoning about fine-grained interactions. To overcome this limitation, we present Human-Like Retrieval-Augmented Generation (HuLiRAG), a framework that stages multimodal reasoning as a ``what--where--reweight'' cascade. Queries are first anchored to candidate referents via open-vocabulary detection (what), then spatially resolved with SAM-derived masks to recover fine-grained precision (where), and adaptively prioritized through the trade-off between local and global alignment (reweight). Mask-guided fine-tuning further injects spatial evidence into the generation process, transforming grounding from a passive bias into an explicit constraint on answer formulation. Extensive experiments demonstrate that this human-like cascade improves grounding fidelity and factual consistency while reducing hallucinations, advancing multimodal question answering toward trustworthy reasoning. 

---
# RobotFleet: An Open-Source Framework for Centralized Multi-Robot Task Planning 

**Authors**: Rohan Gupta, Trevor Asbery, Zain Merchant, Abrar Anwar, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2510.10379)  

**Abstract**: Coordinating heterogeneous robot fleets to achieve multiple goals is challenging in multi-robot systems. We introduce an open-source and extensible framework for centralized multi-robot task planning and scheduling that leverages LLMs to enable fleets of heterogeneous robots to accomplish multiple tasks. RobotFleet provides abstractions for planning, scheduling, and execution across robots deployed as containerized services to simplify fleet scaling and management. The framework maintains a shared declarative world state and two-way communication for task execution and replanning. By modularizing each layer of the autonomy stack and using LLMs for open-world reasoning, RobotFleet lowers the barrier to building scalable multi-robot systems. The code can be found here: this https URL. 

---
# Bridging Semantics & Structure for Software Vulnerability Detection using Hybrid Network Models 

**Authors**: Jugal Gajjar, Kaustik Ranaware, Kamalasankari Subramaniakuppusamy  

**Link**: [PDF](https://arxiv.org/pdf/2510.10321)  

**Abstract**: Software vulnerabilities remain a persistent risk, yet static and dynamic analyses often overlook structural dependencies that shape insecure behaviors. Viewing programs as heterogeneous graphs, we capture control- and data-flow relations as complex interaction networks. Our hybrid framework combines these graph representations with light-weight (<4B) local LLMs, uniting topological features with semantic reasoning while avoiding the cost and privacy concerns of large cloud models. Evaluated on Java vulnerability detection (binary classification), our method achieves 93.57% accuracy-an 8.36% gain over Graph Attention Network-based embeddings and 17.81% over pretrained LLM baselines such as Qwen2.5 Coder 3B. Beyond accuracy, the approach extracts salient subgraphs and generates natural language explanations, improving interpretability for developers. These results pave the way for scalable, explainable, and locally deployable tools that can shift vulnerability analysis from purely syntactic checks to deeper structural and semantic insights, facilitating broader adoption in real-world secure software development. 

---
# Simulating Viva Voce Examinations to Evaluate Clinical Reasoning in Large Language Models 

**Authors**: Christopher Chiu, Silviu Pitis, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2510.10278)  

**Abstract**: Clinical reasoning in medicine is a hypothesis-driven process where physicians refine diagnoses from limited information through targeted history, physical examination, and diagnostic investigations. In contrast, current medical benchmarks for large language models (LLMs) primarily assess knowledge recall through single-turn questions, where complete clinical information is provided upfront. To address this gap, we introduce VivaBench, a multi-turn benchmark that evaluates sequential clinical reasoning in LLM agents. Our dataset consists of 1762 physician-curated clinical vignettes structured as interactive scenarios that simulate a (oral) examination in medical training, requiring agents to actively probe for relevant findings, select appropriate investigations, and synthesize information across multiple steps to reach a diagnosis. While current LLMs demonstrate competence in diagnosing conditions from well-described clinical presentations, their performance degrades significantly when required to navigate iterative diagnostic reasoning under uncertainty in our evaluation. Our analysis identified several failure modes that mirror common cognitive errors in clinical practice, including: (1) fixation on initial hypotheses, (2) inappropriate investigation ordering, (3) premature diagnostic closure, and (4) failing to screen for critical conditions. These patterns reveal fundamental limitations in how current LLMs reason and make decisions under uncertainty. Through VivaBench, we provide a standardized benchmark for evaluating conversational medical AI systems for real-world clinical decision support. Beyond medical applications, we contribute to the larger corpus of research on agentic AI by demonstrating how sequential reasoning trajectories can diverge in complex decision-making environments. 

---
# From Programs to Poses: Factored Real-World Scene Generation via Learned Program Libraries 

**Authors**: Joy Hsu, Emily Jin, Jiajun Wu, Niloy J. Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2510.10292)  

**Abstract**: Real-world scenes, such as those in ScanNet, are difficult to capture, with highly limited data available. Generating realistic scenes with varied object poses remains an open and challenging task. In this work, we propose FactoredScenes, a framework that synthesizes realistic 3D scenes by leveraging the underlying structure of rooms while learning the variation of object poses from lived-in scenes. We introduce a factored representation that decomposes scenes into hierarchically organized concepts of room programs and object poses. To encode structure, FactoredScenes learns a library of functions capturing reusable layout patterns from which scenes are drawn, then uses large language models to generate high-level programs, regularized by the learned library. To represent scene variations, FactoredScenes learns a program-conditioned model to hierarchically predict object poses, and retrieves and places 3D objects in a scene. We show that FactoredScenes generates realistic, real-world rooms that are difficult to distinguish from real ScanNet scenes. 

---
# Reasoning-Enhanced Large Language Models for Molecular Property Prediction 

**Authors**: Jiaxi Zhuang, Yaorui Shi, Jue Hou, Yunong He, Mingwei Ye, Mingjun Xu, Yuming Su, Linfeng Zhang, Linfeng Zhang, Guolin Ke, Hengxing Cai  

**Link**: [PDF](https://arxiv.org/pdf/2510.10248)  

**Abstract**: Molecular property prediction is crucial for drug discovery and materials science, yet existing approaches suffer from limited interpretability, poor cross-task generalization, and lack of chemical reasoning capabilities. Traditional machine learning models struggle with task transferability, while specialized molecular language models provide little insight into their decision-making processes. To address these limitations, we propose \textbf{MPPReasoner}, a multimodal large language model that incorporates chemical reasoning for molecular property prediction. Our approach, built upon Qwen2.5-VL-7B-Instruct, integrates molecular images with SMILES strings to enable comprehensive molecular understanding. We develop a two-stage training strategy: supervised fine-tuning (SFT) using 16,000 high-quality reasoning trajectories generated through expert knowledge and multiple teacher models, followed by Reinforcement Learning from Principle-Guided Rewards (RLPGR). RLPGR employs verifiable, rule-based rewards that systematically evaluate chemical principle application, molecular structure analysis, and logical consistency through computational verification. Extensive experiments across 8 datasets demonstrate significant performance improvements, with MPPReasoner outperforming the best baselines by 7.91\% and 4.53\% on in-distribution and out-of-distribution tasks respectively. MPPReasoner exhibits exceptional cross-task generalization and generates chemically sound reasoning paths that provide valuable insights into molecular property analysis, substantially enhancing both interpretability and practical utility for chemists. Code is available at this https URL. 

---
# LLMs are All You Need? Improving Fuzz Testing for MOJO with Large Language Models 

**Authors**: Linghan Huang, Peizhou Zhao, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10179)  

**Abstract**: The rapid development of large language models (LLMs) has revolutionized software testing, particularly fuzz testing, by automating the generation of diverse and effective test inputs. This advancement holds great promise for improving software reliability. Meanwhile, the introduction of MOJO, a high-performance AI programming language blending Python's usability with the efficiency of C and C++, presents new opportunities to enhance AI model scalability and programmability. However, as a new language, MOJO lacks comprehensive testing frameworks and a sufficient corpus for LLM-based testing, which exacerbates model hallucination. In this case, LLMs will generate syntactically valid but semantically incorrect code, significantly reducing the effectiveness of fuzz testing. To address this challenge, we propose MOJOFuzzer, the first adaptive LLM-based fuzzing framework designed for zero-shot learning environments of emerging programming languages. MOJOFuzzer integrates a mutil-phase framework that systematically eliminates low-quality generated inputs before execution, significantly improving test case validity. Furthermore, MOJOFuzzer dynamically adapts LLM prompts based on runtime feedback for test case mutation, enabling an iterative learning process that continuously enhances fuzzing efficiency and bug detection performance. Our experimental results demonstrate that MOJOFuzzer significantly enhances test validity, API coverage, and bug detection performance, outperforming traditional fuzz testing and state-of-the-art LLM-based fuzzing approaches. Using MOJOFuzzer, we have conducted a first large-scale fuzz testing evaluation of MOJO, uncorvering 13 previous unknown bugs. This study not only advances the field of LLM-driven software testing but also establishes a foundational methodology for leveraging LLMs in the testing of emerging programming languages. 

---
# Training-Free In-Context Forensic Chain for Image Manipulation Detection and Localization 

**Authors**: Rui Chen, Bin Liu, Changtao Miao, Xinghao Wang, Yi Li, Tao Gong, Qi Chu, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10111)  

**Abstract**: Advances in image tampering pose serious security threats, underscoring the need for effective image manipulation localization (IML). While supervised IML achieves strong performance, it depends on costly pixel-level annotations. Existing weakly supervised or training-free alternatives often underperform and lack interpretability. We propose the In-Context Forensic Chain (ICFC), a training-free framework that leverages multi-modal large language models (MLLMs) for interpretable IML tasks. ICFC integrates an objectified rule construction with adaptive filtering to build a reliable knowledge base and a multi-step progressive reasoning pipeline that mirrors expert forensic workflows from coarse proposals to fine-grained forensics results. This design enables systematic exploitation of MLLM reasoning for image-level classification, pixel-level localization, and text-level interpretability. Across multiple benchmarks, ICFC not only surpasses state-of-the-art training-free methods but also achieves competitive or superior performance compared to weakly and fully supervised approaches. 

---
# Efficient Onboard Vision-Language Inference in UAV-Enabled Low-Altitude Economy Networks via LLM-Enhanced Optimization 

**Authors**: Yang Li, Ruichen Zhang, Yinqiu Liu, Guangyuan Liu, Dusit Niyato, Abbas Jamalipour, Xianbin Wang, Dong In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.10028)  

**Abstract**: The rapid advancement of Low-Altitude Economy Networks (LAENets) has enabled a variety of applications, including aerial surveillance, environmental sensing, and semantic data collection. To support these scenarios, unmanned aerial vehicles (UAVs) equipped with onboard vision-language models (VLMs) offer a promising solution for real-time multimodal inference. However, ensuring both inference accuracy and communication efficiency remains a significant challenge due to limited onboard resources and dynamic network conditions. In this paper, we first propose a UAV-enabled LAENet system model that jointly captures UAV mobility, user-UAV communication, and the onboard visual question answering (VQA) pipeline. Based on this model, we formulate a mixed-integer non-convex optimization problem to minimize task latency and power consumption under user-specific accuracy constraints. To solve the problem, we design a hierarchical optimization framework composed of two parts: (i) an Alternating Resolution and Power Optimization (ARPO) algorithm for resource allocation under accuracy constraints, and (ii) a Large Language Model-augmented Reinforcement Learning Approach (LLaRA) for adaptive UAV trajectory optimization. The large language model (LLM) serves as an expert in refining reward design of reinforcement learning in an offline fashion, introducing no additional latency in real-time decision-making. Numerical results demonstrate the efficacy of our proposed framework in improving inference performance and communication efficiency under dynamic LAENet conditions. 

---
# Skill-Targeted Adaptive Training 

**Authors**: Yinghui He, Abhishek Panigrahi, Yong Lin, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2510.10023)  

**Abstract**: Language models often show little to no improvement (i.e., "saturation") when trained via vanilla supervised fine-tuning (SFT) on data similar to what they saw in their training set (e.g., MATH). We introduce a new fine-tuning strategy, STAT, to train such a student model by using the metacognition ability of a stronger large language model (LLM) as the teacher. The teacher uses the task dataset to create a list of skills needed for the task, and then labels each data point with its required skills (Didolkar et al., 2024). By monitoring the student's answers, the teacher creates a Missing-Skill-Profile for the student, tracking how often they failed to apply each skill in their responses. We use this idea to build a modified training set in one of two ways. In STAT-Sel, the teacher uses an existing set of training examples but adaptively reweights them according to the Missing-Skill-Profile. In STAT-Syn, the teacher synthesizes additional examples involving missing skills. Across extensive experiments on Llama and Qwen models, our methods yield improvements of up to 7.5% on MATH, whereas SFT provides only limited gains. Furthermore, STAT enhances performance on out-of-distribution benchmarks (e.g., AIME24/25, AMC23, etc.) by an average of 4.6%. Crucially, we find that STAT is complementary to RL via GRPO (Shao et al., 2024): after the model is improved using STAT to address skill gaps, GRPO continues to add further gains. We conclude that skill-targeted adaptive training should broadly improve current training pipelines. Our code is available at: this https URL. 

---
# Agentic Property-Based Testing: Finding Bugs Across the Python Ecosystem 

**Authors**: Muhammad Maaz, Liam DeVoe, Zac Hatfield-Dodds, Nicholas Carlini  

**Link**: [PDF](https://arxiv.org/pdf/2510.09907)  

**Abstract**: Property-based testing (PBT) is a lightweight formal method, typically implemented as a randomized testing framework. Users specify the input domain for their test using combinators supplied by the PBT framework, and the expected properties or invariants as a unit-test function. The framework then searches for a counterexample, e.g. by generating inputs and calling the test function. In this work, we demonstrate an LLM-based agent which analyzes Python modules, infers function-specific and cross-function properties from code and documentation, synthesizes and executes PBTs, reflects on outputs of these tests to confirm true bugs, and finally outputs actionable bug reports for the developer. We perform an extensive evaluation of our agent across 100 popular Python packages. Of the bug reports generated by the agent, we found after manual review that 56\% were valid bugs and 32\% were valid bugs that we would report to maintainers. We then developed a ranking rubric to surface high-priority valid bugs to developers, and found that of the 21 top-scoring bugs, 86\% were valid and 81\% we would report. The bugs span diverse failure modes from serialization failures to numerical precision errors to flawed cache implementations. We reported 5 bugs, 4 with patches, including to NumPy and cloud computing SDKs, with 3 patches merged successfully. Our results suggest that LLMs with PBT provides a rigorous and scalable method for autonomously testing software. Our code and artifacts are available at: this https URL. 

---
# ROBOPSY PL[AI]: Using Role-Play to Investigate how LLMs Present Collective Memory 

**Authors**: Margarete Jahrmann, Thomas Brandstetter, Stefan Glasauer  

**Link**: [PDF](https://arxiv.org/pdf/2510.09874)  

**Abstract**: The paper presents the first results of an artistic research project investigating how Large Language Models (LLMs) curate and present collective memory. In a public installation exhibited during two months in Vienna in 2025, visitors could interact with five different LLMs (ChatGPT with GPT 4o and GPT 4o mini, Mistral Large, DeepSeek-Chat, and a locally run Llama 3.1 model), which were instructed to act as narrators, implementing a role-playing game revolving around the murder of Austrian philosopher Moritz Schlick in 1936. Results of the investigation include protocols of LLM-user interactions during the game and qualitative conversations after the play experience to get insight into the players' reactions to the game. In a quantitative analysis 115 introductory texts for role-playing generated by the LLMs were examined by different methods of natural language processing, including semantic similarity and sentiment analysis. While the qualitative player feedback allowed to distinguish three distinct types of users, the quantitative text analysis showed significant differences between how the different LLMs presented the historical content. Our study thus adds to ongoing efforts to analyse LLM performance, but also suggests a way of how these efforts can be disseminated in a playful way to a general audience. 

---
# Large Language Models for Imbalanced Classification: Diversity makes the difference 

**Authors**: Dang Nguyen, Sunil Gupta, Kien Do, Thin Nguyen, Taylor Braund, Alexis Whitton, Svetha Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2510.09783)  

**Abstract**: Oversampling is one of the most widely used approaches for addressing imbalanced classification. The core idea is to generate additional minority samples to rebalance the dataset. Most existing methods, such as SMOTE, require converting categorical variables into numerical vectors, which often leads to information loss. Recently, large language model (LLM)-based methods have been introduced to overcome this limitation. However, current LLM-based approaches typically generate minority samples with limited diversity, reducing robustness and generalizability in downstream classification tasks. To address this gap, we propose a novel LLM-based oversampling method designed to enhance diversity. First, we introduce a sampling strategy that conditions synthetic sample generation on both minority labels and features. Second, we develop a new permutation strategy for fine-tuning pre-trained LLMs. Third, we fine-tune the LLM not only on minority samples but also on interpolated samples to further enrich variability. Extensive experiments on 10 tabular datasets demonstrate that our method significantly outperforms eight SOTA baselines. The generated synthetic samples are both realistic and diverse. Moreover, we provide theoretical analysis through an entropy-based perspective, proving that our method encourages diversity in the generated samples. 

---
# Evaluating LLM-Based Process Explanations under Progressive Behavioral-Input Reduction 

**Authors**: P. van Oerle, R. H. Bemthuis, F. A. Bukhsh  

**Link**: [PDF](https://arxiv.org/pdf/2510.09732)  

**Abstract**: Large Language Models (LLMs) are increasingly used to generate textual explanations of process models discovered from event logs. Producing explanations from large behavioral abstractions (e.g., directly-follows graphs or Petri nets) can be computationally expensive. This paper reports an exploratory evaluation of explanation quality under progressive behavioral-input reduction, where models are discovered from progressively smaller prefixes of a fixed log. Our pipeline (i) discovers models at multiple input sizes, (ii) prompts an LLM to generate explanations, and (iii) uses a second LLM to assess completeness, bottleneck identification, and suggested improvements. On synthetic logs, explanation quality is largely preserved under moderate reduction, indicating a practical cost-quality trade-off. The study is exploratory, as the scores are LLM-based (comparative signals rather than ground truth) and the data are synthetic. The results suggest a path toward more computationally efficient, LLM-assisted process analysis in resource-constrained settings. 

---
# InterCorpRel-LLM: Enhancing Financial Relational Understanding with Graph-Language Models 

**Authors**: Qianyou Sun, Jiexin Zheng, Bohan Jin, Lihua Chen, Yijie Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09735)  

**Abstract**: Identifying inter-firm relationships such as supply and competitive ties is critical for financial analysis and corporate governance, yet remains challenging due to the scale, sparsity, and contextual dependence of corporate data. Graph-based methods capture structure but miss semantic depth, while large language models (LLMs) excel at text but remain limited in their ability to represent relational dependencies. To address this, we propose InterCorpRel-LLM, a cross-modal framework that integrates GNNs with LLMs, supported by a proprietary dataset derived from FactSet supply chain records and three tailored training tasks: company graph matching, industry classification, and supply relation prediction. This design enables effective joint modeling of structure and semantics. Experiments show that InterCorpRel-LLM substantially outperforms strong baselines, including GPT-5, on a supply relation identification task, achieving an F-score of 0.8543 vs. 0.2287 with only a 7B-parameter backbone and lightweight training. The model also generalizes to zero-shot competitor identification, underscoring its ability to capture nuanced inter-firm dynamics. Our framework thus provides analysts and strategists with a robust tool for mapping and reasoning about complex corporate networks, enhancing decision-making and risk management in dynamic markets. 

---
# InteractScience: Programmatic and Visually-Grounded Evaluation of Interactive Scientific Demonstration Code Generation 

**Authors**: Qiaosheng Chen, Yang Liu, Lei Li, Kai Chen, Qipeng Guo, Gong Cheng, Fei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.09724)  

**Abstract**: Large Language Models (LLMs) are increasingly capable of generating complete applications from natural language instructions, creating new opportunities in science and education. In these domains, interactive scientific demonstrations are particularly valuable for explaining concepts, supporting new teaching methods, and presenting research findings. Generating such demonstrations requires models to combine accurate scientific knowledge with the ability to implement interactive front-end code that behaves correctly and responds to user actions. This capability goes beyond the scope of existing benchmarks, which typically evaluate either knowledge question answering without grounding in code or static web code generation without scientific interactivity. To evaluate this integrated ability, we design a hybrid framework that combines programmatic functional testing to rigorously verify interaction logic with visually-grounded qualitative testing to assess rendered outputs against reference snapshots. Building on this framework, we present InteractScience, a benchmark consisting of a substantial set of carefully designed questions across five scientific domains, each paired with unit tests, reference snapshots, and checklists. We evaluate 30 leading open- and closed-source LLMs and report results that highlight ongoing weaknesses in integrating domain knowledge with interactive front-end coding. Our work positions InteractScience as the first benchmark to automatically measure this combined capability with realistic interactive operations, providing a foundation for advancing reliable and educationally useful scientific demonstration code generation. All code and data are publicly available at this https URL. 

---
# High-Power Training Data Identification with Provable Statistical Guarantees 

**Authors**: Zhenlong Liu, Hao Zeng, Weiran Huang, Hongxin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.09717)  

**Abstract**: Identifying training data within large-scale models is critical for copyright litigation, privacy auditing, and ensuring fair evaluation. The conventional approaches treat it as a simple binary classification task without statistical guarantees. A recent approach is designed to control the false discovery rate (FDR), but its guarantees rely on strong, easily violated assumptions. In this paper, we introduce Provable Training Data Identification (PTDI), a rigorous method that identifies a set of training data with strict false discovery rate (FDR) control. Specifically, our method computes p-values for each data point using a set of known unseen data, and then constructs a conservative estimator for the data usage proportion of the test set, which allows us to scale these p-values. Our approach then selects the final set of training data by identifying all points whose scaled p-values fall below a data-dependent threshold. This entire procedure enables the discovery of training data with provable, strict FDR control and significantly boosted power. Extensive experiments across a wide range of models (LLMs and VLMs), and datasets demonstrate that PTDI strictly controls the FDR and achieves higher power. 

---
# ICL-Router: In-Context Learned Model Representations for LLM Routing 

**Authors**: Chenxu Wang, Hao Li, Yiqun Zhang, Linyao Chen, Jianhao Chen, Ping Jian, Peng Ye, Qiaosheng Zhang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09719)  

**Abstract**: Large language models (LLMs) often exhibit complementary strengths. Model routing harnesses these strengths by dynamically directing each query to the most suitable model, given a candidate model pool. However, routing performance relies on accurate model representations, and adding new models typically requires retraining, limiting scalability. To address these challenges, we propose a novel routing method using in-context vectors to represent model capabilities. The method proceeds in two stages. First, queries are embedded and projected into vectors, with a projector and LLM-based router trained to reconstruct the original queries, aligning vector representations with the router's semantic space. Second, each candidate model is profiled on a query set, and the router learns -- based on in-context vectors of query and model performance -- to predict whether each model can correctly answer new queries. Extensive experiments demonstrate that our method achieves state-of-the-art routing performance in both in-distribution and out-of-distribution tasks. Moreover, our method allows for seamless integration of new models without retraining the router. The code is available at this https URL. 

---
# CREST-Search: Comprehensive Red-teaming for Evaluating Safety Threats in Large Language Models Powered by Web Search 

**Authors**: Haoran Ou, Kangjie Chen, Xingshuo Han, Gelei Deng, Jie Zhang, Han Qiu, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09689)  

**Abstract**: Large Language Models (LLMs) excel at tasks such as dialogue, summarization, and question answering, yet they struggle to adapt to specialized domains and evolving facts. To overcome this, web search has been integrated into LLMs, allowing real-time access to online content. However, this connection magnifies safety risks, as adversarial prompts combined with untrusted sources can cause severe vulnerabilities. We investigate red teaming for LLMs with web search and present CREST-Search, a framework that systematically exposes risks in such systems. Unlike existing methods for standalone LLMs, CREST-Search addresses the complex workflow of search-enabled models by generating adversarial queries with in-context learning and refining them through iterative feedback. We further construct WebSearch-Harm, a search-specific dataset to fine-tune LLMs into efficient red-teaming agents. Experiments show that CREST-Search effectively bypasses safety filters and reveals vulnerabilities in modern web-augmented LLMs, underscoring the need for specialized defenses to ensure trustworthy deployment. 

---
# Fortifying LLM-Based Code Generation with Graph-Based Reasoning on Secure Coding Practices 

**Authors**: Rupam Patir, Keyan Guo, Haipeng Cai, Hongxin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.09682)  

**Abstract**: The code generation capabilities of Large Language Models (LLMs) have transformed the field of software development. However, this advancement also presents significant security challenges, as LLM-generated code often contains vulnerabilities. One direction of research strengthens LLMs by injecting or refining security knowledge through curated datasets, model tuning, or static analyzers. While effective in certain settings, these methods can be resource-intensive, less adaptable to zero-day vulnerabilities, and often inapplicable to proprietary models. To address these challenges, we introduce GRASP, which explores a new direction that focuses on structured reasoning over Secure Coding Practices(SCPs) rather than additional training or external feedback. GRASP comprises two key ideas: (1) an SCP graph that organizes SCPs into a Directed Acyclic Graph (DAG) capturing dependencies and relationships, and (2) a graph-based reasoning process that systematically guides LLMs through relevant SCPs for code generation. This design enables interpretable, model-agnostic, and scalable security improvements, particularly for previously unseen vulnerabilities. Our evaluation shows that GRASP consistently achieves Security Rates (SR) exceeding 80% across multiple LLMs, and delivers up to 88% improvements over baselines on zero-day vulnerabilities. 

---
# Real-Time Health Analytics Using Ontology-Driven Complex Event Processing and LLM Reasoning: A Tuberculosis Case Study 

**Authors**: Ritesh Chandra, Sonali Agarwal, Navjot Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.09646)  

**Abstract**: Timely detection of critical health conditions remains a major challenge in public health analytics, especially in Big Data environments characterized by high volume, rapid velocity, and diverse variety of clinical data. This study presents an ontology-enabled real-time analytics framework that integrates Complex Event Processing (CEP) and Large Language Models (LLMs) to enable intelligent health event detection and semantic reasoning over heterogeneous, high-velocity health data streams. The architecture leverages the Basic Formal Ontology (BFO) and Semantic Web Rule Language (SWRL) to model diagnostic rules and domain knowledge. Patient data is ingested and processed using Apache Kafka and Spark Streaming, where CEP engines detect clinically significant event patterns. LLMs support adaptive reasoning, event interpretation, and ontology refinement. Clinical information is semantically structured as Resource Description Framework (RDF) triples in Graph DB, enabling SPARQL-based querying and knowledge-driven decision support. The framework is evaluated using a dataset of 1,000 Tuberculosis (TB) patients as a use case, demonstrating low-latency event detection, scalable reasoning, and high model performance (in terms of precision, recall, and F1-score). These results validate the system's potential for generalizable, real-time health analytics in complex Big Data scenarios. 

---
# Mission Impossible: Feedback-Guided Dynamic Interactive Planning for Improving Reasoning on LLMs 

**Authors**: Dong Yan, Gaochen Wu, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.05577)  

**Abstract**: Recent advancements in language agents have led to significant improvements in multi-hop reasoning tasks. However, existing approaches often struggle with handling open-domain problems, which require massive information retrieval due to their reliance on a fixed sequence of actions. To address this, we propose Feedback-Guided Dynamic Interactive Planning (FGDIP), a novel framework tailored to enhance reasoning in LLMs by utilizing dynamic and adaptive strategies for information exploration in open-domain multi-hop reasoning tasks. Our approach begins by identifying key entities relevant to the problem, which serve as the initial nodes in the reasoning process. From these initial nodes, we then generate reasoning child nodes with the process being refined through a combination of historical error analysis and real-time feedback, which allows the framework to dynamically adjust and optimize its reasoning strategies. By integrating depth-first search with an innovative node generation technique, our framework adapts based on both prior error paths and concurrently generated nodes at the same hierarchical level. This dynamic strategy effectively expands the search space while ensuring the reasoning process systematically converges toward accurate solutions. Experimental results show that FGDIP achieved up to 54.47% F1 score on the HotpotQA dataset and 70.05% on the StrategyQA dataset, surpassing the best baseline by 5.03% and 7.25% respectively, highlighting its versatility and potential to enhance language agents in multi-hop reasoning tasks. 

---
