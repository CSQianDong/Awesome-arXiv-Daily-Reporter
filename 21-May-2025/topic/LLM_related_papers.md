# Field Matters: A lightweight LLM-enhanced Method for CTR Prediction 

**Authors**: Yu Cui, Feng Liu, Jiawei Chen, Xingyu Lou, Changwang Zhang, Jun Wang, Yuegang Sun, Xiaohu Yang, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14057)  

**Abstract**: Click-through rate (CTR) prediction is a fundamental task in modern recommender systems. In recent years, the integration of large language models (LLMs) has been shown to effectively enhance the performance of traditional CTR methods. However, existing LLM-enhanced methods often require extensive processing of detailed textual descriptions for large-scale instances or user/item entities, leading to substantial computational overhead. To address this challenge, this work introduces LLaCTR, a novel and lightweight LLM-enhanced CTR method that employs a field-level enhancement paradigm. Specifically, LLaCTR first utilizes LLMs to distill crucial and lightweight semantic knowledge from small-scale feature fields through self-supervised field-feature fine-tuning. Subsequently, it leverages this field-level semantic knowledge to enhance both feature representation and feature interactions. In our experiments, we integrate LLaCTR with six representative CTR models across four datasets, demonstrating its superior performance in terms of both effectiveness and efficiency compared to existing LLM-enhanced methods. Our code is available at this https URL. 

---
# Information Extraction from Visually Rich Documents using LLM-based Organization of Documents into Independent Textual Segments 

**Authors**: Aniket Bhattacharyya, Anurag Tripathi, Ujjal Das, Archan Karmakar, Amit Pathak, Maneesh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.13535)  

**Abstract**: Information extraction (IE) from Visually Rich Documents (VRDs) containing layout features along with text is a critical and well-studied task. Specialized non-LLM NLP-based solutions typically involve training models using both textual and geometric information to label sequences/tokens as named entities or answers to specific questions. However, these approaches lack reasoning, are not able to infer values not explicitly present in documents, and do not generalize well to new formats. Generative LLM-based approaches proposed recently are capable of reasoning, but struggle to comprehend clues from document layout especially in previously unseen document formats, and do not show competitive performance in heterogeneous VRD benchmark datasets. In this paper, we propose BLOCKIE, a novel LLM-based approach that organizes VRDs into localized, reusable semantic textual segments called $\textit{semantic blocks}$, which are processed independently. Through focused and more generalizable reasoning,our approach outperforms the state-of-the-art on public VRD benchmarks by 1-3% in F1 scores, is resilient to document formats previously not encountered and shows abilities to correctly extract information not explicitly present in documents. 

---
# LLM-Based User Simulation for Low-Knowledge Shilling Attacks on Recommender Systems 

**Authors**: Shengkang Gu, Jiahao Liu, Dongsheng Li, Guangping Zhang, Mingzhe Han, Hansu Gu, Peng Zhang, Ning Gu, Li Shang, Tun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13528)  

**Abstract**: Recommender systems (RS) are increasingly vulnerable to shilling attacks, where adversaries inject fake user profiles to manipulate system outputs. Traditional attack strategies often rely on simplistic heuristics, require access to internal RS data, and overlook the manipulation potential of textual reviews. In this work, we introduce Agent4SR, a novel framework that leverages Large Language Model (LLM)-based agents to perform low-knowledge, high-impact shilling attacks through both rating and review generation. Agent4SR simulates realistic user behavior by orchestrating adversarial interactions, selecting items, assigning ratings, and crafting reviews, while maintaining behavioral plausibility. Our design includes targeted profile construction, hybrid memory retrieval, and a review attack strategy that propagates target item features across unrelated reviews to amplify manipulation. Extensive experiments on multiple datasets and RS architectures demonstrate that Agent4SR outperforms existing low-knowledge baselines in both effectiveness and stealth. Our findings reveal a new class of emergent threats posed by LLM-driven agents, underscoring the urgent need for enhanced defenses in modern recommender systems. 

---
# Rank-K: Test-Time Reasoning for Listwise Reranking 

**Authors**: Eugene Yang, Andrew Yates, Kathryn Ricci, Orion Weller, Vivek Chari, Benjamin Van Durme, Dawn Lawrie  

**Link**: [PDF](https://arxiv.org/pdf/2505.14432)  

**Abstract**: Retrieve-and-rerank is a popular retrieval pipeline because of its ability to make slow but effective rerankers efficient enough at query time by reducing the number of comparisons. Recent works in neural rerankers take advantage of large language models for their capability in reasoning between queries and passages and have achieved state-of-the-art retrieval effectiveness. However, such rerankers are resource-intensive, even after heavy optimization. In this work, we introduce Rank-K, a listwise passage reranking model that leverages the reasoning capability of the reasoning language model at query time that provides test time scalability to serve hard queries. We show that Rank-K improves retrieval effectiveness by 23\% over the RankZephyr, the state-of-the-art listwise reranker, when reranking a BM25 initial ranked list and 19\% when reranking strong retrieval results by SPLADE-v3. Since Rank-K is inherently a multilingual model, we found that it ranks passages based on queries in different languages as effectively as it does in monolingual retrieval. 

---
# Know Or Not: a library for evaluating out-of-knowledge base robustness 

**Authors**: Jessica Foo, Pradyumna Shyama Prasad, Shaun Khoo  

**Link**: [PDF](https://arxiv.org/pdf/2505.13545)  

**Abstract**: While the capabilities of large language models (LLMs) have progressed significantly, their use in high-stakes applications have been limited due to risks of hallucination. One key approach in reducing hallucination is retrieval-augmented generation (RAG), but even in such setups, LLMs may still hallucinate when presented with questions outside of the knowledge base. Such behavior is unacceptable in high-stake applications where LLMs are expected to abstain from answering queries it does not have sufficient context on. In this work, we present a novel methodology for systematically evaluating out-of-knowledge base (OOKB) robustness of LLMs (whether LLMs know or do not know) in the RAG setting, without the need for manual annotation of gold standard answers. We implement our methodology in knowornot, an open-source library that enables users to develop their own customized evaluation data and pipelines for OOKB robustness. knowornot comprises four main features. Firstly, it provides a unified, high-level API that streamlines the process of setting up and running robustness benchmarks. Secondly, its modular architecture emphasizes extensibility and flexibility, allowing users to easily integrate their own LLM clients and RAG settings. Thirdly, its rigorous data modeling design ensures experiment reproducibility, reliability and traceability. Lastly, it implements a comprehensive suite of tools for users to customize their pipelines. We demonstrate the utility of knowornot by developing a challenging benchmark, PolicyBench, which spans four Question-Answer (QA) chatbots on government policies, and analyze its OOKB robustness. The source code of knowornot is available this https URL. 

---
# Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning 

**Authors**: Wenlin Zhang, Xiangyang Li, Kuicai Dong, Yichao Wang, Pengyue Jia, Xiaopeng Li, Yingyi Zhang, Derong Xu, Zhaocheng Du, Huifeng Guo, Ruiming Tang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.14069)  

**Abstract**: Retrieval-augmented generation (RAG) enhances the text generation capabilities of large language models (LLMs) by integrating external knowledge and up-to-date information. However, traditional RAG systems are limited by static workflows and lack the adaptability required for multistep reasoning and complex task management. To address these limitations, agentic RAG systems (e.g., DeepResearch) have been proposed, enabling dynamic retrieval strategies, iterative context refinement, and adaptive workflows for handling complex search queries beyond the capabilities of conventional RAG. Recent advances, such as Search-R1, have demonstrated promising gains using outcome-based reinforcement learning, where the correctness of the final answer serves as the reward signal. Nevertheless, such outcome-supervised agentic RAG methods face challenges including low exploration efficiency, gradient conflict, and sparse reward signals. To overcome these challenges, we propose to utilize fine-grained, process-level rewards to improve training stability, reduce computational costs, and enhance efficiency. Specifically, we introduce a novel method ReasonRAG that automatically constructs RAG-ProGuide, a high-quality dataset providing process-level rewards for (i) query generation, (ii) evidence extraction, and (iii) answer generation, thereby enhancing model inherent capabilities via process-supervised reinforcement learning. With the process-level policy optimization, the proposed framework empowers LLMs to autonomously invoke search, generate queries, extract relevant evidence, and produce final answers. Compared to existing approaches such as Search-R1 and traditional RAG systems, ReasonRAG, leveraging RAG-ProGuide, achieves superior performance on five benchmark datasets using only 5k training instances, significantly fewer than the 90k training instances required by Search-R1. 

---
# Unify Graph Learning with Text: Unleashing LLM Potentials for Session Search 

**Authors**: Songhao Wu, Quan Tu, Hong Liu, Jia Xu, Zhongyi Liu, Guannan Zhang, Ran Wang, Xiuying Chen, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14156)  

**Abstract**: Session search involves a series of interactive queries and actions to fulfill user's complex information need. Current strategies typically prioritize sequential modeling for deep semantic understanding, overlooking the graph structure in interactions. While some approaches focus on capturing structural information, they use a generalized representation for documents, neglecting the word-level semantic modeling. In this paper, we propose Symbolic Graph Ranker (SGR), which aims to take advantage of both text-based and graph-based approaches by leveraging the power of recent Large Language Models (LLMs). Concretely, we first introduce a set of symbolic grammar rules to convert session graph into text. This allows integrating session history, interaction process, and task instruction seamlessly as inputs for the LLM. Moreover, given the natural discrepancy between LLMs pre-trained on textual corpora, and the symbolic language we produce using our graph-to-text grammar, our objective is to enhance LLMs' ability to capture graph structures within a textual format. To achieve this, we introduce a set of self-supervised symbolic learning tasks including link prediction, node content generation, and generative contrastive learning, to enable LLMs to capture the topological information from coarse-grained to fine-grained. Experiment results and comprehensive analysis on two benchmark datasets, AOL and Tiangong-ST, confirm the superiority of our approach. Our paradigm also offers a novel and effective methodology that bridges the gap between traditional search strategies and modern LLMs. 

---
# LLM-Based Compact Reranking with Document Features for Scientific Retrieval 

**Authors**: Runchu Tian, Xueqiang Xu, Bowen Jin, SeongKu Kang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.13757)  

**Abstract**: Scientific retrieval is essential for advancing academic discovery. Within this process, document reranking plays a critical role by refining first-stage retrieval results. However, large language model (LLM) listwise reranking faces unique challenges in the scientific domain. First-stage retrieval is often suboptimal in the scientific domain, so relevant documents are ranked lower. Moreover, conventional listwise reranking uses the full text of candidate documents in the context window, limiting the number of candidates that can be considered. As a result, many relevant documents are excluded before reranking, which constrains overall retrieval performance. To address these challenges, we explore compact document representations based on semantic features such as categories, sections, and keywords, and propose a training-free, model-agnostic reranking framework for scientific retrieval called CoRank. The framework involves three stages: (i) offline extraction of document-level features, (ii) coarse reranking using these compact representations, and (iii) fine-grained reranking on full texts of the top candidates from stage (ii). This hybrid design provides a high-level abstraction of document semantics, expands candidate coverage, and retains critical details required for precise ranking. Experiments on LitSearch and CSFCube show that CoRank significantly improves reranking performance across different LLM backbones, increasing nDCG@10 from 32.0 to 39.7. Overall, these results highlight the value of information extraction for reranking in scientific retrieval. 

---
# An agentic system with reinforcement-learned subsystem improvements for parsing form-like documents 

**Authors**: Ayesha Amjad, Saurav Sthapit, Tahir Qasim Syed  

**Link**: [PDF](https://arxiv.org/pdf/2505.13504)  

**Abstract**: Extracting alphanumeric data from form-like documents such as invoices, purchase orders, bills, and financial documents is often performed via vision (OCR) and learning algorithms or monolithic pipelines with limited potential for systemic improvements. We propose an agentic AI system that leverages Large Language Model (LLM) agents and a reinforcement learning (RL) driver agent to automate consistent, self-improving extraction under LLM inference uncertainty. Our work highlights the limitations of monolithic LLM-based extraction and introduces a modular, multi-agent framework with task-specific prompts and an RL policy of rewards and penalties to guide a meta-prompting agent to learn from past errors and improve prompt-based actor agents. This self-corrective adaptive system handles diverse documents, file formats, layouts, and LLMs, aiming to automate accurate information extraction without the need for human intervention. Results as reported on two benchmark datasets of SOIRE, and CORD, are promising for the agentic AI framework. 

---
# Beyond Chains: Bridging Large Language Models and Knowledge Bases in Complex Question Answering 

**Authors**: Yihua Zhu, Qianying Liu, Akiko Aizawa, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2505.14099)  

**Abstract**: Knowledge Base Question Answering (KBQA) aims to answer natural language questions using structured knowledge from KBs. While LLM-only approaches offer generalization, they suffer from outdated knowledge, hallucinations, and lack of transparency. Chain-based KG-RAG methods address these issues by incorporating external KBs, but are limited to simple chain-structured questions due to the absence of planning and logical structuring. Inspired by semantic parsing methods, we propose PDRR: a four-stage framework consisting of Predict, Decompose, Retrieve, and Reason. Our method first predicts the question type and decomposes the question into structured triples. Then retrieves relevant information from KBs and guides the LLM as an agent to reason over and complete the decomposed triples. Experimental results demonstrate that PDRR consistently outperforms existing methods across various LLM backbones and achieves superior performance on both chain-structured and non-chain complex questions. 

---
# Geography-Aware Large Language Models for Next POI Recommendation 

**Authors**: Zhao Liu, Wei Liu, Huajie Zhu, Jianxing Yu, Jian Yin, Wang-Chien Lee, Shun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13526)  

**Abstract**: The next Point-of-Interest (POI) recommendation task aims to predict users' next destinations based on their historical movement data and plays a key role in location-based services and personalized applications. Accurate next POI recommendation depends on effectively modeling geographic information and POI transition relations, which are crucial for capturing spatial dependencies and user movement patterns. While Large Language Models (LLMs) exhibit strong capabilities in semantic understanding and contextual reasoning, applying them to spatial tasks like next POI recommendation remains challenging. First, the infrequent nature of specific GPS coordinates makes it difficult for LLMs to model precise spatial contexts. Second, the lack of knowledge about POI transitions limits their ability to capture potential POI-POI relationships. To address these issues, we propose GA-LLM (Geography-Aware Large Language Model), a novel framework that enhances LLMs with two specialized components. The Geographic Coordinate Injection Module (GCIM) transforms GPS coordinates into spatial representations using hierarchical and Fourier-based positional encoding, enabling the model to understand geographic features from multiple perspectives. The POI Alignment Module (PAM) incorporates POI transition relations into the LLM's semantic space, allowing it to infer global POI relationships and generalize to unseen POIs. Experiments on three real-world datasets demonstrate the state-of-the-art performance of GA-LLM. 

---
# Q${}^2$Forge: Minting Competency Questions and SPARQL Queries for Question-Answering Over Knowledge Graphs 

**Authors**: Yousouf Taghzouti, Franck Michel, Tao Jiang, Louis-Félix Nothias, Fabien Gandon  

**Link**: [PDF](https://arxiv.org/pdf/2505.13572)  

**Abstract**: The SPARQL query language is the standard method to access knowledge graphs (KGs). However, formulating SPARQL queries is a significant challenge for non-expert users, and remains time-consuming for the experienced ones. Best practices recommend to document KGs with competency questions and example queries to contextualise the knowledge they contain and illustrate their potential applications. In practice, however, this is either not the case or the examples are provided in limited numbers. Large Language Models (LLMs) are being used in conversational agents and are proving to be an attractive solution with a wide range of applications, from simple question-answering about common knowledge to generating code in a targeted programming language. However, training and testing these models to produce high quality SPARQL queries from natural language questions requires substantial datasets of question-query pairs. In this paper, we present Q${}^2$Forge that addresses the challenge of generating new competency questions for a KG and corresponding SPARQL queries. It iteratively validates those queries with human feedback and LLM as a judge. Q${}^2$Forge is open source, generic, extensible and modular, meaning that the different modules of the application (CQ generation, query generation and query refinement) can be used separately, as an integrated pipeline, or replaced by alternative services. The result is a complete pipeline from competency question formulation to query evaluation, supporting the creation of reference query sets for any target KG. 

---
# General-Reasoner: Advancing LLM Reasoning Across All Domains 

**Authors**: Xueguang Ma, Qian Liu, Dongfu Jiang, Ge Zhang, Zejun Ma, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.14652)  

**Abstract**: Reinforcement learning (RL) has recently demonstrated strong potential in enhancing the reasoning capabilities of large language models (LLMs). Particularly, the "Zero" reinforcement learning introduced by Deepseek-R1-Zero, enables direct RL training of base LLMs without relying on an intermediate supervised fine-tuning stage. Despite these advancements, current works for LLM reasoning mainly focus on mathematical and coding domains, largely due to data abundance and the ease of answer verification. This limits the applicability and generalization of such models to broader domains, where questions often have diverse answer representations, and data is more scarce. In this paper, we propose General-Reasoner, a novel training paradigm designed to enhance LLM reasoning capabilities across diverse domains. Our key contributions include: (1) constructing a large-scale, high-quality dataset of questions with verifiable answers curated by web crawling, covering a wide range of disciplines; and (2) developing a generative model-based answer verifier, which replaces traditional rule-based verification with the capability of chain-of-thought and context-awareness. We train a series of models and evaluate them on a wide range of datasets covering wide domains like physics, chemistry, finance, electronics etc. Our comprehensive evaluation across these 12 benchmarks (e.g. MMLU-Pro, GPQA, SuperGPQA, TheoremQA, BBEH and MATH AMC) demonstrates that General-Reasoner outperforms existing baseline methods, achieving robust and generalizable reasoning performance while maintaining superior effectiveness in mathematical reasoning tasks. 

---
# Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning 

**Authors**: Haolei Xu, Yuchen Yan, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Shengpei Jiang, Kaitao Song, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14684)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress on mathemati-cal tasks through Chain-of-Thought (CoT) reasoning. However, existing mathematical CoT datasets often suffer from Thought Leaps due to experts omitting intermediate steps, which negatively impacts model learning and generalization. We propose the CoT Thought Leap Bridge Task, which aims to automatically detect leaps and generate missing intermediate reasoning steps to restore the completeness and coherence of CoT. To facilitate this, we constructed a specialized training dataset called ScaleQM+, based on the structured ScaleQuestMath dataset, and trained CoT-Bridge to bridge thought leaps. Through comprehensive experiments on mathematical reasoning benchmarks, we demonstrate that models fine-tuned on bridged datasets consistently outperform those trained on original datasets, with improvements of up to +5.87% on NuminaMath. Our approach effectively enhances distilled data (+3.02%) and provides better starting points for reinforcement learning (+3.1%), functioning as a plug-and-play module compatible with existing optimization techniques. Furthermore, CoT-Bridge demonstrate improved generalization to out-of-domain logical reasoning tasks, confirming that enhancing reasoning completeness yields broadly applicable benefits. 

---
# Language Models use Lookbacks to Track Beliefs 

**Authors**: Nikhil Prakash, Natalie Shapira, Arnab Sen Sharma, Christoph Riedl, Yonatan Belinkov, Tamar Rott Shaham, David Bau, Atticus Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2505.14685)  

**Abstract**: How do language models (LMs) represent characters' beliefs, especially when those beliefs may differ from reality? This question lies at the heart of understanding the Theory of Mind (ToM) capabilities of LMs. We analyze Llama-3-70B-Instruct's ability to reason about characters' beliefs using causal mediation and abstraction. We construct a dataset that consists of simple stories where two characters each separately change the state of two objects, potentially unaware of each other's actions. Our investigation uncovered a pervasive algorithmic pattern that we call a lookback mechanism, which enables the LM to recall important information when it becomes necessary. The LM binds each character-object-state triple together by co-locating reference information about them, represented as their Ordering IDs (OIs) in low rank subspaces of the state token's residual stream. When asked about a character's beliefs regarding the state of an object, the binding lookback retrieves the corresponding state OI and then an answer lookback retrieves the state token. When we introduce text specifying that one character is (not) visible to the other, we find that the LM first generates a visibility ID encoding the relation between the observing and the observed character OIs. In a visibility lookback, this ID is used to retrieve information about the observed character and update the observing character's beliefs. Our work provides insights into the LM's belief tracking mechanisms, taking a step toward reverse-engineering ToM reasoning in LMs. 

---
# Think Only When You Need with Large Hybrid-Reasoning Models 

**Authors**: Lingjie Jiang, Xun Wu, Shaohan Huang, Qingxiu Dong, Zewen Chi, Li Dong, Xingxing Zhang, Tengchao Lv, Lei Cui, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.14631)  

**Abstract**: Recent Large Reasoning Models (LRMs) have shown substantially improved reasoning capabilities over traditional Large Language Models (LLMs) by incorporating extended thinking processes prior to producing final responses. However, excessively lengthy thinking introduces substantial overhead in terms of token consumption and latency, which is particularly unnecessary for simple queries. In this work, we introduce Large Hybrid-Reasoning Models (LHRMs), the first kind of model capable of adaptively determining whether to perform thinking based on the contextual information of user queries. To achieve this, we propose a two-stage training pipeline comprising Hybrid Fine-Tuning (HFT) as a cold start, followed by online reinforcement learning with the proposed Hybrid Group Policy Optimization (HGPO) to implicitly learn to select the appropriate thinking mode. Furthermore, we introduce a metric called Hybrid Accuracy to quantitatively assess the model's capability for hybrid thinking. Extensive experimental results show that LHRMs can adaptively perform hybrid thinking on queries of varying difficulty and type. It outperforms existing LRMs and LLMs in reasoning and general capabilities while significantly improving efficiency. Together, our work advocates for a reconsideration of the appropriate use of extended thinking processes and provides a solid starting point for building hybrid thinking systems. 

---
# Linear Control of Test Awareness Reveals Differential Compliance in Reasoning Models 

**Authors**: Sahar Abdelnabi, Ahmed Salem  

**Link**: [PDF](https://arxiv.org/pdf/2505.14617)  

**Abstract**: Reasoning-focused large language models (LLMs) sometimes alter their behavior when they detect that they are being evaluated, an effect analogous to the Hawthorne phenomenon, which can lead them to optimize for test-passing performance or to comply more readily with harmful prompts if real-world consequences appear absent. We present the first quantitative study of how such "test awareness" impacts model behavior, particularly its safety alignment. We introduce a white-box probing framework that (i) linearly identifies awareness-related activations and (ii) steers models toward or away from test awareness while monitoring downstream performance. We apply our method to different state-of-the-art open-source reasoning LLMs across both realistic and hypothetical tasks. Our results demonstrate that test awareness significantly impact safety alignment, and is different for different models. By providing fine-grained control over this latent effect, our work aims to increase trust in how we perform safety evaluation. 

---
# UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Large Language Models 

**Authors**: Xiaojie Gu, Guangxu Chen, Jungang Li, Jia-Chen Gu, Xuming Hu, Kai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14679)  

**Abstract**: Lifelong learning enables large language models (LLMs) to adapt to evolving information by continually updating their internal knowledge. An ideal system should support efficient, wide-ranging updates while preserving existing capabilities and ensuring reliable deployment. Model editing stands out as a promising solution for this goal, offering a focused and efficient way to revise a model's internal knowledge. Although recent paradigms have made notable progress, they often struggle to meet the demands of practical lifelong adaptation at scale. To bridge this gap, we propose ULTRAEDIT-a fundamentally new editing solution that is training-, subject- and memory-free, making it particularly well-suited for ultra-scalable, real-world lifelong model editing. ULTRAEDIT performs editing through a self-contained process that relies solely on lightweight linear algebra operations to compute parameter shifts, enabling fast and consistent parameter modifications with minimal overhead. To improve scalability in lifelong settings, ULTRAEDIT employs a lifelong normalization strategy that continuously updates feature statistics across turns, allowing it to adapt to distributional shifts and maintain consistency over time. ULTRAEDIT achieves editing speeds over 7x faster than the previous state-of-the-art method-which was also the fastest known approach-while consuming less than 1/3 the VRAM, making it the only method currently capable of editing a 7B LLM on a 24GB consumer-grade GPU. Furthermore, we construct ULTRAEDITBENCH-the largest dataset in the field to date, with over 2M editing pairs-and demonstrate that our method supports up to 1M edits while maintaining high accuracy. Comprehensive experiments on four datasets and six models show that ULTRAEDIT consistently achieves superior performance across diverse model editing scenarios. Our code is available at: this https URL. 

---
# sudoLLM : On Multi-role Alignment of Language Models 

**Authors**: Soumadeep Saha, Akshay Chaturvedi, Joy Mahapatra, Utpal Garain  

**Link**: [PDF](https://arxiv.org/pdf/2505.14607)  

**Abstract**: User authorization-based access privileges are a key feature in many safety-critical systems, but have thus far been absent from the large language model (LLM) realm. In this work, drawing inspiration from such access control systems, we introduce sudoLLM, a novel framework that results in multi-role aligned LLMs, i.e., LLMs that account for, and behave in accordance with, user access rights. sudoLLM injects subtle user-based biases into queries and trains an LLM to utilize this bias signal in order to produce sensitive information if and only if the user is authorized. We present empirical results demonstrating that this approach shows substantially improved alignment, generalization, and resistance to prompt-based jailbreaking attacks. The persistent tension between the language modeling objective and safety alignment, which is often exploited to jailbreak LLMs, is somewhat resolved with the aid of the injected bias signal. Our framework is meant as an additional security layer, and complements existing guardrail mechanisms for enhanced end-to-end safety with LLMs. 

---
# Toward Reliable Biomedical Hypothesis Generation: Evaluating Truthfulness and Hallucination in Large Language Models 

**Authors**: Guangzhi Xiong, Eric Xie, Corey Williams, Myles Kim, Amir Hassan Shariatmadari, Sikun Guo, Stefan Bekiranov, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14599)  

**Abstract**: Large language models (LLMs) have shown significant potential in scientific disciplines such as biomedicine, particularly in hypothesis generation, where they can analyze vast literature, identify patterns, and suggest research directions. However, a key challenge lies in evaluating the truthfulness of generated hypotheses, as verifying their accuracy often requires substantial time and resources. Additionally, the hallucination problem in LLMs can lead to the generation of hypotheses that appear plausible but are ultimately incorrect, undermining their reliability. To facilitate the systematic study of these challenges, we introduce TruthHypo, a benchmark for assessing the capabilities of LLMs in generating truthful biomedical hypotheses, and KnowHD, a knowledge-based hallucination detector to evaluate how well hypotheses are grounded in existing knowledge. Our results show that LLMs struggle to generate truthful hypotheses. By analyzing hallucinations in reasoning steps, we demonstrate that the groundedness scores provided by KnowHD serve as an effective metric for filtering truthful hypotheses from the diverse outputs of LLMs. Human evaluations further validate the utility of KnowHD in identifying truthful hypotheses and accelerating scientific discovery. Our data and source code are available at this https URL. 

---
# TRATES: Trait-Specific Rubric-Assisted Cross-Prompt Essay Scoring 

**Authors**: Sohaila Eltanbouly, Salam Albatarni, Tamer Elsayed  

**Link**: [PDF](https://arxiv.org/pdf/2505.14577)  

**Abstract**: Research on holistic Automated Essay Scoring (AES) is long-dated; yet, there is a notable lack of attention for assessing essays according to individual traits. In this work, we propose TRATES, a novel trait-specific and rubric-based cross-prompt AES framework that is generic yet specific to the underlying trait. The framework leverages a Large Language Model (LLM) that utilizes the trait grading rubrics to generate trait-specific features (represented by assessment questions), then assesses those features given an essay. The trait-specific features are eventually combined with generic writing-quality and prompt-specific features to train a simple classical regression model that predicts trait scores of essays from an unseen prompt. Experiments show that TRATES achieves a new state-of-the-art performance across all traits on a widely-used dataset, with the generated LLM-based features being the most significant. 

---
# Reward Reasoning Model 

**Authors**: Jiaxin Guo, Zewen Chi, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.14674)  

**Abstract**: Reward models play a critical role in guiding large language models toward outputs that align with human expectations. However, an open challenge remains in effectively utilizing test-time compute to enhance reward model performance. In this work, we introduce Reward Reasoning Models (RRMs), which are specifically designed to execute a deliberate reasoning process before generating final rewards. Through chain-of-thought reasoning, RRMs leverage additional test-time compute for complex queries where appropriate rewards are not immediately apparent. To develop RRMs, we implement a reinforcement learning framework that fosters self-evolved reward reasoning capabilities without requiring explicit reasoning traces as training data. Experimental results demonstrate that RRMs achieve superior performance on reward modeling benchmarks across diverse domains. Notably, we show that RRMs can adaptively exploit test-time compute to further improve reward accuracy. The pretrained reward reasoning models are available at this https URL. 

---
# Can Pruning Improve Reasoning? Revisiting Long-CoT Compression with Capability in Mind for Better Reasoning 

**Authors**: Shangziqi Zhao, Jiahao Yuan, Guisong Yang, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2505.14582)  

**Abstract**: Long chain-of-thought (Long-CoT) reasoning improves accuracy in LLMs, yet its verbose, self-reflective style often hinders effective distillation into small language models (SLMs). We revisit Long-CoT compression through the lens of capability alignment and ask: Can pruning improve reasoning? We propose Prune-on-Logic, a structure-aware framework that transforms Long-CoT into logic graphs and selectively prunes low-utility reasoning steps under self-verification constraints. Through systematic analysis across three pruning strategies -- targeting entire chains, core reasoning, and verification -- we find that pruning verification steps yields consistent accuracy gains while reducing inference cost, outperforming token-level baselines and uncompressed fine-tuning. In contrast, pruning reasoning or all-chain steps degrades performance, revealing that small models benefit not from shorter CoTs, but from semantically leaner ones. Our findings highlight pruning as a structural optimization strategy for aligning CoT reasoning with SLM capacity. 

---
# Breaking Bad Tokens: Detoxification of LLMs Using Sparse Autoencoders 

**Authors**: Agam Goyal, Vedant Rathi, William Yeh, Yian Wang, Yuen Chen, Hari Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2505.14536)  

**Abstract**: Large language models (LLMs) are now ubiquitous in user-facing applications, yet they still generate undesirable toxic outputs, including profanity, vulgarity, and derogatory remarks. Although numerous detoxification methods exist, most apply broad, surface-level fixes and can therefore easily be circumvented by jailbreak attacks. In this paper we leverage sparse autoencoders (SAEs) to identify toxicity-related directions in the residual stream of models and perform targeted activation steering using the corresponding decoder vectors. We introduce three tiers of steering aggressiveness and evaluate them on GPT-2 Small and Gemma-2-2B, revealing trade-offs between toxicity reduction and language fluency. At stronger steering strengths, these causal interventions surpass competitive baselines in reducing toxicity by up to 20%, though fluency can degrade noticeably on GPT-2 Small depending on the aggressiveness. Crucially, standard NLP benchmark scores upon steering remain stable, indicating that the model's knowledge and general abilities are preserved. We further show that feature-splitting in wider SAEs hampers safety interventions, underscoring the importance of disentangled feature learning. Our findings highlight both the promise and the current limitations of SAE-based causal interventions for LLM detoxification, further suggesting practical guidelines for safer language-model deployment. 

---
# Internal Chain-of-Thought: Empirical Evidence for Layer-wise Subtask Scheduling in LLMs 

**Authors**: Zhipeng Yang, Junzhuo Li, Siyu Xia, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14530)  

**Abstract**: We show that large language models (LLMs) exhibit an $\textit{internal chain-of-thought}$: they sequentially decompose and execute composite tasks layer-by-layer. Two claims ground our study: (i) distinct subtasks are learned at different network depths, and (ii) these subtasks are executed sequentially across layers. On a benchmark of 15 two-step composite tasks, we employ layer-from context-masking and propose a novel cross-task patching method, confirming (i). To examine claim (ii), we apply LogitLens to decode hidden states, revealing a consistent layerwise execution pattern. We further replicate our analysis on the real-world $\text{TRACE}$ benchmark, observing the same stepwise dynamics. Together, our results enhance LLMs transparency by showing their capacity to internally plan and execute subtasks (or instructions), opening avenues for fine-grained, instruction-level activation steering. 

---
# KORGym: A Dynamic Game Platform for LLM Reasoning Evaluation 

**Authors**: Jiajun Shi, Jian Yang, Jiaheng Liu, Xingyuan Bu, Jiangjie Chen, Junting Zhou, Kaijing Ma, Zhoufutu Wen, Bingli Wang, Yancheng He, Liang Song, Hualei Zhu, Shilong Li, Xingjian Wang, Wei Zhang, Ruibin Yuan, Yifan Yao, Wenjun Yang, Yunli Wang, Siyuan Fang, Siyu Yuan, Qianyu He, Xiangru Tang, Yingshui Tan, Wangchunshu Zhou, Zhaoxiang Zhang, Zhoujun Li, Wenhao Huang, Ge Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14552)  

**Abstract**: Recent advancements in large language models (LLMs) underscore the need for more comprehensive evaluation methods to accurately assess their reasoning capabilities. Existing benchmarks are often domain-specific and thus cannot fully capture an LLM's general reasoning potential. To address this limitation, we introduce the Knowledge Orthogonal Reasoning Gymnasium (KORGym), a dynamic evaluation platform inspired by KOR-Bench and Gymnasium. KORGym offers over fifty games in either textual or visual formats and supports interactive, multi-turn assessments with reinforcement learning scenarios. Using KORGym, we conduct extensive experiments on 19 LLMs and 8 VLMs, revealing consistent reasoning patterns within model families and demonstrating the superior performance of closed-source models. Further analysis examines the effects of modality, reasoning strategies, reinforcement learning techniques, and response length on model performance. We expect KORGym to become a valuable resource for advancing LLM reasoning research and developing evaluation methodologies suited to complex, interactive environments. 

---
# ModRWKV: Transformer Multimodality in Linear Time 

**Authors**: Jiale Kang, Ziyin Yue, Qingyu Yin, Jiang Rui, Weile Li, Zening Lu, Zhouran Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.14505)  

**Abstract**: Currently, most multimodal studies are based on large language models (LLMs) with quadratic-complexity Transformer architectures. While linear models like RNNs enjoy low inference costs, their application has been largely limited to the text-only modality. This work explores the capabilities of modern RNN architectures in multimodal contexts. We propose ModRWKV-a decoupled multimodal framework built upon the RWKV7 architecture as its LLM backbone-which achieves multi-source information fusion through dynamically adaptable heterogeneous modality encoders. We designed the multimodal modules in ModRWKV with an extremely lightweight architecture and, through extensive experiments, identified a configuration that achieves an optimal balance between performance and computational efficiency. ModRWKV leverages the pretrained weights of the RWKV7 LLM for initialization, which significantly accelerates multimodal training. Comparative experiments with different pretrained checkpoints further demonstrate that such initialization plays a crucial role in enhancing the model's ability to understand multimodal signals. Supported by extensive experiments, we conclude that modern RNN architectures present a viable alternative to Transformers in the domain of multimodal large language models (MLLMs). Furthermore, we identify the optimal configuration of the ModRWKV architecture through systematic exploration. 

---
# Enhanced Multimodal Aspect-Based Sentiment Analysis by LLM-Generated Rationales 

**Authors**: Jun Cao, Jiyi Li, Ziwei Yang, Renjie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14499)  

**Abstract**: There has been growing interest in Multimodal Aspect-Based Sentiment Analysis (MABSA) in recent years. Existing methods predominantly rely on pre-trained small language models (SLMs) to collect information related to aspects and sentiments from both image and text, with an aim to align these two modalities. However, small SLMs possess limited capacity and knowledge, often resulting in inaccurate identification of meaning, aspects, sentiments, and their interconnections in textual and visual data. On the other hand, Large language models (LLMs) have shown exceptional capabilities in various tasks by effectively exploring fine-grained information in multimodal data. However, some studies indicate that LLMs still fall short compared to fine-tuned small models in the field of ABSA. Based on these findings, we propose a novel framework, termed LRSA, which combines the decision-making capabilities of SLMs with additional information provided by LLMs for MABSA. Specifically, we inject explanations generated by LLMs as rationales into SLMs and employ a dual cross-attention mechanism for enhancing feature interaction and fusion, thereby augmenting the SLMs' ability to identify aspects and sentiments. We evaluated our method using two baseline models, numerous experiments highlight the superiority of our approach on three widely-used benchmarks, indicating its generalizability and applicability to most pre-trained models for MABSA. 

---
# MoMoE: Mixture of Moderation Experts Framework for AI-Assisted Online Governance 

**Authors**: Agam Goyal, Xianyang Zhan, Yilun Chen, Koustuv Saha, Eshwar Chandrasekharan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14483)  

**Abstract**: Large language models (LLMs) have shown great potential in flagging harmful content in online communities. Yet, existing approaches for moderation require a separate model for every community and are opaque in their decision-making, limiting real-world adoption. We introduce Mixture of Moderation Experts (MoMoE), a modular, cross-community framework that adds post-hoc explanations to scalable content moderation. MoMoE orchestrates four operators -- Allocate, Predict, Aggregate, Explain -- and is instantiated as seven community-specialized experts (MoMoE-Community) and five norm-violation experts (MoMoE-NormVio). On 30 unseen subreddits, the best variants obtain Micro-F1 scores of 0.72 and 0.67, respectively, matching or surpassing strong fine-tuned baselines while consistently producing concise and reliable explanations. Although community-specialized experts deliver the highest peak accuracy, norm-violation experts provide steadier performance across domains. These findings show that MoMoE yields scalable, transparent moderation without needing per-community fine-tuning. More broadly, they suggest that lightweight, explainable expert ensembles can guide future NLP and HCI research on trustworthy human-AI governance of online communities. 

---
# Creative Preference Optimization 

**Authors**: Mete Ismayilzada, Antonio Laverghetta Jr., Simone A. Luchini, Reet Patel, Antoine Bosselut, Lonneke van der Plas, Roger Beaty  

**Link**: [PDF](https://arxiv.org/pdf/2505.14442)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive performance across natural language generation tasks, their ability to generate truly creative content-characterized by novelty, diversity, surprise, and quality-remains limited. Existing methods for enhancing LLM creativity often focus narrowly on diversity or specific tasks, failing to address creativity's multifaceted nature in a generalizable way. In this work, we propose Creative Preference Optimization (CrPO), a novel alignment method that injects signals from multiple creativity dimensions into the preference optimization objective in a modular fashion. We train and evaluate creativity-augmented versions of several models using CrPO and MuCE, a new large-scale human preference dataset spanning over 200,000 human-generated responses and ratings from more than 30 psychological creativity assessments. Our models outperform strong baselines, including GPT-4o, on both automated and human evaluations, producing more novel, diverse, and surprising generations while maintaining high output quality. Additional evaluations on NoveltyBench further confirm the generalizability of our approach. Together, our results demonstrate that directly optimizing for creativity within preference frameworks is a promising direction for advancing the creative capabilities of LLMs without compromising output quality. 

---
# Attributional Safety Failures in Large Language Models under Code-Mixed Perturbations 

**Authors**: Somnath Banerjee, Pratyush Chatterjee, Shanu Kumar, Sayan Layek, Parag Agrawal, Rima Hazra, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.14469)  

**Abstract**: Recent advancements in LLMs have raised significant safety concerns, particularly when dealing with code-mixed inputs and outputs. Our study systematically investigates the increased susceptibility of LLMs to produce unsafe outputs from code-mixed prompts compared to monolingual English prompts. Utilizing explainability methods, we dissect the internal attribution shifts causing model's harmful behaviors. In addition, we explore cultural dimensions by distinguishing between universally unsafe and culturally-specific unsafe queries. This paper presents novel experimental insights, clarifying the mechanisms driving this phenomenon. 

---
# From Templates to Natural Language: Generalization Challenges in Instruction-Tuned LLMs for Spatial Reasoning 

**Authors**: Chalamalasetti Kranti, Sherzod Hakimov, David Schlangen  

**Link**: [PDF](https://arxiv.org/pdf/2505.14425)  

**Abstract**: Instruction-tuned large language models (LLMs) have shown strong performance on a variety of tasks; however, generalizing from synthetic to human-authored instructions in grounded environments remains a challenge for them. In this work, we study generalization challenges in spatial grounding tasks where models interpret and translate instructions for building object arrangements on a $2.5$D grid. We fine-tune LLMs using only synthetic instructions and evaluate their performance on a benchmark dataset containing both synthetic and human-written instructions. Our results reveal that while models generalize well on simple tasks, their performance degrades significantly on more complex tasks. We present a detailed error analysis of the gaps in instruction generalization. 

---
# Log-Augmented Generation: Scaling Test-Time Reasoning with Reusable Computation 

**Authors**: Peter Baile Chen, Yi Zhang, Dan Roth, Samuel Madden, Jacob Andreas, Michael Cafarella  

**Link**: [PDF](https://arxiv.org/pdf/2505.14398)  

**Abstract**: While humans naturally learn and adapt from past experiences, large language models (LLMs) and their agentic counterparts struggle to retain reasoning from previous tasks and apply them in future contexts. To address this limitation, we propose a novel framework, log-augmented generation (LAG) that directly reuses prior computation and reasoning from past logs at test time to enhance model's ability to learn from previous tasks and perform better on new, unseen challenges, all while keeping the system efficient and scalable. Specifically, our system represents task logs using key-value (KV) caches, encoding the full reasoning context of prior tasks while storing KV caches for only a selected subset of tokens. When a new task arises, LAG retrieves the KV values from relevant logs to augment generation. Our approach differs from reflection-based memory mechanisms by directly reusing prior reasoning and computations without requiring additional steps for knowledge extraction or distillation. Our method also goes beyond existing KV caching techniques, which primarily target efficiency gains rather than improving accuracy. Experiments on knowledge- and reasoning-intensive datasets demonstrate that our method significantly outperforms standard agentic systems that do not utilize logs, as well as existing solutions based on reflection and KV cache techniques. 

---
# MUG-Eval: A Proxy Evaluation Framework for Multilingual Generation Capabilities in Any Language 

**Authors**: Seyoung Song, Seogyeong Jeong, Eunsu Kim, Jiho Jin, Dongkwan Kim, Jay Shin, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2505.14395)  

**Abstract**: Evaluating text generation capabilities of large language models (LLMs) is challenging, particularly for low-resource languages where methods for direct assessment are scarce. We propose MUG-Eval, a novel framework that evaluates LLMs' multilingual generation capabilities by transforming existing benchmarks into conversational tasks and measuring the LLMs' accuracies on those tasks. We specifically designed these conversational tasks to require effective communication in the target language. Then, we simply use task success rate as a proxy of successful conversation generation. Our approach offers two key advantages: it is independent of language-specific NLP tools or annotated datasets, which are limited for most languages, and it does not rely on LLMs-as-judges, whose evaluation quality degrades outside a few high-resource languages. We evaluate 8 LLMs across 30 languages spanning high, mid, and low-resource categories, and we find that MUG-Eval correlates strongly with established benchmarks ($r$ > 0.75) while enabling standardized comparisons across languages and models. Our framework provides a robust and resource-efficient solution for evaluating multilingual generation that can be extended to thousands of languages. 

---
# Hidden Ghost Hand: Unveiling Backdoor Vulnerabilities in MLLM-Powered Mobile GUI Agents 

**Authors**: Pengzhou Cheng, Haowen Hu, Zheng Wu, Zongru Wu, Tianjie Ju, Daizong Ding, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14418)  

**Abstract**: Graphical user interface (GUI) agents powered by multimodal large language models (MLLMs) have shown greater promise for human-interaction. However, due to the high fine-tuning cost, users often rely on open-source GUI agents or APIs offered by AI providers, which introduces a critical but underexplored supply chain threat: backdoor attacks. In this work, we first unveil that MLLM-powered GUI agents naturally expose multiple interaction-level triggers, such as historical steps, environment states, and task progress. Based on this observation, we introduce AgentGhost, an effective and stealthy framework for red-teaming backdoor attacks. Specifically, we first construct composite triggers by combining goal and interaction levels, allowing GUI agents to unintentionally activate backdoors while ensuring task utility. Then, we formulate backdoor injection as a Min-Max optimization problem that uses supervised contrastive learning to maximize the feature difference across sample classes at the representation space, improving flexibility of the backdoor. Meanwhile, it adopts supervised fine-tuning to minimize the discrepancy between backdoor and clean behavior generation, enhancing effectiveness and utility. Extensive evaluations of various agent models in two established mobile benchmarks show that AgentGhost is effective and generic, with attack accuracy that reaches 99.7\% on three attack objectives, and shows stealthiness with only 1\% utility degradation. Furthermore, we tailor a defense method against AgentGhost that reduces the attack accuracy to 22.1\%. Our code is available at \texttt{anonymous}. 

---
# Dual Decomposition of Weights and Singular Value Low Rank Adaptation 

**Authors**: Jialong Han, Si Zhang, Ke Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14367)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) has emerged as a critical paradigm for adapting Large Language Models (LLMs) to downstream tasks, among which Low-rank Adaptation (LoRA) represents one of the most widely adopted methodologies. However, existing LoRA-based approaches exhibit two fundamental limitations: unstable training dynamics and inefficient knowledge transfer from pre-trained models, both stemming from random initialization of adapter parameters. To overcome these challenges, we propose DuDe, a novel approach that decomposes weight matrices into magnitude and direction components, employing Singular Value Decomposition (SVD) for principled initialization. Our comprehensive evaluation demonstrates DuDe's superior performance and robustness, achieving up to 48.35\% accuracy on MMLU and 62.53\% ($\pm$ 1.59) accuracy on GSM8K. Our theoretical analysis and empirical validation collectively demonstrate that DuDe's decomposition strategy enhances optimization stability and better preserves pre-trained representations, particularly for domain-specific tasks requiring specialized knowledge. The combination of robust empirical performance and rigorous theoretical foundations establishes DuDe as a significant contribution to PEFT methodologies for LLMs. 

---
# Pierce the Mists, Greet the Sky: Decipher Knowledge Overshadowing via Knowledge Circuit Analysis 

**Authors**: Haoming Huang, Yibo Yan, Jiahao Huo, Xin Zou, Xinfeng Li, Kun Wang, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14406)  

**Abstract**: Large Language Models (LLMs), despite their remarkable capabilities, are hampered by hallucinations. A particularly challenging variant, knowledge overshadowing, occurs when one piece of activated knowledge inadvertently masks another relevant piece, leading to erroneous outputs even with high-quality training data. Current understanding of overshadowing is largely confined to inference-time observations, lacking deep insights into its origins and internal mechanisms during model training. Therefore, we introduce PhantomCircuit, a novel framework designed to comprehensively analyze and detect knowledge overshadowing. By innovatively employing knowledge circuit analysis, PhantomCircuit dissects the internal workings of attention heads, tracing how competing knowledge pathways contribute to the overshadowing phenomenon and its evolution throughout the training process. Extensive experiments demonstrate PhantomCircuit's effectiveness in identifying such instances, offering novel insights into this elusive hallucination and providing the research community with a new methodological lens for its potential mitigation. 

---
# A MIND for Reasoning: Meta-learning for In-context Deduction 

**Authors**: Leonardo Bertolazzi, Manuel Vargas Guzmán, Raffaella Bernardi, Maciej Malicki, Jakub Szymanik  

**Link**: [PDF](https://arxiv.org/pdf/2505.14313)  

**Abstract**: Large language models (LLMs) are increasingly evaluated on formal tasks, where strong reasoning abilities define the state of the art. However, their ability to generalize to out-of-distribution problems remains limited. In this paper, we investigate how LLMs can achieve a systematic understanding of deductive rules. Our focus is on the task of identifying the appropriate subset of premises within a knowledge base needed to derive a given hypothesis. To tackle this challenge, we propose Meta-learning for In-context Deduction (MIND), a novel few-shot meta-learning fine-tuning approach. The goal of MIND is to enable models to generalize more effectively to unseen knowledge bases and to systematically apply inference rules. Our results show that MIND significantly improves generalization in small LMs ranging from 1.5B to 7B parameters. The benefits are especially pronounced in smaller models and low-data settings. Remarkably, small models fine-tuned with MIND outperform state-of-the-art LLMs, such as GPT-4o and o3-mini, on this task. 

---
# OSoRA: Output-Dimension and Singular-Value Initialized Low-Rank Adaptation 

**Authors**: Jialong Han, Si Zhang, Ke Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14350)  

**Abstract**: Fine-tuning Large Language Models (LLMs) has become increasingly challenging due to their massive scale and associated computational costs. Parameter-Efficient Fine-Tuning (PEFT) methodologies have been proposed as computational alternatives; however, their implementations still require significant resources. In this paper, we present OSoRA (Output-Dimension and Singular-Value Initialized Low-Rank Adaptation), a novel PEFT method for LLMs. OSoRA extends Low-Rank Adaptation (LoRA) by integrating Singular Value Decomposition (SVD) with learnable scaling vectors in a unified framework. It first performs an SVD of pre-trained weight matrices, then optimizes an output-dimension vector during training, while keeping the corresponding singular vector matrices frozen. OSoRA substantially reduces computational resource requirements by minimizing the number of trainable parameters during fine-tuning. Comprehensive evaluations across mathematical reasoning, common sense reasoning, and other benchmarks demonstrate that OSoRA achieves comparable or superior performance to state-of-the-art methods like LoRA and VeRA, while maintaining a linear parameter scaling even as the rank increases to higher dimensions. Our ablation studies further confirm that jointly training both the singular values and the output-dimension vector is critical for optimal performance. 

---
# JOLT-SQL: Joint Loss Tuning of Text-to-SQL with Confusion-aware Noisy Schema Sampling 

**Authors**: Jinwang Song, Hongying Zan, Kunli Zhang, Lingling Mu, Yingjie Han, Haobo Hua, Min Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.14305)  

**Abstract**: Text-to-SQL, which maps natural language to SQL queries, has benefited greatly from recent advances in Large Language Models (LLMs). While LLMs offer various paradigms for this task, including prompting and supervised fine-tuning (SFT), SFT approaches still face challenges such as complex multi-stage pipelines and poor robustness to noisy schema information. To address these limitations, we present JOLT-SQL, a streamlined single-stage SFT framework that jointly optimizes schema linking and SQL generation via a unified loss. JOLT-SQL employs discriminative schema linking, enhanced by local bidirectional attention, alongside a confusion-aware noisy schema sampling strategy with selective attention to improve robustness under noisy schema conditions. Experiments on the Spider and BIRD benchmarks demonstrate that JOLT-SQL achieves state-of-the-art execution accuracy among comparable-size open-source models, while significantly improving both training and inference efficiency. 

---
# HausaNLP: Current Status, Challenges and Future Directions for Hausa Natural Language Processing 

**Authors**: Shamsuddeen Hassan Muhammad, Ibrahim Said Ahmad, Idris Abdulmumin, Falalu Ibrahim Lawan, Babangida Sani, Sukairaj Hafiz Imam, Yusuf Aliyu, Sani Abdullahi Sani, Ali Usman Umar, Kenneth Church, Vukosi Marivate  

**Link**: [PDF](https://arxiv.org/pdf/2505.14311)  

**Abstract**: Hausa Natural Language Processing (NLP) has gained increasing attention in recent years, yet remains understudied as a low-resource language despite having over 120 million first-language (L1) and 80 million second-language (L2) speakers worldwide. While significant advances have been made in high-resource languages, Hausa NLP faces persistent challenges, including limited open-source datasets and inadequate model representation. This paper presents an overview of the current state of Hausa NLP, systematically examining existing resources, research contributions, and gaps across fundamental NLP tasks: text classification, machine translation, named entity recognition, speech recognition, and question answering. We introduce HausaNLP (this https URL), a curated catalog that aggregates datasets, tools, and research works to enhance accessibility and drive further development. Furthermore, we discuss challenges in integrating Hausa into large language models (LLMs), addressing issues of suboptimal tokenization and dialectal variation. Finally, we propose strategic research directions emphasizing dataset expansion, improved language modeling approaches, and strengthened community collaboration to advance Hausa NLP. Our work provides both a foundation for accelerating Hausa NLP progress and valuable insights for broader multilingual NLP research. 

---
# YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering 

**Authors**: Jennifer D'Souza, Hamed Babaei Giglou, Quentin Münch  

**Link**: [PDF](https://arxiv.org/pdf/2505.14279)  

**Abstract**: Large Language Models (LLMs) drive scientific question-answering on modern search engines, yet their evaluation robustness remains underexplored. We introduce YESciEval, an open-source framework that combines fine-grained rubric-based assessment with reinforcement learning to mitigate optimism bias in LLM evaluators. We release multidisciplinary scienceQ&A datasets, including adversarial variants, with evaluation scores from multiple LLMs. Independent of proprietary models and human feedback, our approach enables scalable, cost-free evaluation. By advancing reliable LLM-as-a-judge models, this work supports AI alignment and fosters robust, transparent evaluation essential for scientific inquiry and artificial general intelligence. 

---
# FuxiMT: Sparsifying Large Language Models for Chinese-Centric Multilingual Machine Translation 

**Authors**: Shaolin Zhu, Tianyu Dong, Bo Li, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.14256)  

**Abstract**: In this paper, we present FuxiMT, a novel Chinese-centric multilingual machine translation model powered by a sparsified large language model (LLM). We adopt a two-stage strategy to train FuxiMT. We first pre-train the model on a massive Chinese corpus and then conduct multilingual fine-tuning on a large parallel dataset encompassing 65 languages. FuxiMT incorporates Mixture-of-Experts (MoEs) and employs a curriculum learning strategy for robust performance across various resource levels. Experimental results demonstrate that FuxiMT significantly outperforms strong baselines, including state-of-the-art LLMs and machine translation models, particularly under low-resource scenarios. Furthermore, FuxiMT exhibits remarkable zero-shot translation capabilities for unseen language pairs, indicating its potential to bridge communication gaps where parallel data are scarce or unavailable. 

---
# ABBA: Highly Expressive Hadamard Product Adaptation for Large Language Models 

**Authors**: Raghav Singhal, Kaustubh Ponkshe, Rohit Vartak, Praneeth Vepakomma  

**Link**: [PDF](https://arxiv.org/pdf/2505.14238)  

**Abstract**: Large Language Models have demonstrated strong performance across a wide range of tasks, but adapting them efficiently to new domains remains a key challenge. Parameter-Efficient Fine-Tuning (PEFT) methods address this by introducing lightweight, trainable modules while keeping most pre-trained weights fixed. The prevailing approach, LoRA, models updates using a low-rank decomposition, but its expressivity is inherently constrained by the rank. Recent methods like HiRA aim to increase expressivity by incorporating a Hadamard product with the frozen weights, but still rely on the structure of the pre-trained model. We introduce ABBA, a new PEFT architecture that reparameterizes the update as a Hadamard product of two independently learnable low-rank matrices. In contrast to prior work, ABBA fully decouples the update from the pre-trained weights, enabling both components to be optimized freely. This leads to significantly higher expressivity under the same parameter budget. We formally analyze ABBA's expressive capacity and validate its advantages through matrix reconstruction experiments. Empirically, ABBA achieves state-of-the-art results on arithmetic and commonsense reasoning benchmarks, consistently outperforming existing PEFT methods by a significant margin across multiple models. Our code is publicly available at: this https URL. 

---
# Universal Acoustic Adversarial Attacks for Flexible Control of Speech-LLMs 

**Authors**: Rao Ma, Mengjie Qian, Vyas Raina, Mark Gales, Kate Knill  

**Link**: [PDF](https://arxiv.org/pdf/2505.14286)  

**Abstract**: The combination of pre-trained speech encoders with large language models has enabled the development of speech LLMs that can handle a wide range of spoken language processing tasks. While these models are powerful and flexible, this very flexibility may make them more vulnerable to adversarial attacks. To examine the extent of this problem, in this work we investigate universal acoustic adversarial attacks on speech LLMs. Here a fixed, universal, adversarial audio segment is prepended to the original input audio. We initially investigate attacks that cause the model to either produce no output or to perform a modified task overriding the original prompt. We then extend the nature of the attack to be selective so that it activates only when specific input attributes, such as a speaker gender or spoken language, are present. Inputs without the targeted attribute should be unaffected, allowing fine-grained control over the model outputs. Our findings reveal critical vulnerabilities in Qwen2-Audio and Granite-Speech and suggest that similar speech LLMs may be susceptible to universal adversarial attacks. This highlights the need for more robust training strategies and improved resistance to adversarial attacks. 

---
# "Haet Bhasha aur Diskrimineshun": Phonetic Perturbations in Code-Mixed Hinglish to Red-Team LLMs 

**Authors**: Darpan Aswal, Siddharth D Jaiswal  

**Link**: [PDF](https://arxiv.org/pdf/2505.14226)  

**Abstract**: Large Language Models (LLMs) have become increasingly powerful, with multilingual and multimodal capabilities improving by the day. These models are being evaluated through audits, alignment studies and red-teaming efforts to expose model vulnerabilities towards generating harmful, biased and unfair content. Existing red-teaming efforts have previously focused on the English language, using fixed template-based attacks; thus, models continue to be susceptible to multilingual jailbreaking strategies, especially in the multimodal context. In this study, we introduce a novel strategy that leverages code-mixing and phonetic perturbations to jailbreak LLMs for both text and image generation tasks. We also introduce two new jailbreak strategies that show higher effectiveness than baseline strategies. Our work presents a method to effectively bypass safety filters in LLMs while maintaining interpretability by applying phonetic misspellings to sensitive words in code-mixed prompts. Our novel prompts achieve a 99% Attack Success Rate for text generation and 78% for image generation, with Attack Relevance Rate of 100% for text generation and 95% for image generation when using the phonetically perturbed code-mixed prompts. Our interpretability experiments reveal that phonetic perturbations impact word tokenization, leading to jailbreak success. Our study motivates increasing the focus towards more generalizable safety alignment for multilingual multimodal models, especially in real-world settings wherein prompts can have misspelt words. 

---
# Automatic Dataset Generation for Knowledge Intensive Question Answering Tasks 

**Authors**: Sizhe Yuen, Ting Su, Ziyang Wang, Yali Du, Adam J. Sobey  

**Link**: [PDF](https://arxiv.org/pdf/2505.14212)  

**Abstract**: A question-answering (QA) system is to search suitable answers within a knowledge base. Current QA systems struggle with queries requiring complex reasoning or real-time knowledge integration. They are often supplemented with retrieval techniques on a data source such as Retrieval-Augmented Generation (RAG). However, RAG continues to face challenges in handling complex reasoning and logical connections between multiple sources of information. A novel approach for enhancing Large Language Models (LLMs) in knowledge-intensive QA tasks is presented through the automated generation of context-based QA pairs. This methodology leverages LLMs to create fine-tuning data, reducing reliance on human labelling and improving model comprehension and reasoning capabilities. The proposed system includes an automated QA generator and a model fine-tuner, evaluated using perplexity, ROUGE, BLEU, and BERTScore. Comprehensive experiments demonstrate improvements in logical coherence and factual accuracy, with implications for developing adaptable Artificial Intelligence (AI) systems. Mistral-7b-v0.3 outperforms Llama-3-8b with BERT F1, BLEU, and ROUGE scores 0.858, 0.172, and 0.260 of for the LLM generated QA pairs compared to scores of 0.836, 0.083, and 0.139 for the human annotated QA pairs. 

---
# Think-J: Learning to Think for Generative LLM-as-a-Judge 

**Authors**: Hui Huang, Yancheng He, Hongli Zhou, Rui Zhang, Wei Liu, Weixun Wang, Wenbo Su, Bo Zheng, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14268)  

**Abstract**: LLM-as-a-Judge refers to the automatic modeling of preferences for responses generated by Large Language Models (LLMs), which is of significant importance for both LLM evaluation and reward modeling. Although generative LLMs have made substantial progress in various tasks, their performance as LLM-Judge still falls short of expectations. In this work, we propose Think-J, which improves generative LLM-as-a-Judge by learning how to think. We first utilized a small amount of curated data to develop the model with initial judgment thinking capabilities. Subsequently, we optimize the judgment thinking traces based on reinforcement learning (RL). We propose two methods for judgment thinking optimization, based on offline and online RL, respectively. The offline RL requires training a critic model to construct positive and negative examples for learning. The online method defines rule-based reward as feedback for optimization. Experimental results showed that our approach can significantly enhance the evaluation capability of generative LLM-Judge, surpassing both generative and classifier-based LLM-Judge without requiring extra human annotations. 

---
# Tokenization Constraints in LLMs: A Study of Symbolic and Arithmetic Reasoning Limits 

**Authors**: Xiang Zhang, Juntai Cao, Jiaqi Wei, Yiwei Xu, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2505.14178)  

**Abstract**: Tokenization is the first - and often underappreciated - layer of computation in language models. While Chain-of-Thought (CoT) prompting enables transformer models to approximate recurrent computation by externalizing intermediate steps, we show that the success of such reasoning is fundamentally bounded by the structure of tokenized inputs. This work presents a theoretical and empirical investigation into how tokenization schemes, particularly subword-based methods like byte-pair encoding (BPE), impede symbolic computation by merging or obscuring atomic reasoning units. We introduce the notion of Token Awareness to formalize how poor token granularity disrupts logical alignment and prevents models from generalizing symbolic procedures. Through systematic evaluation on arithmetic and symbolic tasks, we demonstrate that token structure dramatically affect reasoning performance, causing failure even with CoT, while atomically-aligned formats unlock strong generalization, allowing small models (e.g., GPT-4o-mini) to outperform larger systems (e.g., o1) in structured reasoning. Our findings reveal that symbolic reasoning ability in LLMs is not purely architectural, but deeply conditioned on token-level representations. 

---
# The Strawberry Problem: Emergence of Character-level Understanding in Tokenized Language Models 

**Authors**: Adrian Cosma, Stefan Ruseti, Emilian Radoi, Mihai Dascalu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14172)  

**Abstract**: Despite their remarkable progress across diverse domains, Large Language Models (LLMs) consistently fail at simple character-level tasks, such as counting letters in words, due to a fundamental limitation: tokenization. In this work, we frame this limitation as a problem of low mutual information and analyze it in terms of concept emergence. Using a suite of 19 synthetic tasks that isolate character-level reasoning in a controlled setting, we show that such capabilities emerge slowly, suddenly, and only late in training. We further show that percolation-based models of concept emergence explain these patterns, suggesting that learning character composition is not fundamentally different from learning commonsense knowledge. To address this bottleneck, we propose a lightweight architectural modification that significantly improves character-level reasoning while preserving the inductive advantages of subword models. Together, our results bridge low-level perceptual gaps in tokenized LMs and provide a principled framework for understanding and mitigating their structural blind spots. We make our code publicly available. 

---
# QA-prompting: Improving Summarization with Large Language Models using Question-Answering 

**Authors**: Neelabh Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2505.14347)  

**Abstract**: Language Models (LMs) have revolutionized natural language processing, enabling high-quality text generation through prompting and in-context learning. However, models often struggle with long-context summarization due to positional biases, leading to suboptimal extraction of critical information. There are techniques to improve this with fine-tuning, pipelining, or using complex techniques, which have their own challenges. To solve these challenges, we propose QA-prompting - a simple prompting method for summarization that utilizes question-answering as an intermediate step prior to summary generation. Our method extracts key information and enriches the context of text to mitigate positional biases and improve summarization in a single LM call per task without requiring fine-tuning or pipelining. Experiments on multiple datasets belonging to different domains using ten state-of-the-art pre-trained models demonstrate that QA-prompting outperforms baseline and other state-of-the-art methods, achieving up to 29% improvement in ROUGE scores. This provides an effective and scalable solution for summarization and highlights the importance of domain-specific question selection for optimal performance. 

---
# Self-Reasoning Language Models: Unfold Hidden Reasoning Chains with Few Reasoning Catalyst 

**Authors**: Hongru Wang, Deng Cai, Wanjun Zhong, Shijue Huang, Jeff Z. Pan, Zeming Liu, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2505.14116)  

**Abstract**: Inference-time scaling has attracted much attention which significantly enhance the performance of Large Language Models (LLMs) in complex reasoning tasks by increasing the length of Chain-of-Thought. These longer intermediate reasoning rationales embody various meta-reasoning skills in human cognition, such as reflection and decomposition, being difficult to create and acquire. In this work, we introduce \textit{Self-Reasoning Language Model} (SRLM), where the model itself can synthesize longer CoT data and iteratively improve performance through self-training. By incorporating a few demonstration examples (i.e., 1,000 samples) on how to unfold hidden reasoning chains from existing responses, which act as a reasoning catalyst, we demonstrate that SRLM not only enhances the model's initial performance but also ensures more stable and consistent improvements in subsequent iterations. Our proposed SRLM achieves an average absolute improvement of more than $+2.5$ points across five reasoning tasks: MMLU, GSM8K, ARC-C, HellaSwag, and BBH on two backbone models. Moreover, it brings more improvements with more times of sampling during inference, such as absolute $+7.89$ average improvement with $64$ sampling times, revealing the in-depth, diverse and creative reasoning paths in SRLM against the strong baseline. 

---
# ThinkSwitcher: When to Think Hard, When to Think Fast 

**Authors**: Guosheng Liang, Longguang Zhong, Ziyi Yang, Xiaojun Quan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14183)  

**Abstract**: Large reasoning models (LRMs) excel at solving complex tasks by leveraging long chain-of-thought (CoT) reasoning. However, this often leads to overthinking on simple tasks, resulting in unnecessary computational overhead. We observe that LRMs inherently possess the capability for efficient short CoT reasoning, which can be reliably elicited through prompt design. To leverage this capability, we propose ThinkSwitcher, a framework that enables a single LRM to dynamically switch between short and long CoT modes based on task complexity. ThinkSwitcher introduces a lightweight switching module trained with supervision signals derived from the relative performance of each reasoning mode across tasks. Experiments on multiple reasoning benchmarks show that ThinkSwitcher reduces computational cost by 20-30% while maintaining high accuracy on complex tasks. This demonstrates the effectiveness of ThinkSwitcher as a scalable and efficient solution for unified LRM deployment. 

---
# Temporal Alignment of Time Sensitive Facts with Activation Engineering 

**Authors**: Sanjay Govindan, Maurice Pagnucco, Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.14158)  

**Abstract**: Large Language Models (LLMs) are trained on diverse and often conflicting knowledge spanning multiple domains and time periods. Some of this knowledge is only valid within specific temporal contexts, such as answering the question, "Who is the President of the United States in 2022?" Ensuring LLMs generate time appropriate responses is crucial for maintaining relevance and accuracy. In this work we explore activation engineering as a method for temporally aligning LLMs to improve factual recall without any training or dataset creation. In this research we explore an activation engineering technique to ground three versions of LLaMA 2 to specific points in time and examine the effects of varying injection layers and prompting strategies. Our experiments demonstrate up to a 44% and 16% improvement in relative and explicit prompting respectively, achieving comparable performance to the fine-tuning method proposed by Zhao et al. (2024) . Notably, our approach achieves similar results to the fine-tuning baseline while being significantly more computationally efficient and requiring no pre-aligned datasets. 

---
# DiagnosisArena: Benchmarking Diagnostic Reasoning for Large Language Models 

**Authors**: Yakun Zhu, Zhongzhen Huang, Linjie Mu, Yutong Huang, Wei Nie, Shaoting Zhang, Pengfei Liu, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14107)  

**Abstract**: The emergence of groundbreaking large language models capable of performing complex reasoning tasks holds significant promise for addressing various scientific challenges, including those arising in complex clinical scenarios. To enable their safe and effective deployment in real-world healthcare settings, it is urgently necessary to benchmark the diagnostic capabilities of current models systematically. Given the limitations of existing medical benchmarks in evaluating advanced diagnostic reasoning, we present DiagnosisArena, a comprehensive and challenging benchmark designed to rigorously assess professional-level diagnostic competence. DiagnosisArena consists of 1,113 pairs of segmented patient cases and corresponding diagnoses, spanning 28 medical specialties, deriving from clinical case reports published in 10 top-tier medical journals. The benchmark is developed through a meticulous construction pipeline, involving multiple rounds of screening and review by both AI systems and human experts, with thorough checks conducted to prevent data leakage. Our study reveals that even the most advanced reasoning models, o3-mini, o1, and DeepSeek-R1, achieve only 45.82%, 31.09%, and 17.79% accuracy, respectively. This finding highlights a significant generalization bottleneck in current large language models when faced with clinical diagnostic reasoning challenges. Through DiagnosisArena, we aim to drive further advancements in AIs diagnostic reasoning capabilities, enabling more effective solutions for real-world clinical diagnostic challenges. We provide the benchmark and evaluation tools for further research and development this https URL. 

---
# A Personalized Conversational Benchmark: Towards Simulating Personalized Conversations 

**Authors**: Li Li, Peilin Cai, Ryan A. Rossi, Franck Dernoncourt, Branislav Kveton, Junda Wu, Tong Yu, Linxin Song, Tiankai Yang, Yuehan Qin, Nesreen K. Ahmed, Samyadeep Basu, Subhojyoti Mukherjee, Ruiyi Zhang, Zhengmian Hu, Bo Ni, Yuxiao Zhou, Zichao Wang, Yue Huang, Yu Wang, Xiangliang Zhang, Philip S. Yu, Xiyang Hu, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.14106)  

**Abstract**: We present PersonaConvBench, a large-scale benchmark for evaluating personalized reasoning and generation in multi-turn conversations with large language models (LLMs). Unlike existing work that focuses on either personalization or conversational structure in isolation, PersonaConvBench integrates both, offering three core tasks: sentence classification, impact regression, and user-centric text generation across ten diverse Reddit-based domains. This design enables systematic analysis of how personalized conversational context shapes LLM outputs in realistic multi-user scenarios. We benchmark several commercial and open-source LLMs under a unified prompting setup and observe that incorporating personalized history yields substantial performance improvements, including a 198 percent relative gain over the best non-conversational baseline in sentiment classification. By releasing PersonaConvBench with evaluations and code, we aim to support research on LLMs that adapt to individual styles, track long-term context, and produce contextually rich, engaging responses. 

---
# Legal Rule Induction: Towards Generalizable Principle Discovery from Analogous Judicial Precedents 

**Authors**: Wei Fan, Tianshi Zheng, Yiran Hu, Zheye Deng, Weiqi Wang, Baixuan Xu, Chunyang Li, Haoran Li, Weixing Shen, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.14104)  

**Abstract**: Legal rules encompass not only codified statutes but also implicit adjudicatory principles derived from precedents that contain discretionary norms, social morality, and policy. While computational legal research has advanced in applying established rules to cases, inducing legal rules from judicial decisions remains understudied, constrained by limitations in model inference efficacy and symbolic reasoning capability. The advent of Large Language Models (LLMs) offers unprecedented opportunities for automating the extraction of such latent principles, yet progress is stymied by the absence of formal task definitions, benchmark datasets, and methodologies. To address this gap, we formalize Legal Rule Induction (LRI) as the task of deriving concise, generalizable doctrinal rules from sets of analogous precedents, distilling their shared preconditions, normative behaviors, and legal consequences. We introduce the first LRI benchmark, comprising 5,121 case sets (38,088 Chinese cases in total) for model tuning and 216 expert-annotated gold test sets. Experimental results reveal that: 1) State-of-the-art LLMs struggle with over-generalization and hallucination; 2) Training on our dataset markedly enhances LLMs capabilities in capturing nuanced rule patterns across similar cases. 

---
# Texts or Images? A Fine-grained Analysis on the Effectiveness of Input Representations and Models for Table Question Answering 

**Authors**: Wei Zhou, Mohsen Mesgar, Heike Adel, Annemarie Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2505.14131)  

**Abstract**: In table question answering (TQA), tables are encoded as either texts or images. Prior work suggests that passing images of tables to multi-modal large language models (MLLMs) performs comparably to or even better than using textual input with large language models (LLMs). However, the lack of controlled setups limits fine-grained distinctions between these approaches. In this paper, we conduct the first controlled study on the effectiveness of several combinations of table representations and models from two perspectives: question complexity and table size. We build a new benchmark based on existing TQA datasets. In a systematic analysis of seven pairs of MLLMs and LLMs, we find that the best combination of table representation and model varies across setups. We propose FRES, a method selecting table representations dynamically, and observe a 10% average performance improvement compared to using both representations indiscriminately. 

---
# BAR: A Backward Reasoning based Agent for Complex Minecraft Tasks 

**Authors**: Weihong Du, Wenrui Liao, Binyu Yan, Hongru Liang, Anthony G. Cohn, Wenqiang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2505.14079)  

**Abstract**: Large language model (LLM) based agents have shown great potential in following human instructions and automatically completing various tasks. To complete a task, the agent needs to decompose it into easily executed steps by planning. Existing studies mainly conduct the planning by inferring what steps should be executed next starting from the agent's initial state. However, this forward reasoning paradigm doesn't work well for complex tasks. We propose to study this issue in Minecraft, a virtual environment that simulates complex tasks based on real-world scenarios. We believe that the failure of forward reasoning is caused by the big perception gap between the agent's initial state and task goal. To this end, we leverage backward reasoning and make the planning starting from the terminal state, which can directly achieve the task goal in one step. Specifically, we design a BAckward Reasoning based agent (BAR). It is equipped with a recursive goal decomposition module, a state consistency maintaining module and a stage memory module to make robust, consistent, and efficient planning starting from the terminal state. Experimental results demonstrate the superiority of BAR over existing methods and the effectiveness of proposed modules. 

---
# From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way Parallel Corpora 

**Authors**: Yingli Shen, Wen Lai, Shuo Wang, Kangyang Luo, Alexander Fraser, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.14045)  

**Abstract**: Continued pretraining and instruction tuning on large-scale multilingual data have proven to be effective in scaling large language models (LLMs) to low-resource languages. However, the unaligned nature of such data limits its ability to effectively capture cross-lingual semantics. In contrast, multi-way parallel data, where identical content is aligned across multiple languages, provides stronger cross-lingual consistency and offers greater potential for improving multilingual performance. In this paper, we introduce a large-scale, high-quality multi-way parallel corpus, TED2025, based on TED Talks. The corpus spans 113 languages, with up to 50 languages aligned in parallel, ensuring extensive multilingual coverage. Using this dataset, we investigate best practices for leveraging multi-way parallel data to enhance LLMs, including strategies for continued pretraining, instruction tuning, and the analysis of key influencing factors. Experiments on six multilingual benchmarks show that models trained on multiway parallel data consistently outperform those trained on unaligned multilingual data. 

---
# AUTOLAW: Enhancing Legal Compliance in Large Language Models via Case Law Generation and Jury-Inspired Deliberation 

**Authors**: Tai D. Nguyen, Long H. Pham, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.14015)  

**Abstract**: The rapid advancement of domain-specific large language models (LLMs) in fields like law necessitates frameworks that account for nuanced regional legal distinctions, which are critical for ensuring compliance and trustworthiness. Existing legal evaluation benchmarks often lack adaptability and fail to address diverse local contexts, limiting their utility in dynamically evolving regulatory landscapes. To address these gaps, we propose AutoLaw, a novel violation detection framework that combines adversarial data generation with a jury-inspired deliberation process to enhance legal compliance of LLMs. Unlike static approaches, AutoLaw dynamically synthesizes case law to reflect local regulations and employs a pool of LLM-based "jurors" to simulate judicial decision-making. Jurors are ranked and selected based on synthesized legal expertise, enabling a deliberation process that minimizes bias and improves detection accuracy. Evaluations across three benchmarks: Law-SG, Case-SG (legality), and Unfair-TOS (policy), demonstrate AutoLaw's effectiveness: adversarial data generation improves LLM discrimination, while the jury-based voting strategy significantly boosts violation detection rates. Our results highlight the framework's ability to adaptively probe legal misalignments and deliver reliable, context-aware judgments, offering a scalable solution for evaluating and enhancing LLMs in legally sensitive applications. 

---
# DecIF: Improving Instruction-Following through Meta-Decomposition 

**Authors**: Tingfeng Hui, Pengyu Zhu, Bowen Ping, Ling Tang, Yaqi Zhang, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.13990)  

**Abstract**: Instruction-following has emerged as a crucial capability for large language models (LLMs). However, existing approaches often rely on pre-existing documents or external resources to synthesize instruction-following data, which limits their flexibility and generalizability. In this paper, we introduce DecIF, a fully autonomous, meta-decomposition guided framework that generates diverse and high-quality instruction-following data using only LLMs. DecIF is grounded in the principle of decomposition. For instruction generation, we guide LLMs to iteratively produce various types of meta-information, which are then combined with response constraints to form well-structured and semantically rich instructions. We further utilize LLMs to detect and resolve potential inconsistencies within the generated instructions. Regarding response generation, we decompose each instruction into atomic-level evaluation criteria, enabling rigorous validation and the elimination of inaccurate instruction-response pairs. Extensive experiments across a wide range of scenarios and settings demonstrate DecIF's superior performance on instruction-following tasks. Further analysis highlights its strong flexibility, scalability, and generalizability in automatically synthesizing high-quality instruction data. 

---
# Enhancing LLMs via High-Knowledge Data Selection 

**Authors**: Feiyu Duan, Xuemiao Zhang, Sirui Wang, Haoran Que, Yuqi Liu, Wenge Rong, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2505.14070)  

**Abstract**: The performance of Large Language Models (LLMs) is intrinsically linked to the quality of its training data. Although several studies have proposed methods for high-quality data selection, they do not consider the importance of knowledge richness in text corpora. In this paper, we propose a novel and gradient-free High-Knowledge Scorer (HKS) to select high-quality data from the dimension of knowledge, to alleviate the problem of knowledge scarcity in the pre-trained corpus. We propose a comprehensive multi-domain knowledge element pool and introduce knowledge density and coverage as metrics to assess the knowledge content of the text. Based on this, we propose a comprehensive knowledge scorer to select data with intensive knowledge, which can also be utilized for domain-specific high-knowledge data selection by restricting knowledge elements to the specific domain. We train models on a high-knowledge bilingual dataset, and experimental results demonstrate that our scorer improves the model's performance in knowledge-intensive and general comprehension tasks, and is effective in enhancing both the generic and domain-specific capabilities of the model. 

---
# DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models 

**Authors**: Yuxuan Jiang, Dawei Li, Frank Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2505.13975)  

**Abstract**: While Large Reasoning Models (LRMs) have demonstrated success in complex reasoning tasks through long chain-of-thought (CoT) reasoning, their inference often involves excessively verbose reasoning traces, resulting in substantial inefficiency. To address this, we propose Distilled Reasoning Pruning (DRP), a hybrid framework that combines inference-time pruning with tuning-based distillation, two widely used strategies for efficient reasoning. DRP uses a teacher model to perform skill-aware step decomposition and content pruning, and then distills the pruned reasoning paths into a student model, enabling it to reason both efficiently and accurately. Across several challenging mathematical reasoning datasets, we find that models trained with DRP achieve substantial improvements in token efficiency without sacrificing accuracy. Specifically, DRP reduces average token usage on GSM8K from 917 to 328 while improving accuracy from 91.7% to 94.1%, and achieves a 43% token reduction on AIME with no performance drop. Further analysis shows that aligning the reasoning structure of training CoTs with the student's reasoning capacity is critical for effective knowledge transfer and performance gains. 

---
# The Hallucination Tax of Reinforcement Finetuning 

**Authors**: Linxin Song, Taiwei Shi, Jieyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13988)  

**Abstract**: Reinforcement finetuning (RFT) has become a standard approach for enhancing the reasoning capabilities of large language models (LLMs). However, its impact on model trustworthiness remains underexplored. In this work, we identify and systematically study a critical side effect of RFT, which we term the hallucination tax: a degradation in refusal behavior causing models to produce hallucinated answers to unanswerable questions confidently. To investigate this, we introduce SUM (Synthetic Unanswerable Math), a high-quality dataset of unanswerable math problems designed to probe models' ability to recognize an unanswerable question by reasoning from the insufficient or ambiguous information. Our results show that standard RFT training could reduce model refusal rates by more than 80%, which significantly increases model's tendency to hallucinate. We further demonstrate that incorporating just 10% SUM during RFT substantially restores appropriate refusal behavior, with minimal accuracy trade-offs on solvable tasks. Crucially, this approach enables LLMs to leverage inference-time compute to reason about their own uncertainty and knowledge boundaries, improving generalization not only to out-of-domain math problems but also to factual question answering tasks. 

---
# Truth or Twist? Optimal Model Selection for Reliable Label Flipping Evaluation in LLM-based Counterfactuals 

**Authors**: Qianli Wang, Van Bach Nguyen, Nils Feldhus, Luis Felipe Villa-Arenas, Christin Seifert, Sebastian Möller, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2505.13972)  

**Abstract**: Counterfactual examples are widely employed to enhance the performance and robustness of large language models (LLMs) through counterfactual data augmentation (CDA). However, the selection of the judge model used to evaluate label flipping, the primary metric for assessing the validity of generated counterfactuals for CDA, yields inconsistent results. To decipher this, we define four types of relationships between the counterfactual generator and judge models. Through extensive experiments involving two state-of-the-art LLM-based methods, three datasets, five generator models, and 15 judge models, complemented by a user study (n = 90), we demonstrate that judge models with an independent, non-fine-tuned relationship to the generator model provide the most reliable label flipping evaluations. Relationships between the generator and judge models, which are closely aligned with the user study for CDA, result in better model performance and robustness. Nevertheless, we find that the gap between the most effective judge models and the results obtained from the user study remains considerably large. This suggests that a fully automated pipeline for CDA may be inadequate and requires human intervention. 

---
# SlangDIT: Benchmarking LLMs in Interpretative Slang Translation 

**Authors**: Yunlong Liang, Fandong Meng, Jiaan Wang, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14181)  

**Abstract**: The challenge of slang translation lies in capturing context-dependent semantic extensions, as slang terms often convey meanings beyond their literal interpretation. While slang detection, explanation, and translation have been studied as isolated tasks in the era of large language models (LLMs), their intrinsic interdependence remains underexplored. The main reason is lacking of a benchmark where the two tasks can be a prerequisite for the third one, which can facilitate idiomatic translation. In this paper, we introduce the interpretative slang translation task (named SlangDIT) consisting of three sub-tasks: slang detection, cross-lingual slang explanation, and slang translation within the current context, aiming to generate more accurate translation with the help of slang detection and slang explanation. To this end, we construct a SlangDIT dataset, containing over 25k English-Chinese sentence pairs. Each source sentence mentions at least one slang term and is labeled with corresponding cross-lingual slang explanation. Based on the benchmark, we propose a deep thinking model, named SlangOWL. It firstly identifies whether the sentence contains a slang, and then judges whether the slang is polysemous and analyze its possible meaning. Further, the SlangOWL provides the best explanation of the slang term targeting on the current context. Finally, according to the whole thought, the SlangOWL offers a suitable translation. Our experiments on LLMs (\emph{e.g.}, Qwen2.5 and LLama-3.1), show that our deep thinking approach indeed enhances the performance of LLMs where the proposed SLangOWL significantly surpasses the vanilla models and supervised fine-tuned models without thinking. 

---
# FlashThink: An Early Exit Method For Efficient Reasoning 

**Authors**: Guochao Jiang, Guofeng Quan, Zepeng Ding, Ziqin Luo, Dixuan Wang, Zheng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13949)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance in reasoning tasks. However, LLMs tend to generate excessively long reasoning content, leading to significant computational overhead. Our observations indicate that even on simple problems, LLMs tend to produce unnecessarily lengthy reasoning content, which is against intuitive expectations. Preliminary experiments show that at a certain point during the generation process, the model is already capable of producing the correct solution without completing the full reasoning content. Therefore, we consider that the reasoning process of the model can be exited early to achieve the purpose of efficient reasoning. We introduce a verification model that identifies the exact moment when the model can stop reasoning and still provide the correct answer. Comprehensive experiments on four different benchmarks demonstrate that our proposed method, FlashThink, effectively shortens the reasoning content while preserving the model accuracy. For the Deepseek-R1 and QwQ-32B models, we reduced the length of reasoning content by 77.04% and 77.47%, respectively, without reducing the accuracy. 

---
# Through a Compressed Lens: Investigating the Impact of Quantization on LLM Explainability and Interpretability 

**Authors**: Qianli Wang, Mingyang Wang, Nils Feldhus, Simon Ostermann, Yuan Cao, Hinrich Schütze, Sebastian Möller, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2505.13963)  

**Abstract**: Quantization methods are widely used to accelerate inference and streamline the deployment of large language models (LLMs). While prior research has extensively investigated the degradation of various LLM capabilities due to quantization, its effects on model explainability and interpretability, which are crucial for understanding decision-making processes, remain unexplored. To address this gap, we conduct comprehensive experiments using three common quantization techniques at distinct bit widths, in conjunction with two explainability methods, counterfactual examples and natural language explanations, as well as two interpretability approaches, knowledge memorization analysis and latent multi-hop reasoning analysis. We complement our analysis with a thorough user study, evaluating selected explainability methods. Our findings reveal that, depending on the configuration, quantization can significantly impact model explainability and interpretability. Notably, the direction of this effect is not consistent, as it strongly depends on (1) the quantization method, (2) the explainability or interpretability approach, and (3) the evaluation protocol. In some settings, human evaluation shows that quantization degrades explainability, while in others, it even leads to improvements. Our work serves as a cautionary tale, demonstrating that quantization can unpredictably affect model transparency. This insight has important implications for deploying LLMs in applications where transparency is a critical requirement. 

---
# Activation-Guided Consensus Merging for Large Language Models 

**Authors**: Yuxuan Yao, Shuqi Liu, Zehua Liu, Qintong Li, Mingyang Liu, Xiongwei Han, Zhijiang Guo, Han Wu, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.14009)  

**Abstract**: Recent research has increasingly focused on reconciling the reasoning capabilities of System 2 with the efficiency of System 1. While existing training-based and prompt-based approaches face significant challenges in terms of efficiency and stability, model merging emerges as a promising strategy to integrate the diverse capabilities of different Large Language Models (LLMs) into a unified model. However, conventional model merging methods often assume uniform importance across layers, overlooking the functional heterogeneity inherent in neural components. To address this limitation, we propose \textbf{A}ctivation-Guided \textbf{C}onsensus \textbf{M}erging (\textbf{ACM}), a plug-and-play merging framework that determines layer-specific merging coefficients based on mutual information between activations of pre-trained and fine-tuned models. ACM effectively preserves task-specific capabilities without requiring gradient computations or additional training. Extensive experiments on Long-to-Short (L2S) and general merging tasks demonstrate that ACM consistently outperforms all baseline methods. For instance, in the case of Qwen-7B models, TIES-Merging equipped with ACM achieves a \textbf{55.3\%} reduction in response length while simultaneously improving reasoning accuracy by \textbf{1.3} points. We submit the code with the paper for reproducibility, and it will be publicly available. 

---
# Code2Logic: Game-Code-Driven Data Synthesis for Enhancing VLMs General Reasoning 

**Authors**: Jingqi Tong, Jixin Tang, Hangcheng Li, Yurong Mou, Ming Zhang, Jun Zhao, Yanbo Wen, Fan Song, Jiahao Zhan, Yuyang Lu, Chaoran Tao, Zhiyuan Guo, Jizhou Yu, Tianhao Cheng, Changhao Jiang, Zhen Wang, Tao Liang, Zhihui Fei, Mingyang Wan, Guojun Ma, Weifeng Ge, Guanhua Chen, Tao Gui, Xipeng Qiu, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13886)  

**Abstract**: Visual-language Chain-of-Thought (CoT) data resources are relatively scarce compared to text-only counterparts, limiting the improvement of reasoning capabilities in Vision Language Models (VLMs). However, high-quality vision-language reasoning data is expensive and labor-intensive to annotate. To address this issue, we leverage a promising resource: game code, which naturally contains logical structures and state transition processes. Therefore, we propose Code2Logic, a novel game-code-driven approach for multimodal reasoning data synthesis. Our approach leverages Large Language Models (LLMs) to adapt game code, enabling automatic acquisition of reasoning processes and results through code execution. Using the Code2Logic approach, we developed the GameQA dataset to train and evaluate VLMs. GameQA is cost-effective and scalable to produce, challenging for state-of-the-art models, and diverse with 30 games and 158 tasks. Surprisingly, despite training solely on game data, VLMs demonstrated out of domain generalization, specifically Qwen2.5-VL-7B improving performance by 2.33\% across 7 diverse vision-language benchmarks. Our code and dataset are available at this https URL. 

---
# Reasoning Path Compression: Compressing Generation Trajectories for Efficient LLM Reasoning 

**Authors**: Jiwon Song, Dongwon Jo, Yulhwa Kim, Jae-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.13866)  

**Abstract**: Recent reasoning-focused language models achieve high accuracy by generating lengthy intermediate reasoning paths before producing final answers. While this approach is effective in solving problems that require logical thinking, long reasoning paths significantly increase memory usage and throughput of token generation, limiting the practical deployment of such models. We propose Reasoning Path Compression (RPC), a training-free method that accelerates inference by leveraging the semantic sparsity of reasoning paths. RPC periodically compresses the KV cache by retaining KV cache that receive high importance score, which are computed using a selector window composed of recently generated queries. Experiments show that RPC improves generation throughput of QwQ-32B by up to 1.60$\times$ compared to the inference with full KV cache, with an accuracy drop of 1.2% on the AIME 2024 benchmark. Our findings demonstrate that semantic sparsity in reasoning traces can be effectively exploited for compression, offering a practical path toward efficient deployment of reasoning LLMs. Our code is available at this https URL. 

---
# Let's Verify Math Questions Step by Step 

**Authors**: Chengyu Shen, Zhen Hao Wong, Runming He, Hao Liang, Meiyi Qiang, Zimo Meng, Zhengyang Zhao, Bohan Zeng, Zhengzhou Zhu, Bin Cui, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13903)  

**Abstract**: Large Language Models (LLMs) have recently achieved remarkable progress in mathematical reasoning. To enable such capabilities, many existing works distill strong reasoning models into long chains of thought or design algorithms to construct high-quality math QA data for training. However, these efforts primarily focus on generating correct reasoning paths and answers, while largely overlooking the validity of the questions themselves. In this work, we propose Math Question Verification (MathQ-Verify), a novel five-stage pipeline designed to rigorously filter ill-posed or under-specified math problems. MathQ-Verify first performs format-level validation to remove redundant instructions and ensure that each question is syntactically well-formed. It then formalizes each question, decomposes it into atomic conditions, and verifies them against mathematical definitions. Next, it detects logical contradictions among these conditions, followed by a goal-oriented completeness check to ensure the question provides sufficient information for solving. To evaluate this task, we use existing benchmarks along with an additional dataset we construct, containing 2,147 math questions with diverse error types, each manually double-validated. Experiments show that MathQ-Verify achieves state-of-the-art performance across multiple benchmarks, improving the F1 score by up to 25 percentage points over the direct verification baseline. It further attains approximately 90% precision and 63% recall through a lightweight model voting scheme. MathQ-Verify offers a scalable and accurate solution for curating reliable mathematical datasets, reducing label noise and avoiding unnecessary computation on invalid questions. Our code and data are available at this https URL. 

---
# EfficientLLM: Efficiency in Large Language Models 

**Authors**: Zhengqing Yuan, Weixiang Sun, Yixin Liu, Huichi Zhou, Rong Zhou, Yiyang Li, Zheyuan Zhang, Wei Song, Yue Huang, Haolong Jia, Keerthiram Murugesan, Yu Wang, Lifang He, Jianfeng Gao, Lichao Sun, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.13840)  

**Abstract**: Large Language Models (LLMs) have driven significant progress, yet their growing parameter counts and context windows incur prohibitive compute, energy, and monetary costs. We introduce EfficientLLM, a novel benchmark and the first comprehensive empirical study evaluating efficiency techniques for LLMs at scale. Conducted on a production-class cluster (48xGH200, 8xH200 GPUs), our study systematically explores three key axes: (1) architecture pretraining (efficient attention variants: MQA, GQA, MLA, NSA; sparse Mixture-of-Experts (MoE)), (2) fine-tuning (parameter-efficient methods: LoRA, RSLoRA, DoRA), and (3) inference (quantization methods: int4, float16). We define six fine-grained metrics (Memory Utilization, Compute Utilization, Latency, Throughput, Energy Consumption, Compression Rate) to capture hardware saturation, latency-throughput balance, and carbon cost. Evaluating over 100 model-technique pairs (0.5B-72B parameters), we derive three core insights: (i) Efficiency involves quantifiable trade-offs: no single method is universally optimal; e.g., MoE reduces FLOPs and improves accuracy but increases VRAM by 40%, while int4 quantization cuts memory/energy by up to 3.9x at a 3-5% accuracy drop. (ii) Optima are task- and scale-dependent: MQA offers optimal memory-latency trade-offs for constrained devices, MLA achieves lowest perplexity for quality-critical tasks, and RSLoRA surpasses LoRA efficiency only beyond 14B parameters. (iii) Techniques generalize across modalities: we extend evaluations to Large Vision Models (Stable Diffusion 3.5, Wan 2.1) and Vision-Language Models (Qwen2.5-VL), confirming effective transferability. By open-sourcing datasets, evaluation pipelines, and leaderboards, EfficientLLM provides essential guidance for researchers and engineers navigating the efficiency-performance landscape of next-generation foundation models. 

---
# CAFES: A Collaborative Multi-Agent Framework for Multi-Granular Multimodal Essay Scoring 

**Authors**: Jiamin Su, Yibo Yan, Zhuoran Gao, Han Zhang, Xiang Liu, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13965)  

**Abstract**: Automated Essay Scoring (AES) is crucial for modern education, particularly with the increasing prevalence of multimodal assessments. However, traditional AES methods struggle with evaluation generalizability and multimodal perception, while even recent Multimodal Large Language Model (MLLM)-based approaches can produce hallucinated justifications and scores misaligned with human judgment. To address the limitations, we introduce CAFES, the first collaborative multi-agent framework specifically designed for AES. It orchestrates three specialized agents: an Initial Scorer for rapid, trait-specific evaluations; a Feedback Pool Manager to aggregate detailed, evidence-grounded strengths; and a Reflective Scorer that iteratively refines scores based on this feedback to enhance human alignment. Extensive experiments, using state-of-the-art MLLMs, achieve an average relative improvement of 21% in Quadratic Weighted Kappa (QWK) against ground truth, especially for grammatical and lexical diversity. Our proposed CAFES framework paves the way for an intelligent multimodal AES system. The code will be available upon acceptance. 

---
# Krikri: Advancing Open Large Language Models for Greek 

**Authors**: Dimitris Roussis, Leon Voukoutis, Georgios Paraskevopoulos, Sokratis Sofianopoulos, Prokopis Prokopidis, Vassilis Papavasileiou, Athanasios Katsamanis, Stelios Piperidis, Vassilis Katsouros  

**Link**: [PDF](https://arxiv.org/pdf/2505.13772)  

**Abstract**: We introduce Llama-Krikri-8B, a cutting-edge Large Language Model tailored for the Greek language, built on Meta's Llama 3.1-8B. Llama-Krikri-8B has been extensively trained on high-quality Greek data to ensure superior adaptation to linguistic nuances. With 8 billion parameters, it offers advanced capabilities while maintaining efficient computational performance. Llama-Krikri-8B supports both Modern Greek and English, and is also equipped to handle polytonic text and Ancient Greek. The chat version of Llama-Krikri-8B features a multi-stage post-training pipeline, utilizing both human and synthetic instruction and preference data, by applying techniques such as MAGPIE. In addition, for evaluation, we propose three novel public benchmarks for Greek. Our evaluation on existing as well as the proposed benchmarks shows notable improvements over comparable Greek and multilingual LLMs in both natural language understanding and generation as well as code generation. 

---
# InfiGFusion: Graph-on-Logits Distillation via Efficient Gromov-Wasserstein for Model Fusion 

**Authors**: Yuanyi Wang, Zhaoyi Yan, Yiming Zhang, Qi Zhou, Yanggan Gu, Fei Wu, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13893)  

**Abstract**: Recent advances in large language models (LLMs) have intensified efforts to fuse heterogeneous open-source models into a unified system that inherits their complementary strengths. Existing logit-based fusion methods maintain inference efficiency but treat vocabulary dimensions independently, overlooking semantic dependencies encoded by cross-dimension interactions. These dependencies reflect how token types interact under a model's internal reasoning and are essential for aligning models with diverse generation behaviors. To explicitly model these dependencies, we propose \textbf{InfiGFusion}, the first structure-aware fusion framework with a novel \textit{Graph-on-Logits Distillation} (GLD) loss. Specifically, we retain the top-$k$ logits per output and aggregate their outer products across sequence positions to form a global co-activation graph, where nodes represent vocabulary channels and edges quantify their joint activations. To ensure scalability and efficiency, we design a sorting-based closed-form approximation that reduces the original $O(n^4)$ cost of Gromov-Wasserstein distance to $O(n \log n)$, with provable approximation guarantees. Experiments across multiple fusion settings show that GLD consistently improves fusion quality and stability. InfiGFusion outperforms SOTA models and fusion baselines across 11 benchmarks spanning reasoning, coding, and mathematics. It shows particular strength in complex reasoning tasks, with +35.6 improvement on Multistep Arithmetic and +37.06 on Causal Judgement over SFT, demonstrating superior multi-step and relational inference. 

---
# Simulation Agent: A Framework for Integrating Simulation and Large Language Models for Enhanced Decision-Making 

**Authors**: Jacob Kleiman, Kevin Frank, Sindy Campagna  

**Link**: [PDF](https://arxiv.org/pdf/2505.13761)  

**Abstract**: Simulations, although powerful in accurately replicating real-world systems, often remain inaccessible to non-technical users due to their complexity. Conversely, large language models (LLMs) provide intuitive, language-based interactions but can lack the structured, causal understanding required to reliably model complex real-world dynamics. We introduce our simulation agent framework, a novel approach that integrates the strengths of both simulation models and LLMs. This framework helps empower users by leveraging the conversational capabilities of LLMs to interact seamlessly with sophisticated simulation systems, while simultaneously utilizing the simulations to ground the LLMs in accurate and structured representations of real-world phenomena. This integrated approach helps provide a robust and generalizable foundation for empirical validation and offers broad applicability across diverse domains. 

---
# SQLForge: Synthesizing Reliable and Diverse Data to Enhance Text-to-SQL Reasoning in LLMs 

**Authors**: Yu Guo, Dong Jin, Shenghao Ye, Shuangwu Chen, Jian Yang, Xiaobin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13725)  

**Abstract**: Large Language models (LLMs) have demonstrated significant potential in text-to-SQL reasoning tasks, yet a substantial performance gap persists between existing open-source models and their closed-source counterparts. In this paper, we introduce SQLForge, a novel approach for synthesizing reliable and diverse data to enhance text-to-SQL reasoning in LLMs. We improve data reliability through SQL syntax constraints and SQL-to-question reverse translation, ensuring data logic at both structural and semantic levels. We also propose an SQL template enrichment and iterative data domain exploration mechanism to boost data diversity. Building on the augmented data, we fine-tune a variety of open-source models with different architectures and parameter sizes, resulting in a family of models termed SQLForge-LM. SQLForge-LM achieves the state-of-the-art performance on the widely recognized Spider and BIRD benchmarks among the open-source models. Specifically, SQLForge-LM achieves EX accuracy of 85.7% on Spider Dev and 59.8% on BIRD Dev, significantly narrowing the performance gap with closed-source methods. 

---
# CS-Sum: A Benchmark for Code-Switching Dialogue Summarization and the Limits of Large Language Models 

**Authors**: Sathya Krishnan Suresh, Tanmay Surana, Lim Zhi Hao, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2505.13559)  

**Abstract**: Code-switching (CS) poses a significant challenge for Large Language Models (LLMs), yet its comprehensibility remains underexplored in LLMs. We introduce CS-Sum, to evaluate the comprehensibility of CS by the LLMs through CS dialogue to English summarization. CS-Sum is the first benchmark for CS dialogue summarization across Mandarin-English (EN-ZH), Tamil-English (EN-TA), and Malay-English (EN-MS), with 900-1300 human-annotated dialogues per language pair. Evaluating ten LLMs, including open and closed-source models, we analyze performance across few-shot, translate-summarize, and fine-tuning (LoRA, QLoRA on synthetic data) approaches. Our findings show that though the scores on automated metrics are high, LLMs make subtle mistakes that alter the complete meaning of the dialogue. To this end, we introduce 3 most common type of errors that LLMs make when handling CS input. Error rates vary across CS pairs and LLMs, with some LLMs showing more frequent errors on certain language pairs, underscoring the need for specialized training on code-switched data. 

---
# Are Large Language Models Good at Detecting Propaganda? 

**Authors**: Julia Jose, Rachel Greenstadt  

**Link**: [PDF](https://arxiv.org/pdf/2505.13706)  

**Abstract**: Propagandists use rhetorical devices that rely on logical fallacies and emotional appeals to advance their agendas. Recognizing these techniques is key to making informed decisions. Recent advances in Natural Language Processing (NLP) have enabled the development of systems capable of detecting manipulative content. In this study, we look at several Large Language Models and their performance in detecting propaganda techniques in news articles. We compare the performance of these LLMs with transformer-based models. We find that, while GPT-4 demonstrates superior F1 scores (F1=0.16) compared to GPT-3.5 and Claude 3 Opus, it does not outperform a RoBERTa-CRF baseline (F1=0.67). Additionally, we find that all three LLMs outperform a MultiGranularity Network (MGN) baseline in detecting instances of one out of six propaganda techniques (name-calling), with GPT-3.5 and GPT-4 also outperforming the MGN baseline in detecting instances of appeal to fear and flag-waving. 

---
# Combining the Best of Both Worlds: A Method for Hybrid NMT and LLM Translation 

**Authors**: Zhanglin Wu, Daimeng Wei, Xiaoyu Chen, Hengchao Shang, Jiaxin Guo, Zongyao Li, Yuanchang Luo, Jinlong Yang, Zhiqiang Rao, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13554)  

**Abstract**: Large language model (LLM) shows promising performances in a variety of downstream tasks, such as machine translation (MT). However, using LLMs for translation suffers from high computational costs and significant latency. Based on our evaluation, in most cases, translations using LLMs are comparable to that generated by neural machine translation (NMT) systems. Only in particular scenarios, LLM and NMT models show respective advantages. As a result, integrating NMT and LLM for translation and using LLM only when necessary seems to be a sound solution. A scheduling policy that optimizes translation result while ensuring fast speed and as little LLM usage as possible is thereby required. We compare several scheduling policies and propose a novel and straightforward decider that leverages source sentence features. We conduct extensive experiments on multilingual test sets and the result shows that we can achieve optimal translation performance with minimal LLM usage, demonstrating effectiveness of our decider. 

---
# Improve Language Model and Brain Alignment via Associative Memory 

**Authors**: Congchi Yin, Yongpeng Zhang, Xuyun Wen, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13844)  

**Abstract**: Associative memory engages in the integration of relevant information for comprehension in the human cognition system. In this work, we seek to improve alignment between language models and human brain while processing speech information by integrating associative memory. After verifying the alignment between language model and brain by mapping language model activations to brain activity, the original text stimuli expanded with simulated associative memory are regarded as input to computational language models. We find the alignment between language model and brain is improved in brain regions closely related to associative memory processing. We also demonstrate large language models after specific supervised fine-tuning better align with brain response, by building the \textit{Association} dataset containing 1000 samples of stories, with instructions encouraging associative memory as input and associated content as output. 

---
# Induction Head Toxicity Mechanistically Explains Repetition Curse in Large Language Models 

**Authors**: Shuxun Wang, Qingyu Yin, Chak Tou Leong, Qiang Zhang, Linyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13514)  

**Abstract**: Repetition curse is a phenomenon where Large Language Models (LLMs) generate repetitive sequences of tokens or cyclic sequences. While the repetition curse has been widely observed, its underlying mechanisms remain poorly understood. In this work, we investigate the role of induction heads--a specific type of attention head known for their ability to perform in-context learning--in driving this repetitive behavior. Specifically, we focus on the "toxicity" of induction heads, which we define as their tendency to dominate the model's output logits during repetition, effectively excluding other attention heads from contributing to the generation process. Our findings have important implications for the design and training of LLMs. By identifying induction heads as a key driver of the repetition curse, we provide a mechanistic explanation for this phenomenon and suggest potential avenues for mitigation. We also propose a technique with attention head regularization that could be employed to reduce the dominance of induction heads during generation, thereby promoting more diverse and coherent outputs. 

---
# Mapping the Minds of LLMs: A Graph-Based Analysis of Reasoning LLM 

**Authors**: Zhen Xiong, Yujun Cai, Zhecheng Li, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13890)  

**Abstract**: Recent advances in test-time scaling have enabled Large Language Models (LLMs) to display sophisticated reasoning abilities via extended Chain-of-Thought (CoT) generation. Despite their potential, these Reasoning LLMs (RLMs) often demonstrate counterintuitive and unstable behaviors, such as performance degradation under few-shot prompting, that challenge our current understanding of RLMs. In this work, we introduce a unified graph-based analytical framework for better modeling the reasoning processes of RLMs. Our method first clusters long, verbose CoT outputs into semantically coherent reasoning steps, then constructs directed reasoning graphs to capture contextual and logical dependencies among these steps. Through comprehensive analysis across models and prompting regimes, we reveal that structural properties, such as exploration density, branching, and convergence ratios, strongly correlate with reasoning accuracy. Our findings demonstrate how prompting strategies substantially reshape the internal reasoning structure of RLMs, directly affecting task outcomes. The proposed framework not only enables quantitative evaluation of reasoning quality beyond conventional metrics but also provides practical insights for prompt engineering and the cognitive analysis of LLMs. Code and resources will be released to facilitate future research in this direction. 

---
# Noise Injection Systemically Degrades Large Language Model Safety Guardrails 

**Authors**: Prithviraj Singh Shahani, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2505.13500)  

**Abstract**: Safety guardrails in large language models (LLMs) are a critical component in preventing harmful outputs. Yet, their resilience under perturbation remains poorly understood. In this paper, we investigate the robustness of safety fine-tuning in LLMs by systematically injecting Gaussian noise into model activations. We show across multiple open-weight models that (1) Gaussian noise raises harmful-output rates (p < 0.001) by up to 27%, (2) that deeper safety fine-tuning affords no extra protection, and (3) that chain-of-thought reasoning remains largely intact. The findings reveal critical vulnerabilities in current safety alignment techniques and highlight the potential of reasoning-based and reinforcement learning approaches as promising direction for developing more robust AI safety systems. These results have important implications for real-world deployment of LLMs in safety-critical applications as these results imply that widely-deployed safety tuning methods can fail even without adversarial prompts. 

---
# IRLBench: A Multi-modal, Culturally Grounded, Parallel Irish-English Benchmark for Open-Ended LLM Reasoning Evaluation 

**Authors**: Khanh-Tung Tran, Barry O'Sullivan, Hoang D. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13498)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated promising knowledge and reasoning abilities, yet their performance in multilingual and low-resource settings remains underexplored. Existing benchmarks often exhibit cultural bias, restrict evaluation to text-only, rely on multiple-choice formats, and, more importantly, are limited for extremely low-resource languages. To address these gaps, we introduce IRLBench, presented in parallel English and Irish, which is considered definitely endangered by UNESCO. Our benchmark consists of 12 representative subjects developed from the 2024 Irish Leaving Certificate exams, enabling fine-grained analysis of model capabilities across domains. By framing the task as long-form generation and leveraging the official marking scheme, it does not only support a comprehensive evaluation of correctness but also language fidelity. Our extensive experiments of leading closed-source and open-source LLMs reveal a persistent performance gap between English and Irish, in which models produce valid Irish responses less than 80\% of the time, and answer correctly 55.8\% of the time compared to 76.2\% in English for the best-performing model. We release IRLBench (this https URL) and an accompanying evaluation codebase (this https URL) to enable future research on robust, culturally aware multilingual AI development. 

---
# Source framing triggers systematic evaluation bias in Large Language Models 

**Authors**: Federico Germani, Giovanni Spitale  

**Link**: [PDF](https://arxiv.org/pdf/2505.13488)  

**Abstract**: Large Language Models (LLMs) are increasingly used not only to generate text but also to evaluate it, raising urgent questions about whether their judgments are consistent, unbiased, and robust to framing effects. In this study, we systematically examine inter- and intra-model agreement across four state-of-the-art LLMs (OpenAI o3-mini, Deepseek Reasoner, xAI Grok 2, and Mistral) tasked with evaluating 4,800 narrative statements on 24 different topics of social, political, and public health relevance, for a total of 192,000 assessments. We manipulate the disclosed source of each statement to assess how attribution to either another LLM or a human author of specified nationality affects evaluation outcomes. We find that, in the blind condition, different LLMs display a remarkably high degree of inter- and intra-model agreement across topics. However, this alignment breaks down when source framing is introduced. Here we show that attributing statements to Chinese individuals systematically lowers agreement scores across all models, and in particular for Deepseek Reasoner. Our findings reveal that framing effects can deeply affect text evaluation, with significant implications for the integrity, neutrality, and fairness of LLM-mediated information systems. 

---
# Detecting Prefix Bias in LLM-based Reward Models 

**Authors**: Ashwin Kumar, Yuzi He, Aram H. Markosyan, Bobbie Chern, Imanol Arrieta-Ibarra  

**Link**: [PDF](https://arxiv.org/pdf/2505.13487)  

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) has emerged as a key paradigm for task-specific fine-tuning of language models using human preference data. While numerous publicly available preference datasets provide pairwise comparisons of responses, the potential for biases in the resulting reward models remains underexplored. In this work, we introduce novel methods to detect and evaluate prefix bias -- a systematic shift in model preferences triggered by minor variations in query prefixes -- in LLM-based reward models trained on such datasets. We leverage these metrics to reveal significant biases in preference models across racial and gender dimensions. Our comprehensive evaluation spans diverse open-source preference datasets and reward model architectures, demonstrating susceptibility to this kind of bias regardless of the underlying model architecture. Furthermore, we propose a data augmentation strategy to mitigate these biases, showing its effectiveness in reducing the impact of prefix bias. Our findings highlight the critical need for bias-aware dataset design and evaluation in developing fair and reliable reward models, contributing to the broader discourse on fairness in AI. 

---
# LLM4CD: Leveraging Large Language Models for Open-World Knowledge Augmented Cognitive Diagnosis 

**Authors**: Weiming Zhang, Lingyue Fu, Qingyao Li, Kounianhua Du, Jianghao Lin, Jingwei Yu, Wei Xia, Weinan Zhang, Ruiming Tang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13492)  

**Abstract**: Cognitive diagnosis (CD) plays a crucial role in intelligent education, evaluating students' comprehension of knowledge concepts based on their test histories. However, current CD methods often model students, exercises, and knowledge concepts solely on their ID relationships, neglecting the abundant semantic relationships present within educational data space. Furthermore, contemporary intelligent tutoring systems (ITS) frequently involve the addition of new students and exercises, a situation that ID-based methods find challenging to manage effectively. The advent of large language models (LLMs) offers the potential for overcoming this challenge with open-world knowledge. In this paper, we propose LLM4CD, which Leverages Large Language Models for Open-World Knowledge Augmented Cognitive Diagnosis. Our method utilizes the open-world knowledge of LLMs to construct cognitively expressive textual representations, which are then encoded to introduce rich semantic information into the CD task. Additionally, we propose an innovative bi-level encoder framework that models students' test histories through two levels of encoders: a macro-level cognitive text encoder and a micro-level knowledge state encoder. This approach substitutes traditional ID embeddings with semantic representations, enabling the model to accommodate new students and exercises with open-world knowledge and address the cold-start problem. Extensive experimental results demonstrate that our proposed method consistently outperforms previous CD models on multiple real-world datasets, validating the effectiveness of leveraging LLMs to introduce rich semantic information into the CD task. 

---
# KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models 

**Authors**: Fnu Mohbat, Mohammed J Zaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.14629)  

**Abstract**: Recent advances in large language models (LLMs) and the abundance of food data have resulted in studies to improve food understanding using LLMs. Despite several recommendation systems utilizing LLMs and Knowledge Graphs (KGs), there has been limited research on integrating food related KGs with LLMs. We introduce KERL, a unified system that leverages food KGs and LLMs to provide personalized food recommendations and generates recipes with associated micro-nutritional information. Given a natural language question, KERL extracts entities, retrieves subgraphs from the KG, which are then fed into the LLM as context to select the recipes that satisfy the constraints. Next, our system generates the cooking steps and nutritional information for each recipe. To evaluate our approach, we also develop a benchmark dataset by curating recipe related questions, combined with constraints and personal preferences. Through extensive experiments, we show that our proposed KG-augmented LLM significantly outperforms existing approaches, offering a complete and coherent solution for food recommendation, recipe generation, and nutritional analysis. Our code and benchmark datasets are publicly available at this https URL. 

---
# Debating for Better Reasoning: An Unsupervised Multimodal Approach 

**Authors**: Ashutosh Adhikari, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2505.14627)  

**Abstract**: As Large Language Models (LLMs) gain expertise across diverse domains and modalities, scalable oversight becomes increasingly challenging, particularly when their capabilities may surpass human evaluators. Debate has emerged as a promising mechanism for enabling such oversight. In this work, we extend the debate paradigm to a multimodal setting, exploring its potential for weaker models to supervise and enhance the performance of stronger models. We focus on visual question answering (VQA), where two "sighted" expert vision-language models debate an answer, while a "blind" (text-only) judge adjudicates based solely on the quality of the arguments. In our framework, the experts defend only answers aligned with their beliefs, thereby obviating the need for explicit role-playing and concentrating the debate on instances of expert disagreement. Experiments on several multimodal tasks demonstrate that the debate framework consistently outperforms individual expert models. Moreover, judgments from weaker LLMs can help instill reasoning capabilities in vision-language models through finetuning. 

---
# Enhancing Learned Knowledge in LoRA Adapters Through Efficient Contrastive Decoding on Ascend NPUs 

**Authors**: Morgan Lindsay Heisler, Linzi Xing, Ge Shi, Hanieh Sadri, Gursimran Singh, Weiwei Zhang, Tao Ye, Ying Xiong, Yong Zhang, Zhenan Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14620)  

**Abstract**: Huawei Cloud users leverage LoRA (Low-Rank Adaptation) as an efficient and scalable method to fine-tune and customize large language models (LLMs) for application-specific needs. However, tasks that require complex reasoning or deep contextual understanding are often hindered by biases or interference from the base model when using typical decoding methods like greedy or beam search. These biases can lead to generic or task-agnostic responses from the base model instead of leveraging the LoRA-specific adaptations. In this paper, we introduce Contrastive LoRA Decoding (CoLD), a novel decoding framework designed to maximize the use of task-specific knowledge in LoRA-adapted models, resulting in better downstream performance. CoLD uses contrastive decoding by scoring candidate tokens based on the divergence between the probability distributions of a LoRA-adapted expert model and the corresponding base model. This approach prioritizes tokens that better align with the LoRA's learned representations, enhancing performance for specialized tasks. While effective, a naive implementation of CoLD is computationally expensive because each decoding step requires evaluating multiple token candidates across both models. To address this, we developed an optimized kernel for Huawei's Ascend NPU. CoLD achieves up to a 5.54% increase in task accuracy while reducing end-to-end latency by 28% compared to greedy decoding. This work provides practical and efficient decoding strategies for fine-tuned LLMs in resource-constrained environments and has broad implications for applied data science in both cloud and on-premises settings. 

---
# SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle Generation from SAT Formulas 

**Authors**: Anjiang Wei, Yuheng Wu, Yingjia Wan, Tarun Suresh, Huanmi Tan, Zhanke Zhou, Sanmi Koyejo, Ke Wang, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2505.14615)  

**Abstract**: We introduce SATBench, a benchmark for evaluating the logical reasoning capabilities of large language models (LLMs) through logical puzzles derived from Boolean satisfiability (SAT) problems. Unlike prior work that focuses on inference rule-based reasoning, which often involves deducing conclusions from a set of premises, our approach leverages the search-based nature of SAT problems, where the objective is to find a solution that fulfills a specified set of logical constraints. Each instance in SATBench is generated from a SAT formula, then translated into a story context and conditions using LLMs. The generation process is fully automated and allows for adjustable difficulty by varying the number of clauses. All 2100 puzzles are validated through both LLM-assisted and solver-based consistency checks, with human validation on a subset. Experimental results show that even the strongest model, o4-mini, achieves only 65.0% accuracy on hard UNSAT problems, close to the random baseline of 50%. SATBench exposes fundamental limitations in the search-based logical reasoning abilities of current LLMs and provides a scalable testbed for future research in logical reasoning. 

---
# Time-R1: Towards Comprehensive Temporal Reasoning in LLMs 

**Authors**: Zijia Liu, Peixuan Han, Haofei Yu, Haoru Li, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2505.13508)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities but lack robust temporal intelligence, struggling to integrate reasoning about the past with predictions and plausible generations of the future. Meanwhile, existing methods typically target isolated temporal skills, such as question answering about past events or basic forecasting, and exhibit poor generalization, particularly when dealing with events beyond their knowledge cutoff or requiring creative foresight. To address these limitations, we introduce \textit{Time-R1}, the first framework to endow a moderate-sized (3B-parameter) LLM with comprehensive temporal abilities: understanding, prediction, and creative generation. Our approach features a novel three-stage development path; the first two constitute a \textit{reinforcement learning (RL) curriculum} driven by a meticulously designed dynamic rule-based reward system. This framework progressively builds (1) foundational temporal understanding and logical event-time mappings from historical data, (2) future event prediction skills for events beyond its knowledge cutoff, and finally (3) enables remarkable generalization to creative future scenario generation without any fine-tuning. Strikingly, experiments demonstrate that Time-R1 outperforms models over 200 times larger, including the state-of-the-art 671B DeepSeek-R1, on highly challenging future event prediction and creative scenario generation benchmarks. This work provides strong evidence that thoughtfully engineered, progressive RL fine-tuning allows smaller, efficient models to achieve superior temporal performance, offering a practical and scalable path towards truly time-aware AI. To foster further research, we also release \textit{Time-Bench}, a large-scale multi-task temporal reasoning dataset derived from 10 years of news data, and our series of \textit{Time-R1} checkpoints. 

---
# Teaching Audio-Aware Large Language Models What Does Not Hear: Mitigating Hallucinations through Synthesized Negative Samples 

**Authors**: Chun-Yi Kuan, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.14518)  

**Abstract**: Recent advancements in audio-aware large language models (ALLMs) enable them to process and understand audio inputs. However, these models often hallucinate non-existent sound events, reducing their reliability in real-world applications. To address this, we propose LISTEN (Learning to Identify Sounds Through Extended Negative Samples), a contrastive-like training method that enhances ALLMs' ability to distinguish between present and absent sounds using synthesized data from the backbone LLM. Unlike prior approaches, our method requires no modification to LLM parameters and efficiently integrates audio representations via a lightweight adapter. Experiments show that LISTEN effectively mitigates hallucinations while maintaining impressive performance on existing audio question and reasoning benchmarks. At the same time, it is more efficient in both data and computation. 

---
# Reasoning Models Better Express Their Confidence 

**Authors**: Dongkeun Yoon, Seungone Kim, Sohee Yang, Sunkyoung Kim, Soyeon Kim, Yongil Kim, Eunbi Choi, Yireun Kim, Minjoon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2505.14489)  

**Abstract**: Despite their strengths, large language models (LLMs) often fail to communicate their confidence accurately, making it difficult to assess when they might be wrong and limiting their reliability. In this work, we demonstrate that reasoning models-LLMs that engage in extended chain-of-thought (CoT) reasoning-exhibit superior performance not only in problem-solving but also in accurately expressing their confidence. Specifically, we benchmark six reasoning models across six datasets and find that they achieve strictly better confidence calibration than their non-reasoning counterparts in 33 out of the 36 settings. Our detailed analysis reveals that these gains in calibration stem from the slow thinking behaviors of reasoning models-such as exploring alternative approaches and backtracking-which enable them to adjust their confidence dynamically throughout their CoT, making it progressively more accurate. In particular, we find that reasoning models become increasingly better calibrated as their CoT unfolds, a trend not observed in non-reasoning models. Moreover, removing slow thinking behaviors from the CoT leads to a significant drop in calibration. Lastly, we show that these gains are not exclusive to reasoning models-non-reasoning models also benefit when guided to perform slow thinking via in-context learning. 

---
# Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach 

**Authors**: Oren Sultan, Eitan Stern, Dafna Shahaf  

**Link**: [PDF](https://arxiv.org/pdf/2505.14479)  

**Abstract**: Large language models (LLMs) struggle with formal domains that require rigorous logical deduction and symbolic reasoning, such as mathematical proof generation. We propose a neuro-symbolic approach that combines LLMs' generative strengths with structured components to overcome this challenge. As a proof-of-concept, we focus on geometry problems. Our approach is two-fold: (1) we retrieve analogous problems and use their proofs to guide the LLM, and (2) a formal verifier evaluates the generated proofs and provides feedback, helping the model fix incorrect proofs. We demonstrate that our method significantly improves proof accuracy for OpenAI's o1 model (58%-70% improvement); both analogous problems and the verifier's feedback contribute to these gains. More broadly, shifting to LLMs that generate provably correct conclusions could dramatically improve their reliability, accuracy and consistency, unlocking complex tasks and critical real-world applications that require trustworthiness. 

---
# SAFEPATH: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment 

**Authors**: Wonje Jeung, Sangyeon Yoon, Minsuk Kahng, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2505.14667)  

**Abstract**: Large Reasoning Models (LRMs) have become powerful tools for complex problem solving, but their structured reasoning pathways can lead to unsafe outputs when exposed to harmful prompts. Existing safety alignment methods reduce harmful outputs but can degrade reasoning depth, leading to significant trade-offs in complex, multi-step tasks, and remain vulnerable to sophisticated jailbreak attacks. To address this, we introduce SAFEPATH, a lightweight alignment method that fine-tunes LRMs to emit a short, 8-token Safety Primer at the start of their reasoning, in response to harmful prompts, while leaving the rest of the reasoning process unsupervised. Empirical results across multiple benchmarks indicate that SAFEPATH effectively reduces harmful outputs while maintaining reasoning performance. Specifically, SAFEPATH reduces harmful responses by up to 90.0% and blocks 83.3% of jailbreak attempts in the DeepSeek-R1-Distill-Llama-8B model, while requiring 295.9x less compute than Direct Refusal and 314.1x less than SafeChain. We further introduce a zero-shot variant that requires no fine-tuning. In addition, we provide a comprehensive analysis of how existing methods in LLMs generalize, or fail, when applied to reasoning-centric models, revealing critical gaps and new directions for safer AI. 

---
# Beyond Words: Multimodal LLM Knows When to Speak 

**Authors**: Zikai Liao, Yi Ouyang, Yi-Lun Lee, Chen-Ping Yu, Yi-Hsuan Tsai, Zhaozheng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.14654)  

**Abstract**: While large language model (LLM)-based chatbots have demonstrated strong capabilities in generating coherent and contextually relevant responses, they often struggle with understanding when to speak, particularly in delivering brief, timely reactions during ongoing conversations. This limitation arises largely from their reliance on text input, lacking the rich contextual cues in real-world human dialogue. In this work, we focus on real-time prediction of response types, with an emphasis on short, reactive utterances that depend on subtle, multimodal signals across vision, audio, and text. To support this, we introduce a new multimodal dataset constructed from real-world conversational videos, containing temporally aligned visual, auditory, and textual streams. This dataset enables fine-grained modeling of response timing in dyadic interactions. Building on this dataset, we propose MM-When2Speak, a multimodal LLM-based model that adaptively integrates visual, auditory, and textual context to predict when a response should occur, and what type of response is appropriate. Experiments show that MM-When2Speak significantly outperforms state-of-the-art unimodal and LLM-based baselines, achieving up to a 4x improvement in response timing accuracy over leading commercial LLMs. These results underscore the importance of multimodal inputs for producing timely, natural, and engaging conversational AI. 

---
# TinyV: Reducing False Negatives in Verification Improves RL for LLM Reasoning 

**Authors**: Zhangchen Xu, Yuetai Li, Fengqing Jiang, Bhaskar Ramasubramanian, Luyao Niu, Bill Yuchen Lin, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2505.14625)  

**Abstract**: Reinforcement Learning (RL) has become a powerful tool for enhancing the reasoning abilities of large language models (LLMs) by optimizing their policies with reward signals. Yet, RL's success relies on the reliability of rewards, which are provided by verifiers. In this paper, we expose and analyze a widespread problem--false negatives--where verifiers wrongly reject correct model outputs. Our in-depth study of the Big-Math-RL-Verified dataset reveals that over 38% of model-generated responses suffer from false negatives, where the verifier fails to recognize correct answers. We show, both empirically and theoretically, that these false negatives severely impair RL training by depriving the model of informative gradient signals and slowing convergence. To mitigate this, we propose tinyV, a lightweight LLM-based verifier that augments existing rule-based methods, which dynamically identifies potential false negatives and recovers valid responses to produce more accurate reward estimates. Across multiple math-reasoning benchmarks, integrating TinyV boosts pass rates by up to 10% and accelerates convergence relative to the baseline. Our findings highlight the critical importance of addressing verifier false negatives and offer a practical approach to improve RL-based fine-tuning of LLMs. Our code is available at this https URL. 

---
# Causal Cartographer: From Mapping to Reasoning Over Counterfactual Worlds 

**Authors**: Gaël Gendron, Jože M. Rožanec, Michael Witbrock, Gillian Dobbie  

**Link**: [PDF](https://arxiv.org/pdf/2505.14396)  

**Abstract**: Causal world models are systems that can answer counterfactual questions about an environment of interest, i.e. predict how it would have evolved if an arbitrary subset of events had been realized differently. It requires understanding the underlying causes behind chains of events and conducting causal inference for arbitrary unseen distributions. So far, this task eludes foundation models, notably large language models (LLMs), which do not have demonstrated causal reasoning capabilities beyond the memorization of existing causal relationships. Furthermore, evaluating counterfactuals in real-world applications is challenging since only the factual world is observed, limiting evaluation to synthetic datasets. We address these problems by explicitly extracting and modeling causal relationships and propose the Causal Cartographer framework. First, we introduce a graph retrieval-augmented generation agent tasked to retrieve causal relationships from data. This approach allows us to construct a large network of real-world causal relationships that can serve as a repository of causal knowledge and build real-world counterfactuals. In addition, we create a counterfactual reasoning agent constrained by causal relationships to perform reliable step-by-step causal inference. We show that our approach can extract causal knowledge and improve the robustness of LLMs for causal reasoning tasks while reducing inference costs and spurious correlations. 

---
# Is Your Prompt Safe? Investigating Prompt Injection Attacks Against Open-Source LLMs 

**Authors**: Jiawen Wang, Pritha Gupta, Ivan Habernal, Eyke Hüllermeier  

**Link**: [PDF](https://arxiv.org/pdf/2505.14368)  

**Abstract**: Recent studies demonstrate that Large Language Models (LLMs) are vulnerable to different prompt-based attacks, generating harmful content or sensitive information. Both closed-source and open-source LLMs are underinvestigated for these attacks. This paper studies effective prompt injection attacks against the $\mathbf{14}$ most popular open-source LLMs on five attack benchmarks. Current metrics only consider successful attacks, whereas our proposed Attack Success Probability (ASP) also captures uncertainty in the model's response, reflecting ambiguity in attack feasibility. By comprehensively analyzing the effectiveness of prompt injection attacks, we propose a simple and effective hypnotism attack; results show that this attack causes aligned language models, including Stablelm2, Mistral, Openchat, and Vicuna, to generate objectionable behaviors, achieving around $90$% ASP. They also indicate that our ignore prefix attacks can break all $\mathbf{14}$ open-source LLMs, achieving over $60$% ASP on a multi-categorical dataset. We find that moderately well-known LLMs exhibit higher vulnerability to prompt injection attacks, highlighting the need to raise public awareness and prioritize efficient mitigation strategies. 

---
# PRL: Prompts from Reinforcement Learning 

**Authors**: Paweł Batorski, Adrian Kosmala, Paul Swoboda  

**Link**: [PDF](https://arxiv.org/pdf/2505.14412)  

**Abstract**: Effective prompt engineering remains a central challenge in fully harnessing the capabilities of LLMs. While well-designed prompts can dramatically enhance performance, crafting them typically demands expert intuition and a nuanced understanding of the task. Moreover, the most impactful prompts often hinge on subtle semantic cues, ones that may elude human perception but are crucial for guiding LLM behavior. In this paper, we introduce PRL (Prompts from Reinforcement Learning), a novel RL-based approach for automatic prompt generation. Unlike previous methods, PRL can produce novel few-shot examples that were not seen during training. Our approach achieves state-of-the-art performance across a range of benchmarks, including text classification, simplification, and summarization. On the classification task, it surpasses prior methods by 2.58% over APE and 1.00% over EvoPrompt. Additionally, it improves the average ROUGE scores on the summarization task by 4.32 over APE and by 2.12 over EvoPrompt and the SARI score on simplification by 6.93 over APE and by 6.01 over EvoPrompt. Our code is available at this https URL . 

---
# Evaluating Reasoning LLMs for Suicide Screening with the Columbia-Suicide Severity Rating Scale 

**Authors**: Avinash Patil, Siru Tao, Amardeep Gedhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13480)  

**Abstract**: Suicide prevention remains a critical public health challenge. While online platforms such as Reddit's r/SuicideWatch have historically provided spaces for individuals to express suicidal thoughts and seek community support, the advent of large language models (LLMs) introduces a new paradigm-where individuals may begin disclosing ideation to AI systems instead of humans. This study evaluates the capability of LLMs to perform automated suicide risk assessment using the Columbia-Suicide Severity Rating Scale (C-SSRS). We assess the zero-shot performance of six models-including Claude, GPT, Mistral, and LLaMA-in classifying posts across a 7-point severity scale (Levels 0-6). Results indicate that Claude and GPT closely align with human annotations, while Mistral achieves the lowest ordinal prediction error. Most models exhibit ordinal sensitivity, with misclassifications typically occurring between adjacent severity levels. We further analyze confusion patterns, misclassification sources, and ethical considerations, underscoring the importance of human oversight, transparency, and cautious deployment. Full code and supplementary materials are available at this https URL. 

---
# ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions 

**Authors**: Bufang Yang, Lilin Xu, Liekang Zeng, Kaiwei Liu, Siyang Jiang, Wenrui Lu, Hongkai Chen, Xiaofan Jiang, Guoliang Xing, Zhenyu Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14668)  

**Abstract**: Recent advances in Large Language Models (LLMs) have propelled intelligent agents from reactive responses to proactive support. While promising, existing proactive agents either rely exclusively on observations from enclosed environments (e.g., desktop UIs) with direct LLM inference or employ rule-based proactive notifications, leading to suboptimal user intent understanding and limited functionality for proactive service. In this paper, we introduce ContextAgent, the first context-aware proactive agent that incorporates extensive sensory contexts to enhance the proactive capabilities of LLM agents. ContextAgent first extracts multi-dimensional contexts from massive sensory perceptions on wearables (e.g., video and audio) to understand user intentions. ContextAgent then leverages the sensory contexts and the persona contexts from historical data to predict the necessity for proactive services. When proactive assistance is needed, ContextAgent further automatically calls the necessary tools to assist users unobtrusively. To evaluate this new task, we curate ContextAgentBench, the first benchmark for evaluating context-aware proactive LLM agents, covering 1,000 samples across nine daily scenarios and twenty tools. Experiments on ContextAgentBench show that ContextAgent outperforms baselines by achieving up to 8.5% and 6.0% higher accuracy in proactive predictions and tool calling, respectively. We hope our research can inspire the development of more advanced, human-centric, proactive AI assistants. 

---
# AAPO: Enhance the Reasoning Capabilities of LLMs with Advantage Momentum 

**Authors**: Jian Xiong, Jingbo Zhou, Jingyong Ye, Dejing Dou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14264)  

**Abstract**: Reinforcement learning (RL) has emerged as an effective approach for enhancing the reasoning capabilities of large language models (LLMs), especially in scenarios where supervised fine-tuning (SFT) falls short due to limited chain-of-thought (CoT) data. Among RL-based post-training methods, group relative advantage estimation, as exemplified by Group Relative Policy Optimization (GRPO), has attracted considerable attention for eliminating the dependency on the value model, thereby simplifying training compared to traditional approaches like Proximal Policy Optimization (PPO). However, we observe that exsiting group relative advantage estimation method still suffers from training inefficiencies, particularly when the estimated advantage approaches zero. To address this limitation, we propose Advantage-Augmented Policy Optimization (AAPO), a novel RL algorithm that optimizes the cross-entropy (CE) loss using advantages enhanced through a momentum-based estimation scheme. This approach effectively mitigates the inefficiencies associated with group relative advantage estimation. Experimental results on multiple mathematical reasoning benchmarks demonstrate the superior performance of AAPO. 

---
# s3: You Don't Need That Much Data to Train a Search Agent via RL 

**Authors**: Pengcheng Jiang, Xueqiang Xu, Jiacheng Lin, Jinfeng Xiao, Zifeng Wang, Jimeng Sun, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.14146)  

**Abstract**: Retrieval-augmented generation (RAG) systems empower large language models (LLMs) to access external knowledge during inference. Recent advances have enabled LLMs to act as search agents via reinforcement learning (RL), improving information acquisition through multi-turn interactions with retrieval engines. However, existing approaches either optimize retrieval using search-only metrics (e.g., NDCG) that ignore downstream utility or fine-tune the entire LLM to jointly reason and retrieve-entangling retrieval with generation and limiting the real search utility and compatibility with frozen or proprietary models. In this work, we propose s3, a lightweight, model-agnostic framework that decouples the searcher from the generator and trains the searcher using a Gain Beyond RAG reward: the improvement in generation accuracy over naive RAG. s3 requires only 2.4k training samples to outperform baselines trained on over 70x more data, consistently delivering stronger downstream performance across six general QA and five medical QA benchmarks. 

---
# ProMind-LLM: Proactive Mental Health Care via Causal Reasoning with Sensor Data 

**Authors**: Xinzhe Zheng, Sijie Ji, Jiawei Sun, Renqi Chen, Wei Gao, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2505.14038)  

**Abstract**: Mental health risk is a critical global public health challenge, necessitating innovative and reliable assessment methods. With the development of large language models (LLMs), they stand out to be a promising tool for explainable mental health care applications. Nevertheless, existing approaches predominantly rely on subjective textual mental records, which can be distorted by inherent mental uncertainties, leading to inconsistent and unreliable predictions. To address these limitations, this paper introduces ProMind-LLM. We investigate an innovative approach integrating objective behavior data as complementary information alongside subjective mental records for robust mental health risk assessment. Specifically, ProMind-LLM incorporates a comprehensive pipeline that includes domain-specific pretraining to tailor the LLM for mental health contexts, a self-refine mechanism to optimize the processing of numerical behavioral data, and causal chain-of-thought reasoning to enhance the reliability and interpretability of its predictions. Evaluations of two real-world datasets, PMData and Globem, demonstrate the effectiveness of our proposed methods, achieving substantial improvements over general LLMs. We anticipate that ProMind-LLM will pave the way for more dependable, interpretable, and scalable mental health case solutions. 

---
# MLZero: A Multi-Agent System for End-to-end Machine Learning Automation 

**Authors**: Haoyang Fang, Boran Han, Nick Erickson, Xiyuan Zhang, Su Zhou, Anirudh Dagar, Jiani Zhang, Ali Caner Turkmen, Cuixiong Hu, Huzefa Rangwala, Ying Nian Wu, Bernie Wang, George Karypis  

**Link**: [PDF](https://arxiv.org/pdf/2505.13941)  

**Abstract**: Existing AutoML systems have advanced the automation of machine learning (ML); however, they still require substantial manual configuration and expert input, particularly when handling multimodal data. We introduce MLZero, a novel multi-agent framework powered by Large Language Models (LLMs) that enables end-to-end ML automation across diverse data modalities with minimal human intervention. A cognitive perception module is first employed, transforming raw multimodal inputs into perceptual context that effectively guides the subsequent workflow. To address key limitations of LLMs, such as hallucinated code generation and outdated API knowledge, we enhance the iterative code generation process with semantic and episodic memory. MLZero demonstrates superior performance on MLE-Bench Lite, outperforming all competitors in both success rate and solution quality, securing six gold medals. Additionally, when evaluated on our Multimodal AutoML Agent Benchmark, which includes 25 more challenging tasks spanning diverse data modalities, MLZero outperforms the competing methods by a large margin with a success rate of 0.92 (+263.6\%) and an average rank of 2.28. Our approach maintains its robust effectiveness even with a compact 8B LLM, outperforming full-size systems from existing solutions. 

---
# InfiFPO: Implicit Model Fusion via Preference Optimization in Large Language Models 

**Authors**: Yanggan Gu, Zhaoyi Yan, Yuanyi Wang, Yiming Zhang, Qi Zhou, Fei Wu, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13878)  

**Abstract**: Model fusion combines multiple Large Language Models (LLMs) with different strengths into a more powerful, integrated model through lightweight training methods. Existing works on model fusion focus primarily on supervised fine-tuning (SFT), leaving preference alignment (PA) --a critical phase for enhancing LLM performance--largely unexplored. The current few fusion methods on PA phase, like WRPO, simplify the process by utilizing only response outputs from source models while discarding their probability information. To address this limitation, we propose InfiFPO, a preference optimization method for implicit model fusion. InfiFPO replaces the reference model in Direct Preference Optimization (DPO) with a fused source model that synthesizes multi-source probabilities at the sequence level, circumventing complex vocabulary alignment challenges in previous works and meanwhile maintaining the probability information. By introducing probability clipping and max-margin fusion strategies, InfiFPO enables the pivot model to align with human preferences while effectively distilling knowledge from source models. Comprehensive experiments on 11 widely-used benchmarks demonstrate that InfiFPO consistently outperforms existing model fusion and preference optimization methods. When using Phi-4 as the pivot model, InfiFPO improve its average performance from 79.95 to 83.33 on 11 benchmarks, significantly improving its capabilities in mathematics, coding, and reasoning tasks. 

---
# PandaGuard: Systematic Evaluation of LLM Safety in the Era of Jailbreaking Attacks 

**Authors**: Guobin Shen, Dongcheng Zhao, Linghao Feng, Xiang He, Jihang Wang, Sicheng Shen, Haibo Tong, Yiting Dong, Jindong Li, Xiang Zheng, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.13862)  

**Abstract**: Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety. 

---
# Ice Cream Doesn't Cause Drowning: Benchmarking LLMs Against Statistical Pitfalls in Causal Inference 

**Authors**: Jin Du, Li Chen, Xun Xian, An Luo, Fangqiao Tian, Ganghua Wang, Charles Doss, Xiaotong Shen, Jie Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.13770)  

**Abstract**: Reliable causal inference is essential for making decisions in high-stakes areas like medicine, economics, and public policy. However, it remains unclear whether large language models (LLMs) can handle rigorous and trustworthy statistical causal inference. Current benchmarks usually involve simplified tasks. For example, these tasks might only ask LLMs to identify semantic causal relationships or draw conclusions directly from raw data. As a result, models may overlook important statistical pitfalls, such as Simpson's paradox or selection bias. This oversight limits the applicability of LLMs in the real world. To address these limitations, we propose CausalPitfalls, a comprehensive benchmark designed to rigorously evaluate the capability of LLMs in overcoming common causal inference pitfalls. Our benchmark features structured challenges across multiple difficulty levels, each paired with grading rubrics. This approach allows us to quantitatively measure both causal reasoning capabilities and the reliability of LLMs' responses. We evaluate models using two protocols: (1) direct prompting, which assesses intrinsic causal reasoning, and (2) code-assisted prompting, where models generate executable code for explicit statistical analysis. Additionally, we validate the effectiveness of this judge by comparing its scoring with assessments from human experts. Our results reveal significant limitations in current LLMs when performing statistical causal inference. The CausalPitfalls benchmark provides essential guidance and quantitative metrics to advance the development of trustworthy causal reasoning systems. 

---
# Advancing Software Quality: A Standards-Focused Review of LLM-Based Assurance Techniques 

**Authors**: Avinash Patil  

**Link**: [PDF](https://arxiv.org/pdf/2505.13766)  

**Abstract**: Software Quality Assurance (SQA) is critical for delivering reliable, secure, and efficient software products. The Software Quality Assurance Process aims to provide assurance that work products and processes comply with predefined provisions and plans. Recent advancements in Large Language Models (LLMs) present new opportunities to enhance existing SQA processes by automating tasks like requirement analysis, code review, test generation, and compliance checks. Simultaneously, established standards such as ISO/IEC 12207, ISO/IEC 25010, ISO/IEC 5055, ISO 9001/ISO/IEC 90003, CMMI, and TMM provide structured frameworks for ensuring robust quality practices. This paper surveys the intersection of LLM-based SQA methods and these recognized standards, highlighting how AI-driven solutions can augment traditional approaches while maintaining compliance and process maturity. We first review the foundational software quality standards and the technical fundamentals of LLMs in software engineering. Next, we explore various LLM-based SQA applications, including requirement validation, defect detection, test generation, and documentation maintenance. We then map these applications to key software quality frameworks, illustrating how LLMs can address specific requirements and metrics within each standard. Empirical case studies and open-source initiatives demonstrate the practical viability of these methods. At the same time, discussions on challenges (e.g., data privacy, model bias, explainability) underscore the need for deliberate governance and auditing. Finally, we propose future directions encompassing adaptive learning, privacy-focused deployments, multimodal analysis, and evolving standards for AI-driven software quality. 

---
# Structured Agent Distillation for Large Language Model 

**Authors**: Jun Liu, Zhenglun Kong, Peiyan Dong, Changdi Yang, Tianqi Li, Hao Tang, Geng Yuan, Wei Niu, Wenbin Zhang, Pu Zhao, Xue Lin, Dong Huang, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13820)  

**Abstract**: Large language models (LLMs) exhibit strong capabilities as decision-making agents by interleaving reasoning and actions, as seen in ReAct-style frameworks. Yet, their practical deployment is constrained by high inference costs and large model sizes. We propose Structured Agent Distillation, a framework that compresses large LLM-based agents into smaller student models while preserving both reasoning fidelity and action consistency. Unlike standard token-level distillation, our method segments trajectories into {[REASON]} and {[ACT]} spans, applying segment-specific losses to align each component with the teacher's behavior. This structure-aware supervision enables compact agents to better replicate the teacher's decision process. Experiments on ALFWorld, HotPotQA-ReAct, and WebShop show that our approach consistently outperforms token-level and imitation learning baselines, achieving significant compression with minimal performance drop. Scaling and ablation results further highlight the importance of span-level alignment for efficient and deployable agents. 

---
# SafetyNet: Detecting Harmful Outputs in LLMs by Modeling and Monitoring Deceptive Behaviors 

**Authors**: Maheep Chaudhary, Fazl Barez  

**Link**: [PDF](https://arxiv.org/pdf/2505.14300)  

**Abstract**: High-risk industries like nuclear and aviation use real-time monitoring to detect dangerous system conditions. Similarly, Large Language Models (LLMs) need monitoring safeguards. We propose a real-time framework to predict harmful AI outputs before they occur by using an unsupervised approach that treats normal behavior as the baseline and harmful outputs as outliers. Our study focuses specifically on backdoor-triggered responses -- where specific input phrases activate hidden vulnerabilities causing the model to generate unsafe content like violence, pornography, or hate speech. We address two key challenges: (1) identifying true causal indicators rather than surface correlations, and (2) preventing advanced models from deception -- deliberately evading monitoring systems. Hence, we approach this problem from an unsupervised lens by drawing parallels to human deception: just as humans exhibit physical indicators while lying, we investigate whether LLMs display distinct internal behavioral signatures when generating harmful content. Our study addresses two critical challenges: 1) designing monitoring systems that capture true causal indicators rather than superficial correlations; and 2)preventing intentional evasion by increasingly capable "Future models''. Our findings show that models can produce harmful content through causal mechanisms and can become deceptive by: (a) alternating between linear and non-linear representations, and (b) modifying feature relationships. To counter this, we developed Safety-Net -- a multi-detector framework that monitors different representation dimensions, successfully detecting harmful behavior even when information is shifted across representational spaces to evade individual monitors. Our evaluation shows 96% accuracy in detecting harmful cases using our unsupervised ensemble approach. 

---
# Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations 

**Authors**: Li Ji-An, Hua-Dong Xiong, Robert C. Wilson, Marcelo G. Mattar, Marcus K. Benna  

**Link**: [PDF](https://arxiv.org/pdf/2505.13763)  

**Abstract**: Large language models (LLMs) can sometimes report the strategies they actually use to solve tasks, but they can also fail to do so. This suggests some degree of metacognition -- the capacity to monitor one's own cognitive processes for subsequent reporting and self-control. Metacognitive abilities enhance AI capabilities but raise safety concerns, as models might obscure their internal processes to evade neural-activation-based oversight mechanisms designed to detect harmful behaviors. Given society's increased reliance on these models, it is critical that we understand the limits of their metacognitive abilities, particularly their ability to monitor their internal activations. To address this, we introduce a neuroscience-inspired neurofeedback paradigm designed to quantify the ability of LLMs to explicitly report and control their activation patterns. By presenting models with sentence-label pairs where labels correspond to sentence-elicited internal activations along specific directions in the neural representation space, we demonstrate that LLMs can learn to report and control these activations. The performance varies with several factors: the number of example pairs provided, the semantic interpretability of the target neural direction, and the variance explained by that direction. These results reveal a "metacognitive space" with dimensionality much lower than the model's neural space, suggesting LLMs can monitor only a subset of their neural mechanisms. Our findings provide empirical evidence quantifying metacognitive capabilities in LLMs, with significant implications for AI safety. 

---
# Guided Search Strategies in Non-Serializable Environments with Applications to Software Engineering Agents 

**Authors**: Karina Zainullina, Alexander Golubev, Maria Trofimova, Sergei Polezhaev, Ibragim Badertdinov, Daria Litvintseva, Simon Karasik, Filipp Fisin, Sergei Skvortsov, Maksim Nekrashevich, Anton Shevtsov, Boris Yangel  

**Link**: [PDF](https://arxiv.org/pdf/2505.13652)  

**Abstract**: Large language models (LLMs) have recently achieved remarkable results in complex multi-step tasks, such as mathematical reasoning and agentic software engineering. However, they often struggle to maintain consistent performance across multiple solution attempts. One effective approach to narrow the gap between average-case and best-case performance is guided test-time search, which explores multiple solution paths to identify the most promising one. Unfortunately, effective search techniques (e.g. MCTS) are often unsuitable for non-serializable RL environments, such as Docker containers, where intermediate environment states cannot be easily saved and restored. We investigate two complementary search strategies applicable to such environments: 1-step lookahead and trajectory selection, both guided by a learned action-value function estimator. On the SWE-bench Verified benchmark, a key testbed for agentic software engineering, we find these methods to double the average success rate of a fine-tuned Qwen-72B model, achieving 40.8%, the new state-of-the-art for open-weights models. Additionally, we show that these techniques are transferable to more advanced closed models, yielding similar improvements with GPT-4o. 

---
# Warm Up Before You Train: Unlocking General Reasoning in Resource-Constrained Settings 

**Authors**: Safal Shrestha, Minwu Kim, Aadim Nepal, Anubhav Shrestha, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2505.13718)  

**Abstract**: Designing effective reasoning-capable LLMs typically requires training using Reinforcement Learning with Verifiable Rewards (RLVR) or distillation with carefully curated Long Chain of Thoughts (CoT), both of which depend heavily on extensive training data. This creates a major challenge when the amount of quality training data is scarce. We propose a sample-efficient, two-stage training strategy to develop reasoning LLMs under limited supervision. In the first stage, we "warm up" the model by distilling Long CoTs from a toy domain, namely, Knights \& Knaves (K\&K) logic puzzles to acquire general reasoning skills. In the second stage, we apply RLVR to the warmed-up model using a limited set of target-domain examples. Our experiments demonstrate that this two-phase approach offers several benefits: $(i)$ the warmup phase alone facilitates generalized reasoning, leading to performance improvements across a range of tasks, including MATH, HumanEval$^{+}$, and MMLU-Pro. $(ii)$ When both the base model and the warmed-up model are RLVR trained on the same small dataset ($\leq100$ examples), the warmed-up model consistently outperforms the base model; $(iii)$ Warming up before RLVR training allows a model to maintain cross-domain generalizability even after training on a specific domain; $(iv)$ Introducing warmup in the pipeline improves not only accuracy but also overall sample efficiency during RLVR training. The results in this paper highlight the promise of warmup for building robust reasoning LLMs in data-scarce environments. 

---
# RAR: Setting Knowledge Tripwires for Retrieval Augmented Rejection 

**Authors**: Tommaso Mario Buonocore, Enea Parimbelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.13581)  

**Abstract**: Content moderation for large language models (LLMs) remains a significant challenge, requiring flexible and adaptable solutions that can quickly respond to emerging threats. This paper introduces Retrieval Augmented Rejection (RAR), a novel approach that leverages a retrieval-augmented generation (RAG) architecture to dynamically reject unsafe user queries without model retraining. By strategically inserting and marking malicious documents into the vector database, the system can identify and reject harmful requests when these documents are retrieved. Our preliminary results show that RAR achieves comparable performance to embedded moderation in LLMs like Claude 3.5 Sonnet, while offering superior flexibility and real-time customization capabilities, a fundamental feature to timely address critical vulnerabilities. This approach introduces no architectural changes to existing RAG systems, requiring only the addition of specially crafted documents and a simple rejection mechanism based on retrieval results. 

---
# AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference 

**Authors**: Shitong Duan, Xiaoyuan Yi, Peng Zhang, Dongkuan Xu, Jing Yao, Tun Lu, Ning Gu, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.13531)  

**Abstract**: Assessing Large Language Models (LLMs)' underlying value differences enables comprehensive comparison of their misalignment, cultural adaptability, and biases. Nevertheless, current value measurement datasets face the informativeness challenge: with often outdated, contaminated, or generic test questions, they can only capture the shared value orientations among different LLMs, leading to saturated and thus uninformative results. To address this problem, we introduce AdAEM, a novel, self-extensible assessment framework for revealing LLMs' inclinations. Distinct from previous static benchmarks, AdAEM can automatically and adaptively generate and extend its test questions. This is achieved by probing the internal value boundaries of a diverse set of LLMs developed across cultures and time periods in an in-context optimization manner. The optimization process theoretically maximizes an information-theoretic objective to extract the latest or culturally controversial topics, providing more distinguishable and informative insights about models' value differences. In this way, AdAEM is able to co-evolve with the development of LLMs, consistently tracking their value dynamics. Using AdAEM, we generate 12,310 questions grounded in Schwartz Value Theory, conduct an extensive analysis to manifest our method's validity and effectiveness, and benchmark the values of 16 LLMs, laying the groundwork for better value research. 

---
# Can AI Freelancers Compete? Benchmarking Earnings, Reliability, and Task Success at Scale 

**Authors**: David Noever, Forrest McKee  

**Link**: [PDF](https://arxiv.org/pdf/2505.13511)  

**Abstract**: This study explores Large Language Models (LLMs) as autonomous agents for real-world tasks, including freelance software development. This work presents a new benchmark that evaluates LLMs on freelance programming and data analysis tasks derived from economic data. We construct the benchmark using synthetic tasks created from a Kaggle Freelancer dataset of job postings, with all job prices standardized to USD (median fixed-project price around $250, and an average of $306). Each task is accompanied by structured input-output test cases and an estimated price tag, enabling automated correctness checking and a monetary performance valuation. This approach is inspired by OpenAI's recent SWE-Lancer benchmark (1,400 real Upwork tasks worth $1M total). Still, our framework simplifies evaluation using programmatically testable tasks and predicted price values, making it highly scalable and repeatable. On this benchmark, we evaluate four modern LLMs - Claude 3.5 Haiku, GPT-4o-mini, Qwen 2.5, and Mistral. We report each model's accuracy (task success rate and test-case pass rate) and the total "freelance earnings" it achieves (sum of prices of solved tasks). Our results show that Claude 3.5 Haiku performs best, earning approximately $1.52 million USD, followed closely by GPT-4o-mini at $1.49 million, then Qwen 2.5 ($1.33M) and Mistral ($0.70M). We analyze the distribution of errors per task and observe that the strongest models solve the most tasks and rarely fail completely on any project. We discuss the implications of these results for the feasibility of AI as a freelance developer, the advantages and limitations of our automated benchmark approach, and the gap between performance on structured tasks versus the true complexity of real-world freelance jobs. 

---
# Contrastive Cross-Course Knowledge Tracing via Concept Graph Guided Knowledge Transfer 

**Authors**: Wenkang Han, Wang Lin, Liya Hu, Zhenlong Dai, Yiyun Zhou, Mengze Li, Zemin Liu, Chang Yao, Jingyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13489)  

**Abstract**: Knowledge tracing (KT) aims to predict learners' future performance based on historical learning interactions. However, existing KT models predominantly focus on data from a single course, limiting their ability to capture a comprehensive understanding of learners' knowledge states. In this paper, we propose TransKT, a contrastive cross-course knowledge tracing method that leverages concept graph guided knowledge transfer to model the relationships between learning behaviors across different courses, thereby enhancing knowledge state estimation. Specifically, TransKT constructs a cross-course concept graph by leveraging zero-shot Large Language Model (LLM) prompts to establish implicit links between related concepts across different courses. This graph serves as the foundation for knowledge transfer, enabling the model to integrate and enhance the semantic features of learners' interactions across courses. Furthermore, TransKT includes an LLM-to-LM pipeline for incorporating summarized semantic features, which significantly improves the performance of Graph Convolutional Networks (GCNs) used for knowledge transfer. Additionally, TransKT employs a contrastive objective that aligns single-course and cross-course knowledge states, thereby refining the model's ability to provide a more robust and accurate representation of learners' overall knowledge states. 

---
# LoRASuite: Efficient LoRA Adaptation Across Large Language Model Upgrades 

**Authors**: Yanan Li, Fanxu Meng, Muhan Zhang, Shiai Zhu, Shangguang Wang, Mengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13515)  

**Abstract**: As Large Language Models (LLMs) are frequently updated, LoRA weights trained on earlier versions quickly become obsolete. The conventional practice of retraining LoRA weights from scratch on the latest model is costly, time-consuming, and environmentally detrimental, particularly as the diversity of LLMs and downstream tasks expands. This motivates a critical question: "How can we efficiently leverage existing LoRA weights to adapt to newer model versions?" To address this, we propose LoRASuite, a modular approach tailored specifically to various types of LLM updates. First, we compute a transfer matrix utilizing known parameters from both old and new LLMs. Next, we allocate corresponding layers and attention heads based on centered kernel alignment and cosine similarity metrics, respectively. A subsequent small-scale, skillful fine-tuning step ensures numerical stability. Experimental evaluations demonstrate that LoRASuite consistently surpasses small-scale vanilla LoRA methods. Notably, on backbone LLMs such as MiniCPM and Qwen, LoRASuite even exceeds the performance of full-scale LoRA retraining, with average improvements of +1.4 and +6.6 points on math tasks, respectively. Additionally, LoRASuite significantly reduces memory consumption by 5.5 GB and computational time by 78.23%. 

---
# Evaluating Large Language Models for Real-World Engineering Tasks 

**Authors**: Rene Heesch, Sebastian Eilermann, Alexander Windmann, Alexander Diedrich, Philipp Rosenthal, Oliver Niggemann  

**Link**: [PDF](https://arxiv.org/pdf/2505.13484)  

**Abstract**: Large Language Models (LLMs) are transformative not only for daily activities but also for engineering tasks. However, current evaluations of LLMs in engineering exhibit two critical shortcomings: (i) the reliance on simplified use cases, often adapted from examination materials where correctness is easily verifiable, and (ii) the use of ad hoc scenarios that insufficiently capture critical engineering competencies. Consequently, the assessment of LLMs on complex, real-world engineering problems remains largely unexplored. This paper addresses this gap by introducing a curated database comprising over 100 questions derived from authentic, production-oriented engineering scenarios, systematically designed to cover core competencies such as product design, prognosis, and diagnosis. Using this dataset, we evaluate four state-of-the-art LLMs, including both cloud-based and locally hosted instances, to systematically investigate their performance on complex engineering tasks. Our results show that LLMs demonstrate strengths in basic temporal and structural reasoning but struggle significantly with abstract reasoning, formal modeling, and context-sensitive engineering logic. 

---
# RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection 

**Authors**: Wenjun Hou, Yi Cheng, Kaishuai Xu, Heng Li, Yan Hu, Wenjie Li, Jiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14318)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in various domains, including radiology report generation. Previous approaches have attempted to utilize multimodal LLMs for this task, enhancing their performance through the integration of domain-specific knowledge retrieval. However, these approaches often overlook the knowledge already embedded within the LLMs, leading to redundant information integration and inefficient utilization of learned representations. To address this limitation, we propose RADAR, a framework for enhancing radiology report generation with supplementary knowledge injection. RADAR improves report generation by systematically leveraging both the internal knowledge of an LLM and externally retrieved information. Specifically, it first extracts the model's acquired knowledge that aligns with expert image-based classification outputs. It then retrieves relevant supplementary knowledge to further enrich this information. Finally, by aggregating both sources, RADAR generates more accurate and informative radiology reports. Extensive experiments on MIMIC-CXR, CheXpert-Plus, and IU X-ray demonstrate that our model outperforms state-of-the-art LLMs in both language quality and clinical accuracy 

---
# BARREL: Boundary-Aware Reasoning for Factual and Reliable LRMs 

**Authors**: Junxiao Yang, Jinzhe Tu, Haoran Liu, Xiaoce Wang, Chujie Zheng, Zhexin Zhang, Shiyao Cui, Caishun Chen, Tiantian He, Hongning Wang, Yew-Soon Ong, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13529)  

**Abstract**: Recent advances in Large Reasoning Models (LRMs) have shown impressive capabilities in mathematical and logical reasoning. However, current LRMs rarely admit ignorance or respond with "I don't know". Instead, they often produce incorrect answers while showing undue confidence, raising concerns about their factual reliability. In this work, we identify two pathological reasoning patterns characterized by overthinking that contribute to the overconfident and incorrect answers: last-minute guessing and second-thought spiraling. To address these issues, we propose BARREL-a novel framework that promotes concise and boundary-aware factual reasoning. Our experiments show that BARREL-training increases the reliability of DeepSeek-R1-Distill-Llama-8B from 39.33% to 61.48%, while still achieving accuracy comparable to models finetuned on reasoning data generated by R1. These results demonstrate that our pilot study is inspiring to build more reliable and factual System 2 LRMs. 

---
# Textual Steering Vectors Can Improve Visual Understanding in Multimodal Large Language Models 

**Authors**: Woody Haosheng Gan, Deqing Fu, Julian Asilis, Ollie Liu, Dani Yogatama, Vatsal Sharan, Robin Jia, Willie Neiswanger  

**Link**: [PDF](https://arxiv.org/pdf/2505.14071)  

**Abstract**: Steering methods have emerged as effective and targeted tools for guiding large language models' (LLMs) behavior without modifying their parameters. Multimodal large language models (MLLMs), however, do not currently enjoy the same suite of techniques, due in part to their recency and architectural diversity. Inspired by this gap, we investigate whether MLLMs can be steered using vectors derived from their text-only LLM backbone, via sparse autoencoders (SAEs), mean shift, and linear probing. We find that text-derived steering consistently enhances multimodal accuracy across diverse MLLM architectures and visual tasks. In particular, mean shift boosts spatial relationship accuracy on CV-Bench by up to +7.3% and counting accuracy by up to +3.3%, outperforming prompting and exhibiting strong generalization to out-of-distribution datasets. These results highlight textual steering vectors as a powerful, efficient mechanism for enhancing grounding in MLLMs with minimal additional data collection and computational overhead. 

---
# Cost-Augmented Monte Carlo Tree Search for LLM-Assisted Planning 

**Authors**: Zihao Zhang, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14656)  

**Abstract**: While LLMs excel at open-ended reasoning, they often struggle with cost-sensitive planning, either treating all actions as having equal cost or failing to stay within strict budgets. In this paper, we introduce Cost-Augmented Monte Carlo Tree Search (CATS), a novel approach that brings explicit cost-awareness into LLM-guided planning. Tight cost constraints push the planner to quickly identify infeasible solutions, while looser constraints encourage optimization for minimal cost. We benchmark top LLMs such as GPT-4.1, Claude-3.7-Sonnet, and DeepSeek-R1, against our CATS planner to evaluate their performance in cost-sensitive scenarios. Our experiments suggest that raw LLMs such as GPT-4.1 often falter under tight budgets, whereas CATS consistently delivers strong performance, achieving higher task success rates and better cost efficiency. CATS provides an effective solution for budget-aware decision-making by combining the reasoning power of LLMs with structured search. 

---
# Let LLMs Break Free from Overthinking via Self-Braking Tuning 

**Authors**: Haoran Zhao, Yuchen Yan, Yongliang Shen, Haolei Xu, Wenqi Zhang, Kaitao Song, Jian Shao, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14604)  

**Abstract**: Large reasoning models (LRMs), such as OpenAI o1 and DeepSeek-R1, have significantly enhanced their reasoning capabilities by generating longer chains of thought, demonstrating outstanding performance across a variety of tasks. However, this performance gain comes at the cost of a substantial increase in redundant reasoning during the generation process, leading to high computational overhead and exacerbating the issue of overthinking. Although numerous existing approaches aim to address the problem of overthinking, they often rely on external interventions. In this paper, we propose a novel framework, Self-Braking Tuning (SBT), which tackles overthinking from the perspective of allowing the model to regulate its own reasoning process, thus eliminating the reliance on external control mechanisms. We construct a set of overthinking identification metrics based on standard answers and design a systematic method to detect redundant reasoning. This method accurately identifies unnecessary steps within the reasoning trajectory and generates training signals for learning self-regulation behaviors. Building on this foundation, we develop a complete strategy for constructing data with adaptive reasoning lengths and introduce an innovative braking prompt mechanism that enables the model to naturally learn when to terminate reasoning at an appropriate point. Experiments across mathematical benchmarks (AIME, AMC, MATH500, GSM8K) demonstrate that our method reduces token consumption by up to 60% while maintaining comparable accuracy to unconstrained models. 

---
# Unearthing Gems from Stones: Policy Optimization with Negative Sample Augmentation for LLM Reasoning 

**Authors**: Zhaohui Yang, Shilei Jiang, Chen Hu, Linjing Li, Shihong Deng, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14403)  

**Abstract**: Recent advances in reasoning language models have witnessed a paradigm shift from short to long CoT pattern. Given the substantial computational cost of rollouts in long CoT models, maximizing the utility of fixed training datasets becomes crucial. Our analysis reveals that negative responses contain valuable components such as self-reflection and error-correction steps, yet primary existing methods either completely discard negative samples (RFT) or apply equal penalization across all tokens (RL), failing to leverage these potential learning signals. In light of this, we propose Behavior Constrained Policy Gradient with Negative Sample Augmentation (BCPG-NSA), a fine-grained offline RL framework that encompasses three stages: 1) sample segmentation, 2) consensus-based step correctness assessment combining LLM and PRM judgers, and 3) policy optimization with NSA designed to effectively mine positive steps within negative samples. Experimental results show that BCPG-NSA outperforms baselines on several challenging math/coding reasoning benchmarks using the same training dataset, achieving improved sample efficiency and demonstrating robustness and scalability when extended to multiple iterations. 

---
# Beyond the First Error: Process Reward Models for Reflective Mathematical Reasoning 

**Authors**: Zhaohui Yang, Chenghua He, Xiaowen Shi, Linjing Li, Qiyue Yin, Shihong Deng, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14391)  

**Abstract**: Many studies focus on data annotation techniques for training effective PRMs. However, current methods encounter a significant issue when applied to long CoT reasoning processes: they tend to focus solely on the first incorrect step and all preceding steps, assuming that all subsequent steps are incorrect. These methods overlook the unique self-correction and reflection mechanisms inherent in long CoT, where correct reasoning steps may still occur after initial reasoning mistakes. To address this issue, we propose a novel data annotation method for PRMs specifically designed to score the long CoT reasoning process. Given that under the reflection pattern, correct and incorrect steps often alternate, we introduce the concepts of Error Propagation and Error Cessation, enhancing PRMs' ability to identify both effective self-correction behaviors and reasoning based on erroneous steps. Leveraging an LLM-based judger for annotation, we collect 1.7 million data samples to train a 7B PRM and evaluate it at both solution and step levels. Experimental results demonstrate that compared to existing open-source PRMs and PRMs trained on open-source datasets, our PRM achieves superior performance across various metrics, including search guidance, BoN, and F1 scores. Compared to widely used MC-based annotation methods, our annotation approach not only achieves higher data efficiency but also delivers superior performance. Detailed analysis is also conducted to demonstrate the stability and generalizability of our method. 

---
# Knowledge Graph Based Repository-Level Code Generation 

**Authors**: Mihir Athale, Vishal Vaddina  

**Link**: [PDF](https://arxiv.org/pdf/2505.14394)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have transformed code generation from natural language queries. However, despite their extensive knowledge and ability to produce high-quality code, LLMs often struggle with contextual accuracy, particularly in evolving codebases. Current code search and retrieval methods frequently lack robustness in both the quality and contextual relevance of retrieved results, leading to suboptimal code generation. This paper introduces a novel knowledge graph-based approach to improve code search and retrieval leading to better quality of code generation in the context of repository-level tasks. The proposed approach represents code repositories as graphs, capturing structural and relational information for enhanced context-aware code generation. Our framework employs a hybrid approach for code retrieval to improve contextual relevance, track inter-file modular dependencies, generate more robust code and ensure consistency with the existing codebase. We benchmark the proposed approach on the Evolutionary Code Benchmark (EvoCodeBench) dataset, a repository-level code generation benchmark, and demonstrate that our method significantly outperforms the baseline approach. These findings suggest that knowledge graph based code generation could advance robust, context-sensitive coding assistance tools. 

---
# DSMentor: Enhancing Data Science Agents with Curriculum Learning and Online Knowledge Accumulation 

**Authors**: He Wang, Alexander Hanbo Li, Yiqun Hu, Sheng Zhang, Hideo Kobayashi, Jiani Zhang, Henry Zhu, Chung-Wei Hang, Patrick Ng  

**Link**: [PDF](https://arxiv.org/pdf/2505.14163)  

**Abstract**: Large language model (LLM) agents have shown promising performance in generating code for solving complex data science problems. Recent studies primarily focus on enhancing in-context learning through improved search, sampling, and planning techniques, while overlooking the importance of the order in which problems are tackled during inference. In this work, we develop a novel inference-time optimization framework, referred to as DSMentor, which leverages curriculum learning -- a strategy that introduces simpler task first and progressively moves to more complex ones as the learner improves -- to enhance LLM agent performance in challenging data science tasks. Our mentor-guided framework organizes data science tasks in order of increasing difficulty and incorporates a growing long-term memory to retain prior experiences, guiding the agent's learning progression and enabling more effective utilization of accumulated knowledge. We evaluate DSMentor through extensive experiments on DSEval and QRData benchmarks. Experiments show that DSMentor using Claude-3.5-Sonnet improves the pass rate by up to 5.2% on DSEval and QRData compared to baseline agents. Furthermore, DSMentor demonstrates stronger causal reasoning ability, improving the pass rate by 8.8% on the causality problems compared to GPT-4 using Program-of-Thoughts prompts. Our work underscores the importance of developing effective strategies for accumulating and utilizing knowledge during inference, mirroring the human learning process and opening new avenues for improving LLM performance through curriculum-based inference optimization. 

---
# SHARP: Synthesizing High-quality Aligned Reasoning Problems for Large Reasoning Models Reinforcement Learning 

**Authors**: Xiong Jun Wu, Zhenduo Zhang, ZuJie Wen, Zhiqiang Zhang, Wang Ren, Lei Shi, Cai Chen, Deng Zhao, Dingnan Jin, Qing Cui, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14147)  

**Abstract**: Training large reasoning models (LRMs) with reinforcement learning in STEM domains is hindered by the scarcity of high-quality, diverse, and verifiable problem sets. Existing synthesis methods, such as Chain-of-Thought prompting, often generate oversimplified or uncheckable data, limiting model advancement on complex tasks. To address these challenges, we introduce SHARP, a unified approach to Synthesizing High-quality Aligned Reasoning Problems for LRMs reinforcement learning with verifiable rewards (RLVR). SHARP encompasses a strategic set of self-alignment principles -- targeting graduate and Olympiad-level difficulty, rigorous logical consistency, and unambiguous, verifiable answers -- and a structured three-phase framework (Alignment, Instantiation, Inference) that ensures thematic diversity and fine-grained control over problem generation. We implement SHARP by leveraging a state-of-the-art LRM to infer and verify challenging STEM questions, then employ a reinforcement learning loop to refine the model's reasoning through verifiable reward signals. Experiments on benchmarks such as GPQA demonstrate that SHARP-augmented training substantially outperforms existing methods, markedly improving complex reasoning accuracy and pushing LRM performance closer to expert-level proficiency. Our contributions include the SHARP strategy, framework design, end-to-end implementation, and experimental evaluation of its effectiveness in elevating LRM reasoning capabilities. 

---
# RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning 

**Authors**: Qianyue Hao, Sibo Li, Jian Yuan, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.14140)  

**Abstract**: Despite rapid advancements in large language models (LLMs), the token-level autoregressive nature constrains their complex reasoning capabilities. To enhance LLM reasoning, inference-time techniques, including Chain/Tree/Graph-of-Thought(s), successfully improve the performance, as they are fairly cost-effective by guiding reasoning through sophisticated logical structures without modifying LLMs' parameters. However, these manually predefined, task-agnostic frameworks are applied uniformly across diverse tasks, lacking adaptability. To improve this, we propose RL-of-Thoughts (RLoT), where we train a lightweight navigator model with reinforcement learning (RL) to adaptively enhance LLM reasoning at inference time. Specifically, we design five basic logic blocks from the perspective of human cognition. During the reasoning process, the trained RL navigator dynamically selects the suitable logic blocks and combines them into task-specific logical structures according to problem characteristics. Experiments across multiple reasoning benchmarks (AIME, MATH, GPQA, etc.) with multiple LLMs (GPT, Llama, Qwen, and DeepSeek) illustrate that RLoT outperforms established inference-time techniques by up to 13.4%. Remarkably, with less than 3K parameters, our RL navigator is able to make sub-10B LLMs comparable to 100B-scale counterparts. Moreover, the RL navigator demonstrates strong transferability: a model trained on one specific LLM-task pair can effectively generalize to unseen LLMs and tasks. Our code is open-source at this https URL for reproducibility. 

---
# MM-Agent: LLM as Agents for Real-world Mathematical Modeling Problem 

**Authors**: Fan Liu, Zherui Yang, Cancheng Liu, Tianrui Song, Xiaofeng Gao, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14148)  

**Abstract**: Mathematical modeling is a cornerstone of scientific discovery and engineering practice, enabling the translation of real-world problems into formal systems across domains such as physics, biology, and economics. Unlike mathematical reasoning, which assumes a predefined formulation, modeling requires open-ended problem analysis, abstraction, and principled formalization. While Large Language Models (LLMs) have shown strong reasoning capabilities, they fall short in rigorous model construction, limiting their utility in real-world problem-solving. To this end, we formalize the task of LLM-powered real-world mathematical modeling, where agents must analyze problems, construct domain-appropriate formulations, and generate complete end-to-end solutions. We introduce MM-Bench, a curated benchmark of 111 problems from the Mathematical Contest in Modeling (MCM/ICM), spanning the years 2000 to 2025 and across ten diverse domains such as physics, biology, and economics. To tackle this task, we propose MM-Agent, an expert-inspired framework that decomposes mathematical modeling into four stages: open-ended problem analysis, structured model formulation, computational problem solving, and report generation. Experiments on MM-Bench show that MM-Agent significantly outperforms baseline agents, achieving an 11.88\% improvement over human expert solutions while requiring only 15 minutes and \$0.88 per task using GPT-4o. Furthermore, under official MCM/ICM protocols, MM-Agent assisted two undergraduate teams in winning the Finalist Award (\textbf{top 2.0\% among 27,456 teams}) in MCM/ICM 2025, demonstrating its practical effectiveness as a modeling copilot. Our code is available at this https URL 

---
# Guarded Query Routing for Large Language Models 

**Authors**: Richard Šléher, William Brach, Tibor Sloboda, Kristián Košťál, Lukas Galke  

**Link**: [PDF](https://arxiv.org/pdf/2505.14524)  

**Abstract**: Query routing, the task to route user queries to different large language model (LLM) endpoints, can be considered as a text classification problem. However, out-of-distribution queries must be handled properly, as those could be questions about unrelated domains, queries in other languages, or even contain unsafe text. Here, we thus study a \emph{guarded} query routing problem, for which we first introduce the Guarded Query Routing Benchmark (GQR-Bench), which covers three exemplary target domains (law, finance, and healthcare), and seven datasets to test robustness against out-of-distribution queries. We then use GQR-Bench to contrast the effectiveness and efficiency of LLM-based routing mechanisms (GPT-4o-mini, Llama-3.2-3B, and Llama-3.1-8B), standard LLM-based guardrail approaches (LlamaGuard and NVIDIA NeMo Guardrails), continuous bag-of-words classifiers (WideMLP, fastText), and traditional machine learning models (SVM, XGBoost). Our results show that WideMLP, enhanced with out-of-domain detection capabilities, yields the best trade-off between accuracy (88\%) and speed (<4ms). The embedding-based fastText excels at speed (<1ms) with acceptable accuracy (80\%), whereas LLMs yield the highest accuracy (91\%) but are comparatively slow (62ms for local Llama-3.1:8B and 669ms for remote GPT-4o-mini calls). Our findings challenge the automatic reliance on LLMs for (guarded) query routing and provide concrete recommendations for practical applications. GQR-Bench will be released as a Python package -- \texttt{gqr}. 

---
# Building a Stable Planner: An Extended Finite State Machine Based Planning Module for Mobile GUI Agent 

**Authors**: Fanglin Mo, Junzhe Chen, Haoxuan Zhu, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14141)  

**Abstract**: Mobile GUI agents execute user commands by directly interacting with the graphical user interface (GUI) of mobile devices, demonstrating significant potential to enhance user convenience. However, these agents face considerable challenges in task planning, as they must continuously analyze the GUI and generate operation instructions step by step. This process often leads to difficulties in making accurate task plans, as GUI agents lack a deep understanding of how to effectively use the target applications, which can cause them to become "lost" during task execution. To address the task planning issue, we propose SPlanner, a plug-and-play planning module to generate execution plans that guide vision language model(VLMs) in executing tasks. The proposed planning module utilizes extended finite state machines (EFSMs) to model the control logits and configurations of mobile applications. It then decomposes a user instruction into a sequence of primary function modeled in EFSMs, and generate the execution path by traversing the EFSMs. We further refine the execution path into a natural language plan using an LLM. The final plan is concise and actionable, and effectively guides VLMs to generate interactive GUI actions to accomplish user tasks. SPlanner demonstrates strong performance on dynamic benchmarks reflecting real-world mobile usage. On the AndroidWorld benchmark, SPlanner achieves a 63.8% task success rate when paired with Qwen2.5-VL-72B as the VLM executor, yielding a 28.8 percentage point improvement compared to using Qwen2.5-VL-72B without planning assistance. 

---
# CoIn: Counting the Invisible Reasoning Tokens in Commercial Opaque LLM APIs 

**Authors**: Guoheng Sun, Ziyao Wang, Bowei Tian, Meng Liu, Zheyu Shen, Shwai He, Yexiao He, Wanghao Ye, Yiting Wang, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13778)  

**Abstract**: As post-training techniques evolve, large language models (LLMs) are increasingly augmented with structured multi-step reasoning abilities, often optimized through reinforcement learning. These reasoning-enhanced models outperform standard LLMs on complex tasks and now underpin many commercial LLM APIs. However, to protect proprietary behavior and reduce verbosity, providers typically conceal the reasoning traces while returning only the final answer. This opacity introduces a critical transparency gap: users are billed for invisible reasoning tokens, which often account for the majority of the cost, yet have no means to verify their authenticity. This opens the door to token count inflation, where providers may overreport token usage or inject synthetic, low-effort tokens to inflate charges. To address this issue, we propose CoIn, a verification framework that audits both the quantity and semantic validity of hidden tokens. CoIn constructs a verifiable hash tree from token embedding fingerprints to check token counts, and uses embedding-based relevance matching to detect fabricated reasoning content. Experiments demonstrate that CoIn, when deployed as a trusted third-party auditor, can effectively detect token count inflation with a success rate reaching up to 94.7%, showing the strong ability to restore billing transparency in opaque LLM services. The dataset and code are available at this https URL. 

---
# TelePlanNet: An AI-Driven Framework for Efficient Telecom Network Planning 

**Authors**: Zongyuan Deng, Yujie Cai, Qing Liu, Shiyao Mu, Bin Lyu, Zhen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13831)  

**Abstract**: The selection of base station sites is a critical challenge in 5G network planning, which requires efficient optimization of coverage, cost, user satisfaction, and practical constraints. Traditional manual methods, reliant on human expertise, suffer from inefficiencies and are limited to an unsatisfied planning-construction consistency. Existing AI tools, despite improving efficiency in certain aspects, still struggle to meet the dynamic network conditions and multi-objective needs of telecom operators' networks. To address these challenges, we propose TelePlanNet, an AI-driven framework tailored for the selection of base station sites, integrating a three-layer architecture for efficient planning and large-scale automation. By leveraging large language models (LLMs) for real-time user input processing and intent alignment with base station planning, combined with training the planning model using the improved group relative policy optimization (GRPO) reinforcement learning, the proposed TelePlanNet can effectively address multi-objective optimization, evaluates candidate sites, and delivers practical solutions. Experiments results show that the proposed TelePlanNet can improve the consistency to 78%, which is superior to the manual methods, providing telecom operators with an efficient and scalable tool that significantly advances cellular network planning. 

---
# Measuring the Faithfulness of Thinking Drafts in Large Reasoning Models 

**Authors**: Zidi Xiong, Chen Shan, Zhenting Qi, Himabindu Lakkaraju  

**Link**: [PDF](https://arxiv.org/pdf/2505.13774)  

**Abstract**: Large Reasoning Models (LRMs) have significantly enhanced their capabilities in complex problem-solving by introducing a thinking draft that enables multi-path Chain-of-Thought explorations before producing final answers. Ensuring the faithfulness of these intermediate reasoning processes is crucial for reliable monitoring, interpretation, and effective control. In this paper, we propose a systematic counterfactual intervention framework to rigorously evaluate thinking draft faithfulness. Our approach focuses on two complementary dimensions: (1) Intra-Draft Faithfulness, which assesses whether individual reasoning steps causally influence subsequent steps and the final draft conclusion through counterfactual step insertions; and (2) Draft-to-Answer Faithfulness, which evaluates whether final answers are logically consistent with and dependent on the thinking draft, by perturbing the draft's concluding logic. We conduct extensive experiments across six state-of-the-art LRMs. Our findings show that current LRMs demonstrate selective faithfulness to intermediate reasoning steps and frequently fail to faithfully align with the draft conclusions. These results underscore the need for more faithful and interpretable reasoning in advanced LRMs. 

---
# DrugPilot: LLM-based Parameterized Reasoning Agent for Drug Discovery 

**Authors**: Kun Li, Zhennan Wu, Shoupeng Wang, Wenbin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13940)  

**Abstract**: In the field of AI4Science, large-scale language models (LLMs) show great potential to parse complex scientific semantics, integrate cross-disciplinary knowledge, and assist critical task research. However, in the field of drug discovery, despite the optimization through professional data pre-training, context window expansion, and internet search, the existing LLMs are still facing challenges such as massive multi-modal and heterogeneous data processing, domain knowledge dynamic updating delay, and insufficient confidence in predicting the results of complex computational tasks. To address these challenges, we propose the DrugPilot, an LLM-based agent with parameterized reasoning for drug discovery. DrugPilot addresses key limitations of traditional end-to-end LLM prediction approaches through its parametric inference architecture. This agent system supports major phases of the drug discovery pipeline, facilitating automated planning and execution of multi-stage research tasks. To address the critical challenge of multi-modal drug data analysis (incorporating both public datasets and user-submitted data), we developed an interactive parameterized memory pool. This innovative component standardizes real-world drug data into parametric representations, simultaneously enabling efficient knowledge retrieval in multi-turn dialogue while mitigating the information loss inherent in text-based data transmission. Additionally, we created a drug instruct dataset across 8 essential drug discovery tasks for model fine-tuning and evaluation. Based on the Berkeley function calling evaluation framework, DrugPilot demonstrated the most advanced tool calling capabilities on our drug discovery tool instruction dataset, outperforming existing agents (e.g., ReAct, LoT). Specifically, it achieves task completion rates of 98.0%, 93.5%, and 64.0% on simple, multiple, and multi-turn tasks, respectively. 

---
# LLM-based Evaluation Policy Extraction for Ecological Modeling 

**Authors**: Qi Cheng, Licheng Liu, Qing Zhu, Runlong Yu, Zhenong Jin, Yiqun Xie, Xiaowei Jia  

**Link**: [PDF](https://arxiv.org/pdf/2505.13794)  

**Abstract**: Evaluating ecological time series is critical for benchmarking model performance in many important applications, including predicting greenhouse gas fluxes, capturing carbon-nitrogen dynamics, and monitoring hydrological cycles. Traditional numerical metrics (e.g., R-squared, root mean square error) have been widely used to quantify the similarity between modeled and observed ecosystem variables, but they often fail to capture domain-specific temporal patterns critical to ecological processes. As a result, these methods are often accompanied by expert visual inspection, which requires substantial human labor and limits the applicability to large-scale evaluation. To address these challenges, we propose a novel framework that integrates metric learning with large language model (LLM)-based natural language policy extraction to develop interpretable evaluation criteria. The proposed method processes pairwise annotations and implements a policy optimization mechanism to generate and combine different assessment metrics. The results obtained on multiple datasets for evaluating the predictions of crop gross primary production and carbon dioxide flux have confirmed the effectiveness of the proposed method in capturing target assessment preferences, including both synthetically generated and expert-annotated model comparisons. The proposed framework bridges the gap between numerical metrics and expert knowledge while providing interpretable evaluation policies that accommodate the diverse needs of different ecosystem modeling studies. 

---
# Language and Thought: The View from LLMs 

**Authors**: Daniel Rothschild  

**Link**: [PDF](https://arxiv.org/pdf/2505.13561)  

**Abstract**: Daniel Dennett speculated in *Kinds of Minds* 1996: "Perhaps the kind of mind you get when you add language to it is so different from the kind of mind you can have without language that calling them both minds is a mistake." Recent work in AI can be seen as testing Dennett's thesis by exploring the performance of AI systems with and without linguistic training. I argue that the success of Large Language Models at inferential reasoning, limited though it may be, supports Dennett's radical view about the effect of language on thought. I suggest it is the abstractness and efficiency of linguistic encoding that lies behind the capacity of LLMs to perform inferences across a wide range of domains. In a slogan, language makes inference computationally tractable. I assess what these results in AI indicate about the role of language in the workings of our own biological minds. 

---
# Prompt Stability Matters: Evaluating and Optimizing Auto-Generated Prompt in General-Purpose Systems 

**Authors**: Ke Chen, Yufei Zhou, Xitong Zhang, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13546)  

**Abstract**: Automatic prompt generation plays a crucial role in enabling general-purpose multi-agent systems to perform diverse tasks autonomously. Existing methods typically evaluate prompts based on their immediate task performance, overlooking the intrinsic qualities that determine their reliability. This outcome-centric view not only limits interpretability but also fails to account for the inherent stochasticity of large language models (LLMs). In this work, we bring attention to prompt stability-the consistency of model responses across repeated executions-as a key factor for building robust and effective prompt generation systems. To quantify this, we propose semantic stability as a criterion for assessing the response consistency of prompts, and fine-tune a LLaMA-based evaluator to measure it automatically across tasks. These components have enabled us to develop the first stability-aware general-purpose prompt generation system that leverages stability feedback to iteratively enhance both prompt quality and system-level performance. Furthermore, we establish a logical chain between prompt stability and task success by analyzing the structural dependencies within our system, proving stability as a necessary condition for effective system-level execution. Empirical results across general and domain-specific tasks demonstrate that our stability-aware framework improves both accuracy and output consistency. By shifting the focus from one-off results to persistent reliability, our work offers a new perspective on prompt design and contributes practical tools for building more trustworthy general-purpose systems. 

---
# FinMaster: A Holistic Benchmark for Mastering Full-Pipeline Financial Workflows with LLMs 

**Authors**: Junzhe Jiang, Chang Yang, Aixin Cui, Sihan Jin, Ruiyu Wang, Bo Li, Xiao Huang, Dongning Sun, Xinrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13533)  

**Abstract**: Financial tasks are pivotal to global economic stability; however, their execution faces challenges including labor intensive processes, low error tolerance, data fragmentation, and tool limitations. Although large language models (LLMs) have succeeded in various natural language processing tasks and have shown potential in automating workflows through reasoning and contextual understanding, current benchmarks for evaluating LLMs in finance lack sufficient domain-specific data, have simplistic task design, and incomplete evaluation frameworks. To address these gaps, this article presents FinMaster, a comprehensive financial benchmark designed to systematically assess the capabilities of LLM in financial literacy, accounting, auditing, and consulting. Specifically, FinMaster comprises three main modules: i) FinSim, which builds simulators that generate synthetic, privacy-compliant financial data for companies to replicate market dynamics; ii) FinSuite, which provides tasks in core financial domains, spanning 183 tasks of various types and difficulty levels; and iii) FinEval, which develops a unified interface for evaluation. Extensive experiments over state-of-the-art LLMs reveal critical capability gaps in financial reasoning, with accuracy dropping from over 90% on basic tasks to merely 40% on complex scenarios requiring multi-step reasoning. This degradation exhibits the propagation of computational errors, where single-metric calculations initially demonstrating 58% accuracy decreased to 37% in multimetric scenarios. To the best of our knowledge, FinMaster is the first benchmark that covers full-pipeline financial workflows with challenging tasks. We hope that FinMaster can bridge the gap between research and industry practitioners, driving the adoption of LLMs in real-world financial practices to enhance efficiency and accuracy. 

---
# AgentSGEN: Multi-Agent LLM in the Loop for Semantic Collaboration and GENeration of Synthetic Data 

**Authors**: Vu Dinh Xuan, Hao Vo, David Murphy, Hoang D. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13466)  

**Abstract**: The scarcity of data depicting dangerous situations presents a major obstacle to training AI systems for safety-critical applications, such as construction safety, where ethical and logistical barriers hinder real-world data collection. This creates an urgent need for an end-to-end framework to generate synthetic data that can bridge this gap. While existing methods can produce synthetic scenes, they often lack the semantic depth required for scene simulations, limiting their effectiveness. To address this, we propose a novel multi-agent framework that employs an iterative, in-the-loop collaboration between two agents: an Evaluator Agent, acting as an LLM-based judge to enforce semantic consistency and safety-specific constraints, and an Editor Agent, which generates and refines scenes based on this guidance. Powered by LLM's capabilities to reasoning and common-sense knowledge, this collaborative design produces synthetic images tailored to safety-critical scenarios. Our experiments suggest this design can generate useful scenes based on realistic specifications that address the shortcomings of prior approaches, balancing safety requirements with visual semantics. This iterative process holds promise for delivering robust, aesthetically sound simulations, offering a potential solution to the data scarcity challenge in multimedia safety applications. 

---
# Multimodal RAG-driven Anomaly Detection and Classification in Laser Powder Bed Fusion using Large Language Models 

**Authors**: Kiarash Naghavi Khanghah, Zhiling Chen, Lela Romeo, Qian Yang, Rajiv Malhotra, Farhad Imani, Hongyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13828)  

**Abstract**: Additive manufacturing enables the fabrication of complex designs while minimizing waste, but faces challenges related to defects and process anomalies. This study presents a novel multimodal Retrieval-Augmented Generation-based framework that automates anomaly detection across various Additive Manufacturing processes leveraging retrieved information from literature, including images and descriptive text, rather than training datasets. This framework integrates text and image retrieval from scientific literature and multimodal generation models to perform zero-shot anomaly identification, classification, and explanation generation in a Laser Powder Bed Fusion setting. The proposed framework is evaluated on four L-PBF manufacturing datasets from Oak Ridge National Laboratory, featuring various printer makes, models, and materials. This evaluation demonstrates the framework's adaptability and generalizability across diverse images without requiring additional training. Comparative analysis using Qwen2-VL-2B and GPT-4o-mini as MLLM within the proposed framework highlights that GPT-4o-mini outperforms Qwen2-VL-2B and proportional random baseline in manufacturing anomalies classification. Additionally, the evaluation of the RAG system confirms that incorporating retrieval mechanisms improves average accuracy by 12% by reducing the risk of hallucination and providing additional information. The proposed framework can be continuously updated by integrating emerging research, allowing seamless adaptation to the evolving landscape of AM technologies. This scalable, automated, and zero-shot-capable framework streamlines AM anomaly analysis, enhancing efficiency and accuracy. 

---
# Latent Flow Transformer 

**Authors**: Yen-Chen Wu, Feng-Ting Liao, Meng-Hsi Chen, Pei-Chen Ho, Farhang Nabiei, Da-shan Shiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14513)  

**Abstract**: Transformers, the standard implementation for large language models (LLMs), typically consist of tens to hundreds of discrete layers. While more layers can lead to better performance, this approach has been challenged as far from efficient, especially given the superiority of continuous layers demonstrated by diffusion and flow-based models for image generation. We propose the Latent Flow Transformer (LFT), which replaces a block of layers with a single learned transport operator trained via flow matching, offering significant compression while maintaining compatibility with the original architecture. Additionally, we address the limitations of existing flow-based methods in \textit{preserving coupling} by introducing the Flow Walking (FW) algorithm. On the Pythia-410M model, LFT trained with flow matching compresses 6 of 24 layers and outperforms directly skipping 2 layers (KL Divergence of LM logits at 0.407 vs. 0.529), demonstrating the feasibility of this design. When trained with FW, LFT further distills 12 layers into one while reducing the KL to 0.736 surpassing that from skipping 3 layers (0.932), significantly narrowing the gap between autoregressive and flow-based generation paradigms. 

---
# Can Large Language Models Really Recognize Your Name? 

**Authors**: Dzung Pham, Peter Kairouz, Niloofar Mireshghallah, Eugene Bagdasarian, Chau Minh Pham, Amir Houmansadr  

**Link**: [PDF](https://arxiv.org/pdf/2505.14549)  

**Abstract**: Large language models (LLMs) are increasingly being used to protect sensitive user data. However, current LLM-based privacy solutions assume that these models can reliably detect personally identifiable information (PII), particularly named entities. In this paper, we challenge that assumption by revealing systematic failures in LLM-based privacy tasks. Specifically, we show that modern LLMs regularly overlook human names even in short text snippets due to ambiguous contexts, which cause the names to be misinterpreted or mishandled. We propose AMBENCH, a benchmark dataset of seemingly ambiguous human names, leveraging the name regularity bias phenomenon, embedded within concise text snippets along with benign prompt injections. Our experiments on modern LLMs tasked to detect PII as well as specialized tools show that recall of ambiguous names drops by 20--40% compared to more recognizable names. Furthermore, ambiguous human names are four times more likely to be ignored in supposedly privacy-preserving summaries generated by LLMs when benign prompt injections are present. These findings highlight the underexplored risks of relying solely on LLMs to safeguard user privacy and underscore the need for a more systematic investigation into their privacy failure modes. 

---
# Exploring Jailbreak Attacks on LLMs through Intent Concealment and Diversion 

**Authors**: Tiehan Cui, Yanxu Mao, Peipei Liu, Congying Liu, Datao You  

**Link**: [PDF](https://arxiv.org/pdf/2505.14316)  

**Abstract**: Although large language models (LLMs) have achieved remarkable advancements, their security remains a pressing concern. One major threat is jailbreak attacks, where adversarial prompts bypass model safeguards to generate harmful or objectionable content. Researchers study jailbreak attacks to understand security and robustness of LLMs. However, existing jailbreak attack methods face two main challenges: (1) an excessive number of iterative queries, and (2) poor generalization across models. In addition, recent jailbreak evaluation datasets focus primarily on question-answering scenarios, lacking attention to text generation tasks that require accurate regeneration of toxic content. To tackle these challenges, we propose two contributions: (1) ICE, a novel black-box jailbreak method that employs Intent Concealment and divErsion to effectively circumvent security constraints. ICE achieves high attack success rates (ASR) with a single query, significantly improving efficiency and transferability across different models. (2) BiSceneEval, a comprehensive dataset designed for assessing LLM robustness in question-answering and text-generation tasks. Experimental results demonstrate that ICE outperforms existing jailbreak techniques, revealing critical vulnerabilities in current defense mechanisms. Our findings underscore the necessity of a hybrid security strategy that integrates predefined security mechanisms with real-time semantic decomposition to enhance the security of LLMs. 

---
# Choosing a Model, Shaping a Future: Comparing LLM Perspectives on Sustainability and its Relationship with AI 

**Authors**: Annika Bush, Meltem Aksoy, Markus Pauly, Greta Ontrup  

**Link**: [PDF](https://arxiv.org/pdf/2505.14435)  

**Abstract**: As organizations increasingly rely on AI systems for decision support in sustainability contexts, it becomes critical to understand the inherent biases and perspectives embedded in Large Language Models (LLMs). This study systematically investigates how five state-of-the-art LLMs -- Claude, DeepSeek, GPT, LLaMA, and Mistral - conceptualize sustainability and its relationship with AI. We administered validated, psychometric sustainability-related questionnaires - each 100 times per model -- to capture response patterns and variability. Our findings revealed significant inter-model differences: For example, GPT exhibited skepticism about the compatibility of AI and sustainability, whereas LLaMA demonstrated extreme techno-optimism with perfect scores for several Sustainable Development Goals (SDGs). Models also diverged in attributing institutional responsibility for AI and sustainability integration, a results that holds implications for technology governance approaches. Our results demonstrate that model selection could substantially influence organizational sustainability strategies, highlighting the need for awareness of model-specific biases when deploying LLMs for sustainability-related decision-making. 

---
# A*-Decoding: Token-Efficient Inference Scaling 

**Authors**: Giannis Chatziveroglou  

**Link**: [PDF](https://arxiv.org/pdf/2505.13672)  

**Abstract**: Inference-time scaling has emerged as a powerful alternative to parameter scaling for improving language model performance on complex reasoning tasks. While existing methods have shown strong performance gains under fixed compute budgets, there has been little focus on optimally utilizing that budget during inference. In this work, we introduce A*-decoding, a search-based inference-time strategy that builds on the A* search algorithm to optimally utilize a fixed compute budget by prioritizing high-quality reasoning paths during generation. We frame language model decoding as a structured search in a state space of partial solutions, applying the A* transition model to identify promising continuations guided by an external process supervision signal. In our experiments, A*-decoding reaches the performance levels of strong inference scaling baselines like best-of-N and particle filtering while using up to 3x fewer tokens and 30% fewer PRM passes under equivalent compute budgets. On the MATH500 and AIME 2024 benchmarks, A*-decoding enables Llama-3.2-1B-Instruct to match the performance of the 70x larger Llama-3.1-70B-Instruct, and allows Qwen3-1.7B to reach o1-like reasoning accuracy. These results highlight the power of structured search in decoding, offering an alternative to brute-force sampling or scale-driven gains. Our work demonstrates how thoughtful inference-time strategies can enhance reasoning in SLMs, pointing toward future advances in more efficient and scalable language model deployment. 

---
# Speculative Decoding Reimagined for Multimodal Large Language Models 

**Authors**: Luxi Lin, Zhihang Lin, Zhanpeng Zeng, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.14260)  

**Abstract**: This paper introduces Multimodal Speculative Decoding (MSD) to accelerate Multimodal Large Language Models (MLLMs) inference. Speculative decoding has been shown to accelerate Large Language Models (LLMs) without sacrificing accuracy. However, current speculative decoding methods for MLLMs fail to achieve the same speedup as they do for LLMs. To address this, we reimagine speculative decoding specifically for MLLMs. Our analysis of MLLM characteristics reveals two key design principles for MSD: (1) Text and visual tokens have fundamentally different characteristics and need to be processed separately during drafting. (2) Both language modeling ability and visual perception capability are crucial for the draft model. For the first principle, MSD decouples text and visual tokens in the draft model, allowing each to be handled based on its own characteristics. For the second principle, MSD uses a two-stage training strategy: In stage one, the draft model is trained on text-only instruction-tuning datasets to improve its language modeling ability. In stage two, MSD gradually introduces multimodal data to enhance the visual perception capability of the draft model. Experiments show that MSD boosts inference speed by up to $2.29\times$ for LLaVA-1.5-7B and up to $2.46\times$ for LLaVA-1.5-13B on multimodal benchmarks, demonstrating its effectiveness. Our code is available at this https URL. 

---
# When LLMs meet open-world graph learning: a new perspective for unlabeled data uncertainty 

**Authors**: Yanzhe Wen, Xunkai Li, Qi Zhang, Zhu Lei, Guang Zeng, Rong-Hua Li, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13989)  

**Abstract**: Recently, large language models (LLMs) have significantly advanced text-attributed graph (TAG) learning. However, existing methods inadequately handle data uncertainty in open-world scenarios, especially concerning limited labeling and unknown-class nodes. Prior solutions typically rely on isolated semantic or structural approaches for unknown-class rejection, lacking effective annotation pipelines. To address these limitations, we propose Open-world Graph Assistant (OGA), an LLM-based framework that combines adaptive label traceability, which integrates semantics and topology for unknown-class rejection, and a graph label annotator to enable model updates using newly annotated nodes. Comprehensive experiments demonstrate OGA's effectiveness and practicality. 

---
# APEX: Empowering LLMs with Physics-Based Task Planning for Real-time Insight 

**Authors**: Wanjing Huang, Weixiang Yan, Zhen Zhang, Ambuj Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.13921)  

**Abstract**: Large Language Models (LLMs) demonstrate strong reasoning and task planning capabilities but remain fundamentally limited in physical interaction modeling. Existing approaches integrate perception via Vision-Language Models (VLMs) or adaptive decision-making through Reinforcement Learning (RL), but they fail to capture dynamic object interactions or require task-specific training, limiting their real-world applicability. We introduce APEX (Anticipatory Physics-Enhanced Execution), a framework that equips LLMs with physics-driven foresight for real-time task planning. APEX constructs structured graphs to identify and model the most relevant dynamic interactions in the environment, providing LLMs with explicit physical state updates. Simultaneously, APEX provides low-latency forward simulations of physically feasible actions, allowing LLMs to select optimal strategies based on predictive outcomes rather than static observations. We evaluate APEX on three benchmarks designed to assess perception, prediction, and decision-making: (1) Physics Reasoning Benchmark, testing causal inference and object motion prediction; (2) Tetris, evaluating whether physics-informed prediction enhances decision-making performance in long-horizon planning tasks; (3) Dynamic Obstacle Avoidance, assessing the immediate integration of perception and action feasibility analysis. APEX significantly outperforms standard LLMs and VLM-based models, demonstrating the necessity of explicit physics reasoning for bridging the gap between language-based intelligence and real-world task execution. The source code and experiment setup are publicly available at this https URL . 

---
# SayCoNav: Utilizing Large Language Models for Adaptive Collaboration in Decentralized Multi-Robot Navigation 

**Authors**: Abhinav Rajvanshi, Pritish Sahu, Tixiao Shan, Karan Sikka, Han-Pang Chiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13729)  

**Abstract**: Adaptive collaboration is critical to a team of autonomous robots to perform complicated navigation tasks in large-scale unknown environments. An effective collaboration strategy should be determined and adapted according to each robot's skills and current status to successfully achieve the shared goal. We present SayCoNav, a new approach that leverages large language models (LLMs) for automatically generating this collaboration strategy among a team of robots. Building on the collaboration strategy, each robot uses the LLM to generate its plans and actions in a decentralized way. By sharing information to each other during navigation, each robot also continuously updates its step-by-step plans accordingly. We evaluate SayCoNav on Multi-Object Navigation (MultiON) tasks, that require the team of the robots to utilize their complementary strengths to efficiently search multiple different objects in unknown environments. By validating SayCoNav with varied team compositions and conditions against baseline methods, our experimental results show that SayCoNav can improve search efficiency by at most 44.28% through effective collaboration among heterogeneous robots. It can also dynamically adapt to the changing conditions during task execution. 

---
# VocalAgent: Large Language Models for Vocal Health Diagnostics with Safety-Aware Evaluation 

**Authors**: Yubin Kim, Taehan Kim, Wonjune Kang, Eugene Park, Joonsik Yoon, Dongjae Lee, Xin Liu, Daniel McDuff, Hyeonhoon Lee, Cynthia Breazeal, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.13577)  

**Abstract**: Vocal health plays a crucial role in peoples' lives, significantly impacting their communicative abilities and interactions. However, despite the global prevalence of voice disorders, many lack access to convenient diagnosis and treatment. This paper introduces VocalAgent, an audio large language model (LLM) to address these challenges through vocal health diagnosis. We leverage Qwen-Audio-Chat fine-tuned on three datasets collected in-situ from hospital patients, and present a multifaceted evaluation framework encompassing a safety assessment to mitigate diagnostic biases, cross-lingual performance analysis, and modality ablation studies. VocalAgent demonstrates superior accuracy on voice disorder classification compared to state-of-the-art baselines. Its LLM-based method offers a scalable solution for broader adoption of health diagnostics, while underscoring the importance of ethical and technical validation. 

---
# Breaking the Compression Ceiling: Data-Free Pipeline for Ultra-Efficient Delta Compression 

**Authors**: Xiaohui Wang, Peng Ye, Chenyu Huang, Shenghe Zheng, Bo Zhang, Wanli Ouyang, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13563)  

**Abstract**: With the rise of the fine-tuned--pretrained paradigm, storing numerous fine-tuned models for multi-tasking creates significant storage overhead. Delta compression alleviates this by storing only the pretrained model and the highly compressed delta weights (the differences between fine-tuned and pretrained model weights). However, existing methods fail to maintain both high compression and performance, and often rely on data. To address these challenges, we propose UltraDelta, the first data-free delta compression pipeline that achieves both ultra-high compression and strong performance. UltraDelta is designed to minimize redundancy, maximize information, and stabilize performance across inter-layer, intra-layer, and global dimensions, using three key components: (1) Variance-Based Mixed Sparsity Allocation assigns sparsity based on variance, giving lower sparsity to high-variance layers to preserve inter-layer information. (2) Distribution-Aware Compression applies uniform quantization and then groups parameters by value, followed by group-wise pruning, to better preserve intra-layer distribution. (3) Trace-Norm-Guided Rescaling uses the trace norm of delta weights to estimate a global rescaling factor, improving model stability under higher compression. Extensive experiments across (a) large language models (fine-tuned on LLaMA-2 7B and 13B) with up to 133x, (b) general NLP models (RoBERTa-base, T5-base) with up to 800x, (c) vision models (ViT-B/32, ViT-L/14) with up to 400x, and (d) multi-modal models (BEiT-3) with 40x compression ratio, demonstrate that UltraDelta consistently outperforms existing methods, especially under ultra-high compression. 

---
# EcoSafeRAG: Efficient Security through Context Analysis in Retrieval-Augmented Generation 

**Authors**: Ruobing Yao, Yifei Zhang, Shuang Song, Neng Gao, Chenyang Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13506)  

**Abstract**: Retrieval-Augmented Generation (RAG) compensates for the static knowledge limitations of Large Language Models (LLMs) by integrating external knowledge, producing responses with enhanced factual correctness and query-specific contextualization. However, it also introduces new attack surfaces such as corpus poisoning at the same time. Most of the existing defense methods rely on the internal knowledge of the model, which conflicts with the design concept of RAG. To bridge the gap, EcoSafeRAG uses sentence-level processing and bait-guided context diversity detection to identify malicious content by analyzing the context diversity of candidate documents without relying on LLM internal knowledge. Experiments show EcoSafeRAG delivers state-of-the-art security with plug-and-play deployment, simultaneously improving clean-scenario RAG performance while maintaining practical operational costs (relatively 1.2$\times$ latency, 48\%-80\% token reduction versus Vanilla RAG). 

---
# ProdRev: A DNN framework for empowering customers using generative pre-trained transformers 

**Authors**: Aakash Gupta, Nataraj Das  

**Link**: [PDF](https://arxiv.org/pdf/2505.13491)  

**Abstract**: Following the pandemic, customers, preference for using e-commerce has accelerated. Since much information is available in multiple reviews (sometimes running in thousands) for a single product, it can create decision paralysis for the buyer. This scenario disempowers the consumer, who cannot be expected to go over so many reviews since its time consuming and can confuse them. Various commercial tools are available, that use a scoring mechanism to arrive at an adjusted score. It can alert the user to potential review manipulations. This paper proposes a framework that fine-tunes a generative pre-trained transformer to understand these reviews better. Furthermore, using "common-sense" to make better decisions. These models have more than 13 billion parameters. To fine-tune the model for our requirement, we use the curie engine from generative pre-trained transformer (GPT3). By using generative models, we are introducing abstractive summarization. Instead of using a simple extractive method of summarizing the reviews. This brings out the true relationship between the reviews and not simply copy-paste. This introduces an element of "common sense" for the user and helps them to quickly make the right decisions. The user is provided the pros and cons of the processed reviews. Thus the user/customer can take their own decisions. 

---
# LODGE: Joint Hierarchical Task Planning and Learning of Domain Models with Grounded Execution 

**Authors**: Claudius Kienle, Benjamin Alt, Oleg Arenz, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2505.13497)  

**Abstract**: Large Language Models (LLMs) enable planning from natural language instructions using implicit world knowledge, but often produce flawed plans that require refinement. Instead of directly predicting plans, recent methods aim to learn a problem domain that can be solved for different goal states using classical planners. However, these approaches require significant human feedback to obtain useful models. We address this shortcoming by learning hierarchical domains, where low-level predicates and actions are composed into higher-level counterparts, and by leveraging simulation to validate their preconditions and effects. This hierarchical approach is particularly powerful for long-horizon planning, where LLM-based planning approaches typically struggle. Furthermore, we introduce a central error reasoner to ensure consistency among the different planning levels. Evaluation on two challenging International Planning Competition (IPC) domains and a long-horizon robot manipulation task demonstrates higher planning success rates than state-of-the-art domain synthesis and LLM-modulo planning methods, while constructing high-quality models of the domain. Resources, videos and detailed experiment results are available at this https URL. 

---
# Pel, A Programming Language for Orchestrating AI Agents 

**Authors**: Behnam Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.13453)  

**Abstract**: The proliferation of Large Language Models (LLMs) has opened new frontiers in computing, yet controlling and orchestrating their capabilities beyond simple text generation remains a challenge. Current methods, such as function/tool calling and direct code generation, suffer from limitations in expressiveness, scalability, cost, security, and the ability to enforce fine-grained control. This paper introduces Pel, a novel programming language specifically designed to bridge this gap. Inspired by the strengths of Lisp, Elixir, Gleam, and Haskell, Pel provides a syntactically simple, homoiconic, and semantically rich platform for LLMs to express complex actions, control flow, and inter-agent communication safely and efficiently. Pel's design emphasizes a minimal, easily modifiable grammar suitable for constrained LLM generation, eliminating the need for complex sandboxing by enabling capability control at the syntax level. Key features include a powerful piping mechanism for linear composition, first-class closures enabling easy partial application and functional patterns, built-in support for natural language conditions evaluated by LLMs, and an advanced Read-Eval-Print-Loop (REPeL) with Common Lisp-style restarts and LLM-powered helper agents for automated error correction. Furthermore, Pel incorporates automatic parallelization of independent operations via static dependency analysis, crucial for performant agentic systems. We argue that Pel offers a more robust, secure, and expressive paradigm for LLM orchestration, paving the way for more sophisticated and reliable AI agentic frameworks. 

---
# LLM Context Conditioning and PWP Prompting for Multimodal Validation of Chemical Formulas 

**Authors**: Evgeny Markhasin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12257)  

**Abstract**: Identifying subtle technical errors within complex scientific and technical documents, especially those requiring multimodal interpretation (e.g., formulas in images), presents a significant hurdle for Large Language Models (LLMs) whose inherent error-correction tendencies can mask inaccuracies. This exploratory proof-of-concept (PoC) study investigates structured LLM context conditioning, informed by Persistent Workflow Prompting (PWP) principles, as a methodological strategy to modulate this LLM behavior at inference time. The approach is designed to enhance the reliability of readily available, general-purpose LLMs (specifically Gemini 2.5 Pro and ChatGPT Plus o3) for precise validation tasks, crucially relying only on their standard chat interfaces without API access or model modifications. To explore this methodology, we focused on validating chemical formulas within a single, complex test paper with known textual and image-based errors. Several prompting strategies were evaluated: while basic prompts proved unreliable, an approach adapting PWP structures to rigorously condition the LLM's analytical mindset appeared to improve textual error identification with both models. Notably, this method also guided Gemini 2.5 Pro to repeatedly identify a subtle image-based formula error previously overlooked during manual review, a task where ChatGPT Plus o3 failed in our tests. These preliminary findings highlight specific LLM operational modes that impede detail-oriented validation and suggest that PWP-informed context conditioning offers a promising and highly accessible technique for developing more robust LLM-driven analytical workflows, particularly for tasks requiring meticulous error detection in scientific and technical documents. Extensive validation beyond this limited PoC is necessary to ascertain broader applicability. 

---
# SLOT: Sample-specific Language Model Optimization at Test-time 

**Authors**: Yang Hu, Xingyu Zhang, Xueji Fang, Zhiyang Chen, Xiao Wang, Huatian Zhang, Guojun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12392)  

**Abstract**: We propose SLOT (Sample-specific Language Model Optimization at Test-time), a novel and parameter-efficient test-time inference approach that enhances a language model's ability to more accurately respond to individual prompts. Existing Large Language Models (LLMs) often struggle with complex instructions, leading to poor performances on those not well represented among general samples. To address this, SLOT conducts few optimization steps at test-time to update a light-weight sample-specific parameter vector. It is added to the final hidden layer before the output head, and enables efficient adaptation by caching the last layer features during per-sample optimization. By minimizing the cross-entropy loss on the input prompt only, SLOT helps the model better aligned with and follow each given instruction. In experiments, we demonstrate that our method outperforms the compared models across multiple benchmarks and LLMs. For example, Qwen2.5-7B with SLOT achieves an accuracy gain of 8.6% on GSM8K from 57.54% to 66.19%, while DeepSeek-R1-Distill-Llama-70B with SLOT achieves a SOTA accuracy of 68.69% on GPQA among 70B-level models. Our code is available at this https URL. 

---
